from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from rheon.config import GeneratorConfig, RESOURCE_KEY, ROLE_KEY
from rheon.drift.common import (
    DriftPlan,
    gradual_probability,
    incremental_index,
    is_gradual,
    recurring_index,
)
from rheon.metadata import DriftMetadata


@dataclass
class ResourceState:
    resources: list[str]
    active_resources: list[str]
    role_by_resource: dict[str, str]
    activity_probs: dict[str, np.ndarray]
    workload_probs: np.ndarray
    handover_probs: dict[str, np.ndarray]
    resource_speed: dict[str, float]  # multiplier per resource: >1 slower, <1 faster

    def clone(self) -> "ResourceState":
        return copy.deepcopy(self)

    def active_masked_probs(self, probs: np.ndarray) -> np.ndarray:
        mask = np.array([resource in set(self.active_resources) for resource in self.resources])
        masked = probs * mask
        if masked.sum() <= 0:
            masked = mask.astype(float)
        return masked / masked.sum()


@dataclass
class ResourceDrift:
    plan: DriftPlan
    versions: list[ResourceState]
    details: dict[str, object]

    @property
    def final_state(self) -> ResourceState:
        return self.versions[-1]


@dataclass
class ResourceRuntime:
    initial_state: ResourceState
    drifts: list[ResourceDrift]
    metadata: list[DriftMetadata]
    config: GeneratorConfig

    def state_for(self, timestamp: datetime, rng: np.random.Generator) -> ResourceState:
        current = self.initial_state
        for drift in self.drifts:
            plan = drift.plan
            start = plan.overlap_start or plan.change_point
            end = plan.overlap_end or plan.change_point
            if timestamp < start:
                return current
            if is_gradual(plan.drift_type):
                if timestamp < end:
                    p_new = gradual_probability(plan.drift_type, timestamp, start, end)
                    return drift.versions[1] if rng.random() < p_new else drift.versions[0]
                current = drift.final_state
            elif plan.drift_type == "incremental":
                if timestamp < end:
                    idx = incremental_index(timestamp, start, end, len(drift.versions))
                    return drift.versions[idx]
                current = drift.final_state
            elif plan.drift_type == "recurring":
                if timestamp < end:
                    return drift.versions[recurring_index(timestamp, plan, self.config)]
                current = drift.final_state
            else:
                current = drift.final_state
        return current

    def assign(
        self,
        activity: str,
        previous_resource: str | None,
        timestamp: datetime,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        state = self.state_for(timestamp, rng)
        probs = state.activity_probs.get(activity, state.workload_probs)
        if previous_resource is not None and previous_resource in state.handover_probs and rng.random() < 0.45:
            probs = state.handover_probs[previous_resource]
        probs = state.active_masked_probs(probs)
        resource = str(rng.choice(state.resources, p=probs))
        return resource, state.role_by_resource[resource]

    def speed_for(self, resource: str, timestamp: datetime, rng: np.random.Generator) -> float:
        state = self.state_for(timestamp, rng)
        return state.resource_speed.get(resource, 1.0)


def build_resource_runtime(
    config: GeneratorConfig,
    plans: list[DriftPlan],
    activities: list[str],
    rng: np.random.Generator,
) -> ResourceRuntime:
    initial = _initial_state(config, activities, rng)
    current = initial
    drifts: list[ResourceDrift] = []
    metadata: list[DriftMetadata] = []
    for plan in sorted([p for p in plans if p.perspective == "resource"], key=lambda item: item.change_point):
        drift, meta = _build_resource_drift(config, plan, current, activities, rng)
        drifts.append(drift)
        metadata.append(meta)
        current = drift.final_state
    return ResourceRuntime(initial, drifts, metadata, config)


def _initial_state(
    config: GeneratorConfig,
    activities: list[str],
    rng: np.random.Generator,
) -> ResourceState:
    resources = config.resources
    roles = config.roles
    role_by_resource = {
        resource: roles[idx % len(roles)] for idx, resource in enumerate(resources)
    }
    base = np.ones(len(resources), dtype=float) / len(resources)
    activity_probs = {}
    for activity in activities:
        dominant = int(rng.integers(0, len(resources)))
        probs = np.ones(len(resources), dtype=float) * 0.2 / max(1, len(resources) - 1)
        probs[dominant] = 0.8
        activity_probs[activity] = probs / probs.sum()
    return ResourceState(
        resources=resources,
        active_resources=list(resources),
        role_by_resource=role_by_resource,
        activity_probs=activity_probs,
        workload_probs=base,
        handover_probs={resource: base.copy() for resource in resources},
        resource_speed={resource: 1.0 for resource in resources},
    )


def _build_resource_drift(
    config: GeneratorConfig,
    plan: DriftPlan,
    current: ResourceState,
    activities: list[str],
    rng: np.random.Generator,
) -> tuple[ResourceDrift, DriftMetadata]:
    if plan.drift_type == "incremental":
        version_count = int(plan.spec.get("num_versions", rng.integers(3, 6)))
        version_count = max(3, min(5, version_count))
        versions = [current]
        changes: list[dict[str, object]] = []
        for _ in range(version_count - 1):
            new_state, details = _mutate_state(config, plan.subtype, versions[-1], activities, plan, rng)
            versions.append(new_state)
            changes.append(details)
        details = {
            "version_count": version_count,
            "steps": changes,
            "version_change_points": [
                point.isoformat() for point in _mini_change_points(plan, version_count)
            ],
        }
    else:
        new_state, details = _mutate_state(config, plan.subtype, current, activities, plan, rng)
        versions = [current, new_state]
    drift = ResourceDrift(plan=plan, versions=versions, details=details)
    meta = DriftMetadata(
        drift_id=plan.drift_id,
        perspective="resource",
        subtype=plan.subtype,
        drift_type=plan.drift_type,
        change_point_timestamp=plan.change_point,
        overlap_window_start=plan.overlap_start,
        overlap_window_end=plan.overlap_end,
        affected_columns=[RESOURCE_KEY, ROLE_KEY],
        change_details=details,
    )
    return drift, meta


def _mutate_state(
    config: GeneratorConfig,
    subtype: str,
    state: ResourceState,
    activities: list[str],
    plan: DriftPlan,
    rng: np.random.Generator,
) -> tuple[ResourceState, dict[str, object]]:
    if subtype == "pool_size":
        return _pool_size_drift(state, rng)
    if subtype == "handover":
        return _handover_drift(state, rng)
    if subtype == "service_time":
        return _service_time_drift(state, plan.spec, rng)
    if subtype == "workload_distribution":
        return _workload_distribution_drift(state, activities, plan.spec, rng)
    return _reassignment_drift(state, activities, plan.spec, rng)


def _reassignment_drift(
    state: ResourceState,
    activities: list[str],
    spec: dict,
    rng: np.random.Generator,
) -> tuple[ResourceState, dict[str, object]]:
    activities_spec = spec.get("activities", "all")
    targets = activities if activities_spec == "all" else [a for a in activities_spec if a in activities]
    new_state = state.clone()
    reassignments = []
    for activity in targets:
        old_probs = new_state.activity_probs.get(activity, new_state.workload_probs)
        old_resource = new_state.resources[int(np.argmax(old_probs))]
        choices = [resource for resource in new_state.active_resources if resource != old_resource]
        new_resource = str(rng.choice(choices or new_state.active_resources))
        probs = np.ones(len(new_state.resources), dtype=float) * 0.2 / max(1, len(new_state.resources) - 1)
        probs[new_state.resources.index(new_resource)] = 0.8
        new_state.activity_probs[activity] = probs / probs.sum()
        reassignments.append({
            "activity": activity,
            "old_dominant_resource": old_resource,
            "new_dominant_resource": new_resource,
            "post_drift_probability": 0.8,
        })
    return new_state, {"activities": activities_spec, "reassignments": reassignments}


def _service_time_drift(
    state: ResourceState,
    spec: dict,
    rng: np.random.Generator,
) -> tuple[ResourceState, dict[str, object]]:
    resources_spec = spec.get("resources", "all")
    targets = (
        state.active_resources
        if resources_spec == "all"
        else [r for r in resources_spec if r in state.resources]
    )
    new_state = state.clone()
    changes = {}
    for resource in targets:
        old = new_state.resource_speed.get(resource, 1.0)
        # sample a new multiplier clearly different from the current one
        new_mult = float(rng.uniform(0.3, 3.0))
        new_state.resource_speed[resource] = new_mult
        changes[resource] = {"old_multiplier": round(old, 4), "new_multiplier": round(new_mult, 4)}
    return new_state, {
        "resources": resources_spec,
        "multiplier_changes": changes,
        "note": "multiplier > 1.0 means slower (longer wait + processing), < 1.0 means faster",
    }


def _workload_distribution_drift(
    state: ResourceState,
    activities: list[str],
    spec: dict,
    rng: np.random.Generator,
) -> tuple[ResourceState, dict[str, object]]:
    resources_spec = spec.get("resources", "all")
    affected = _target_resources(state, resources_spec)
    if resources_spec == "all":
        heavy = _sample_heavy_resources(affected, rng)
    else:
        heavy = affected

    new_state = state.clone()
    if not affected:
        return new_state, {
            "resources": resources_spec,
            "affected_resources": [],
            "heavy_resources": [],
            "heavy_resource_share": 0.0,
            "probability_changes": {},
        }

    new_probs = _event_count_probs(new_state, heavy)
    old_probs = _mean_activity_probs(state, activities)
    new_state.workload_probs = new_probs
    for activity in list(new_state.activity_probs):
        new_state.activity_probs[activity] = new_probs.copy()

    changes = {
        resource: {
            "old_probability": round(float(old_probs[new_state.resources.index(resource)]), 4),
            "new_probability": round(float(new_probs[new_state.resources.index(resource)]), 4),
        }
        for resource in new_state.active_resources
    }
    heavy_indices = [new_state.resources.index(resource) for resource in heavy]
    return new_state, {
        "resources": resources_spec,
        "affected_resources": affected,
        "heavy_resources": heavy,
        "heavy_resource_share": round(float(new_probs[heavy_indices].sum()), 4) if heavy else 0.0,
        "probability_changes": changes,
    }


def _target_resources(state: ResourceState, resources_spec: object) -> list[str]:
    if resources_spec == "all":
        return list(state.active_resources)
    if not isinstance(resources_spec, list):
        return []
    return [str(resource) for resource in resources_spec if str(resource) in state.active_resources]


def _sample_heavy_resources(resources: list[str], rng: np.random.Generator) -> list[str]:
    if len(resources) <= 1:
        return list(resources)
    heavy_count = max(1, min(len(resources) - 1, int(round(len(resources) * 0.35))))
    return sorted(str(resource) for resource in rng.choice(resources, size=heavy_count, replace=False))


def _event_count_probs(state: ResourceState, heavy_resources: list[str]) -> np.ndarray:
    active = list(state.active_resources)
    heavy = [resource for resource in heavy_resources if resource in active]
    if not heavy:
        return state.active_masked_probs(state.workload_probs)

    probs = np.zeros(len(state.resources), dtype=float)
    light = [resource for resource in active if resource not in set(heavy)]
    heavy_share = 1.0 if not light else 0.7
    light_share = 1.0 - heavy_share
    for resource in heavy:
        probs[state.resources.index(resource)] = heavy_share / len(heavy)
    for resource in light:
        probs[state.resources.index(resource)] = light_share / len(light)
    return probs / probs.sum()


def _mean_activity_probs(state: ResourceState, activities: list[str]) -> np.ndarray:
    vectors = [
        state.active_masked_probs(state.activity_probs.get(activity, state.workload_probs))
        for activity in activities
    ]
    if not vectors:
        return state.active_masked_probs(state.workload_probs)
    return np.mean(vectors, axis=0)


def _pool_size_drift(
    state: ResourceState,
    rng: np.random.Generator,
) -> tuple[ResourceState, dict[str, object]]:
    new_state = state.clone()
    current_size = len(new_state.active_resources)
    max_size = len(new_state.resources)
    if current_size > max(2, int(max_size * 0.65)):
        new_size = max(2, int(round(current_size * 0.6)))
    else:
        new_size = min(max_size, max(current_size + 1, int(round(max_size * 0.8))))
    selected = list(rng.choice(new_state.resources, size=new_size, replace=False))
    new_state.active_resources = sorted(str(resource) for resource in selected)
    return new_state, {
        "old_pool_size": current_size,
        "new_pool_size": new_size,
        "active_resources": new_state.active_resources,
    }


def _handover_drift(
    state: ResourceState,
    rng: np.random.Generator,
) -> tuple[ResourceState, dict[str, object]]:
    new_state = state.clone()
    source = str(rng.choice(new_state.active_resources))
    old_probs = new_state.handover_probs[source]
    old_target = new_state.resources[int(np.argmax(old_probs))]
    candidates = [resource for resource in new_state.active_resources if resource != old_target]
    target = str(rng.choice(candidates or new_state.active_resources))
    probs = np.ones(len(new_state.resources), dtype=float) * 0.2 / max(1, len(new_state.resources) - 1)
    probs[new_state.resources.index(target)] = 0.8
    new_state.handover_probs[source] = probs / probs.sum()
    return new_state, {
        "source_resource": source,
        "old_dominant_target": old_target,
        "new_dominant_target": target,
        "post_drift_probability": 0.8,
    }



def _mini_change_points(plan: DriftPlan, version_count: int) -> list[datetime]:
    start = plan.overlap_start or plan.change_point
    end = plan.overlap_end or (plan.change_point + timedelta(seconds=1))
    total = max(1.0, (end - start).total_seconds())
    return [
        start + timedelta(seconds=total * idx / version_count)
        for idx in range(1, version_count)
    ]
