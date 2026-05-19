from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from rheon.config import CASE_ID_KEY, CASE_TYPE_KEY, GeneratorConfig, START_TIMESTAMP_KEY, TIMESTAMP_KEY
from rheon.drift.common import (
    DriftPlan,
    gradual_probability,
    incremental_index,
    is_gradual,
    recurring_index,
)
from rheon.metadata import DriftMetadata


@dataclass
class InterCaseState:
    arrival_mean_multiplier: float
    burstiness_shape: str
    case_type_probs: np.ndarray

    def clone(self) -> "InterCaseState":
        return copy.deepcopy(self)


@dataclass
class InterCaseDrift:
    plan: DriftPlan
    versions: list[InterCaseState]
    details: dict[str, object]

    @property
    def final_state(self) -> InterCaseState:
        return self.versions[-1]


@dataclass
class InterCaseRuntime:
    initial_state: InterCaseState
    drifts: list[InterCaseDrift]
    metadata: list[DriftMetadata]
    config: GeneratorConfig

    def state_for(self, timestamp: datetime, rng: np.random.Generator) -> InterCaseState:
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

    def sample_case_type(self, timestamp: datetime, rng: np.random.Generator) -> str:
        state = self.state_for(timestamp, rng)
        return str(rng.choice(self.config.case_types, p=state.case_type_probs))


def build_inter_case_runtime(
    config: GeneratorConfig,
    plans: list[DriftPlan],
    rng: np.random.Generator,
) -> InterCaseRuntime:
    initial = InterCaseState(
        arrival_mean_multiplier=1.0,
        burstiness_shape="exponential",
        case_type_probs=np.ones(config.num_case_types) / config.num_case_types,
    )
    current = initial
    drifts: list[InterCaseDrift] = []
    metadata: list[DriftMetadata] = []
    for plan in sorted([p for p in plans if p.perspective == "inter_case"], key=lambda item: item.change_point):
        drift, meta = _build_drift(config, plan, current, rng)
        drifts.append(drift)
        metadata.append(meta)
        current = drift.final_state
    return InterCaseRuntime(initial, drifts, metadata, config)


def generate_case_arrivals(
    config: GeneratorConfig,
    runtime: InterCaseRuntime,
    horizon_start: datetime,
    horizon_end: datetime,
    rng: np.random.Generator,
) -> list[datetime]:
    horizon_min = max(1.0, (horizon_end - horizon_start).total_seconds() / 60.0)
    base_mean = max(config.inter_arrival_mean_min, horizon_min / max(1, config.num_traces))
    has_arrival_drift = any(
        drift.plan.subtype in {"arrival_rate", "burstiness"} for drift in runtime.drifts
    )
    arrivals: list[datetime] = []
    current = horizon_start
    max_cases = max(config.num_traces * 3, config.num_traces + 10)
    target_cases = max(1, config.num_traces)

    while len(arrivals) < max_cases:
        state = runtime.state_for(current, rng)
        mean = base_mean * state.arrival_mean_multiplier
        delta = _sample_interarrival(mean, state.burstiness_shape, rng)
        current = current + timedelta(minutes=delta)
        if current > horizon_end:
            break
        arrivals.append(current)
        if not has_arrival_drift and len(arrivals) >= target_cases:
            break

    if not arrivals:
        arrivals.append(horizon_start)
    if not has_arrival_drift and len(arrivals) < target_cases:
        while len(arrivals) < target_cases:
            arrivals.append(arrivals[-1] + timedelta(minutes=base_mean))
    return arrivals


def _build_drift(
    config: GeneratorConfig,
    plan: DriftPlan,
    current: InterCaseState,
    rng: np.random.Generator,
) -> tuple[InterCaseDrift, DriftMetadata]:
    if plan.drift_type == "incremental":
        version_count = int(plan.spec.get("num_versions", rng.integers(3, 6)))
        version_count = max(3, min(5, version_count))
        versions = [current]
        steps: list[dict[str, object]] = []
        for _ in range(version_count - 1):
            new_state, details = _mutate(config, plan.subtype, versions[-1], rng)
            versions.append(new_state)
            steps.append(details)
        details = {
            "version_count": version_count,
            "steps": steps,
            "version_change_points": [
                point.isoformat() for point in _mini_change_points(plan, version_count)
            ],
        }
    else:
        new_state, details = _mutate(config, plan.subtype, current, rng)
        versions = [current, new_state]
    affected = {
        "arrival_rate": [CASE_ID_KEY, START_TIMESTAMP_KEY, TIMESTAMP_KEY],
        "burstiness": [CASE_ID_KEY, START_TIMESTAMP_KEY, TIMESTAMP_KEY],
        "case_mix": [CASE_TYPE_KEY],
        "concurrency": [],
    }.get(plan.subtype, [CASE_TYPE_KEY])
    meta = DriftMetadata(
        drift_id=plan.drift_id,
        perspective="inter_case",
        subtype=plan.subtype,
        drift_type=plan.drift_type,
        change_point_timestamp=plan.change_point,
        overlap_window_start=plan.overlap_start,
        overlap_window_end=plan.overlap_end,
        affected_columns=affected,
        change_details=details,
    )
    return InterCaseDrift(plan, versions, details), meta


def _mutate(
    config: GeneratorConfig,
    subtype: str,
    state: InterCaseState,
    rng: np.random.Generator,
) -> tuple[InterCaseState, dict[str, object]]:
    new_state = state.clone()
    if subtype == "arrival_rate":
        old = new_state.arrival_mean_multiplier
        new_state.arrival_mean_multiplier = 0.5 if old >= 1.0 else 1.8
        return new_state, {
            "old_interarrival_mean_multiplier": old,
            "new_interarrival_mean_multiplier": new_state.arrival_mean_multiplier,
        }
    if subtype == "burstiness":
        old = new_state.burstiness_shape
        new_state.burstiness_shape = "bimodal_pareto" if old == "exponential" else "exponential"
        return new_state, {
            "old_shape": old,
            "new_shape": new_state.burstiness_shape,
            "mean_preserved": True,
        }
    if subtype == "concurrency":
        # Concurrency is an emergent property of overlapping case intervals. It is
        # not directly injected; arrival and duration choices can change it.
        return new_state, {
            "direct_injection": False,
            "note": "Concurrency is emergent from arrivals and service times.",
        }
    old_probs = new_state.case_type_probs.copy()
    probs = np.ones(config.num_case_types) * 0.3 / max(1, config.num_case_types - 1)
    dominant = int(rng.integers(0, config.num_case_types))
    probs[dominant] = 0.7
    new_state.case_type_probs = probs / probs.sum()
    return new_state, {
        "old_case_type_distribution": old_probs.round(4).tolist(),
        "new_case_type_distribution": new_state.case_type_probs.round(4).tolist(),
        "dominant_case_type": config.case_types[dominant],
    }


def _sample_interarrival(mean_min: float, shape: str, rng: np.random.Generator) -> float:
    mean_min = max(0.001, mean_min)
    if shape == "bimodal_pareto":
        if rng.random() < 0.8:
            return float(rng.exponential(mean_min * 0.2))
        alpha = 2.5
        scale = mean_min * 4.0 * (alpha - 1.0) / alpha
        return float((rng.pareto(alpha) + 1.0) * scale)
    return float(rng.exponential(mean_min))


def _mini_change_points(plan: DriftPlan, version_count: int) -> list[datetime]:
    start = plan.overlap_start or plan.change_point
    end = plan.overlap_end or (plan.change_point + timedelta(seconds=1))
    total = max(1.0, (end - start).total_seconds())
    return [
        start + timedelta(seconds=total * idx / version_count)
        for idx in range(1, version_count)
    ]
