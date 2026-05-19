from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from pm4py.objects.process_tree.obj import ProcessTree

from rheon.config import ACTIVITY_KEY, GeneratorConfig
from rheon.drift.common import (
    DriftPlan,
    gradual_probability,
    incremental_index,
    is_gradual,
    recurring_index,
)
from rheon.metadata import DriftMetadata
from rheon.process_tree import (
    TreeMutationResult,
    generate_process_tree,
    mutate_tree,
)


@dataclass
class ControlFlowDrift:
    plan: DriftPlan
    versions: list[ProcessTree]
    details: dict[str, object]

    @property
    def final_tree(self) -> ProcessTree:
        return self.versions[-1]


@dataclass
class ControlFlowModel:
    initial_tree: ProcessTree
    drifts: list[ControlFlowDrift]
    metadata: list[DriftMetadata]
    config: GeneratorConfig

    def tree_for(self, timestamp: datetime, rng: np.random.Generator) -> ProcessTree:
        current = self.initial_tree
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
                current = drift.final_tree
            elif plan.drift_type == "incremental":
                if timestamp < end:
                    idx = incremental_index(timestamp, start, end, len(drift.versions))
                    return drift.versions[idx]
                current = drift.final_tree
            elif plan.drift_type == "recurring":
                if timestamp < end:
                    return drift.versions[recurring_index(timestamp, plan, self.config)]
                current = drift.final_tree
            else:
                current = drift.final_tree
        return current

def build_control_flow_model(
    config: GeneratorConfig,
    plans: list[DriftPlan],
    rng: np.random.Generator,
) -> ControlFlowModel:
    initial = generate_process_tree(config, rng)
    current = initial
    drifts: list[ControlFlowDrift] = []
    metadata: list[DriftMetadata] = []

    for plan in sorted(plans, key=lambda item: item.change_point):
        if plan.perspective != "control_flow":
            continue
        if plan.drift_type == "incremental":
            runtime, meta = _build_incremental(config, plan, current, rng)
        else:
            mutation = mutate_tree(current, config, rng, _intensity(plan))
            runtime = ControlFlowDrift(
                plan=plan,
                versions=[current, mutation.tree],
                details=mutation.change_details(),
            )
            meta = _metadata(plan, mutation.change_details())
        drifts.append(runtime)
        metadata.append(meta)
        current = runtime.final_tree
    return ControlFlowModel(initial_tree=initial, drifts=drifts, metadata=metadata, config=config)


def _build_incremental(
    config: GeneratorConfig,
    plan: DriftPlan,
    current: ProcessTree,
    rng: np.random.Generator,
) -> tuple[ControlFlowDrift, DriftMetadata]:
    version_count = int(plan.spec.get("num_versions", rng.integers(3, 6)))
    version_count = max(3, min(5, version_count))
    versions = [current]
    added: list[str] = []
    deleted: list[str] = []
    moved: list[str] = []
    swaps: list[dict[str, str]] = []
    for _ in range(version_count - 1):
        mutation = mutate_tree(versions[-1], config, rng, _intensity(plan) / 2)
        versions.append(mutation.tree)
        added.extend(mutation.activities_added)
        deleted.extend(mutation.activities_deleted)
        moved.extend(mutation.activities_moved)
        swaps.extend(mutation.operator_swaps)
    mini_points = _mini_change_points(plan, version_count)
    details = {
        "activities_added": added,
        "activities_deleted": deleted,
        "activities_moved": moved,
        "operator_swaps": swaps,
        "version_count": version_count,
        "version_change_points": [point.isoformat() for point in mini_points],
    }
    drift = ControlFlowDrift(plan=plan, versions=versions, details=details)
    return drift, _metadata(plan, details)


def _metadata(plan: DriftPlan, details: dict[str, object]) -> DriftMetadata:
    return DriftMetadata(
        drift_id=plan.drift_id,
        perspective="control_flow",
        subtype=plan.subtype,
        drift_type=plan.drift_type,
        change_point_timestamp=plan.change_point,
        overlap_window_start=plan.overlap_start,
        overlap_window_end=plan.overlap_end,
        affected_columns=[ACTIVITY_KEY],
        change_details=details,
    )


def _mini_change_points(plan: DriftPlan, version_count: int) -> list[datetime]:
    start = plan.overlap_start or plan.change_point
    end = plan.overlap_end or (plan.change_point + timedelta(seconds=1))
    total = max(1.0, (end - start).total_seconds())
    return [
        start + timedelta(seconds=total * idx / version_count)
        for idx in range(1, version_count)
    ]


def _intensity(plan: DriftPlan) -> float:
    return float(plan.spec.get("change_proportion", plan.spec.get("intensity", 0.25)))
