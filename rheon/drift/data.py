from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from rheon.config import AMOUNT_KEY, GeneratorConfig, REGION_KEY
from rheon.drift.common import (
    DriftPlan,
    gradual_probability,
    incremental_index,
    is_gradual,
    recurring_index,
)
from rheon.metadata import DriftMetadata


@dataclass
class DataState:
    amount_mean: float
    amount_std: float
    region_probs: np.ndarray

    def clone(self) -> "DataState":
        return copy.deepcopy(self)


@dataclass
class DataDrift:
    plan: DriftPlan
    versions: list[DataState]
    details: dict[str, object]

    @property
    def final_state(self) -> DataState:
        return self.versions[-1]


@dataclass
class DataRuntime:
    initial_state: DataState
    drifts: list[DataDrift]
    metadata: list[DriftMetadata]
    config: GeneratorConfig

    def state_for(self, timestamp: datetime, rng: np.random.Generator) -> DataState:
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

    def sample_amount_region(
        self,
        timestamp: datetime,
        rng: np.random.Generator,
    ) -> tuple[float, str]:
        state = self.state_for(timestamp, rng)
        amount = max(0.0, float(rng.normal(state.amount_mean, state.amount_std)))
        region = str(rng.choice(self.config.regions, p=state.region_probs))
        return round(amount, 2), region


def build_data_runtime(
    config: GeneratorConfig,
    plans: list[DriftPlan],
    rng: np.random.Generator,
) -> DataRuntime:
    initial = DataState(
        amount_mean=1000.0,
        amount_std=250.0,
        region_probs=np.ones(len(config.regions)) / len(config.regions),
    )
    current = initial
    drifts: list[DataDrift] = []
    metadata: list[DriftMetadata] = []
    for plan in sorted([p for p in plans if p.perspective == "data"], key=lambda item: item.change_point):
        drift, meta = _build_drift(config, plan, current, rng)
        drifts.append(drift)
        metadata.append(meta)
        current = drift.final_state
    return DataRuntime(initial, drifts, metadata, config)


def _build_drift(
    config: GeneratorConfig,
    plan: DriftPlan,
    current: DataState,
    rng: np.random.Generator,
) -> tuple[DataDrift, DriftMetadata]:
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
    affected = [AMOUNT_KEY] if plan.subtype == "numeric" else [REGION_KEY]
    meta = DriftMetadata(
        drift_id=plan.drift_id,
        perspective="data",
        subtype=plan.subtype,
        drift_type=plan.drift_type,
        change_point_timestamp=plan.change_point,
        overlap_window_start=plan.overlap_start,
        overlap_window_end=plan.overlap_end,
        affected_columns=affected,
        change_details=details,
    )
    return DataDrift(plan, versions, details), meta


def _mutate(
    config: GeneratorConfig,
    subtype: str,
    state: DataState,
    rng: np.random.Generator,
) -> tuple[DataState, dict[str, object]]:
    new_state = state.clone()
    if subtype == "categorical":
        old_probs = new_state.region_probs.copy()
        probs = np.ones(len(config.regions)) * 0.3 / max(1, len(config.regions) - 1)
        dominant = int(rng.integers(0, len(config.regions)))
        probs[dominant] = 0.7
        new_state.region_probs = probs / probs.sum()
        return new_state, {
            "old_region_distribution": old_probs.round(4).tolist(),
            "new_region_distribution": new_state.region_probs.round(4).tolist(),
            "dominant_region": config.regions[dominant],
        }
    old_mean = new_state.amount_mean
    old_std = new_state.amount_std
    new_state.amount_mean = old_mean * float(rng.choice([0.65, 1.45]))
    new_state.amount_std = old_std * float(rng.choice([0.7, 1.6]))
    return new_state, {
        "old_amount_mean": round(old_mean, 4),
        "new_amount_mean": round(new_state.amount_mean, 4),
        "old_amount_std": round(old_std, 4),
        "new_amount_std": round(new_state.amount_std, 4),
    }


def _mini_change_points(plan: DriftPlan, version_count: int) -> list[datetime]:
    start = plan.overlap_start or plan.change_point
    end = plan.overlap_end or (plan.change_point + timedelta(seconds=1))
    total = max(1.0, (end - start).total_seconds())
    return [
        start + timedelta(seconds=total * idx / version_count)
        for idx in range(1, version_count)
    ]
