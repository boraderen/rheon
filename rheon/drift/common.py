from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from rheon.config import GeneratorConfig


@dataclass(frozen=True)
class DriftPlan:
    drift_id: str
    perspective: str
    subtype: str
    drift_type: str
    change_point: datetime
    overlap_start: datetime | None
    overlap_end: datetime | None
    spec: dict[str, Any]


def normalize_perspective_drifts(
    drifts: list[dict[str, Any]],
    default_perspective: str,
) -> list[dict[str, Any]]:
    result = []
    for drift in drifts:
        enriched = dict(drift)
        enriched.setdefault("perspective", default_perspective)
        result.append(enriched)
    return result


def sample_drift_plans(
    drifts: list[dict[str, Any]],
    *,
    config: GeneratorConfig,
    horizon_start: datetime,
    horizon_end: datetime,
    rng: np.random.Generator,
    default_perspective: str,
) -> list[DriftPlan]:
    normalized = normalize_perspective_drifts(drifts, default_perspective)
    by_perspective: dict[str, list[dict[str, Any]]] = {}
    for drift in normalized:
        by_perspective.setdefault(str(drift["perspective"]), []).append(drift)

    plans: list[DriftPlan] = []
    horizon_seconds = max(1.0, (horizon_end - horizon_start).total_seconds())
    for perspective, perspective_drifts in by_perspective.items():
        count = len(perspective_drifts)
        for index, spec in enumerate(perspective_drifts):
            fraction = _change_fraction(spec, index, count, rng)
            change_point = horizon_start + timedelta(seconds=horizon_seconds * fraction)
            drift_type = str(spec.get("drift_type", "sudden"))
            overlap_start: datetime | None = None
            overlap_end: datetime | None = None
            if is_gradual(drift_type):
                window = timedelta(seconds=horizon_seconds * config.gradual_overlap_fraction)
                overlap_start = max_datetime(horizon_start, change_point - window / 2)
                overlap_end = min_datetime(horizon_end, change_point + window / 2)
            elif drift_type == "incremental":
                window = timedelta(seconds=horizon_seconds * config.gradual_overlap_fraction)
                overlap_start = change_point
                overlap_end = min_datetime(horizon_end, change_point + window)
            elif drift_type == "recurring":
                period = recurring_period(config, horizon_start, horizon_end, spec)
                overlap_start = change_point
                overlap_end = min_datetime(horizon_end, change_point + period * 4)
            plans.append(
                DriftPlan(
                    drift_id=f"d{len(plans) + 1:02d}",
                    perspective=perspective,
                    subtype=str(spec.get("subtype", "unknown")),
                    drift_type=drift_type,
                    change_point=change_point,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                    spec=dict(spec),
                )
            )
    return sorted(plans, key=lambda item: (item.perspective, item.change_point))


def is_gradual(drift_type: str) -> bool:
    return drift_type in {"gradual", "gradual_linear", "gradual_exponential"}


def gradual_probability(
    drift_type: str,
    timestamp: datetime,
    overlap_start: datetime,
    overlap_end: datetime,
) -> float:
    total = max(1e-9, (overlap_end - overlap_start).total_seconds())
    x = min(1.0, max(0.0, (timestamp - overlap_start).total_seconds() / total))
    if drift_type == "gradual_exponential":
        k = 5.0
        return float((np.exp(k * x) - 1.0) / (np.exp(k) - 1.0))
    return x


def incremental_index(
    timestamp: datetime,
    overlap_start: datetime,
    overlap_end: datetime,
    n_versions: int,
) -> int:
    if timestamp < overlap_start:
        return 0
    if timestamp >= overlap_end:
        return n_versions - 1
    total = max(1e-9, (overlap_end - overlap_start).total_seconds())
    x = (timestamp - overlap_start).total_seconds() / total
    return min(n_versions - 1, int(np.floor(x * n_versions)))


def recurring_index(
    timestamp: datetime,
    plan: DriftPlan,
    config: GeneratorConfig,
) -> int:
    if plan.overlap_start is None or plan.overlap_end is None:
        return 0
    if timestamp < plan.overlap_start:
        return 0
    if timestamp >= plan.overlap_end:
        return 1
    period = recurring_period(config, plan.overlap_start, plan.overlap_end, plan.spec)
    elapsed = max(0.0, (timestamp - plan.overlap_start).total_seconds())
    phase = int(elapsed // max(1.0, period.total_seconds()))
    return phase % 2


def recurring_period(
    config: GeneratorConfig,
    horizon_start: datetime,
    horizon_end: datetime,
    spec: dict[str, Any],
) -> timedelta:
    if "period_days" in spec:
        return timedelta(days=float(spec["period_days"]))
    seconds = max(1.0, (horizon_end - horizon_start).total_seconds())
    return timedelta(seconds=max(1.0, seconds * config.recurring_period_fraction))


def max_datetime(left: datetime, right: datetime) -> datetime:
    return left if left >= right else right


def min_datetime(left: datetime, right: datetime) -> datetime:
    return left if left <= right else right


def _change_fraction(
    spec: dict[str, Any],
    index: int,
    count: int,
    rng: np.random.Generator,
) -> float:
    explicit = spec.get("change_point")
    if explicit is not None:
        return min(0.95, max(0.05, float(explicit)))
    left = index / max(1, count)
    right = (index + 1) / max(1, count)
    center = (index + 1) / (count + 1)
    jitter = 0.20 / (count + 1)
    value = float(center + rng.uniform(-jitter, jitter))
    return min(right - 0.03, max(left + 0.03, value))
