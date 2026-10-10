"""Drift specifications, validation of the user-supplied drift list, and timeline helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from rheon.config import (
    GeneratorConfig,
    check_positive,
    check_positive_int,
    check_tree_weights,
    check_variance,
)


# Known drift types grouped by the perspective they belong to (perspective is
# documentation only - the engine dispatches purely on the drift `type`).
DRIFT_TYPES = {
    "control_flow": "intra-case",
    "pool_size": "resource",
    "reassignment": "resource",
    "workload": "resource",
    "duration": "resource",
    "waiting_time": "inter-case",
    "amount": "inter-case",
    "arrival_rate": "inter-case",
    "region": "inter-case",
}

# The type-specific parameters each drift type accepts.
DRIFT_PARAMS = {
    "control_flow": {"num_activities", "tree_weights"},
    "pool_size": {"delta", "duration_factor"},
    "reassignment": set(),
    "workload": {"workload_factor"},
    "duration": {"resources", "factor"},
    "waiting_time": {"mean", "variance"},
    "amount": {"mean", "variance"},
    "arrival_rate": {"inter_arrival", "factor"},
    "region": set(),
}

# The position keys each mode requires.
POSITION_KEYS = {"sudden": ["drift_point"], "gradual": ["start_point", "end_point"]}


@dataclass
class Drift:
    """One normalized drift: its type, mode, transition window and type-specific params."""

    index: int
    type: str
    mode: str
    start_frac: float          # == drift_point for sudden drifts
    end_frac: float            # == drift_point for sudden drifts
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Stable identifier such as `d01`."""
        return f"d{self.index:02d}"

    def ramp(self, position: float) -> float:
        """Fraction of the change in effect at a horizon position (0 before, 1 after)."""
        if position < self.start_frac:
            return 0.0
        if position >= self.end_frac:
            return 1.0
        return (position - self.start_frac) / (self.end_frac - self.start_frac)

    def window_dates(self, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """Drift point / window as both horizon fractions and absolute timestamps."""
        span = end_date - start_date

        def at(frac: float) -> str:
            return (start_date + timedelta(seconds=span.total_seconds() * frac)).isoformat()

        if self.mode == "sudden":
            return {"drift_point": self.start_frac, "drift_point_date": at(self.start_frac)}
        return {
            "start_point": self.start_frac,
            "end_point": self.end_frac,
            "start_point_date": at(self.start_frac),
            "end_point_date": at(self.end_frac),
        }


def normalize_drifts(drifts: list[dict[str, Any]], config: GeneratorConfig | None = None) -> list[Drift]:
    """Validate the raw drift dicts and turn them into Drift objects, raising on any problem."""
    if not isinstance(drifts, list):
        raise ValueError("drifts must be a list of drift dicts")
    config = config or GeneratorConfig()

    result: list[Drift] = []
    for index, raw in enumerate(drifts, start=1):
        result.append(_normalize_one(raw, index, config))

    _check_no_overlap(result)
    return result


def _normalize_one(raw: dict[str, Any], index: int, config: GeneratorConfig) -> Drift:
    """Validate a single drift dict and build its Drift object."""
    label = f"drift d{index:02d}"
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a dict, got {raw!r}")

    drift_type = raw.get("type")
    if drift_type not in DRIFT_TYPES:
        raise ValueError(f"{label}: unknown type {drift_type!r}; choose one of {sorted(DRIFT_TYPES)}")
    label = f"{label} ({drift_type})"

    mode = raw.get("mode")
    if mode not in POSITION_KEYS:
        raise ValueError(f"{label}: mode must be 'sudden' or 'gradual', got {mode!r}")

    allowed = {"type", "mode", *POSITION_KEYS[mode], *DRIFT_PARAMS[drift_type]}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label}: unexpected keys {unknown}; a {mode} {drift_type} drift accepts {sorted(allowed)}")
    missing = [key for key in POSITION_KEYS[mode] if key not in raw]
    if missing:
        raise ValueError(f"{label}: a {mode} drift needs {missing}")

    for key in POSITION_KEYS[mode]:
        _check_fraction(raw[key], label, key)
    if mode == "sudden":
        start_frac = end_frac = float(raw["drift_point"])
    else:
        start_frac, end_frac = float(raw["start_point"]), float(raw["end_point"])
        if start_frac >= end_frac:
            raise ValueError(f"{label}: start_point must be smaller than end_point")

    params = {key: value for key, value in raw.items() if key in DRIFT_PARAMS[drift_type]}
    _check_params(drift_type, params, label, config)
    return Drift(index=index, type=drift_type, mode=mode, start_frac=start_frac, end_frac=end_frac, params=params)


def _check_fraction(value: Any, label: str, field_name: str) -> None:
    """Ensure a drift position is a number strictly inside the (0, 1) horizon."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value < 1.0:
        raise ValueError(f"{label}: {field_name} must be between 0 and 1 (exclusive), got {value!r}")


def _check_params(drift_type: str, params: dict[str, Any], label: str, config: GeneratorConfig) -> None:
    """Ensure the type-specific parameters are present and valid, so every drift changes something."""
    if "num_activities" in params:
        check_positive_int(params["num_activities"], f"{label}: num_activities")
    if "tree_weights" in params:
        check_tree_weights(params["tree_weights"], f"{label}: tree_weights")
    if "duration_factor" in params:
        check_positive(params["duration_factor"], f"{label}: duration_factor")
    if "inter_arrival" in params:
        check_positive(params["inter_arrival"], f"{label}: inter_arrival")
    if "mean" in params:
        check_positive(params["mean"], f"{label}: mean")
    if "variance" in params:
        check_variance(params["variance"], f"{label}: variance")
    for key in ("factor", "workload_factor"):
        if key in params:
            check_positive(params[key], f"{label}: {key}")
            if params[key] == 1:
                raise ValueError(f"{label}: {key} of 1 changes nothing")

    if drift_type == "pool_size" and "delta" in params:
        delta = params["delta"]
        if isinstance(delta, bool) or not isinstance(delta, int) or delta == 0:
            raise ValueError(f"{label}: delta must be a non-zero integer, got {delta!r}")
    if drift_type == "workload" and "workload_factor" not in params:
        raise ValueError(f"{label}: needs workload_factor")
    if drift_type in ("waiting_time", "amount") and not {"mean", "variance"} & set(params):
        raise ValueError(f"{label}: needs mean and/or variance")
    if drift_type == "arrival_rate" and len({"inter_arrival", "factor"} & set(params)) != 1:
        raise ValueError(f"{label}: needs exactly one of inter_arrival or factor")
    if drift_type == "region" and config.num_regions < 2:
        raise ValueError(f"{label}: needs num_regions of at least 2")
    if drift_type == "duration":
        _check_resources_spec(params.get("resources", "all"), label)


def _check_resources_spec(spec: Any, label: str) -> None:
    """Ensure a duration drift's `resources` is 'all', a positive count, or a non-empty list of names."""
    is_count = isinstance(spec, int) and not isinstance(spec, bool) and spec > 0
    is_names = isinstance(spec, (list, tuple)) and len(spec) > 0 and all(isinstance(r, str) for r in spec)
    if spec != "all" and not is_count and not is_names:
        raise ValueError(f"{label}: resources must be 'all', a positive count or a list of names, got {spec!r}")


def _check_no_overlap(drifts: list[Drift]) -> None:
    """Ensure drifts of the same type have disjoint windows (sudden drifts may not share a point)."""
    by_type: dict[str, list[Drift]] = {}
    for drift in drifts:
        by_type.setdefault(drift.type, []).append(drift)

    for drift_type, group in by_type.items():
        ordered = sorted(group, key=lambda d: d.start_frac)
        for earlier, later in zip(ordered, ordered[1:]):
            if later.start_frac <= earlier.end_frac:
                raise ValueError(
                    f"drifts {earlier.name} and {later.name} ({drift_type}) overlap: "
                    f"{earlier.name} ends at {earlier.end_frac} and {later.name} starts at {later.start_frac}"
                )


def drifts_of(drifts: list[Drift], drift_type: str) -> list[Drift]:
    """Return all drifts of a given type, ordered by where they start."""
    return sorted((d for d in drifts if d.type == drift_type), key=lambda d: d.start_frac)
