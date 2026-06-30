"""Drift specifications, validation of the user-supplied drift list, and timeline helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


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

MODES = {"sudden", "gradual"}


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
            "drift_start": self.start_frac,
            "drift_end": self.end_frac,
            "drift_start_date": at(self.start_frac),
            "drift_end_date": at(self.end_frac),
        }


def normalize_drifts(drifts: list[dict[str, Any]]) -> list[Drift]:
    """Validate the raw drift dicts and turn them into Drift objects, raising on any problem."""
    if not isinstance(drifts, list):
        raise ValueError("drifts must be a list of drift dicts")

    result: list[Drift] = []
    for index, raw in enumerate(drifts, start=1):
        result.append(_normalize_one(raw, index))

    _check_no_overlap(result)
    return result


def _normalize_one(raw: dict[str, Any], index: int) -> Drift:
    """Validate a single drift dict and build its Drift object."""
    if not isinstance(raw, dict):
        raise ValueError(f"drift #{index} must be a dict")

    drift_type = raw.get("type")
    if drift_type not in DRIFT_TYPES:
        raise ValueError(
            f"drift #{index}: unknown type {drift_type!r}; "
            f"choose one of {sorted(DRIFT_TYPES)}"
        )

    mode = raw.get("mode", "sudden")
    if mode not in MODES:
        raise ValueError(f"drift #{index}: mode must be 'sudden' or 'gradual', got {mode!r}")

    if mode == "sudden":
        point = float(raw.get("drift_point", 0.5))
        _check_fraction(point, index, "drift_point")
        start_frac = end_frac = point
    else:
        if "start_point" not in raw or "end_point" not in raw:
            raise ValueError(f"drift #{index}: gradual drift needs start_point and end_point")
        start_frac = float(raw["start_point"])
        end_frac = float(raw["end_point"])
        _check_fraction(start_frac, index, "start_point")
        _check_fraction(end_frac, index, "end_point")
        if start_frac >= end_frac:
            raise ValueError(f"drift #{index}: start_point must be smaller than end_point")

    params = {
        key: value
        for key, value in raw.items()
        if key not in {"type", "mode", "drift_point", "start_point", "end_point"}
    }
    return Drift(index=index, type=drift_type, mode=mode, start_frac=start_frac, end_frac=end_frac, params=params)


def _check_fraction(value: float, index: int, field_name: str) -> None:
    """Ensure a drift position is strictly inside the (0, 1) horizon."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"drift #{index}: {field_name} must be between 0 and 1 (exclusive), got {value}")


def _check_no_overlap(drifts: list[Drift]) -> None:
    """Ensure drifts of the same type do not have overlapping transition windows."""
    by_type: dict[str, list[Drift]] = {}
    for drift in drifts:
        by_type.setdefault(drift.type, []).append(drift)

    for drift_type, group in by_type.items():
        ordered = sorted(group, key=lambda d: d.start_frac)
        for earlier, later in zip(ordered, ordered[1:]):
            if later.start_frac < earlier.end_frac:
                raise ValueError(
                    f"two {drift_type} drifts overlap: "
                    f"{earlier.name} ends at {earlier.end_frac} but {later.name} starts at {later.start_frac}"
                )


def drifts_of(drifts: list[Drift], drift_type: str) -> list[Drift]:
    """Return all drifts of a given type, ordered by where they start."""
    return sorted((d for d in drifts if d.type == drift_type), key=lambda d: d.start_frac)
