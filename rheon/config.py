"""Column keys, label helpers and the bundle of generation parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


# --- XES / DataFrame column names -------------------------------------------

EVENT_ID_KEY = "event:id"
CASE_ID_KEY = "case:concept:name"
ACTIVITY_KEY = "concept:name"
START_TIMESTAMP_KEY = "start_timestamp"
TIMESTAMP_KEY = "time:timestamp"
DURATION_KEY = "event:duration_min"
RESOURCE_KEY = "org:resource"
AMOUNT_KEY = "case:amount"
REGION_KEY = "case:region"

SCHEMA_COLUMNS = [
    EVENT_ID_KEY,
    CASE_ID_KEY,
    ACTIVITY_KEY,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
    DURATION_KEY,
    RESOURCE_KEY,
    AMOUNT_KEY,
    REGION_KEY,
]


# --- Defaults ----------------------------------------------------------------

DEFAULT_TREE_WEIGHTS = {"sequence": 0.6, "choice": 0.25, "parallel": 0.1, "loop": 0.05}
DEFAULT_START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
DEFAULT_END_DATE = datetime(2020, 12, 31, tzinfo=timezone.utc)


def make_rng(seed: int | np.random.Generator) -> np.random.Generator:
    """Return a numpy Generator, passing an existing Generator through unchanged."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def activity_labels(count: int) -> list[str]:
    """Return `count` short activity names: a, b, ..., z, aa, ab, ..."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    labels: list[str] = []
    for idx in range(count):
        if idx < len(alphabet):
            labels.append(alphabet[idx])
        else:
            labels.append(alphabet[(idx // len(alphabet)) - 1] + alphabet[idx % len(alphabet)])
    return labels


def resource_label(index: int) -> str:
    """Return the canonical name of the resource with the given 1-based index."""
    return f"res_{index:02d}"


def region_label(index: int) -> str:
    """Return the canonical name of the region with the given 1-based index."""
    return f"region_{index}"


@dataclass(frozen=True)
class GeneratorConfig:
    """All structural, temporal and attribute parameters for one base process tree."""

    num_traces: int = 1000
    num_activities: int = 10
    num_resources: int = 8
    num_regions: int = 4

    tree_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TREE_WEIGHTS))

    start_date: datetime = DEFAULT_START_DATE
    end_date: datetime = DEFAULT_END_DATE

    inter_arrival: float = 60.0          # mean minutes between case starts
    activity_duration: tuple[float, float] = (30.0, 100.0)  # (mean, variance) minutes
    waiting_time: tuple[float, float] = (15.0, 50.0)        # (mean, variance) minutes
    amount: tuple[float, float] = (1000.0, 40000.0)         # (mean, variance)

    seed: int = 42

    @property
    def activities(self) -> list[str]:
        """The base activity names a, b, c, ..."""
        return activity_labels(self.num_activities)

    @property
    def resources(self) -> list[str]:
        """The base resource pool res_01, res_02, ..."""
        return [resource_label(idx) for idx in range(1, self.num_resources + 1)]

    @property
    def regions(self) -> list[str]:
        """The region pool region_1, region_2, ..."""
        return [region_label(idx) for idx in range(1, self.num_regions + 1)]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of the parameters for metadata (dates as ISO strings)."""
        return {
            "num_traces": self.num_traces,
            "num_activities": self.num_activities,
            "num_resources": self.num_resources,
            "num_regions": self.num_regions,
            "tree_weights": dict(self.tree_weights),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "inter_arrival": self.inter_arrival,
            "activity_duration": list(self.activity_duration),
            "waiting_time": list(self.waiting_time),
            "amount": list(self.amount),
            "seed": self.seed,
        }
