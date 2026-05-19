from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np


EVENT_ID_KEY = "event:id"
CASE_ID_KEY = "case:concept:name"
ACTIVITY_KEY = "concept:name"
START_TIMESTAMP_KEY = "start_timestamp"
TIMESTAMP_KEY = "time:timestamp"
DURATION_KEY = "event:duration_min"
RESOURCE_KEY = "org:resource"
ROLE_KEY = "org:role"
CASE_TYPE_KEY = "case:case_type"
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
    ROLE_KEY,
    CASE_TYPE_KEY,
    AMOUNT_KEY,
    REGION_KEY,
]

LEGACY_TO_NATIVE_COLUMNS = {
    "event_id": EVENT_ID_KEY,
    "case_id": CASE_ID_KEY,
    "activity": ACTIVITY_KEY,
    "start_timestamp": START_TIMESTAMP_KEY,
    "timestamp": TIMESTAMP_KEY,
    "event_duration_min": DURATION_KEY,
    "resource": RESOURCE_KEY,
    "role": ROLE_KEY,
    "case_type": CASE_TYPE_KEY,
    "amount": AMOUNT_KEY,
    "region": REGION_KEY,
}


DEFAULT_REGIONS = ["DE-NRW", "DE-BY", "DE-HE", "DE-BW", "DE-BE"]
DEFAULT_START_TIMESTAMP = datetime(2020, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class GeneratorConfig:
    global_seed: int = 42

    num_traces: int = 2000
    min_trace_length: int = 5
    max_trace_length: int = 50
    avg_trace_length: int = 15
    trace_length_variance: int = 25

    horizon_min_days: int = 30
    horizon_max_days: int = 365

    min_activities: int = 6
    max_activities: int = 20
    tree_depth_min: int = 3
    tree_depth_max: int = 6
    tree_generation_attempts: int = 32
    sequence_weight: float = 0.70
    choice_weight: float = 0.20
    parallel_weight: float = 0.07
    loop_weight: float = 0.02
    or_weight: float = 0.01
    silent_transition_prob: float = 0.05
    duplicate_activity_prob: float = 0.0

    num_resources: int = 20
    num_roles: int = 5
    num_case_types: int = 3
    regions: list[str] = field(default_factory=lambda: list(DEFAULT_REGIONS))

    inter_arrival_mean_min: float = 5.0
    service_time_mean_min: float = 10.0
    service_time_std_min: float = 5.0

    noise_probability: float = 0.05
    noise_similar_vs_random: float = 0.5

    gradual_overlap_fraction: float = 0.10
    recurring_period_fraction: float = 0.20
    start_timestamp: datetime = DEFAULT_START_TIMESTAMP

    @classmethod
    def from_uppercase(cls, values: dict[str, Any]) -> "GeneratorConfig":
        mapping = {config_field.name.upper(): config_field.name for config_field in fields(cls)}
        kwargs = {attr: values[key] for key, attr in mapping.items() if key in values}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_timestamp"] = self.start_timestamp.isoformat()
        return data

    @property
    def resources(self) -> list[str]:
        return [f"res_{idx:03d}" for idx in range(1, self.num_resources + 1)]

    @property
    def roles(self) -> list[str]:
        return [f"role_{idx:02d}" for idx in range(1, self.num_roles + 1)]

    @property
    def case_types(self) -> list[str]:
        return [f"type_{idx:02d}" for idx in range(1, self.num_case_types + 1)]

    @property
    def activity_pool(self) -> list[str]:
        return activity_labels(self.max_activities)


def make_rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def activity_labels(count: int) -> list[str]:
    labels: list[str] = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx in range(count):
        if idx < len(alphabet):
            labels.append(alphabet[idx])
        else:
            first = alphabet[(idx // len(alphabet)) - 1]
            second = alphabet[idx % len(alphabet)]
            labels.append(first + second)
    return labels


def sample_horizon(config: GeneratorConfig, rng: np.random.Generator) -> tuple[datetime, datetime]:
    base_days = float(rng.uniform(config.horizon_min_days, config.horizon_max_days))
    baseline_events = 2000 * 15
    requested_events = max(1, config.num_traces * config.avg_trace_length)
    scaled_days = max(1.0, base_days * requested_events / baseline_events)
    start = config.start_timestamp
    return start, start + timedelta(days=scaled_days)
