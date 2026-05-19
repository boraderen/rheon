from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from rheon.config import GeneratorConfig


def sample_service_duration_min(
    config: GeneratorConfig, rng: np.random.Generator
) -> float:
    value = float(
        rng.normal(config.service_time_mean_min, max(0.001, config.service_time_std_min))
    )
    return max(0.1, value)


def assign_event_times(
    case_start: datetime,
    trace_length: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[tuple[datetime, datetime, float]]:
    times: list[tuple[datetime, datetime, float]] = []
    cursor = case_start
    for _ in range(trace_length):
        wait_min = float(rng.exponential(1.0))
        start = cursor + timedelta(minutes=wait_min)
        duration_min = sample_service_duration_min(config, rng)
        end = start + timedelta(minutes=duration_min)
        times.append((start, end, duration_min))
        cursor = end
    return times
