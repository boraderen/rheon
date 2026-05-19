from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rheon.composer import GeneratedLog, generate_and_write_log
from rheon.config import DEFAULT_REGIONS, DEFAULT_START_TIMESTAMP, GeneratorConfig, make_rng


def generate_log(
    drifts: list[dict[str, Any]],
    output_path: str | Path,
    *,
    log_name: str | None = None,
    rng: int | np.random.Generator | None = None,
    global_seed: int = 42,
    num_traces: int = 2000,
    min_trace_length: int = 5,
    max_trace_length: int = 50,
    avg_trace_length: int = 15,
    trace_length_variance: int = 25,
    horizon_min_days: int = 30,
    horizon_max_days: int = 365,
    min_activities: int = 6,
    max_activities: int = 20,
    tree_depth_min: int = 3,
    tree_depth_max: int = 6,
    tree_generation_attempts: int = 32,
    sequence_weight: float = 0.70,
    choice_weight: float = 0.20,
    parallel_weight: float = 0.07,
    loop_weight: float = 0.02,
    or_weight: float = 0.01,
    silent_transition_prob: float = 0.05,
    duplicate_activity_prob: float = 0.0,
    num_resources: int = 20,
    num_roles: int = 5,
    num_case_types: int = 3,
    regions: list[str] | None = None,
    inter_arrival_mean_min: float = 5.0,
    service_time_mean_min: float = 10.0,
    service_time_std_min: float = 5.0,
    noise_probability: float = 0.05,
    noise_similar_vs_random: float = 0.5,
    gradual_overlap_fraction: float = 0.10,
    recurring_period_fraction: float = 0.20,
    start_timestamp: datetime | None = None,
) -> GeneratedLog:
    config = GeneratorConfig(
        global_seed=global_seed,
        num_traces=num_traces,
        min_trace_length=min_trace_length,
        max_trace_length=max_trace_length,
        avg_trace_length=avg_trace_length,
        trace_length_variance=trace_length_variance,
        horizon_min_days=horizon_min_days,
        horizon_max_days=horizon_max_days,
        min_activities=min_activities,
        max_activities=max_activities,
        tree_generation_attempts=tree_generation_attempts,
        sequence_weight=sequence_weight,
        choice_weight=choice_weight,
        parallel_weight=parallel_weight,
        loop_weight=loop_weight,
        or_weight=or_weight,
        silent_transition_prob=silent_transition_prob,
        duplicate_activity_prob=duplicate_activity_prob,
        num_resources=num_resources,
        num_roles=num_roles,
        num_case_types=num_case_types,
        regions=list(DEFAULT_REGIONS) if regions is None else regions,
        inter_arrival_mean_min=inter_arrival_mean_min,
        service_time_mean_min=service_time_mean_min,
        service_time_std_min=service_time_std_min,
        noise_probability=noise_probability,
        noise_similar_vs_random=noise_similar_vs_random,
        gradual_overlap_fraction=gradual_overlap_fraction,
        recurring_period_fraction=recurring_period_fraction,
        start_timestamp=DEFAULT_START_TIMESTAMP if start_timestamp is None else start_timestamp,
    )
    resolved_rng = _resolve_rng(rng)
    return generate_and_write_log(
        config,
        drifts,
        output_path,
        log_name=log_name,
        rng=resolved_rng,
    )


def _resolve_rng(rng: int | np.random.Generator | None) -> np.random.Generator | None:
    if rng is None:
        return None
    return make_rng(rng)
