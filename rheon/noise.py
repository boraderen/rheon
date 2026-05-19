from __future__ import annotations

import numpy as np
import pandas as pd

from rheon.config import (
    ACTIVITY_KEY,
    CASE_ID_KEY,
    DURATION_KEY,
    GeneratorConfig,
    RESOURCE_KEY,
    ROLE_KEY,
    SCHEMA_COLUMNS,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
)
from rheon.timestamp_assignment import assign_event_times


def inject_noise(
    df: pd.DataFrame,
    config: GeneratorConfig,
    activity_pool: list[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if df.empty or config.noise_probability <= 0:
        return df.copy(), {"noise_probability": 0.0, "noisy_traces": 0}
    case_ids = sorted(df[CASE_ID_KEY].unique())
    noisy_count = int(round(len(case_ids) * config.noise_probability))
    if noisy_count <= 0:
        return df.copy(), {"noise_probability": config.noise_probability, "noisy_traces": 0}
    selected = set(rng.choice(case_ids, size=min(noisy_count, len(case_ids)), replace=False))
    frames: list[pd.DataFrame] = []
    similar_count = 0
    random_count = 0
    for case_id, group in df.groupby(CASE_ID_KEY, sort=False):
        trace = group.sort_values(TIMESTAMP_KEY).reset_index(drop=True).copy()
        if case_id in selected:
            if rng.random() < config.noise_similar_vs_random:
                trace = _similar_trace_noise(trace, config, rng)
                similar_count += 1
            else:
                trace = _fully_random_trace_noise(trace, config, activity_pool, rng)
                random_count += 1
        frames.append(trace)
    result = pd.concat(frames, ignore_index=True)
    result = result[SCHEMA_COLUMNS]
    return result, {
        "noise_probability": config.noise_probability,
        "similar_vs_random": config.noise_similar_vs_random,
        "noisy_traces": noisy_count,
        "similar_trace_noise": similar_count,
        "fully_random_noise": random_count,
    }


def _similar_trace_noise(
    trace: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    activities = trace[ACTIVITY_KEY].astype(str).tolist()
    operations = ["delete", "swap", "duplicate"]
    for _ in range(int(rng.integers(1, 3))):
        op = str(rng.choice(operations))
        if op == "delete" and len(activities) > max(1, config.min_trace_length):
            del activities[int(rng.integers(0, len(activities)))]
        elif op == "swap" and len(activities) >= 2:
            idx = int(rng.integers(0, len(activities) - 1))
            activities[idx], activities[idx + 1] = activities[idx + 1], activities[idx]
        elif op == "duplicate" and len(activities) < config.max_trace_length:
            idx = int(rng.integers(0, len(activities)))
            activities.insert(idx, activities[idx])
    return _rebuild_trace(trace, activities, config, rng)


def _fully_random_trace_noise(
    trace: pd.DataFrame,
    config: GeneratorConfig,
    activity_pool: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    length = int(rng.integers(config.min_trace_length, config.max_trace_length + 1))
    activities = [str(rng.choice(activity_pool)) for _ in range(length)]
    return _rebuild_trace(trace, activities, config, rng, randomize_resource=True)


def _rebuild_trace(
    trace: pd.DataFrame,
    activities: list[str],
    config: GeneratorConfig,
    rng: np.random.Generator,
    *,
    randomize_resource: bool = False,
) -> pd.DataFrame:
    template = trace.iloc[0].to_dict()
    case_start = pd.to_datetime(trace[START_TIMESTAMP_KEY].min()).to_pydatetime()
    event_times = assign_event_times(case_start, len(activities), config, rng)
    role_by_resource = {
        resource: config.roles[idx % len(config.roles)]
        for idx, resource in enumerate(config.resources)
    }
    rows = []
    for idx, activity in enumerate(activities):
        row = dict(template)
        source_idx = min(idx, len(trace) - 1)
        if not randomize_resource:
            row.update(trace.iloc[source_idx].to_dict())
        else:
            resource = str(rng.choice(config.resources))
            row[RESOURCE_KEY] = resource
            row[ROLE_KEY] = role_by_resource[resource]
        start, end, duration = event_times[idx]
        row[ACTIVITY_KEY] = activity
        row[START_TIMESTAMP_KEY] = start
        row[TIMESTAMP_KEY] = end
        row[DURATION_KEY] = duration
        rows.append(row)
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
