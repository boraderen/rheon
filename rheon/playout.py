from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pm4py.algo.simulation.playout.process_tree import algorithm as playout_alg
from pm4py.objects.process_tree.obj import ProcessTree

from rheon.config import GeneratorConfig
from rheon.process_tree import next_pm4py_seed, pm4py_seed


def play_trace_pool(
    tree: ProcessTree,
    num_traces: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[list[str]]:
    target = max(1, num_traces)
    candidates = _valid_extensive_traces(tree, target, config, rng)
    candidates.extend(_valid_stochastic_traces(tree, target, config, rng))
    buckets = _bucket_by_length(candidates, config)
    if not buckets:
        buckets = _fallback_buckets(config, target, rng)
    return _sample_by_target_length(buckets, target, config, rng)


def normalize_trace_length(
    activities: Sequence[str],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[str]:
    trace = list(activities)
    if len(trace) > config.max_trace_length:
        return trace[: config.max_trace_length]
    return trace


def sample_from_pool(pool: list[list[str]], rng: np.random.Generator) -> list[str]:
    if not pool:
        raise ValueError("trace pool is empty")
    return list(pool[int(rng.integers(0, len(pool)))])


def _valid_extensive_traces(
    tree: ProcessTree,
    target: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[list[str]]:
    max_limit = min(50_000, max(100, target * 2))
    try:
        with pm4py_seed(next_pm4py_seed(rng)):
            log = playout_alg.apply(
                tree,
                variant=playout_alg.Variants.EXTENSIVE,
                parameters={
                    "min_trace_length": config.min_trace_length,
                    "max_trace_length": config.max_trace_length,
                    "max_loop_occ": max(1, config.max_trace_length // 2),
                    "max_limit_num_traces": max_limit,
                },
            )
    except Exception:
        return []
    return [
        trace
        for trace in (_trace_activities(trace) for trace in log)
        if _valid_length(trace, config)
    ]


def _valid_stochastic_traces(
    tree: ProcessTree,
    target: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[list[str]]:
    traces: list[list[str]] = []
    goal = min(50_000, max(target * 3, target + 100))
    attempts = 0
    while len(traces) < goal and attempts < 8:
        attempts += 1
        batch_size = min(50_000, max(target, int((goal - len(traces)) * 2.0)))
        with pm4py_seed(next_pm4py_seed(rng)):
            log = playout_alg.apply(
                tree,
                variant=playout_alg.Variants.BASIC_PLAYOUT,
                parameters={"num_traces": batch_size},
            )
        for trace in log:
            activities = _trace_activities(trace)
            if _valid_length(activities, config):
                traces.append(activities)
            if len(traces) >= goal:
                break
    return traces


def _trace_activities(trace: Sequence[object]) -> list[str]:
    activities: list[str] = []
    for event in trace:
        name = event.get("concept:name") if hasattr(event, "get") else None
        if name is not None:
            activities.append(str(name))
    return activities


def _valid_length(trace: Sequence[str], config: GeneratorConfig) -> bool:
    return config.min_trace_length <= len(trace) <= config.max_trace_length


def _bucket_by_length(
    traces: list[list[str]],
    config: GeneratorConfig,
) -> dict[int, list[list[str]]]:
    buckets: dict[int, list[list[str]]] = {}
    for trace in traces:
        if _valid_length(trace, config):
            buckets.setdefault(len(trace), []).append(trace)
    for length, bucket in buckets.items():
        buckets[length] = sorted(bucket)
    return buckets


def _fallback_buckets(
    config: GeneratorConfig,
    target: int,
    rng: np.random.Generator,
) -> dict[int, list[list[str]]]:
    buckets: dict[int, list[list[str]]] = {}
    for length in _target_lengths(config, target, rng):
        pool = config.activity_pool[: max(1, config.max_activities)]
        trace = [str(pool[int(rng.integers(0, len(pool)))]) for _ in range(int(length))]
        buckets.setdefault(int(length), []).append(trace)
    return buckets


def _sample_by_target_length(
    buckets: dict[int, list[list[str]]],
    count: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[list[str]]:
    available_lengths = np.array(sorted(buckets), dtype=int)
    sampled: list[list[str]] = []
    for target_length in _target_lengths(config, count, rng):
        length = int(_nearest_length(int(target_length), available_lengths))
        bucket = buckets[length]
        sampled.append(list(bucket[int(rng.integers(0, len(bucket)))]))
    return sampled


def _target_lengths(
    config: GeneratorConfig,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    low = int(config.min_trace_length)
    high = int(config.max_trace_length)
    if high < low:
        raise ValueError("max_trace_length must be greater than or equal to min_trace_length")
    mean = min(float(high), max(float(low), float(config.avg_trace_length)))
    std = float(np.sqrt(max(0.0, config.trace_length_variance)))
    if std <= 0:
        return np.full(count, int(round(mean)), dtype=int)

    values: list[int] = []
    while len(values) < count:
        samples = rng.normal(mean, std, size=max(128, (count - len(values)) * 3))
        samples = np.rint(samples).astype(int)
        accepted = samples[(samples >= low) & (samples <= high)]
        values.extend(int(value) for value in accepted[: count - len(values)])
    return np.array(values, dtype=int)


def _nearest_length(target_length: int, available_lengths: np.ndarray) -> int:
    if available_lengths.size == 0:
        raise ValueError("no trace lengths are available")
    index = int(np.searchsorted(available_lengths, target_length))
    if index <= 0:
        return int(available_lengths[0])
    if index >= len(available_lengths):
        return int(available_lengths[-1])
    left = int(available_lengths[index - 1])
    right = int(available_lengths[index])
    return left if abs(target_length - left) <= abs(right - target_length) else right
