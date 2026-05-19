from __future__ import annotations

from dataclasses import replace

from pm4py.objects.process_tree.obj import Operator, ProcessTree

from rheon.config import make_rng
from rheon.playout import play_trace_pool
from rheon.process_tree import generate_process_tree


def test_playout_returns_schema_activity_traces(small_config, rng):
    tree = generate_process_tree(small_config, rng)
    traces = play_trace_pool(tree, 12, small_config, rng)

    assert len(traces) == 12
    assert all(small_config.min_trace_length <= len(trace) <= small_config.max_trace_length for trace in traces)
    assert all(set(trace).issubset(set(small_config.activity_pool)) for trace in traces)


def test_playout_is_reproducible_for_same_seed(small_config):
    rng_one = make_rng(99)
    tree_one = generate_process_tree(small_config, rng_one)
    traces_one = play_trace_pool(tree_one, 8, small_config, rng_one)

    rng_two = make_rng(99)
    tree_two = generate_process_tree(small_config, rng_two)
    traces_two = play_trace_pool(tree_two, 8, small_config, rng_two)

    assert traces_one == traces_two


def test_playout_uses_target_length_fallback_instead_of_min_padding(small_config):
    config = replace(
        small_config,
        min_trace_length=8,
        max_trace_length=20,
        avg_trace_length=15,
        trace_length_variance=9,
    )
    tree = ProcessTree(operator=Operator.SEQUENCE)
    for label in ["a", "b"]:
        tree.children.append(ProcessTree(label=label, parent=tree))

    traces = play_trace_pool(tree, 200, config, make_rng(1234))
    lengths = [len(trace) for trace in traces]

    assert all(config.min_trace_length <= length <= config.max_trace_length for length in lengths)
    assert len(set(lengths)) > 1
    assert lengths.count(config.min_trace_length) < len(lengths) * 0.5
