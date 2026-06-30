"""Build process trees from operator weights and play out activity sequences."""

from __future__ import annotations

import contextlib
import random
from typing import Iterator

import numpy as np
from pm4py.algo.simulation.playout.process_tree import algorithm as playout_alg
from pm4py.algo.simulation.tree_generator import algorithm as tree_gen
from pm4py.objects.process_tree.obj import ProcessTree

from rheon.config import activity_labels


@contextlib.contextmanager
def _pm4py_seed(seed: int) -> Iterator[None]:
    """Temporarily seed the global RNGs PM4PY relies on, then restore them."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _next_seed(rng: np.random.Generator) -> int:
    """Draw a fresh integer seed for a PM4PY call from our own generator."""
    return int(rng.integers(0, 2**31 - 1))


def build_tree(
    num_activities: int,
    tree_weights: dict[str, float],
    rng: np.random.Generator,
) -> ProcessTree:
    """Generate a process tree with the given number of activities and operator weights."""
    params = {
        "min": num_activities,
        "max": num_activities,
        "mode": num_activities,
        "sequence": float(tree_weights.get("sequence", 0.6)),
        "choice": float(tree_weights.get("choice", 0.25)),
        "parallel": float(tree_weights.get("parallel", 0.1)),
        "loop": float(tree_weights.get("loop", 0.05)),
        "or": 0.0,
        "silent": 0.0,
        "duplicate": 0.0,
        "no_models": 1,
    }
    with _pm4py_seed(_next_seed(rng)):
        tree = tree_gen.apply(parameters=params)
    _relabel(tree)
    return tree


def _relabel(tree: ProcessTree) -> None:
    """Rename the visible leaves to clean activity names a, b, c, ... in traversal order."""
    leaves = [node for node in _all_nodes(tree) if node.operator is None and node.label is not None]
    labels = activity_labels(len(leaves))
    for leaf, label in zip(leaves, labels):
        leaf.label = label


def activities_in_tree(tree: ProcessTree) -> list[str]:
    """Sorted list of distinct activity names that appear in a tree."""
    labels = {
        str(node.label)
        for node in _all_nodes(tree)
        if node.operator is None and node.label is not None
    }
    return sorted(labels)


def playout_pool(
    tree: ProcessTree,
    size: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    """Play out a pool of traces (activity sequences) to later sample cases from."""
    with _pm4py_seed(_next_seed(rng)):
        log = playout_alg.apply(
            tree,
            variant=playout_alg.Variants.BASIC_PLAYOUT,
            parameters={"num_traces": max(1, size)},
        )
    pool = [_trace_activities(trace) for trace in log]
    pool = [trace for trace in pool if trace]
    if not pool:
        pool = [activities_in_tree(tree)]
    return pool


def sample_trace(pool: list[list[str]], rng: np.random.Generator) -> list[str]:
    """Pick one trace (activity sequence) from a pool."""
    return list(pool[int(rng.integers(0, len(pool)))])


def _trace_activities(trace) -> list[str]:
    """Extract the visible activity names from one played-out PM4PY trace."""
    return [str(event["concept:name"]) for event in trace if event.get("concept:name") is not None]


def _all_nodes(tree: ProcessTree) -> list[ProcessTree]:
    """Every node in the tree (depth-first)."""
    nodes = [tree]
    for child in tree.children:
        nodes.extend(_all_nodes(child))
    return nodes
