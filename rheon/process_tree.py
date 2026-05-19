from __future__ import annotations

import contextlib
import copy
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from pm4py.algo.simulation.playout.process_tree import algorithm as playout_alg
from pm4py.algo.simulation.tree_generator import algorithm as tree_gen
from pm4py.objects.process_tree.obj import Operator, ProcessTree

from rheon.config import GeneratorConfig


@dataclass
class TreeMutationResult:
    tree: ProcessTree
    activities_added: list[str]
    activities_deleted: list[str]
    activities_moved: list[str]
    operator_swaps: list[dict[str, str]]

    def change_details(self) -> dict[str, object]:
        return {
            "activities_added": self.activities_added,
            "activities_deleted": self.activities_deleted,
            "activities_moved": self.activities_moved,
            "operator_swaps": self.operator_swaps,
        }


@contextlib.contextmanager
def pm4py_seed(seed: int) -> Iterator[None]:
    state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(state)
        np.random.set_state(np_state)


def next_pm4py_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def generate_process_tree(config: GeneratorConfig, rng: np.random.Generator) -> ProcessTree:
    params = {
        "mode": int(round((config.min_activities + config.max_activities) / 2)),
        "min": config.min_activities,
        "max": config.max_activities,
        "sequence": config.sequence_weight,
        "choice": config.choice_weight,
        "parallel": config.parallel_weight,
        "loop": config.loop_weight,
        "or": config.or_weight,
        "silent": config.silent_transition_prob,
        "duplicate": config.duplicate_activity_prob,
        "no_models": 1,
    }
    best_tree: ProcessTree | None = None
    best_score = float("inf")
    for _ in range(max(1, config.tree_generation_attempts)):
        with pm4py_seed(next_pm4py_seed(rng)):
            tree = tree_gen.apply(parameters=params)
        score = _trace_length_score(tree, config, rng)
        if score < best_score:
            best_tree = tree
            best_score = score
        if score <= 0.20:
            break
    if best_tree is None:
        raise RuntimeError("PM4PY did not return a process tree")
    return best_tree


def clone_tree(tree: ProcessTree) -> ProcessTree:
    return copy.deepcopy(tree)


def _trace_length_score(
    tree: ProcessTree,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> float:
    sample_size = 256
    with pm4py_seed(next_pm4py_seed(rng)):
        log = playout_alg.apply(
            tree,
            variant=playout_alg.Variants.BASIC_PLAYOUT,
            parameters={"num_traces": sample_size},
        )
    lengths = np.array(
        [
            len([event for event in trace if event.get("concept:name") is not None])
            for trace in log
        ],
        dtype=float,
    )
    if lengths.size == 0:
        return float("inf")
    width = max(1.0, float(config.max_trace_length - config.min_trace_length))
    mean_error = abs(float(lengths.mean()) - float(config.avg_trace_length)) / width
    target_std = float(np.sqrt(max(0.0, config.trace_length_variance)))
    std_error = abs(float(lengths.std()) - target_std) / width
    unique_lengths = len(set(int(length) for length in lengths))
    diversity_penalty = max(0.0, (4.0 - float(unique_lengths)) / 4.0)
    below_share = float(np.mean(lengths < config.min_trace_length))
    above_share = float(np.mean(lengths > config.max_trace_length))
    short_share = float(np.mean(lengths <= config.min_trace_length))
    valid_share = float(np.mean((lengths >= config.min_trace_length) & (lengths <= config.max_trace_length)))
    return (
        mean_error
        + 0.75 * std_error
        + 0.75 * diversity_penalty
        + 2.0 * below_share
        + 1.0 * above_share
        + 0.5 * short_share
        + (1.0 - valid_share)
    )


def activities_in_tree(tree: ProcessTree) -> list[str]:
    labels: list[str] = []
    for leaf in _visible_leaves(tree):
        if leaf.label not in labels:
            labels.append(str(leaf.label))
    return sorted(labels)


def count_operators(tree: ProcessTree, operator: Operator | None = None) -> int:
    nodes = _operator_nodes(tree)
    if operator is None:
        return len(nodes)
    return sum(1 for node in nodes if node.operator is operator)


def estimate_variant_count(tree: ProcessTree, cap: int = 100_000) -> int:
    def walk(node: ProcessTree) -> int:
        if node.operator is None:
            return 1 if node.label is not None else 1
        child_counts = [walk(child) for child in node.children] or [1]
        if node.operator is Operator.XOR:
            value = sum(child_counts)
        elif node.operator is Operator.OR:
            value = max(sum(child_counts), 2 ** min(len(child_counts), 12) - 1)
        elif node.operator is Operator.LOOP:
            value = child_counts[0] * max(1, sum(child_counts[1:]) + 1) * 3
        else:
            value = 1
            for child_count in child_counts:
                value *= child_count
        return min(cap, max(1, int(value)))

    return walk(tree)


def mutate_tree(
    tree: ProcessTree,
    config: GeneratorConfig,
    rng: np.random.Generator,
    intensity: float = 0.25,
) -> TreeMutationResult:
    drift_tree = clone_tree(tree)
    visible_count = max(1, len(_visible_leaves(drift_tree)))
    rounds = max(1, int(np.ceil(visible_count * max(0.05, intensity))))
    added: list[str] = []
    deleted: list[str] = []
    moved: list[str] = []
    swaps: list[dict[str, str]] = []

    operations = ["add", "delete", "move", "operator_swap"]
    for _ in range(rounds):
        op = str(rng.choice(operations))
        if op == "add" and _add_activity(drift_tree, config, rng, added):
            continue
        if op == "delete" and _delete_activity(drift_tree, rng, deleted):
            continue
        if op == "move" and _move_activity(drift_tree, rng, moved):
            continue
        if _swap_operator(drift_tree, rng, swaps, moved):
            continue
        if _add_activity(drift_tree, config, rng, added):
            continue
        if _move_activity(drift_tree, rng, moved):
            continue
        _delete_activity(drift_tree, rng, deleted)

    return TreeMutationResult(
        tree=drift_tree,
        activities_added=added,
        activities_deleted=deleted,
        activities_moved=moved,
        operator_swaps=swaps,
    )


def _add_activity(
    tree: ProcessTree,
    config: GeneratorConfig,
    rng: np.random.Generator,
    added: list[str],
) -> bool:
    existing = set(activities_in_tree(tree))
    available = [activity for activity in config.activity_pool if activity not in existing]
    operators = _operator_nodes(tree)
    if not available or not operators:
        return False
    label = str(rng.choice(available))
    parent = operators[int(rng.integers(0, len(operators)))]
    new_leaf = ProcessTree(label=label, parent=parent)
    parent.children.append(new_leaf)
    added.append(label)
    return True


def _delete_activity(tree: ProcessTree, rng: np.random.Generator, deleted: list[str]) -> bool:
    candidates = [
        leaf
        for leaf in _visible_leaves(tree)
        if leaf.parent is not None and len(leaf.parent.children) > 1
    ]
    if len(_visible_leaves(tree)) <= 1 or not candidates:
        return False
    leaf = candidates[int(rng.integers(0, len(candidates)))]
    label = str(leaf.label)
    leaf.label = None
    deleted.append(label)
    return True


def _move_activity(tree: ProcessTree, rng: np.random.Generator, moved: list[str]) -> bool:
    leaves = [
        leaf
        for leaf in _visible_leaves(tree)
        if leaf.parent is not None and len(leaf.parent.children) > 1
    ]
    operators = _operator_nodes(tree)
    if not leaves or len(operators) < 2:
        return False
    leaf = leaves[int(rng.integers(0, len(leaves)))]
    targets = [
        node
        for node in operators
        if node is not leaf.parent and not _is_descendant(node, leaf)
    ]
    if not targets:
        return False
    old_parent = leaf.parent
    target = targets[int(rng.integers(0, len(targets)))]
    old_parent.children.remove(leaf)
    leaf.parent = target
    target.children.append(leaf)
    moved.append(str(leaf.label))
    return True


def _swap_operator(
    tree: ProcessTree,
    rng: np.random.Generator,
    swaps: list[dict[str, str]],
    moved: list[str],
) -> bool:
    candidates = _operator_nodes(tree)
    if not candidates:
        return False
    node = candidates[int(rng.integers(0, len(candidates)))]
    choices = [Operator.SEQUENCE, Operator.XOR, Operator.PARALLEL, Operator.OR]
    if len(node.children) >= 2:
        choices.append(Operator.LOOP)
    choices = [choice for choice in choices if choice is not node.operator]
    if not choices:
        return False
    old_name = _operator_name(node.operator)
    new_operator = choices[int(rng.integers(0, len(choices)))]
    node.operator = new_operator
    new_name = _operator_name(new_operator)
    affected = [str(leaf.label) for leaf in _visible_leaves(node)]
    moved.extend(affected)
    swaps.append({"from": old_name, "to": new_name, "affected_activities": ",".join(affected)})
    return True


def _visible_leaves(tree: ProcessTree) -> list[ProcessTree]:
    return [leaf for leaf in tree._get_leaves() if leaf.label is not None]


def _operator_nodes(tree: ProcessTree) -> list[ProcessTree]:
    return [node for node in _all_nodes(tree) if node.operator is not None]


def _all_nodes(tree: ProcessTree) -> list[ProcessTree]:
    nodes = [tree]
    for child in tree.children:
        nodes.extend(_all_nodes(child))
    return nodes


def _is_descendant(candidate: ProcessTree, node: ProcessTree) -> bool:
    current = candidate
    while current is not None:
        if current is node:
            return True
        current = current.parent
    return False


def _operator_name(operator: Operator | None) -> str:
    if operator is Operator.SEQUENCE:
        return "sequence"
    if operator is Operator.XOR:
        return "choice"
    if operator is Operator.PARALLEL:
        return "parallel"
    if operator is Operator.LOOP:
        return "loop"
    if operator is Operator.OR:
        return "or"
    return "leaf"
