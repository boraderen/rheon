from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from pm4py.objects.process_tree.obj import ProcessTree

from rheon.config import (
    ACTIVITY_KEY,
    AMOUNT_KEY,
    CASE_ID_KEY,
    CASE_TYPE_KEY,
    DURATION_KEY,
    EVENT_ID_KEY,
    GeneratorConfig,
    REGION_KEY,
    RESOURCE_KEY,
    ROLE_KEY,
    SCHEMA_COLUMNS,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
    make_rng,
    sample_horizon,
)
from rheon.drift.common import sample_drift_plans
from rheon.drift.control_flow import ControlFlowModel, build_control_flow_model
from rheon.drift.data import build_data_runtime
from rheon.drift.inter_case import build_inter_case_runtime, generate_case_arrivals
from rheon.drift.resource import build_resource_runtime
from rheon.metadata import DriftMetadata, metadata_payload
from rheon.noise import inject_noise
from rheon.playout import play_trace_pool, sample_from_pool
from rheon.process_tree import activities_in_tree
from rheon.resource_assignment import assign_resource_and_role
from rheon.timestamp_assignment import sample_service_duration_min
from rheon.xes_writer import write_xes_with_metadata


@dataclass
class GeneratedLog:
    dataframe: pd.DataFrame
    metadata: dict[str, Any]
    xes_path: Path | None = None


def generate_log(
    config: GeneratorConfig,
    drifts: list[dict[str, Any]],
    *,
    log_name: str = "rheon_log",
    default_perspective: str = "control_flow",
    rng: np.random.Generator | None = None,
) -> GeneratedLog:
    rng = make_rng(config.global_seed) if rng is None else rng
    horizon_start, horizon_end = sample_horizon(config, rng)
    plans = sample_drift_plans(
        drifts,
        config=config,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        rng=rng,
        default_perspective=default_perspective,
    )

    control_flow = build_control_flow_model(config, plans, rng)
    activity_pool = _activity_union(control_flow)
    resource_runtime = build_resource_runtime(config, plans, activity_pool, rng)
    inter_case_runtime = build_inter_case_runtime(config, plans, rng)
    data_runtime = build_data_runtime(config, plans, rng)

    arrivals = generate_case_arrivals(config, inter_case_runtime, horizon_start, horizon_end, rng)
    trace_pools = _trace_pools(control_flow, max(1, len(arrivals)), config, rng)
    rows: list[dict[str, object]] = []
    observed_variants_by_tree: dict[int, set[tuple[str, ...]]] = {}

    for case_index, case_start in enumerate(arrivals, start=1):
        case_id = f"case_{case_index:06d}"
        tree = control_flow.tree_for(case_start, rng)
        trace = sample_from_pool(trace_pools[id(tree)], rng)
        observed_variants_by_tree.setdefault(id(tree), set()).add(tuple(trace))
        case_type = inter_case_runtime.sample_case_type(case_start, rng)
        amount, region = data_runtime.sample_amount_region(case_start, rng)
        previous_resource: str | None = None
        cursor = case_start
        for event_index, activity in enumerate(trace, start=1):
            # generate base (unscaled) wait and processing time
            base_wait_min = float(rng.exponential(1.0))
            base_duration_min = sample_service_duration_min(config, rng)
            base_start = cursor + timedelta(minutes=base_wait_min)
            base_end = base_start + timedelta(minutes=base_duration_min)

            # assign resource using the base completion timestamp
            resource, role = assign_resource_and_role(
                resource_runtime,
                str(activity),
                previous_resource,
                base_end,
                rng,
            )
            previous_resource = resource

            # scale both wait and processing time by the resource's speed multiplier
            speed = resource_runtime.speed_for(resource, base_end, rng)
            wait_min = base_wait_min * speed
            duration_min = base_duration_min * speed
            start_timestamp = cursor + timedelta(minutes=wait_min)
            timestamp = start_timestamp + timedelta(minutes=duration_min)
            cursor = timestamp

            rows.append(
                {
                    EVENT_ID_KEY: "",
                    CASE_ID_KEY: case_id,
                    ACTIVITY_KEY: str(activity),
                    START_TIMESTAMP_KEY: start_timestamp,
                    TIMESTAMP_KEY: timestamp,
                    DURATION_KEY: duration_min,
                    RESOURCE_KEY: resource,
                    ROLE_KEY: role,
                    CASE_TYPE_KEY: case_type,
                    AMOUNT_KEY: amount,
                    REGION_KEY: region,
                }
            )

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    if df.empty:
        df = pd.DataFrame(columns=SCHEMA_COLUMNS)
    pre_noise_variant_count = len(set().union(*observed_variants_by_tree.values())) if observed_variants_by_tree else 0
    df, noise_metadata = inject_noise(df, config, activity_pool, rng)
    df = finalize_dataframe(df)
    _enrich_control_flow_variant_counts(control_flow, observed_variants_by_tree)

    metadata = metadata_payload(
        log_name=log_name,
        config=config.to_dict()
        | {
            "horizon_start": horizon_start.isoformat(),
            "horizon_end": horizon_end.isoformat(),
            "actual_num_traces": int(df[CASE_ID_KEY].nunique()) if not df.empty else 0,
            "actual_num_events": int(len(df)),
            "num_trace_variants": _trace_variant_count(df),
            "num_trace_variants_before_noise": pre_noise_variant_count,
        },
        drifts=_all_metadata(control_flow, resource_runtime, inter_case_runtime, data_runtime),
        noise=noise_metadata,
    )
    return GeneratedLog(dataframe=df, metadata=metadata)


def write_generated_log(
    generated: GeneratedLog,
    output_path: str | Path,
) -> GeneratedLog:
    path = write_xes_with_metadata(generated.dataframe, output_path, generated.metadata)
    generated.xes_path = path
    return generated


def generate_and_write_log(
    config: GeneratorConfig,
    drifts: list[dict[str, Any]],
    output_path: str | Path,
    *,
    log_name: str | None = None,
    default_perspective: str = "control_flow",
    rng: np.random.Generator | None = None,
) -> GeneratedLog:
    name = log_name or Path(output_path).stem
    generated = generate_log(
        config,
        drifts,
        log_name=name,
        default_perspective=default_perspective,
        rng=rng,
    )
    return write_generated_log(generated, output_path)


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        result = pd.DataFrame(columns=SCHEMA_COLUMNS)
        return result
    result = df.copy()
    result[START_TIMESTAMP_KEY] = pd.to_datetime(result[START_TIMESTAMP_KEY], utc=True)
    result[TIMESTAMP_KEY] = pd.to_datetime(result[TIMESTAMP_KEY], utc=True)
    result = result.sort_values([CASE_ID_KEY, START_TIMESTAMP_KEY, TIMESTAMP_KEY, ACTIVITY_KEY]).reset_index(drop=True)
    result[DURATION_KEY] = (
        result[TIMESTAMP_KEY] - result[START_TIMESTAMP_KEY]
    ).dt.total_seconds() / 60.0
    result[EVENT_ID_KEY] = [f"evt_{idx:08d}" for idx in range(1, len(result) + 1)]
    for column in [CASE_ID_KEY, ACTIVITY_KEY, RESOURCE_KEY, ROLE_KEY, CASE_TYPE_KEY, REGION_KEY]:
        result[column] = result[column].astype(str)
    result[AMOUNT_KEY] = result[AMOUNT_KEY].astype(float)
    return result[SCHEMA_COLUMNS]


def _trace_pools(
    control_flow: ControlFlowModel,
    num_cases: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> dict[int, list[list[str]]]:
    trees = _all_trees(control_flow)
    per_tree = max(10, min(num_cases, max(config.num_traces, 1)))
    return {
        id(tree): play_trace_pool(tree, per_tree, config, rng)
        for tree in trees
    }


def _all_trees(control_flow: ControlFlowModel) -> list[ProcessTree]:
    trees: list[ProcessTree] = [control_flow.initial_tree]
    seen = {id(control_flow.initial_tree)}
    for drift in control_flow.drifts:
        for tree in drift.versions:
            if id(tree) not in seen:
                trees.append(tree)
                seen.add(id(tree))
    return trees


def _activity_union(control_flow: ControlFlowModel) -> list[str]:
    labels: set[str] = set()
    for tree in _all_trees(control_flow):
        labels.update(activities_in_tree(tree))
    return sorted(labels)


def _all_metadata(*runtimes: object) -> list[DriftMetadata]:
    drifts: list[DriftMetadata] = []
    for runtime in runtimes:
        drifts.extend(getattr(runtime, "metadata", []))
    return sorted(drifts, key=lambda item: item.drift_id)


def _enrich_control_flow_variant_counts(
    control_flow: ControlFlowModel,
    observed_variants_by_tree: dict[int, set[tuple[str, ...]]],
) -> None:
    metadata_by_id = {metadata.drift_id: metadata for metadata in control_flow.metadata}
    for drift in control_flow.drifts:
        counts = [
            len(observed_variants_by_tree.get(id(tree), set()))
            for tree in drift.versions
        ]
        drift.details["variant_counts_by_version"] = counts
        if counts:
            drift.details["variant_count_before"] = counts[0]
            drift.details["variant_count_after"] = counts[-1]

        metadata = metadata_by_id.get(drift.plan.drift_id)
        if metadata is not None:
            metadata.change_details.update(drift.details)


def _trace_variant_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    variants = {
        tuple(group.sort_values([START_TIMESTAMP_KEY, TIMESTAMP_KEY])[ACTIVITY_KEY].astype(str))
        for _, group in df.groupby(CASE_ID_KEY, sort=False)
    }
    return len(variants)
