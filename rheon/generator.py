"""The generation engine: turn parameters and drifts into an event-log DataFrame plus metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheon.config import (
    ACTIVITY_KEY,
    AMOUNT_KEY,
    CASE_ID_KEY,
    DURATION_KEY,
    EVENT_ID_KEY,
    GeneratorConfig,
    REGION_KEY,
    RESOURCE_KEY,
    SCHEMA_COLUMNS,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
    make_rng,
    resource_label,
)
from rheon.drifts import Drift, drifts_of, normalize_drifts
from rheon.metadata import build_metadata
from rheon.process_tree import activities_in_tree, build_tree, playout_pool, sample_trace
from rheon.xes import write_csv, write_metadata_file, write_xes


# --- Internal working structures --------------------------------------------


@dataclass
class Distributions:
    """Per-activity timing and dominance distributions plus the case-level amount/region defaults."""

    activities: list[str]
    duration_mean: dict[str, float]
    duration_var: float
    waiting_mean: dict[str, float]
    waiting_var: float
    dominant_resource: dict[str, str]
    dominant_region: str
    amount_mean: float
    amount_var: float


@dataclass
class Case:
    """One in-progress case with per-event timing/resource lists and case-level attributes."""

    start: datetime
    activities: list[str]
    gaps: list[float]          # minutes before each event; gaps[0] is 0
    durations: list[float]     # minutes of each event
    resources: list[str]
    base_starts: list[datetime]  # un-drifted reference timeline used to place drifts
    region: str
    amount: float
    case_id: str = ""


# --- Public entry point ------------------------------------------------------


def generate_log(
    drifts: list[dict[str, Any]],
    output_path: str | Path,
    *,
    log_name: str | None = None,
    format: str = "xes",
    **params: Any,
) -> None:
    """Generate a drifted event log and save it as XES or CSV, with a metadata sidecar."""
    if format not in {"xes", "csv"}:
        raise ValueError(f"format must be 'xes' or 'csv', got {format!r}")
    config = GeneratorConfig(**params)
    parsed = normalize_drifts(drifts)
    rng = make_rng(config.seed)

    output = Path(output_path).with_suffix(f".{format}")
    name = log_name or Path(output_path).stem
    dataframe, metadata = _generate(config, parsed, name, rng)

    if format == "csv":
        write_csv(dataframe, output)
    else:
        write_xes(dataframe, output, metadata)
    write_metadata_file(metadata, output)


# --- Orchestration (the six generation steps) -------------------------------


def _generate(
    config: GeneratorConfig,
    drifts: list[Drift],
    log_name: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full pipeline and return the finished DataFrame and its metadata."""
    # 1. control flow: build the tree versions and the activity universe
    versions, pools = _build_versions(config, drifts, rng)
    activities = sorted({a for tree in versions for a in activities_in_tree(tree)})
    distributions = _init_distributions(config, activities, rng)

    # 2. case start times along the horizon
    arrivals, horizon_end = _generate_arrivals(config, drifts_of(drifts, "arrival_rate"), rng)
    position = _position_fn(config.start_date, horizon_end)

    # 3-4. activity sequences, base event times and attribute columns
    cases = _build_cases(config, drifts, versions, pools, arrivals, distributions, position, rng)

    # 5. apply the remaining drifts (attributes / timing first, workload last)
    records, state = _apply_drifts(config, cases, drifts, distributions, position, rng)
    workload_record = _apply_workload(config, cases, drifts, distributions, state, horizon_end, rng)

    # 6. materialise the event table and the metadata
    dataframe = _assemble(cases)
    metadata = build_metadata(
        log_name=log_name,
        config=config,
        horizon_end=horizon_end,
        drifts=drifts,
        distributions=distributions,
        drift_records=records,
        workload_record=workload_record,
        final_state=state,
        dataframe=dataframe,
    )
    return dataframe, metadata


# --- Step 1: control-flow tree versions -------------------------------------


def _build_versions(
    config: GeneratorConfig,
    drifts: list[Drift],
    rng: np.random.Generator,
) -> tuple[list, list]:
    """Build the base tree plus one extra tree per control-flow drift, with a playout pool each."""
    versions = [build_tree(config.num_activities, config.tree_weights, rng)]
    for drift in drifts_of(drifts, "control_flow"):
        num_activities = int(drift.params.get("num_activities", config.num_activities))
        tree_weights = drift.params.get("tree_weights", config.tree_weights)
        versions.append(build_tree(num_activities, tree_weights, rng))
    pool_size = max(200, config.num_traces)
    pools = [playout_pool(tree, pool_size, rng) for tree in versions]
    return versions, pools


def _tree_version_for(
    cf_drifts: list[Drift],
    position_value: float,
    rng: np.random.Generator,
) -> int:
    """Pick which tree version a case at this horizon position should be played out from."""
    version = 0
    for offset, drift in enumerate(cf_drifts, start=1):
        ramp = drift.ramp(position_value)
        if ramp <= 0.0:
            break
        if ramp >= 1.0:
            version = offset
            continue
        return offset if rng.random() < ramp else version
    return version


# --- Step 1b: per-activity distributions ------------------------------------


def _init_distributions(
    config: GeneratorConfig,
    activities: list[str],
    rng: np.random.Generator,
) -> Distributions:
    """Initialise each activity's duration/waiting means, dominant resource and the case defaults."""
    dur_mean, dur_var = config.activity_duration
    wait_mean, wait_var = config.waiting_time
    amount_mean, amount_var = config.amount
    resources = config.resources
    return Distributions(
        activities=activities,
        duration_mean={a: dur_mean * float(rng.uniform(0.6, 1.4)) for a in activities},
        duration_var=float(dur_var),
        waiting_mean={a: wait_mean * float(rng.uniform(0.6, 1.4)) for a in activities},
        waiting_var=float(wait_var),
        dominant_resource={a: str(rng.choice(resources)) for a in activities},
        dominant_region=str(rng.choice(config.regions)),
        amount_mean=float(amount_mean),
        amount_var=float(amount_var),
    )


# --- Step 2: case arrivals ---------------------------------------------------


def _generate_arrivals(
    config: GeneratorConfig,
    arrival_drifts: list[Drift],
    rng: np.random.Generator,
) -> tuple[list[datetime], datetime]:
    """Fill the fixed [start_date, end_date] horizon with case starts; the count is approximate."""
    start, end = config.start_date, config.end_date
    horizon_seconds = max(1.0, (end - start).total_seconds())
    base_mean = config.base_inter_arrival
    max_cases = config.num_traces * 5 + 100  # safety cap if a drift makes arrivals very dense
    starts = [start]
    cursor = start
    while len(starts) < max_cases:
        position_value = (cursor - start).total_seconds() / horizon_seconds
        mean = _effective_inter_arrival(base_mean, arrival_drifts, position_value)
        cursor = cursor + timedelta(minutes=float(rng.exponential(max(1e-6, mean))))
        if cursor > end:
            break
        starts.append(cursor)
    return starts, end


def _effective_inter_arrival(base: float, arrival_drifts: list[Drift], position_value: float) -> float:
    """Blend the base inter-arrival mean with any active arrival_rate drift targets."""
    mean = base
    for drift in arrival_drifts:
        ramp = drift.ramp(position_value)
        if ramp <= 0.0:
            continue
        target = float(drift.params.get("inter_arrival", base * float(drift.params.get("factor", 1.0))))
        mean = mean * (1.0 - ramp) + target * ramp
    return mean


# --- Steps 3-4: build cases --------------------------------------------------


def _build_cases(
    config: GeneratorConfig,
    drifts: list[Drift],
    versions: list,
    pools: list,
    arrivals: list[datetime],
    dist: Distributions,
    position: Any,
    rng: np.random.Generator,
) -> list[Case]:
    """Create every case: its activity sequence, base event timing and attribute columns."""
    cf_drifts = drifts_of(drifts, "control_flow")
    cases: list[Case] = []
    for start in arrivals:
        version = _tree_version_for(cf_drifts, position(start), rng)
        activities = sample_trace(pools[version], rng)

        gaps = [0.0]
        durations = [_positive_normal(dist.duration_mean[activities[0]], dist.duration_var, rng)]
        for activity in activities[1:]:
            gaps.append(_positive_normal(dist.waiting_mean[activity], dist.waiting_var, rng))
            durations.append(_positive_normal(dist.duration_mean[activity], dist.duration_var, rng))

        base_starts = _base_timeline(start, gaps, durations)
        resources = [_draw_dominant(dist.dominant_resource[a], config.resources, rng) for a in activities]
        cases.append(
            Case(
                start=start,
                activities=activities,
                gaps=gaps,
                durations=durations,
                resources=resources,
                base_starts=base_starts,
                region=_draw_dominant(dist.dominant_region, config.regions, rng),
                amount=_positive_normal(dist.amount_mean, dist.amount_var, rng),
            )
        )
    return cases


def _base_timeline(start: datetime, gaps: list[float], durations: list[float]) -> list[datetime]:
    """Compute each event's un-drifted start time from the gaps and durations."""
    base_starts: list[datetime] = []
    cursor = start
    for gap, duration in zip(gaps, durations):
        event_start = cursor + timedelta(minutes=gap)
        base_starts.append(event_start)
        cursor = event_start + timedelta(minutes=duration)
    return base_starts


# --- Step 5: apply the non-control-flow drifts ------------------------------


def _apply_drifts(
    config: GeneratorConfig,
    cases: list[Case],
    drifts: list[Drift],
    dist: Distributions,
    position: Any,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply attribute and timing drifts in chronological order, recording what each changed."""
    state = {
        "dominant_resource": dict(dist.dominant_resource),
        "active_resources": list(config.resources),
        "all_resources": list(config.resources),
        "dominant_region": dist.dominant_region,
        "amount_mean": dist.amount_mean,
        "amount_var": dist.amount_var,
    }
    handlers = {
        "reassignment": _apply_reassignment,
        "region": _apply_region,
        "amount": _apply_amount,
        "waiting_time": _apply_waiting_time,
        "duration": _apply_duration,
        "pool_size": _apply_pool_size,
    }
    records: dict[str, dict[str, Any]] = {}
    ordered = sorted(
        (d for d in drifts if d.type in handlers),
        key=lambda d: d.start_frac,
    )
    for drift in ordered:
        records[drift.name] = handlers[drift.type](config, cases, drift, dist, state, position, rng)
    return records, state


def _apply_reassignment(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Pick a new dominant resource per activity and re-fill the resource column after the drift."""
    old = dict(state["dominant_resource"])
    active = state["active_resources"]
    new = {}
    for activity in dist.activities:
        choices = [r for r in active if r != old.get(activity)]
        new[activity] = str(rng.choice(choices)) if choices else old.get(activity, active[0])
    for case in cases:
        for index, base_start in enumerate(case.base_starts):
            ramp = drift.ramp(position(base_start))
            if ramp > 0 and (drift.mode == "sudden" or rng.random() < ramp):
                case.resources[index] = _draw_dominant(new[case.activities[index]], active, rng)
    state["dominant_resource"] = new
    return {"dominant_before": old, "dominant_after": new}


def _apply_region(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Choose a new dominant region and re-fill the region column for later cases."""
    old = state["dominant_region"]
    choices = [r for r in config.regions if r != old]
    new = str(rng.choice(choices)) if choices else old
    for case in cases:
        ramp = drift.ramp(position(case.start))
        if ramp > 0 and (drift.mode == "sudden" or rng.random() < ramp):
            case.region = _draw_dominant(new, config.regions, rng)
    state["dominant_region"] = new
    return {"dominant_region_before": old, "dominant_region_after": new}


def _apply_amount(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Re-draw case amounts from a shifted mean/variance after the drift."""
    old_mean, old_var = state["amount_mean"], state["amount_var"]
    new_mean = float(drift.params.get("mean", old_mean))
    new_var = float(drift.params.get("variance", old_var))
    for case in cases:
        ramp = drift.ramp(position(case.start))
        if ramp <= 0:
            continue
        case.amount = _positive_normal(old_mean * (1 - ramp) + new_mean * ramp,
                                       old_var * (1 - ramp) + new_var * ramp, rng)
    state["amount_mean"], state["amount_var"] = new_mean, new_var
    return {"amount_mean_before": old_mean, "amount_mean_after": new_mean,
            "amount_var_before": old_var, "amount_var_after": new_var}


def _apply_waiting_time(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Shift the waiting-gap mean for events after the drift (driving throughput time)."""
    base_mean = float(np.mean(list(dist.waiting_mean.values()))) if dist.waiting_mean else 0.0
    new_mean = float(drift.params.get("mean", base_mean))
    new_var = float(drift.params.get("variance", dist.waiting_var))
    for case in cases:
        for index in range(1, len(case.activities)):
            ramp = drift.ramp(position(case.base_starts[index]))
            if ramp <= 0:
                continue
            new_gap = _positive_normal(new_mean, new_var, rng)
            case.gaps[index] = case.gaps[index] * (1 - ramp) + new_gap * ramp
    return {
        "waiting_mean_before": round(base_mean, 2),
        "waiting_mean_after": round(new_mean, 2),
        "waiting_var_before": round(dist.waiting_var, 2),
        "waiting_var_after": round(new_var, 2),
    }


def _apply_duration(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Scale the processing time of the given resources' events after the drift."""
    affected = _resolve_resources(drift.params.get("resources", "all"), state["all_resources"])
    factor = float(drift.params.get("factor", 1.5))
    for case in cases:
        for index, base_start in enumerate(case.base_starts):
            if case.resources[index] not in affected:
                continue
            ramp = drift.ramp(position(base_start))
            if ramp > 0:
                case.durations[index] *= (1 - ramp) + factor * ramp
    return {"affected_resources": sorted(affected), "factor": factor}


def _apply_pool_size(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Grow or shrink the resource pool and scale durations the opposite way."""
    delta = int(drift.params.get("delta", -2))
    duration_factor = float(drift.params.get("duration_factor", 1.2))
    active = state["active_resources"]
    dominant = state["dominant_resource"]
    old_size = len(active)

    if delta < 0:
        record = _shrink_pool(cases, drift, dist, active, dominant, -delta, duration_factor, position, rng)
    elif delta > 0:
        record = _grow_pool(cases, drift, dist, state, delta, duration_factor, position, rng)
    else:
        record = {}
    record.update({"old_pool_size": old_size, "new_pool_size": len(state["active_resources"])})
    return record


def _shrink_pool(cases, drift, dist, active, dominant, count, duration_factor, position, rng) -> dict[str, Any]:
    """Remove resources, spread their events onto the rest, and lengthen durations."""
    count = min(count, len(active) - 1)
    removed = {str(r) for r in rng.choice(active, size=count, replace=False)} if count > 0 else set()
    remaining = [r for r in active if r not in removed]
    reassigned: dict[str, str] = {}
    for activity in dist.activities:
        if dominant.get(activity) in removed:
            dominant[activity] = str(rng.choice(remaining))
            reassigned[activity] = dominant[activity]
    for case in cases:
        for index, base_start in enumerate(case.base_starts):
            ramp = drift.ramp(position(base_start))
            if ramp <= 0:
                continue
            case.durations[index] *= (1 - ramp) + duration_factor * ramp
            if case.resources[index] in removed and (drift.mode == "sudden" or rng.random() < ramp):
                case.resources[index] = str(rng.choice(remaining))
    active[:] = remaining
    return {
        "removed_resources": sorted(removed),
        "reassigned_dominants": reassigned,
        "duration_factor": duration_factor,
    }


def _grow_pool(cases, drift, dist, state, count, duration_factor, position, rng) -> dict[str, Any]:
    """Add resources, make each dominate a free activity, and shorten durations."""
    active = state["active_resources"]
    dominant = state["dominant_resource"]
    added: list[str] = []
    for _ in range(count):
        new = resource_label(len(state["all_resources"]) + 1)
        state["all_resources"].append(new)
        active.append(new)
        added.append(new)

    claimed: dict[str, str] = {}
    free = list(dist.activities)
    rng.shuffle(free)
    for new in added:
        if not free:
            break
        activity = free.pop()
        dominant[activity] = new
        claimed[activity] = new

    for case in cases:
        for index, base_start in enumerate(case.base_starts):
            ramp = drift.ramp(position(base_start))
            if ramp <= 0:
                continue
            case.durations[index] *= (1 - ramp) + (1.0 / duration_factor) * ramp
            activity = case.activities[index]
            if activity in claimed and (drift.mode == "sudden" or rng.random() < ramp):
                case.resources[index] = _draw_dominant(dominant[activity], active, rng)
    return {"added_resources": added, "claimed_activities": claimed, "duration_factor": duration_factor}


# --- Step 5b: workload (runs last) ------------------------------------------


def _apply_workload(
    config: GeneratorConfig,
    cases: list[Case],
    drifts: list[Drift],
    dist: Distributions,
    state: dict[str, Any],
    horizon_end: datetime,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Add duplicate traces (or remove some) after the drift to change per-resource case load."""
    workload = drifts_of(drifts, "workload")
    if not workload:
        return None
    drift = workload[0]
    factor = float(drift.params.get("workload_factor", 1.0))
    affected = [c for c in cases if _frac(c.start, config.start_date, horizon_end) >= drift.start_frac]
    change = int(round(len(affected) * (factor - 1.0)))

    if change > 0:
        horizon = (horizon_end - config.start_date).total_seconds()
        window_start = config.start_date + timedelta(seconds=horizon * drift.start_frac)
        added = _duplicate_cases(config, cases, affected, change, dist, state, window_start, horizon_end, rng)
        return {"workload_factor": factor, "added_traces": added, "removed_traces": 0}
    if change < 0:
        removed = _remove_cases(cases, affected, -change, rng)
        return {"workload_factor": factor, "added_traces": 0, "removed_traces": removed}
    return {"workload_factor": factor, "added_traces": 0, "removed_traces": 0}


def _duplicate_cases(config, cases, affected, count, dist, state, window_start, window_end, rng) -> int:
    """Clone random affected cases with freshly drawn timing, amount, resources and region."""
    dominant = state["dominant_resource"]
    active = state["active_resources"]
    span = max(1.0, (window_end - window_start).total_seconds())
    for _ in range(count):
        source = affected[int(rng.integers(0, len(affected)))]
        activities = list(source.activities)
        gaps = [0.0]
        durations = [_positive_normal(dist.duration_mean[activities[0]], dist.duration_var, rng)]
        for activity in activities[1:]:
            gaps.append(_positive_normal(dist.waiting_mean[activity], dist.waiting_var, rng))
            durations.append(_positive_normal(dist.duration_mean[activity], dist.duration_var, rng))
        start = window_start + timedelta(seconds=float(rng.uniform(0, span)))
        cases.append(
            Case(
                start=start,
                activities=activities,
                gaps=gaps,
                durations=durations,
                resources=[_draw_dominant(dominant.get(a, active[0]), active, rng) for a in activities],
                base_starts=_base_timeline(start, gaps, durations),
                region=_draw_dominant(state["dominant_region"], config.regions, rng),
                amount=_positive_normal(state["amount_mean"], state["amount_var"], rng),
            )
        )
    return count


def _remove_cases(cases, affected, count, rng) -> int:
    """Delete random affected cases, but never the last instance of a trace variant."""
    variant_counts: dict[tuple[str, ...], int] = {}
    for case in cases:
        variant_counts[tuple(case.activities)] = variant_counts.get(tuple(case.activities), 0) + 1
    order = list(affected)
    rng.shuffle(order)
    removed = 0
    to_drop = set()
    for case in order:
        if removed >= count:
            break
        variant = tuple(case.activities)
        if variant_counts[variant] <= 1:
            continue
        variant_counts[variant] -= 1
        to_drop.add(id(case))
        removed += 1
    cases[:] = [c for c in cases if id(c) not in to_drop]
    return removed


# --- Step 6: assemble the DataFrame -----------------------------------------


def _assemble(cases: list[Case]) -> pd.DataFrame:
    """Recompute final timestamps from the gaps/durations and build the event-log DataFrame."""
    ordered = sorted(cases, key=lambda c: c.start)
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(ordered, start=1):
        case_id = f"case_{case_index:06d}"
        cursor = case.start
        for activity, gap, duration, resource in zip(case.activities, case.gaps, case.durations, case.resources):
            start = cursor + timedelta(minutes=gap)
            end = start + timedelta(minutes=duration)
            rows.append(
                {
                    EVENT_ID_KEY: "",
                    CASE_ID_KEY: case_id,
                    ACTIVITY_KEY: activity,
                    START_TIMESTAMP_KEY: start,
                    TIMESTAMP_KEY: end,
                    DURATION_KEY: duration,
                    RESOURCE_KEY: resource,
                    AMOUNT_KEY: round(case.amount, 2),
                    REGION_KEY: case.region,
                }
            )
            cursor = end

    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    df[START_TIMESTAMP_KEY] = pd.to_datetime(df[START_TIMESTAMP_KEY], utc=True)
    df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY], utc=True)
    df = df.sort_values([CASE_ID_KEY, START_TIMESTAMP_KEY]).reset_index(drop=True)
    df[DURATION_KEY] = (df[TIMESTAMP_KEY] - df[START_TIMESTAMP_KEY]).dt.total_seconds() / 60.0
    df[EVENT_ID_KEY] = [f"evt_{idx:08d}" for idx in range(1, len(df) + 1)]
    return df


# --- Small helpers -----------------------------------------------------------


def _position_fn(start_date: datetime, horizon_end: datetime):
    """Return a function mapping a timestamp to its [0, 1] position in the horizon."""
    span = max(1.0, (horizon_end - start_date).total_seconds())

    def position(timestamp: datetime) -> float:
        return min(1.0, max(0.0, (timestamp - start_date).total_seconds() / span))

    return position


def _frac(timestamp: datetime, start_date: datetime, horizon_end: datetime) -> float:
    """Position of a single timestamp within the horizon (clamped to [0, 1])."""
    span = max(1.0, (horizon_end - start_date).total_seconds())
    return min(1.0, max(0.0, (timestamp - start_date).total_seconds() / span))


def _positive_normal(mean: float, var: float, rng: np.random.Generator) -> float:
    """Draw a strictly positive value from a normal distribution."""
    return max(0.1, float(rng.normal(mean, np.sqrt(max(0.0, var)))))


def _draw_dominant(dominant: str, pool: list[str], rng: np.random.Generator) -> str:
    """Return the dominant resource/region 80% of the time, otherwise a random pool member."""
    if dominant in pool and rng.random() < 0.8:
        return dominant
    return str(rng.choice(pool))


def _resolve_resources(spec: Any, all_resources: list[str]) -> set[str]:
    """Interpret a drift's `resources` spec: 'all', an int count, or an explicit list."""
    if spec == "all" or spec is None:
        return set(all_resources)
    if isinstance(spec, int):
        return set(all_resources[: max(0, spec)])
    return {str(item) for item in spec}
