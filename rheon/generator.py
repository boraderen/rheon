"""The generation engine: turn parameters and drifts into an event-log DataFrame plus metadata."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

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
from rheon.write import write_csv, write_metadata_file, write_xes


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
    options = [option.name for option in fields(GeneratorConfig)]
    unknown = sorted(set(params) - set(options))
    if unknown:
        raise ValueError(f"unknown options {unknown}; choose from {options}")
    config = GeneratorConfig(**params)
    parsed = normalize_drifts(drifts, config)
    rng = make_rng(config.seed)

    output = Path(output_path).with_suffix(f".{format}")
    name = log_name or output.stem
    dataframe, metadata = _generate(config, parsed, name, rng)

    if format == "csv":
        write_csv(dataframe, output)
    else:
        write_xes(dataframe, output, metadata)
    write_metadata_file(metadata, output)


# --- Orchestration (the five generation steps) ------------------------------


def _generate(
    config: GeneratorConfig,
    drifts: list[Drift],
    log_name: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full pipeline and return the finished DataFrame and its metadata."""
    records: dict[str, dict[str, Any]] = {}

    # 1. control flow: build the tree versions and the activity universe
    trees, pools = _build_versions(config, drifts, records, rng)
    activities = sorted({a for tree in trees for a in activities_in_tree(tree)})
    distributions = _init_distributions(config, activities, rng)

    # 2. case start times along the horizon (arrival_rate and workload drifts)
    arrivals = _generate_arrivals(config, drifts, records, rng)
    position = _position_fn(config.start_date, config.end_date)

    # 3. activity sequences, base event times and attribute columns
    cases = _build_cases(config, drifts, pools, arrivals, distributions, position, rng)

    # 4. apply the remaining drifts to the cases
    _apply_drifts(config, cases, drifts, distributions, position, records, rng)

    # 5. materialise the event table and the metadata
    dataframe = _assemble(cases)
    metadata = build_metadata(
        log_name=log_name,
        config=config,
        drifts=drifts,
        distributions=distributions,
        process_tree=str(trees[0]),
        drift_records=records,
        dataframe=dataframe,
    )
    return dataframe, metadata


# --- Step 1: control-flow tree versions -------------------------------------


def _build_versions(
    config: GeneratorConfig,
    drifts: list[Drift],
    records: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[list, list]:
    """Build the base tree plus one different tree per control-flow drift, with a playout pool each."""
    trees = [build_tree(config.num_activities, config.tree_weights, rng)]
    for drift in drifts_of(drifts, "control_flow"):
        num_activities = drift.params.get("num_activities", config.num_activities)
        tree_weights = drift.params.get("tree_weights", config.tree_weights)
        tree = build_tree(num_activities, tree_weights, rng, differ_from=trees[-1])
        records[drift.name] = {"process_tree_before": str(trees[-1]), "process_tree_after": str(tree)}
        trees.append(tree)
    pool_size = max(200, config.num_traces)
    pools = [playout_pool(tree, pool_size, rng) for tree in trees]
    return trees, pools


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
    drifts: list[Drift],
    records: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> list[datetime]:
    """Fill the fixed [start_date, end_date] horizon with case starts; the count is approximate.

    arrival_rate drifts move the mean inter-arrival gap; workload drifts scale how many cases
    arrive per unit of time, so the extra (or missing) cases follow every other drift naturally.
    """
    base = config.base_inter_arrival
    gaps = _levels(
        drifts_of(drifts, "arrival_rate"), base,
        lambda drift, level: float(drift.params["inter_arrival"]) if "inter_arrival" in drift.params
        else level * float(drift.params["factor"]),
    )
    loads = _levels(
        drifts_of(drifts, "workload"), 1.0,
        lambda drift, level: level * float(drift.params["workload_factor"]),
    )
    for drift, before, after in gaps:
        records[drift.name] = {"inter_arrival_before": round(before, 2), "inter_arrival_after": round(after, 2)}
    for drift, before, after in loads:
        records[drift.name] = {"workload_before": round(before, 4), "workload_after": round(after, 4)}

    start, end = config.start_date, config.end_date
    horizon_seconds = (end - start).total_seconds()
    max_cases = 100 * config.num_traces + 1000  # guard against drifts that make arrivals explode
    starts = [start]
    cursor = start
    while True:
        position_value = (cursor - start).total_seconds() / horizon_seconds
        mean = _level_at(gaps, base, position_value) / _level_at(loads, 1.0, position_value)
        cursor = cursor + timedelta(minutes=float(rng.exponential(mean)))
        if cursor > end:
            return starts
        starts.append(cursor)
        if len(starts) > max_cases:
            raise ValueError(f"arrival_rate/workload drifts would create more than {max_cases} cases")


def _levels(
    drifts: list[Drift],
    base: float,
    target: Callable[[Drift, float], float],
) -> list[tuple[Drift, float, float]]:
    """Pair each (chronological) drift with the level it starts from and the level it moves to."""
    levels: list[tuple[Drift, float, float]] = []
    level = base
    for drift in drifts:
        after = target(drift, level)
        levels.append((drift, level, after))
        level = after
    return levels


def _level_at(levels: list[tuple[Drift, float, float]], base: float, position_value: float) -> float:
    """The level in effect at a horizon position, interpolated inside a gradual window."""
    value = base
    for drift, before, after in levels:
        ramp = drift.ramp(position_value)
        if ramp > 0.0:
            value = before + (after - before) * ramp
    return value


# --- Step 3: build cases -----------------------------------------------------


def _build_cases(
    config: GeneratorConfig,
    drifts: list[Drift],
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


# --- Step 4: apply the attribute, resource and timing drifts ----------------


def _apply_drifts(
    config: GeneratorConfig,
    cases: list[Case],
    drifts: list[Drift],
    dist: Distributions,
    position: Any,
    records: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> None:
    """Apply attribute drifts by start fraction, then duration drifts, recording each change.

    Duration drifts run last, so a slowdown stays with its resource even if a later
    reassignment or pool change moves events to other resources.
    """
    state = {
        "dominant_resource": dict(dist.dominant_resource),
        "active_resources": list(config.resources),
        "all_resources": list(config.resources),
        "dominant_region": dist.dominant_region,
        "amount_mean": dist.amount_mean,
        "amount_var": dist.amount_var,
        "waiting_mean": dict(dist.waiting_mean),
        "waiting_var": dist.waiting_var,
    }
    handlers = {
        "reassignment": _apply_reassignment,
        "region": _apply_region,
        "amount": _apply_amount,
        "waiting_time": _apply_waiting_time,
        "pool_size": _apply_pool_size,
        "duration": _apply_duration,
    }
    ordered = sorted(
        (d for d in drifts if d.type in handlers),
        key=lambda d: (d.type == "duration", d.start_frac),
    )
    for drift in ordered:
        records[drift.name] = handlers[drift.type](config, cases, drift, dist, state, position, rng)


def _apply_reassignment(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Pick a new dominant resource per activity and re-fill the resource column after the drift."""
    old = dict(state["dominant_resource"])
    active = state["active_resources"]
    if len(active) < 2:
        raise ValueError(f"drift {drift.name} (reassignment): needs at least two active resources")
    new = {a: str(rng.choice([r for r in active if r != old[a]])) for a in dist.activities}
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
    new = str(rng.choice([r for r in config.regions if r != old]))
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
    return {"amount_mean_before": round(old_mean, 2), "amount_mean_after": round(new_mean, 2),
            "amount_var_before": round(old_var, 2), "amount_var_after": round(new_var, 2)}


def _apply_waiting_time(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Re-draw the waiting gaps before later events from a shifted mean/variance (all activities at once)."""
    old_mean, old_var = state["waiting_mean"], state["waiting_var"]
    new_mean = {a: float(drift.params.get("mean", old_mean[a])) for a in old_mean}
    new_var = float(drift.params.get("variance", old_var))
    for case in cases:
        for index in range(1, len(case.activities)):
            ramp = drift.ramp(position(case.base_starts[index]))
            if ramp <= 0:
                continue
            activity = case.activities[index]
            case.gaps[index] = _positive_normal(old_mean[activity] * (1 - ramp) + new_mean[activity] * ramp,
                                                old_var * (1 - ramp) + new_var * ramp, rng)
    state["waiting_mean"], state["waiting_var"] = new_mean, new_var
    return {
        "waiting_mean_before": round(float(np.mean(list(old_mean.values()))), 2),
        "waiting_mean_after": round(float(np.mean(list(new_mean.values()))), 2),
        "waiting_var_before": round(old_var, 2),
        "waiting_var_after": round(new_var, 2),
    }


def _apply_duration(config, cases, drift, dist, state, position, rng) -> dict[str, Any]:
    """Scale the processing time of the given resources' events after the drift."""
    affected = _resolve_resources(drift, state["all_resources"])
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
    """Grow or shrink the pool; scale durations by the reciprocal or direct duration factor."""
    delta = int(drift.params.get("delta", -2))
    duration_factor = float(drift.params.get("duration_factor", 1.2))
    old_size = len(state["active_resources"])

    if delta < 0:
        record = _shrink_pool(cases, drift, dist, state, -delta, duration_factor, position, rng)
    else:
        record = _grow_pool(cases, drift, dist, state, delta, duration_factor, position, rng)
    return {"pool_size_before": old_size, "pool_size_after": len(state["active_resources"]), **record}


def _shrink_pool(cases, drift, dist, state, count, duration_factor, position, rng) -> dict[str, Any]:
    """Remove resources, move their events to the new dominants, and multiply durations by the factor."""
    active = state["active_resources"]
    dominant = state["dominant_resource"]
    if count >= len(active):
        raise ValueError(f"drift {drift.name} (pool_size): cannot remove {count} of the {len(active)} active resources")
    removed = {str(r) for r in rng.choice(active, size=count, replace=False)}
    remaining = [r for r in active if r not in removed]
    reassigned: dict[str, str] = {}
    for activity in dist.activities:
        if dominant[activity] in removed:
            dominant[activity] = str(rng.choice(remaining))
            reassigned[activity] = dominant[activity]
    for case in cases:
        for index, base_start in enumerate(case.base_starts):
            ramp = drift.ramp(position(base_start))
            if ramp <= 0:
                continue
            case.durations[index] *= (1 - ramp) + duration_factor * ramp
            if case.resources[index] in removed and (drift.mode == "sudden" or rng.random() < ramp):
                case.resources[index] = _draw_dominant(dominant[case.activities[index]], remaining, rng)
    active[:] = remaining
    return {
        "removed_resources": sorted(removed),
        "reassigned_dominants": reassigned,
        "duration_factor": duration_factor,
    }


def _grow_pool(cases, drift, dist, state, count, duration_factor, position, rng) -> dict[str, Any]:
    """Add resources, claim distinct activities while available, and apply reciprocal duration scaling."""
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


# --- Step 5: assemble the DataFrame -----------------------------------------


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

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    # whole seconds give every timestamp the same format; rounding is monotonic, so order is kept
    df[START_TIMESTAMP_KEY] = pd.to_datetime(df[START_TIMESTAMP_KEY], utc=True).dt.round("s")
    df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY], utc=True).dt.round("s")
    df = df.sort_values([CASE_ID_KEY, START_TIMESTAMP_KEY]).reset_index(drop=True)
    df[DURATION_KEY] = (df[TIMESTAMP_KEY] - df[START_TIMESTAMP_KEY]).dt.total_seconds() / 60.0
    df[EVENT_ID_KEY] = [f"evt_{idx:08d}" for idx in range(1, len(df) + 1)]
    return df


# --- Small helpers -----------------------------------------------------------


def _position_fn(start_date: datetime, end_date: datetime):
    """Return a function mapping a timestamp to its [0, 1] position in the horizon."""
    span = (end_date - start_date).total_seconds()

    def position(timestamp: datetime) -> float:
        return min(1.0, max(0.0, (timestamp - start_date).total_seconds() / span))

    return position


def _positive_normal(mean: float, var: float, rng: np.random.Generator) -> float:
    """Draw a strictly positive value (at least 0.1) from a normal distribution."""
    return max(0.1, float(rng.normal(mean, np.sqrt(max(0.0, var)))))


def _draw_dominant(dominant: str, pool: list[str], rng: np.random.Generator) -> str:
    """Return the dominant resource/region 80% of the time, otherwise a random pool member."""
    if dominant in pool and rng.random() < 0.8:
        return dominant
    return str(rng.choice(pool))


def _resolve_resources(drift: Drift, all_resources: list[str]) -> set[str]:
    """Interpret a duration drift's `resources`: 'all', a count of the first resources, or a list of names."""
    spec = drift.params.get("resources", "all")
    if spec == "all":
        return set(all_resources)
    if isinstance(spec, (list, tuple)):
        unknown = sorted(set(spec) - set(all_resources))
        if unknown:
            raise ValueError(f"drift {drift.name} (duration): unknown resources {unknown}; the pool is {all_resources}")
        return set(spec)
    if spec > len(all_resources):
        raise ValueError(f"drift {drift.name} (duration): resources={spec} but the pool only has {len(all_resources)}")
    return set(all_resources[:spec])
