"""Assemble the ground-truth metadata payload and render its human-readable Markdown file."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from rheon.config import ACTIVITY_KEY, CASE_ID_KEY, GeneratorConfig, START_TIMESTAMP_KEY
from rheon.drifts import DRIFT_TYPES, Drift


def build_metadata(
    *,
    log_name: str,
    config: GeneratorConfig,
    horizon_end: datetime,
    drifts: list[Drift],
    distributions,
    drift_records: dict[str, dict[str, Any]],
    workload_record: dict[str, Any] | None,
    final_state: dict[str, Any],
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Collect the base distributions and each drift's changes into one ground-truth payload."""
    parameters = config.to_dict()
    parameters["end_date"] = horizon_end.isoformat()
    parameters["generated_traces"] = int(dataframe[CASE_ID_KEY].nunique()) if not dataframe.empty else 0
    parameters["generated_events"] = int(len(dataframe))
    parameters["num_trace_variants"] = _trace_variant_count(dataframe)

    base = {
        "activities": [
            {
                "activity": activity,
                "dominant_resource": distributions.dominant_resource[activity],
                "duration_mean": round(distributions.duration_mean[activity], 2),
                "duration_var": round(distributions.duration_var, 2),
                "waiting_mean": round(distributions.waiting_mean[activity], 2),
                "waiting_var": round(distributions.waiting_var, 2),
            }
            for activity in distributions.activities
        ],
        "amount": {"mean": round(distributions.amount_mean, 2), "var": round(distributions.amount_var, 2)},
        "inter_arrival": round(config.base_inter_arrival, 2),
        "dominant_region": distributions.dominant_region,
        "regions": config.regions,
        "resources": config.resources,
    }

    drift_entries = [
        {
            "id": drift.name,
            "type": drift.type,
            "perspective": DRIFT_TYPES[drift.type],
            "mode": drift.mode,
            **drift.window_dates(config.start_date, horizon_end),
            "params": drift.params,
            "changes": _change_record(drift, config, drift_records, workload_record),
        }
        for drift in drifts
    ]

    return {"log_name": log_name, "parameters": parameters, "base": base, "drifts": drift_entries}


def _change_record(
    drift: Drift,
    config: GeneratorConfig,
    drift_records: dict[str, dict[str, Any]],
    workload_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the recorded change for a drift, synthesising it for drifts applied outside the engine."""
    if drift.type == "workload":
        return workload_record or {}
    if drift.type == "arrival_rate":
        base = config.base_inter_arrival
        after = float(drift.params.get("inter_arrival", base * float(drift.params.get("factor", 1.0))))
        return {"inter_arrival_before": round(base, 2), "inter_arrival_after": round(after, 2)}
    if drift.type == "control_flow":
        return {
            "num_activities": int(drift.params.get("num_activities", config.num_activities)),
            "tree_weights": dict(drift.params.get("tree_weights", config.tree_weights)),
        }
    return drift_records.get(drift.name, {})


def compact_json(payload: dict[str, Any]) -> str:
    """Compact JSON string of the payload, for embedding inside the XES log attributes."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


# --- Markdown rendering ------------------------------------------------------


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the metadata payload as a readable Markdown report."""
    lines = [f"# Rheon log: {payload['log_name']}", "", "## General parameters", ""]
    lines += _table(["Parameter", "Value"], [[key, _fmt(value)] for key, value in payload["parameters"].items()])

    base = payload["base"]
    lines += ["", "## Base distributions", "", "### Activities", ""]
    lines += _table(
        ["Activity", "Dominant resource", "Duration mean", "Duration var", "Waiting mean", "Waiting var"],
        [
            [row["activity"], row["dominant_resource"], row["duration_mean"],
             row["duration_var"], row["waiting_mean"], row["waiting_var"]]
            for row in base["activities"]
        ],
    )
    lines += ["", "### Case attributes", ""]
    lines += [
        f"- Amount: mean={_fmt(base['amount']['mean'])}, variance={_fmt(base['amount']['var'])}",
        f"- Inter-arrival mean (derived from horizon / num_traces): {_fmt(base['inter_arrival'])} minutes",
        f"- Dominant region: {base['dominant_region']} (of {', '.join(base['regions'])})",
        f"- Resources: {', '.join(base['resources'])}",
    ]

    lines += ["", "## Drifts", ""]
    if not payload["drifts"]:
        lines.append("_No drifts._")
    for drift in payload["drifts"]:
        lines += _drift_section(drift)
    return "\n".join(lines)


def _drift_section(drift: dict[str, Any]) -> list[str]:
    """Render one drift: its position, then the distributions/assignments it changed."""
    lines = [f"### {drift['id']} — {drift['type']} ({drift['perspective']}, {drift['mode']})", ""]
    if drift["mode"] == "sudden":
        lines.append(f"- Drift point: {drift['drift_point']:.3f} ({drift['drift_point_date']})")
    else:
        lines.append(f"- Drift window: {drift['drift_start']:.3f} → {drift['drift_end']:.3f}")
        lines.append(f"  ({drift['drift_start_date']} → {drift['drift_end_date']})")
    lines += _changes_lines(drift["type"], drift.get("changes") or {})
    lines.append("")
    return lines


def _changes_lines(drift_type: str, changes: dict[str, Any]) -> list[str]:
    """Render the type-specific changed distributions/assignments for one drift."""
    if drift_type == "control_flow":
        return [f"- New process tree: num_activities={changes.get('num_activities')}, "
                f"weights {_fmt(changes.get('tree_weights', {}))}"]

    if drift_type == "reassignment":
        before, after = changes.get("dominant_before", {}), changes.get("dominant_after", {})
        lines = ["- New dominant resource per activity:"]
        return lines + _indent(_assignment_table(before, after))

    if drift_type == "pool_size":
        lines = [f"- Pool size: {changes.get('old_pool_size')} → {changes.get('new_pool_size')}"]
        if changes.get("removed_resources"):
            lines.append(f"- Removed resources: {_fmt(changes['removed_resources'])}")
            if changes.get("reassigned_dominants"):
                lines.append("- Reassigned dominant resource:")
                lines += _indent(_map_table(changes["reassigned_dominants"]))
        if changes.get("added_resources"):
            lines.append(f"- Added resources: {_fmt(changes['added_resources'])}")
            if changes.get("claimed_activities"):
                lines.append("- New resource now dominant for:")
                lines += _indent(_map_table(changes["claimed_activities"]))
        factor = changes.get("duration_factor")
        if factor is not None:
            lines.append(f"- Duration scaling factor: {_fmt(factor)} (fewer resources → slower, more → faster)")
        return lines

    if drift_type == "duration":
        return [f"- Affected resources: {_fmt(changes.get('affected_resources', []))}",
                f"- Processing time multiplied by {_fmt(changes.get('factor'))}"]

    if drift_type == "waiting_time":
        return [f"- Waiting-gap mean: {_fmt(changes.get('waiting_mean_before'))} → {_fmt(changes.get('waiting_mean_after'))}",
                f"- Waiting-gap variance: {_fmt(changes.get('waiting_var_before'))} → {_fmt(changes.get('waiting_var_after'))}"]

    if drift_type == "amount":
        return [f"- Amount mean: {_fmt(changes.get('amount_mean_before'))} → {_fmt(changes.get('amount_mean_after'))}",
                f"- Amount variance: {_fmt(changes.get('amount_var_before'))} → {_fmt(changes.get('amount_var_after'))}"]

    if drift_type == "arrival_rate":
        return [f"- Inter-arrival mean: {_fmt(changes.get('inter_arrival_before'))} → "
                f"{_fmt(changes.get('inter_arrival_after'))} minutes"]

    if drift_type == "region":
        return [f"- Dominant region: {changes.get('dominant_region_before')} → {changes.get('dominant_region_after')}"]

    if drift_type == "workload":
        return [f"- Workload factor: {_fmt(changes.get('workload_factor'))}",
                f"- Traces added: {changes.get('added_traces', 0)}, removed: {changes.get('removed_traces', 0)}"]

    return []


def _assignment_table(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Render an activity → resource (before → after) table, showing only the activities that changed."""
    rows = [[activity, f"{before[activity]} → {after[activity]}"]
            for activity in before if before.get(activity) != after.get(activity)]
    if not rows:
        return ["_(no change)_"]
    return _table(["Activity", "Resource (before → after)"], rows)


def _map_table(mapping: dict[str, str]) -> list[str]:
    """Render an activity → resource table."""
    return _table(["Activity", "Resource"], [[key, value] for key, value in mapping.items()])


def _indent(lines: list[str]) -> list[str]:
    """Indent a block of lines so it nests under a bullet."""
    return ["  " + line for line in lines]


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    """Render a Markdown table from a header and rows."""
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return out


def _fmt(value: Any) -> str:
    """Format a value compactly for Markdown."""
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _trace_variant_count(df: pd.DataFrame) -> int:
    """Number of distinct activity sequences in the log."""
    if df.empty:
        return 0
    variants = {
        tuple(group.sort_values(START_TIMESTAMP_KEY)[ACTIVITY_KEY].astype(str))
        for _, group in df.groupby(CASE_ID_KEY, sort=False)
    }
    return len(variants)
