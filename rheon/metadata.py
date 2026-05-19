from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DriftMetadata:
    drift_id: str
    perspective: str
    subtype: str
    drift_type: str
    change_point_timestamp: datetime
    overlap_window_start: datetime | None
    overlap_window_end: datetime | None
    affected_columns: list[str]
    change_details: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "perspective": self.perspective,
            "subtype": self.subtype,
            "drift_type": self.drift_type,
            "change_point_timestamp": self.change_point_timestamp.isoformat(),
            "overlap_window_start": self.overlap_window_start.isoformat()
            if self.overlap_window_start
            else None,
            "overlap_window_end": self.overlap_window_end.isoformat()
            if self.overlap_window_end
            else None,
            "affected_columns": self.affected_columns,
            "change_details": self.change_details,
        }


def metadata_payload(
    *,
    log_name: str,
    config: dict[str, Any],
    drifts: list[DriftMetadata],
    noise: dict[str, Any],
) -> dict[str, Any]:
    return {
        "log_name": log_name,
        "config": config,
        "drifts": [drift.to_json_dict() for drift in drifts],
        "noise": noise,
    }


def drift_info_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload["drifts"], sort_keys=True, separators=(",", ":"))


def log_attributes_payload(payload: dict[str, Any]) -> dict[str, str]:
    """Return the compact JSON strings to embed in the XES log-level attributes."""
    import json

    return {
        "drift_info": drift_info_json(payload),
        "config_info": json.dumps(payload.get("config", {}), sort_keys=True, separators=(",", ":")),
        "noise_info": json.dumps(payload.get("noise", {}), sort_keys=True, separators=(",", ":")),
        "log_name": payload.get("log_name", ""),
    }


def _kv_table(items: dict[str, Any]) -> list[str]:
    """Render a small key/value dict as a Markdown table."""
    if not items:
        return ["_(empty)_", ""]
    rows = ["| Setting | Value |", "| --- | --- |"]
    for key in sorted(items):
        value = items[key]
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        rows.append(f"| {key} | {value} |")
    rows.append("")
    return rows


def _format_drift(drift: dict[str, Any], idx: int) -> list[str]:
    """Render a single drift entry as a Markdown section."""
    header = f"### #{idx} · {drift.get('drift_type', '?')} {drift.get('subtype', '?')} ({drift.get('perspective', '?')})"
    cp = drift.get("change_point_timestamp", "?")
    ows = drift.get("overlap_window_start")
    owe = drift.get("overlap_window_end")
    lines = [header, "", f"- **Drift id:** `{drift.get('drift_id', '?')}`", f"- **Change point:** {cp}"]
    if ows or owe:
        lines.append(f"- **Overlap window:** {ows} → {owe}")
    cols = drift.get("affected_columns") or []
    if cols:
        lines.append(f"- **Affected columns:** {', '.join(cols)}")
    details = drift.get("change_details") or {}
    for key in sorted(details):
        lines.append(f"- **{key}:** {details[key]}")
    lines.append("")
    return lines


def drift_metadata_md(payload: dict[str, Any]) -> str:
    """Human-readable Markdown summary of a rheon metadata payload."""
    config = payload.get("config", {}) or {}
    drifts = payload.get("drifts", []) or []
    noise = payload.get("noise", {}) or {}
    n_events = config.get("actual_num_events", "?")
    n_traces = config.get("actual_num_traces", "?")
    h_start = config.get("horizon_start", "?")
    h_end = config.get("horizon_end", "?")
    parts: list[str] = [
        f"# Rheon log: `{payload.get('log_name', '?')}`",
        "",
        f"- **Events:** {n_events:,}" if isinstance(n_events, int) else f"- **Events:** {n_events}",
        f"- **Cases:** {n_traces:,}" if isinstance(n_traces, int) else f"- **Cases:** {n_traces}",
        f"- **Time horizon:** {h_start} → {h_end}",
        f"- **Injected drifts:** {len(drifts)}",
        "",
        "## Injected drifts",
        "",
    ]
    if not drifts:
        parts.extend(["_No drifts injected._", ""])
    for i, drift in enumerate(drifts, start=1):
        parts.extend(_format_drift(drift, i))
    parts.extend(["## Configuration", ""])
    parts.extend(_kv_table(config))
    parts.extend(["## Noise", ""])
    parts.extend(_kv_table(noise))
    return "\n".join(parts)
