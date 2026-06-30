"""Write the event log to XES (with a metadata sidecar) and validate an existing XES file."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import Event, EventLog, Trace

from rheon.config import (
    ACTIVITY_KEY,
    AMOUNT_KEY,
    CASE_ID_KEY,
    DURATION_KEY,
    EVENT_ID_KEY,
    REGION_KEY,
    RESOURCE_KEY,
    SCHEMA_COLUMNS,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
)
from rheon.metadata import compact_json, render_markdown


EVENT_COLUMNS = [EVENT_ID_KEY, ACTIVITY_KEY, START_TIMESTAMP_KEY, TIMESTAMP_KEY, DURATION_KEY, RESOURCE_KEY]
METADATA_KEY = "rheon:metadata"


def write_xes(df: pd.DataFrame, output_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Export the DataFrame as an XES file with the metadata embedded as a log attribute."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log = _dataframe_to_log(df, metadata)
    xes_exporter.apply(log, str(path), variant=xes_exporter.Variants.LINE_BY_LINE)
    return path


def write_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the event log as a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_metadata_file(metadata: dict[str, Any], output_path: str | Path) -> Path:
    """Write the `<log_name>_meta.md` ground-truth sidecar next to the output file."""
    path = Path(output_path)
    name = metadata.get("log_name") or path.stem
    sidecar = path.parent / f"{name}_meta.md"
    sidecar.write_text(render_markdown(metadata), encoding="utf-8")
    return sidecar


def _dataframe_to_log(df: pd.DataFrame, metadata: dict[str, Any]) -> EventLog:
    """Build a PM4PY EventLog with the metadata stored as a log-level attribute."""
    log = EventLog()
    log.attributes["generator"] = "rheon"
    log.attributes[METADATA_KEY] = compact_json(metadata)
    for case_id, group in df.groupby(CASE_ID_KEY, sort=False):
        first = group.iloc[0]
        trace = Trace(attributes={
            "concept:name": str(case_id),
            "amount": float(first[AMOUNT_KEY]),
            "region": str(first[REGION_KEY]),
        })
        for _, row in group.iterrows():
            event = Event()
            for column in EVENT_COLUMNS:
                event[column] = _python_value(column, row[column])
            trace.append(event)
        log.append(trace)
    return log


def _python_value(column: str, value: Any) -> Any:
    """Coerce a DataFrame cell to the plain Python type XES expects for that column."""
    if column in {START_TIMESTAMP_KEY, TIMESTAMP_KEY}:
        return pd.to_datetime(value).to_pydatetime()
    if column == DURATION_KEY:
        return float(value)
    return str(value)


# --- Validation --------------------------------------------------------------

Issue = dict[str, Any]


def validate_xes(path: str | Path) -> list[Issue]:
    """Check that an XES file is well-formed, schema-correct and consistent with its metadata."""
    issues: list[Issue] = []
    log = _read_log(Path(path), issues)
    if log is None:
        return issues

    df = _log_to_dataframe(log)
    missing = [column for column in SCHEMA_COLUMNS if column not in df.columns]
    if missing:
        issues.append(_issue("error", "missing_columns", f"missing columns: {missing}"))
        return issues

    _check_timestamps(df, issues)
    _check_drift_points(df, log, issues)
    return issues


def validation_passed(issues: list[Issue]) -> bool:
    """True when no issue has error severity."""
    return not any(issue["severity"] == "error" for issue in issues)


def _read_log(path: Path, issues: list[Issue]):
    """Read and parse the XES file, recording an error if it cannot be loaded."""
    if not path.exists():
        issues.append(_issue("error", "file_missing", f"{path} does not exist"))
        return None
    try:
        ET.parse(path)
        log = xes_importer.apply(str(path))
    except Exception as exc:  # noqa: BLE001 - report any parse/import failure as an issue
        issues.append(_issue("error", "unreadable", str(exc)))
        return None
    if len(log) == 0:
        issues.append(_issue("error", "empty_log", "the log has no traces"))
    return log


def _log_to_dataframe(log) -> pd.DataFrame:
    """Flatten a loaded XES log back into the rheon event schema."""
    rows: list[dict[str, Any]] = []
    for trace in log:
        case_id = trace.attributes.get("concept:name", "")
        amount = trace.attributes.get("amount", 0.0)
        region = trace.attributes.get("region", "")
        for event in trace:
            rows.append({
                EVENT_ID_KEY: event.get(EVENT_ID_KEY, ""),
                CASE_ID_KEY: case_id,
                ACTIVITY_KEY: event.get(ACTIVITY_KEY, ""),
                START_TIMESTAMP_KEY: event.get(START_TIMESTAMP_KEY),
                TIMESTAMP_KEY: event.get(TIMESTAMP_KEY),
                DURATION_KEY: event.get(DURATION_KEY),
                RESOURCE_KEY: event.get(RESOURCE_KEY, ""),
                AMOUNT_KEY: amount,
                REGION_KEY: region,
            })
    return pd.DataFrame(rows)


def _check_timestamps(df: pd.DataFrame, issues: list[Issue]) -> None:
    """Verify start <= end and non-decreasing timestamps within each case."""
    starts = pd.to_datetime(df[START_TIMESTAMP_KEY], utc=True, errors="coerce")
    ends = pd.to_datetime(df[TIMESTAMP_KEY], utc=True, errors="coerce")
    if starts.isna().any() or ends.isna().any():
        issues.append(_issue("error", "bad_timestamps", "some timestamps are missing or unparseable"))
        return
    if (starts > ends).any():
        issues.append(_issue("error", "start_after_end", "an event starts after it ends"))
    ordered = df.assign(_s=starts).groupby(CASE_ID_KEY, sort=False)["_s"]
    if not ordered.apply(lambda s: s.is_monotonic_increasing).all():
        issues.append(_issue("error", "not_monotonic", "events are not time-ordered within a case"))


def _check_drift_points(df: pd.DataFrame, log, issues: list[Issue]) -> None:
    """Check that every drift point recorded in the metadata falls inside the log's time range."""
    raw = log.attributes.get(METADATA_KEY)
    if not raw:
        issues.append(_issue("warning", "metadata_missing", f"log-level {METADATA_KEY} attribute is absent"))
        return
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        issues.append(_issue("error", "metadata_invalid", str(exc)))
        return
    start = pd.to_datetime(df[START_TIMESTAMP_KEY], utc=True).min()
    end = pd.to_datetime(df[TIMESTAMP_KEY], utc=True).max()
    for drift in metadata.get("drifts", []):
        for key in ("drift_point_date", "drift_start_date", "drift_end_date"):
            if key not in drift:
                continue
            point = pd.to_datetime(drift[key], utc=True)
            if point < start or point > end:
                issues.append(_issue(
                    "error", "drift_outside_log",
                    f"drift {drift.get('id')} {key} {point} is outside the log range",
                ))


def _issue(severity: str, code: str, message: str) -> Issue:
    """Build a single validation issue record."""
    return {"severity": severity, "code": code, "message": message}
