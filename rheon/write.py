"""Write the event log to XES or CSV, plus the metadata sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import Event, EventLog, Trace

from rheon.config import (
    ACTIVITY_KEY,
    AMOUNT_KEY,
    CASE_ID_KEY,
    DURATION_KEY,
    EVENT_ID_KEY,
    REGION_KEY,
    RESOURCE_KEY,
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
