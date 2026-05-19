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
    CASE_TYPE_KEY,
    DURATION_KEY,
    EVENT_ID_KEY,
    REGION_KEY,
    RESOURCE_KEY,
    ROLE_KEY,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
)
from rheon.metadata import drift_metadata_md, log_attributes_payload


EVENT_LEVEL_COLUMNS = [
    EVENT_ID_KEY,
    ACTIVITY_KEY,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
    DURATION_KEY,
    RESOURCE_KEY,
    ROLE_KEY,
]


def dataframe_to_event_log(df: pd.DataFrame, payload: dict[str, Any]) -> EventLog:
    log = EventLog()
    for key, value in log_attributes_payload(payload).items():
        log.attributes[key] = value
    log.attributes["generator"] = "rheon"
    for case_id, group in df.sort_values([CASE_ID_KEY, TIMESTAMP_KEY]).groupby(CASE_ID_KEY, sort=False):
        first = group.iloc[0]
        trace = Trace(
            attributes={
                "concept:name": str(case_id),
                "case_type": str(first[CASE_TYPE_KEY]),
                "amount": float(first[AMOUNT_KEY]),
                "region": str(first[REGION_KEY]),
            }
        )
        for _, row in group.iterrows():
            event = Event()
            for column in EVENT_LEVEL_COLUMNS:
                event[column] = _to_python_value(column, row[column])
            trace.append(event)
        log.append(trace)
    return log


def write_xes_with_metadata(
    df: pd.DataFrame,
    output_path: str | Path,
    payload: dict[str, Any],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log = dataframe_to_event_log(df, payload)
    xes_exporter.apply(log, str(path), variant=xes_exporter.Variants.LINE_BY_LINE)
    sidecar = path.with_name(path.name + ".metadata.md")
    sidecar.write_text(drift_metadata_md(payload), encoding="utf-8")
    return path


def _to_python_value(column: str, value: object) -> object:
    if column in {TIMESTAMP_KEY, START_TIMESTAMP_KEY}:
        return pd.to_datetime(value).to_pydatetime()
    if column in {DURATION_KEY, AMOUNT_KEY}:
        return float(value)
    return str(value)
