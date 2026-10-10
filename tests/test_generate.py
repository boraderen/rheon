from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pm4py
import pytest
from pm4py.objects.log.importer.xes import importer as xes_importer

import rheon
from rheon.config import (
    ACTIVITY_KEY,
    CASE_ID_KEY,
    DURATION_KEY,
    RESOURCE_KEY,
    SCHEMA_COLUMNS,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
)


ALL_DRIFTS = [
    [{"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 7}],
    [{"type": "control_flow", "mode": "gradual", "start_point": 0.4, "end_point": 0.6}],
    [{"type": "reassignment", "mode": "sudden", "drift_point": 0.5}],
    [{"type": "pool_size", "mode": "sudden", "drift_point": 0.5, "delta": -2}],
    [{"type": "pool_size", "mode": "gradual", "start_point": 0.4, "end_point": 0.7, "delta": 2}],
    [{"type": "duration", "mode": "sudden", "drift_point": 0.5, "resources": 2, "factor": 2.0}],
    [{"type": "waiting_time", "mode": "gradual", "start_point": 0.4, "end_point": 0.7, "mean": 90}],
    [{"type": "amount", "mode": "sudden", "drift_point": 0.5, "mean": 5000}],
    [{"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5, "factor": 0.5}],
    [{"type": "region", "mode": "sudden", "drift_point": 0.5}],
    [{"type": "workload", "mode": "gradual", "start_point": 0.4, "end_point": 0.6, "workload_factor": 1.4}],
]


def _read_log(xes_path):
    """Load a written XES file with PM4PY."""
    return xes_importer.apply(str(xes_path))


def _read_embedded_metadata(xes_path) -> dict:
    """Read the ground-truth metadata that is embedded in a written XES file."""
    return json.loads(_read_log(xes_path).attributes["rheon:metadata"])


def _generate_csv(tmp_path, drifts, **params) -> pd.DataFrame:
    """Generate a CSV log and read it back with parsed timestamps."""
    rheon.generate_log(drifts, tmp_path / "log.csv", format="csv", **params)
    return pd.read_csv(tmp_path / "log.csv", parse_dates=[START_TIMESTAMP_KEY, TIMESTAMP_KEY])


def _position(timestamps):
    """Horizon position (0-1) of timestamps within the default 2020 horizon."""
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
    return (timestamps - start) / (end - start)


def _case_positions(df: pd.DataFrame) -> pd.Series:
    """Horizon position of every case start."""
    return _position(df.groupby(CASE_ID_KEY)[START_TIMESTAMP_KEY].min())


# --- Output files ------------------------------------------------------------


@pytest.mark.parametrize("drifts", ALL_DRIFTS, ids=[d[0]["type"] + "_" + d[0]["mode"] for d in ALL_DRIFTS])
def test_each_drift_writes_a_readable_log(tmp_path, drifts):
    out = tmp_path / "log.xes"
    rheon.generate_log(drifts, out, num_traces=120, num_activities=6, num_resources=5)

    assert out.exists()
    assert (tmp_path / "log_meta.md").exists()
    assert len(_read_log(out)) > 0


def test_generate_log_returns_none(tmp_path):
    assert rheon.generate_log(ALL_DRIFTS[2], tmp_path / "log.xes", num_traces=50) is None


def test_csv_format_writes_the_dataframe(tmp_path):
    df = _generate_csv(tmp_path, ALL_DRIFTS[7], num_traces=100, num_activities=5)

    assert (tmp_path / "log_meta.md").exists()
    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df) > 0
    assert isinstance(df[START_TIMESTAMP_KEY].dtype, pd.DatetimeTZDtype)
    assert isinstance(df[TIMESTAMP_KEY].dtype, pd.DatetimeTZDtype)


def test_format_drives_the_file_suffix(tmp_path):
    rheon.generate_log(ALL_DRIFTS[7], tmp_path / "log.xes", format="csv", num_traces=50)
    assert (tmp_path / "log.csv").exists()
    assert not (tmp_path / "log.xes").exists()


def test_invalid_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="format"):
        rheon.generate_log(ALL_DRIFTS[7], tmp_path / "log.json", format="json", num_traces=10)


@pytest.mark.parametrize("params", [
    {"num_trace": 100},
    {"num_resources": 0},
    {"num_traces": 0},
    {"end_date": datetime(2019, 1, 1)},
    {"activity_duration": (-1.0, 5.0)},
    {"tree_weights": {"sequence": 0.0}},
])
def test_invalid_parameters_are_rejected(tmp_path, params):
    with pytest.raises(ValueError):
        rheon.generate_log([], tmp_path / "log.csv", format="csv", **params)


def test_naive_dates_are_read_as_utc(tmp_path):
    df = _generate_csv(tmp_path, [], num_traces=20, start_date=datetime(2020, 6, 1))
    assert df[START_TIMESTAMP_KEY].min() == pd.Timestamp("2020-06-01", tz="UTC")


# --- Metadata ----------------------------------------------------------------


def test_metadata_records_base_and_drift_changes(tmp_path):
    out = tmp_path / "log.xes"
    rheon.generate_log([{"type": "reassignment", "mode": "sudden", "drift_point": 0.5}],
                       out, num_traces=100, num_activities=5)

    meta = _read_embedded_metadata(out)
    assert meta["parameters"]["num_activities"] == 5
    assert len(meta["base"]["activities"]) == 5
    [drift] = meta["drifts"]
    assert drift["type"] == "reassignment"
    assert drift["perspective"] == "resource"
    assert drift["drift_point"] == 0.5
    assert "dominant_before" in drift["changes"]
    assert "dominant_after" in drift["changes"]


def test_control_flow_drift_records_a_different_tree(tmp_path):
    out = tmp_path / "log.xes"
    rheon.generate_log([{"type": "control_flow", "mode": "sudden", "drift_point": 0.5}],
                       out, num_traces=50, num_activities=2)

    meta = _read_embedded_metadata(out)
    changes = meta["drifts"][0]["changes"]
    assert changes["process_tree_before"] == meta["base"]["process_tree"]
    assert changes["process_tree_after"] != changes["process_tree_before"]


def test_chained_drifts_record_their_own_before_values(tmp_path):
    out = tmp_path / "log.xes"
    rheon.generate_log([
        {"type": "arrival_rate", "mode": "sudden", "drift_point": 0.3, "inter_arrival": 600},
        {"type": "arrival_rate", "mode": "sudden", "drift_point": 0.6, "factor": 0.5},
        {"type": "waiting_time", "mode": "sudden", "drift_point": 0.3, "mean": 100},
        {"type": "waiting_time", "mode": "sudden", "drift_point": 0.6, "mean": 200},
    ], out, num_traces=200)

    changes = {drift["id"]: drift["changes"] for drift in _read_embedded_metadata(out)["drifts"]}
    assert changes["d02"] == {"inter_arrival_before": 600, "inter_arrival_after": 300}
    assert changes["d04"]["waiting_mean_before"] == 100
    assert changes["d04"]["waiting_mean_after"] == 200


# --- Drift effects -----------------------------------------------------------


def test_pool_grow_adds_resources(tmp_path):
    df = _generate_csv(tmp_path, [{"type": "pool_size", "mode": "sudden", "drift_point": 0.3, "delta": 3}],
                       num_traces=200, num_resources=5)
    assert df[RESOURCE_KEY].nunique() > 5


def test_pool_shrink_moves_events_to_the_new_dominant(tmp_path):
    out = tmp_path / "log.xes"
    rheon.generate_log([{"type": "pool_size", "mode": "sudden", "drift_point": 0.5, "delta": -6}],
                       out, num_traces=600, num_resources=8, num_activities=6)
    reassigned = _read_embedded_metadata(out)["drifts"][0]["changes"]["reassigned_dominants"]
    df = pm4py.convert_to_dataframe(_read_log(out))
    late = df[_position(df[START_TIMESTAMP_KEY]) > 0.55]

    assert reassigned
    for activity, resource in reassigned.items():
        assert (late[late[ACTIVITY_KEY] == activity][RESOURCE_KEY] == resource).mean() > 0.75


def test_every_workload_drift_scales_case_arrivals(tmp_path):
    positions = _case_positions(_generate_csv(tmp_path, [
        {"type": "workload", "mode": "sudden", "drift_point": 0.3, "workload_factor": 2.0},
        {"type": "workload", "mode": "sudden", "drift_point": 0.7, "workload_factor": 2.0},
    ], num_traces=600))

    rate_before = (positions < 0.3).sum() / 0.3
    assert ((positions >= 0.3) & (positions < 0.7)).sum() / 0.4 == pytest.approx(2 * rate_before, rel=0.25)
    assert (positions >= 0.7).sum() / 0.3 == pytest.approx(4 * rate_before, rel=0.25)


def test_gradual_workload_ramps_across_the_window(tmp_path):
    positions = _case_positions(_generate_csv(tmp_path, [
        {"type": "workload", "mode": "gradual", "start_point": 0.2, "end_point": 0.8, "workload_factor": 3.0},
    ], num_traces=1000))

    before, window_start, _, window_end, after = np.histogram(positions, bins=[0, 0.2, 0.3, 0.7, 0.8, 1.0])[0]
    assert window_start < 0.6 * window_end
    assert after == pytest.approx(3 * before, rel=0.25)


def test_extra_workload_cases_follow_the_other_drifts(tmp_path):
    df = _generate_csv(tmp_path, [
        {"type": "waiting_time", "mode": "sudden", "drift_point": 0.1, "mean": 300, "variance": 10},
        {"type": "duration", "mode": "sudden", "drift_point": 0.1, "factor": 3.0},
        {"type": "workload", "mode": "sudden", "drift_point": 0.3, "workload_factor": 2.0},
        {"type": "control_flow", "mode": "sudden", "drift_point": 0.7, "num_activities": 14},
    ], num_traces=600, num_activities=10)

    case_start = df.groupby(CASE_ID_KEY)[START_TIMESTAMP_KEY].transform("min")
    early = df[_position(case_start) < 0.7]
    assert not early[ACTIVITY_KEY].isin(["k", "l", "m", "n"]).any()

    position = _position(df[START_TIMESTAMP_KEY])
    gaps = (df[START_TIMESTAMP_KEY] - df.groupby(CASE_ID_KEY)[TIMESTAMP_KEY].shift()).dt.total_seconds() / 60
    assert gaps[position > 0.2].dropna().min() > 250

    base_duration = df[position < 0.1].groupby(ACTIVITY_KEY)[DURATION_KEY].mean()
    busy = df[(position > 0.35) & (position < 0.65)]
    assert (busy[DURATION_KEY] / busy[ACTIVITY_KEY].map(base_duration)).mean() == pytest.approx(3.0, rel=0.15)


def test_duration_drift_stays_with_its_resource(tmp_path):
    df = _generate_csv(tmp_path, [
        {"type": "duration", "mode": "sudden", "drift_point": 0.3, "resources": ["res_01"], "factor": 3.0},
        {"type": "reassignment", "mode": "sudden", "drift_point": 0.6},
    ], num_traces=600)

    base_duration = df[_position(df[START_TIMESTAMP_KEY]) < 0.3].groupby(ACTIVITY_KEY)[DURATION_KEY].mean()
    late = df[_position(df[START_TIMESTAMP_KEY]) > 0.65]
    slowdown = (late[DURATION_KEY] / late[ACTIVITY_KEY].map(base_duration)).groupby(late[RESOURCE_KEY]).mean()
    assert slowdown["res_01"] == pytest.approx(3.0, rel=0.2)
    assert slowdown.drop("res_01").max() < 1.3
