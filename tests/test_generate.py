from __future__ import annotations

import json

import pandas as pd
import pytest
from pm4py.objects.log.importer.xes import importer as xes_importer

import rheon
from rheon.config import CASE_ID_KEY, RESOURCE_KEY, SCHEMA_COLUMNS


ALL_DRIFTS = [
    [{"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 7}],
    [{"type": "reassignment", "mode": "sudden", "drift_point": 0.5}],
    [{"type": "pool_size", "mode": "sudden", "drift_point": 0.5, "delta": -2}],
    [{"type": "pool_size", "mode": "gradual", "start_point": 0.4, "end_point": 0.7, "delta": 2}],
    [{"type": "duration", "mode": "sudden", "drift_point": 0.5, "resources": 2, "factor": 2.0}],
    [{"type": "waiting_time", "mode": "gradual", "start_point": 0.4, "end_point": 0.7, "mean": 90}],
    [{"type": "amount", "mode": "sudden", "drift_point": 0.5, "mean": 5000}],
    [{"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5, "factor": 0.5}],
    [{"type": "region", "mode": "sudden", "drift_point": 0.5}],
    [{"type": "workload", "mode": "sudden", "drift_point": 0.5, "workload_factor": 1.4}],
]


def _read_embedded_metadata(xes_path) -> dict:
    """Read the ground-truth metadata that is embedded in a written XES file."""
    log = xes_importer.apply(str(xes_path))
    return json.loads(log.attributes["rheon:metadata"])


@pytest.mark.parametrize("drifts", ALL_DRIFTS, ids=[d[0]["type"] + "_" + d[0]["mode"] for d in ALL_DRIFTS])
def test_each_drift_generates_a_valid_log(tmp_path, drifts):
    out = tmp_path / "log.xes"
    rheon.generate_log(drifts, out, num_traces=120, num_activities=6, num_resources=5)

    assert out.exists()
    assert (tmp_path / "log_meta.md").exists()
    assert rheon.validation_passed(rheon.validate_xes(out))


def test_generate_log_returns_none(tmp_path):
    assert rheon.generate_log(ALL_DRIFTS[1], tmp_path / "log.xes", num_traces=50) is None


def test_csv_format_writes_the_dataframe(tmp_path):
    rheon.generate_log(ALL_DRIFTS[6], tmp_path / "log.csv", format="csv", num_traces=100, num_activities=5)

    csv = tmp_path / "log.csv"
    assert csv.exists()
    assert (tmp_path / "log_meta.md").exists()
    df = pd.read_csv(csv)
    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df) > 0


def test_format_drives_the_file_suffix(tmp_path):
    rheon.generate_log(ALL_DRIFTS[6], tmp_path / "log.xes", format="csv", num_traces=50)
    assert (tmp_path / "log.csv").exists()
    assert not (tmp_path / "log.xes").exists()


def test_invalid_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="format"):
        rheon.generate_log(ALL_DRIFTS[6], tmp_path / "log.json", format="json", num_traces=10)


def test_metadata_records_base_and_drift_changes(tmp_path):
    out = tmp_path / "log.xes"
    rheon.generate_log([{"type": "reassignment", "mode": "sudden", "drift_point": 0.5}],
                       out, num_traces=100, num_activities=5)

    meta = _read_embedded_metadata(out)
    assert meta["parameters"]["num_activities"] == 5
    assert len(meta["base"]["activities"]) >= 5
    [drift] = meta["drifts"]
    assert drift["type"] == "reassignment"
    assert drift["perspective"] == "resource"
    assert "dominant_before" in drift["changes"]
    assert "dominant_after" in drift["changes"]


def test_pool_grow_adds_resources(tmp_path):
    rheon.generate_log([{"type": "pool_size", "mode": "sudden", "drift_point": 0.3, "delta": 3}],
                       tmp_path / "log.csv", format="csv", num_traces=200, num_resources=5)
    df = pd.read_csv(tmp_path / "log.csv")
    assert df[RESOURCE_KEY].nunique() > 5


def test_workload_increase_adds_traces(tmp_path):
    rheon.generate_log([{"type": "amount", "mode": "sudden", "drift_point": 0.5}],
                       tmp_path / "base.csv", format="csv", num_traces=200, seed=1)
    rheon.generate_log([{"type": "workload", "mode": "sudden", "drift_point": 0.5, "workload_factor": 1.5}],
                       tmp_path / "grown.csv", format="csv", num_traces=200, seed=1)
    base = pd.read_csv(tmp_path / "base.csv")
    grown = pd.read_csv(tmp_path / "grown.csv")
    assert grown[CASE_ID_KEY].nunique() > base[CASE_ID_KEY].nunique()
