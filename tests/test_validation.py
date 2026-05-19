from __future__ import annotations

from pm4py.read import read_xes
from pm4py.stats import get_event_attributes, get_trace_attributes

from rheon.composer import generate_and_write_log
from rheon.config import AMOUNT_KEY, CASE_ID_KEY, CASE_TYPE_KEY, DURATION_KEY, REGION_KEY, make_rng
from rheon.metadata import metadata_payload
from rheon.xes_writer import write_xes_with_metadata
from rheon.validation import validate_xes, validation_passed


def test_validate_generated_xes_passes(tmp_path, small_config):
    out = tmp_path / "valid.xes"
    generate_and_write_log(small_config, [{"perspective": "data", "subtype": "numeric", "drift_type": "sudden"}], out, rng=make_rng(31))

    issues = validate_xes(out)

    assert validation_passed(issues)
    assert not [issue for issue in issues if issue["severity"] == "error"]


def test_validate_detects_duration_mismatch(tmp_path, small_config):
    generated = generate_and_write_log(small_config, [], tmp_path / "base.xes", rng=make_rng(32))
    corrupted = generated.dataframe.copy()
    corrupted.loc[0, DURATION_KEY] += 99
    payload = metadata_payload(
        log_name="corrupted",
        config=small_config.to_dict(),
        drifts=[],
        noise={"noise_probability": 0.0, "noisy_traces": 0},
    )
    out = tmp_path / "corrupted.xes"
    write_xes_with_metadata(corrupted, out, payload)

    issues = validate_xes(out, small_config)

    assert any(issue["code"] == "duration_mismatch" for issue in issues)


def test_pm4py_read_xes_does_not_create_double_case_prefixes(tmp_path, small_config):
    out = tmp_path / "native_pm4py_attrs.xes"
    generate_and_write_log(small_config, [], out, rng=make_rng(33))

    log = read_xes(str(out))
    trace_attributes = get_trace_attributes(log)
    event_attributes = get_event_attributes(log)

    assert not [column for column in trace_attributes + event_attributes if column.startswith("case:case:")]
    assert {CASE_ID_KEY, CASE_TYPE_KEY, AMOUNT_KEY, REGION_KEY}.issubset(set(trace_attributes))
    assert {CASE_ID_KEY, CASE_TYPE_KEY, AMOUNT_KEY, REGION_KEY}.issubset(set(event_attributes))
