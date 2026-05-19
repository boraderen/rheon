from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from scripts import generate_control_flow_log, generate_data_log, generate_inter_case_log, generate_multi_perspective_log, generate_resource_log
from rheon.composer import generate_and_write_log, generate_log
from rheon.config import GeneratorConfig, make_rng
from rheon.timestamp_assignment import assign_event_times
from rheon.validation import validate_xes, validation_passed


@pytest.mark.parametrize(
    "module,default_perspective,prefix",
    [
        (generate_control_flow_log, "control_flow", "control_flow"),
        (generate_resource_log, "resource", "resource"),
        (generate_inter_case_log, "inter_case", "inter_case"),
        (generate_data_log, "data", "data"),
        (generate_multi_perspective_log, "control_flow", "multi"),
    ],
)
def test_full_pipeline_for_each_generation_script(tmp_path, small_config, module, default_perspective, prefix):
    config = replace(small_config, output_path=str(tmp_path), noise_probability=0.0)
    out = tmp_path / f"{prefix}.xes"

    generate_and_write_log(
        config,
        module.DRIFTS,
        out,
        default_perspective=default_perspective,
        rng=make_rng(101),
    )
    issues = validate_xes(out)

    assert validation_passed(issues), issues


def test_reproducibility_same_seed_identical_log(small_config):
    drifts = [{"perspective": "data", "subtype": "numeric", "drift_type": "sudden"}]

    first = generate_log(small_config, drifts)
    second = generate_log(small_config, drifts)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.metadata["drifts"] == second.metadata["drifts"]


def test_control_flow_metadata_uses_observed_variant_counts_only(small_config):
    generated = generate_log(
        small_config,
        [{"subtype": "tree_mutation", "drift_type": "sudden"}],
        default_perspective="control_flow",
        rng=make_rng(55),
    )

    drift = generated.metadata["drifts"][0]
    details = drift["change_details"]

    assert "variant_count_before" in details
    assert "variant_count_after" in details
    assert "variant_counts_by_version" in details
    assert "structural_variant_estimate_before" not in details
    assert "structural_variant_estimate_after" not in details
    assert "observed_variant_count_before" not in details
    assert "observed_variant_count_after" not in details
    assert "observed_variant_counts_by_version" not in details
    assert generated.metadata["config"]["num_trace_variants"] >= 1


@given(length=st.integers(min_value=1, max_value=30), seed=st.integers(min_value=1, max_value=2**31 - 1))
@settings(max_examples=35)
def test_timestamp_assignment_monotonicity_property(length, seed):
    config = GeneratorConfig(service_time_mean_min=10, service_time_std_min=5)
    times = assign_event_times(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        length,
        config,
        make_rng(seed),
    )

    previous_completion = None
    for start, end, duration in times:
        assert start <= end
        assert duration == pytest.approx((end - start).total_seconds() / 60.0)
        if previous_completion is not None:
            assert previous_completion <= start
        previous_completion = end
