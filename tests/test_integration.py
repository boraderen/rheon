from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import rheon
from rheon.composer import generate_log as generate_log_in_memory
from rheon.config import GeneratorConfig, make_rng
from rheon.timestamp_assignment import assign_event_times
from rheon.validation import validate_xes, validation_passed


def _fixed_pipeline_options() -> dict[str, object]:
    return {
        "num_traces": 12,
        "min_trace_length": 3,
        "max_trace_length": 5,
        "avg_trace_length": 4,
        "trace_length_variance": 1,
        "min_activities": 4,
        "max_activities": 4,
        "tree_generation_attempts": 1,
        "sequence_weight": 1.0,
        "choice_weight": 0.0,
        "parallel_weight": 0.0,
        "loop_weight": 0.0,
        "or_weight": 0.0,
        "silent_transition_prob": 0.0,
        "duplicate_activity_prob": 0.0,
        "num_resources": 4,
        "num_roles": 2,
        "num_case_types": 2,
        "horizon_min_days": 1,
        "horizon_max_days": 1,
        "noise_probability": 0.0,
        "global_seed": 123,
    }


@pytest.mark.parametrize(
    "drifts,prefix",
    [
        (
            [{"perspective": "control_flow", "subtype": "tree_mutation", "drift_type": "sudden", "change_point": 0.5, "change_proportion": 0.05}],
            "control_flow",
        ),
        (
            [{"perspective": "resource", "subtype": "pool_size", "drift_type": "sudden", "change_point": 0.5}],
            "resource",
        ),
        (
            [{"perspective": "inter_case", "subtype": "arrival_rate", "drift_type": "sudden", "change_point": 0.5}],
            "inter_case",
        ),
        (
            [{"perspective": "data", "subtype": "numeric", "drift_type": "sudden", "change_point": 0.5}],
            "data",
        ),
        (
            [
                {"perspective": "resource", "subtype": "pool_size", "drift_type": "sudden", "change_point": 0.5},
                {"perspective": "data", "subtype": "numeric", "drift_type": "sudden", "change_point": 0.5},
            ],
            "multi",
        ),
    ],
    ids=["control_flow", "resource", "inter_case", "data", "multi"],
)
def test_full_pipeline_for_small_fixed_scenarios(tmp_path, small_config, drifts, prefix):
    out = tmp_path / f"{prefix}.xes"

    rheon.generate_log(
        drifts,
        out,
        rng=101,
        **_fixed_pipeline_options(),
    )
    issues = validate_xes(out)

    assert validation_passed(issues), issues


def test_reproducibility_same_seed_identical_log(small_config):
    drifts = [{"perspective": "data", "subtype": "numeric", "drift_type": "sudden"}]

    first = generate_log_in_memory(small_config, drifts)
    second = generate_log_in_memory(small_config, drifts)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.metadata["drifts"] == second.metadata["drifts"]


def test_control_flow_metadata_uses_observed_variant_counts_only(small_config):
    generated = generate_log_in_memory(
        small_config,
        [{"perspective": "control_flow", "subtype": "tree_mutation", "drift_type": "sudden"}],
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
