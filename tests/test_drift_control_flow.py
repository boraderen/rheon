from __future__ import annotations

import pytest

from rheon.config import ACTIVITY_KEY, sample_horizon
from rheon.drift.common import gradual_probability, sample_drift_plans
from rheon.drift.control_flow import build_control_flow_model


def test_gradual_overlap_math_is_linear_and_exponential(small_config, rng):
    start, end = sample_horizon(small_config, rng)
    midpoint = start + (end - start) / 2

    assert gradual_probability("gradual_linear", start, start, end) == 0.0
    assert gradual_probability("gradual_linear", midpoint, start, end) == 0.5
    assert gradual_probability("gradual_linear", end, start, end) == 1.0
    assert gradual_probability("gradual_exponential", midpoint, start, end) < 0.5


def test_control_flow_drift_builds_versions_and_gold_standard(small_config, rng):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "control_flow", "subtype": "tree_mutation", "drift_type": "sudden", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    model = build_control_flow_model(small_config, plans, rng)

    assert len(model.drifts) == 1
    assert len(model.drifts[0].versions) == 2
    assert model.metadata[0].perspective == "control_flow"
    assert model.metadata[0].affected_columns == [ACTIVITY_KEY]
    assert "activities_added" in model.metadata[0].change_details


@pytest.mark.parametrize(
    "drift_type,min_versions",
    [
        ("sudden", 2),
        ("gradual_linear", 2),
        ("gradual_exponential", 2),
        ("incremental", 3),
        ("recurring", 2),
    ],
)
def test_control_flow_supports_all_drift_types(small_config, rng, drift_type, min_versions):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "control_flow", "subtype": "tree_mutation", "drift_type": drift_type, "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    model = build_control_flow_model(small_config, plans, rng)

    assert len(model.drifts[0].versions) >= min_versions
    assert start < plans[0].change_point < end
    if drift_type.startswith("gradual"):
        assert plans[0].overlap_start < plans[0].change_point < plans[0].overlap_end
