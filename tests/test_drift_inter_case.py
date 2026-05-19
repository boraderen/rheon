from __future__ import annotations

import pytest

from rheon.config import sample_horizon
from rheon.drift.common import sample_drift_plans
from rheon.drift.inter_case import build_inter_case_runtime, generate_case_arrivals


@pytest.mark.parametrize(
    "subtype,expected_key",
    [
        ("arrival_rate", "new_interarrival_mean_multiplier"),
        ("burstiness", "new_shape"),
        ("case_mix", "dominant_case_type"),
        ("concurrency", "direct_injection"),
    ],
)
def test_inter_case_drift_subtypes_are_recorded(small_config, rng, subtype, expected_key):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "inter_case", "subtype": subtype, "drift_type": "sudden", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    runtime = build_inter_case_runtime(small_config, plans, rng)

    assert len(runtime.drifts) == 1
    assert expected_key in runtime.drifts[0].details
    assert runtime.metadata[0].perspective == "inter_case"
    if subtype == "concurrency":
        assert runtime.drifts[0].details["direct_injection"] is False


def test_arrival_rate_drift_changes_case_count_from_fixed_target(small_config, rng):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "inter_case", "subtype": "arrival_rate", "drift_type": "sudden", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    runtime = build_inter_case_runtime(small_config, plans, rng)
    arrivals = generate_case_arrivals(small_config, runtime, start, end, rng)

    assert arrivals
    assert len(arrivals) != small_config.num_traces or runtime.drifts[0].details["new_interarrival_mean_multiplier"] != 1.0
    assert all(start <= arrival <= end for arrival in arrivals)
