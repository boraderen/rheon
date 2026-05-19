from __future__ import annotations

import pytest

from rheon.config import RESOURCE_KEY, ROLE_KEY, sample_horizon
from rheon.drift.common import sample_drift_plans
from rheon.drift.resource import build_resource_runtime


@pytest.mark.parametrize(
    "subtype,expected_key",
    [
        ("reassignment", "reassignments"),
        ("pool_size", "new_pool_size"),
        ("handover", "new_dominant_target"),
        ("workload_distribution", "heavy_resources"),
        ("service_time", "multiplier_changes"),
    ],
)
def test_resource_drift_subtypes_change_state(small_config, rng, subtype, expected_key):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "resource", "subtype": subtype, "drift_type": "sudden", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    runtime = build_resource_runtime(small_config, plans, small_config.activity_pool[:5], rng)

    assert len(runtime.drifts) == 1
    assert expected_key in runtime.drifts[0].details
    assert runtime.metadata[0].perspective == "resource"
    assert runtime.metadata[0].affected_columns == [RESOURCE_KEY, ROLE_KEY]


def test_resource_gradual_runtime_samples_both_versions(small_config, rng):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "resource", "subtype": "pool_size", "drift_type": "gradual_linear", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    runtime = build_resource_runtime(small_config, plans, small_config.activity_pool[:5], rng)
    plan = plans[0]

    old_size = len(runtime.drifts[0].versions[0].active_resources)
    new_size = len(runtime.drifts[0].versions[1].active_resources)
    sampled_sizes = {
        len(runtime.state_for(plan.change_point, rng).active_resources)
        for _ in range(100)
    }

    assert {old_size, new_size}.issubset(sampled_sizes)


def test_workload_distribution_targets_selected_resources(small_config, rng):
    resources = small_config.resources[:2]
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{
            "perspective": "resource",
            "subtype": "workload_distribution",
            "drift_type": "sudden",
            "change_point": 0.5,
            "resources": resources,
        }],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    runtime = build_resource_runtime(small_config, plans, small_config.activity_pool[:5], rng)
    final_state = runtime.drifts[0].final_state
    target_indices = [final_state.resources.index(resource) for resource in resources]

    assert runtime.drifts[0].details["affected_resources"] == resources
    assert runtime.drifts[0].details["heavy_resources"] == resources
    for probs in final_state.activity_probs.values():
        assert sum(probs[index] for index in target_indices) == pytest.approx(0.7)
