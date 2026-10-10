from __future__ import annotations

import pytest

from rheon.config import GeneratorConfig
from rheon.drifts import normalize_drifts


def test_sudden_drift_defaults_window_to_point():
    [drift] = normalize_drifts([{"type": "amount", "mode": "sudden", "drift_point": 0.4, "mean": 5000}])
    assert drift.start_frac == drift.end_frac == 0.4
    assert drift.ramp(0.3) == 0.0
    assert drift.ramp(0.5) == 1.0


def test_gradual_drift_ramps_linearly():
    [drift] = normalize_drifts([{"type": "region", "mode": "gradual", "start_point": 0.4, "end_point": 0.6}])
    assert drift.ramp(0.4) == 0.0
    assert drift.ramp(0.5) == pytest.approx(0.5)
    assert drift.ramp(0.6) == 1.0


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="unknown type"):
        normalize_drifts([{"type": "nope", "mode": "sudden", "drift_point": 0.5}])


def test_gradual_needs_ordered_window():
    with pytest.raises(ValueError, match="start_point must be smaller"):
        normalize_drifts([{"type": "region", "mode": "gradual", "start_point": 0.6, "end_point": 0.4}])


def test_drift_point_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_drifts([{"type": "region", "mode": "sudden", "drift_point": 1.5}])


@pytest.mark.parametrize("raw, message", [
    ({"type": "region", "drift_point": 0.5}, "mode must be"),
    ({"type": "region", "mode": "sudden"}, "needs"),
    ({"type": "region", "mode": "gradual", "start_point": 0.2}, "needs"),
    ({"type": "region", "mode": "sudden", "start_point": 0.2, "end_point": 0.3}, "unexpected keys"),
    ({"type": "amount", "mode": "sudden", "drift_point": 0.5, "maen": 4000}, "unexpected keys"),
])
def test_mode_position_and_keys_are_checked(raw, message):
    with pytest.raises(ValueError, match=message):
        normalize_drifts([raw])


@pytest.mark.parametrize("raw, message", [
    ({"type": "workload", "mode": "sudden", "drift_point": 0.5}, "needs workload_factor"),
    ({"type": "workload", "mode": "sudden", "drift_point": 0.5, "workload_factor": 1}, "changes nothing"),
    ({"type": "amount", "mode": "sudden", "drift_point": 0.5}, "needs mean and/or variance"),
    ({"type": "waiting_time", "mode": "sudden", "drift_point": 0.5, "mean": -5}, "positive"),
    ({"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5}, "exactly one"),
    ({"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5, "inter_arrival": 60, "factor": 0.5}, "exactly one"),
    ({"type": "pool_size", "mode": "sudden", "drift_point": 0.5, "delta": 0}, "non-zero"),
    ({"type": "duration", "mode": "sudden", "drift_point": 0.5, "resources": "res_01"}, "resources"),
    ({"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "tree_weights": {"xor": 1}}, "tree_weights"),
])
def test_invalid_or_no_op_parameters_are_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        normalize_drifts([raw])


def test_region_drift_needs_two_regions():
    with pytest.raises(ValueError, match="num_regions"):
        normalize_drifts([{"type": "region", "mode": "sudden", "drift_point": 0.5}], GeneratorConfig(num_regions=1))


def test_two_control_flow_drifts_may_not_overlap():
    with pytest.raises(ValueError, match="overlap"):
        normalize_drifts([
            {"type": "control_flow", "mode": "gradual", "start_point": 0.3, "end_point": 0.6},
            {"type": "control_flow", "mode": "gradual", "start_point": 0.5, "end_point": 0.8},
        ])


def test_two_sudden_drifts_of_one_type_may_not_share_a_point():
    with pytest.raises(ValueError, match="overlap"):
        normalize_drifts([
            {"type": "amount", "mode": "sudden", "drift_point": 0.5, "mean": 10},
            {"type": "amount", "mode": "sudden", "drift_point": 0.5, "mean": 20},
        ])


def test_non_overlapping_same_type_is_allowed():
    drifts = normalize_drifts([
        {"type": "control_flow", "mode": "gradual", "start_point": 0.3, "end_point": 0.5},
        {"type": "control_flow", "mode": "sudden", "drift_point": 0.7},
    ])
    assert len(drifts) == 2
