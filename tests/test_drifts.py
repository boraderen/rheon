from __future__ import annotations

import pytest

from rheon.drifts import normalize_drifts


def test_sudden_drift_defaults_window_to_point():
    [drift] = normalize_drifts([{"type": "amount", "mode": "sudden", "drift_point": 0.4}])
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
        normalize_drifts([{"type": "amount", "mode": "gradual", "start_point": 0.6, "end_point": 0.4}])


def test_drift_point_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_drifts([{"type": "amount", "mode": "sudden", "drift_point": 1.5}])


def test_two_control_flow_drifts_may_not_overlap():
    with pytest.raises(ValueError, match="overlap"):
        normalize_drifts([
            {"type": "control_flow", "mode": "gradual", "start_point": 0.3, "end_point": 0.6},
            {"type": "control_flow", "mode": "gradual", "start_point": 0.5, "end_point": 0.8},
        ])


def test_non_overlapping_same_type_is_allowed():
    drifts = normalize_drifts([
        {"type": "control_flow", "mode": "gradual", "start_point": 0.3, "end_point": 0.5},
        {"type": "control_flow", "mode": "sudden", "drift_point": 0.7},
    ])
    assert len(drifts) == 2
