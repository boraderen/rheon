from __future__ import annotations

import pandas as pd
import pytest

from rheon.case_attributes import fit_encoding_schema, transform_with_encoding_schema
from rheon.config import AMOUNT_KEY, REGION_KEY, sample_horizon
from rheon.drift.common import sample_drift_plans
from rheon.drift.data import build_data_runtime


@pytest.mark.parametrize(
    "subtype,affected",
    [
        ("numeric", [AMOUNT_KEY]),
        ("categorical", [REGION_KEY]),
    ],
)
def test_data_drift_subtypes_change_state_and_metadata(small_config, rng, subtype, affected):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "data", "subtype": subtype, "drift_type": "sudden", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    runtime = build_data_runtime(small_config, plans, rng)

    assert len(runtime.drifts) == 1
    assert runtime.metadata[0].affected_columns == affected
    assert runtime.metadata[0].perspective == "data"
    if subtype == "numeric":
        assert runtime.drifts[0].versions[0].amount_mean != runtime.drifts[0].versions[1].amount_mean
    else:
        assert runtime.drifts[0].versions[0].region_probs.tolist() != runtime.drifts[0].versions[1].region_probs.tolist()


def test_data_incremental_drift_has_multiple_versions(small_config, rng):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"perspective": "data", "subtype": "numeric", "drift_type": "incremental", "change_point": 0.5, "num_versions": 4}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
    )
    runtime = build_data_runtime(small_config, plans, rng)

    assert len(runtime.drifts[0].versions) == 4
    assert runtime.metadata[0].change_details["version_count"] == 4


def test_case_attribute_encoding_standardizes_numeric_and_uses_unk_category():
    df = pd.DataFrame({"amount": [10.0, 20.0, 30.0], "region": ["DE-NRW", "DE-BY", "DE-NRW"]})
    schema = fit_encoding_schema(df, ["amount"], ["region"])
    transformed = transform_with_encoding_schema(
        pd.DataFrame({"amount": [20.0, 40.0], "region": ["DE-BY", "DE-UNKNOWN"]}),
        schema,
    )

    assert transformed["amount"].iloc[0] == pytest.approx(0.0)
    assert "region=UNK" in transformed.columns
    assert transformed["region=UNK"].iloc[1] == 1.0
