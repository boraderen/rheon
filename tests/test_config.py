from __future__ import annotations

from rheon.config import GeneratorConfig


def test_from_uppercase_accepts_all_config_fields():
    config = GeneratorConfig.from_uppercase(
        {
            "TREE_GENERATION_ATTEMPTS": 1,
            "GRADUAL_OVERLAP_FRACTION": 0.25,
            "RECURRING_PERIOD_FRACTION": 0.35,
        }
    )

    assert config.tree_generation_attempts == 1
    assert config.gradual_overlap_fraction == 0.25
    assert config.recurring_period_fraction == 0.35
