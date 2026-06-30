from __future__ import annotations

import pytest

from rheon.config import GeneratorConfig


@pytest.fixture
def small_config() -> GeneratorConfig:
    """A small, fast configuration for tests."""
    return GeneratorConfig(num_traces=80, num_activities=6, num_resources=5, num_regions=3, seed=123)
