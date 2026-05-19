from __future__ import annotations

from datetime import datetime

import numpy as np

from rheon.drift.resource import ResourceRuntime


def assign_resource_and_role(
    runtime: ResourceRuntime,
    activity: str,
    previous_resource: str | None,
    timestamp: datetime,
    rng: np.random.Generator,
) -> tuple[str, str]:
    return runtime.assign(activity, previous_resource, timestamp, rng)
