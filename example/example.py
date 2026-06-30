"""Generate one mixed-drift example log.

Run with:  uv run python example/example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rheon


# A scenario that touches every perspective: one intra-case, two resource and
# two inter-case drifts, mixing sudden and gradual modes.
DRIFTS = [
    {"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 9},
    {"type": "reassignment", "mode": "gradual", "start_point": 0.30, "end_point": 0.45},
    {"type": "duration", "mode": "sudden", "drift_point": 0.6, "resources": 2, "factor": 1.8},
    {"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5, "factor": 0.5},
    {"type": "amount", "mode": "gradual", "start_point": 0.65, "end_point": 0.85, "mean": 4000},
]


def main() -> None:
    """Generate the example log as XES (and again as CSV) with its metadata sidecar."""
    rheon.generate_log(
        DRIFTS,
        "data/example/example.xes",
        log_name="example",
        num_traces=2000,
        num_activities=8,
        num_resources=8,
        num_regions=4,
    )
    rheon.generate_log(
        DRIFTS,
        "data/example/example.csv",
        log_name="example",
        format="csv",
        num_traces=2000,
        num_activities=8,
        num_resources=8,
        num_regions=4,
    )
    print("wrote data/example/example.xes, example.csv and example_meta.md")


if __name__ == "__main__":
    main()
