from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rheon.config import GeneratorConfig
from rheon.script_api import run_generation


# Output
OUTPUT_PATH = "data/data-attributes/"
NUM_LOGS = 1
GLOBAL_SEED = 7

NUM_TRACES = 6000
MIN_TRACE_LENGTH = 4
MAX_TRACE_LENGTH = 8
AVG_TRACE_LENGTH = 6
TRACE_LENGTH_VARIANCE = 25

HORIZON_MIN_DAYS = 300
HORIZON_MAX_DAYS = 365

MIN_ACTIVITIES = 5
MAX_ACTIVITIES = 6
TREE_DEPTH_MIN = 2
TREE_DEPTH_MAX = 2
SEQUENCE_WEIGHT = 0.85
CHOICE_WEIGHT = 0.10
PARALLEL_WEIGHT = 0.01
LOOP_WEIGHT = 0.0
OR_WEIGHT = 0.01
SILENT_TRANSITION_PROB = 0.0
DUPLICATE_ACTIVITY_PROB = 0.0

NUM_RESOURCES = 10
NUM_ROLES = 3
NUM_CASE_TYPES = 3
REGIONS = ["DE-NRW", "DE-BY", "DE-HE", "DE-BW", "DE-BE"]

INTER_ARRIVAL_MEAN_MIN = 30
SERVICE_TIME_MEAN_MIN = 15
SERVICE_TIME_STD_MIN = 7

NOISE_PROBABILITY = 0
NOISE_SIMILAR_VS_RANDOM = 0

# Three sudden data drifts: two numeric jumps and one categorical shift, all
# affecting case-level attributes (case:amount and case:region).
DRIFTS = [
    {"subtype": "numeric", "drift_type": "sudden"},
    {"subtype": "categorical", "drift_type": "sudden"},
    {"subtype": "numeric", "drift_type": "sudden"},
]


def build_config() -> GeneratorConfig:
    return GeneratorConfig.from_uppercase(globals())


def generate_logs():
    return run_generation(
        build_config(),
        DRIFTS,
        filename_prefix="data",
        default_perspective="data",
    )


if __name__ == "__main__":
    generate_logs()
