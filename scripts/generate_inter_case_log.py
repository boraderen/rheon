from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rheon


# Output
OUTPUT_PATH = "data/inter-case/"
NUM_LOGS = 1
GLOBAL_SEED = 7

# Big and long; simple tree so the timing/case-mix signal isn't drowned
# out by control-flow variation.
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

# Three sudden timing shifts (arrival rate, burstiness, concurrency) and a
# case-mix change. All sudden so the change points are easy to read.
DRIFTS = [
    {"perspective": "inter_case", "subtype": "arrival_rate", "drift_type": "sudden"},
    {"perspective": "inter_case", "subtype": "burstiness", "drift_type": "sudden"},
    {"perspective": "inter_case", "subtype": "case_mix", "drift_type": "sudden"},
    {"perspective": "inter_case", "subtype": "concurrency", "drift_type": "sudden"},
]


if __name__ == "__main__":
    for index in range(NUM_LOGS):
        stem = f"inter_case_{index + 1:03d}"
        target_path = Path(OUTPUT_PATH) / f"{stem}.xes"
        rheon.generate_log(
            DRIFTS,
            target_path,
            log_name=stem,
            global_seed=GLOBAL_SEED + index,
            num_traces=NUM_TRACES,
            min_trace_length=MIN_TRACE_LENGTH,
            max_trace_length=MAX_TRACE_LENGTH,
            avg_trace_length=AVG_TRACE_LENGTH,
            trace_length_variance=TRACE_LENGTH_VARIANCE,
            horizon_min_days=HORIZON_MIN_DAYS,
            horizon_max_days=HORIZON_MAX_DAYS,
            min_activities=MIN_ACTIVITIES,
            max_activities=MAX_ACTIVITIES,
            tree_depth_min=TREE_DEPTH_MIN,
            tree_depth_max=TREE_DEPTH_MAX,
            sequence_weight=SEQUENCE_WEIGHT,
            choice_weight=CHOICE_WEIGHT,
            parallel_weight=PARALLEL_WEIGHT,
            loop_weight=LOOP_WEIGHT,
            or_weight=OR_WEIGHT,
            silent_transition_prob=SILENT_TRANSITION_PROB,
            duplicate_activity_prob=DUPLICATE_ACTIVITY_PROB,
            num_resources=NUM_RESOURCES,
            num_roles=NUM_ROLES,
            num_case_types=NUM_CASE_TYPES,
            regions=REGIONS,
            inter_arrival_mean_min=INTER_ARRIVAL_MEAN_MIN,
            service_time_mean_min=SERVICE_TIME_MEAN_MIN,
            service_time_std_min=SERVICE_TIME_STD_MIN,
            noise_probability=NOISE_PROBABILITY,
            noise_similar_vs_random=NOISE_SIMILAR_VS_RANDOM,
        )
        print(f"wrote {target_path}")
