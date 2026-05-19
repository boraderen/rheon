from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rheon


# Output
OUTPUT_PATH = "data/resource/"
NUM_LOGS = 1
GLOBAL_SEED = 7

# Log size - very big and long so drift regimes are unmistakable.
NUM_TRACES = 1000
MIN_TRACE_LENGTH = 5
MAX_TRACE_LENGTH = 7
AVG_TRACE_LENGTH = 6
TRACE_LENGTH_VARIANCE = 9

# Time horizon (auto-scaled to event count)
HORIZON_MIN_DAYS = 365
HORIZON_MAX_DAYS = 365

# Process complexity - kept simple so the resource signal isn't drowned out
# by control-flow variation.
MIN_ACTIVITIES = 6
MAX_ACTIVITIES = 6
TREE_DEPTH_MIN = 2
TREE_DEPTH_MAX = 2
SEQUENCE_WEIGHT = 0.85
CHOICE_WEIGHT = 0.10
PARALLEL_WEIGHT = 0.02
LOOP_WEIGHT = 0.0
OR_WEIGHT = 0.01
SILENT_TRANSITION_PROB = 0.0
DUPLICATE_ACTIVITY_PROB = 0.0

# Resources
NUM_RESOURCES = 8
NUM_ROLES = 2
NUM_CASE_TYPES = 3
REGIONS = ["DE-NRW", "DE-BY", "DE-HE", "DE-BW", "DE-BE"]

# Timing model
INTER_ARRIVAL_MEAN_MIN = 30
SERVICE_TIME_MEAN_MIN = 15
SERVICE_TIME_STD_MIN = 5

# Noise
NOISE_PROBABILITY = 0
NOISE_SIMILAR_VS_RANDOM = 0

# Drift configuration - only resource drifts (no CF / inter-case), placed at
# fixed positions so each dashboard feature family has a clean interval to
# react to. The resource-specific subtypes use their current targeting knobs:
# activity lists for reassignment and resource lists / "all" for workload and
# service-time shifts.
DRIFTS = [
    {"perspective": "resource", "subtype": "pool_size", "drift_type": "sudden", "change_point": 0.50},
    {"perspective": "resource", "subtype": "workload_distribution", "drift_type": "sudden", "change_point": 0.50, "resources": "all"},
    {"perspective": "resource", "subtype": "reassignment", "drift_type": "sudden", "change_point": 0.50, "activities": ["a", "c"]},
    {"perspective": "resource", "subtype": "service_time", "drift_type": "sudden", "change_point": 0.50, "resources": "all"},
    {"perspective": "resource", "subtype": "handover", "drift_type": "sudden", "change_point": 0.50},
]


if __name__ == "__main__":
    for index in range(NUM_LOGS):
        stem = f"resource_{index + 1:03d}"
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
