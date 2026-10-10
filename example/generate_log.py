"""Generate an example log with one drift per perspective."""

import rheon

DRIFTS = [
    {"type": "control_flow", "mode": "sudden", "drift_point": 0.35, "num_activities": 8},
    {"type": "duration", "mode": "gradual", "start_point": 0.5, "end_point": 0.6, "resources": ["res_01"], "factor": 2.0},
    {"type": "arrival_rate", "mode": "sudden", "drift_point": 0.7, "factor": 0.5},
]


def main():
    rheon.generate_log(DRIFTS, "example/example.csv", format="csv", num_traces=2000)
    print("wrote example/example.csv and example/example_meta.md")


if __name__ == "__main__":
    main()
