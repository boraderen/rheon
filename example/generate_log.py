import rheon

DRIFTS = [
    {"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 9},
    {"type": "reassignment", "mode": "gradual", "start_point": 0.30, "end_point": 0.45},
    {"type": "duration", "mode": "sudden", "drift_point": 0.6, "resources": 2, "factor": 1.8},
    {"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5, "factor": 0.5},
    {"type": "amount", "mode": "gradual", "start_point": 0.65, "end_point": 0.85, "mean": 4000},
]

def main():
    rheon.generate_log(
        DRIFTS,
        "example/example.csv",
        format="csv",
        num_traces=2000,
        num_activities=8,
        num_resources=8,
        num_regions=4,
    )
    print("wrote example.csv and example_meta.md")


if __name__ == "__main__":
    main()
