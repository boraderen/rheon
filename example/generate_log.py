import rheon

DRIFTS = [
    {
        "type": "control_flow",
        "mode": "sudden",
        "drift_point": 0.35,
        "num_activities": 8,
        "tree_weights": {"sequence": 0.4, "choice": 0.4, "parallel": 0.15, "loop": 0.07},
    }
    #{"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 10},
    #{"type": "reassignment", "mode": "gradual", "start_point": 0.30, "end_point": 0.45},
    #{"type": "duration", "mode": "sudden", "drift_point": 0.6, "resources": 2, "factor": 1.8},
    #{"type": "arrival_rate", "mode": "sudden", "drift_point": 0.5, "factor": 0.5},
    #{"type": "amount", "mode": "gradual", "start_point": 0.65, "end_point": 0.85, "mean": 4000},
]

def main():
    rheon.generate_log(
        DRIFTS,
        "example/example.csv",
        format="csv",
        num_traces=10000,
        num_activities=10,
        num_resources=3,
        num_regions=3,
    )
    print("wrote example.csv and example_meta.md")


if __name__ == "__main__":
    main()
