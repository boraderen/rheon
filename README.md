<img src="rheon.png" alt="Rheon icon" width="96">

# Rheon

A library for synthetic event log generation with injected concept drifts. You describe a base
process and a list of drifts, and Rheon writes the log (XES or CSV) plus a metadata file that records
exactly where each drift is and what it changed.

**Documentation:** https://boraderen.github.io/rheon/

## Install

```bash
git clone git@github.com:boraderen/rheon.git
cd rheon
uv sync
```

## Quick start

```python
import rheon

drifts = [
    {"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 9},
    {"type": "reassignment", "mode": "gradual", "start_point": 0.3, "end_point": 0.45},
    {"type": "amount", "mode": "sudden", "drift_point": 0.6, "mean": 4000},
]

rheon.generate_log(drifts, "out/example.xes", num_traces=2000, num_activities=8)
```

This writes the log `out/example.xes` and its ground truth `out/example_meta.md`. The drift modes,
the nine drift types and all options are described in the
[documentation](https://boraderen.github.io/rheon/).

## Generation and output

Rheon validates the configuration and drift list, builds process-tree versions and playout pools,
and initializes distributions for the union of their activities. It samples case arrivals with
both `arrival_rate` and `workload` drift, then samples each case's activity sequence from the
appropriate pool. Reassignment, region, amount, waiting-time and pool-size drifts run by start
fraction; duration drifts run afterward so they follow final resource assignments. Ties retain
input order. Workload changes arrival density; it does not duplicate or delete existing cases.

Event drifts use the original event start times stored before attribute changes. Final timestamps
are recomputed afterward, rounded to UTC whole seconds, and durations are recalculated in minutes.
The horizon bounds case arrivals, so events can finish beyond its end. The CSV columns are
`event:id`, `case:concept:name`, `concept:name`, `start_timestamp`, `time:timestamp`,
`event:duration_min`, `org:resource`, `case:amount` and `case:region`. Amount and region repeat
on each case's events; in XES they are trace attributes, and full metadata is embedded in
`rheon:metadata`. Reference event times and per-event drift labels are not exported.

Successive arrival-rate factors multiply the previous gap level; workload factors multiply the
previous workload level. Pool shrinkage multiplies durations by `duration_factor`, while growth
uses its reciprocal. Any positive factor is accepted, so a value below 1 reverses the default
slower-on-shrinkage behavior. Added resources receive distinct dominant activities only until
the activity list runs out. Resources have no capacity limit.

The seed controls random draws, but PM4Py can interleave parallel branches differently between
runs. Save generated logs and metadata when exact experimental replication is needed.

## Example and tests

```bash
uv run python example/generate_log.py   # writes example/example.csv and example/example_meta.md
uv run pytest
```
