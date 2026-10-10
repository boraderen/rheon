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

The paper examples use three logs, each with one sudden midpoint drift, and one plot per log:
`control_flow.csv` changes a five-step sequence into two short branches (side-by-side directly-follows count matrices),
`resource_reassignment.csv` uses three resources and changes each activity's dominant resource
(side-by-side activity-resource frequency matrices), and `arrival_rate.csv` increases the arrival gap by 20× so the arrival rate drops to
5% (dotted chart). The matrices use the entire log, split by case arrival for directly-follows counts and event
start for activity-resource counts. The plots retain the notebooks' counting and case-ordering logic. Rebuild them with:

```bash
uv run --with matplotlib python example/generate_paper_examples.py
uv run --with matplotlib python example/generate_paper_examples.py --plots-only
uv run --with matplotlib python example/generate_paper_examples.py --regenerate resource_reassignment
latexmk -pdf -cd paper/paper.tex
```

The first command resamples the logs and updates their metadata, plots and paper table rows.
The second rebuilds plots and tables from saved logs. The third resamples only the resource log
and rebuilds the plots and tables. Generated files are under `paper/logs`,
`paper/figures` and `paper/generated`; the manifest records settings, package versions and CSV hashes.
The manifest also records measured drift signals. Each base process has five activities, no
parallelism or loops and three regions. The reassignment example has three resources; the other
two have four; the control-flow change adds a choice.
The local `paper/` directory is ignored by Git, so keep its artifacts alongside the paper when sharing it.
