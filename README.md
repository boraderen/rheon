# Rheon

A library for synthetic event log generation with injected concept drifts. You describe a base process and a list of drifts and Rheon produces the log plus a metadata file that records exactly where each drift is and what it changed.

The only Rheon function you need to use is `rheon.generate_log()`.

## Install

```bash
git clone <repo-url> rheon
cd rheon
uv sync
```

### Example

Run the bundled script to generate a labeled log:

```bash
uv run python example/generate_log.py
```

It injects drifts across every perspective and writes `example/example.csv` plus its
`example_meta.md` ground-truth sidecar.

## Quickstart

Describe a base process and a list of drifts, then write a labeled log.

```python
import rheon

drifts = [
    {"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 9},
    {"type": "reassignment", "mode": "gradual", "start_point": 0.3, "end_point": 0.45},
    {"type": "amount", "mode": "sudden", "drift_point": 0.6, "mean": 4000},
]

rheon.generate_log(drifts, "out/example.xes", num_traces=2000, num_activities=8)
```

This writes two files: the log `out/example.xes` and a ground-truth metadata sidecar
`out/example_meta.md`.

Read the generated log back with pm4py:

```python
import pm4py

log = pm4py.read_xes("out/example.xes")
```



## Drift types

Every drift is one dict with a `type`, a `mode` (`"sudden"` or `"gradual"`), and a position:
`drift_point` for sudden drifts, or `start_point`/`end_point` for gradual ones. All positions are
fractions of the time horizon in `(0, 1)`. The remaining keys are type-specific.


| Perspective | Type           | What changes                                                                                    | Type-specific params             |
| ----------- | -------------- | ----------------------------------------------------------------------------------------------- | -------------------------------- |
| intra-case  | `control_flow` | activity-ordering structure: cases after the drift are played out from a different process tree | `num_activities`, `tree_weights` |
| resource    | `pool_size`    | resources are added or removed; durations scale the opposite way                                | `delta`, `duration_factor`       |
| resource    | `reassignment` | a new dominant resource is chosen for every activity                                            | —                                |
| resource    | `workload`     | traces are duplicated (or dropped); per-resource case load shifts                               | `workload_factor`                |
| resource    | `duration`     | the processing time of the given resources is scaled                                            | `resources`, `factor`            |
| inter-case  | `waiting_time` | the mean waiting gap between events shifts                                                      | `mean`, `variance`               |
| inter-case  | `amount`       | case amounts are drawn from a shifted distribution                                              | `mean`, `variance`               |
| inter-case  | `arrival_rate` | the mean gap between case arrivals changes                                                      | `inter_arrival` or `factor`      |
| inter-case  | `region`       | a new dominant region is chosen for later cases                                                 | —                                |




## Parameters

Function signature and generator options for `rheon.generate_log()`:

```python
rheon.generate_log(
    drifts,
    output_path,
    *,
    log_name=None,
    format="xes",          # "xes" or "csv"
    num_traces=1000,
    num_activities=10,
    num_resources=8,
    num_regions=4,
    tree_weights={"sequence": 0.6, "choice": 0.25, "parallel": 0.1, "loop": 0.05},
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 12, 31),
    activity_duration=(30.0, 100.0),
    waiting_time=(15.0, 50.0),
    amount=(1000.0, 40000.0),
    seed=42,
)
```

`drifts` is a list of dictionaries, each with a `type`, a `mode`, and a position. The parameters
below all describe the single base process.


| Argument            | Default             | Description                                                                    |
| ------------------- | ------------------- | ------------------------------------------------------------------------------ |
| `drifts`            | —                   | List of drift dictionaries to inject.                                          |
| `output_path`       | —                   | Destination of the log.                                                        |
| `log_name`          | file stem           | Name used in metadata and the sidecar filename.                                |
| `format`            | `"xes"`             | `"xes"` or `"csv"`.                                                            |
| `num_traces`        | `1000`              | Approximate number of cases (not strict).                                      |
| `num_activities`    | `10`                | Activities in the base process tree.                                           |
| `num_resources`     | `8`                 | Size of the resource pool.                                                     |
| `num_regions`       | `4`                 | Number of regions.                                                             |
| `tree_weights`      | see above           | Operator weights for the base tree (`sequence`, `choice`, `parallel`, `loop`). |
| `start_date`        | `2020-01-01`        | Start of the time horizon.                                                     |
| `end_date`          | `2020-12-31`        | End of the time horizon (with `start_date` it fixes the window).               |
| `activity_duration` | `(30.0, 100.0)`     | `(mean, variance)` of activity processing time in minutes.                     |
| `waiting_time`      | `(15.0, 50.0)`      | `(mean, variance)` of the waiting gap between events.                          |
| `amount`            | `(1000.0, 40000.0)` | `(mean, variance)` of the case amount.                                         |
| `seed`              | `42`                | Random seed.                                                                   |


Cases are spread across `[start_date, end_date]`. The mean inter-arrival gap is derived as
`(end_date − start_date) / num_traces`. The `arrival_rate` drift changes that derived rate after
its drift point.

## Output files and metadata

Each run writes two files next to `output_path`:

- `<name>.xes` (or `<name>.csv`) — the event log. Events carry `concept:name`,
`start_timestamp`, `time:timestamp`, `event:duration_min` and `org:resource`; each case carries
`amount` and `region`. For XES the full metadata is also embedded as a log-level `rheon:metadata`
attribute.
- `<name>_meta.md` — a short, readable ground-truth report with three sections:
  1. **General parameters** — the structural, temporal and attribute parameters of the run.
  2. **Base distributions** — the starting state: every activity's dominant resource and its duration
    and waiting distributions `(mean, var)`, plus the case-level amount distribution, inter-arrival
     mean and dominant region.
  3. **Drifts** — each drift with its mode and drift point / window (as both a horizon fraction and an
    absolute timestamp), followed by exactly which distributions or assignments it changed, per
     activity / resource (e.g. a reassignment's before → after resource table, or an amount drift's
     `mean: 1000 → 4000`).



## License & status

Rheon is research software for generating drift benchmarks.