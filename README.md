# Rheon

Rheon generates synthetic **XES event logs with known concept drift** for process-mining
experiments. You describe a base process and a list of drifts; Rheon produces the log plus a
metadata file that records exactly where each drift is and what it changed — the ground truth you
can evaluate a drift-detection pipeline against.

## Install

```bash
git clone <repo-url> rheon
cd rheon
uv sync
```

## Quickstart

```python
import rheon

drifts = [
    {"type": "control_flow", "mode": "sudden", "drift_point": 0.5, "num_activities": 9},
    {"type": "reassignment", "mode": "gradual", "start_point": 0.3, "end_point": 0.45},
    {"type": "amount", "mode": "sudden", "drift_point": 0.6, "mean": 4000},
]

# Writes the log and a metadata sidecar; returns nothing.
rheon.generate_log(drifts, "out/example.xes", num_traces=2000, num_activities=8)

# Pass format="csv" to save the log as CSV instead of XES.
rheon.generate_log(drifts, "out/example.csv", format="csv", num_traces=2000, num_activities=8)
```

`generate_log` does not return anything — it saves the files. To work with the log, read it back:

```python
import pm4py
log = pm4py.read_xes("out/example.xes")     # XES → DataFrame

import pandas as pd
df = pd.read_csv("out/example.csv")          # CSV → DataFrame
```

## Drift types

Every drift is one dict with a `type`, a `mode` (`"sudden"` or `"gradual"`), and a position:
`drift_point` for sudden drifts, or `start_point`/`end_point` for gradual ones. All positions are
fractions of the time horizon in `(0, 1)`. The remaining keys are type-specific.

| Perspective | Type | What changes | Type-specific params |
| --- | --- | --- | --- |
| intra-case | `control_flow` | activity-ordering structure: cases after the drift are played out from a different process tree | `num_activities`, `tree_weights` |
| resource | `pool_size` | resources are added or removed; durations scale the opposite way | `delta`, `duration_factor` |
| resource | `reassignment` | a new dominant resource is chosen for every activity | — |
| resource | `workload` | traces are duplicated (or dropped); per-resource case load shifts | `workload_factor` |
| resource | `duration` | the processing time of the given resources is scaled | `resources`, `factor` |
| inter-case | `waiting_time` | the mean waiting gap between events shifts | `mean`, `variance` |
| inter-case | `amount` | case amounts are drawn from a shifted distribution | `mean`, `variance` |
| inter-case | `arrival_rate` | the mean gap between case arrivals changes | `inter_arrival` or `factor` |
| inter-case | `region` | a new dominant region is chosen for later cases | — |

Rheon validates the drift list before generating and raises a clear `ValueError` for problems such
as an out-of-range position, a gradual window with `start_point >= end_point`, or two drifts of the
same type whose transition windows overlap.

## Parameters

`generate_log(drifts, output_path, *, log_name=None, format="xes", **params)` accepts these
generation parameters (all optional, sensible defaults shown). `format` is `"xes"` or `"csv"` and
also drives the output file's suffix.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_traces` | `1000` | target number of cases (the `arrival_rate` and `workload` drifts can change the final count) |
| `num_activities` | `10` | activities in the base process tree |
| `num_resources` | `8` | size of the resource pool |
| `num_regions` | `4` | number of regions |
| `tree_weights` | `{sequence:.6, choice:.25, parallel:.1, loop:.05}` | operator weights for the base tree |
| `start_date` | `2020-01-01` | start of the time horizon |
| `end_date` | `2020-12-31` | end of the time horizon (with `start_date` it fixes the window) |
| `activity_duration` | `(30.0, 100.0)` | `(mean, variance)` of activity processing time in minutes |
| `waiting_time` | `(15.0, 50.0)` | `(mean, variance)` of the waiting gap between events |
| `amount` | `(1000.0, 40000.0)` | `(mean, variance)` of the case amount |
| `seed` | `42` | random seed |

Cases are spread across `[start_date, end_date]`. The mean inter-arrival gap is **derived** as
`(end_date − start_date) / num_traces` — there is no `inter_arrival` parameter. The `arrival_rate`
drift changes that derived rate after its drift point, so the realised case count then drifts away
from `num_traces`.

## Output files and metadata

Each run writes two files next to `output_path`:

- **`<name>.xes`** (or **`<name>.csv`**) — the event log. Events carry `concept:name`,
  `start_timestamp`, `time:timestamp`, `event:duration_min` and `org:resource`; each case carries
  `amount` and `region`. For XES the full metadata is also embedded as a log-level `rheon:metadata`
  attribute.
- **`<name>_meta.md`** — a short, readable ground-truth report with three sections:
  1. **General parameters** — the structural, temporal and attribute parameters of the run.
  2. **Base distributions** — the starting state: every activity's dominant resource and its duration
     and waiting distributions `(mean, var)`, plus the case-level amount distribution, inter-arrival
     mean and dominant region.
  3. **Drifts** — each drift with its mode and drift point / window (as both a horizon fraction and an
     absolute timestamp), followed by exactly which distributions or assignments it changed, per
     activity / resource (e.g. a reassignment's before → after resource table, or an amount drift's
     `mean: 1000 → 4000`).

## Example

A single ready-to-run script generates one mixed-drift log:

```bash
uv run python example/generate_log.py
```

## Development

```bash
git clone <repo-url> rheon
cd rheon
uv sync
uv run pytest
```

## License & status

Rheon is research software for generating drift benchmarks. See the repository for license details.
