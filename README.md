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

## Example and tests

```bash
uv run python example/generate_log.py   # writes example/example.csv and example/example_meta.md
uv run pytest
```
