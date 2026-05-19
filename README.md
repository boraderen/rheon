# Rheon

Synthetic multi-perspective concept drift log generation for process mining experiments.

Rheon generates XES event logs with known drifts across control-flow, resource, inter-case, and data perspectives.

Rheon is not published on PyPI yet. Install it from a local clone.

## Local Setup

```bash
git clone <repo-url> rheon
cd rheon
uv sync
```

Run a generator:

```bash
uv run python scripts/generate_control_flow_log.py
uv run python scripts/generate_resource_log.py
uv run python scripts/generate_inter_case_log.py
uv run python scripts/generate_data_log.py
```

## Use From Another Project

With `uv`:

```bash
cd /path/to/other-project
uv add --editable /path/to/rheon
```

Or add it manually:

```toml
[project]
dependencies = ["rheon"]

[tool.uv.sources]
rheon = { path = "../rheon", editable = true }
```

Then import it:

```python
import rheon

drifts = [{"perspective": "data", "subtype": "numeric", "drift_type": "sudden"}]

generated = rheon.generate_log(
    drifts,
    "data/example/data_001.xes",
    default_perspective="data",
    num_traces=100,
    global_seed=7,
)
```

The generator scripts use the same function with uppercase constants such as
`NUM_TRACES` and `GLOBAL_SEED`; direct imports can use lowercase keyword
arguments.

With `pip`:

```bash
python -m pip install -e /path/to/rheon
```
