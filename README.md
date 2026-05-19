# rheon

Synthetic multi-perspective concept drift log generation.

Rheon generates synthetic event logs with injected concept drifts across four
perspectives: control flow, resource, inter-case, and data. Each generated log
ships with an XES file and a Markdown sidecar describing the injected drifts.

## Install

```bash
pip install -e .
```

## Usage

```python
from rheon import GeneratorConfig, generate_log

config = GeneratorConfig(num_traces=1000, global_seed=7)
drifts = [
    {"subtype": "tree_mutation", "drift_type": "sudden", "change_proportion": 0.3},
]
generated = generate_log(config, drifts, log_name="example")
```

The `scripts/` directory contains end-to-end examples that produce XES logs for
each perspective.

## Tests

```bash
pytest
```
