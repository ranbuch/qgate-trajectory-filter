---
description: >-
  Quick start guide for qgate. Create a GateConfig, choose an adapter, run the
  trajectory filter, log results, and use the CLI in under 5 minutes.
keywords: qgate quickstart, trajectory filter tutorial, quantum error suppression tutorial, GateConfig, MockAdapter, Python quantum computing
---

# Quick Start

## 1. Create a configuration

All configuration is **immutable** — once created, a `GateConfig` cannot be
mutated.  Create a new instance to change parameters.

```python
from qgate import GateConfig

config = GateConfig(
    n_subsystems=4,
    n_cycles=2,
    shots=1024,
    variant="score_fusion",
)
```

## 2. Choose an adapter

```python
from qgate.adapters import MockAdapter

adapter = MockAdapter(error_rate=0.05, seed=42)
```

## 3. Run the trajectory filter

```python
from qgate import TrajectoryFilter

tf = TrajectoryFilter(config, adapter)
result = tf.run()

print(f"Accepted: {result.accepted_shots}/{result.total_shots}")
print(f"P_accept: {result.acceptance_probability:.4f}")
print(f"TTS:      {result.tts:.2f}")
```

## 4. Log results

Use `RunLogger` as a context manager — it flushes buffered records
(especially Parquet) automatically on exit.

```python
from qgate.run_logging import RunLogger

with RunLogger("results.jsonl") as logger:
    result = tf.run()
    logger.log(result)
```

Supported formats: `.jsonl` (default, no extra deps), `.csv` (requires
`qgate[csv]`), `.parquet` (requires `qgate[parquet]`).

## 5. Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Now all qgate.* loggers emit debug messages
```

## CLI

```bash
# Validate a config file
qgate validate config.json

# Run with mock adapter
qgate run config.json --adapter mock --seed 42

# Run with output logging and verbose mode
qgate run config.json --adapter mock --output results.jsonl --verbose

# Override error rate for quick experiments
qgate run config.json --adapter mock --error-rate 0.1 --quiet

# List installed adapters
qgate adapters
```
