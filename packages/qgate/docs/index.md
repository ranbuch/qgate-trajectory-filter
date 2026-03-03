---
description: >-
  qgate is a hardware-agnostic quantum error suppression middleware for NISQ devices.
  Runtime trajectory filtering via Bell-pair post-selection, score fusion, and Galton
  adaptive thresholding. Validated on IBM Quantum hardware with up to 7.3× fidelity improvement.
keywords: quantum error mitigation, NISQ, qiskit, trajectory filter, post-selection, Bell pair, score fusion, Galton thresholding, quantum computing, qgate, IBM Quantum
faq:
  - q: What is qgate?
    a: qgate is a Python middleware library for quantum error suppression on NISQ devices. It filters quantum computation trajectories using Bell-pair post-selection conditioning, score fusion, and adaptive thresholding.
  - q: What quantum hardware does qgate support?
    a: qgate supports IBM Quantum hardware via Qiskit, with adapter stubs for Google Cirq and Xanadu PennyLane. It has been validated on IBM Fez (156 qubits) and IBM Torino (133 qubits).
  - q: How much does qgate improve quantum circuit fidelity?
    a: In IBM hardware experiments, qgate achieved up to 7.3× fidelity improvement for Grover search (IBM Fez), 1.88× for QAOA MaxCut (IBM Torino), and barren plateau avoidance for VQE (IBM Fez).
  - q: How do I install qgate?
    a: Install from PyPI with pip install qgate. For IBM Quantum support use pip install qgate[qiskit]. Requires Python 3.9 or higher.
  - q: Is qgate open source?
    a: qgate is released under the QGATE Source Available Evaluation License v1.2, which permits academic research and internal corporate evaluation. Commercial deployment requires a separate license. Patent pending (US 63/983,831, US 63/989,632, IL 326915).
---

# qgate — Quantum Trajectory Filter

[![PyPI](https://img.shields.io/pypi/v/qgate)](https://pypi.org/project/qgate/)
[![Python](https://img.shields.io/pypi/pyversions/qgate)](https://pypi.org/project/qgate/)
[![License](https://img.shields.io/badge/license-QGATE%20SAE%20v1.2-blue)](license.md)
[![Tests](https://img.shields.io/badge/tests-376%20passing-brightgreen)]()

**Hardware-agnostic quantum error suppression middleware for NISQ devices.**

qgate provides a framework-agnostic Python toolkit for filtering quantum
computation trajectories based on mid-circuit measurement outcomes. It
implements multiple conditioning strategies (global, hierarchical k-of-N,
and continuous score fusion) with dynamic threshold adaptation — including
the novel **Galton adaptive thresholding** method.

---

## :material-trophy: IBM Hardware Results

Validated on real IBM Quantum processors with up to **7.3× fidelity improvement**:

| Algorithm | Backend | Metric | Standard | TSVF | Advantage |
|---|---|---|---|---|---|
| **Grover** (iter=4) | IBM Fez | Success probability | 0.0830 | **0.6105** | :material-fire: **7.3×** |
| **QAOA** (p=1) | IBM Torino | Approximation ratio | 0.4268 | **0.8029** | :material-fire: **1.88×** |
| **VQE** (L=3) | IBM Fez | Energy gap to ground | 2.398 | **1.291** | :material-fire: **1.86×** closer |
| **QPE** (t=7) | IBM Fez | Phase fidelity | **0.1569** | 0.0076 | :material-close: N/A |

!!! info "Why QPE doesn't benefit"
    TSVF works for **amplitude-encoded** algorithms (Grover, QAOA, VQE) but not
    for **phase-coherence-encoded** algorithms (QPE). See [Hardware Experiments](experiments/index.md)
    for full analysis.

---

## Key Features

- :dart: **TrajectoryFilter** — Main API class that orchestrates build → execute → filter
- :electric_plug: **Adapter pattern** — Pluggable backends (Qiskit, Cirq, PennyLane) + algorithm-specific TSVF adapters
- :bar_chart: **Score fusion** — α-weighted LF/HF multi-rate fusion scoring
- :chart_with_upwards_trend: **Dynamic thresholding** — Rolling z-score gating that adapts to hardware drift
- :game_die: **Galton adaptive thresholding** — Distribution-aware gating with empirical quantile or robust z-score sub-modes
- :zap: **CLI** — `qgate run`, `qgate validate`, `--verbose`/`--quiet`/`--error-rate`
- :memo: **Structured logging** — JSONL (zero deps), CSV, Parquet output with `RunLogger` context manager
- :lock: **Immutable config** — All Pydantic models are `frozen=True`
- :rocket: **Vectorised internals** — NumPy-backed `ParityOutcome` and batch scoring
- :evergreen_tree: **stdlib logging** — All modules use `logging.getLogger("qgate.*")`

## Quick Start

```python
from qgate import TrajectoryFilter, GateConfig
from qgate.adapters import MockAdapter

config = GateConfig(n_subsystems=4, n_cycles=2, shots=1024)
adapter = MockAdapter(error_rate=0.05, seed=42)
tf = TrajectoryFilter(config, adapter)
result = tf.run()
print(f"Accepted: {result.accepted_shots}/{result.total_shots}")
```

## What's New in v0.5.0

- :material-atom: **Algorithm TSVF adapters** — Grover, QAOA, VQE, QPE adapters with
  Two-State Vector Formalism trajectory filtering
- :material-chart-bell-curve: **Galton adaptive thresholding** — Distribution-aware gating with
  empirical quantile and robust z-score sub-modes
- :material-chip: **IBM hardware validation** — Experiments on IBM Fez & Torino with
  up to 7.3× fidelity improvement
- :material-test-tube: **376 tests** passing across Python 3.9–3.13
- :material-speedometer: **NumPy vectorisation** — Batch operations for scoring and filtering
- :material-lock-check: **Frozen config** — All Pydantic models are immutable

See the full [Changelog](https://github.com/ranbuch/qgate-trajectory-filter/blob/main/packages/qgate/CHANGELOG.md).

## Patent Notice

!!! warning "Patent Pending"
    This package explores runtime trajectory filtering concepts from
    US Patent Application Nos. 63/983,831 & 63/989,632 and Israeli Patent
    Application No. 326915. The underlying invention is patent pending.
    See [License](license.md) for details.
