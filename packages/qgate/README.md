# qgate — Quantum Trajectory Filter

> Runtime post-selection conditioning for quantum circuits.

[![CI](https://github.com/ranbuch/qgate-trajectory-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/ranbuch/qgate-trajectory-filter/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/qgate.svg)](https://pypi.org/project/qgate/)
[![License: Source Available](https://img.shields.io/badge/license-Source%20Available%20v1.2-blue.svg)](https://github.com/ranbuch/qgate-trajectory-filter/blob/main/packages/qgate/LICENSE)

This package explores runtime trajectory filtering concepts from
**US Patent Application Nos. 63/983,831 & 63/989,632** and
**Israeli Patent Application No. 326915**.
The underlying invention is patent pending.

---

## Installation

```bash
pip install qgate                # core (numpy, pydantic, typer)
pip install qgate[csv]           # + pandas for CSV logging
pip install qgate[parquet]       # + pandas + pyarrow for Parquet logging
pip install qgate[qiskit]        # + IBM Qiskit adapter
pip install qgate[cirq]          # + Google Cirq adapter (stub)
pip install qgate[pennylane]     # + PennyLane adapter (stub)
pip install qgate[all]           # everything
```

> **Note:** pandas is *not* required for core filtering; it is only needed
> if you write logs to CSV or Parquet.  The default JSONL logger uses only
> the standard library.

For development:

```bash
git clone https://github.com/ranbuch/qgate-trajectory-filter.git
cd qgate-trajectory-filter/packages/qgate
pip install -e ".[dev]"
```

---

## Quick Start

### New API — TrajectoryFilter

```python
from qgate import TrajectoryFilter, GateConfig
from qgate.adapters import MockAdapter

config = GateConfig(
    n_subsystems=4,
    n_cycles=2,
    shots=1024,
    variant="score_fusion",
)
adapter = MockAdapter(error_rate=0.05, seed=42)
tf = TrajectoryFilter(config, adapter)
result = tf.run()

print(f"Accepted: {result.accepted_shots}/{result.total_shots}")
print(f"P_accept: {result.acceptance_probability:.4f}")
print(f"TTS:      {result.tts:.2f}")
```

### Dynamic Thresholding

```python
from qgate import GateConfig, DynamicThresholdConfig, TrajectoryFilter
from qgate.adapters import MockAdapter

config = GateConfig(
    shots=200,
    dynamic_threshold=DynamicThresholdConfig(
        enabled=True, baseline=0.65, z_factor=1.0,
    ),
)
tf = TrajectoryFilter(config, MockAdapter(seed=42))

for _ in range(10):
    result = tf.run()
    print(f"threshold={tf.current_threshold:.4f}  P_acc={result.acceptance_probability:.4f}")
```

### Galton Adaptive Thresholding

Distribution-aware gating that maintains a stable acceptance fraction
by adapting to the empirical score distribution.  Supports quantile and
robust z-score sub-modes.

```python
from qgate import GateConfig, DynamicThresholdConfig, TrajectoryFilter
from qgate.adapters import MockAdapter

config = GateConfig(
    shots=1000,
    variant="score_fusion",
    dynamic_threshold=DynamicThresholdConfig(
        mode="galton",            # distribution-aware adaptive gating
        window_size=1000,         # per-shot rolling window
        min_window_size=100,      # warmup period
        target_acceptance=0.05,   # target ~5% acceptance
        robust_stats=True,        # MAD-based sigma (outlier-resilient)
        use_quantile=True,        # empirical quantile (recommended)
    ),
)
tf = TrajectoryFilter(config, MockAdapter(error_rate=0.1, seed=42))
result = tf.run()

print(f"Threshold: {tf.current_threshold:.4f}")
print(f"Accepted:  {result.accepted_shots}/{result.total_shots}")
# Galton telemetry in result.metadata["galton"]
```

### CLI

```bash
qgate version
qgate validate config.json
qgate run config.json --adapter mock --seed 42
qgate run config.json --adapter mock --output results.jsonl
qgate run config.json --adapter mock --error-rate 0.1 --verbose
qgate run config.json --adapter mock --quiet          # suppress info logs
qgate adapters                                         # list installed adapters
qgate schema                                           # print JSON Schema for GateConfig
```

### Legacy API (backward compatible)

```python
from qgate import decide_hierarchical, MultiRateMonitor
from qgate.conditioning import ParityOutcome

outcome = ParityOutcome(4, 2, [[0, 0, 1, 0], [0, 0, 0, 0]])
assert decide_hierarchical(outcome, k_fraction=0.75) is True
```

---

## Architecture

```
qgate/
├── __init__.py        # Public API re-exports (backward-compatible)
├── config.py          # Pydantic v2 config models (GateConfig, FusionConfig, …)
│                        All models are frozen & extra="forbid"
├── filter.py          # TrajectoryFilter — main API class
│                        Vectorised scoring via score_batch()
├── scoring.py         # Fuse-scores, score_outcome, score_batch (NumPy)
├── threshold.py       # Dynamic threshold: rolling z-score + Galton adaptive
├── run_logging.py     # JSON-Lines / CSV / Parquet structured logging
│                        RunLogger is a context manager; Parquet is buffered
├── cli.py             # Typer CLI (run, validate, schema, adapters, version)
├── compat/            # Backward-compatible wrappers
│   ├── conditioning.py    # ParityOutcome (ndarray), decision rules
│   └── monitors.py        # MultiRateMonitor, should_abort_batch
├── conditioning.py    # Re-export from compat/conditioning
├── monitors.py        # Re-export from compat/monitors
└── adapters/
    ├── base.py            # BaseAdapter ABC + MockAdapter
    ├── registry.py        # list_adapters() / load_adapter()
    ├── qiskit_adapter.py  # Full Qiskit implementation (copy-safe)
    ├── grover_adapter.py  # Grover vs TSVF-Chaotic Grover adapter
    ├── qaoa_adapter.py    # QAOA vs TSVF-QAOA MaxCut adapter
    ├── vqe_adapter.py     # VQE vs TSVF-VQE (TFIM) adapter
    ├── qpe_adapter.py     # QPE vs TSVF-QPE Phase Estimation adapter
    ├── cirq_adapter.py    # Stub
    └── pennylane_adapter.py  # Stub
```

### Key Design Decisions

- **ParityOutcome stores an `np.ndarray`** — shape `(n_cycles, n_subsystems)`,
  dtype `int8`.  Lists are coerced on construction.
- **All configuration is immutable** — `GateConfig` and sub-models are
  Pydantic `frozen` models.  Create a new config to change parameters.
- **pandas is optional** — only imported lazily when CSV/Parquet logging
  is used.  Core filtering needs only numpy + pydantic.
- **Structured logging** — all modules use `logging.getLogger("qgate.*")`
  so users can control verbosity with standard Python logging.

---

## Conditioning Strategies

| Strategy | Config | Scaling |
|---|---|---|
| **Global** | `variant="global"` | Exponential decay at N ≥ 2 |
| **Hierarchical k-of-N** | `variant="hierarchical", k_fraction=0.9` | O(1) — scales to N = 64+ |
| **Score Fusion** | `variant="score_fusion"` | Most robust on real hardware |

---

## Algorithm-Specific TSVF Adapters

qgate ships with four algorithm-specific adapters that apply trajectory
filtering to canonical quantum algorithms. Each adapter builds both a
standard circuit and a TSVF variant (with chaotic perturbation + ancilla
phase/parity probe), runs them, and extracts algorithm-specific metrics.

| Adapter | Algorithm | Entry Point | IBM Validated |
|---|---|---|---|
| `GroverTSVFAdapter` | Grover search | `grover_tsvf` | ✅ IBM Fez — **7.3× advantage** |
| `QAOATSVFAdapter` | QAOA MaxCut | `qaoa_tsvf` | ✅ IBM Torino — **1.88× advantage** |
| `VQETSVFAdapter` | VQE for TFIM | `vqe_tsvf` | ✅ IBM Fez — **barren plateau avoidance** |
| `QPETSVFAdapter` | QPE phase est. | `qpe_tsvf` | ✅ IBM Fez — phase coherence study |

### Utility-Scale Validation (IBM Torino, 133 Qubits)

The `VQETSVFAdapter` has been stress-tested at utility scale on IBM Torino
(133 physical qubits, 16,709 ISA gate depth). At 37× T₁ decoherence, the
Galton filter achieved a negative cooling delta (**Δ = −0.080**), extracting
correlated thermodynamic signal from ~99% thermal noise. See the
[simulations/tfim_127q](../../simulations/tfim_127q/) directory for full
reproduction steps and raw data.

```python
from qgate.adapters.grover_adapter import GroverTSVFAdapter
from qgate.adapters.qaoa_adapter import QAOATSVFAdapter
from qgate.adapters.vqe_adapter import VQETSVFAdapter
from qgate.adapters.qpe_adapter import QPETSVFAdapter
```

---

## Run Logging

```python
from qgate import TrajectoryFilter, GateConfig
from qgate.adapters import MockAdapter
from qgate.run_logging import RunLogger

config = GateConfig(shots=500, variant="score_fusion")
tf = TrajectoryFilter(config, MockAdapter(seed=42))

with RunLogger("results.jsonl") as logger:
    result = tf.run()
    logger.log(result)
    # For CSV: RunLogger("results.csv")   — requires pip install qgate[csv]
    # For Parquet: RunLogger("results.parquet") — requires pip install qgate[parquet]
```

Each logged record includes a deterministic **run ID** (SHA-256 prefix),
full config JSON, acceptance probability, TTS, and a UTC timestamp.

---

## Running Tests

```bash
cd packages/qgate
pip install -e ".[dev]"
pytest -v tests/            # 376 tests, ~3 s
pytest --cov=qgate tests/   # with coverage
```

---

## License

QGATE Source Available Evaluation License v1.2 — see [LICENSE](https://github.com/ranbuch/qgate-trajectory-filter/blob/main/packages/qgate/LICENSE).

Academic research, peer review, and internal corporate evaluation are freely permitted.
Commercial deployment requires a separate license.
