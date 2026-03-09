---
description: >-
  qgate is a hardware-agnostic quantum error suppression middleware for NISQ devices.
  Runtime trajectory filtering via Bell-pair post-selection, score fusion, and Galton
  adaptive thresholding. Validated on IBM Quantum hardware with up to 7.3× fidelity
  improvement. Systematic bias study shows up to 20.7% MSE reduction, 5,360× variance
  collapse, algorithm-agnostic improvement across VQE, QAOA, and Grover, and 14.7% MSE
  reduction on a blind test set with a frozen threshold proving generalisation.
keywords: quantum error mitigation, NISQ, qiskit, trajectory filter, post-selection, Bell pair, score fusion, Galton thresholding, quantum computing, qgate, IBM Quantum, bias study, MSE reduction, variance collapse, noise robustness, qubit scaling, VQE, QAOA, Grover, train test split, frozen threshold, generalisation, calibration
faq:
  - q: What is qgate?
    a: qgate is a Python middleware library for quantum error suppression on NISQ devices. It filters quantum computation trajectories using Bell-pair post-selection conditioning, score fusion, and adaptive thresholding.
  - q: What quantum hardware does qgate support?
    a: qgate supports IBM Quantum hardware via Qiskit, with adapter stubs for Google Cirq and Xanadu PennyLane. It has been validated on IBM Fez (156 qubits) and IBM Torino (133 qubits).
  - q: How much does qgate improve quantum circuit fidelity?
    a: In IBM hardware experiments, qgate achieved up to 7.3× fidelity improvement for Grover search (IBM Fez), 1.88× for QAOA MaxCut (IBM Torino), and barren plateau avoidance for VQE (IBM Fez). In systematic bias studies, it reduced MSE by up to 20.7% and variance by up to 5,360×.
  - q: How do I install qgate?
    a: Install from PyPI with pip install qgate. For IBM Quantum support use pip install qgate[qiskit]. Requires Python 3.9 or higher. Use QgateSampler for instant filtered results with zero circuit changes.
  - q: Is qgate open source?
    a: qgate is released under the QGATE Source Available Evaluation License v1.2, which permits academic research and internal corporate evaluation. Commercial deployment requires a separate license. Patent pending (US 63/983,831, US 63/989,632, IL 326915).
---

# qgate — Quantum Trajectory Filter

[![PyPI](https://img.shields.io/pypi/v/qgate)](https://pypi.org/project/qgate/)
[![Python](https://img.shields.io/pypi/pyversions/qgate)](https://pypi.org/project/qgate/)
[![License](https://img.shields.io/badge/license-QGATE%20SAE%20v1.2-blue)](license.md)
[![Tests](https://img.shields.io/badge/tests-406%20passing-brightgreen)]()

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
| **Utility-Scale** (133Q) | IBM Torino | Cooling delta | −4.108 | **−4.188** | :material-fire: **Δ = −0.080** |

!!! info "Why QPE doesn't benefit"
    TSVF works for **amplitude-encoded** algorithms (Grover, QAOA, VQE) but not
    for **phase-coherence-encoded** algorithms (QPE). See [Hardware Experiments](experiments/index.md)
    for full analysis.

!!! success "NEW: Utility-Scale Stress Test (IBM Torino, 133 Qubits)"
    At 16,709 ISA gate depth ($37\times T_1$), `qgate` Galton filtering achieved
    a negative cooling delta (**Δ = −0.080**), extracting correlated signal from
    ~99% thermal noise across 133 physical qubits — with zero variational
    optimization overhead. See [Utility-Scale Stress Test](experiments/utility-scale.md).

---

## :material-new-box: QgateSampler — Drop-in SamplerV2 Middleware

!!! tip "New in v0.6.0 — the fastest way to get these results"
    **One `import` swap. Zero circuit changes. Measurable physics improvement.**

    Wrap any IBM backend with `QgateSampler` and every call is automatically
    enhanced with probe injection, Galton-filtered post-selection, and clean
    result reconstruction.

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qgate import QgateSampler

service = QiskitRuntimeService()
backend = service.backend("ibm_fez")

# Wrap the backend — that's it
sampler = QgateSampler(backend=backend)

# Use exactly like SamplerV2
job = sampler.run([(qc,)])
result = job.result()
counts = result[0].data.meas.get_counts()  # higher-fidelity shots only
```

Validated on **IBM Fez** (95% Bell fidelity), **IBM Torino** (133Q utility-scale),
and **IBM Brisbane** (6.6% acceptance vs 0% raw post-selection).

:material-arrow-right: **[Full QgateSampler documentation →](middleware/qgate-sampler.md)**

---

## :material-chart-bell-curve: Statistical Validation (NEW — Mar 2026)

Systematic bias study: **15 independent trials × 100,000 shots**, IBM Heron-class noise model.

| Experiment | Key Finding | Significance |
|---|---|---|
| **Noise Robustness** | MSE reduction grows **13.6% → 20.7%** as noise increases | All $p < 10^{-23}$ |
| **Qubit Scaling** (8–16q) | Stable 14–17% MSE; **variance collapse up to 5,360×** | All $p < 10^{-46}$ |
| **Cross-Algorithm** | VQE **+14.8%**, QAOA **+48.8%**, Grover **+24.4%** | All $p < 10^{-17}$ |
| **Train/Test Split** | Frozen threshold generalises: **14.7% MSE↓** on blind test set | $p = 0.001$ \*\*\* |

!!! tip "The Anti-Decoherence Property"
    Unlike most error mitigation techniques that degrade under heavy noise,
    qgate **improves** with noise — the filter's MSE reduction scales from
    13.6% (ideal) to **20.7%** at the highest noise level tested.
    It thrives exactly where current NISQ hardware operates.

!!! success "NEW: Calibrate Once, Deploy Forever"
    Experiment 4 proves the Galton threshold is a **stable physical constant**.
    Enterprises can run a cheap calibration circuit to find θ, freeze it,
    and apply it to massive production runs — saving compute while retaining
    full **14.7% MSE reduction** on completely unseen data ($p = 0.001$).

:material-arrow-right: **[Full bias study results →](experiments/bias-study.md)**

---

## Key Features

- :material-swap-horizontal: **QgateSampler** — :material-new-box: Transparent SamplerV2 drop-in: wrap any backend, get filtered results. [Try it →](middleware/qgate-sampler.md)
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

=== "QgateSampler (recommended)"

    ```python
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qgate import QgateSampler

    service = QiskitRuntimeService()
    backend = service.backend("ibm_fez")

    sampler = QgateSampler(backend=backend)
    job = sampler.run([(qc,)])
    result = job.result()
    counts = result[0].data.meas.get_counts()
    ```

    :material-arrow-right: [Full QgateSampler docs →](middleware/qgate-sampler.md)

=== "TrajectoryFilter (advanced)"

    ```python
    from qgate import TrajectoryFilter, GateConfig
    from qgate.adapters import MockAdapter

    config = GateConfig(n_subsystems=4, n_cycles=2, shots=1024)
    adapter = MockAdapter(error_rate=0.05, seed=42)
    tf = TrajectoryFilter(config, adapter)
    result = tf.run()
    print(f"Accepted: {result.accepted_shots}/{result.total_shots}")
    ```

    :material-arrow-right: [Full Quick Start guide →](getting-started/quickstart.md)

## What's New in v0.6.0

- :material-swap-horizontal: **QgateSampler middleware** — Transparent SamplerV2 drop-in replacement with
  autonomous probe injection and Galton-filtered post-selection. One import swap,
  zero circuit changes. [Try it →](middleware/qgate-sampler.md)
- :material-atom: **Algorithm TSVF adapters** — Grover, QAOA, VQE, QPE adapters with
  Two-State Vector Formalism trajectory filtering
- :material-chart-bell-curve: **Galton adaptive thresholding** — Distribution-aware gating with
  empirical quantile and robust z-score sub-modes
- :material-chip: **IBM hardware validation** — Experiments on IBM Fez, Torino & Brisbane with
  up to 7.3× fidelity improvement, 95% Bell fidelity on filtered shots
- :material-test-tube: **406 tests** passing across Python 3.9–3.13
- :material-speedometer: **NumPy vectorisation** — Batch operations for scoring and filtering
- :material-lock-check: **Frozen config** — All Pydantic models are immutable

See the full [Changelog](https://github.com/ranbuch/qgate-trajectory-filter/blob/main/packages/qgate/CHANGELOG.md).

## Patent Notice

!!! warning "Patent Pending"
    This package explores runtime trajectory filtering concepts from
    US Patent Application Nos. 63/983,831 & 63/989,632 and Israeli Patent
    Application No. 326915. The underlying invention is patent pending.
    See [License](license.md) for details.
