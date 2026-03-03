# qgate: Time-Symmetric Trajectory Filtering for NISQ Hardware

[![CI](https://github.com/ranbuch/qgate-trajectory-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/ranbuch/qgate-trajectory-filter/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qgate.svg)](https://pypi.org/project/qgate/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: Source Available](https://img.shields.io/badge/license-Source%20Available%20v1.2-blue.svg)](LICENSE)

**Status:** Patent Pending — U.S. App. Nos. 63/983,831 & 63/989,632 | Israel App. No. 326915

qgate is a hardware-agnostic middleware that uses Score Fusion and Galton-based dynamic thresholding to rescue deep quantum circuits from combinatorial noise collapse, achieving "QEC-like" error suppression on existing NISQ hardware.

### Empirical Hardware Validation

**Evading the QEC Wall** (IBM Marrakesh, 156-qubit): At $N=8, D=8$, standard global conditioning yields a 0.0% acceptance rate (complete signal collapse). qgate's Score Fusion recovers a statistically robust 6.64% high-fidelity survival rate.

**Adaptive Fidelity Control** (IBM Fez, 156-qubit): While standard moving averages (rolling_z) inadvertently accept $>39\%$ of noisy trajectories, qgate's galton_quantile algorithm mathematically isolates the pristine statistical tail (adapting from an 8.6% acceptance at $N=2$ to 13.3% at $N=8$) to optimize both Time-to-Solution (TTS) and expectation fidelity.

### TSVF Algorithm Experiments (IBM Fez / IBM Torino, Feb–Mar 2026)

| Algorithm | Backend | Key Result | Advantage |
|---|---|---|---|
| **Grover Search** | IBM Fez | 7.3× higher success at iteration 4 | ✅ TSVF filters noise-corrupted amplitude paths |
| **QAOA MaxCut** | IBM Torino | 1.88× approximation ratio at p=1 | ✅ TSVF rescues shallow variational circuits |
| **VQE (TFIM)** | IBM Fez | Barren plateau avoidance at L=3 | ✅ TSVF selects low-energy trajectories |
| **QPE (φ=1/3)** | IBM Fez | Standard QPE retains phase; TSVF disrupts | ❌ Phase coherence incompatible with perturbation |
| **Utility-Scale (133Q)** | IBM Torino | Cooling delta Δ = −0.080 at 16,709 ISA depth | ✅ Galton filter extracts signal from 37× T₁ decoherence |

> **Key insight:** TSVF post-selection delivers significant advantage for
> amplitude-based algorithms (Grover, QAOA, VQE) but is incompatible with
> phase-coherence algorithms (QPE), where even mild perturbation destroys
> the inverse QFT interference pattern.

### Utility-Scale Stress Testing: "Works Clean, Scales Dirty"

To validate `qgate` across both theoretical and physical extremes, we ran a two-phase TFIM benchmark at the quantum critical point ($h/J \approx 3.04$):

| Scale | Environment | ISA Depth | Noise Regime | qgate Advantage | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16-Qubit** | Aer Simulator (Clean) | 2,290 | Simulated | **+0.7% Improvement** | ✅ **Works Clean:** Validates the underlying mathematical model |
| **133-Qubit** | IBM Torino (Physical) | 16,709 | Extreme ($37\times T_1$) | **Δ = −0.0798** | ✅ **Scales Dirty:** Galton filter survives massive hardware decoherence |

> **The 133-Qubit Torino Result:** At an ISA depth of 16,709 gates, standard
> global parity checks collapse entirely, and unmitigated circuit output is
> dominated by thermal decoherence. Despite this, the `qgate` Galton threshold
> dynamically isolated a pristine 11.95% statistical tail. By filtering out
> the hottest trajectories, `qgate` achieved a negative energy cooling delta
> (Δ = −0.0798) entirely at the classical post-processing layer, with zero
> variational optimization overhead.

---

## The Problem: Noise Kills Quantum Advantage

Every NISQ circuit runs inside a storm of decoherence, crosstalk, and gate
errors. As system size $N$ grows, the probability that **all** subsystems
remain coherent collapses exponentially — a hard wall that blocks any
global post-selection scheme from scaling.

**qgate breaks through that wall.**

Instead of demanding perfection from every qubit, qgate applies
*trajectory-level* statistical filtering: Score Fusion weighs multi-rate
parity signals, and a Galton-board quantile algorithm dynamically adapts
the acceptance threshold to the noise floor — all at the classical
post-processing layer, with **zero quantum overhead**.

---

## Hardware Results at a Glance

### Acceptance probability vs. system size (IBM Marrakesh, 10K shots per config)

<p align="center">
  <img src="simulations/ibm_hardware/acceptance_vs_N.png" width="720" alt="Acceptance probability vs N — Score Fusion maintains signal where Global collapses to 0%"/>
</p>

> **Key insight:** Global conditioning (red) collapses to 0 % acceptance at
> $N \geq 2$. Score Fusion (blue) sustains a robust acceptance rate across
> all tested system sizes — rescuing circuits that would otherwise be
> completely discarded.

### Time-to-Solution vs. system size (IBM Marrakesh, 10K shots per config)

<p align="center">
  <img src="simulations/ibm_hardware/tts_vs_N.png" width="720" alt="Time-to-Solution vs N — Score Fusion delivers orders-of-magnitude TTS improvement"/>
</p>

> **Key insight:** By maintaining acceptance probability, Score Fusion keeps
> TTS bounded while global conditioning's TTS diverges to infinity (no
> accepted shots = no solution).

---

## Three Conditioning Strategies

| Strategy | How It Works | Scaling Behaviour |
|---|---|---|
| **Global** | All $N$ subsystems × all $D$ cycles must be error-free | Exponential collapse at $N \geq 2$ |
| **Hierarchical k-of-N** | ≥ $\lceil k \cdot N \rceil$ subsystems pass each cycle | **O(1) — tested to N = 64** |
| **Score Fusion** | α-weighted blend of low-freq + high-freq parity scores ≥ θ | **Best on real hardware** |

### Scaling table (QuTiP simulation, 405K+ trials)

| N | Global | Hierarchical (k=0.9) | Score Fusion |
|---|---|---|---|
| 1 | 100 % | 100 % | 100 % |
| 2 | 0 % | **100 %** | **100 %** |
| 4 | 0 % | **100 %** | **100 %** |
| 8 | 0 % | **100 %** | **100 %** |
| 16 | 0 % | **100 %** | **100 %** |
| 32 | 0 % | **100 %** | **100 %** |
| 64 | 0 % | **100 %** | **100 %** |

---

## Quick Start

> **Requires Python 3.9+** — tested with Qiskit 2.x and qiskit-ibm-runtime 0.41+

```bash
pip install qgate
```

```python
from qgate import TrajectoryFilter, GateConfig, DynamicThresholdConfig

# ── Configure: Score Fusion + Galton adaptive thresholding ──────────
config = GateConfig(
    n_subsystems=8,               # 8 Bell-pair subsystems (the "QEC wall")
    n_cycles=8,                   # 8 monitoring cycles per shot
    shots=10_000,                 # 10K trajectories
    variant="score_fusion",       # α-weighted LF + HF parity blend
    fusion={"alpha": 0.7},        # 70% low-freq, 30% high-freq weight
    dynamic_threshold=DynamicThresholdConfig(
        mode="galton",            # Galton-board quantile gating
        target_acceptance=0.10,   # isolate the top-10% statistical tail
        window_size=500,          # rolling window of per-shot scores
        use_quantile=True,        # empirical quantile (no distributional assumptions)
    ),
)

# ── Run: build circuit → execute → score → Galton-filter ───────────
tf = TrajectoryFilter(config, adapter="mock")   # swap "mock" → "qiskit" for IBM hardware
result = tf.run()

print(f"Acceptance rate : {result.acceptance_probability:.2%}")
print(f"Galton threshold: {result.metadata.get('galton_threshold', 'N/A')}")
print(f"Shots accepted  : {result.n_accepted} / {config.shots}")
```

> **What just happened?** Score Fusion blended multi-rate parity signals into
> a single fidelity score per trajectory. The Galton quantile algorithm then
> adapted the acceptance threshold to the *actual* noise floor — isolating
> only the pristine statistical tail. No hand-tuned thresholds, no
> distributional assumptions, zero quantum overhead.

Optional extras:
```bash
pip install qgate[qiskit]     # IBM Quantum hardware support
pip install qgate[qutip]      # QuTiP simulation support
pip install qgate[all]         # Everything
```

---

## Experimental Validation

All claims are backed by reproducible experiments on **real IBM Quantum hardware**:

### Core Conditioning Experiments

| Experiment | Backend | Qubits | Shots | Key Finding |
|---|---|---|---|---|
| [Score Fusion sweep](simulations/ibm_hardware/README.md) | IBM Marrakesh | 156 | 120K | Score Fusion best across all N, D |
| [Galton threshold comparison](simulations/ibm_hardware/galton_experiment/) | IBM Fez | 156 | 120K | Galton quantile isolates high-fidelity tail |
| [QuTiP high-noise sweep](simulations/qutip_sims/runs/high_noise_20260126_233522/README.md) | Simulation | — | 405K | Global collapses at N ≥ 2 |
| [N=64 scaling test](simulations/qutip_sims/runs/followup_20260127_133350/incremental_20260128_084503/README.md) | Simulation | — | 222 configs | Hierarchical holds at N=64 |

### TSVF Algorithm Experiments (IBM Hardware, Feb–Mar 2026)

Detailed telemetry, circuit depths, raw histograms, and full reproduction
steps for each algorithm are in the respective directories below:

| Experiment | Backend | Circuit Depth | Shots | Key Finding |
|---|---|---|---|---|
| [Grover vs TSVF-Grover](simulations/grover_tsvf/README.md) | IBM Fez | 42–200 | 8,192 | **7.3× success probability** at iteration 4 |
| [QAOA vs TSVF-QAOA](simulations/qaoa_tsvf/README.md) | IBM Torino | 26–149 | 2,000 | **1.88× approximation ratio** at p=1 |
| [VQE vs TSVF-VQE](simulations/vqe_tsvf/README.md) | IBM Fez | 33–192 | 4,000 | **Barren plateau avoidance** at L=3 (1.107 energy gap) |
| [QPE vs TSVF-QPE](simulations/qpe_tsvf/README.md) | IBM Fez | 97–567 | 8,192 | Standard QPE retains phase; perturbation incompatible with QPE |
| [Utility-Scale TFIM](simulations/tfim_127q/README.md) | IBM Torino | 97–16,709 | 100,000 | **Cooling Δ = −0.080** at 133Q, 37× T₁ decoherence |

---

## Repository Structure

```
qgate-trajectory-filter/
├── packages/qgate/              # 📦 pip-installable Python package
│   ├── src/qgate/               #    Core conditioning & monitoring
│   │   └── adapters/            #    Mock, Qiskit, Grover, QAOA, VQE, QPE adapters
│   ├── tests/                   #    376 unit tests (pytest)
│   └── pyproject.toml           #    Build config (hatchling)
│
├── simulations/
│   ├── ibm_hardware/            # 🔬 IBM Quantum conditioning experiments
│   │   ├── results.csv          #    120 rows (Marrakesh)
│   │   ├── acceptance_vs_N.png  #    ← Figure above
│   │   ├── tts_vs_N.png         #    ← Figure above
│   │   └── galton_experiment/   #    Galton threshold (Fez)
│   ├── grover_tsvf/             # 🔍 Grover vs TSVF-Grover (IBM Fez)
│   ├── qaoa_tsvf/               # 🔗 QAOA vs TSVF-QAOA MaxCut (IBM Torino)
│   ├── vqe_tsvf/                # ⚛️  VQE vs TSVF-VQE TFIM (IBM Fez)
│   ├── qpe_tsvf/                # 📐 QPE vs TSVF-QPE Phase Est. (IBM Fez)
│   ├── tfim_127q/               # 🔬 133-Qubit Utility-Scale TFIM (IBM Torino)
│   └── qutip_sims/              # 🧪 QuTiP master-equation sims
│       └── runs/                #    576+ configs, per-run READMEs
│
├── examples/                    # 💡 Usage examples
├── docs/architecture.md         # 📖 Full methodology & patent mapping
└── src/sim.py                   # Core QuTiP simulation engine
```

---

## Documentation

| Resource | Description |
|---|---|
| [Architecture & Methodology](docs/architecture.md) | System design, conditioning strategies, TSVF experiments, validation chain |
| [Package API Reference](packages/qgate/README.md) | Full `qgate` API docs, install guide, class/function reference |
| [Grover vs TSVF-Grover](simulations/grover_tsvf/README.md) | IBM Fez — 7.3× advantage at iteration 4 |
| [QAOA vs TSVF-QAOA](simulations/qaoa_tsvf/README.md) | IBM Torino — 1.88× advantage at p=1 |
| [VQE vs TSVF-VQE](simulations/vqe_tsvf/README.md) | IBM Fez — barren plateau avoidance at L=3 |
| [QPE vs TSVF-QPE](simulations/qpe_tsvf/README.md) | IBM Fez — phase coherence study |
| [Utility-Scale TFIM (133Q)](simulations/tfim_127q/README.md) | IBM Torino — cooling Δ = −0.080 at 37× T₁ |
| Per-run READMEs | Each `simulations/` directory has objectives, parameters, results, reproduction steps |

---

## For Contributors / Local Development

```bash
git clone https://github.com/ranbuch/qgate-trajectory-filter.git
cd qgate-trajectory-filter
pip install -e "packages/qgate[dev]"
pytest packages/qgate/tests/ -v        # 376 tests
ruff check packages/qgate/src/         # lint
mypy packages/qgate/src/               # type-check
```

---

## License & IP

**QGATE Source Available Evaluation License v1.2** — see [`LICENSE`](packages/qgate/LICENSE).

Academic research, internal corporate evaluation (including on IBM Quantum / AWS Braket),
and peer review are freely permitted. Commercial deployment requires a separate license.

**For commercial licensing, hardware integration rights, or IP acquisition
inquiries:** [ranbuch@gmail.com](mailto:ranbuch@gmail.com)
