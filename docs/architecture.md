# Architecture & Methodology

> **Patent notice:** US Patent Application Nos. 63/983,831 & 63/989,632 | Israeli Patent Application No. 326915. The underlying invention is patent pending.

---

## Overview

This repository implements and validates **quantum error suppression via
post-selection conditioning on Bell-pair subsystems**. The approach uses
mid-circuit measurements to monitor subsystem fidelity and applies decision
rules to accept or reject quantum computation outcomes — thereby suppressing
errors without full quantum error correction.

---

## Core Concept

```
 ┌─────────────────────────────────────────────────────────┐
 │                    Quantum Circuit                       │
 │                                                          │
 │  ┌──────────┐   ┌───────────┐   ┌─────────────┐        │
 │  │ Bell-pair │──▶│ Scramble  │──▶│ Mid-circuit │──┐     │
 │  │ prep ×N  │   │ layers ×D │   │ Z-parity    │  │     │
 │  └──────────┘   └───────────┘   │ measure     │  │     │
 │                                  └─────────────┘  │     │
 │                                       ▼           │     │
 │                               ┌───────────────┐   │     │
 │                               │ Conditioning  │◀──┘     │
 │                               │ Decision Rule │         │
 │                               └───────┬───────┘         │
 │                                       │                  │
 │                              ┌────────┴────────┐        │
 │                              │                 │        │
 │                          ACCEPT            REJECT       │
 │                        (keep shot)     (discard shot)   │
 └─────────────────────────────────────────────────────────┘
```

### Bell-Pair Subsystems

Each subsystem is a 2-qubit Bell pair (|00⟩ + |11⟩)/√2. Under noise, the
pair's parity may flip. Mid-circuit Z-parity measurements detect these
flips without collapsing the computational state.

### Multi-Rate Monitoring

- **HF (high-frequency):** Z-parity measured every monitoring cycle
- **LF (low-frequency):** Z-parity measured every 2nd cycle (0, 2, 4, …)
- The two rates provide complementary signal: LF captures slow drift, HF
  catches fast errors

---

## Conditioning Strategies

### 1. Global Conditioning

All N subsystems must pass all W monitoring cycles.

$$P_{\text{accept}}^{\text{global}} = \prod_{w=1}^{W} \prod_{i=1}^{N} p_i^{(w)}$$

**Limitation:** Exponential decay with N — unusable at N ≥ 2 under real noise.

### 2. Hierarchical k-of-N Conditioning

Accept if at least ⌈k·N⌉ subsystems pass each cycle:

$$P_{\text{accept}}^{\text{hier}} = \prod_{w=1}^{W} P\!\left(\sum_{i=1}^{N} X_i^{(w)} \ge \lceil k \cdot N \rceil\right)$$

**Advantage:** O(1) scaling — maintains high acceptance from N = 1 to N = 64.

### 3. Score Fusion Conditioning

Continuous metric combining LF and HF scores:

$$S_{\text{combined}} = \alpha \cdot \bar{S}_{\text{LF}} + (1 - \alpha) \cdot \bar{S}_{\text{HF}}$$

Accept if $S_{\text{combined}} \ge \theta$.

**Advantage:** Soft decision boundary absorbs noise spikes that break logical
(hard) fusion. The most robust strategy on real IBM hardware.

---

## Simulation Backends

### QuTiP (master-equation simulation)

The `src/sim.py` module simulates a driven qubit with pure dephasing:

$$H = \frac{\Omega}{2}\sigma_x + \frac{\omega}{2}\sigma_z$$

$$\mathcal{L}[\rho] = \gamma \left(\sigma_z \rho \sigma_z - \rho\right)$$

Fidelity is computed as $F(t) = \langle\psi_0|\rho(t)|\psi_0\rangle$ and
evaluated within a trailing time window.

### Qiskit (IBM Quantum hardware)

Dynamic circuits with mid-circuit measurements on real IBM processors.
Bell pairs are prepared, scrambled with random rotations, and measured via
ancilla-based Z-parity checks with reset and reuse.

---

## Project Structure

```
qgate-trajectory-filter/
├── packages/
│   └── qgate/                    # Pip-installable developer toolkit
│       ├── src/qgate/            # Core library (conditioning + monitors)
│       │   └── adapters/         # Mock, Qiskit, Grover, QAOA, VQE, QPE
│       ├── tests/                # 806 unit tests
│       └── pyproject.toml        # Build configuration
│
├── simulations/
│   ├── qutip_sims/              # QuTiP master-equation simulations
│   │   ├── scripts/             # Sweep runner scripts
│   │   ├── configs/             # Sweep parameter configs
│   │   └── runs/                # Results (CSV, JSON, figures, READMEs)
│   │
│   ├── ibm_hardware/            # IBM Quantum conditioning experiments
│   │   ├── circuits.py          # Dynamic circuit construction
│   │   ├── conditioning.py      # Post-selection logic
│   │   ├── sweep.py             # Parameter sweep engine
│   │   ├── plots.py             # Figure generation
│   │   └── run_ibm_experiment.py
│   │
│   ├── grover_tsvf/             # Grover vs TSVF-Grover (IBM Fez)
│   ├── qaoa_tsvf/               # QAOA vs TSVF-QAOA MaxCut (IBM Torino)
│   ├── vqe_tsvf/                # VQE vs TSVF-VQE TFIM (IBM Fez)
│   └── qpe_tsvf/                # QPE vs TSVF-QPE Phase Est. (IBM Fez)
│
├── examples/                    # Usage examples for the wrapper
├── docs/                        # Documentation
└── src/sim.py                   # Core QuTiP simulation engine
```

---

## Validation Chain

The research follows a progression from theory to hardware:

```
 QuTiP Simulations              IBM Quantum Hardware
 ═══════════════════             ════════════════════════
 1. High-noise sweep      ──▶   5. IBM Marrakesh experiment
    (300 configs, 405K           (120 rows, 5000 shots each,
     trials)                      ≈ 6 min on real hardware)
                                      │
 2. k-of-N follow-up      ──▶   Both confirm:
    (216 configs, N=1-32)        • Global collapses at N≥2
                                 • Hierarchical scales to N=64
 3. Incremental N=64             • Score fusion is most robust
    (6 new configs)                on real hardware
                                      │
 4. Multi-frequency sweep  ──▶   Score fusion absorbs HF noise
    (54 configs, 3 variants)     that destroys logical fusion
```

---

## Empirical Validation

| Result | Evidence | Location |
|---|---|---|
| Global conditioning collapses exponentially | High-noise sweep: 0% acceptance at N ≥ 2 | `simulations/qutip_sims/runs/high_noise_*/` |
| Hierarchical conditioning scales O(1) | Follow-up + incremental: 100% acceptance N = 1–64 | `simulations/qutip_sims/runs/followup_*/` |
| Multi-rate monitoring improves detection | Multi-freq sweep: score fusion at γ = 10 | `simulations/qutip_sims/runs/multifreq/` |
| Score fusion outperforms logical fusion | Multi-freq: 50% vs 0% acceptance at extreme noise | `simulations/qutip_sims/runs/multifreq/` |
| Hardware validation on IBM Quantum | IBM Marrakesh: score fusion best on real device | `simulations/ibm_hardware/` |

---

## TSVF Algorithm Experiments (IBM Hardware, Feb–Mar 2026)

These experiments extend qgate's trajectory filtering beyond Bell-pair
conditioning to four canonical quantum algorithms. The TSVF (Two-State
Vector Formalism) approach injects a mild chaotic perturbation and uses an
ancilla-based probe to create a post-selectable quality signal — then
filters for high-fidelity execution trajectories.

### Methodology

```
Standard Algorithm:  H → Algorithm Gates → Measure
                          ↓
TSVF Variant:       H → Algorithm Gates → Chaotic Perturbation → Probe Ancilla → Measure
                                                                        ↓
                                                              Post-select on ancilla |1⟩
```

The chaotic perturbation is deliberately mild — small random rotations
scaled as $\pi / (c \cdot \sqrt{d})$ where $d$ is the circuit depth parameter.
The probe ancilla applies controlled rotations that reward bitstrings
consistent with the expected solution structure. Post-selection on the
ancilla measuring $|1\rangle$ retains only trajectories that survived both
the hardware noise and the perturbation — a form of trajectory-level
quality filtering.

### Results Summary

| Algorithm | Backend | Metric | Standard | TSVF | Advantage |
|---|---|---|---|---|---|
| **Grover** (iter=4) | IBM Fez | Success probability | 0.0830 | **0.6105** | **7.3×** |
| **QAOA** (p=1) | IBM Torino | Approximation ratio | 0.4268 | **0.8029** | **1.88×** |
| **VQE** (L=3) | IBM Fez | Energy gap to ground | 2.398 | **1.291** | **1.86×** closer |
| **QPE** (t=7) | IBM Fez | Phase fidelity | **0.1569** | 0.0076 | ❌ N/A |

### Why TSVF Works for Some Algorithms but Not Others

The critical distinction is between **amplitude-encoded** and
**phase-coherence-encoded** information:

| Property | Grover / QAOA / VQE | QPE |
|---|---|---|
| Answer encoding | Amplitude pattern in computational basis | Phase coherence across precision register |
| Perturbation effect | Slightly scrambles amplitudes | Destroys inverse QFT interference |
| Post-selection recovers? | ✅ Yes — filters trajectories where signal survives | ❌ No — destroyed phase info is unrecoverable |
| Depth sensitivity | Moderate — noise accumulates gradually | High — single perturbation collapses peak |

**Grover, QAOA, VQE:** The answer is spread across computational basis
state amplitudes. A mild perturbation slightly degrades these amplitudes,
but the probe ancilla can detect which trajectories retained the signal.
Post-selection filters out noise-corrupted paths, yielding a smaller but
higher-fidelity sample.

**QPE:** The answer is encoded in the *relative phases* between precision
qubits, which the inverse QFT converts to a sharp probability peak.
Any perturbation — even $\pi/(6\sqrt{t})$ rotations — disrupts this
phase coherence, and the inverse QFT produces a diffuse rather than
peaked distribution. Post-selection cannot reconstruct the destroyed
phase information.

### Detailed Experiment Results

#### Grover vs TSVF-Grover (IBM Fez)

Target: 5-qubit marked state search, iterations 1–10, 8,192 shots.

| Iteration | P(success) std | P(success) TSVF | Ratio |
|:-:|:-:|:-:|:-:|
| 1 | 0.2131 | 0.1953 | 0.92× |
| 2 | 0.4329 | 0.3618 | 0.84× |
| 3 | 0.1801 | 0.4764 | 2.65× |
| 4 | 0.0830 | 0.6105 | **7.36×** |
| 5 | 0.0552 | 0.4318 | 7.82× |

> TSVF advantage grows as standard Grover's signal degrades at higher
> iterations. At iteration 4+ where standard success drops below 10%,
> TSVF maintains 40–60% success through trajectory filtering.

#### QAOA vs TSVF-QAOA MaxCut (IBM Torino)

Target: 6-node random MaxCut graph, QAOA layers p=1–5, 2,000 shots.

| p (layers) | AR std | AR TSVF | Ratio | Accept% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.4268 | 0.8029 | **1.88×** | 33.5% |
| 2 | 0.7036 | 0.7024 | 1.00× | 32.0% |
| 3 | 0.6975 | 0.6987 | 1.00× | 35.2% |

> Strongest advantage at p=1 where hardware noise most severely
> degrades the shallow QAOA circuit. TSVF rescues nearly 2× the
> approximation quality.

#### VQE vs TSVF-VQE for TFIM (IBM Fez)

Target: 4-qubit Transverse-Field Ising Model, ansatz layers L=1–6, 4,000 shots.

| L (layers) | Energy std | Energy TSVF | Gap std | Gap TSVF |
|:-:|:-:|:-:|:-:|:-:|
| 1 | −2.921 | −2.977 | 1.079 | 1.023 |
| 2 | −2.804 | −2.880 | 1.196 | 1.120 |
| 3 | −1.602 | −2.709 | **2.398** | **1.291** |
| 4 | −1.468 | −2.501 | 2.532 | 1.499 |

> Standard VQE hits a **barren plateau** at L=3 — energy suddenly jumps
> by ~1.2 units. TSVF maintains smooth energy descent through L=3,
> demonstrating barren plateau avoidance via trajectory filtering.

#### QPE vs TSVF-QPE Phase Estimation (IBM Fez)

Target: eigenphase φ = 1/3, precision qubits t=3–7, 8,192 shots.

| t (precision) | Fidelity std | Fidelity TSVF | Correct phase? (std) | Correct phase? (TSVF) |
|:-:|:-:|:-:|:-:|:-:|
| 3 | **0.582** | 0.064 | ✅ `011` | ❌ `110` |
| 5 | **0.369** | 0.035 | ✅ `01011` | ❌ `11010` |
| 7 | **0.157** | 0.008 | ✅ `0101011` | ❌ `0001110` |

> Standard QPE correctly identifies φ ≈ 1/3 at all precision levels on
> IBM Fez hardware. TSVF perturbation destroys phase coherence — the
> post-selected histogram shows near-uniform randomness rather than a
> sharp peak at the correct binary fraction.
