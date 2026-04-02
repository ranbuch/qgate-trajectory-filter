---
description: >-
  Architecture and methodology of qgate's quantum error suppression system.
  Bell-pair subsystems, mid-circuit Z-parity measurements, conditioning strategies
  (global, hierarchical k-of-N, score fusion), and validation from QuTiP simulation to IBM hardware.
keywords: quantum error suppression architecture, Bell pair conditioning, Z-parity measurement, score fusion, hierarchical k-of-N, QuTiP simulation, IBM Quantum validation
faq:
  - q: How does qgate suppress quantum errors?
    a: qgate uses Bell-pair subsystems as noise probes. Mid-circuit Z-parity measurements detect errors without collapsing the computational state. Decision rules (global, hierarchical, or score fusion) then accept or reject each shot based on subsystem fidelity.
  - q: What is score fusion conditioning?
    a: Score fusion combines high-frequency and low-frequency parity monitoring scores into a continuous metric using an alpha-weighted average. This soft decision boundary is the most robust strategy on real IBM hardware.
  - q: What is hierarchical k-of-N conditioning?
    a: Instead of requiring all subsystems to pass (which fails at scale), hierarchical conditioning accepts shots where at least k out of N subsystems pass each monitoring cycle. This achieves O(1) scaling from N=1 to N=64.
---

# Architecture & Methodology

> **Patent notice:** US Patent App. Nos. 63/983,831 & 63/989,632 | Israeli Patent App. No. 326915. The underlying invention is patent pending.

---

## Overview

This repository implements and validates **quantum error suppression via
post-selection conditioning on Bell-pair subsystems**. The approach uses
mid-circuit measurements to monitor subsystem fidelity and applies decision
rules to accept or reject quantum computation outcomes — thereby suppressing
errors without full quantum error correction.

<figure markdown="span">
  ![qgate conceptual schematic showing the full trajectory filtering pipeline from Bell-pair preparation through conditioning and decision](assets/images/architecture/conceptual-schematic.svg){ width="800" loading="lazy" }
  <figcaption>Conceptual schematic of the qgate trajectory filtering pipeline. Bell-pair subsystems are prepared, scrambled under noise, and monitored via mid-circuit Z-parity measurements. The conditioning decision rule (global, hierarchical, or score fusion) accepts or rejects each shot based on subsystem fidelity.</figcaption>
</figure>

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

Each subsystem is a 2-qubit Bell pair $(\lvert00\rangle + \lvert11\rangle)/\sqrt{2}$.
Under noise, the pair's parity may flip. Mid-circuit Z-parity measurements
detect these flips without collapsing the computational state.

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

!!! warning "Limitation"
    Exponential decay with N — unusable at N ≥ 2 under real noise.

### 2. Hierarchical k-of-N Conditioning

Accept if at least ⌈k·N⌉ subsystems pass each cycle:

$$P_{\text{accept}}^{\text{hier}} = \prod_{w=1}^{W} P\!\left(\sum_{i=1}^{N} X_i^{(w)} \ge \lceil k \cdot N \rceil\right)$$

!!! success "Advantage"
    O(1) scaling — maintains high acceptance from N = 1 to N = 64.

### 3. Score Fusion Conditioning

Continuous metric combining LF and HF scores:

$$S_{\text{combined}} = \alpha \cdot \bar{S}_{\text{LF}} + (1 - \alpha) \cdot \bar{S}_{\text{HF}}$$

Accept if $S_{\text{combined}} \ge \theta$.

!!! success "Advantage"
    Soft decision boundary absorbs noise spikes that break logical
    (hard) fusion. The most robust strategy on real IBM hardware.

---

## Simulation Backends

### QuTiP (Master-Equation Simulation)

The simulation module models a driven qubit with pure dephasing:

$$H = \frac{\Omega}{2}\sigma_x + \frac{\omega}{2}\sigma_z$$

$$\mathcal{L}[\rho] = \gamma \left(\sigma_z \rho \sigma_z - \rho\right)$$

Fidelity is computed as $F(t) = \langle\psi_0\lvert\rho(t)\lvert\psi_0\rangle$ and
evaluated within a trailing time window.

### Qiskit (IBM Quantum Hardware)

Dynamic circuits with mid-circuit measurements on real IBM processors.
Bell pairs are prepared, scrambled with random rotations, and measured via
ancilla-based Z-parity checks with reset and reuse.

---

## Project Structure

```
qgate-trajectory-filter/
├── packages/
│   └── qgate/                    # Pip-installable developer toolkit
│       ├── src/qgate/            # Core library
│       │   ├── adapters/         # Mock, Qiskit, Grover, QAOA, VQE, QPE
│       │   ├── mitigation.py     # TelemetryMitigator (Level-2 ML)
│       │   ├── pulse_mitigator.py# PulseMitigator (Level-1 IQ ML)
│       │   ├── transpiler.py     # QgateTranspiler + Uzdin utilities
│       │   ├── compressor.py     # TelemetryCompressor (utility-scale)
│       │   └── tvs.py            # TVS: HF/LF fusion + Galton filter + adaptive schedule
│       ├── tests/                # 806 unit tests
│       └── pyproject.toml        # Build configuration
│
├── simulations/
│   ├── qutip_sims/              # QuTiP master-equation simulations
│   ├── ibm_hardware/            # IBM Quantum conditioning experiments
│   ├── grover_tsvf/             # Grover vs TSVF-Grover (IBM Fez)
│   ├── qaoa_tsvf/               # QAOA vs TSVF-QAOA MaxCut (IBM Torino)
│   ├── vqe_tsvf/                # VQE vs TSVF-VQE TFIM (IBM Fez)
│   ├── qpe_tsvf/                # QPE vs TSVF-QPE Phase Est. (IBM Fez)
│   ├── ml_trajectory_mitigation/# Full-stack ML benchmarks (T1–T10)
│   └── vns_compatibility/       # VNS experiment (Uzdin-compliant)
│
├── examples/                    # Usage examples
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

## ML-Augmented Pipeline (CIP Addendum)

!!! warning "Patent Pending — Confidential"
    US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915.
    CIP addendum — ML-augmented TSVF trajectory mitigation.

The CIP extends the binary trajectory filter with three ML modules that
compose to reduce QPU cost by up to 10× while improving estimation accuracy.

```
 Binary Pipeline (v0.6)          ML-Augmented Pipeline (CIP)
 ═══════════════════════          ═══════════════════════════════
 Chaotic padding (10× depth)     QgateTranspiler (ML-aware)
       │                                │
 10× shot oversampling            1.2× oversampling (probes only)
       │                                │
 Binary accept/reject             TelemetryMitigator (Level-2 ML)
       │                           + Galton filter → ML regression
       ▼                                │
 Filtered result                  PulseMitigator (Level-1 IQ ML)
                                   + Active cancellation
                                        │
                                        ▼
                                  Mitigated expectation value
```

### Module Responsibility Boundary

| Module | Does | Does NOT |
|---|---|---|
| `QgateTranspiler` | Probe injection, padding, shots | Noise amplification, ZNE |
| `TelemetryMitigator` | Galton filter + ML regression | Circuit modification |
| `PulseMitigator` | IQ drift prediction + cancellation | Bit-string processing |
| `process_telemetry_batch` | HF/LF fusion, Galton filtering, ML features | ML regression, circuit ops |
| `TelemetryCompressor` | Spatial pooling + Gini pruning | Training downstream models |
| Uzdin utilities | Circuit folding validation | Automatic invocation |

See [ML-Augmented Mitigation](concepts/ml-mitigation.md) for full documentation.

---

## Empirical Validation

| Result | Evidence |
|---|---|
| Global conditioning collapses exponentially | High-noise sweep: 0% acceptance at N ≥ 2 |
| Hierarchical conditioning scales O(1) | Follow-up + incremental: 100% acceptance N = 1–64 |
| Multi-rate monitoring improves detection | Multi-freq sweep: score fusion at γ = 10 |
| Score fusion outperforms logical fusion | Multi-freq: 50% vs 0% acceptance at extreme noise |
| Hardware validation on IBM Quantum | IBM Marrakesh: score fusion best on real device |

---

## TSVF Algorithm Extensions

Beyond Bell-pair conditioning, qgate extends trajectory filtering to
canonical quantum algorithms via the TSVF (Two-State Vector Formalism)
approach. See [Hardware Experiments](experiments/index.md) for full results.

| Algorithm | Backend | Advantage |
|---|---|---|
| [Grover](experiments/grover.md) | IBM Fez | **7.3×** at iteration 4 |
| [QAOA](experiments/qaoa.md) | IBM Torino | **1.88×** at p=1 |
| [VQE](experiments/vqe.md) | IBM Fez | **1.86×** closer (barren plateau avoidance) |
| [QPE](experiments/qpe.md) | IBM Fez | N/A (phase-coherence incompatible) |
