---
description: >-
  ML-augmented error mitigation architecture in qgate. Three-tier pipeline:
  TelemetryMitigator (Level-2 bit-string ML), PulseMitigator (Level-1 IQ analog ML),
  and QgateTranspiler (ML-aware circuit compilation). How they compose with the
  original trajectory filter and when to use each tier.
keywords: quantum error mitigation, machine learning, TelemetryMitigator, PulseMitigator, QgateTranspiler, ML pipeline, NISQ, qgate, two-stage mitigation, pulse-level, IQ readout, active cancellation
faq:
  - q: What is ML-augmented error mitigation in qgate?
    a: qgate extends its original binary trajectory filter with three ML modules — TelemetryMitigator (Level-2 bit-string regression), PulseMitigator (Level-1 IQ-based active cancellation), and QgateTranspiler (ML-aware circuit compilation) — that compose to reduce QPU cost by up to 10× while improving fidelity.
  - q: How does TelemetryMitigator differ from the original TrajectoryFilter?
    a: TrajectoryFilter is binary accept/reject. TelemetryMitigator adds a second stage — an ML regressor trained on calibration circuits predicts the residual error after Galton filtering, converting a binary filter into a continuous correction.
  - q: Does qgate automatically handle noise amplification for ZNE?
    a: No. The transpiler pipeline handles probe injection and circuit padding. Noise amplification for ZNE is a separate concern — qgate provides opt-in utilities (validate_noise_scale_factor, apply_uzdin_unitary_folding) but never applies them automatically.
---

# ML-Augmented Error Mitigation

!!! warning "Patent Pending — Confidential"
    US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915.
    CIP addendum — ML-augmented TSVF trajectory mitigation.
    **DO NOT PUSH. DO NOT PUBLISH.**

---

## Overview

The original qgate pipeline uses **binary trajectory filtering** — each shot
is either accepted or rejected based on parity-check scores.  This works, but
it requires aggressive chaotic padding (inflating circuit depth) and 10×
shot oversampling to compensate for the high rejection rate.

The ML-augmented pipeline replaces the binary decision with **continuous
error correction**, reducing QPU cost by an order of magnitude while
improving estimation accuracy.

The architecture has three tiers:

| Tier | Module | Abstraction Level | What It Does |
|---|---|---|---|
| **Level 2** | [`TelemetryMitigator`](../api.md#qgate.mitigation) | Bit-string (digital) | Two-stage Galton filter + ML regression |
| **Level 1** | [`PulseMitigator`](../api.md#qgate.pulse_mitigator) | IQ readout (analog) | Real-time TLS drift prediction + active cancellation |
| **Compiler** | [`QgateTranspiler`](../api.md#qgate.transpiler) | Circuit compilation | ML-aware probe injection + conditional padding |

Each tier can be used independently or composed together.

---

## Architecture

```
                    ┌───────────────────────────────────┐
                    │         QgateTranspiler            │
                    │  (ML-aware circuit compilation)    │
                    │                                    │
                    │  1. Telemetry probe injection      │
                    │  2. Conditional chaotic padding     │
                    │     • legacy_filter → full padding  │
                    │     • ml_extrapolation → SKIP       │
                    │     • pulse_active → SKIP           │
                    │  3. Shot optimisation (1.2× vs 10×) │
                    └──────────────┬────────────────────┘
                                   │ compiled circuit
                                   ▼
                    ┌───────────────────────────────────┐
                    │         QPU Execution              │
                    │   (IBM Quantum / Aer Simulator)    │
                    └──────────┬────────────┬───────────┘
                               │            │
                    Level-2 bits    Level-1 IQ data
                               │            │
                    ┌──────────▼──────┐  ┌──▼──────────────────┐
                    │ TelemetryMitigator│  │   PulseMitigator    │
                    │                   │  │                     │
                    │ Stage 1: Galton   │  │ IQ feature extract  │
                    │   filter (~70%    │  │ → predict TLS drift │
                    │   retained)       │  │ → inject inverse    │
                    │                   │  │   frequency shift   │
                    │ Stage 2: ML       │  │   into drive pulse  │
                    │   regression      │  │                     │
                    │   (predict        │  │ (active cancellation│
                    │    residual)      │  │  at firmware level) │
                    └──────────┬───────┘  └──────────┬──────────┘
                               │                     │
                               ▼                     ▼
                    ┌───────────────────────────────────┐
                    │   Mitigated expectation value      │
                    └───────────────────────────────────┘
```

---

## Tier 1: TelemetryMitigator (Level-2 Bit-String ML)

**Module:** `qgate.mitigation`

The TelemetryMitigator operates on standard measurement outcomes
(bit-strings / expectation values) — the same data any quantum
circuit produces.

### Two-Stage Pipeline

**Stage 1 — Galton Filtering:**
Apply the existing qgate trajectory filter with adaptive thresholding.
This retains approximately 70% of shots (the highest-fidelity fraction)
and provides a first-pass error suppression.

**Stage 2 — ML Regression:**
A scikit-learn regressor (default: `RandomForestRegressor`) is trained
on near-Clifford calibration circuits whose ideal values are efficiently
simulable.  The model learns to predict the *residual correction*:

$$\hat{c} = f_\theta(\mathbf{x})$$

where $\mathbf{x}$ is a 6-dimensional telemetry feature vector:

| Feature | Description |
|---|---|
| `energy` | Raw (or Galton-filtered) expectation value |
| `acceptance` | Galton trajectory acceptance rate (0–1) |
| `variance` | Shot-to-shot variance |
| `abs_energy` | $\lvert E \rvert$ — magnitude symmetry feature |
| `energy_x_acceptance` | $E \times A$ — cross term |
| `residual_from_mean` | $E - \bar{E}$ — deviation from batch mean |

The mitigated estimate is then:

$$\hat{E}_{\text{mitigated}} = E_{\text{filtered}} + \hat{c}$$

### Quick Start

```python
from qgate.mitigation import TelemetryMitigator, MitigatorConfig

config = MitigatorConfig(keep_fraction=0.70)
mitigator = TelemetryMitigator(config=config)

# Step 1 — calibrate with known circuits
cal = mitigator.calibrate(
    calibration_data=[
        {"energy": -1.02, "acceptance": 0.73, "variance": 0.04, "ideal": -1.0},
        {"energy": -0.91, "acceptance": 0.65, "variance": 0.06, "ideal": -0.95},
        # ... 10–50 near-Clifford calibration points
    ]
)
print(f"R² = {cal.r2_score:.3f}")

# Step 2 — mitigate a new measurement
result = mitigator.estimate(
    raw_energy=-1.08,
    acceptance=0.68,
    variance=0.05,
)
print(f"Mitigated: {result.mitigated_value:.4f}")
```

### When to Use

- You have access to standard measurement outcomes (Level-2 data)
- You can run 10–50 near-Clifford calibration circuits
- You want to improve on binary trajectory filtering without hardware changes

---

## Tier 2: PulseMitigator (Level-1 IQ Analog ML)

**Module:** `qgate.pulse_mitigator`

The PulseMitigator operates *below* the gate abstraction, on raw
In-phase / Quadrature (IQ) readout voltages.  It predicts Two-Level
System (TLS) frequency drift in real time and injects compensating
frequency shifts into the qubit drive pulse.

### Pipeline

1. **Calibrate** — execute circuits with known artificial detunings;
   for each shot, extract a 5-dimensional IQ feature vector:

    | Feature | Description |
    |---|---|
    | `magnitude` | $\sqrt{I^2 + Q^2}$ |
    | `phase` | $\arctan(Q / I)$ |
    | `distance_to_0` | Euclidean distance to $\lvert 0 \rangle$ centroid |
    | `distance_to_1` | Euclidean distance to $\lvert 1 \rangle$ centroid |
    | `temporal_delta_phase` | Phase change from rolling average |

2. **Predict drift** — feed a new IQ measurement through the trained
   model to obtain the predicted TLS detuning in Hz.

3. **Active cancellation** — inject a `ShiftFrequency(−Δf)` instruction
   into the drive pulse, so the subsequent gate operates at the
   corrected transition frequency.

### Qiskit Pulse Compatibility

`qiskit.pulse` was removed in Qiskit ≥ 2.0.  PulseMitigator handles
both environments transparently:

| Mode | Qiskit Version | Pulse Schedules | ML Pipeline |
|---|---|---|---|
| **Pulse mode** | 1.x | Real `ScheduleBlock` + `ShiftFrequency` | ✅ |
| **Simulation mode** | 2.x+ | Lightweight `SimulatedPulseSchedule` dataclass | ✅ |

Both modes share identical feature engineering and ML prediction logic.

### Quick Start

```python
from qgate.pulse_mitigator import PulseMitigator, PulseMitigatorConfig

config = PulseMitigatorConfig(target_qubit=0)
pm = PulseMitigator(config=config)

# Calibrate with known detunings
pm.calibrate(iq_shots=[...], detunings_hz=[0, 500, -500, 1000, -1000])

# Predict drift from a new IQ probe
drift = pm.predict_drift(i=0.0012, q=-0.0034)
print(f"Predicted TLS drift: {drift.predicted_detuning_hz:.1f} Hz")

# Full active cancellation
corrected = pm.run_with_active_cancellation(
    target_circuit=qc,
    probe_i=0.0012,
    probe_q=-0.0034,
)
```

### When to Use

- You have access to Level-1 IQ readout data (IBM backends with `meas_level=1`)
- You are targeting TLS drift / frequency drift errors
- You want firmware-level mitigation that corrects errors *before* they happen

---

## Tier 3: QgateTranspiler (ML-Aware Compilation)

**Module:** `qgate.transpiler`

The transpiler sits *upstream* of execution — it prepares circuits for
the active mitigation strategy.  See [Transpiler](transpiler.md) for
full documentation.

### Key Insight: What the Transpiler Does and Does NOT Do

| ✅ The transpiler does | ❌ The transpiler does NOT |
|---|---|
| Inject telemetry probes (ancilla qubits) | Perform noise amplification / ZNE |
| Conditionally apply chaotic padding | Insert circuit folding (U·U†·U) |
| Optimise shot count for active mitigator | Build or manipulate noise models |
| Auto-tune settings based on mitigation mode | Extrapolate to zero noise |

!!! important "Noise amplification is NOT automatic"
    The transpiler **never** inserts noise amplification into circuits.
    ZNE (Zero-Noise Extrapolation) circuit folding is a separate concern
    handled by **opt-in utility functions**:

    - `validate_noise_scale_factor()` — validates odd-integer λ values
    - `apply_uzdin_unitary_folding()` — applies U→U·(U†·U)ⁿ folding

    See [Uzdin Unitary Folding](uzdin-folding.md) for details.

---

## Cost Comparison

| Metric | Legacy Filter | ML Extrapolation | Pulse Active |
|---|---|---|---|
| **Chaotic padding** | Full (2Q gates × depth × qubits) | None | None |
| **Shot oversampling** | 10× | 1.2× | 1.2× |
| **Circuit depth overhead** | High | Probes only | Probes only |
| **QPU cost factor** | 1.0× (baseline) | ~0.12× | ~0.12× |
| **Requires calibration?** | No | Yes (10–50 circuits) | Yes (IQ probe bank) |
| **Requires Level-1 data?** | No | No | Yes |

---

## Composing the Tiers

The three tiers are designed to compose:

```python
from qgate.transpiler import QgateTranspiler, QgateTranspilerConfig
from qgate.mitigation import TelemetryMitigator, MitigatorConfig

# 1. Transpiler — compile with ML-aware settings
config = QgateTranspilerConfig.for_mode("ml_extrapolation")
transpiler = QgateTranspiler(config=config)
result = transpiler.compile(qc, base_shots=4096)
compiled_circuit = result.circuit       # probes injected, no padding
optimized_shots = result.optimized_shots  # 4916 (4096 × 1.2)

# 2. Execute on hardware
# job = backend.run(compiled_circuit, shots=optimized_shots)

# 3. Mitigate with TelemetryMitigator
mitigator = TelemetryMitigator(MitigatorConfig(keep_fraction=0.70))
mitigator.calibrate(calibration_data=[...])
mitigated = mitigator.estimate(
    raw_energy=measured_energy,
    acceptance=galton_rate,
    variance=shot_variance,
)
```

Or compose all three tiers:

```python
from qgate.pulse_mitigator import PulseMitigator, PulseMitigatorConfig

# 1. Transpile (pulse_active mode)
config = QgateTranspilerConfig.for_mode("pulse_active")
transpiler = QgateTranspiler(config=config)
compiled = transpiler.compile(qc, base_shots=4096)

# 2. Active cancellation at pulse level
pm = PulseMitigator(PulseMitigatorConfig(target_qubit=0))
pm.calibrate(iq_shots=[...], detunings_hz=[...])
corrected = pm.run_with_active_cancellation(
    target_circuit=compiled.circuit,
    probe_i=0.0012, probe_q=-0.0034,
)

# 3. Post-execution bit-string mitigation
mitigator = TelemetryMitigator(MitigatorConfig())
mitigator.calibrate(calibration_data=[...])
final = mitigator.estimate(raw_energy=..., acceptance=..., variance=...)
```

---

## Benchmark Results

Full-stack benchmarks validate the ML pipeline across 10 tiers (49 metrics):

| Tier | What It Tests | Key Result |
|---|---|---|
| T1 | Synthetic PoC | 16.1% MSE reduction |
| T2 | Model comparison (RF vs GB vs Ridge) | 461–2536× improvement |
| T3 | Cross-algorithm generalisation | 57–1847× improvement |
| T4 | Real-data pipeline | 1.6–2.9× improvement |
| T5 | Full stack (Transpiler + Mitigator + Pulse) | 2129× combined |
| T6 | Transpiler efficiency | 8.3× shot reduction, 10.6× QPU saving |
| T7 | Depth-scaling survival (adaptive Galton) | 146–3052× in-training; 1.3–3.0× extrapolation (d ≤ 1000) |
| T8 | Shot efficiency curve | 75% accuracy at 20% shot budget |
| T9 | Noise-regime phase diagram | ML dominates across all regimes |
| T10 | TVS Level-1 vs Level-2 fusion | Level-2 TVS gives 3–4× additional lift |

See `simulations/ml_trajectory_mitigation/run_full_stack_benchmark.py` for
the complete benchmark suite.

---

## Next Steps

- [QgateTranspiler](transpiler.md) — detailed transpiler documentation
- [Uzdin Unitary Folding](uzdin-folding.md) — noise amplification utilities
- [API Reference](../api.md) — auto-generated API docs
