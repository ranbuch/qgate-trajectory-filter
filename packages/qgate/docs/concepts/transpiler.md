---
description: >-
  QgateTranspiler — ML-aware quantum circuit compiler. Automatic probe injection,
  conditional chaotic padding, shot optimisation, and mode-specific auto-tuning.
  Clear boundary: the transpiler does NOT perform noise amplification or ZNE.
keywords: QgateTranspiler, quantum transpiler, ML-aware compilation, telemetry probes, chaotic padding, shot optimisation, mitigation mode, qgate
faq:
  - q: What does QgateTranspiler do?
    a: It prepares circuits for execution by injecting telemetry probes, conditionally adding chaotic padding (only in legacy mode), and optimising the shot count based on the active mitigation strategy.
  - q: Does QgateTranspiler do ZNE or noise amplification?
    a: No. The transpiler never inserts noise amplification. ZNE circuit folding is handled by separate opt-in utilities (validate_noise_scale_factor, apply_uzdin_unitary_folding) in the same module.
  - q: What are the three mitigation modes?
    a: legacy_filter (full padding + 10× oversampling), ml_extrapolation (probes only + 1.2× oversampling for TelemetryMitigator), and pulse_active (probes only + 1.2× oversampling for PulseMitigator).
---

# QgateTranspiler

!!! warning "Patent Pending — Confidential"
    US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915.
    **DO NOT PUSH. DO NOT PUBLISH.**

---

## What It Does

The `QgateTranspiler` is an **ML-aware quantum circuit compiler** that
sits between your algorithm circuit and the QPU.  Its job is to prepare
the circuit for whichever error-mitigation strategy you've chosen.

The transpiler handles three things automatically:

1. **Telemetry probe injection** — adds weak-measurement ancilla qubits
2. **Conditional chaotic padding** — adds pseudo-random mixing gates
   (legacy mode only)
3. **Shot optimisation** — calculates the shot count needed after filtering

### What It Does NOT Do

!!! important "No automatic noise amplification"
    The transpiler **never** performs:

    - Zero-Noise Extrapolation (ZNE)
    - Circuit folding (U·U†·U patterns)
    - Noise model construction
    - Noise scaling of any kind

    These are separate concerns.  For ZNE circuit folding, see
    [Uzdin Unitary Folding](uzdin-folding.md) — opt-in utility functions
    in the same module.

---

## Pipeline

```
 Input Circuit              QgateTranspiler                 Output
 ════════════     ════════════════════════════════     ════════════════
                  ┌──────────────────────────────┐
  QuantumCircuit  │ 1. Inject telemetry probes   │   CompilationResult
  (n qubits)  ──▶│ 2. Add chaotic padding?       │──▶  .circuit (n+1 qubits)
                  │    • legacy → YES             │     .optimized_shots
  base_shots  ──▶│    • ml/pulse → NO            │     .metadata
                  │ 3. shots = base × oversample  │
                  └──────────────────────────────┘
```

### Step 1: Telemetry Probe Injection (All Modes)

One ancilla qubit and one classical bit are added.  The ancilla is
entangled with nearest-neighbour system qubit pairs via controlled-RY
gates that reward ferromagnetic alignment ($\lvert 00 \rangle$ and
$\lvert 11 \rangle$):

$$\theta_{\text{pair}} = \frac{\theta_{\text{probe}}}{\max(n_{\text{system}} - 1,\; 1)}$$

This implements the standard qgate energy-probe protocol — identical
across all three modes.

### Step 2: Conditional Chaotic Padding

| Mode | Padding Applied? | Rationale |
|---|---|---|
| `legacy_filter` | ✅ Yes | Binary filter needs decorrelated shots |
| `ml_extrapolation` | ❌ No | ML learns from features, not shot decorrelation |
| `pulse_active` | ❌ No | Firmware cancellation needs shallow depth |

When applied, chaotic padding inserts `mixing_depth` layers of
pseudo-random 2-qubit gates across all qubit pairs, seeded by
`mixing_seed` for reproducibility.

### Step 3: Shot Optimisation

$$\text{optimized\_shots} = \lceil \text{base\_shots} \times \text{oversampling\_factor} \rceil$$

| Mode | Oversampling Factor | Why |
|---|---|---|
| `legacy_filter` | 10.0× | ~90% of shots are rejected |
| `ml_extrapolation` | 1.2× | ~30% truncated by Galton filter |
| `pulse_active` | 1.2× | ~30% truncated by Galton filter |

---

## Configuration

### QgateTranspilerConfig

All configuration is controlled by the frozen (immutable) `QgateTranspilerConfig`
dataclass.  The recommended way to create it is via the factory method:

```python
from qgate.transpiler import QgateTranspilerConfig

# Recommended — auto-tuned for your mitigation strategy
config = QgateTranspilerConfig.for_mode("ml_extrapolation")

# These are set automatically:
assert config.aggressive_mixing is False
assert config.oversampling_factor == 1.2
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mitigation_mode` | `str` | `"legacy_filter"` | Active error-mitigation strategy |
| `aggressive_mixing` | `bool` | *auto* | Whether to inject chaotic padding gates |
| `oversampling_factor` | `float` | *auto* | Shot inflation multiplier |
| `probe_angle` | `float` | `π/6` | Total weak-rotation angle (rad) |
| `mixing_depth` | `int` | `3` | Number of chaotic padding layers |
| `mixing_seed` | `int` | `42` | RNG seed for reproducible padding |

!!! note "Auto-tuning"
    When you set `mitigation_mode`, the config **automatically** adjusts
    `aggressive_mixing` and `oversampling_factor`.  You can still override
    them explicitly if needed.

### Mode Defaults

```python
# legacy_filter → full padding, max oversampling
cfg = QgateTranspilerConfig(mitigation_mode="legacy_filter")
assert cfg.aggressive_mixing is True
assert cfg.oversampling_factor == 10.0

# ml_extrapolation → no padding, minimal oversampling
cfg = QgateTranspilerConfig(mitigation_mode="ml_extrapolation")
assert cfg.aggressive_mixing is False
assert cfg.oversampling_factor == 1.2

# pulse_active → no padding, minimal oversampling
cfg = QgateTranspilerConfig(mitigation_mode="pulse_active")
assert cfg.aggressive_mixing is False
assert cfg.oversampling_factor == 1.2

# Explicit override — ML mode but custom oversampling
cfg = QgateTranspilerConfig(
    mitigation_mode="ml_extrapolation",
    oversampling_factor=2.0,
)
assert cfg.oversampling_factor == 2.0  # your override is preserved
```

---

## Usage

### Basic Compilation

```python
from qiskit import QuantumCircuit
from qgate.transpiler import QgateTranspiler, QgateTranspilerConfig

# Create a circuit
qc = QuantumCircuit(4)
qc.h(range(4))
qc.measure_all()

# Compile with ML-aware settings
config = QgateTranspilerConfig.for_mode("ml_extrapolation")
transpiler = QgateTranspiler(config=config)
result = transpiler.compile(qc, base_shots=4096)

print(result.circuit.num_qubits)         # 5 (4 system + 1 probe)
print(result.optimized_shots)             # 4916 (4096 × 1.2)
print(result.chaotic_padding_applied)     # False
print(result.probes_injected)             # True
```

### CompilationResult

The `compile()` method returns a `CompilationResult` with:

| Field | Type | Description |
|---|---|---|
| `circuit` | `QuantumCircuit` | Transpiled circuit with probes |
| `optimized_shots` | `int` | Shot count after oversampling |
| `probes_injected` | `bool` | Whether telemetry probes were added |
| `chaotic_padding_applied` | `bool` | Whether chaotic mixing was injected |
| `metadata` | `dict` | Compiler telemetry (mode, angles, qubit counts) |

### Integration with TelemetryMitigator

```python
from qgate.transpiler import QgateTranspiler, QgateTranspilerConfig
from qgate.mitigation import TelemetryMitigator, MitigatorConfig

# 1. Compile — probes only, 1.2× oversampling
transpiler = QgateTranspiler(
    QgateTranspilerConfig.for_mode("ml_extrapolation")
)
compiled = transpiler.compile(qc, base_shots=4096)

# 2. Execute on backend
# job = backend.run(compiled.circuit, shots=compiled.optimized_shots)

# 3. Mitigate
mitigator = TelemetryMitigator(MitigatorConfig(keep_fraction=0.70))
mitigator.calibrate(calibration_data=[...])
result = mitigator.estimate(
    raw_energy=measured_energy,
    acceptance=galton_rate,
    variance=shot_variance,
)
```

---

## QPU Cost Savings

The key insight: when ML mitigators handle error correction, the
transpiler can skip expensive circuit padding.

### Legacy Mode (Binary Filter)

```
 Circuit depth:  D_original + D_probes + D_chaotic_padding
 Shot count:     base_shots × 10
 QPU cost:       ~10× baseline
```

### ML Mode

```
 Circuit depth:  D_original + D_probes   (no padding!)
 Shot count:     base_shots × 1.2
 QPU cost:       ~0.12× baseline         (83× cheaper than legacy)
```

This cost reduction is validated in benchmark tier T6 (Transpiler
Efficiency), which measures:

- **8.3× shot reduction** (10.0× → 1.2×)
- **Depth ratio 1.0–1.3×** (padding removed)
- **Combined QPU saving 8.6–10.6×**

---

## See Also

- [ML-Augmented Mitigation](ml-mitigation.md) — three-tier architecture overview
- [Uzdin Unitary Folding](uzdin-folding.md) — opt-in ZNE noise amplification utilities
- [API Reference](../api.md) — auto-generated docstrings
