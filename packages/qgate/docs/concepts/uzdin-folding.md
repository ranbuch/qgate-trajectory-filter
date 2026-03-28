---
description: >-
  Uzdin odd-factor unitary folding in qgate. Why noise amplification must use
  positive odd integers, the difference between unitary folding and digital folding,
  validate_noise_scale_factor() and apply_uzdin_unitary_folding() API documentation,
  and the critical distinction between qgate's automatic pipeline and opt-in ZNE.
keywords: Uzdin rule, unitary folding, ZNE, zero-noise extrapolation, noise amplification, odd integer, digital folding, coherent error, qgate, quantum error mitigation
faq:
  - q: Why must ZNE scale factors be odd integers?
    a: Strict unitary folding maps U → U·(U†·U)ⁿ, which requires an odd number of copies. Even or non-integer factors require "digital folding" — inserting partial-gate pairs that introduce coherent errors not present in the original noise channel.
  - q: Does qgate automatically apply ZNE noise amplification?
    a: No. The qgate transpiler handles probe injection and padding. ZNE folding is a separate, opt-in utility via validate_noise_scale_factor() and apply_uzdin_unitary_folding().
  - q: What happens if I pass an even scale factor?
    a: validate_noise_scale_factor() raises a ValueError explaining the Uzdin rule violation. apply_uzdin_unitary_folding() calls the validator internally, so it will also reject even factors.
---

# Uzdin Unitary Folding

!!! warning "Patent Pending — Confidential"
    US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915.
    **DO NOT PUSH. DO NOT PUBLISH.**

---

## The Problem: Digital Folding Trap

Zero-Noise Extrapolation (ZNE) is a popular error mitigation technique.
The idea is simple: run circuits at several **amplified noise levels**
$\lambda \in \{1, 3, 5, 7, \ldots\}$, then extrapolate the observable
back to $\lambda = 0$ (zero noise).

The dangerous part is **how** you amplify the noise.

### Two Approaches

| Approach | Method | Physically Correct? |
|---|---|---|
| **Unitary folding** | $U \to U \cdot (U^\dagger \cdot U)^n$ | ✅ Yes — strict KIK identity |
| **Digital folding** | Insert partial-gate pairs, fractional scaling | ❌ No — introduces coherent errors |

### Why Unitary Folding Requires Odd Integers

In unitary folding, each gate $U$ is replaced by:

$$U \to U \cdot \underbrace{(U^\dagger \cdot U) \cdot (U^\dagger \cdot U) \cdot \ldots}_{n \text{ pairs}}$$

Count the total copies of $U$: the original plus $n$ forward copies from
the pairs = $1 + 2n$.  This is always **odd**.

$$\lambda = 1 + 2n, \quad n \in \{0, 1, 2, 3, \ldots\}$$

$$\therefore \lambda \in \{1, 3, 5, 7, 9, \ldots\}$$

There is no way to get $\lambda = 2$ or $\lambda = 1.5$ with strict
unitary folding.

### What Goes Wrong with Even/Non-Integer Factors

To achieve $\lambda = 2$, you would need to:

- Insert a $U^\dagger$ without a matching $U$, or
- Apply a "partial fold" using fractional gate powers

Both techniques break the KIK (Kraus-Inverse-Kraus) identity.  The
noise channel of the folded circuit is **no longer** a simple scaling
of the original noise channel — it includes **new coherent error terms**
that weren't present in the unscaled circuit.

The consequence: your extrapolation curve is fitting the wrong model.
The zero-noise limit you extrapolate to is **wrong**.

!!! danger "The Uzdin Rule"
    *Any noise amplification factor that is not a positive odd integer
    introduces coherent errors not present in the original noise channel,
    making the ZNE extrapolation fundamentally unreliable.*

    **Reference:** R. Uzdin, *"KIK identity and noise scaling"*,
    supplementary material to Temme et al., PRL 119, 180509 (2017).

---

## qgate's Boundary: Automatic vs Opt-In

This is a critical architectural distinction:

| Component | What It Does | Automatic? |
|---|---|---|
| `QgateTranspiler.compile()` | Probe injection, chaotic padding, shot optimisation | ✅ **Automatic** |
| `validate_noise_scale_factor()` | Validates λ is a positive odd integer | ❌ **Opt-in** |
| `apply_uzdin_unitary_folding()` | Applies U→U·(U†·U)ⁿ circuit folding | ❌ **Opt-in** |

The transpiler pipeline **never performs noise amplification**.  It handles
circuit preparation (probes + padding + shots).  ZNE is a separate concern
that the user invokes explicitly when needed.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    qgate.transpiler module                       │
 │                                                                  │
 │  ┌───────────────────────────────┐  ┌─────────────────────────┐ │
 │  │ QgateTranspiler (AUTOMATIC)   │  │ Uzdin Utilities (OPT-IN)│ │
 │  │                               │  │                         │ │
 │  │ • compile()                   │  │ • validate_noise_scale_ │ │
 │  │   → probe injection           │  │   factor()              │ │
 │  │   → conditional padding       │  │ • apply_uzdin_unitary_  │ │
 │  │   → shot optimisation         │  │   folding()             │ │
 │  │                               │  │                         │ │
 │  │ NO noise amplification.       │  │ For ZNE experiments     │ │
 │  │ NO circuit folding.           │  │ that need circuit-level  │ │
 │  │ NO noise model construction.  │  │ noise scaling.          │ │
 │  └───────────────────────────────┘  └─────────────────────────┘ │
 └─────────────────────────────────────────────────────────────────┘
```

---

## API Reference

### `validate_noise_scale_factor(scale_factor)`

Validates that a noise-amplification scale factor is a positive odd integer.

```python
from qgate import validate_noise_scale_factor

validate_noise_scale_factor(1)   # ✅ OK
validate_noise_scale_factor(3)   # ✅ OK
validate_noise_scale_factor(5)   # ✅ OK
validate_noise_scale_factor(7)   # ✅ OK

validate_noise_scale_factor(2)   # ❌ ValueError — even
validate_noise_scale_factor(1.5) # ❌ ValueError — not integer
validate_noise_scale_factor(0)   # ❌ ValueError — not positive
validate_noise_scale_factor(-3)  # ❌ ValueError — not positive
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `scale_factor` | `int` | The noise amplification factor to validate |

**Raises:** `ValueError` if not a positive odd integer.

**Error messages** include an explanation of the Uzdin rule and why
non-odd factors are forbidden, to help users avoid the digital folding trap.

---

### `apply_uzdin_unitary_folding(circuit, scale_factor)`

Apply noise amplification via strict odd-integer unitary folding.

Each gate $U$ in the circuit is replaced by:

$$U \to U \cdot (U^\dagger \cdot U)^{(\lambda - 1) / 2}$$

**Non-invertible operations** (measurements, barriers, resets, delays) are
passed through unchanged — they cannot be folded.

```python
from qiskit import QuantumCircuit
from qgate import apply_uzdin_unitary_folding

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 3× folding: each gate becomes U·U†·U
folded_3 = apply_uzdin_unitary_folding(qc, scale_factor=3)
assert folded_3.size() == 6   # 2 gates × 3 copies each

# 5× folding: each gate becomes U·U†·U·U†·U
folded_5 = apply_uzdin_unitary_folding(qc, scale_factor=5)
assert folded_5.size() == 10  # 2 gates × 5 copies each

# λ=1 returns a copy (no folding)
folded_1 = apply_uzdin_unitary_folding(qc, scale_factor=1)
assert folded_1.size() == 2
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `circuit` | `QuantumCircuit` | The Qiskit circuit to fold |
| `scale_factor` | `int` | Positive odd integer (validated internally) |

**Returns:** A new `QuantumCircuit` with folded gates.

**Raises:**

- `ValueError` if `scale_factor` is not a positive odd integer
- `TypeError` if `circuit` is not a `QuantumCircuit`

### Handling Measurements

Circuits with measurements are handled correctly — measurements are
passed through without folding:

```python
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
# H and CX are each folded 3×, measure is passed through
# 2 gates × 3 + 2 measures = 8 operations
```

The set of non-foldable operations:

| Operation | Why Not Foldable |
|---|---|
| `measure` | Irreversible — no inverse |
| `barrier` | Scheduling hint — no gate content |
| `reset` | Irreversible — projects to $\lvert 0 \rangle$ |
| `delay` | Timing instruction — no unitary |

---

## Example: ZNE with Uzdin Folding

Here's a complete example of using the Uzdin utilities for a ZNE
experiment:

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qgate import validate_noise_scale_factor, apply_uzdin_unitary_folding

# Define the circuit
qc = QuantumCircuit(4)
qc.h(range(4))
for i in range(3):
    qc.cx(i, i + 1)
qc.measure_all()

# ZNE scale factors — MUST be odd integers
scale_factors = [1, 3, 5, 7, 9]

# Validate all factors upfront
for lam in scale_factors:
    validate_noise_scale_factor(lam)  # raises if invalid

# Run at each noise level
results = {}
backend = AerSimulator()
for lam in scale_factors:
    folded = apply_uzdin_unitary_folding(qc, scale_factor=lam)
    job = backend.run(folded, shots=8192)
    counts = job.result().get_counts()
    # ... compute observable from counts
    results[lam] = observable_value

# Extrapolate to λ=0 (your choice of polynomial/exponential fit)
# ...
```

!!! tip "Combine with the transpiler"
    You can fold first, then compile through the transpiler:

    ```python
    folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
    result = transpiler.compile(folded, base_shots=4096)
    # result.circuit has probes injected into the folded circuit
    ```

---

## Background: The VNS Experiment Fix

The Uzdin utilities were added after an audit of the codebase revealed
that the VNS (Variational Noise Scaling) experiment in
`simulations/vns_compatibility/run_vns_experiment.py` used non-compliant
scale factors:

```python
# ❌ BEFORE (Uzdin violation)
LAMBDA_VALUES = [1.0, 1.5, 2.0, 3.0, 5.0]
#                      ^^^  ^^^  non-odd factors

# ✅ AFTER (Uzdin compliant)
LAMBDA_VALUES = [1, 3, 5, 7, 9]
```

The fix also refactored the noise model construction:

- `build_noise_model(lam)` — public, validates odd integer via Uzdin guard
- `_build_noise_model_raw(lam)` — internal, accepts float for drift simulation
- `build_drifted_noise_model(lam, rng)` — validates base λ, applies stochastic
  TLS drift (fractional result is physically correct for environment modelling)

---

## See Also

- [QgateTranspiler](transpiler.md) — the automatic compilation pipeline
- [ML-Augmented Mitigation](ml-mitigation.md) — three-tier architecture overview
- [API Reference](../api.md) — auto-generated docstrings
