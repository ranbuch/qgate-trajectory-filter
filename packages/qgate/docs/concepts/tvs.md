---
description: >-
  Trajectory Viability Score (TVS) — fuses Level-1 (I/Q) and Level-2 (binary)
  high-frequency telemetry with low-frequency drift scores via Kalman-style
  dynamic alpha weighting.  Normalised fusion feeds Stage-1 Galton percentile
  filtering, producing a surviving-shot mask and ML features for Stage-2
  regressors.  Fully vectorised NumPy — no Python for-loops.
keywords: trajectory viability score, TVS, HF LF fusion, I/Q readout, Level-1, Level-2, Kalman alpha, Galton filter, RBF, quantum telemetry, qgate, soft-decision decoding, hard-decision decoding, fusion normalisation
faq:
  - q: What is the Trajectory Viability Score (TVS)?
    a: The TVS fuses high-frequency (HF) and low-frequency (LF) telemetry into a single per-shot viability metric bounded in [0, 1], then applies Galton percentile filtering to reject outlier trajectories before Stage-2 ML regression.
  - q: What is the difference between Level-1 and Level-2 mode?
    a: Level-2 uses binary bit-string readout (hard-decision decoding) with static alpha fusion. Level-1 uses raw complex I/Q microwave samples (soft-decision decoding) with a per-shot dynamic alpha derived from measurement confidence.
  - q: Is the fusion score normalised?
    a: Yes. Both HF and LF scores are bounded [0, 1] and alpha is bounded [0, 1], so the convex combination is guaranteed to lie in [0, 1] by construction — no post-hoc rescaling is needed.
---

# Trajectory Viability Score (TVS)

!!! warning "Patent Pending — Confidential"
    US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915.
    CIP addendum — Level-1 I/Q trajectory viability scoring.
    **DO NOT PUSH. DO NOT PUBLISH.**

---

## Overview

The **Trajectory Viability Score (TVS)** module (`qgate.tvs`) is the
signal-processing front end of the ML-augmented mitigation pipeline.
It sits *between* the raw hardware readout and the Stage-2 ML regressors:

```
 Hardware readout        TVS pipeline             Stage-2 ML
 ════════════════        ═══════════════════       ═══════════
 Level-2 bits      ──▶  HF normalisation   ──▶   TelemetryMitigator
   or                        │                        │
 Level-1 I/Q             Fusion (α blend)        Corrected ⟨O⟩
                             │
                        Galton filter
                             │
                        surviving mask
                        + ml_features
```

The module is **fully vectorised** — every operation uses NumPy array
math with zero Python `for` loops — and processes 50 000+ shots in
sub-millisecond wall time.

---

## Hardware Modes

The caller selects the hardware abstraction level **dynamically** via
the `mode` parameter:

| Mode | Data Type | Decoding | Alpha Strategy |
|---|---|---|---|
| `level_2` | Binary (0/1) | Hard-decision | Static α (user-specified) |
| `level_1` | Complex (I + iQ) | Soft-decision | Per-shot dynamic α (Kalman-style) |

### Level-2 — Hard-Decision Decoding (Fallback)

Standard discriminated bit-string readout.  The HF array contains
integer 0s and 1s produced by the firmware's threshold discriminator.

- Bit `0` (no error detected) → `hf_score = 1.0`
- Bit `1` (error detected) → `hf_score = 0.0`

### Level-1 — Soft-Decision Decoding (Premium)

Raw microwave baseband readout.  The HF array contains complex numbers
$I + iQ$ from the digitiser **before** firmware discrimination.  This
preserves analog information that hard thresholding would discard.

Each I/Q point is scored by its proximity to a calibrated |0⟩ centroid
using a Gaussian Radial Basis Function (RBF):

$$d_i = \lvert \text{hf}[i] - z_0 \rvert$$

$$\text{hf\_score}[i] = \exp\!\left( -\frac{d_i^2}{\sigma^2} \right)$$

where $z_0$ is the `zero_centroid` and $\sigma^2$ is the `variance`
parameter.  Points near the centroid score ≈ 1; distant points score ≈ 0.

---

## Dynamic Alpha Fusion

The TVS fuses the HF viability score with a **low-frequency (LF)** drift
score (historical, already bounded [0, 1]) using an α-weighted blend:

$$\text{fusion}[i] = \alpha_i \cdot \text{hf\_score}[i] + (1 - \alpha_i) \cdot \text{lf\_score}[i]$$

### Level-2 Static Alpha

A single user-specified α (default 0.5) is broadcast to all shots:

$$\alpha_i = \alpha_{\text{static}} \quad \forall\, i$$

### Level-1 Dynamic Alpha (Kalman-Style)

Per-shot α is computed from the **confidence** of the HF measurement.
Confidence is maximal when `hf_score` is near 0 or 1 (unambiguous
discrimination), and minimal when near 0.5 (ambiguous — the I/Q point
lies on the decision boundary):

$$\text{confidence}[i] = 2 \cdot \lvert \text{hf\_score}[i] - 0.5 \rvert$$

$$\alpha_i = \alpha_{\min} + \text{confidence}[i] \cdot (\alpha_{\max} - \alpha_{\min})$$

This is a Kalman-style update: when HF is confident, it dominates the
fusion (high α); when HF is ambiguous, LF drift history fills the gap
(low α).

### Normalisation Guarantee

Both `hf_score` and `lf_score` are clamped to [0, 1], and α is
bounded in [0, 1].  The convex combination of [0, 1] values with
[0, 1] weights is **guaranteed** to lie in [0, 1] — no post-hoc
rescaling is required.

---

## Stage-1 Galton Filter

After fusion, a **percentile-based outlier rejection** step discards
the lowest-viability trajectories:

1. Compute the `drop_percentile`-th percentile of `fusion_scores`
   (default: 25th percentile).
2. Create a boolean mask: `True` where `fusion_score ≥ threshold`.
3. Shots below the threshold are rejected.

This is the same Galton-inspired filtering used throughout qgate — the
threshold adapts to the actual score distribution rather than relying
on a fixed cutoff.

### Adaptive Galton Schedule (Depth-Aware Rejection)

For deep circuits the fixed 25 % rejection is insufficient: depolarisation
noise accumulates with depth and the majority of shots thermalise,
overwhelming the ML mitigator during extrapolation.

`adaptive_galton_schedule` computes a **per-depth rejection percentile**
using a sigmoid ramp:

$$
p(d) = p_{\text{base}} + (p_{\text{max}} - p_{\text{base}}) \cdot \sigma\!\bigl(s \cdot (d / d_{\text{knee}} - 1)\bigr)
$$

where $\sigma(x) = 1/(1+e^{-x})$ and defaults are
$p_{\text{base}} = 25$, $p_{\text{max}} = 75$, $d_{\text{knee}} = 300$, $s = 3$.

The knee is set at $d = 300$, well past the training boundary ($d \le 100$),
so the schedule stays gentle during training and only ramps aggressively for
truly deep, thermalised circuits.

| Depth | Drop % | Oversample |
|------:|-------:|-----------:|
|    10 |  27.6 |  1.38×     |
|   100 |  31.0 |  1.45×     |
|   300 |  50.0 |  2.00×     |
|   500 |  69.0 |  3.23×     |
|  1000 |  75.0 |  4.00×     |

The **oversampling factor** $100/(100 - p(d))$ compensates for the
more aggressive rejection so that the effective surviving sample size
remains approximately constant across depths.

```python
from qgate.tvs import adaptive_galton_schedule
import numpy as np

depths = np.array([10, 50, 100, 500, 1000])
schedule = adaptive_galton_schedule(depths)
oversample = 100.0 / (100.0 - schedule)
```

---

## API

### `process_telemetry_batch`

The main entry point.  Takes raw HF data, LF scores, mode flag, and
calibration parameters; returns a dict with the surviving mask, fusion
scores, ML features, and diagnostics.

```python
from qgate.tvs import process_telemetry_batch
import numpy as np

# Level-2 (binary)
result = process_telemetry_batch(
    hf_data=np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0]),
    lf_scores=np.random.default_rng(42).uniform(0.4, 1.0, 10),
    mode="level_2",
    alpha=0.6,
)

# Level-1 (I/Q)
rng = np.random.default_rng(42)
result = process_telemetry_batch(
    hf_data=rng.normal(0.9, 0.1, 1000) + 1j * rng.normal(0.02, 0.05, 1000),
    lf_scores=rng.uniform(0.5, 1.0, 1000),
    mode="level_1",
    zero_centroid=0.95 + 0.02j,
    variance=0.15,
)
```

**Return dict keys:**

| Key | Type | Description |
|---|---|---|
| `surviving_mask` | `bool ndarray` | Per-shot filter pass/fail |
| `fusion_scores` | `float64 ndarray` | Fused TVS values ∈ [0, 1] |
| `ml_features` | `ndarray` | Surviving shots' raw data for Stage-2 |
| `hf_scores` | `float64 ndarray` | Normalised HF scores |
| `alpha_values` | `float64 ndarray` | Per-shot α values |
| `threshold` | `float` | Galton percentile threshold |
| `n_shots` | `int` | Total input shots |
| `n_surviving` | `int` | Shots that passed |
| `mode` | `str` | Hardware mode used |

### Helper functions

All helpers are public and can be used independently:

| Function | Purpose |
|---|---|
| `normalise_hf_level2` | Binary → viability score |
| `normalise_hf_level1` | I/Q RBF → viability score |
| `compute_alpha_static` | Constant α array |
| `compute_alpha_dynamic` | Confidence-weighted per-shot α |
| `fuse_scores` | Weighted HF + LF fusion |
| `galton_filter` | Percentile-based boolean mask |
| `adaptive_galton_schedule` | Depth-aware sigmoid rejection schedule |

---

## Integration with Pipeline

### With TelemetryMitigator (Level-2 path)

```python
from qgate.tvs import process_telemetry_batch
from qgate import TelemetryMitigator, MitigatorConfig

# Stage 0: TVS filtering
tvs = process_telemetry_batch(hf_bits, lf_drift, mode="level_2")

# Stage 1+2: ML mitigation on surviving shots only
mitigator = TelemetryMitigator(config=MitigatorConfig())
mitigator.calibrate(calibration_data)
result = mitigator.estimate(
    raw_energy=energy_from_surviving_shots,
    acceptance=tvs["n_surviving"] / tvs["n_shots"],
    variance=np.var(tvs["fusion_scores"]),
)
```

### With TelemetryCompressor (utility-scale)

```python
from qgate.tvs import process_telemetry_batch
from qgate import TelemetryCompressor

# TVS produces ml_features (surviving I/Q data)
tvs = process_telemetry_batch(iq_data, lf_drift, mode="level_1", ...)

# Compress 780-dim I/Q → 6 latent features
compressor = TelemetryCompressor(subsystem_map, retain_ratio=0.20)
compressor.fit(X_calibration, y_ideal)
X_compressed = compressor.transform(X_surviving)
```

---

## Performance

All operations are O(n) vectorised NumPy:

| Operation | Complexity | Loops |
|---|---|---|
| HF normalisation | O(n) | 0 |
| Alpha calculation | O(n) | 0 |
| Fusion | O(n) | 0 |
| Galton filter | O(n log n) | 0 (percentile uses partial sort) |

Measured: **50 000 shots in < 2 ms** on Apple M2 (single thread).
