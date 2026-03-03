---
description: >-
  Dynamic thresholding in qgate: rolling z-score, Galton adaptive thresholding with
  empirical quantile and robust z-score sub-modes, warmup periods, and telemetry for
  hardware drift adaptation.
keywords: dynamic thresholding, Galton adaptive threshold, rolling z-score, empirical quantile, robust z-score, hardware drift, quantum threshold adaptation
---

# Dynamic Thresholding

qgate supports three threshold adaptation strategies, configured via
`DynamicThresholdConfig.mode`:

| Mode | Description |
|---|---|
| `"fixed"` | Static threshold — no adaptation (default). |
| `"rolling_z"` | Rolling z-score on batch means (legacy). |
| `"galton"` | Distribution-aware adaptive gating on per-shot scores. |

---

## Rolling Z-Score Mode (`rolling_z`)

Adapts based on recent **batch-level** mean scores:

$$\theta_t = \text{clamp}\!\left(\mu_{\text{rolling}} + z \cdot \sigma_{\text{rolling}},\; \theta_{\min},\; \theta_{\max}\right)$$

Where:

- $\mu_{\text{rolling}}$ = mean of recent batch scores
- $\sigma_{\text{rolling}}$ = std-dev of recent batch scores
- $z$ = z-factor multiplier
- $\theta_{\min}, \theta_{\max}$ = floor / ceiling bounds

### Configuration

```python
from qgate import DynamicThresholdConfig

dt_config = DynamicThresholdConfig(
    mode="rolling_z",    # or: enabled=True (legacy shorthand)
    baseline=0.65,       # Starting threshold
    z_factor=1.5,        # Std-dev multiplier
    window_size=10,      # Rolling window (batches)
    min_threshold=0.3,   # Floor
    max_threshold=0.95,  # Ceiling
)
```

### Behaviour

- **Hardware behaving well** (high scores) → threshold tightens → better filtering
- **Hardware drifting** (lower scores) → threshold loosens → maintains reasonable acceptance
- **Few samples** (< 2 in window) → falls back to baseline

### Example

```python
from qgate import GateConfig, TrajectoryFilter, DynamicThresholdConfig
from qgate.adapters import MockAdapter

config = GateConfig(
    shots=200,
    dynamic_threshold=DynamicThresholdConfig(mode="rolling_z", z_factor=1.0),
)
tf = TrajectoryFilter(config, MockAdapter(seed=42))

for _ in range(10):
    result = tf.run()
    print(f"threshold={tf.current_threshold:.4f}  P_acc={result.acceptance_probability:.4f}")
```

---

## Galton Adaptive Mode (`galton`)

Distribution-aware gating inspired by diffusion / central-limit
principles.  Maintains a rolling window of **per-shot** combined scores
and computes a threshold targeting a stable acceptance fraction.

> **Note:** This is *distribution-aware quantile/z-score adaptive
> gating* — it does **not** assume exact Gaussianity.

### Sub-modes

**Quantile mode** (default, `use_quantile=True`)

$$\theta = Q_{1 - \alpha}(\text{window})$$

where $\alpha$ = `target_acceptance`.  This is the most robust option —
no distributional assumptions, naturally tracks hardware drift.

**Z-score mode** (`use_quantile=False`)

$$\theta = \mu + z_{\sigma} \cdot \sigma$$

When `robust_stats=True` (default):
- $\mu$ = median(window)
- $\sigma$ = MAD × 1.4826

When `robust_stats=False`:
- $\mu$ = mean(window)
- $\sigma$ = std(window)

### Warmup

While the window contains fewer than `min_window_size` scores the
threshold falls back to `baseline`.  This prevents noisy estimates from
too few observations.

### Configuration

```python
from qgate import GateConfig, DynamicThresholdConfig

config = GateConfig(
    shots=2000,
    dynamic_threshold=DynamicThresholdConfig(
        mode="galton",
        window_size=1000,        # per-shot rolling window capacity
        min_window_size=200,     # warmup: wait for 200 scores
        target_acceptance=0.05,  # keep ~5% acceptance fraction
        robust_stats=True,       # MAD-based sigma (outlier-resilient)
        use_quantile=True,       # empirical quantile (recommended)
        min_threshold=0.3,       # floor
        max_threshold=0.95,      # ceiling
    ),
)
```

### Example

```python
from qgate import GateConfig, TrajectoryFilter, DynamicThresholdConfig
from qgate.adapters import MockAdapter

config = GateConfig(
    shots=1000,
    variant="score_fusion",
    dynamic_threshold=DynamicThresholdConfig(
        mode="galton",
        window_size=1000,
        min_window_size=100,
        target_acceptance=0.10,
    ),
)
tf = TrajectoryFilter(config, MockAdapter(error_rate=0.1, seed=42))
result = tf.run()

print(f"Threshold: {tf.current_threshold:.4f}")
print(f"Accepted:  {result.accepted_shots}/{result.total_shots}")
print(f"P_accept:  {result.acceptance_probability:.4f}")

# Galton telemetry is in result.metadata["galton"]
galton = result.metadata.get("galton", {})
print(f"Rolling mean:     {galton.get('galton_rolling_mean', 'N/A')}")
print(f"Rolling sigma:    {galton.get('galton_rolling_sigma', 'N/A')}")
print(f"Window size:      {galton.get('galton_window_size_current', 'N/A')}")
print(f"Accept rate (win): {galton.get('galton_acceptance_rate_rolling', 'N/A')}")
```

### Telemetry

When galton mode is active, the following fields are added to
`FilterResult.metadata["galton"]`:

| Field | Description |
|---|---|
| `galton_rolling_mean` | Window centre (median or mean) |
| `galton_rolling_sigma` | Window dispersion (MAD-σ or std) |
| `galton_rolling_quantile` | Empirical quantile at 1 − target |
| `galton_effective_threshold` | Threshold actually used |
| `galton_window_size_current` | Scores currently in the window |
| `galton_acceptance_rate_rolling` | Fraction of window ≥ threshold |
| `galton_in_warmup` | True if still in warmup period |
