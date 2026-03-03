---
description: >-
  Score fusion in qgate. Alpha-weighted combination of low-frequency and high-frequency
  monitoring scores for continuous quality assessment of quantum computation trajectories.
keywords: score fusion, alpha-weighted scoring, LF HF monitoring, quantum trajectory quality, continuous scoring, quantum error assessment
---

# Fusion Scoring

Score fusion combines low-frequency and high-frequency monitoring scores
into a single continuous metric.

## Formula

$$S_{\text{combined}} = \alpha \cdot \bar{S}_{\text{LF}} + (1 - \alpha) \cdot \bar{S}_{\text{HF}}$$

Where:

- $\bar{S}_{\text{LF}}$ = mean subsystem pass-rate over LF cycles
- $\bar{S}_{\text{HF}}$ = mean subsystem pass-rate over HF cycles
- $\alpha$ = weight for the LF component (0 ≤ α ≤ 1)

## Configuration

```python
from qgate import FusionConfig

fusion = FusionConfig(
    alpha=0.5,         # Equal LF/HF weighting
    threshold=0.65,    # Accept if combined ≥ 0.65
    hf_cycles=None,    # Default: every cycle
    lf_cycles=None,    # Default: every 2nd cycle (0, 2, 4, ...)
)
```

## Why Score Fusion?

On real IBM hardware, **logical (hard) fusion** with per-frequency
thresholds is extremely sensitive to HF noise, causing 100% false-reject
at moderate noise levels.  Score fusion provides a **soft decision
boundary** that absorbs these spikes.

| Method | γ ≤ 5.0 | γ = 10.0 |
|---|---|---|
| Logical fusion | 100% false reject | ~50% accept |
| Score fusion | **Robust** | **Robust** |
