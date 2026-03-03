---
description: >-
  Conditioning strategies in qgate: global conditioning, hierarchical k-of-N conditioning,
  and continuous score fusion. Comparison of scaling behavior and acceptance rates.
keywords: conditioning strategies, global conditioning, hierarchical k-of-N, score fusion, quantum post-selection, acceptance probability, quantum error suppression
---

# Conditioning Strategies

qgate supports three conditioning strategies, configured via the
`variant` field in `GateConfig`.

## Global Conditioning

All N subsystems must pass all W monitoring cycles.

$$P_{\text{accept}}^{\text{global}} = \prod_{w=1}^{W} \prod_{i=1}^{N} p_i^{(w)}$$

**Limitation:** Exponential decay with N — unusable at N ≥ 2 under noise.

```python
config = GateConfig(variant="global")
```

## Hierarchical k-of-N

Accept if at least ⌈k·N⌉ subsystems pass each cycle.

$$P_{\text{accept}}^{\text{hier}} = \prod_{w=1}^{W} P\!\left(\sum_{i=1}^{N} X_i^{(w)} \ge \lceil k \cdot N \rceil\right)$$

**Advantage:** O(1) scaling — maintains high acceptance from N = 1 to N = 64.

```python
config = GateConfig(variant="hierarchical", k_fraction=0.9)
```

## Score Fusion

Continuous metric combining LF and HF scores:

$$S_{\text{combined}} = \alpha \cdot \bar{S}_{\text{LF}} + (1 - \alpha) \cdot \bar{S}_{\text{HF}}$$

Accept if $S_{\text{combined}} \ge \theta$.

**Advantage:** Soft decision boundary absorbs noise spikes.

```python
from qgate import GateConfig, FusionConfig

config = GateConfig(
    variant="score_fusion",
    fusion=FusionConfig(alpha=0.5, threshold=0.65),
)
```
