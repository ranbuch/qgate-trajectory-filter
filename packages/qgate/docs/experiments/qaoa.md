---
description: >-
  QAOA vs TSVF-QAOA MaxCut experiment on IBM Torino hardware. 1.88× approximation ratio
  improvement at p=1 using trajectory filtering for shallow variational quantum circuits.
keywords: QAOA, MaxCut, TSVF, IBM Torino, quantum trajectory filtering, 1.88x improvement, variational quantum algorithm, NISQ
faq:
  - q: How much does TSVF improve QAOA MaxCut on IBM hardware?
    a: At p=1 on IBM Torino, TSVF-QAOA achieves an approximation ratio of 0.80 compared to 0.43 for standard QAOA — a 1.88× improvement. The advantage is strongest at shallow circuit depths.
  - q: Why does TSVF help QAOA most at p=1?
    a: At p=1 (shallowest depth), hardware noise most severely degrades the single QAOA layer. TSVF post-selection filters for high-quality trajectories. At higher p, the variational ansatz has enough expressivity to partially self-correct.
---

# QAOA vs TSVF-QAOA MaxCut (IBM Torino)

> **Patent notice:** US Patent App. Nos. 63/983,831 & 63/989,632 | Israeli Patent App. No. 326915

## Objective

Test whether TSVF trajectory filtering improves QAOA MaxCut performance
on real hardware, particularly at shallow circuit depths (low p) where
hardware noise has the most severe impact on variational quality.

## Setup

| Parameter | Value |
|---|---|
| **Backend** | IBM Torino (133 qubits) |
| **Algorithm** | QAOA for MaxCut on a 6-node random graph |
| **Layers** | p = 1–5 |
| **Shots** | 2,000 per configuration |
| **TSVF variant** | Chaotic perturbation + cut-quality probe ancilla |
| **Date** | February 2026 |

## TSVF Approach

1. **Standard QAOA:** Cost layer (ZZ interactions from graph edges) +
   Mixer layer (Rx rotations), repeated p times
2. **TSVF-QAOA:** Same + chaotic perturbation + ancilla probe that
   rewards bitstrings with high cut fractions via controlled-Ry gates
3. **Post-selection:** Accept only shots where ancilla measures $\lvert1\rangle$

## Key Results

| p (layers) | AR std | AR TSVF | Ratio | Accept% |
|:-:|:-:|:-:|:-:|:-:|
| **1** | **0.4268** | **0.8029** | **1.88×** | **33.5%** |
| 2 | 0.7036 | 0.7024 | 1.00× | 32.0% |
| 3 | 0.6975 | 0.6987 | 1.00× | 35.2% |
| 4 | 0.6841 | 0.6912 | 1.01× | 34.8% |
| 5 | 0.6753 | 0.6802 | 1.01× | 36.1% |

!!! tip "Headline: 1.88× TSVF advantage at p=1"
    At p=1 (shallowest depth), hardware noise most severely degrades the
    single QAOA layer. TSVF post-selection nearly doubles the approximation
    ratio. At higher p, the variational ansatz has enough expressivity to
    partially self-correct, so the TSVF advantage narrows.

## Analysis

- **Strongest advantage** at p=1: TSVF rescues nearly 2× the approximation quality
- **Diminishing returns** at p ≥ 2: deeper circuits self-correct, reducing TSVF's marginal benefit
- **Consistent acceptance** at ~33–36%, showing stable post-selection regardless of depth

## Reproduction

=== "IBM Hardware"

    ```bash
    python simulations/qaoa_tsvf/run_qaoa_tsvf_experiment.py \
        --mode ibm --max-layers 5 --shots 2000
    ```

=== "Aer Simulator"

    ```bash
    python simulations/qaoa_tsvf/run_qaoa_tsvf_experiment.py \
        --mode aer --max-layers 5 --shots 4000
    ```

!!! note "Requirements"
    Requires `.secrets.json` with `ibmq_token` for IBM hardware runs.

## Using the qgate Adapter

```python
from qgate.adapters.qaoa_adapter import QAOATSVFAdapter

adapter = QAOATSVFAdapter(
    n_qubits=6,
    edges=[(0,1), (1,2), (2,3), (3,4), (4,5), (0,5)],
    p_layers=1,
)

# Build both standard and TSVF circuits
std_circuit = adapter.build_standard_circuit()
tsvf_circuit = adapter.build_tsvf_circuit()
```
