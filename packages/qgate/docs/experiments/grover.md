---
description: >-
  Grover search vs TSVF-Grover experiment on IBM Fez hardware. 7.3× success probability
  improvement at iteration 4 using trajectory filtering with chaotic perturbation and
  parity probe ancilla post-selection.
keywords: Grover search, TSVF, IBM Fez, quantum trajectory filtering, 7.3x improvement, post-selection, NISQ, quantum search algorithm
faq:
  - q: How much does TSVF improve Grover search on IBM hardware?
    a: At iteration 4 on IBM Fez, TSVF-Grover achieves 61% success probability compared to 8.3% for standard Grover — a 7.3× improvement. The advantage grows as standard Grover's signal degrades at higher iteration counts.
  - q: At what point does TSVF outperform standard Grover?
    a: The crossover occurs at iteration 3. At low iterations (1-2), standard Grover still has strong signal and TSVF adds overhead. At iteration 3+, hardware noise degrades the amplitude pattern, and TSVF post-selection filters for surviving trajectories.
---

# Grover vs TSVF-Grover (IBM Fez)

> **Patent notice:** US Patent App. Nos. 63/983,831 & 63/989,632 | Israeli Patent App. No. 326915

## Objective

Test whether TSVF trajectory filtering can rescue Grover search from
hardware noise degradation at higher iteration counts, where standard
Grover's success probability collapses on real NISQ devices.

## Setup

| Parameter | Value |
|---|---|
| **Backend** | IBM Fez (156 qubits) |
| **Algorithm** | 5-qubit Grover search (marked state $\lvert10101\rangle$) |
| **Iterations** | 1–10 |
| **Shots** | 8,192 per configuration |
| **TSVF variant** | Chaotic perturbation + parity probe ancilla |
| **Date** | February 2026 |

## TSVF Approach

1. **Standard Grover:** Oracle + Diffusion operator, iterated 1–10 times
2. **TSVF-Grover:** Same + chaotic layer (random Rz/Ry/CX) + ancilla
   parity probe (controlled rotations rewarding marked-state bit pattern)
3. **Post-selection:** Accept only shots where ancilla measures $\lvert1\rangle$

## Key Results

| Iteration | P(success) std | P(success) TSVF | Ratio | Accept% |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.2131 | 0.1953 | 0.92× | 29.1% |
| 2 | 0.4329 | 0.3618 | 0.84× | 31.4% |
| 3 | 0.1801 | 0.4764 | 2.65× | 28.7% |
| **4** | **0.0830** | **0.6105** | **7.36×** | **25.3%** |
| 5 | 0.0552 | 0.4318 | 7.82× | 22.8% |

!!! tip "Headline: 7.3× TSVF advantage at iteration 4"
    At low iterations (1–2), standard Grover still has strong signal and TSVF
    adds overhead. At iteration 3+, hardware noise degrades the Grover
    amplitude pattern, and TSVF post-selection filters for trajectories where
    the marked-state amplitude survived — yielding dramatic improvement.

## Analysis

- **Crossover point** at iteration 3: TSVF begins outperforming standard Grover
- **Peak advantage** at iterations 4–5: Standard success drops below 10% while TSVF maintains 40–60%
- **Acceptance rate** stays ~25–30%, confirming the probe ancilla selects a meaningful subset

## Reproduction

=== "IBM Hardware"

    ```bash
    python simulations/grover_tsvf/run_grover_tsvf_experiment.py \
        --mode ibm --max-iter 10 --shots 8192
    ```

=== "Aer Simulator"

    ```bash
    python simulations/grover_tsvf/run_grover_tsvf_experiment.py \
        --mode aer --max-iter 10 --shots 8192
    ```

!!! note "Requirements"
    Requires `.secrets.json` with `ibmq_token` for IBM hardware runs.

## Using the qgate Adapter

```python
from qgate.adapters.grover_adapter import GroverTSVFAdapter

adapter = GroverTSVFAdapter(
    n_qubits=5,
    marked_state="10101",
    n_iterations=4,
)

# Build both standard and TSVF circuits
std_circuit = adapter.build_standard_circuit()
tsvf_circuit = adapter.build_tsvf_circuit()
```
