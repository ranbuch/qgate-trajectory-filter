---
description: >-
  QPE vs TSVF-QPE phase estimation experiment on IBM Fez. Demonstrates that TSVF trajectory
  filtering is incompatible with phase-coherence-encoded algorithms — perturbation destroys
  the inverse QFT interference pattern. A negative but scientifically valuable result.
keywords: QPE, quantum phase estimation, TSVF, IBM Fez, phase coherence, inverse QFT, negative result, quantum algorithm limitations
faq:
  - q: Why doesn't TSVF work for QPE?
    a: QPE encodes its answer in phase coherence between precision qubits. Any perturbation — even mild rotations — disrupts these phase relationships, causing the inverse QFT to produce a diffuse distribution instead of a sharp peak. Post-selection cannot recover destroyed phase information.
  - q: Does standard QPE work on IBM Fez hardware?
    a: Yes, standard QPE correctly identifies the eigenphase φ ≈ 1/3 at all precision levels (t=3 to t=7) on IBM Fez, even at circuit depth 411. The hardware preserves phase structure well enough without trajectory filtering.
---

# QPE vs TSVF-QPE Phase Estimation (IBM Fez)

> **Patent notice:** US Patent App. Nos. 63/983,831 & 63/989,632 | Israeli Patent App. No. 326915

## Objective

Test whether TSVF trajectory filtering can "anchor" the phase estimate in
Quantum Phase Estimation, keeping a sharp probability spike on the correct
phase binary fraction despite hardware noise as precision qubits increase.

## Setup

| Parameter | Value |
|---|---|
| **Backend** | IBM Fez (156 qubits) |
| **Algorithm** | QPE for $U = R_z(2\pi\phi)$ with eigenphase $\phi = 1/3$ |
| **Precision qubits** | t = 3–7 |
| **Total qubits** | t + 1 (eigenstate) + 1 (ancilla) = 5–9 |
| **Shots** | 8,192 per configuration |
| **TSVF variant** | Mild perturbation + phase probe ancilla |
| **Date** | March 2026 |

## Why φ = 1/3?

The eigenphase $\phi = 1/3 = 0.\overline{01}$ in binary is irrational —
it cannot be exactly represented in any finite binary fraction. This makes
it a good stress test: even a perfect QPE circuit will have inherent
approximation error that shrinks as $2^{-t}$, and hardware noise
compounds on top of that.

## TSVF Approach

1. **Standard QPE:** Hadamard on precision register → controlled-$U^{2^k}$
   kicks → inverse QFT → measure precision qubits
2. **TSVF-QPE:** Same + mild chaotic perturbation (Rz/Ry with scale
   $\pi/(6\sqrt{t})$, sparse CZ ring) + ancilla phase probe
   (2-controlled-Ry gates rewarding correct binary fraction bits)
3. **Post-selection:** Accept only shots where ancilla measures $\lvert1\rangle$

## Key Results

| t | Fid(std) | Fid(TSVF) | Err(std) | Err(TSVF) | Ent(std) | Ent(TSVF) | Accept% |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 3 | **0.582** | 0.064 | **0.105** | 0.326 | **1.97** | 2.48 | 35.8% |
| 4 | **0.551** | 0.015 | **0.095** | 0.267 | **2.49** | 3.71 | 32.0% |
| 5 | **0.369** | 0.035 | **0.148** | 0.284 | **3.68** | 4.76 | 50.5% |
| 6 | **0.343** | 0.012 | **0.110** | 0.261 | **4.05** | 5.87 | 60.1% |
| 7 | **0.157** | 0.008 | **0.157** | 0.245 | **5.65** | 6.83 | 49.6% |

### Phase Identification

| t | Correct bits | Std best | ✓ | TSVF best | ✓ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 3 | `011` | `011` | :material-check: | `110` | :material-close: |
| 4 | `0101` | `0101` | :material-check: | `0010` | :material-close: |
| 5 | `01011` | `01011` | :material-check: | `11010` | :material-close: |
| 6 | `010101` | `010101` | :material-check: | `000010` | :material-close: |
| 7 | `0101011` | `0101011` | :material-check: | `0001110` | :material-close: |

!!! failure "Result: TSVF does NOT help QPE"
    Standard QPE correctly identifies φ ≈ 1/3 at **all precision levels** on
    IBM Fez — the hardware is good enough to preserve the phase structure even
    at depth 411 (t=7). The TSVF perturbation, even at the mild scale of
    $\pi/(6\sqrt{t})$, destroys the delicate phase coherence that the inverse
    QFT relies on, producing near-uniform random output.

## Why TSVF Fails for QPE

QPE encodes its answer in the **phase coherence** of the precision register
after the inverse QFT. The controlled-$U^{2^k}$ gates establish precise
phase relationships between qubits, and the inverse QFT converts these into
a probability peak at the correct binary fraction.

Any unitary perturbation on the precision register — even small rotations —
disrupts these phase relationships. The inverse QFT then produces a diffuse
distribution instead of a sharp peak. Post-selection on the ancilla cannot
recover the destroyed phase information because it was lost before
measurement.

!!! abstract "Lesson Learned"
    TSVF trajectory filtering is effective for **amplitude-encoded** algorithms
    but fundamentally incompatible with **phase-coherence-encoded** algorithms.
    This is not a failure of the implementation but an intrinsic limitation of
    the perturbation-based approach.

## Reproduction

=== "IBM Hardware"

    ```bash
    python simulations/qpe_tsvf/run_qpe_tsvf_experiment.py \
        --mode ibm --max-precision 7 --shots 8192
    ```

=== "Aer Simulator"

    ```bash
    python simulations/qpe_tsvf/run_qpe_tsvf_experiment.py \
        --mode aer --max-precision 7 --shots 8192
    ```

!!! note "Requirements"
    Requires `.secrets.json` with `ibmq_token` for IBM hardware runs.

## Using the qgate Adapter

```python
from qgate.adapters.qpe_adapter import QPETSVFAdapter

adapter = QPETSVFAdapter(
    eigenphase=1/3,
    n_precision_qubits=7,
)

# Build both standard and TSVF circuits
std_circuit = adapter.build_standard_circuit()
tsvf_circuit = adapter.build_tsvf_circuit()
```
