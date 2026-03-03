---
description: >-
  IBM Quantum hardware experiments validating TSVF trajectory filtering for Grover search,
  QAOA MaxCut, VQE eigensolvers, and QPE phase estimation. Results from IBM Fez and IBM Torino
  processors showing up to 7.3× fidelity improvement.
keywords: IBM Quantum experiments, TSVF, trajectory filtering, Grover search, QAOA MaxCut, VQE barren plateau, QPE phase estimation, IBM Fez, IBM Torino, quantum hardware validation
faq:
  - q: What is TSVF trajectory filtering?
    a: TSVF (Two-State Vector Formalism) trajectory filtering injects a mild chaotic perturbation and uses an ancilla probe to create a post-selectable quality signal. Post-selection retains only high-fidelity execution trajectories.
  - q: Which quantum algorithms benefit from TSVF?
    a: Amplitude-encoded algorithms like Grover search (7.3× improvement), QAOA MaxCut (1.88× improvement), and VQE (barren plateau avoidance) benefit from TSVF. Phase-coherence algorithms like QPE do not benefit because perturbation destroys the phase structure.
  - q: What IBM hardware was used for validation?
    a: Experiments were run on IBM Fez (156 qubits) for Grover, VQE, and QPE, and IBM Torino (133 qubits) for QAOA, during February–March 2026.
---

# Hardware Experiments

> **Patent notice:** US Patent App. Nos. 63/983,831 & 63/989,632 | Israeli Patent App. No. 326915

## TSVF Algorithm Experiments (IBM Quantum, Feb–Mar 2026)

These experiments extend qgate's trajectory filtering beyond Bell-pair
conditioning to four canonical quantum algorithms. The **TSVF (Two-State
Vector Formalism)** approach injects a mild chaotic perturbation and uses an
ancilla-based probe to create a post-selectable quality signal — then
filters for high-fidelity execution trajectories.

### Methodology

```
Standard Algorithm:  H → Algorithm Gates → Measure
                          ↓
TSVF Variant:       H → Algorithm Gates → Chaotic Perturbation → Probe Ancilla → Measure
                                                                        ↓
                                                              Post-select on ancilla |1⟩
```

The chaotic perturbation is deliberately mild — small random rotations
scaled as $\pi / (c \cdot \sqrt{d})$ where $d$ is the circuit depth parameter.
The probe ancilla applies controlled rotations that reward bitstrings
consistent with the expected solution structure. Post-selection on the
ancilla measuring $|1\rangle$ retains only trajectories that survived both
the hardware noise and the perturbation.

---

## Results Summary

| Algorithm | Backend | Metric | Standard | TSVF | Advantage |
|---|---|---|---|---|---|
| [**Grover**](grover.md) (iter=4) | IBM Fez | Success probability | 0.0830 | **0.6105** | :material-fire: **7.3×** |
| [**QAOA**](qaoa.md) (p=1) | IBM Torino | Approximation ratio | 0.4268 | **0.8029** | :material-fire: **1.88×** |
| [**VQE**](vqe.md) (L=3) | IBM Fez | Energy gap to ground | 2.398 | **1.291** | :material-fire: **1.86×** closer |
| [**QPE**](qpe.md) (t=7) | IBM Fez | Phase fidelity | **0.1569** | 0.0076 | :material-close: N/A |

---

## Why TSVF Works for Some Algorithms but Not Others

The critical distinction is between **amplitude-encoded** and
**phase-coherence-encoded** information:

| Property | Grover / QAOA / VQE | QPE |
|---|---|---|
| Answer encoding | Amplitude pattern in computational basis | Phase coherence across precision register |
| Perturbation effect | Slightly scrambles amplitudes | Destroys inverse QFT interference |
| Post-selection recovers? | :material-check: Yes — filters trajectories where signal survives | :material-close: No — destroyed phase info is unrecoverable |
| Depth sensitivity | Moderate — noise accumulates gradually | High — single perturbation collapses peak |

!!! success "Amplitude-encoded algorithms (Grover, QAOA, VQE)"
    The answer is spread across computational basis state amplitudes. A mild
    perturbation slightly degrades these amplitudes, but the probe ancilla can
    detect which trajectories retained the signal. Post-selection filters out
    noise-corrupted paths, yielding a smaller but higher-fidelity sample.

!!! failure "Phase-coherence algorithms (QPE)"
    The answer is encoded in the *relative phases* between precision qubits,
    which the inverse QFT converts to a sharp probability peak. Any
    perturbation disrupts this phase coherence, and the inverse QFT produces
    a diffuse rather than peaked distribution. Post-selection cannot
    reconstruct the destroyed phase information.

---

## Reproduction

All experiments can be reproduced with a `.secrets.json` file containing
your IBM Quantum token:

```json
{
  "ibmq_token": "your-ibm-quantum-token-here"
}
```

See each experiment page for specific commands.
