---
description: >-
  How qgate's quantum trajectory filtering works. Bell-pair subsystems, mid-circuit
  Z-parity measurements, multi-rate monitoring pipeline, and the build-execute-filter cycle.
keywords: quantum trajectory filtering, Bell pair, Z-parity measurement, mid-circuit measurement, quantum error detection, multi-rate monitoring
---

# How It Works

qgate implements **quantum trajectory filtering** — a technique that uses
mid-circuit measurements to monitor subsystem fidelity and applies decision
rules to accept or reject quantum computation shots.

## The Pipeline

```
 ┌────────────┐    ┌─────────┐    ┌───────────┐    ┌───────────┐
 │   Adapter   │───▶│  Score   │───▶│ Threshold │───▶│  Accept/  │
 │ build + run │    │  Fusion  │    │   Gate    │    │  Reject   │
 └────────────┘    └─────────┘    └───────────┘    └───────────┘
       │                                                   │
       │              TrajectoryFilter                      │
       └───────────────────────────────────────────────────┘
```

1. **Adapter** builds a circuit with Bell-pair subsystems and mid-circuit
   parity measurements, then executes it on a backend.
2. **Scoring** computes per-shot LF/HF scores and fuses them with
   α-weighting.
3. **Thresholding** applies a static or dynamic threshold to determine
   accept/reject.
4. The result is a **FilterResult** with acceptance statistics, scores,
   and full config provenance.

## Bell-Pair Subsystems

Each subsystem is a 2-qubit Bell pair $(|00\rangle + |11\rangle)/\sqrt{2}$.
Under noise, the pair's parity may flip.  Mid-circuit Z-parity measurements
detect these flips without collapsing the computational state.

## Multi-Rate Monitoring

- **HF (high-frequency):** Parity measured every cycle
- **LF (low-frequency):** Parity measured every 2nd cycle

The two rates provide complementary signal: LF captures slow drift,
HF catches fast errors.
