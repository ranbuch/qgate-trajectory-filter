---
description: >-
  qgate adapter system overview. Pluggable backends for Qiskit, Cirq, PennyLane,
  and algorithm-specific TSVF adapters for Grover, QAOA, VQE, and QPE.
  BaseAdapter protocol and custom adapter development guide.
keywords: qgate adapters, Qiskit adapter, Cirq adapter, PennyLane adapter, TSVF adapter, quantum framework adapter, BaseAdapter, plugin architecture
---

# Adapters Overview

qgate uses an **adapter pattern** to support multiple quantum frameworks.
Each adapter translates between a framework's native types and qgate's
internal `ParityOutcome` representation.

## Available Adapters

### Core Framework Adapters

| Adapter | Status | Install |
|---|---|---|
| `MockAdapter` | ✅ Full | Built-in |
| `QiskitAdapter` | ✅ Full | `pip install qgate[qiskit]` |
| `CirqAdapter` | 🔧 Stub | `pip install qgate[cirq]` |
| `PennyLaneAdapter` | 🔧 Stub | `pip install qgate[pennylane]` |

### Algorithm-Specific TSVF Adapters

These adapters build both standard and TSVF-variant circuits for canonical
quantum algorithms, enabling trajectory filtering experiments on real hardware.

| Adapter | Algorithm | Entry Point | IBM Validated |
|---|---|---|---|
| `GroverTSVFAdapter` | Grover search | `grover_tsvf` | ✅ IBM Fez — **7.3× advantage** at iter 4 |
| `QAOATSVFAdapter` | QAOA MaxCut | `qaoa_tsvf` | ✅ IBM Torino — **1.88× advantage** at p=1 |
| `VQETSVFAdapter` | VQE (TFIM) | `vqe_tsvf` | ✅ IBM Fez — barren plateau avoidance at L=3 |
| `QPETSVFAdapter` | QPE phase est. | `qpe_tsvf` | ✅ IBM Fez — phase coherence study |

## The BaseAdapter Protocol

All adapters implement three methods:

```python
class BaseAdapter(ABC):
    def build_circuit(self, n_subsystems, n_cycles, **kwargs) -> Any: ...
    def run(self, circuit, shots, **kwargs) -> Any: ...
    def parse_results(self, raw_results, n_subsystems, n_cycles) -> List[ParityOutcome]: ...
```

Plus a convenience method:

```python
    def build_and_run(self, n_subsystems, n_cycles, shots, ...) -> List[ParityOutcome]: ...
```

## Writing a Custom Adapter

```python
from qgate.adapters.base import BaseAdapter
from qgate.conditioning import ParityOutcome

class MyAdapter(BaseAdapter):
    def build_circuit(self, n_subsystems, n_cycles, **kwargs):
        # Build your circuit here
        ...

    def run(self, circuit, shots, **kwargs):
        # Execute the circuit
        ...

    def parse_results(self, raw_results, n_subsystems, n_cycles):
        # Convert to List[ParityOutcome]
        ...
```
