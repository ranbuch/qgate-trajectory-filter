---
description: >-
  Qiskit adapter for qgate. Dynamic circuits with Bell-pair subsystems, scramble layers,
  and ancilla-based mid-circuit Z-parity measurements. IBM Quantum hardware integration.
keywords: Qiskit adapter, IBM Quantum, dynamic circuits, Bell pair, mid-circuit measurement, quantum hardware, qgate Qiskit
---

# Qiskit Adapter

The `QiskitAdapter` builds dynamic circuits with Bell-pair subsystems,
scramble layers, and ancilla-based mid-circuit Z-parity measurements.

## Installation

```bash
pip install qgate[qiskit]
```

## Usage

```python
from qgate import GateConfig, TrajectoryFilter
from qgate.adapters.qiskit_adapter import QiskitAdapter

config = GateConfig(n_subsystems=3, n_cycles=2, shots=500)
adapter = QiskitAdapter(scramble_depth=1)
tf = TrajectoryFilter(config, adapter)
result = tf.run()
```

## With IBM Quantum Hardware

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.least_busy(min_num_qubits=20, simulator=False)

adapter = QiskitAdapter(backend=backend, optimization_level=2)
```

## Circuit Layout

- **Data qubits:** `0 .. 2N-1` (Bell pairs: `[0,1]`, `[2,3]`, …)
- **Ancilla qubits:** `2N .. 3N-1` (one per pair)
- **Classical registers:** One per monitoring cycle, width N
