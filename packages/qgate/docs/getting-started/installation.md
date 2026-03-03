---
description: >-
  Install qgate from PyPI or source. Core package with numpy/pydantic/typer,
  optional extras for Qiskit, Cirq, PennyLane, CSV, and Parquet logging.
  Requires Python 3.9+.
keywords: qgate installation, pip install qgate, quantum computing setup, qiskit adapter, Python quantum library
---

# Installation

## From source (recommended during development)

```bash
git clone https://github.com/ranbuch/qgate-trajectory-filter.git
cd qgate-trajectory-filter/packages/qgate
pip install -e ".[dev]"
```

## Core only

```bash
pip install qgate
```

The core package depends only on **numpy**, **pydantic**, and **typer**.
pandas is *not* required unless you use CSV or Parquet logging.

## With backend extras

```bash
pip install qgate[csv]          # + pandas (CSV logging)
pip install qgate[parquet]      # + pandas + pyarrow (Parquet logging)
pip install qgate[qiskit]       # IBM Qiskit adapter
pip install qgate[cirq]         # Google Cirq adapter (stub)
pip install qgate[pennylane]    # PennyLane adapter (stub)
pip install qgate[all]          # Everything
```

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- pydantic ≥ 2.0
- typer ≥ 0.9

### Optional

- pandas ≥ 1.5 (for CSV/Parquet logging)
- pyarrow ≥ 14.0 (for Parquet logging)
- qiskit ≥ 1.0 (for IBM hardware)
