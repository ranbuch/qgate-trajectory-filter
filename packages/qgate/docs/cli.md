---
description: >-
  qgate CLI reference. Commands for running trajectory filters, validating configs,
  listing adapters, and exporting JSON schemas. Powered by Typer.
keywords: qgate CLI, command line interface, qgate run, qgate validate, quantum CLI tool, Typer CLI
---

# CLI Reference

qgate includes a command-line interface powered by [typer](https://typer.tiangolo.com).

## Commands

### `qgate version`

Print the installed version.

```bash
$ qgate version
qgate 0.5.0
```

### `qgate validate <config.json>`

Validate a GateConfig JSON file.

```bash
$ qgate validate config.json
✅  Config valid — variant=score_fusion, n_subsystems=4, shots=1024
```

### `qgate run <config.json>`

Run a trajectory filter.

```bash
$ qgate run config.json --adapter mock --seed 42
variant=score_fusion  shots=1024  accepted=950  P_acc=0.9277  TTS=1.08

$ qgate run config.json --adapter mock --output results.jsonl
📝  Logged to results.jsonl

$ qgate run config.json --adapter mock --error-rate 0.1 --quiet
variant=score_fusion  shots=1024  accepted=876  P_acc=0.8555  TTS=1.17
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--adapter`, `-a` | `mock` | Adapter: `mock`, `qiskit`, `cirq`, `pennylane` |
| `--output`, `-o` | None | Log file (`.jsonl`, `.csv`, `.parquet`) |
| `--seed`, `-s` | None | Random seed (mock adapter) |
| `--error-rate`, `-e` | None | Override mock adapter error rate (0.0–1.0) |
| `--verbose`, `-v` | off | Enable DEBUG logging |
| `--quiet`, `-q` | off | Suppress INFO logging (WARNING only) |

### `qgate adapters`

List all registered adapters (built-in and entry-point plugins).

```bash
$ qgate adapters
mock         MockAdapter (built-in)
qiskit       QiskitAdapter
cirq         CirqAdapter (stub)
pennylane    PennyLaneAdapter (stub)
```

### `qgate schema`

Print the JSON Schema for `GateConfig`.

```bash
$ qgate schema | python -m json.tool
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "GateConfig",
  ...
}
```
