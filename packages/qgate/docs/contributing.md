---
description: >-
  Contributing guide for qgate. Development setup, testing, code quality standards,
  project conventions, and areas for contribution including Cirq and PennyLane adapters.
keywords: qgate contributing, development setup, quantum computing open source, Python testing, ruff, mypy, conventional commits
---

# Contributing

Thank you for your interest in contributing to qgate!

## Development Setup

```bash
git clone https://github.com/ranbuch/qgate-trajectory-filter.git
cd qgate-trajectory-filter/packages/qgate
pip install -e ".[dev]"
```

The `[dev]` extra installs all test and quality dependencies:
pytest, pytest-cov, ruff, mypy, pandas, pyarrow, qiskit, etc.

## Running Tests

```bash
pytest -v tests/                 # 806 tests, ~10 s
pytest --cov=qgate tests/        # with coverage report
pytest tests/test_edge_cases.py  # run a single file
pytest -k "test_frozen"          # run by keyword
```

## Code Quality

```bash
ruff check src/ tests/           # lint
ruff format src/ tests/          # auto-format
mypy src/qgate/                  # type-check
```

All code must pass `ruff check` and `mypy --strict` before merge.

## Project Conventions

- **Immutable configs** — `GateConfig` and all sub-models are Pydantic
  `frozen=True` models.  Never mutate after construction.
- **NumPy for hot paths** — `ParityOutcome.parity_matrix` is an
  `np.ndarray`.  Avoid Python loops in scoring and decision code.
- **pandas is optional** — never import pandas at module level.  Use the
  lazy `_get_pandas()` helper in `run_logging.py`.
- **Structured logging** — use `logging.getLogger("qgate.<module>")`.
  Never print to stdout from library code.
- **Type annotations** — all public functions must have full type hints.

## Areas for Contribution

- **Cirq adapter** — Full implementation of `CirqAdapter`
- **PennyLane adapter** — Full implementation of `PennyLaneAdapter`
- **Additional conditioning strategies** — New decision rules beyond
  global / hierarchical / score-fusion
- **Benchmarks** — Performance comparison across backends and problem sizes
- **Documentation** — Examples, tutorials, API docs improvements
- **Property-based tests** — Hypothesis-powered fuzz testing

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/):

```
feat: add CirqAdapter with full Bell-pair circuit support
fix: prevent ndarray aliasing in QiskitAdapter.parse_results
docs: update CLI reference with --error-rate flag
test: add edge-case tests for empty outcomes
```

## Patent Notice

This package explores runtime trajectory filtering concepts from
US Patent Application Nos. 63/983,831 & 63/989,632 and Israeli Patent Application No. 326915.
The underlying invention is patent pending.

## License

QGATE Source Available Evaluation License v1.2 — see [License](license.md)
