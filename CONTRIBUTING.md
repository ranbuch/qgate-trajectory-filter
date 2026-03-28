# Contributing to qgate

Thank you for your interest in contributing!

## License & IP

This project is released under the **QGATE Source Available Evaluation License v1.2**.
By submitting a pull request you agree that your contribution will be licensed
under the same terms. Please read the [LICENSE](LICENSE) before contributing.

The Software implements methods covered by **pending patent applications**
(US 63/983,831, US 63/989,632, IL 326915). Contributions that extend the
patented methods may need additional IP review.

## Development Setup

```bash
git clone https://github.com/ranbuch/qgate-trajectory-filter.git
cd qgate-trajectory-filter
python -m venv .venv && source .venv/bin/activate
pip install -e "packages/qgate[dev]"
```

## Running Tests

```bash
cd packages/qgate
pytest -v tests/             # 806 tests, ~10 s
pytest --cov=qgate tests/    # with coverage
```

## Linting & Type Checking

```bash
cd packages/qgate
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/qgate/ --ignore-missing-imports
```

## Pull Request Guidelines

1. **One feature per PR.** Keep changes focused and reviewable.
2. **Add tests.** Every new public function/class should have unit tests.
3. **Follow existing style.** The codebase uses `ruff` formatting.
4. **Update the CHANGELOG.** Add an entry under `[Unreleased]`.
5. **Docstrings.** All public functions need Google/NumPy-style docstrings.

## Reporting Issues

Use the GitHub issue tracker. Include:
- Python version (`python --version`)
- qgate version (`python -c "import qgate; print(qgate.__version__)"`)
- Full traceback (if applicable)

## Contact

For commercial licensing or IP inquiries: [ranbuch@gmail.com](mailto:ranbuch@gmail.com)
