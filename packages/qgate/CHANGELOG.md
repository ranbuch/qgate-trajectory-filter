# Changelog

All notable changes to **qgate** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.6.0] — 2026-03-20

### Added

- **TVS (Trajectory Viability Score)** module (`qgate.tvs`) — HF/LF
  telemetry fusion with Kalman-style dynamic alpha and Stage-1 Galton
  percentile filtering.  Four hardware modes: `level_1` (I/Q RBF),
  `level_1_cluster` (K-Means + RBF), `level_2` (binary), `auto`
  (autonomous routing).
- **`adaptive_galton_schedule`** — sigmoid-based depth-aware rejection
  percentile schedule.  Raises drop rate from 25 % (shallow) to 75 %
  (deep) with knee at depth 300 (well past training boundary).
  Configurable knee, steepness, and oversampling factor.
  Patent pending — US 63/983,831 & 63/989,632, IL 326915.
- **`compute_iq_snr`** — I/Q separability metric (inter-centroid
  distance / pooled σ) used by autonomous pipeline routing.
- **`normalise_hf_level1_cluster`** — K-Means clustering + per-cluster
  RBF scoring for multi-modal I/Q readout distributions.
- **Autonomous pipeline routing** — `process_telemetry_batch` now
  auto-selects the optimal pipeline (Level-1/Level-1-cluster/Level-2)
  based on dtype detection and SNR thresholding when no `force_mode`
  is specified.
- **`force_mode` keyword-only parameter** — replaces the positional
  `mode` parameter for explicit pipeline selection.  `mode` is
  deprecated and emits `DeprecationWarning`.
- **`VALID_FORCE_MODES`** / **`VALID_MODES`** constants — public
  enumeration of supported pipeline identifiers.
- **TelemetryCompressor** — spatial pooling + Gini pruning for
  utility-scale telemetry reduction.
- **QgateTranspiler** — ML-aware circuit compiler with probe injection,
  Uzdin unitary folding, and mitigation-mode-specific depth/shot
  optimisation.
- **QgateSampler** — transparent `SamplerV2` drop-in replacement with
  autonomous probe injection and Galton-filtered result reconstruction.
- **PulseMitigator** — Level-1 IQ-level drift prediction with active
  cancellation.
- **Algorithm TSVF adapters** — Grover, QAOA, VQE, QPE adapters for
  TSVF-augmented algorithm execution.
- **Full-stack benchmark T1–T10** (49 metrics) — including T7 depth-
  scaling survival with adaptive Galton filtering (129–568× in-training,
  1.2–3.2× extrapolation to d = 1000).
- **806 unit tests** — all passing.

### Changed

- `process_telemetry_batch` signature: positional `mode` parameter
  deprecated in favour of keyword-only `force_mode`.  Legacy
  `mode='hybrid'` maps to auto-routing with a deprecation warning.

### Deprecated

- Positional `mode` parameter in `process_telemetry_batch` — use
  `force_mode` instead.  Will be removed in 0.7.0.

## [0.5.0] — 2026-02-22

### Added

- **Galton adaptive thresholding** — new `mode="galton"` in
  `DynamicThresholdConfig` provides distribution-aware, rolling-window
  gating inspired by diffusion / central-limit principles.  Targets a
  stable acceptance fraction under hardware drift.
  - **Quantile sub-mode** (default, `use_quantile=True`) — sets threshold
    at the empirical (1 − `target_acceptance`) quantile of the score window.
  - **Z-score sub-mode** (`use_quantile=False`) — estimates μ ± z·σ with
    optional robust statistics (median + MAD × 1.4826).
  - **Warmup period** — threshold falls back to `baseline` until
    `min_window_size` scores have been observed.
  - Galton telemetry (rolling mean, sigma, quantile, acceptance rate,
    window size) is logged to `FilterResult.metadata["galton"]`.
- **`GaltonAdaptiveThreshold`** class in `threshold.py` — standalone
  adaptive threshold with `observe()`, `observe_batch()`, `reset()`, and
  a `GaltonSnapshot` dataclass for telemetry.
- **`estimate_diffusion_width()`** utility — variance estimator with
  robust (MAD) and standard modes.
- **`ThresholdMode`** type alias — `Literal["fixed", "rolling_z", "galton"]`.
- **`TrajectoryFilter.galton_snapshot`** property — introspect the latest
  `GaltonSnapshot` when galton mode is active.
- **Config auto-enable** — setting `mode="galton"` or `mode="rolling_z"`
  automatically sets `enabled=True`.
- **27 new unit tests** for galton mode — quantile accuracy, robust stats
  under outliers, warmup, clamping, window management, integration with
  `TrajectoryFilter`, and `estimate_diffusion_width`.

### Changed

- `DynamicThresholdConfig` now accepts `mode`, `min_window_size`,
  `target_acceptance`, `robust_stats`, `use_quantile`, and `z_sigma`
  fields (all optional, backward-compatible defaults).
- `TrajectoryFilter.filter()` routes to `GaltonAdaptiveThreshold` when
  `mode="galton"`, feeding per-shot scores instead of batch means.

## [0.4.0] — 2026-02-19

### Added

- **Vectorised internals** — `ParityOutcome.parity_matrix` is now a
  `numpy.ndarray` (shape `(n_cycles, n_subsystems)`, dtype `int8`).
  All scoring, conditioning, and filtering hot-paths use NumPy instead of
  Python loops.  `score_batch()` stacks matrices into a single 3-D array
  for fully vectorised batch scoring.
- **`pass_rates` property** on `ParityOutcome` — returns per-cycle pass
  rates as an ndarray, used internally by scoring and decision functions.
- **`RunLogger` context-manager** — `with RunLogger("log.jsonl") as rl:`
  now supported; calls `close()` on exit.
- **Parquet buffered writes** — Parquet records accumulate in memory and
  flush on `close()`, avoiding per-shot file rewrites.
- **`TrajectoryFilter.__repr__`** — human-readable summary string.
- **CLI flags** — `--verbose`/`-v`, `--quiet`/`-q`, and
  `--error-rate`/`-e` (mock adapter error rate override).
- **stdlib `logging`** — `filter.py`, `threshold.py`, `run_logging.py`,
  `qiskit_adapter.py` emit structured log messages via
  `logging.getLogger("qgate.*")`.
- **22 new edge-case tests** — empty inputs, `n_subsystems=1`,
  ndarray coercion, frozen config, `filter_counts`, Parquet logging,
  CLI flags, Qiskit copy safety.
- **`csv` optional extra** — `pip install qgate[csv]` installs pandas
  for CSV logging without pulling the full `[all]` bundle.

### Changed

- **`GateConfig` is now frozen** — all Pydantic models use
  `ConfigDict(frozen=True, extra="forbid")`, preventing accidental
  mutation after construction.
- **pandas is no longer a core dependency** — moved to `[csv]` and
  `[parquet]` extras.  Lazy-imported at first use with a clear error
  message when absent.
- **`Dict[str, object]` → `Dict[str, Any]`** for `adapter_options` and
  `metadata` fields in `GateConfig` (fixes downstream typing issues).
- **Removed redundant `k_fraction` validator** — `Field(gt=0.0, le=1.0)`
  constraints are sufficient.
- **`compute_window_metric` de-duplicated** —
  `compat/monitors.py` now re-exports from `scoring.py` instead of
  carrying a copy.
- **`MultiRateMonitor` type hints** — `hf_scores` and `lf_scores` are
  typed as `list[float]`.
- **`threshold.py` docstring** — corrected formula to
  `rolling_mean + z_factor × rolling_std`.
- **MkDocs CDN** — replaced compromised `polyfill.io` with
  `cdnjs.cloudflare.com` for MathJax.

### Fixed

- **Qiskit adapter aliasing bug** — each shot from `parse_results()` now
  receives an independent ndarray copy of the parity matrix.  Previously,
  all shots sharing a bitstring shared the same object; mutating one
  silently corrupted the rest.
- **Empty outcomes** — `TrajectoryFilter.filter([])` returns a zeroed
  `FilterResult` instead of raising.
- **Unknown log extension** — `RunLogger("data.xyz")` now emits a
  warning and falls back to JSONL instead of silently misbehaving.

---

## [0.3.0] — 2025-07-07

### Added

- Multi-rate monitoring (`MultiRateMonitor`, HF / LF cycle partitions).
- Dynamic threshold adaptation with z-factor and rolling window.
- Probe-based early abort (`should_abort_batch`).
- Run logging to JSON-Lines, CSV, and Parquet (`RunLogger`).
- Deterministic SHA-256 run IDs for reproducibility.
- Adapter registry with `list_adapters()` / `load_adapter()`.
- Full Qiskit adapter with scramble layers and mid-circuit measurement.
- CLI (`qgate run`, `qgate validate`, `qgate schema`, `qgate adapters`, `qgate version`).
- MkDocs-based documentation site.
- Comprehensive test suite (152 tests).

### Changed

- Flat module layout refactored to `compat/` sub-package for backward
  compatibility.

---

## [0.2.0] — 2025-06-28

### Added

- Score-fusion conditioning variant.
- Configurable `alpha` blending parameter.
- `GateConfig` Pydantic model with validation.

---

## [0.1.0] — 2025-06-15

### Added

- Initial release.
- Global and hierarchical conditioning.
- `MockAdapter` for testing.
- Basic `TrajectoryFilter.run()` pipeline.
