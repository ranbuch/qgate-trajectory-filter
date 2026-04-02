"""
mitigation.py — TelemetryMitigator: two-stage quantum error mitigation.

Combines **Stage 1** (Galton adaptive trajectory filtering) with
**Stage 2** (machine-learning regression) to produce high-fidelity
expectation-value estimates from noisy hardware measurements.

Pipeline overview
-----------------

1. **Calibrate** — execute a set of near-Clifford calibration circuits
   whose ideal expectation values are efficiently simulable.  For each
   circuit, record hardware energy, Galton acceptance rate, and derived
   telemetry features.  Train a regression model to predict
   ``correction = ideal − filtered_energy``.

2. **Run / Estimate** — execute the target circuit on hardware, apply
   Galton trajectory filtering (Stage 1), extract the same telemetry
   feature vector, and feed it through the trained regressor (Stage 2)
   to obtain a mitigated expectation value.

The user can inject any scikit-learn-compatible regressor via a callable
factory (default: ``RandomForestRegressor``).

Usage::

    from qgate.mitigation import TelemetryMitigator, MitigatorConfig

    config = MitigatorConfig(keep_fraction=0.70)
    mitigator = TelemetryMitigator(config=config)

    # Step 1 — calibrate with known circuits
    cal = mitigator.calibrate(
        calibration_data=[
            {"energy": -1.02, "acceptance": 0.73, "variance": 0.04, "ideal": -1.0},
            ...
        ]
    )

    # Step 2 — mitigate a new measurement
    result = mitigator.estimate(
        raw_energy=-1.08,
        acceptance=0.68,
        variance=0.05,
    )
    print(result.mitigated_value)

Patent reference
----------------
US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
CIP addendum — ML-augmented TSVF trajectory mitigation.

Licensed under the QGATE Source Available Evaluation License v1.2.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("qgate.mitigation")

# ---------------------------------------------------------------------------
# Lazy scikit-learn imports — fail gracefully
# ---------------------------------------------------------------------------

try:
    from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore[import-untyped]
    from sklearn.ensemble import (  # type: ignore[import-untyped]
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    BaseEstimator = None  # type: ignore[assignment,misc]
    RegressorMixin = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Pydantic — with lightweight fallback (mirrors sampler.py pattern)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover — lightweight fallback
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = None  # type: ignore[assignment,misc]

    def _field_fallback(**kw: Any) -> Any:  # type: ignore[misc]
        return kw.get("default")

    Field = _field_fallback  # type: ignore[assignment]


def _require_sklearn() -> None:
    """Raise a helpful ``ImportError`` when scikit-learn is missing."""
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for TelemetryMitigator.  "
            "Install with:  pip install qgate[ml]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering helpers
# ═══════════════════════════════════════════════════════════════════════════

#: Canonical ordered list of features the regressor is trained on.
FEATURE_NAMES: Tuple[str, ...] = (
    "energy",
    "acceptance",
    "variance",
    "abs_energy",
    "energy_x_acceptance",
    "residual_from_mean",
)


def _extract_features(
    energy: float,
    acceptance: float,
    variance: float,
    mean_energy: float,
) -> np.ndarray:
    """Build a 1-D feature vector from raw telemetry scalars.

    The feature set is intentionally compact (6 features) to avoid
    over-fitting on small calibration sets while capturing the key
    signal dimensions.

    Args:
        energy:     Raw (or filtered) expectation-value estimate.
        acceptance: Galton trajectory acceptance rate (0–1).
        variance:   Shot-to-shot variance of the energy estimator.
        mean_energy: Rolling or batch-level mean energy (used for
                     residual feature).

    Returns:
        Feature vector of shape ``(6,)``.
    """
    return np.array(
        [
            energy,
            acceptance,
            variance,
            abs(energy),
            energy * acceptance,
            energy - mean_energy,
        ],
        dtype=np.float64,
    )


def _extract_feature_matrix(
    records: Sequence[Dict[str, float]],
) -> np.ndarray:
    """Build an (N, 6) feature matrix from calibration records.

    Each record **must** contain keys ``"energy"``, ``"acceptance"``,
    ``"variance"``.  A ``"mean_energy"`` key is optional; if absent,
    the batch mean of ``energy`` is used.

    Args:
        records: Sequence of telemetry dicts.

    Returns:
        Feature matrix of shape ``(N, 6)``.
    """
    energies = np.array([r["energy"] for r in records], dtype=np.float64)
    mean_e = float(np.mean(energies))
    rows = []
    for rec in records:
        rows.append(
            _extract_features(
                energy=rec["energy"],
                acceptance=rec["acceptance"],
                variance=rec["variance"],
                mean_energy=rec.get("mean_energy", mean_e),
            )
        )
    return np.vstack(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


class MitigatorConfig(BaseModel):  # type: ignore[misc]
    """Configuration for the two-stage telemetry mitigator.

    All fields are immutable after construction (``frozen=True``) to
    prevent accidental mutation between calibration and estimation.

    Attributes:
        keep_fraction:          Fraction of shots retained by the Galton
                                filter in Stage 1 (0, 1].  Higher values
                                keep more data; lower values are more
                                aggressive at discarding noisy shots.
        n_calibration_circuits: Default number of calibration circuits
                                generated by :meth:`calibrate` when using
                                synthetic / Cliffordised calibration.
        model_name:             Name of the default regression model.
                                One of ``"random_forest"``,
                                ``"gradient_boosting"``, or ``"ridge"``.
        model_params:           Extra keyword arguments forwarded to the
                                scikit-learn estimator constructor.
        scale_features:         Whether to apply ``StandardScaler`` to the
                                feature matrix before training / inference.
        random_state:           RNG seed for reproducibility (passed to
                                the estimator and scaler).
    """

    keep_fraction: float = Field(
        default=0.70,
        gt=0.0,
        le=1.0,
        description="Fraction of shots to keep after Galton filtering (Stage 1)",
    )
    n_calibration_circuits: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Number of calibration circuits for training",
    )
    model_name: str = Field(
        default="random_forest",
        description="Default regressor: random_forest | gradient_boosting | ridge",
    )
    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to the sklearn estimator constructor",
    )
    scale_features: bool = Field(
        default=True,
        description="Apply StandardScaler before training / inference",
    )
    random_state: Optional[int] = Field(
        default=42,
        description="RNG seed for reproducibility",
    )

    if ConfigDict is not None:
        model_config = ConfigDict(frozen=True, extra="forbid")


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CalibrationResult:
    """Artefacts produced by :meth:`TelemetryMitigator.calibrate`.

    Attributes:
        model_name:         Name of the trained regressor.
        n_samples:          Number of calibration samples used for training.
        feature_names:      Ordered feature names matching the model input.
        train_mae:          Mean absolute error on the training set.
        train_rmse:         Root mean squared error on the training set.
        elapsed_seconds:    Wall-clock time for calibration (seconds).
        metadata:           Free-form metadata dict.
    """

    model_name: str
    n_samples: int
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    train_mae: float = 0.0
    train_rmse: float = 0.0
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MitigationResult:
    """Result of a single two-stage mitigation inference.

    Attributes:
        mitigated_value:   The final ML-corrected expectation value.
        raw_energy:        Raw (unfiltered) hardware estimate.
        filtered_energy:   Energy after Stage 1 Galton filtering.
        correction:        Additive correction applied by the ML model
                           (``mitigated = filtered + correction``).
        acceptance:        Galton acceptance rate for this measurement.
        metadata:          Free-form metadata dict.
    """

    mitigated_value: float
    raw_energy: float
    filtered_energy: float
    correction: float
    acceptance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Built-in model factories
# ═══════════════════════════════════════════════════════════════════════════

_BUILTIN_MODELS: Dict[str, str] = {
    "random_forest": "sklearn.ensemble.RandomForestRegressor",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingRegressor",
    "ridge": "sklearn.linear_model.Ridge",
}


def _make_builtin_model(name: str, random_state: Optional[int], **kwargs: Any) -> Any:
    """Instantiate a built-in scikit-learn regressor by short name.

    Args:
        name:         One of ``"random_forest"``, ``"gradient_boosting"``,
                      or ``"ridge"``.
        random_state: RNG seed (forwarded if the estimator supports it).
        **kwargs:     Additional keyword arguments forwarded to the
                      estimator constructor.

    Returns:
        An unfitted scikit-learn regressor instance.
    """
    _require_sklearn()
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower == "random_forest":
        defaults: Dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 3,
            "random_state": random_state,
        }
        defaults.update(kwargs)
        return RandomForestRegressor(**defaults)

    if name_lower == "gradient_boosting":
        defaults = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_samples_leaf": 3,
            "random_state": random_state,
        }
        defaults.update(kwargs)
        return GradientBoostingRegressor(**defaults)

    if name_lower == "ridge":
        defaults = {"alpha": 1.0}
        defaults.update(kwargs)
        return Ridge(**defaults)

    raise ValueError(
        f"Unknown model name {name!r}.  Choose from: "
        f"{', '.join(sorted(_BUILTIN_MODELS))}, or pass a custom model_factory."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TelemetryMitigator
# ═══════════════════════════════════════════════════════════════════════════


class TelemetryMitigator:
    """Two-stage quantum error mitigation via trajectory telemetry + ML.

    **Stage 1 — Galton trajectory filtering** discards noisy shots
    whose combined LF/HF parity-check score falls below an adaptively
    placed threshold.  The remaining high-quality shots yield a
    *filtered* expectation value and a Galton acceptance rate.

    **Stage 2 — ML regression** predicts the residual correction
    ``Δ = ideal − filtered_energy`` from a compact telemetry feature
    vector (energy, acceptance rate, variance, and derived features).
    Adding this correction to the filtered estimate produces the final
    mitigated value.

    The regressor is trained during :meth:`calibrate` on circuits whose
    ideal values are known (e.g.  near-Clifford / Cliffordised circuits).
    After calibration, :meth:`estimate` applies both stages to unseen
    hardware measurements.

    Args:
        config:         Mitigator configuration (immutable after
                        construction).
        model_factory:  Optional callable ``() → estimator`` that returns
                        an unfitted scikit-learn-compatible regressor.
                        If ``None``, a built-in model is instantiated
                        from ``config.model_name``.

    Example::

        mitigator = TelemetryMitigator()
        mitigator.calibrate(calibration_data=[...])
        result = mitigator.estimate(
            raw_energy=-1.08, acceptance=0.68, variance=0.05,
        )
        print(result.mitigated_value)

    Patent reference:
        US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
        CIP addendum — ML-augmented TSVF trajectory mitigation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[MitigatorConfig] = None,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        _require_sklearn()
        self._config: MitigatorConfig = config or MitigatorConfig()
        self._model_factory = model_factory
        self._model: Any = None
        self._scaler: Any = None
        self._calibrated: bool = False
        self._calibration_result: Optional[CalibrationResult] = None
        self._training_mean_energy: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> MitigatorConfig:
        """Return the (immutable) mitigator configuration."""
        return self._config

    @property
    def is_calibrated(self) -> bool:
        """``True`` after a successful :meth:`calibrate` call."""
        return self._calibrated

    @property
    def calibration_result(self) -> Optional[CalibrationResult]:
        """Return the most recent :class:`CalibrationResult`, or ``None``."""
        return self._calibration_result

    @property
    def model(self) -> Any:
        """Return the underlying fitted scikit-learn model (or ``None``)."""
        return self._model

    # ------------------------------------------------------------------
    # Calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        calibration_data: Sequence[Dict[str, float]],
        *,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> CalibrationResult:
        """Train the Stage 2 regression model on known-ideal circuits.

        Each element of *calibration_data* is a dict with keys:

        - ``"energy"``     — filtered expectation value from hardware.
        - ``"acceptance"`` — Galton acceptance rate (0–1).
        - ``"variance"``   — shot-to-shot variance.
        - ``"ideal"``      — exact / classically-simulable ideal value.

        The model learns to predict ``correction = ideal − energy``
        (the additive error) so that at inference time we return
        ``filtered_energy + predicted_correction``.

        Args:
            calibration_data: Sequence of telemetry dicts.  Must contain
                at least 2 records.
            model_factory: Override the instance-level factory for this
                calibration only.

        Returns:
            :class:`CalibrationResult` with training metrics.

        Raises:
            ValueError: If fewer than 2 calibration records are provided.
            ImportError: If scikit-learn is not installed.
        """
        _require_sklearn()
        t0 = time.monotonic()

        # ── Validate ─────────────────────────────────────────────────
        if len(calibration_data) < 2:
            raise ValueError(
                f"calibrate() requires ≥ 2 calibration records, got {len(calibration_data)}"
            )
        required_keys = {"energy", "acceptance", "variance", "ideal"}
        for i, rec in enumerate(calibration_data):
            missing = required_keys - set(rec.keys())
            if missing:
                raise ValueError(
                    f"Calibration record [{i}] is missing keys: {missing}"
                )

        # ── Feature matrix & target vector ────────────────────────────
        X = _extract_feature_matrix(calibration_data)
        ideals = np.array([r["ideal"] for r in calibration_data], dtype=np.float64)
        energies = np.array([r["energy"] for r in calibration_data], dtype=np.float64)
        y = ideals - energies  # correction to predict

        self._training_mean_energy = float(np.mean(energies))

        # ── Optional scaling ──────────────────────────────────────────
        if self._config.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            self._scaler = None

        # ── Instantiate model ─────────────────────────────────────────
        factory = model_factory or self._model_factory
        if factory is not None:
            self._model = factory()
        else:
            self._model = _make_builtin_model(
                self._config.model_name,
                random_state=self._config.random_state,
                **self._config.model_params,
            )

        model_name = type(self._model).__name__

        # ── Train ─────────────────────────────────────────────────────
        logger.info(
            "TelemetryMitigator.calibrate: training %s on %d samples",
            model_name,
            len(calibration_data),
        )
        self._model.fit(X, y)
        self._calibrated = True

        # ── Training metrics ──────────────────────────────────────────
        y_pred = self._model.predict(X)
        residuals = y - y_pred
        train_mae = float(np.mean(np.abs(residuals)))
        train_rmse = float(np.sqrt(np.mean(residuals**2)))

        elapsed = time.monotonic() - t0

        self._calibration_result = CalibrationResult(
            model_name=model_name,
            n_samples=len(calibration_data),
            feature_names=FEATURE_NAMES,
            train_mae=train_mae,
            train_rmse=train_rmse,
            elapsed_seconds=elapsed,
            metadata={
                "config": self._config.model_dump() if hasattr(self._config, "model_dump") else {},
                "training_mean_energy": self._training_mean_energy,
            },
        )

        logger.info(
            "Calibration complete — MAE=%.6f  RMSE=%.6f  (%.2fs)",
            train_mae,
            train_rmse,
            elapsed,
        )

        return self._calibration_result

    # ------------------------------------------------------------------
    # Estimate (single measurement)
    # ------------------------------------------------------------------

    def estimate(
        self,
        raw_energy: float,
        acceptance: float,
        variance: float,
        *,
        filtered_energy: Optional[float] = None,
        mean_energy: Optional[float] = None,
    ) -> MitigationResult:
        """Apply two-stage mitigation to a single hardware measurement.

        Stage 1 is assumed to have already been applied externally (the
        caller provides the filtered energy and acceptance rate from the
        Galton filter).  This method performs **Stage 2** — ML regression
        correction.

        Args:
            raw_energy:      Unfiltered hardware expectation value.
            acceptance:      Galton acceptance rate (fraction of kept shots).
            variance:        Shot-to-shot variance of the energy estimator.
            filtered_energy: Energy after Galton filtering.  If ``None``,
                             defaults to ``raw_energy`` (no Stage 1).
            mean_energy:     Reference mean energy for the residual feature.
                             If ``None``, the training-set mean is used.

        Returns:
            :class:`MitigationResult` with the corrected value.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError(
                "TelemetryMitigator has not been calibrated.  "
                "Call calibrate() before estimate()."
            )

        filt_e = filtered_energy if filtered_energy is not None else raw_energy
        ref_mean = mean_energy if mean_energy is not None else self._training_mean_energy

        # ── Build feature vector ──────────────────────────────────────
        x = _extract_features(
            energy=filt_e,
            acceptance=acceptance,
            variance=variance,
            mean_energy=ref_mean,
        ).reshape(1, -1)

        if self._scaler is not None:
            x = self._scaler.transform(x)

        # ── Predict correction ────────────────────────────────────────
        correction = float(self._model.predict(x)[0])
        mitigated = filt_e + correction

        return MitigationResult(
            mitigated_value=mitigated,
            raw_energy=raw_energy,
            filtered_energy=filt_e,
            correction=correction,
            acceptance=acceptance,
            metadata={
                "model": type(self._model).__name__,
                "features_used": list(FEATURE_NAMES),
                "scaled": self._scaler is not None,
            },
        )

    # ------------------------------------------------------------------
    # Estimate (batch)
    # ------------------------------------------------------------------

    def estimate_batch(
        self,
        records: Sequence[Dict[str, float]],
        *,
        mean_energy: Optional[float] = None,
    ) -> List[MitigationResult]:
        """Apply two-stage mitigation to a batch of measurements.

        Each record dict must contain ``"energy"``, ``"acceptance"``,
        ``"variance"``.  Optional keys: ``"filtered_energy"``,
        ``"raw_energy"``.

        Args:
            records:     Sequence of measurement telemetry dicts.
            mean_energy: Override reference mean for the residual feature.

        Returns:
            List of :class:`MitigationResult`, one per input record.
        """
        if not self._calibrated:
            raise RuntimeError(
                "TelemetryMitigator has not been calibrated.  "
                "Call calibrate() before estimate_batch()."
            )

        ref_mean = mean_energy if mean_energy is not None else self._training_mean_energy

        # ── Build feature matrix for vectorised prediction ────────────
        filt_energies: List[float] = []
        raw_energies: List[float] = []
        acceptances: List[float] = []
        feature_rows: List[np.ndarray] = []

        for rec in records:
            raw_e = rec.get("raw_energy", rec["energy"])
            filt_e = rec.get("filtered_energy", rec["energy"])
            acc = rec["acceptance"]
            var = rec["variance"]

            raw_energies.append(raw_e)
            filt_energies.append(filt_e)
            acceptances.append(acc)
            feature_rows.append(
                _extract_features(
                    energy=filt_e,
                    acceptance=acc,
                    variance=var,
                    mean_energy=rec.get("mean_energy", ref_mean),
                )
            )

        X = np.vstack(feature_rows)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        corrections = self._model.predict(X)

        results: List[MitigationResult] = []
        for i, corr in enumerate(corrections):
            corr_f = float(corr)
            results.append(
                MitigationResult(
                    mitigated_value=filt_energies[i] + corr_f,
                    raw_energy=raw_energies[i],
                    filtered_energy=filt_energies[i],
                    correction=corr_f,
                    acceptance=acceptances[i],
                    metadata={
                        "model": type(self._model).__name__,
                        "batch_index": i,
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Convenience — combined calibrate + estimate
    # ------------------------------------------------------------------

    def calibrate_and_estimate(
        self,
        calibration_data: Sequence[Dict[str, float]],
        test_records: Sequence[Dict[str, float]],
        *,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> Tuple[CalibrationResult, List[MitigationResult]]:
        """Calibrate on known data, then estimate on new measurements.

        This is a convenience wrapper that calls :meth:`calibrate` and
        :meth:`estimate_batch` in sequence.

        Args:
            calibration_data: Training records (with ``"ideal"`` key).
            test_records:     Measurement records to mitigate.
            model_factory:    Optional override model factory.

        Returns:
            Tuple of ``(CalibrationResult, list[MitigationResult])``.
        """
        cal = self.calibrate(calibration_data, model_factory=model_factory)
        results = self.estimate_batch(test_records)
        return cal, results

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "calibrated" if self._calibrated else "uncalibrated"
        model_str = type(self._model).__name__ if self._model else "None"
        return (
            f"TelemetryMitigator("
            f"status={status}, "
            f"model={model_str}, "
            f"keep_fraction={self._config.keep_fraction})"
        )
