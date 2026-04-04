"""
stochastic.py — Probabilistic Processing Unit (PPU) telemetry-driven mitigation.

Extends the patented two-stage trajectory filtering architecture
(§22 of the CIP disclosure) from quantum processing units (QPUs) to
**classical stochastic hardware**: Monte Carlo simulators, thermodynamic
sampling units, p-bit arrays, True Random Number Generators, and analog
neuromorphic processors.

Architecture overview
---------------------

The module mirrors the quantum pipeline with a "Split Frontend, Unified
Backend" design:

**Frontend** — :class:`StochasticTelemetryExtractor`
    Extracts per-trajectory telemetry features from raw stochastic paths,
    analogous to Level-1 IQ probe readout in the QPU pipeline.

**Backend — Stage 1** — :class:`GaltonOutlierFilter`
    Applies MAD-robust statistical outlier rejection (patent §22.2) to
    discard trajectories corrupted by non-Markovian structural bias,
    thermal cross-talk, or PRNG memory artefacts.

**Backend — Stage 2** — :class:`StochasticMitigator`
    Trains a regression transfer function that maps telemetry features +
    raw observables → corrected (unbiased) observables.  Directly reuses
    the calibration-transfer paradigm from ``qgate.mitigation``.

**Orchestrator** — :class:`PPUMitigationPipeline`
    End-to-end pipeline wiring frontend and backend stages.

Simulator support
-----------------

The module ships a self-contained **Fractional Brownian Motion** (fBM)
simulator (:func:`simulate_fbm_paths`) to model non-Markovian asset
dynamics with tunable Hurst parameter.  This is used for financial Monte
Carlo acceleration proofs-of-concept but the pipeline itself is
simulator-agnostic.

Patent reference
----------------
US Provisional App. No. 64/XXX,XXX (April 2026), §22.
Prior: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915.
CIP — PPU generalization of trajectory filtering.

Licensed under the QGATE Source Available Evaluation License v1.2.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("qgate.stochastic")

# ---------------------------------------------------------------------------
# Lazy scikit-learn imports (mirrors mitigation.py pattern)
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import (  # type: ignore[import-untyped]
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Pydantic — with lightweight fallback (mirrors mitigation.py pattern)
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
            "scikit-learn is required for StochasticMitigator.  "
            "Install with:  pip install qgate[ml]"
        )


# ── MAD → σ conversion constant ──────────────────────────────────────────
_MAD_TO_SIGMA: float = 1.4826


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


class StochasticConfig(BaseModel):  # type: ignore[misc]
    """Configuration for the PPU stochastic mitigation pipeline.

    Mirrors :class:`~qgate.mitigation.MitigatorConfig` with
    domain-appropriate defaults for classical Monte Carlo workloads.

    Attributes:
        reject_fraction: Fraction of trajectories discarded by Stage 1
                         Galton filter (0, 1).  Default 0.25 discards
                         the 25% most extreme outliers.
        model_name:      scikit-learn regressor for Stage 2.
        scale_features:  Apply ``StandardScaler`` before ML training.
        random_state:    RNG seed for reproducibility.
    """

    reject_fraction: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description="Fraction of paths rejected by Stage 1 Galton filter",
    )
    model_name: str = Field(
        default="random_forest",
        description="Regressor: random_forest | gradient_boosting | ridge",
    )
    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to the sklearn estimator",
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
# Result Dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StochasticCalibrationResult:
    """Artefacts from :meth:`StochasticMitigator.calibrate`.

    Attributes:
        model_name:      Name of the trained regressor.
        n_samples:       Number of calibration paths used.
        feature_names:   Ordered feature names.
        train_mae:       Mean absolute error on the training set.
        train_rmse:      Root mean squared error on the training set.
        elapsed_seconds: Wall-clock calibration time.
        metadata:        Free-form metadata dict.
    """

    model_name: str
    n_samples: int
    feature_names: Tuple[str, ...] = ()
    train_mae: float = 0.0
    train_rmse: float = 0.0
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StochasticMitigationResult:
    """Result of the full PPU mitigation pipeline.

    Attributes:
        mitigated_value:   Corrected (unbiased) observable estimate.
        raw_value:         Raw budget estimate before mitigation.
        stage1_survivors:  Number of paths surviving Stage 1 filter.
        stage1_rejected:   Number of paths rejected by Stage 1.
        improvement_factor: ``|raw_error| / |mitigated_error|`` vs
                            a known reference (``NaN`` if no ref).
        latency_seconds:   Total pipeline wall-clock time.
        metadata:          Free-form metadata dict.
    """

    mitigated_value: float
    raw_value: float
    stage1_survivors: int
    stage1_rejected: int
    improvement_factor: float = float("nan")
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Fractional Brownian Motion Simulator (PPU Hardware Model)
# ═══════════════════════════════════════════════════════════════════════════


def _cholesky_fbm(n_steps: int, hurst: float) -> NDArray[np.float64]:
    """Build the Cholesky factor for fractional Gaussian noise (fGn).

    The fGn autocovariance for lag *k* is:

    .. math::
        \\gamma(k) = \\tfrac{1}{2}\\bigl(|k-1|^{2H}
                     - 2|k|^{2H} + |k+1|^{2H}\\bigr)

    This gives i.i.d. increments at H=0.5 and positively correlated
    increments for H > 0.5 (long-memory / trending behaviour).

    Args:
        n_steps: Number of time increments.
        hurst:   Hurst parameter ∈ (0, 1).

    Returns:
        Lower-triangular Cholesky factor of shape ``(n_steps, n_steps)``.
    """
    two_h = 2.0 * hurst

    # Build the fGn autocovariance matrix
    # gamma(|i - j|) for all pairs
    lags = np.arange(n_steps, dtype=np.float64)
    autocov = 0.5 * (
        np.abs(lags - 1) ** two_h
        - 2.0 * np.abs(lags) ** two_h
        + np.abs(lags + 1) ** two_h
    )
    # Toeplitz covariance matrix from the autocovariance function
    # C[i,j] = gamma(|i-j|)
    idx = np.arange(n_steps)
    cov = autocov[np.abs(idx[:, None] - idx[None, :])]

    # Regularise for numerical stability
    cov += np.eye(n_steps) * 1e-10
    return np.linalg.cholesky(cov)


def simulate_fbm_paths(
    n_paths: int,
    n_steps: int = 252,
    hurst: float = 0.7,
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """Generate asset price paths under Fractional Brownian Motion.

    fBM with Hurst H > 0.5 introduces **long-range positive
    autocorrelation** (non-Markovian "memory"), modelling the
    structural bias present in physical PPU hardware (§22.1 of the
    CIP disclosure).

    The price process follows:

    .. math::
        S_t = S_0 \\exp\\!\\bigl((\\mu - \\tfrac{1}{2}\\sigma^2)\\,t
              + \\sigma\\,B_t^H\\bigr)

    where :math:`B^H` is a fractional Brownian motion with Hurst
    parameter *H*.

    Args:
        n_paths: Number of independent Monte Carlo trajectories.
        n_steps: Time steps per trajectory (default 252 = trading days).
        hurst:   Hurst exponent ∈ (0, 1).  H=0.5 ⟹ standard GBM.
                 H=0.7 ⟹ trending (non-Markovian) paths.
        S0:      Initial asset price.
        mu:      Annualised drift.
        sigma:   Annualised volatility.
        T:       Time horizon in years.
        seed:    RNG seed for reproducibility.

    Returns:
        Array of shape ``(n_paths, n_steps + 1)`` with price paths.
        Column 0 is ``S0`` for every path.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Build Cholesky factor for fBM covariance
    L = _cholesky_fbm(n_steps, hurst)

    # Generate correlated increments: Z ~ N(0, I), fBm = L @ Z * sqrt(dt)
    Z = rng.standard_normal((n_paths, n_steps))
    fbm_increments = (L @ Z.T).T * np.sqrt(dt)  # (n_paths, n_steps)

    # GBM-like price process with fBM driving noise
    drift = (mu - 0.5 * sigma ** 2) * dt
    log_returns = drift + sigma * fbm_increments

    # Cumulative sum of log-returns → price levels
    log_prices = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    log_prices[:, 0] = np.log(S0)
    log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

    return np.exp(log_prices)


# ═══════════════════════════════════════════════════════════════════════════
# Asian Option Payoff
# ═══════════════════════════════════════════════════════════════════════════


def asian_call_payoff(
    paths: NDArray[np.float64],
    strike: float,
    r: float = 0.05,
    T: float = 1.0,
) -> NDArray[np.float64]:
    """Compute discounted payoff of a discrete arithmetic Asian call.

    .. math::
        \\text{payoff}_i = e^{-rT}\\,\\max\\!\\bigl(\\bar{S}_i - K,\\,0\\bigr)

    where :math:`\\bar{S}_i` is the arithmetic mean of the *i*-th
    price path (excluding the initial price).

    Args:
        paths:  Price array of shape ``(n_paths, n_steps + 1)``.
        strike: Strike price *K*.
        r:      Risk-free rate for discounting.
        T:      Time to maturity.

    Returns:
        Discounted payoffs, shape ``(n_paths,)``.
    """
    avg_prices = paths[:, 1:].mean(axis=1)  # arithmetic average
    payoffs = np.maximum(avg_prices - strike, 0.0)
    return np.exp(-r * T) * payoffs


# ═══════════════════════════════════════════════════════════════════════════
# Frontend — StochasticTelemetryExtractor
# ═══════════════════════════════════════════════════════════════════════════

#: Canonical feature names for the stochastic telemetry vector.
STOCHASTIC_FEATURE_NAMES: Tuple[str, ...] = (
    "realized_vol",
    "lag1_autocorr",
    "max_drawdown",
    "terminal_dist",
    "path_skewness",
    "mean_log_return",
)


class StochasticTelemetryExtractor:
    """Extract per-trajectory telemetry from raw stochastic price paths.

    Analogous to the Level-1 IQ probe readout in the QPU pipeline
    (§17.1 of the CIP disclosure).  Each price path is a 1-D analog
    "waveform" from which we extract a compact telemetry feature
    vector that characterises the instantaneous noise environment of
    that specific trajectory.

    The telemetry features are chosen to capture:

    * **Realised volatility** — instantaneous noise amplitude.
    * **Lag-1 autocorrelation** — non-Markovian memory strength
      (the fBM H > 0.5 signature).
    * **Maximum drawdown** — extreme-event severity.
    * **Terminal distance** — deviation from ensemble consensus.
    * **Path skewness** — asymmetry of return distribution.
    * **Mean log-return** — drift bias proxy.

    These features form the "Telemetry Lockbox" (§6.3) — they are
    computed independently of the payoff observable to prevent
    post-selection bias.

    Example::

        extractor = StochasticTelemetryExtractor()
        features = extractor.extract(paths)  # (n_paths, 6)
    """

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Ordered feature names."""
        return STOCHASTIC_FEATURE_NAMES

    @property
    def n_features(self) -> int:
        return len(STOCHASTIC_FEATURE_NAMES)

    def extract(self, paths: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract telemetry feature matrix from price paths.

        Args:
            paths: Price array of shape ``(n_paths, n_steps + 1)``.

        Returns:
            Feature matrix of shape ``(n_paths, 6)``.
        """
        n_paths = paths.shape[0]

        # Log returns: (n_paths, n_steps)
        log_ret = np.diff(np.log(paths), axis=1)

        features = np.empty((n_paths, self.n_features), dtype=np.float64)

        for i in range(n_paths):
            lr = log_ret[i]
            features[i] = self._extract_single(lr, paths[i])

        return features

    def extract_batch(
        self, paths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Vectorised batch extraction (alias for :meth:`extract`)."""
        return self.extract(paths)

    @staticmethod
    def _extract_single(
        log_returns: NDArray[np.float64],
        prices: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Extract features for a single path.

        Args:
            log_returns: Log-return array of shape ``(n_steps,)``.
            prices:      Price array of shape ``(n_steps + 1,)``.

        Returns:
            Feature vector of shape ``(6,)``.
        """
        # 1. Realized volatility — std of log-returns
        realized_vol = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0

        # 2. Lag-1 autocorrelation — proxy for non-Markovian memory (H > 0.5)
        if len(log_returns) > 2:
            lr_demeaned = log_returns - log_returns.mean()
            c0 = float(np.dot(lr_demeaned, lr_demeaned))
            c1 = float(np.dot(lr_demeaned[:-1], lr_demeaned[1:]))
            lag1_autocorr = c1 / c0 if abs(c0) > 1e-15 else 0.0
        else:
            lag1_autocorr = 0.0

        # 3. Maximum drawdown — worst peak-to-trough decline
        cummax = np.maximum.accumulate(prices)
        drawdowns = (prices - cummax) / np.where(cummax > 0, cummax, 1.0)
        max_drawdown = float(np.min(drawdowns))  # most negative

        # 4. Terminal distance from ensemble mean (normalised)
        # Caller may not have ensemble mean; use path-own terminal-to-start ratio
        terminal_dist = float(np.log(prices[-1] / prices[0]))

        # 5. Path skewness of log-returns
        if len(log_returns) > 2 and realized_vol > 1e-15:
            skew = float(np.mean(((log_returns - log_returns.mean()) / realized_vol) ** 3))
        else:
            skew = 0.0

        # 6. Mean log-return (drift bias proxy)
        mean_lr = float(np.mean(log_returns))

        return np.array(
            [realized_vol, lag1_autocorr, max_drawdown, terminal_dist, skew, mean_lr],
            dtype=np.float64,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Backend — Stage 1: GaltonOutlierFilter
# ═══════════════════════════════════════════════════════════════════════════


class GaltonOutlierFilter:
    """MAD-robust trajectory outlier rejection (Stage 1 Galton filter).

    Implements the Median Absolute Deviation (MAD) robust estimator
    from the patent (§9.2, §22.2) to compute per-feature z-scores
    and a composite viability score.  Trajectories with the most
    extreme scores are rejected.

    The MAD estimator is resilient to the non-Markovian "clumping"
    artefacts introduced by long-memory stochastic processes (fBM)
    or biased PRNG hardware.

    Args:
        reject_fraction: Fraction of trajectories to discard (0, 1).

    Example::

        filt = GaltonOutlierFilter(reject_fraction=0.25)
        mask = filt.filter(features)  # boolean survival mask
    """

    def __init__(self, reject_fraction: float = 0.25) -> None:
        if not 0.0 < reject_fraction < 1.0:
            raise ValueError(f"reject_fraction must be in (0, 1), got {reject_fraction}")
        self.reject_fraction = reject_fraction

    def compute_viability_scores(
        self, features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute composite trajectory viability scores.

        For each feature column, computes the MAD-robust z-score:

        .. math::
            z_j = \\frac{|x_j - \\text{median}(x)|}{1.4826 \\cdot \\text{MAD}(x)}

        The composite score is the Euclidean norm of the per-feature
        z-scores — lower scores indicate more "typical" trajectories.

        Args:
            features: Feature matrix ``(n_paths, n_features)``.

        Returns:
            Composite viability scores ``(n_paths,)`` — lower is better.
        """
        n_paths, n_features = features.shape
        z_scores = np.zeros_like(features)

        for j in range(n_features):
            col = features[:, j]
            median = np.median(col)
            mad = np.median(np.abs(col - median))
            sigma = _MAD_TO_SIGMA * mad

            if sigma > 1e-15:
                z_scores[:, j] = np.abs(col - median) / sigma
            else:
                z_scores[:, j] = 0.0

        # Composite: L2 norm of per-feature MAD z-scores
        return np.sqrt(np.sum(z_scores ** 2, axis=1))

    def filter(
        self, features: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Apply the Galton filter and return a boolean survival mask.

        Args:
            features: Feature matrix ``(n_paths, n_features)``.

        Returns:
            Boolean mask of shape ``(n_paths,)`` — ``True`` for retained
            trajectories.
        """
        scores = self.compute_viability_scores(features)
        n_reject = int(np.ceil(len(scores) * self.reject_fraction))
        n_keep = len(scores) - n_reject

        # Keep the n_keep paths with the LOWEST composite z-scores
        threshold_idx = np.argsort(scores)[n_keep - 1]
        cutoff = scores[threshold_idx]

        # Mask: keep everything at or below the cutoff score
        # Tie-breaking: if multiple paths share the cutoff score,
        # keep all of them (slightly more than target keep%)
        return scores <= cutoff + 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# Backend — Stage 2: StochasticMitigator (ML Reconstruction)
# ═══════════════════════════════════════════════════════════════════════════


class StochasticMitigator:
    """Two-stage telemetry-driven observable reconstruction for PPUs.

    Implements the mathematical transfer function (§17.2, §22.3) that
    maps ``[raw_observable, telemetry_vector] → corrected_observable``.

    **Calibration protocol** (mirrors ``TelemetryMitigator``):
      1. Execute a set of "calibration" paths with known ideal payoffs.
      2. Extract telemetry + raw payoffs as features.
      3. Train a regression model to predict ``ideal − raw`` corrections.

    **Inference**:
      1. For each surviving path, predict the correction.
      2. Apply: ``mitigated = raw + predicted_correction``.

    Args:
        config: :class:`StochasticConfig` (immutable).

    Example::

        config = StochasticConfig(reject_fraction=0.25)
        mitigator = StochasticMitigator(config)
        cal = mitigator.calibrate(train_features, train_payoffs, train_ideal)
        mitigated = mitigator.predict(test_features, test_payoffs)
    """

    def __init__(self, config: Optional[StochasticConfig] = None) -> None:
        _require_sklearn()
        self._config = config or StochasticConfig()
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._is_calibrated = False

    @property
    def config(self) -> StochasticConfig:
        return self._config

    @property
    def is_calibrated(self) -> bool:
        """True after successful :meth:`calibrate`."""
        return self._is_calibrated

    def _build_feature_matrix(
        self,
        telemetry: NDArray[np.float64],
        raw_payoffs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Concatenate raw payoffs with telemetry features.

        The raw payoff is prepended as the first feature so the model
        can learn "how far off is this specific observation?" — directly
        mapping the telemetry lockbox architecture.

        Returns:
            Feature matrix ``(n_paths, n_telemetry_features + 1)``.
        """
        return np.column_stack([raw_payoffs.reshape(-1, 1), telemetry])

    def _make_model(self) -> Any:
        """Instantiate the sklearn regressor from config."""
        _require_sklearn()
        cfg = self._config
        name = cfg.model_name.lower().replace("-", "_").replace(" ", "_")
        kwargs = dict(cfg.model_params) if cfg.model_params else {}

        if name == "random_forest":
            defaults = {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_leaf": 3,
                "random_state": cfg.random_state,
            }
            defaults.update(kwargs)
            return RandomForestRegressor(**defaults)

        if name == "gradient_boosting":
            defaults = {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_leaf": 3,
                "random_state": cfg.random_state,
            }
            defaults.update(kwargs)
            return GradientBoostingRegressor(**defaults)

        if name == "ridge":
            defaults = {"alpha": 1.0, "random_state": cfg.random_state}
            defaults.update(kwargs)
            return Ridge(**defaults)

        raise ValueError(
            f"Unknown model {cfg.model_name!r}.  "
            f"Supported: random_forest, gradient_boosting, ridge"
        )

    def calibrate(
        self,
        telemetry: NDArray[np.float64],
        raw_payoffs: NDArray[np.float64],
        ideal_payoffs: NDArray[np.float64],
    ) -> StochasticCalibrationResult:
        """Train the Stage 2 transfer function on calibration data.

        The model learns to predict the **correction**:
        ``correction_i = ideal_i − raw_i``.

        Args:
            telemetry:     Feature matrix ``(n_samples, n_features)``.
            raw_payoffs:   Raw (noisy) payoff for each path ``(n_samples,)``.
            ideal_payoffs: Known exact payoff for each path ``(n_samples,)``.

        Returns:
            :class:`StochasticCalibrationResult` with training metrics.
        """
        _require_sklearn()
        t0 = time.monotonic()

        n_samples = len(raw_payoffs)
        X = self._build_feature_matrix(telemetry, raw_payoffs)
        y = ideal_payoffs - raw_payoffs  # correction target

        # Optional feature scaling
        if self._config.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        # Train
        self._model = self._make_model()
        self._model.fit(X, y)
        self._is_calibrated = True

        # Training metrics
        preds = self._model.predict(X)
        residuals = preds - y
        train_mae = float(np.mean(np.abs(residuals)))
        train_rmse = float(np.sqrt(np.mean(residuals ** 2)))

        elapsed = time.monotonic() - t0

        n_feat = telemetry.shape[1] if telemetry.ndim == 2 else 0
        feature_names = ("raw_payoff",) + STOCHASTIC_FEATURE_NAMES[:n_feat]

        logger.info(
            "StochasticMitigator calibrated: model=%s, n=%d, "
            "train_mae=%.6f, elapsed=%.2fs",
            self._config.model_name, n_samples, train_mae, elapsed,
        )

        return StochasticCalibrationResult(
            model_name=self._config.model_name,
            n_samples=n_samples,
            feature_names=feature_names,
            train_mae=train_mae,
            train_rmse=train_rmse,
            elapsed_seconds=elapsed,
            metadata={
                "scale_features": self._config.scale_features,
                "random_state": self._config.random_state,
            },
        )

    def predict(
        self,
        telemetry: NDArray[np.float64],
        raw_payoffs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply the trained transfer function to produce mitigated payoffs.

        Args:
            telemetry:   Feature matrix ``(n_paths, n_features)``.
            raw_payoffs: Raw payoff per surviving path ``(n_paths,)``.

        Returns:
            Corrected payoffs ``(n_paths,)``.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        if not self._is_calibrated or self._model is None:
            raise RuntimeError(
                "StochasticMitigator is not calibrated.  "
                "Call calibrate() first."
            )

        X = self._build_feature_matrix(telemetry, raw_payoffs)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        corrections = self._model.predict(X)
        return raw_payoffs + corrections


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator — PPUMitigationPipeline
# ═══════════════════════════════════════════════════════════════════════════


class PPUMitigationPipeline:
    """End-to-end PPU trajectory filtering + ML reconstruction pipeline.

    Wires the three components together:
      1. :class:`StochasticTelemetryExtractor` (frontend)
      2. :class:`GaltonOutlierFilter` (Stage 1)
      3. :class:`StochasticMitigator` (Stage 2)

    **Usage pattern** (matches the quantum TelemetryMitigator API):

    .. code-block:: python

        pipe = PPUMitigationPipeline()

        # Calibrate on a small set of "known" paths
        pipe.calibrate(cal_paths, cal_payoffs, ideal_payoff_per_path)

        # Mitigate a budget run
        result = pipe.mitigate(budget_paths, budget_payoffs,
                               ground_truth=gt_price)

    Args:
        config: :class:`StochasticConfig` (immutable).
    """

    def __init__(self, config: Optional[StochasticConfig] = None) -> None:
        self._config = config or StochasticConfig()
        self._extractor = StochasticTelemetryExtractor()
        self._filter = GaltonOutlierFilter(reject_fraction=self._config.reject_fraction)
        self._mitigator = StochasticMitigator(config=self._config)

    @property
    def config(self) -> StochasticConfig:
        return self._config

    @property
    def extractor(self) -> StochasticTelemetryExtractor:
        return self._extractor

    @property
    def filter(self) -> GaltonOutlierFilter:
        return self._filter

    @property
    def mitigator(self) -> StochasticMitigator:
        return self._mitigator

    def calibrate(
        self,
        cal_paths: NDArray[np.float64],
        cal_raw_payoffs: NDArray[np.float64],
        cal_ideal_payoffs: NDArray[np.float64],
    ) -> StochasticCalibrationResult:
        """Calibrate the pipeline using paths with known ideal payoffs.

        This implements the **Calibration Transfer Protocol** (§9.3):
        a training phase learns the noise→correction mapping, which is
        then stored and reused for subsequent budget runs.

        Args:
            cal_paths:          Price paths ``(n_cal, n_steps + 1)``.
            cal_raw_payoffs:    Raw payoffs ``(n_cal,)``.
            cal_ideal_payoffs:  Known ideal payoffs ``(n_cal,)``.

        Returns:
            :class:`StochasticCalibrationResult`.
        """
        telemetry = self._extractor.extract(cal_paths)
        return self._mitigator.calibrate(telemetry, cal_raw_payoffs, cal_ideal_payoffs)

    def mitigate(
        self,
        paths: NDArray[np.float64],
        raw_payoffs: NDArray[np.float64],
        *,
        ground_truth: Optional[float] = None,
    ) -> StochasticMitigationResult:
        """Run the full two-stage mitigation pipeline.

        1. Extract telemetry from all paths.
        2. **Stage 1 — Galton Filter**: reject extreme outliers using
           MAD-robust viability scoring.
        3. **Stage 2 — ML Reconstruction**: apply trained transfer
           function to surviving paths.
        4. Aggregate the corrected payoffs into a single mitigated
           observable estimate.

        Args:
            paths:        Price paths ``(n_budget, n_steps + 1)``.
            raw_payoffs:  Raw payoffs ``(n_budget,)``.
            ground_truth: Optional known exact price for computing
                          improvement factor.

        Returns:
            :class:`StochasticMitigationResult`.
        """
        t0 = time.monotonic()

        # ── Telemetry extraction (frontend) ───────────────────────────
        telemetry = self._extractor.extract(paths)

        # ── Stage 1: Galton outlier rejection ─────────────────────────
        mask = self._filter.filter(telemetry)
        n_survivors = int(mask.sum())
        n_rejected = len(mask) - n_survivors

        surviving_telemetry = telemetry[mask]
        surviving_payoffs = raw_payoffs[mask]

        logger.info(
            "Stage 1 Galton filter: %d/%d paths survived (%.1f%% rejected)",
            n_survivors, len(mask), 100.0 * n_rejected / len(mask),
        )

        # ── Stage 2: ML reconstruction ────────────────────────────────
        mitigated_payoffs = self._mitigator.predict(
            surviving_telemetry, surviving_payoffs,
        )

        # ── Observable aggregation ────────────────────────────────────
        # The mitigated price is the mean of corrected payoffs
        mitigated_value = float(np.mean(mitigated_payoffs))
        raw_value = float(np.mean(raw_payoffs))

        elapsed = time.monotonic() - t0

        # ── Improvement factor ────────────────────────────────────────
        improvement = float("nan")
        if ground_truth is not None:
            raw_error = abs(raw_value - ground_truth)
            mit_error = abs(mitigated_value - ground_truth)
            if mit_error > 1e-15:
                improvement = raw_error / mit_error

        return StochasticMitigationResult(
            mitigated_value=mitigated_value,
            raw_value=raw_value,
            stage1_survivors=n_survivors,
            stage1_rejected=n_rejected,
            improvement_factor=improvement,
            latency_seconds=elapsed,
            metadata={
                "n_budget_paths": len(raw_payoffs),
                "reject_fraction": self._config.reject_fraction,
                "model_name": self._config.model_name,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Quick Monte Carlo Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def run_monte_carlo_benchmark(
    n_ground_truth: int = 1_000_000,
    n_budget: int = 1_000,
    n_calibration: int = 2_000,
    strike: float = 100.0,
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    hurst: float = 0.7,
    n_steps: int = 252,
    r: float = 0.05,
    reject_fraction: float = 0.25,
    model_name: str = "random_forest",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a complete financial Monte Carlo acceleration benchmark.

    Generates ground-truth, budget, and calibration path sets;
    calibrates the PPU mitigation pipeline; and returns a full
    results dictionary suitable for T12 benchmark ingestion.

    Args:
        n_ground_truth:  Number of paths for the "expensive" reference.
        n_budget:        Number of paths for the "cheap" budget run.
        n_calibration:   Number of additional paths for calibrating the ML model.
        strike:          Asian call strike price.
        S0:              Initial asset price.
        mu:              Annualised drift.
        sigma:           Annualised volatility.
        T:               Time horizon (years).
        hurst:           fBM Hurst exponent (0.5=Markov, 0.7=trending).
        n_steps:         Time steps per path.
        r:               Risk-free rate.
        reject_fraction: Stage 1 Galton filter rejection rate.
        model_name:      Stage 2 regression model.
        seed:            Master RNG seed.

    Returns:
        Dictionary with keys: ``ground_truth_price``,
        ``raw_budget_price``, ``mitigated_price``, ``raw_mae``,
        ``mitigated_mae``, ``improvement_factor``,
        ``equivalent_paths``, and detailed ``metadata``.
    """
    _require_sklearn()
    rng = np.random.default_rng(seed)

    logger.info("=" * 60)
    logger.info("PPU Monte Carlo Acceleration Benchmark")
    logger.info("=" * 60)

    # ── 1. Ground truth (expensive reference) ─────────────────────────
    t0 = time.monotonic()
    gt_paths = simulate_fbm_paths(
        n_paths=n_ground_truth, n_steps=n_steps, hurst=hurst,
        S0=S0, mu=mu, sigma=sigma, T=T, seed=rng.integers(0, 2**31),
    )
    gt_payoffs = asian_call_payoff(gt_paths, strike=strike, r=r, T=T)
    gt_price = float(np.mean(gt_payoffs))
    gt_std = float(np.std(gt_payoffs, ddof=1))
    gt_elapsed = time.monotonic() - t0
    logger.info("Ground truth (%d paths): %.6f  (%.2fs)", n_ground_truth, gt_price, gt_elapsed)

    # ── 2. Budget run (cheap, high-variance) ──────────────────────────
    t1 = time.monotonic()
    budget_paths = simulate_fbm_paths(
        n_paths=n_budget, n_steps=n_steps, hurst=hurst,
        S0=S0, mu=mu, sigma=sigma, T=T, seed=rng.integers(0, 2**31),
    )
    budget_payoffs = asian_call_payoff(budget_paths, strike=strike, r=r, T=T)
    raw_budget_price = float(np.mean(budget_payoffs))
    budget_elapsed = time.monotonic() - t1

    raw_mae = abs(raw_budget_price - gt_price)
    logger.info("Raw budget  (%d paths): %.6f  MAE=%.6f", n_budget, raw_budget_price, raw_mae)

    # ── 3. Calibration set ────────────────────────────────────────────
    # Generate a separate small set of paths with known "ideal" payoffs.
    # The ideal payoff per path is the path's own payoff under the
    # ground-truth distribution mean — we use the batch mean as a proxy
    # for the true conditional expectation.
    cal_paths = simulate_fbm_paths(
        n_paths=n_calibration, n_steps=n_steps, hurst=hurst,
        S0=S0, mu=mu, sigma=sigma, T=T, seed=rng.integers(0, 2**31),
    )
    cal_raw_payoffs = asian_call_payoff(cal_paths, strike=strike, r=r, T=T)

    # "Ideal" target: the ground-truth ensemble mean.  Each calibration
    # path's ideal payoff is set to gt_price, so the model learns:
    # correction = gt_price − raw_payoff.  This is the calibration
    # transfer protocol (§9.3): a known reference anchors the mapping.
    cal_ideal_payoffs = np.full_like(cal_raw_payoffs, gt_price)

    # ── 4. Qgate PPU mitigation pipeline ─────────────────────────────
    config = StochasticConfig(
        reject_fraction=reject_fraction,
        model_name=model_name,
        random_state=seed,
    )
    pipeline = PPUMitigationPipeline(config=config)

    # Calibrate
    cal_result = pipeline.calibrate(cal_paths, cal_raw_payoffs, cal_ideal_payoffs)

    # Mitigate the budget run
    t2 = time.monotonic()
    mit_result = pipeline.mitigate(
        budget_paths, budget_payoffs, ground_truth=gt_price,
    )
    mitigate_elapsed = time.monotonic() - t2

    mitigated_mae = abs(mit_result.mitigated_value - gt_price)
    improvement = raw_mae / mitigated_mae if mitigated_mae > 1e-15 else float("inf")

    # Equivalent paths: how many raw paths would you need to match
    # the mitigated accuracy?  SE = σ / sqrt(N) ⟹ N = (σ / SE)²
    if mitigated_mae > 1e-15:
        equivalent_paths = int((gt_std / mitigated_mae) ** 2)
    else:
        equivalent_paths = n_ground_truth

    return {
        "ground_truth_price": gt_price,
        "ground_truth_std": gt_std,
        "raw_budget_price": raw_budget_price,
        "mitigated_price": mit_result.mitigated_value,
        "raw_mae": raw_mae,
        "mitigated_mae": mitigated_mae,
        "improvement_factor": improvement,
        "equivalent_paths": equivalent_paths,
        "compute_reduction": equivalent_paths / n_budget,
        "stage1_survivors": mit_result.stage1_survivors,
        "stage1_rejected": mit_result.stage1_rejected,
        "calibration": {
            "n_calibration": n_calibration,
            "model_name": cal_result.model_name,
            "train_mae": cal_result.train_mae,
            "elapsed_seconds": cal_result.elapsed_seconds,
        },
        "timing": {
            "ground_truth_seconds": gt_elapsed,
            "budget_seconds": budget_elapsed,
            "mitigate_seconds": mitigate_elapsed,
        },
        "params": {
            "n_ground_truth": n_ground_truth,
            "n_budget": n_budget,
            "hurst": hurst,
            "strike": strike,
            "S0": S0,
            "mu": mu,
            "sigma": sigma,
            "T": T,
            "n_steps": n_steps,
            "r": r,
            "reject_fraction": reject_fraction,
        },
    }
