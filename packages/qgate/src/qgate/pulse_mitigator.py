"""
pulse_mitigator.py — PulseMitigator: firmware-level ML-driven error mitigation.

This module implements **pulse-level machine learning** — an architecture
that operates below the gate abstraction, directly on the analog
microwave control signals sent to superconducting qubits.

How it works
------------

Conventional error mitigation acts on **Level-2** (binary bit-string)
measurement outcomes.  The PulseMitigator instead consumes **Level-1**
(analog) IQ readout data — raw In-phase / Quadrature voltages — and
trains a lightweight ML model to predict Two-Level System (TLS) drift
in real time.

Pipeline overview
~~~~~~~~~~~~~~~~~

1. **Calibrate** — execute a bank of circuits with known artificial
   frequency detunings.  For each shot, extract Level-1 IQ telemetry
   and compute a 5-dimensional feature vector (magnitude, phase,
   distance-to-|0⟩, distance-to-|1⟩, temporal phase delta).
   Train the ML model to map features → detuning (Hz).

2. **Run with Active Cancellation** — execute a fast IQ probe, extract
   features, predict the current TLS drift, and **inject the inverse
   frequency shift** into the drive pulse of the target qubit so the
   subsequent gate operates at the corrected transition frequency.

Qiskit Pulse compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

``qiskit.pulse`` was removed in Qiskit ≥ 2.0.  The PulseMitigator
therefore operates in two modes:

* **Pulse mode** — when ``qiskit.pulse`` is available (Qiskit 1.x),
  actual ``ScheduleBlock`` objects with ``ShiftFrequency`` instructions
  are built and bound to real backend channels.

* **Simulation mode** (default for Qiskit 2.x) — pulse schedules are
  represented as lightweight dataclass objects, and IQ feature
  extraction / ML prediction work identically using NumPy.  This mode
  is suitable for offline development, unit testing, and patent
  evidence generation.

Both modes share the same feature-engineering math and ML pipeline.

Usage::

    from qgate.pulse_mitigator import PulseMitigator, PulseMitigatorConfig

    config = PulseMitigatorConfig(target_qubit=0)
    pm = PulseMitigator(config=config)

    # Calibrate with known detunings
    pm.calibrate(iq_shots=[...], detunings_hz=[...])

    # Predict drift from a new IQ measurement
    result = pm.predict_drift(i=0.0012, q=-0.0034)

    # Or get a fully corrected circuit
    corrected_circuit = pm.run_with_active_cancellation(
        target_circuit=qc,
        probe_i=0.0012,
        probe_q=-0.0034,
    )

Patent reference
----------------
US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
CIP addendum — Pulse-level ML-augmented TSVF firmware mitigation.

Licensed under the QGATE Source Available Evaluation License v1.2.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("qgate.pulse_mitigator")

# ---------------------------------------------------------------------------
# Lazy Qiskit imports — fail gracefully
# ---------------------------------------------------------------------------

try:
    from qiskit import QuantumCircuit  # type: ignore[import-untyped]
    from qiskit.circuit import Gate, Parameter  # type: ignore[import-untyped]

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

# Qiskit Pulse was removed in Qiskit >= 2.0.  Guard gracefully.
try:
    from qiskit.pulse import (  # type: ignore[import-untyped]
        DriveChannel,
        Gaussian,
        Play,
        ScheduleBlock,
        ShiftFrequency,
        build,
    )

    HAS_PULSE = True
except ImportError:
    HAS_PULSE = False

# scikit-learn — optional
try:
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
            "scikit-learn is required for PulseMitigator.  "
            "Install with:  pip install qgate[ml]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# IQ Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

#: Canonical ordered list of IQ features.
IQ_FEATURE_NAMES: Tuple[str, ...] = (
    "magnitude",
    "phase",
    "distance_to_0",
    "distance_to_1",
    "temporal_delta_phase",
)

#: Default number of recent shots for temporal phase averaging.
_PHASE_HISTORY_WINDOW: int = 10


def extract_iq_features(
    i: float,
    q: float,
    centroid_0: complex,
    centroid_1: complex,
    phase_history: Sequence[float] = (),
) -> np.ndarray:
    """Extract a 5-dimensional feature vector from Level-1 IQ readout.

    This function operates on **analog** (Level-1 / ``meas_level=1``)
    In-phase / Quadrature measurement data, NOT on binary bit-string
    (Level-2) outcomes.

    The five features capture the geometric position of the current IQ
    point relative to the calibrated state centroids and the temporal
    evolution of the readout phase:

    .. math::

        R         &= \\sqrt{I^2 + Q^2}                             \\\\
        \\theta   &= \\operatorname{atan2}(Q,\\, I)                \\\\
        D_0       &= |z - c_0|                                     \\\\
        D_1       &= |z - c_1|                                     \\\\
        \\Delta\\theta &= \\theta - \\langle\\theta\\rangle_{\\text{last } N}

    where :math:`z = I + jQ`, :math:`c_0` / :math:`c_1` are the
    calibrated |0⟩ / |1⟩ centroids, and the temporal delta is computed
    over the last :data:`_PHASE_HISTORY_WINDOW` shots.

    Args:
        i:              In-phase (real) voltage.
        q:              Quadrature (imaginary) voltage.
        centroid_0:     Calibrated |0⟩ IQ centroid (complex).
        centroid_1:     Calibrated |1⟩ IQ centroid (complex).
        phase_history:  Recent phase angles (radians) for temporal delta.

    Returns:
        Feature vector of shape ``(5,)`` with dtype ``float64``.
    """
    z = complex(i, q)

    # ── Magnitude (distance from origin) ──────────────────────────────
    magnitude: float = abs(z)

    # ── Phase angle ───────────────────────────────────────────────────
    phase: float = float(np.arctan2(q, i))

    # ── Distance to calibrated centroids ──────────────────────────────
    dist_0: float = abs(z - centroid_0)
    dist_1: float = abs(z - centroid_1)

    # ── Temporal phase delta ──────────────────────────────────────────
    if len(phase_history) > 0:
        recent = phase_history[-_PHASE_HISTORY_WINDOW:]
        avg_phase = float(np.mean(recent))
        delta_phase = phase - avg_phase
    else:
        delta_phase = 0.0

    return np.array(
        [magnitude, phase, dist_0, dist_1, delta_phase],
        dtype=np.float64,
    )


def extract_iq_features_batch(
    iq_shots: Sequence[Tuple[float, float]],
    centroid_0: complex,
    centroid_1: complex,
) -> np.ndarray:
    """Extract features from a batch of IQ shots with rolling phase history.

    Processes shots sequentially, maintaining a rolling window of recent
    phase angles for the temporal delta feature.

    Args:
        iq_shots:    Sequence of ``(I, Q)`` tuples.
        centroid_0:  Calibrated |0⟩ centroid.
        centroid_1:  Calibrated |1⟩ centroid.

    Returns:
        Feature matrix of shape ``(N, 5)``.
    """
    phase_history: List[float] = []
    rows: List[np.ndarray] = []
    for i_val, q_val in iq_shots:
        feat = extract_iq_features(
            i=i_val,
            q=q_val,
            centroid_0=centroid_0,
            centroid_1=centroid_1,
            phase_history=phase_history,
        )
        phase_history.append(feat[1])  # store phase for temporal delta
        rows.append(feat)
    return np.vstack(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation-mode pulse schedule (Qiskit 2.x fallback)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SimulatedPulseSchedule:
    """Lightweight stand-in for a Qiskit Pulse ``ScheduleBlock``.

    Used when ``qiskit.pulse`` is not available (Qiskit ≥ 2.0).
    Stores the parameterised frequency offset so the ML pipeline can
    bind a predicted drift value without requiring real pulse hardware.

    Attributes:
        target_qubit:    Qubit index the schedule targets.
        freq_offset_hz:  The frequency offset (Hz) to apply.  ``None``
                         means the schedule is still parameterised.
        pulse_duration:  Duration of the Gaussian drive pulse (dt units).
        pulse_sigma:     σ of the Gaussian envelope (dt units).
        pulse_amp:       Amplitude of the Gaussian drive pulse.
    """

    target_qubit: int
    freq_offset_hz: Optional[float] = None
    pulse_duration: int = 160
    pulse_sigma: int = 40
    pulse_amp: float = 0.5

    def bind(self, freq_offset_hz: float) -> SimulatedPulseSchedule:
        """Return a new schedule with the frequency offset bound."""
        return SimulatedPulseSchedule(
            target_qubit=self.target_qubit,
            freq_offset_hz=freq_offset_hz,
            pulse_duration=self.pulse_duration,
            pulse_sigma=self.pulse_sigma,
            pulse_amp=self.pulse_amp,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


class PulseMitigatorConfig(BaseModel):  # type: ignore[misc]
    """Configuration for the pulse-level ML mitigator.

    All fields are immutable after construction (``frozen=True``).

    Attributes:
        target_qubit:       Index of the qubit to mitigate.
        model_name:         Short name of the default ML model.
                            ``"ridge"`` (default, low latency) or any
                            name accepted by
                            :func:`~qgate.mitigation._make_builtin_model`.
        model_params:       Extra kwargs forwarded to the estimator
                            constructor.
        scale_features:     Whether to apply ``StandardScaler`` before
                            training / inference.
        pulse_duration:     Duration of the Gaussian drive pulse (dt).
        pulse_sigma:        σ of the Gaussian envelope (dt).
        pulse_amp:          Amplitude of the Gaussian drive pulse.
        phase_history_size: Rolling window size for temporal phase delta.
        random_state:       RNG seed for reproducibility.
    """

    target_qubit: int = Field(
        default=0, ge=0, description="Qubit index to target for pulse mitigation"
    )
    model_name: str = Field(
        default="ridge",
        description="Default regressor: ridge (low latency recommended)",
    )
    model_params: Dict[str, Any] = Field(
        default_factory=lambda: {"alpha": 1.0},
        description="Extra kwargs forwarded to the sklearn estimator constructor",
    )
    scale_features: bool = Field(
        default=True,
        description="Apply StandardScaler before training / inference",
    )
    pulse_duration: int = Field(
        default=160, ge=1, description="Gaussian pulse duration (dt units)"
    )
    pulse_sigma: int = Field(
        default=40, ge=1, description="Gaussian pulse sigma (dt units)"
    )
    pulse_amp: float = Field(
        default=0.5, gt=0.0, le=1.0, description="Gaussian pulse amplitude"
    )
    phase_history_size: int = Field(
        default=10, ge=1, description="Rolling window for temporal phase delta"
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
class PulseCalibrationResult:
    """Artefacts from :meth:`PulseMitigator.calibrate`.

    Attributes:
        model_name:         Name of the trained regressor.
        n_samples:          Number of IQ calibration shots used.
        feature_names:      Ordered feature names.
        centroid_0:         Calibrated |0⟩ IQ centroid.
        centroid_1:         Calibrated |1⟩ IQ centroid.
        train_mae:          Training mean absolute error (Hz).
        train_rmse:         Training root mean squared error (Hz).
        elapsed_seconds:    Calibration wall-clock time.
        metadata:           Free-form metadata dict.
    """

    model_name: str
    n_samples: int
    feature_names: Tuple[str, ...] = IQ_FEATURE_NAMES
    centroid_0: complex = 0j
    centroid_1: complex = 0j
    train_mae: float = 0.0
    train_rmse: float = 0.0
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftPrediction:
    """Result of a single drift prediction.

    Attributes:
        predicted_drift_hz:     Predicted TLS frequency drift (Hz).
        correction_hz:          Correction to apply (negated drift).
        features:               The 5-D IQ feature vector used.
        metadata:               Free-form metadata dict.
    """

    predicted_drift_hz: float
    correction_hz: float
    features: Tuple[float, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveCancellationResult:
    """Result of :meth:`PulseMitigator.run_with_active_cancellation`.

    Attributes:
        corrected_circuit:     The circuit with the ML-corrected pulse
                               calibration injected.
        drift_prediction:      The underlying drift prediction.
        pulse_schedule:        The bound pulse schedule (real or simulated).
        metadata:              Free-form metadata dict.
    """

    corrected_circuit: Any  # QuantumCircuit when available
    drift_prediction: DriftPrediction
    pulse_schedule: Any  # ScheduleBlock or SimulatedPulseSchedule
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# PulseMitigator — the main class
# ═══════════════════════════════════════════════════════════════════════════


class PulseMitigator:
    """Firmware-level ML-driven TLS drift cancellation via IQ telemetry.

    This class operates on **Level-1 (analog)** IQ readout data — raw
    In-phase / Quadrature voltages from the measurement chain — rather
    than on Level-2 (binary) bit-string outcomes.  It trains a
    lightweight ML model (default: Ridge regression for microsecond
    latency) to predict the current Two-Level System (TLS) frequency
    drift from the IQ feature vector, then injects the inverse shift
    into the drive pulse to actively cancel the drift.

    The architecture works in two modes:

    * **Pulse mode** (Qiskit 1.x with ``qiskit.pulse``) — builds real
      ``ScheduleBlock`` objects with ``ShiftFrequency`` instructions.
    * **Simulation mode** (Qiskit ≥ 2.x or no backend) — uses
      :class:`SimulatedPulseSchedule` and custom ``Gate`` injection.

    Both modes share identical IQ feature extraction and ML prediction.

    Args:
        config:         :class:`PulseMitigatorConfig` (immutable).
        backend:        Optional Qiskit backend.  When provided and
                        ``qiskit.pulse`` is available, real pulse
                        schedules are built.
        model_factory:  Optional callable ``() → estimator`` returning
                        an unfitted scikit-learn-compatible regressor.

    Example::

        pm = PulseMitigator(PulseMitigatorConfig(target_qubit=0))
        pm.calibrate(
            iq_shots=[(0.001, -0.003), (0.002, 0.001), ...],
            detunings_hz=[1000, 2000, ...],
        )
        result = pm.predict_drift(i=0.0015, q=-0.002)
        print(f"Predicted drift: {result.predicted_drift_hz:.1f} Hz")

    Patent reference:
        US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
        CIP addendum — Pulse-level ML-augmented TSVF firmware mitigation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[PulseMitigatorConfig] = None,
        backend: Any = None,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        _require_sklearn()

        self._config: PulseMitigatorConfig = config or PulseMitigatorConfig()
        self._backend = backend
        self._model_factory = model_factory

        # ML state
        self._model: Any = None
        self._scaler: Any = None
        self._calibrated: bool = False
        self._calibration_result: Optional[PulseCalibrationResult] = None

        # IQ centroids (set during calibration)
        self._centroid_0: complex = 0j
        self._centroid_1: complex = 0j

        # Phase history for temporal delta feature
        self._phase_history: deque = deque(
            maxlen=self._config.phase_history_size
        )

        # Parameterised pulse schedule
        self._pulse_mode: bool = HAS_PULSE and backend is not None
        self._freq_offset_param: Any = None  # qiskit.circuit.Parameter
        self._parameterized_schedule: Any = None
        self._build_parameterized_schedule()

        logger.info(
            "PulseMitigator initialised — qubit=%d, pulse_mode=%s, model=%s",
            self._config.target_qubit,
            self._pulse_mode,
            self._config.model_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> PulseMitigatorConfig:
        """The (immutable) pulse mitigator configuration."""
        return self._config

    @property
    def is_calibrated(self) -> bool:
        """``True`` after a successful :meth:`calibrate` call."""
        return self._calibrated

    @property
    def calibration_result(self) -> Optional[PulseCalibrationResult]:
        """The most recent :class:`PulseCalibrationResult`, or ``None``."""
        return self._calibration_result

    @property
    def centroid_0(self) -> complex:
        """Calibrated |0⟩ IQ centroid."""
        return self._centroid_0

    @property
    def centroid_1(self) -> complex:
        """Calibrated |1⟩ IQ centroid."""
        return self._centroid_1

    @property
    def pulse_mode(self) -> bool:
        """``True`` if real Qiskit Pulse schedules are being built."""
        return self._pulse_mode

    @property
    def model(self) -> Any:
        """The underlying fitted ML model (or ``None``)."""
        return self._model

    # ------------------------------------------------------------------
    # Parameterized pulse schedule builder
    # ------------------------------------------------------------------

    def _build_parameterized_schedule(self) -> None:
        """Build the parameterised drive pulse with a frequency offset slot.

        In **pulse mode** (Qiskit 1.x), this creates a real
        ``ScheduleBlock`` containing:

        1. ``ShiftFrequency(freq_offset, drive_channel)`` — the ML model
           will dictate this value to cancel TLS drift.
        2. ``Play(Gaussian(...), drive_channel)`` — a standard calibrated
           Gaussian π-pulse.

        In **simulation mode** (Qiskit ≥ 2.x), a lightweight
        :class:`SimulatedPulseSchedule` is stored instead.
        """
        qubit = self._config.target_qubit

        if self._pulse_mode:
            # ── Real Qiskit Pulse schedule ────────────────────────────
            self._freq_offset_param = Parameter("freq_offset")
            drive_chan = DriveChannel(qubit)

            with build(self._backend, name="ml_drift_cancel") as sched:
                ShiftFrequency(self._freq_offset_param, drive_chan)
                Play(
                    Gaussian(
                        duration=self._config.pulse_duration,
                        amp=self._config.pulse_amp,
                        sigma=self._config.pulse_sigma,
                    ),
                    drive_chan,
                )

            self._parameterized_schedule = sched
            logger.debug(
                "Built real pulse schedule for qubit %d (pulse mode)", qubit
            )
        else:
            # ── Simulation fallback ───────────────────────────────────
            if HAS_QISKIT:
                self._freq_offset_param = Parameter("freq_offset")
            else:
                self._freq_offset_param = None

            self._parameterized_schedule = SimulatedPulseSchedule(
                target_qubit=qubit,
                pulse_duration=self._config.pulse_duration,
                pulse_sigma=self._config.pulse_sigma,
                pulse_amp=self._config.pulse_amp,
            )
            logger.debug(
                "Built simulated pulse schedule for qubit %d (simulation mode)",
                qubit,
            )

    # ------------------------------------------------------------------
    # Calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        iq_shots: Sequence[Tuple[float, float]],
        detunings_hz: Sequence[float],
        *,
        labels: Optional[Sequence[int]] = None,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> PulseCalibrationResult:
        """Train the ML model on Level-1 IQ data with known detunings.

        This method accepts raw IQ readout voltages (``meas_level=1``)
        paired with the known artificial frequency detunings that were
        applied during calibration.

        Centroid computation
        ~~~~~~~~~~~~~~~~~~~~

        If *labels* (0 or 1 per shot, indicating the prepared state) are
        provided, centroids are computed as the mean IQ point of each
        class.  Otherwise, k-means-style splitting on the IQ plane is
        used (splitting at the median magnitude).

        Args:
            iq_shots:       Sequence of ``(I, Q)`` tuples — one per shot.
            detunings_hz:   Known frequency detunings (Hz), same length.
            labels:         Optional per-shot state labels (0 or 1) for
                            centroid computation.
            model_factory:  Override model factory for this calibration.

        Returns:
            :class:`PulseCalibrationResult` with training metrics.

        Raises:
            ValueError: If inputs are empty or mismatched in length.
        """
        _require_sklearn()
        t0 = time.monotonic()

        # ── Validate ─────────────────────────────────────────────────
        n = len(iq_shots)
        if n < 2:
            raise ValueError(
                f"calibrate() requires ≥ 2 IQ shots, got {n}"
            )
        if len(detunings_hz) != n:
            raise ValueError(
                f"iq_shots ({n}) and detunings_hz ({len(detunings_hz)}) "
                f"must have the same length"
            )

        # ── Compute centroids ─────────────────────────────────────────
        iq_complex = np.array(
            [complex(i, q) for i, q in iq_shots], dtype=np.complex128
        )
        self._centroid_0, self._centroid_1 = self._compute_centroids(
            iq_complex, labels
        )
        logger.info(
            "Centroids: |0⟩ = (%.6f, %.6fj)  |1⟩ = (%.6f, %.6fj)",
            self._centroid_0.real,
            self._centroid_0.imag,
            self._centroid_1.real,
            self._centroid_1.imag,
        )

        # ── Extract features ──────────────────────────────────────────
        X = extract_iq_features_batch(
            iq_shots, self._centroid_0, self._centroid_1
        )
        y = np.array(detunings_hz, dtype=np.float64)

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
            self._model = self._make_default_model()

        model_name = type(self._model).__name__

        # ── Train ─────────────────────────────────────────────────────
        logger.info(
            "PulseMitigator.calibrate: training %s on %d IQ shots",
            model_name,
            n,
        )
        self._model.fit(X, y)
        self._calibrated = True

        # ── Training metrics ──────────────────────────────────────────
        y_pred = self._model.predict(X)
        residuals = y - y_pred
        train_mae = float(np.mean(np.abs(residuals)))
        train_rmse = float(np.sqrt(np.mean(residuals**2)))

        # Reset phase history for inference
        self._phase_history.clear()

        elapsed = time.monotonic() - t0

        self._calibration_result = PulseCalibrationResult(
            model_name=model_name,
            n_samples=n,
            feature_names=IQ_FEATURE_NAMES,
            centroid_0=self._centroid_0,
            centroid_1=self._centroid_1,
            train_mae=train_mae,
            train_rmse=train_rmse,
            elapsed_seconds=elapsed,
            metadata={
                "config": (
                    self._config.model_dump()
                    if hasattr(self._config, "model_dump")
                    else {}
                ),
                "pulse_mode": self._pulse_mode,
            },
        )

        logger.info(
            "Pulse calibration complete — MAE=%.1f Hz  RMSE=%.1f Hz  (%.2fs)",
            train_mae,
            train_rmse,
            elapsed,
        )

        return self._calibration_result

    # ------------------------------------------------------------------
    # Predict drift (single IQ measurement)
    # ------------------------------------------------------------------

    def predict_drift(
        self,
        i: float,
        q: float,
    ) -> DriftPrediction:
        """Predict TLS frequency drift from a single Level-1 IQ readout.

        This is the core inference step.  It:

        1. Extracts the 5-D IQ feature vector.
        2. Scales features (if enabled).
        3. Calls ``self._model.predict()`` to get predicted drift (Hz).
        4. Updates the internal phase history for future temporal deltas.

        Args:
            i:  In-phase (real) voltage from ``meas_level=1`` readout.
            q:  Quadrature (imaginary) voltage.

        Returns:
            :class:`DriftPrediction` with the predicted drift and
            the negated correction value.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError(
                "PulseMitigator has not been calibrated.  "
                "Call calibrate() before predict_drift()."
            )

        # ── Extract features ──────────────────────────────────────────
        features = extract_iq_features(
            i=i,
            q=q,
            centroid_0=self._centroid_0,
            centroid_1=self._centroid_1,
            phase_history=list(self._phase_history),
        )

        # Update phase history for next call
        self._phase_history.append(features[1])  # phase angle

        # ── Scale & predict ───────────────────────────────────────────
        x = features.reshape(1, -1)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        predicted_drift_hz = float(self._model.predict(x)[0])
        correction_hz = -predicted_drift_hz

        return DriftPrediction(
            predicted_drift_hz=predicted_drift_hz,
            correction_hz=correction_hz,
            features=tuple(features.tolist()),
            metadata={
                "model": type(self._model).__name__,
                "scaled": self._scaler is not None,
                "pulse_mode": self._pulse_mode,
            },
        )

    # ------------------------------------------------------------------
    # Run with active cancellation
    # ------------------------------------------------------------------

    def run_with_active_cancellation(
        self,
        target_circuit: Any,
        probe_i: float,
        probe_q: float,
    ) -> ActiveCancellationResult:
        """Apply ML-predicted frequency correction to a target circuit.

        Given the latest Level-1 IQ probe readout, this method:

        1. Predicts the current TLS drift via :meth:`predict_drift`.
        2. Binds the **inverse** drift to the parameterised pulse schedule.
        3. Copies the target circuit and injects the corrected pulse
           as a custom gate (``ml_rx``) on the target qubit.

        In **pulse mode** the schedule is bound via
        ``assign_parameters`` and attached with ``add_calibration``.
        In **simulation mode** the correction is injected as a custom
        ``Gate('ml_rx', 1, [correction_hz])`` appended to the circuit.

        Args:
            target_circuit: The quantum circuit to correct.
            probe_i:        Latest In-phase IQ probe voltage.
            probe_q:        Latest Quadrature IQ probe voltage.

        Returns:
            :class:`ActiveCancellationResult` with the corrected circuit
            and drift prediction.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError(
                "PulseMitigator has not been calibrated.  "
                "Call calibrate() before run_with_active_cancellation()."
            )

        # ── Predict drift ─────────────────────────────────────────────
        drift = self.predict_drift(i=probe_i, q=probe_q)
        correction_hz = drift.correction_hz

        qubit = self._config.target_qubit

        if self._pulse_mode:
            bound_schedule = self._bind_pulse_schedule(correction_hz)
            corrected = self._inject_pulse_calibration(
                target_circuit, bound_schedule, qubit
            )
        else:
            bound_schedule = self._parameterized_schedule.bind(correction_hz)
            corrected = self._inject_gate_calibration(
                target_circuit, correction_hz, qubit
            )

        return ActiveCancellationResult(
            corrected_circuit=corrected,
            drift_prediction=drift,
            pulse_schedule=bound_schedule,
            metadata={
                "correction_hz": correction_hz,
                "target_qubit": qubit,
                "pulse_mode": self._pulse_mode,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_default_model(self) -> Any:
        """Instantiate the default ML model from config."""
        _require_sklearn()
        name = self._config.model_name.lower().replace("-", "_").replace(" ", "_")
        params = dict(self._config.model_params)

        if name == "ridge":
            params.setdefault("alpha", 1.0)
            return Ridge(**params)

        # Fall through to the mitigation module's factory if available
        try:
            from qgate.mitigation import _make_builtin_model

            return _make_builtin_model(
                name,
                random_state=self._config.random_state,
                **params,
            )
        except (ImportError, ValueError):
            # Default fallback: Ridge
            params.setdefault("alpha", 1.0)
            return Ridge(**params)

    @staticmethod
    def _compute_centroids(
        iq_complex: np.ndarray,
        labels: Optional[Sequence[int]] = None,
    ) -> Tuple[complex, complex]:
        """Compute |0⟩ and |1⟩ centroids from IQ data.

        If *labels* are provided, centroids are the mean of each class.
        Otherwise, a simple magnitude-median split is used: shots with
        magnitude below the median are assigned to |0⟩, above to |1⟩.

        Args:
            iq_complex: Array of complex IQ points.
            labels:     Optional per-shot labels (0 or 1).

        Returns:
            ``(centroid_0, centroid_1)`` as complex numbers.
        """
        if labels is not None:
            labels_arr = np.asarray(labels, dtype=int)
            mask_0 = labels_arr == 0
            mask_1 = labels_arr == 1
            c0 = complex(np.mean(iq_complex[mask_0])) if mask_0.any() else 0j
            c1 = complex(np.mean(iq_complex[mask_1])) if mask_1.any() else 0j
            return c0, c1

        # Auto-split by magnitude median
        magnitudes = np.abs(iq_complex)
        median_mag = float(np.median(magnitudes))
        mask_lo = magnitudes <= median_mag
        mask_hi = magnitudes > median_mag

        c0 = complex(np.mean(iq_complex[mask_lo])) if mask_lo.any() else 0j
        c1 = complex(np.mean(iq_complex[mask_hi])) if mask_hi.any() else 0j
        return c0, c1

    def _bind_pulse_schedule(self, correction_hz: float) -> Any:
        """Bind the frequency offset to the parameterised pulse schedule.

        Only used in pulse mode (Qiskit 1.x).
        """
        return self._parameterized_schedule.assign_parameters(
            {self._freq_offset_param: correction_hz},
            inplace=False,
        )

    @staticmethod
    def _inject_pulse_calibration(
        target_circuit: Any,
        bound_schedule: Any,
        qubit: int,
    ) -> Any:
        """Inject a bound pulse schedule into a circuit via add_calibration.

        Only used in pulse mode (Qiskit 1.x with add_calibration).
        """
        corrected = target_circuit.copy()
        ml_gate = Gate("ml_rx", 1, [])
        corrected.append(ml_gate, [qubit])
        corrected.add_calibration("ml_rx", (qubit,), bound_schedule)
        return corrected

    @staticmethod
    def _inject_gate_calibration(
        target_circuit: Any,
        correction_hz: float,
        qubit: int,
    ) -> Any:
        """Inject the ML correction as a custom Gate (simulation mode).

        Appends a ``Gate('ml_rx', 1, [correction_hz])`` to the target
        qubit.  In simulation mode the gate carries the correction as
        a parameter so downstream tooling can inspect or log it.
        """
        if not HAS_QISKIT:
            # No Qiskit at all — return a dict describing the correction
            return {
                "original_circuit": target_circuit,
                "ml_rx_correction_hz": correction_hz,
                "qubit": qubit,
            }

        corrected = target_circuit.copy()
        ml_gate = Gate("ml_rx", 1, [correction_hz])
        corrected.append(ml_gate, [qubit])
        return corrected

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear calibration state and phase history.

        After calling this, :meth:`calibrate` must be called again
        before :meth:`predict_drift` or
        :meth:`run_with_active_cancellation`.
        """
        self._model = None
        self._scaler = None
        self._calibrated = False
        self._calibration_result = None
        self._centroid_0 = 0j
        self._centroid_1 = 0j
        self._phase_history.clear()
        logger.info("PulseMitigator reset — recalibration required")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "calibrated" if self._calibrated else "uncalibrated"
        model_str = type(self._model).__name__ if self._model else "None"
        mode = "pulse" if self._pulse_mode else "simulation"
        return (
            f"PulseMitigator("
            f"status={status}, "
            f"qubit={self._config.target_qubit}, "
            f"mode={mode}, "
            f"model={model_str})"
        )
