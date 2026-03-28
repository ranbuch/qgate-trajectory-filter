"""Tests for qgate.pulse_mitigator — PulseMitigator firmware-level ML pipeline.

Tests are structured in tiers:
  1. Unit tests for PulseMitigatorConfig (Pydantic validation, frozen, bounds)
  2. Unit tests for IQ feature extraction math
  3. Unit tests for centroid computation
  4. Integration tests for PulseMitigator (calibrate → predict → cancel)

All tests use simulation mode (no qiskit.pulse required) so they run
on Qiskit 2.x and in environments without a real backend.

NOTICE: Pre-patent proprietary code — do NOT push to public repositories.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if scikit-learn not installed
# ---------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn", reason="scikit-learn required for pulse mitigator tests")

from pydantic import ValidationError  # noqa: E402

from qgate.pulse_mitigator import (  # noqa: E402
    IQ_FEATURE_NAMES,
    ActiveCancellationResult,
    DriftPrediction,
    PulseCalibrationResult,
    PulseMitigator,
    PulseMitigatorConfig,
    SimulatedPulseSchedule,
    _PHASE_HISTORY_WINDOW,
    extract_iq_features,
    extract_iq_features_batch,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — synthetic IQ data generators
# ═══════════════════════════════════════════════════════════════════════════


def _make_synthetic_iq_data(
    n: int = 100,
    drift_range_hz: float = 5000.0,
    noise_std: float = 0.001,
    seed: int = 42,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Generate synthetic IQ shots with known detunings.

    Simulates a qubit whose IQ readout shifts linearly with frequency
    detuning (a simplified model of TLS drift).

    Returns:
        ``(iq_shots, detunings_hz)``
    """
    rng = np.random.default_rng(seed)
    detunings = rng.uniform(-drift_range_hz, drift_range_hz, size=n)

    # Simple model: IQ point = base + drift_sensitivity * detuning + noise
    base_i, base_q = 0.005, -0.003
    sensitivity = 1e-7  # Hz → voltage
    iq_shots = []
    for det in detunings:
        i_val = base_i + sensitivity * det + rng.normal(0, noise_std)
        q_val = base_q - sensitivity * det * 0.5 + rng.normal(0, noise_std)
        iq_shots.append((i_val, q_val))
    return iq_shots, detunings.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — PulseMitigatorConfig unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPulseMitigatorConfig:
    """Verify PulseMitigatorConfig Pydantic model validation."""

    def test_defaults(self):
        cfg = PulseMitigatorConfig()
        assert cfg.target_qubit == 0
        assert cfg.model_name == "ridge"
        assert cfg.model_params == {"alpha": 1.0}
        assert cfg.scale_features is True
        assert cfg.pulse_duration == 160
        assert cfg.pulse_sigma == 40
        assert cfg.pulse_amp == pytest.approx(0.5)
        assert cfg.phase_history_size == 10
        assert cfg.random_state == 42

    def test_custom_values(self):
        cfg = PulseMitigatorConfig(
            target_qubit=3,
            model_name="random_forest",
            model_params={"n_estimators": 50},
            scale_features=False,
            pulse_duration=200,
            pulse_sigma=50,
            pulse_amp=0.8,
            phase_history_size=20,
            random_state=123,
        )
        assert cfg.target_qubit == 3
        assert cfg.model_name == "random_forest"
        assert cfg.pulse_duration == 200
        assert cfg.pulse_amp == pytest.approx(0.8)

    def test_frozen(self):
        cfg = PulseMitigatorConfig()
        with pytest.raises(ValidationError):
            cfg.target_qubit = 5  # type: ignore[misc]

    def test_target_qubit_bounds(self):
        with pytest.raises(ValidationError):
            PulseMitigatorConfig(target_qubit=-1)
        # Zero is valid
        cfg = PulseMitigatorConfig(target_qubit=0)
        assert cfg.target_qubit == 0

    def test_pulse_amp_bounds(self):
        with pytest.raises(ValidationError):
            PulseMitigatorConfig(pulse_amp=0.0)
        with pytest.raises(ValidationError):
            PulseMitigatorConfig(pulse_amp=1.5)
        cfg = PulseMitigatorConfig(pulse_amp=1.0)
        assert cfg.pulse_amp == pytest.approx(1.0)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PulseMitigatorConfig(nonexistent=True)  # type: ignore[call-arg]


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — SimulatedPulseSchedule
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulatedPulseSchedule:
    """Verify the simulation-mode pulse schedule dataclass."""

    def test_creation(self):
        sps = SimulatedPulseSchedule(target_qubit=0)
        assert sps.target_qubit == 0
        assert sps.freq_offset_hz is None
        assert sps.pulse_duration == 160
        assert sps.pulse_sigma == 40
        assert sps.pulse_amp == 0.5

    def test_bind(self):
        sps = SimulatedPulseSchedule(target_qubit=2)
        bound = sps.bind(freq_offset_hz=-1500.0)
        assert bound.freq_offset_hz == pytest.approx(-1500.0)
        assert bound.target_qubit == 2
        # Original unchanged (frozen)
        assert sps.freq_offset_hz is None

    def test_frozen(self):
        sps = SimulatedPulseSchedule(target_qubit=0)
        with pytest.raises(AttributeError):
            sps.freq_offset_hz = 100.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — IQ feature extraction math
# ═══════════════════════════════════════════════════════════════════════════


class TestIQFeatureExtraction:
    """Verify extract_iq_features math against hand-calculated values."""

    def test_shape_and_dtype(self):
        feat = extract_iq_features(
            i=0.003, q=-0.002,
            centroid_0=complex(0.001, -0.001),
            centroid_1=complex(0.005, 0.003),
        )
        assert feat.shape == (5,)
        assert feat.dtype == np.float64

    def test_magnitude(self):
        feat = extract_iq_features(
            i=3.0, q=4.0,
            centroid_0=0j, centroid_1=0j,
        )
        # R = sqrt(9 + 16) = 5
        assert feat[0] == pytest.approx(5.0)

    def test_phase(self):
        feat = extract_iq_features(
            i=1.0, q=1.0,
            centroid_0=0j, centroid_1=0j,
        )
        # theta = atan2(1, 1) = pi/4
        assert feat[1] == pytest.approx(math.pi / 4)

    def test_phase_negative_quadrant(self):
        feat = extract_iq_features(
            i=-1.0, q=-1.0,
            centroid_0=0j, centroid_1=0j,
        )
        # theta = atan2(-1, -1) = -3*pi/4
        assert feat[1] == pytest.approx(-3 * math.pi / 4)

    def test_distance_to_centroids(self):
        c0 = complex(1.0, 0.0)
        c1 = complex(0.0, 1.0)
        feat = extract_iq_features(i=0.0, q=0.0, centroid_0=c0, centroid_1=c1)
        # D_0 = |0 - (1+0j)| = 1.0
        assert feat[2] == pytest.approx(1.0)
        # D_1 = |0 - (0+1j)| = 1.0
        assert feat[3] == pytest.approx(1.0)

    def test_distance_asymmetric(self):
        c0 = complex(3.0, 4.0)
        c1 = complex(0.0, 0.0)
        feat = extract_iq_features(i=0.0, q=0.0, centroid_0=c0, centroid_1=c1)
        assert feat[2] == pytest.approx(5.0)  # dist to c0
        assert feat[3] == pytest.approx(0.0)  # dist to c1

    def test_temporal_delta_no_history(self):
        feat = extract_iq_features(
            i=1.0, q=0.0,
            centroid_0=0j, centroid_1=0j,
            phase_history=[],
        )
        # No history → delta = 0
        assert feat[4] == pytest.approx(0.0)

    def test_temporal_delta_with_history(self):
        # Current phase: atan2(1, 1) = pi/4
        history = [0.0, 0.0, 0.0, 0.0]  # avg = 0.0
        feat = extract_iq_features(
            i=1.0, q=1.0,
            centroid_0=0j, centroid_1=0j,
            phase_history=history,
        )
        expected_delta = math.pi / 4 - 0.0
        assert feat[4] == pytest.approx(expected_delta)

    def test_feature_names_length(self):
        assert len(IQ_FEATURE_NAMES) == 5

    def test_phase_history_window_constant(self):
        assert _PHASE_HISTORY_WINDOW == 10


class TestIQFeaturesBatch:
    """Verify batch feature extraction with rolling phase history."""

    def test_batch_shape(self):
        iq = [(0.001, -0.002), (0.003, 0.001), (0.002, -0.001)]
        X = extract_iq_features_batch(iq, centroid_0=0j, centroid_1=complex(0.01, 0.01))
        assert X.shape == (3, 5)

    def test_first_shot_zero_delta(self):
        """First shot has no history, so temporal delta = 0."""
        iq = [(1.0, 0.0), (0.0, 1.0)]
        X = extract_iq_features_batch(iq, centroid_0=0j, centroid_1=0j)
        assert X[0, 4] == pytest.approx(0.0)

    def test_second_shot_uses_history(self):
        """Second shot's delta uses the first shot's phase."""
        iq = [(1.0, 0.0), (0.0, 1.0)]
        X = extract_iq_features_batch(iq, centroid_0=0j, centroid_1=0j)
        # First shot phase: atan2(0, 1) = 0.0
        # Second shot phase: atan2(1, 0) = pi/2
        # Delta = pi/2 - 0.0
        assert X[1, 4] == pytest.approx(math.pi / 2)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Centroid computation
# ═══════════════════════════════════════════════════════════════════════════


class TestCentroidComputation:
    """Verify centroid calculation logic."""

    def test_with_labels(self):
        iq = np.array([1 + 0j, 1 + 0j, 5 + 5j, 5 + 5j], dtype=np.complex128)
        labels = [0, 0, 1, 1]
        c0, c1 = PulseMitigator._compute_centroids(iq, labels)
        assert c0 == pytest.approx(1 + 0j)
        assert c1 == pytest.approx(5 + 5j)

    def test_without_labels_magnitude_split(self):
        # Low magnitude points → |0⟩, high magnitude → |1⟩
        iq = np.array(
            [0.1 + 0.1j, 0.2 + 0.1j, 5.0 + 5.0j, 6.0 + 4.0j],
            dtype=np.complex128,
        )
        c0, c1 = PulseMitigator._compute_centroids(iq, labels=None)
        # Low-mag cluster centroid should be near origin
        assert abs(c0) < 1.0
        # High-mag cluster centroid should be far from origin
        assert abs(c1) > 3.0

    def test_single_class_labels(self):
        """All shots labelled 0 → c1 = 0j."""
        iq = np.array([1 + 1j, 2 + 2j], dtype=np.complex128)
        c0, c1 = PulseMitigator._compute_centroids(iq, labels=[0, 0])
        assert c0 == pytest.approx(1.5 + 1.5j)
        assert c1 == 0j


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — PulseMitigator integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPulseMitigator:
    """Integration tests for the full calibrate → predict → cancel pipeline."""

    def test_construction_defaults(self):
        pm = PulseMitigator()
        assert pm.is_calibrated is False
        assert pm.pulse_mode is False  # Qiskit 2.x → simulation mode
        assert pm.centroid_0 == 0j
        assert pm.centroid_1 == 0j
        assert pm.model is None

    def test_uncalibrated_predict_raises(self):
        pm = PulseMitigator()
        with pytest.raises(RuntimeError, match="not been calibrated"):
            pm.predict_drift(i=0.001, q=-0.002)

    def test_uncalibrated_cancellation_raises(self):
        qiskit = pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        pm = PulseMitigator()
        qc = QuantumCircuit(1)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            pm.run_with_active_cancellation(qc, probe_i=0.001, probe_q=-0.002)

    def test_calibrate_too_few(self):
        pm = PulseMitigator()
        with pytest.raises(ValueError, match="≥ 2"):
            pm.calibrate(iq_shots=[(0.001, -0.002)], detunings_hz=[100.0])

    def test_calibrate_length_mismatch(self):
        pm = PulseMitigator()
        with pytest.raises(ValueError, match="same length"):
            pm.calibrate(
                iq_shots=[(0.001, -0.002), (0.003, 0.001)],
                detunings_hz=[100.0],
            )

    def test_calibrate_returns_result(self):
        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator()
        cal = pm.calibrate(iq, det)

        assert isinstance(cal, PulseCalibrationResult)
        assert cal.n_samples == 50
        assert cal.model_name == "Ridge"
        assert cal.feature_names == IQ_FEATURE_NAMES
        assert cal.centroid_0 != 0j
        assert cal.centroid_1 != 0j
        assert cal.train_mae >= 0
        assert cal.train_rmse >= 0
        assert cal.elapsed_seconds > 0

    def test_is_calibrated_flag(self):
        iq, det = _make_synthetic_iq_data(n=20)
        pm = PulseMitigator()
        assert pm.is_calibrated is False
        pm.calibrate(iq, det)
        assert pm.is_calibrated is True

    def test_centroids_set_after_calibration(self):
        iq, det = _make_synthetic_iq_data(n=30)
        pm = PulseMitigator()
        pm.calibrate(iq, det)
        assert pm.centroid_0 != 0j or pm.centroid_1 != 0j

    def test_calibrate_with_labels(self):
        iq, det = _make_synthetic_iq_data(n=30)
        labels = [0 if i < 15 else 1 for i in range(30)]
        pm = PulseMitigator()
        cal = pm.calibrate(iq, det, labels=labels)
        assert cal.n_samples == 30

    def test_predict_drift_returns_result(self):
        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        pred = pm.predict_drift(i=0.005, q=-0.003)
        assert isinstance(pred, DriftPrediction)
        assert isinstance(pred.predicted_drift_hz, float)
        assert pred.correction_hz == pytest.approx(-pred.predicted_drift_hz)
        assert len(pred.features) == 5

    def test_predict_drift_correction_sign(self):
        """Correction should be the negation of predicted drift."""
        iq, det = _make_synthetic_iq_data(n=80, seed=77)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        pred = pm.predict_drift(i=0.006, q=-0.001)
        assert pred.correction_hz == pytest.approx(-pred.predicted_drift_hz)

    def test_predict_drift_updates_phase_history(self):
        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        # Make multiple predictions — phase history should accumulate
        for _ in range(5):
            pm.predict_drift(i=0.004, q=-0.002)

        assert len(pm._phase_history) == 5

    def test_run_with_active_cancellation_simulation_mode(self):
        """In simulation mode, a custom Gate('ml_rx') is injected."""
        qiskit = pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        qc = QuantumCircuit(1)
        qc.h(0)
        result = pm.run_with_active_cancellation(qc, probe_i=0.005, probe_q=-0.003)

        assert isinstance(result, ActiveCancellationResult)
        assert isinstance(result.drift_prediction, DriftPrediction)
        assert isinstance(result.pulse_schedule, SimulatedPulseSchedule)
        assert result.pulse_schedule.freq_offset_hz is not None

        # Corrected circuit should have more instructions than original
        corrected = result.corrected_circuit
        assert corrected.num_qubits == 1
        gate_names = [inst.operation.name for inst in corrected.data]
        assert "ml_rx" in gate_names

    def test_active_cancellation_preserves_original(self):
        """Original circuit should NOT be modified."""
        qiskit = pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        original_len = len(qc.data)

        pm.run_with_active_cancellation(qc, probe_i=0.005, probe_q=-0.003)

        # Original unchanged
        assert len(qc.data) == original_len

    def test_active_cancellation_metadata(self):
        qiskit = pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        result = pm.run_with_active_cancellation(
            QuantumCircuit(1), probe_i=0.005, probe_q=-0.003
        )
        assert "correction_hz" in result.metadata
        assert "target_qubit" in result.metadata
        assert result.metadata["pulse_mode"] is False

    # ------------------------------------------------------------------
    # Model injection
    # ------------------------------------------------------------------

    def test_custom_model_factory(self):
        from sklearn.ensemble import GradientBoostingRegressor

        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator(
            model_factory=lambda: GradientBoostingRegressor(
                n_estimators=30, random_state=42
            )
        )
        cal = pm.calibrate(iq, det)
        assert cal.model_name == "GradientBoostingRegressor"

    def test_override_model_at_calibrate(self):
        from sklearn.ensemble import RandomForestRegressor

        iq, det = _make_synthetic_iq_data(n=50)
        pm = PulseMitigator()  # default is Ridge
        cal = pm.calibrate(
            iq, det,
            model_factory=lambda: RandomForestRegressor(
                n_estimators=30, random_state=42
            ),
        )
        assert cal.model_name == "RandomForestRegressor"

    def test_config_model_name_random_forest(self):
        iq, det = _make_synthetic_iq_data(n=50)
        cfg = PulseMitigatorConfig(model_name="random_forest", model_params={})
        pm = PulseMitigator(config=cfg)
        cal = pm.calibrate(iq, det)
        assert cal.model_name == "RandomForestRegressor"

    # ------------------------------------------------------------------
    # Scaling toggle
    # ------------------------------------------------------------------

    def test_no_scaling(self):
        iq, det = _make_synthetic_iq_data(n=50)
        cfg = PulseMitigatorConfig(scale_features=False)
        pm = PulseMitigator(config=cfg)
        pm.calibrate(iq, det)

        pred = pm.predict_drift(i=0.005, q=-0.003)
        assert isinstance(pred.predicted_drift_hz, float)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset(self):
        iq, det = _make_synthetic_iq_data(n=30)
        pm = PulseMitigator()
        pm.calibrate(iq, det)
        assert pm.is_calibrated is True

        pm.reset()
        assert pm.is_calibrated is False
        assert pm.model is None
        assert pm.centroid_0 == 0j
        assert pm.centroid_1 == 0j

    def test_reset_then_recalibrate(self):
        iq, det = _make_synthetic_iq_data(n=30, seed=1)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        pm.reset()

        iq2, det2 = _make_synthetic_iq_data(n=50, seed=2)
        cal = pm.calibrate(iq2, det2)
        assert cal.n_samples == 50
        assert pm.is_calibrated is True

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def test_repr_uncalibrated(self):
        pm = PulseMitigator()
        r = repr(pm)
        assert "uncalibrated" in r
        assert "simulation" in r
        assert "qubit=0" in r

    def test_repr_calibrated(self):
        iq, det = _make_synthetic_iq_data(n=20)
        pm = PulseMitigator()
        pm.calibrate(iq, det)
        r = repr(pm)
        assert "calibrated" in r
        assert "Ridge" in r


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — Multi-qubit / multi-call scenarios
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiQubitScenarios:
    """Verify PulseMitigator targets the correct qubit."""

    def test_target_qubit_3(self):
        qiskit = pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        cfg = PulseMitigatorConfig(target_qubit=3)
        pm = PulseMitigator(config=cfg)

        iq, det = _make_synthetic_iq_data(n=30)
        pm.calibrate(iq, det)

        qc = QuantumCircuit(5)
        qc.h(range(5))
        result = pm.run_with_active_cancellation(qc, probe_i=0.005, probe_q=-0.003)

        # ml_rx should be on qubit 3
        corrected = result.corrected_circuit
        for inst in corrected.data:
            if inst.operation.name == "ml_rx":
                qubit_indices = [corrected.find_bit(q).index for q in inst.qubits]
                assert 3 in qubit_indices

    def test_sequential_predictions_vary(self):
        """Sequential predictions with different IQ points should differ."""
        iq, det = _make_synthetic_iq_data(n=100, seed=99)
        pm = PulseMitigator()
        pm.calibrate(iq, det)

        pred1 = pm.predict_drift(i=0.010, q=-0.005)
        pred2 = pm.predict_drift(i=-0.003, q=0.008)

        # Different IQ points should yield different predictions
        assert pred1.predicted_drift_hz != pytest.approx(
            pred2.predicted_drift_hz, abs=1.0
        )
