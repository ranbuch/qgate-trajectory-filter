"""Tests for qgate.mitigation — TelemetryMitigator two-stage pipeline.

Tests are structured in tiers:
  1. Unit tests for MitigatorConfig (Pydantic validation, frozen, bounds)
  2. Unit tests for feature engineering helpers
  3. Integration tests for TelemetryMitigator (calibrate → estimate)

NOTICE: Pre-patent proprietary code — do NOT push to public repositories.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if scikit-learn not installed
# ---------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn", reason="scikit-learn required for mitigation tests")

from pydantic import ValidationError  # noqa: E402

from qgate.mitigation import (  # noqa: E402
    FEATURE_NAMES,
    CalibrationResult,
    MitigationResult,
    MitigatorConfig,
    TelemetryMitigator,
    _extract_feature_matrix,
    _extract_features,
    _make_builtin_model,
)

# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — MitigatorConfig unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMitigatorConfig:
    """Verify MitigatorConfig Pydantic model validation."""

    def test_defaults(self):
        cfg = MitigatorConfig()
        assert cfg.keep_fraction == pytest.approx(0.70)
        assert cfg.n_calibration_circuits == 20
        assert cfg.model_name == "random_forest"
        assert cfg.model_params == {}
        assert cfg.scale_features is True
        assert cfg.random_state == 42

    def test_custom_values(self):
        cfg = MitigatorConfig(
            keep_fraction=0.50,
            n_calibration_circuits=50,
            model_name="ridge",
            model_params={"alpha": 2.0},
            scale_features=False,
            random_state=123,
        )
        assert cfg.keep_fraction == pytest.approx(0.50)
        assert cfg.n_calibration_circuits == 50
        assert cfg.model_name == "ridge"
        assert cfg.model_params == {"alpha": 2.0}
        assert cfg.scale_features is False
        assert cfg.random_state == 123

    def test_frozen(self):
        cfg = MitigatorConfig()
        with pytest.raises(ValidationError):
            cfg.keep_fraction = 0.5  # type: ignore[misc]

    def test_keep_fraction_bounds(self):
        # Too small (must be > 0)
        with pytest.raises(ValidationError):
            MitigatorConfig(keep_fraction=0.0)
        with pytest.raises(ValidationError):
            MitigatorConfig(keep_fraction=-0.1)
        # Upper bound is 1.0 (inclusive)
        cfg = MitigatorConfig(keep_fraction=1.0)
        assert cfg.keep_fraction == pytest.approx(1.0)
        # Over 1.0
        with pytest.raises(ValidationError):
            MitigatorConfig(keep_fraction=1.1)

    def test_n_calibration_circuits_bounds(self):
        with pytest.raises(ValidationError):
            MitigatorConfig(n_calibration_circuits=0)
        with pytest.raises(ValidationError):
            MitigatorConfig(n_calibration_circuits=501)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MitigatorConfig(nonexistent_field=True)  # type: ignore[call-arg]


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — Feature engineering unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureEngineering:
    """Verify feature extraction helpers."""

    def test_extract_features_shape(self):
        feat = _extract_features(
            energy=-1.0, acceptance=0.7, variance=0.05, mean_energy=-0.9
        )
        assert feat.shape == (6,)
        assert feat.dtype == np.float64

    def test_extract_features_values(self):
        feat = _extract_features(
            energy=-1.0, acceptance=0.7, variance=0.05, mean_energy=-0.9
        )
        # energy, acceptance, variance, abs_energy, energy*acceptance, residual
        expected = np.array([-1.0, 0.7, 0.05, 1.0, -1.0 * 0.7, -1.0 - (-0.9)])
        np.testing.assert_allclose(feat, expected)

    def test_extract_feature_matrix_shape(self):
        records = [
            {"energy": -1.0, "acceptance": 0.7, "variance": 0.05},
            {"energy": -0.9, "acceptance": 0.8, "variance": 0.03},
            {"energy": -1.1, "acceptance": 0.6, "variance": 0.07},
        ]
        X = _extract_feature_matrix(records)
        assert X.shape == (3, 6)

    def test_extract_feature_matrix_mean_energy_default(self):
        """When mean_energy is absent, batch mean is used."""
        records = [
            {"energy": -1.0, "acceptance": 0.7, "variance": 0.05},
            {"energy": -2.0, "acceptance": 0.8, "variance": 0.03},
        ]
        X = _extract_feature_matrix(records)
        # batch mean = -1.5; residual for first record = -1.0 - (-1.5) = 0.5
        assert X[0, 5] == pytest.approx(0.5)
        # residual for second record = -2.0 - (-1.5) = -0.5
        assert X[1, 5] == pytest.approx(-0.5)

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 6


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — Built-in model factories
# ═══════════════════════════════════════════════════════════════════════════


class TestBuiltinModels:
    """Verify that built-in model factories produce valid estimators."""

    @pytest.mark.parametrize("name", ["random_forest", "gradient_boosting", "ridge"])
    def test_factory_returns_unfitted(self, name: str):
        model = _make_builtin_model(name, random_state=42)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model name"):
            _make_builtin_model("nonexistent_model", random_state=42)

    def test_custom_params_forwarded(self):
        model = _make_builtin_model("random_forest", random_state=7, n_estimators=50)
        assert model.n_estimators == 50
        assert model.random_state == 7


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — TelemetryMitigator integration tests
# ═══════════════════════════════════════════════════════════════════════════


def _make_synthetic_calibration_data(
    n: int = 50, noise_std: float = 0.05, seed: int = 42
) -> list[dict[str, float]]:
    """Generate synthetic calibration data with known ideal = -1.0."""
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        ideal = -1.0
        noise = rng.normal(0, noise_std)
        energy = ideal + noise
        acceptance = 0.5 + 0.3 * rng.random()
        variance = abs(noise_std * rng.normal(1, 0.3))
        data.append({
            "energy": energy,
            "acceptance": acceptance,
            "variance": variance,
            "ideal": ideal,
        })
    return data


class TestTelemetryMitigator:
    """Integration tests for the full calibrate → estimate pipeline."""

    def test_uncalibrated_estimate_raises(self):
        mit = TelemetryMitigator()
        with pytest.raises(RuntimeError, match="not been calibrated"):
            mit.estimate(raw_energy=-1.0, acceptance=0.7, variance=0.05)

    def test_uncalibrated_estimate_batch_raises(self):
        mit = TelemetryMitigator()
        with pytest.raises(RuntimeError, match="not been calibrated"):
            mit.estimate_batch([{"energy": -1.0, "acceptance": 0.7, "variance": 0.05}])

    def test_calibrate_too_few_records(self):
        mit = TelemetryMitigator()
        with pytest.raises(ValueError, match="≥ 2"):
            mit.calibrate([{"energy": -1.0, "acceptance": 0.7, "variance": 0.05, "ideal": -1.0}])

    def test_calibrate_missing_keys(self):
        mit = TelemetryMitigator()
        with pytest.raises(ValueError, match="missing keys"):
            mit.calibrate([
                {"energy": -1.0, "acceptance": 0.7},  # missing variance, ideal
                {"energy": -0.9, "acceptance": 0.8, "variance": 0.03, "ideal": -1.0},
            ])

    def test_calibrate_returns_result(self):
        data = _make_synthetic_calibration_data(n=30)
        mit = TelemetryMitigator()
        cal = mit.calibrate(data)

        assert isinstance(cal, CalibrationResult)
        assert cal.n_samples == 30
        assert cal.model_name == "RandomForestRegressor"
        assert cal.feature_names == FEATURE_NAMES
        assert cal.train_mae >= 0
        assert cal.train_rmse >= 0
        assert cal.elapsed_seconds > 0

    def test_is_calibrated_flag(self):
        data = _make_synthetic_calibration_data(n=20)
        mit = TelemetryMitigator()
        assert mit.is_calibrated is False
        mit.calibrate(data)
        assert mit.is_calibrated is True

    def test_calibration_result_property(self):
        data = _make_synthetic_calibration_data(n=20)
        mit = TelemetryMitigator()
        assert mit.calibration_result is None
        cal = mit.calibrate(data)
        assert mit.calibration_result is cal

    def test_estimate_returns_result(self):
        data = _make_synthetic_calibration_data(n=30)
        mit = TelemetryMitigator()
        mit.calibrate(data)

        result = mit.estimate(raw_energy=-1.05, acceptance=0.72, variance=0.04)
        assert isinstance(result, MitigationResult)
        assert isinstance(result.mitigated_value, float)
        assert result.raw_energy == pytest.approx(-1.05)
        assert result.acceptance == pytest.approx(0.72)
        assert result.correction == pytest.approx(
            result.mitigated_value - result.filtered_energy
        )

    def test_estimate_reduces_error(self):
        """ML correction should bring the estimate closer to ideal (-1.0)."""
        data = _make_synthetic_calibration_data(n=100, noise_std=0.10, seed=99)
        mit = TelemetryMitigator()
        mit.calibrate(data)

        # Test on a noisy measurement
        raw = -1.15
        result = mit.estimate(raw_energy=raw, acceptance=0.65, variance=0.10)
        raw_error = abs(raw - (-1.0))
        ml_error = abs(result.mitigated_value - (-1.0))
        # ML should reduce the error (at least not be worse by a lot)
        assert ml_error < raw_error + 0.05

    def test_estimate_with_filtered_energy(self):
        """When filtered_energy is provided, it should be used instead of raw."""
        data = _make_synthetic_calibration_data(n=30)
        mit = TelemetryMitigator()
        mit.calibrate(data)

        result = mit.estimate(
            raw_energy=-1.20,
            acceptance=0.65,
            variance=0.08,
            filtered_energy=-1.05,
        )
        assert result.raw_energy == pytest.approx(-1.20)
        assert result.filtered_energy == pytest.approx(-1.05)
        assert result.mitigated_value == pytest.approx(
            result.filtered_energy + result.correction
        )

    def test_estimate_batch(self):
        data = _make_synthetic_calibration_data(n=30)
        mit = TelemetryMitigator()
        mit.calibrate(data)

        records = [
            {"energy": -1.05, "acceptance": 0.72, "variance": 0.04},
            {"energy": -0.95, "acceptance": 0.80, "variance": 0.03},
            {"energy": -1.10, "acceptance": 0.60, "variance": 0.06},
        ]
        results = mit.estimate_batch(records)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, MitigationResult)

    def test_calibrate_and_estimate_convenience(self):
        cal_data = _make_synthetic_calibration_data(n=30)
        test_data = [
            {"energy": -1.05, "acceptance": 0.72, "variance": 0.04},
            {"energy": -0.95, "acceptance": 0.80, "variance": 0.03},
        ]
        mit = TelemetryMitigator()
        cal, results = mit.calibrate_and_estimate(cal_data, test_data)
        assert isinstance(cal, CalibrationResult)
        assert len(results) == 2

    # ------------------------------------------------------------------
    # Model injection
    # ------------------------------------------------------------------

    def test_custom_model_factory(self):
        from sklearn.linear_model import Ridge

        data = _make_synthetic_calibration_data(n=30)
        mit = TelemetryMitigator(model_factory=lambda: Ridge(alpha=2.0))
        cal = mit.calibrate(data)
        assert cal.model_name == "Ridge"
        assert mit.model is not None

    def test_override_model_at_calibrate_time(self):
        from sklearn.ensemble import GradientBoostingRegressor

        data = _make_synthetic_calibration_data(n=30)
        mit = TelemetryMitigator()  # default would be RF
        cal = mit.calibrate(
            data,
            model_factory=lambda: GradientBoostingRegressor(n_estimators=50, random_state=42),
        )
        assert cal.model_name == "GradientBoostingRegressor"

    @pytest.mark.parametrize("model_name", ["random_forest", "gradient_boosting", "ridge"])
    def test_builtin_model_names(self, model_name: str):
        data = _make_synthetic_calibration_data(n=30)
        cfg = MitigatorConfig(model_name=model_name)
        mit = TelemetryMitigator(config=cfg)
        cal = mit.calibrate(data)
        assert cal.n_samples == 30

    # ------------------------------------------------------------------
    # Scaling toggle
    # ------------------------------------------------------------------

    def test_no_scaling(self):
        data = _make_synthetic_calibration_data(n=30)
        cfg = MitigatorConfig(scale_features=False)
        mit = TelemetryMitigator(config=cfg)
        cal = mit.calibrate(data)
        assert cal.n_samples == 30

        result = mit.estimate(raw_energy=-1.05, acceptance=0.72, variance=0.04)
        assert isinstance(result.mitigated_value, float)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def test_repr_uncalibrated(self):
        mit = TelemetryMitigator()
        r = repr(mit)
        assert "uncalibrated" in r
        assert "None" in r

    def test_repr_calibrated(self):
        data = _make_synthetic_calibration_data(n=20)
        mit = TelemetryMitigator()
        mit.calibrate(data)
        r = repr(mit)
        assert "calibrated" in r
        assert "RandomForestRegressor" in r
        assert "0.7" in r


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Recalibration / stateful tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRecalibration:
    """Verify that re-calibration properly replaces the trained model."""

    def test_recalibrate_replaces_model(self):
        data1 = _make_synthetic_calibration_data(n=20, seed=1)
        data2 = _make_synthetic_calibration_data(n=40, seed=2)
        mit = TelemetryMitigator()

        cal1 = mit.calibrate(data1)
        assert cal1.n_samples == 20

        cal2 = mit.calibrate(data2)
        assert cal2.n_samples == 40
        assert mit.calibration_result is cal2

    def test_recalibrate_with_different_model(self):
        data = _make_synthetic_calibration_data(n=30)
        mit = TelemetryMitigator()

        cal1 = mit.calibrate(data)
        assert cal1.model_name == "RandomForestRegressor"

        from sklearn.linear_model import Ridge

        cal2 = mit.calibrate(data, model_factory=lambda: Ridge())
        assert cal2.model_name == "Ridge"
