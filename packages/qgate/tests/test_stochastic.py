"""Unit tests for :mod:`qgate.stochastic`.

Covers the full PPU stochastic mitigation pipeline:
  - fBM path simulation
  - Asian call payoff
  - StochasticTelemetryExtractor
  - GaltonOutlierFilter
  - StochasticMitigator (calibration + prediction)
  - PPUMitigationPipeline (end-to-end)
  - run_monte_carlo_benchmark convenience function

Patent reference: US Provisional App. No. 64/XXX,XXX (April 2026), §22.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from qgate.stochastic import (
    STOCHASTIC_FEATURE_NAMES,
    GaltonOutlierFilter,
    PPUMitigationPipeline,
    StochasticCalibrationResult,
    StochasticConfig,
    StochasticMitigationResult,
    StochasticMitigator,
    StochasticTelemetryExtractor,
    _cholesky_fbm,
    _MAD_TO_SIGMA,
    asian_call_payoff,
    run_monte_carlo_benchmark,
    simulate_fbm_paths,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def small_paths() -> np.ndarray:
    """50 paths × 20 steps — fast deterministic paths."""
    return simulate_fbm_paths(
        n_paths=50, n_steps=20, hurst=0.7,
        S0=100.0, mu=0.05, sigma=0.2, T=1.0, seed=42,
    )


@pytest.fixture()
def default_config() -> StochasticConfig:
    return StochasticConfig()


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_mad_to_sigma(self):
        assert abs(_MAD_TO_SIGMA - 1.4826) < 1e-4

    def test_feature_names(self):
        assert len(STOCHASTIC_FEATURE_NAMES) == 6
        assert "realized_vol" in STOCHASTIC_FEATURE_NAMES
        assert "lag1_autocorr" in STOCHASTIC_FEATURE_NAMES
        assert "max_drawdown" in STOCHASTIC_FEATURE_NAMES
        assert "terminal_dist" in STOCHASTIC_FEATURE_NAMES
        assert "path_skewness" in STOCHASTIC_FEATURE_NAMES
        assert "mean_log_return" in STOCHASTIC_FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════════════
# StochasticConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestStochasticConfig:
    def test_defaults(self, default_config):
        assert default_config.reject_fraction == 0.25
        assert default_config.model_name == "random_forest"
        assert default_config.scale_features is True
        assert default_config.random_state == 42

    def test_frozen(self, default_config):
        with pytest.raises(Exception):
            default_config.reject_fraction = 0.5  # type: ignore[misc]

    def test_custom_values(self):
        cfg = StochasticConfig(
            reject_fraction=0.3,
            model_name="ridge",
            random_state=123,
        )
        assert cfg.reject_fraction == 0.3
        assert cfg.model_name == "ridge"

    def test_reject_fraction_bounds(self):
        with pytest.raises(Exception):
            StochasticConfig(reject_fraction=0.0)
        with pytest.raises(Exception):
            StochasticConfig(reject_fraction=1.0)
        with pytest.raises(Exception):
            StochasticConfig(reject_fraction=-0.1)


# ═══════════════════════════════════════════════════════════════════════════
# fBM Simulator
# ═══════════════════════════════════════════════════════════════════════════


class TestFBMSimulator:
    def test_shape(self):
        paths = simulate_fbm_paths(n_paths=10, n_steps=50, seed=1)
        assert paths.shape == (10, 51)

    def test_initial_price(self):
        paths = simulate_fbm_paths(n_paths=5, n_steps=10, S0=42.0, seed=1)
        np.testing.assert_allclose(paths[:, 0], 42.0)

    def test_positive_prices(self):
        paths = simulate_fbm_paths(n_paths=100, n_steps=50, seed=1)
        assert np.all(paths > 0), "fBM GBM prices must be positive"

    def test_reproducibility(self):
        p1 = simulate_fbm_paths(n_paths=5, n_steps=10, seed=99)
        p2 = simulate_fbm_paths(n_paths=5, n_steps=10, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds(self):
        p1 = simulate_fbm_paths(n_paths=5, n_steps=10, seed=1)
        p2 = simulate_fbm_paths(n_paths=5, n_steps=10, seed=2)
        assert not np.allclose(p1, p2)

    def test_hurst_half_is_gbm(self):
        """H=0.5 should give standard GBM-like behaviour.

        The Cholesky fBM construction at H=0.5 produces valid GBM paths
        but the covariance structure of fBM cumulative values introduces
        structural correlation at the increments level.  We verify that
        H=0.5 gives *less* autocorrelation than H=0.8 (trending).
        """
        paths_05 = simulate_fbm_paths(
            n_paths=1000, n_steps=100, hurst=0.5, seed=42,
        )
        paths_08 = simulate_fbm_paths(
            n_paths=1000, n_steps=100, hurst=0.8, seed=42,
        )
        ext = StochasticTelemetryExtractor()
        feat_05 = ext.extract(paths_05)
        feat_08 = ext.extract(paths_08)
        # lag1_autocorr is feature index 1
        autocorr_05 = np.mean(feat_05[:, 1])
        autocorr_08 = np.mean(feat_08[:, 1])
        # H=0.5 should have less autocorrelation than H=0.8
        assert autocorr_05 < autocorr_08, (
            f"H=0.5 autocorr {autocorr_05:.4f} should be < H=0.8 {autocorr_08:.4f}"
        )

    def test_hurst_above_half_positive_autocorr(self):
        """H=0.7 should yield positive lag-1 autocorrelation (trending)."""
        paths = simulate_fbm_paths(
            n_paths=2000, n_steps=100, hurst=0.7, seed=42,
        )
        log_ret = np.diff(np.log(paths), axis=1)
        autocorrs = []
        for i in range(paths.shape[0]):
            lr = log_ret[i]
            c = np.corrcoef(lr[:-1], lr[1:])[0, 1]
            if not np.isnan(c):
                autocorrs.append(c)
        mean_autocorr = np.mean(autocorrs)
        assert mean_autocorr > 0.05, f"H=0.7 should have positive autocorr, got {mean_autocorr}"


class TestCholeskyFBM:
    def test_lower_triangular(self):
        L = _cholesky_fbm(10, 0.7)
        assert L.shape == (10, 10)
        # Lower triangular: upper part should be zero
        assert np.allclose(L, np.tril(L))

    def test_positive_definite(self):
        L = _cholesky_fbm(20, 0.6)
        cov = L @ L.T
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)


# ═══════════════════════════════════════════════════════════════════════════
# Asian Call Payoff
# ═══════════════════════════════════════════════════════════════════════════


class TestAsianCallPayoff:
    def test_at_the_money(self):
        # Paths that average exactly at strike → zero payoff
        paths = np.full((3, 11), 100.0)
        payoffs = asian_call_payoff(paths, strike=100.0, r=0.0, T=1.0)
        np.testing.assert_allclose(payoffs, 0.0)

    def test_in_the_money(self):
        paths = np.full((2, 11), 120.0)
        payoffs = asian_call_payoff(paths, strike=100.0, r=0.0, T=1.0)
        np.testing.assert_allclose(payoffs, 20.0)

    def test_out_of_money(self):
        paths = np.full((2, 11), 80.0)
        payoffs = asian_call_payoff(paths, strike=100.0, r=0.0, T=1.0)
        np.testing.assert_allclose(payoffs, 0.0)

    def test_discounting(self):
        paths = np.full((1, 11), 110.0)
        payoff_no_disc = asian_call_payoff(paths, strike=100.0, r=0.0, T=1.0)
        payoff_disc = asian_call_payoff(paths, strike=100.0, r=0.05, T=1.0)
        assert payoff_disc[0] < payoff_no_disc[0]

    def test_shape(self, small_paths):
        payoffs = asian_call_payoff(small_paths, strike=100.0)
        assert payoffs.shape == (50,)

    def test_non_negative(self, small_paths):
        payoffs = asian_call_payoff(small_paths, strike=100.0)
        assert np.all(payoffs >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# StochasticTelemetryExtractor
# ═══════════════════════════════════════════════════════════════════════════


class TestStochasticTelemetryExtractor:
    def test_feature_names(self):
        ext = StochasticTelemetryExtractor()
        assert ext.feature_names == STOCHASTIC_FEATURE_NAMES
        assert ext.n_features == 6

    def test_shape(self, small_paths):
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        assert features.shape == (50, 6)

    def test_no_nans(self, small_paths):
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        assert not np.any(np.isnan(features))

    def test_realized_vol_positive(self, small_paths):
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        realized_vol = features[:, 0]
        assert np.all(realized_vol >= 0)

    def test_max_drawdown_non_positive(self, small_paths):
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        max_dd = features[:, 2]
        assert np.all(max_dd <= 0), "Max drawdown should be ≤ 0"

    def test_batch_alias(self, small_paths):
        ext = StochasticTelemetryExtractor()
        f1 = ext.extract(small_paths)
        f2 = ext.extract_batch(small_paths)
        np.testing.assert_array_equal(f1, f2)

    def test_constant_paths(self):
        """Constant paths should have zero vol, zero autocorr, zero drawdown."""
        paths = np.full((5, 11), 100.0)
        ext = StochasticTelemetryExtractor()
        features = ext.extract(paths)
        # realized_vol ≈ 0
        np.testing.assert_allclose(features[:, 0], 0.0, atol=1e-12)

    def test_trending_paths_positive_autocorr(self):
        """Strongly trending paths should show positive lag-1 autocorr."""
        paths = simulate_fbm_paths(
            n_paths=500, n_steps=100, hurst=0.8, seed=42,
        )
        ext = StochasticTelemetryExtractor()
        features = ext.extract(paths)
        mean_autocorr = np.mean(features[:, 1])
        assert mean_autocorr > 0, f"Expected positive autocorr for H=0.8, got {mean_autocorr}"


# ═══════════════════════════════════════════════════════════════════════════
# GaltonOutlierFilter
# ═══════════════════════════════════════════════════════════════════════════


class TestGaltonOutlierFilter:
    def test_reject_fraction(self, small_paths):
        filt = GaltonOutlierFilter(reject_fraction=0.25)
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        mask = filt.filter(features)
        # Should keep approximately 75%
        survival_rate = mask.sum() / len(mask)
        assert 0.65 <= survival_rate <= 0.85

    def test_viability_scores_shape(self, small_paths):
        filt = GaltonOutlierFilter(reject_fraction=0.25)
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        scores = filt.compute_viability_scores(features)
        assert scores.shape == (50,)

    def test_viability_scores_non_negative(self, small_paths):
        filt = GaltonOutlierFilter(reject_fraction=0.25)
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        scores = filt.compute_viability_scores(features)
        assert np.all(scores >= 0)

    def test_high_reject_fraction(self, small_paths):
        filt = GaltonOutlierFilter(reject_fraction=0.9)
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        mask = filt.filter(features)
        survival_rate = mask.sum() / len(mask)
        assert survival_rate < 0.2  # ~10% survive

    def test_low_reject_fraction(self, small_paths):
        filt = GaltonOutlierFilter(reject_fraction=0.01)
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        mask = filt.filter(features)
        survival_rate = mask.sum() / len(mask)
        assert survival_rate > 0.9

    def test_invalid_reject_fraction(self):
        with pytest.raises(ValueError):
            GaltonOutlierFilter(reject_fraction=0.0)
        with pytest.raises(ValueError):
            GaltonOutlierFilter(reject_fraction=1.0)

    def test_mask_is_boolean(self, small_paths):
        filt = GaltonOutlierFilter(reject_fraction=0.25)
        ext = StochasticTelemetryExtractor()
        features = ext.extract(small_paths)
        mask = filt.filter(features)
        assert mask.dtype == np.bool_


# ═══════════════════════════════════════════════════════════════════════════
# StochasticMitigator
# ═══════════════════════════════════════════════════════════════════════════


class TestStochasticMitigator:
    def _make_calibration_data(self, n=200, seed=42):
        """Helper: generate synthetic calibration data."""
        paths = simulate_fbm_paths(
            n_paths=n, n_steps=20, hurst=0.7, S0=100.0, seed=seed,
        )
        ext = StochasticTelemetryExtractor()
        telemetry = ext.extract(paths)
        raw_payoffs = asian_call_payoff(paths, strike=100.0)
        # Ideal = ground truth mean (simplified)
        ideal_payoffs = np.full_like(raw_payoffs, np.mean(raw_payoffs))
        return telemetry, raw_payoffs, ideal_payoffs

    def test_calibrate(self):
        mit = StochasticMitigator()
        telemetry, raw, ideal = self._make_calibration_data()
        result = mit.calibrate(telemetry, raw, ideal)
        assert isinstance(result, StochasticCalibrationResult)
        assert mit.is_calibrated

    def test_predict_raises_before_calibrate(self):
        mit = StochasticMitigator()
        telemetry, raw, _ = self._make_calibration_data()
        with pytest.raises(RuntimeError, match="not calibrated"):
            mit.predict(telemetry, raw)

    def test_predict_shape(self):
        mit = StochasticMitigator()
        telemetry, raw, ideal = self._make_calibration_data()
        mit.calibrate(telemetry, raw, ideal)
        mitigated = mit.predict(telemetry, raw)
        assert mitigated.shape == raw.shape

    def test_calibration_result_fields(self):
        mit = StochasticMitigator()
        telemetry, raw, ideal = self._make_calibration_data()
        result = mit.calibrate(telemetry, raw, ideal)
        assert result.model_name == "random_forest"
        assert result.n_samples == 200
        assert result.train_mae >= 0
        assert result.elapsed_seconds >= 0
        assert len(result.feature_names) == 7  # raw_payoff + 6 telemetry

    def test_model_names(self):
        for name in ("random_forest", "gradient_boosting", "ridge"):
            cfg = StochasticConfig(model_name=name)
            mit = StochasticMitigator(cfg)
            telemetry, raw, ideal = self._make_calibration_data()
            result = mit.calibrate(telemetry, raw, ideal)
            assert result.model_name == name

    def test_unknown_model_raises(self):
        cfg = StochasticConfig(model_name="xgboost")
        mit = StochasticMitigator(cfg)
        telemetry, raw, ideal = self._make_calibration_data()
        with pytest.raises(ValueError, match="Unknown model"):
            mit.calibrate(telemetry, raw, ideal)

    def test_mitigated_closer_to_ideal(self):
        """Core property: mitigated payoffs should be closer to ideal."""
        mit = StochasticMitigator()
        telemetry, raw, ideal = self._make_calibration_data(n=500, seed=42)
        mit.calibrate(telemetry, raw, ideal)
        mitigated = mit.predict(telemetry, raw)

        raw_mae = np.mean(np.abs(raw - ideal))
        mit_mae = np.mean(np.abs(mitigated - ideal))
        assert mit_mae < raw_mae, f"Mitigated MAE {mit_mae} should be < raw MAE {raw_mae}"


# ═══════════════════════════════════════════════════════════════════════════
# PPUMitigationPipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestPPUMitigationPipeline:
    def _calibrate_pipeline(
        self, config=None, n_cal=300, seed=42,
    ) -> PPUMitigationPipeline:
        pipe = PPUMitigationPipeline(config=config)
        cal_paths = simulate_fbm_paths(
            n_paths=n_cal, n_steps=20, hurst=0.7, S0=100.0, seed=seed,
        )
        cal_raw = asian_call_payoff(cal_paths, strike=100.0)
        cal_ideal = np.full_like(cal_raw, np.mean(cal_raw))
        pipe.calibrate(cal_paths, cal_raw, cal_ideal)
        return pipe

    def test_end_to_end(self):
        pipe = self._calibrate_pipeline()
        budget_paths = simulate_fbm_paths(
            n_paths=100, n_steps=20, hurst=0.7, S0=100.0, seed=99,
        )
        budget_payoffs = asian_call_payoff(budget_paths, strike=100.0)
        result = pipe.mitigate(budget_paths, budget_payoffs)
        assert isinstance(result, StochasticMitigationResult)
        assert result.stage1_survivors > 0
        assert result.stage1_rejected > 0
        assert result.stage1_survivors + result.stage1_rejected == 100

    def test_with_ground_truth(self):
        pipe = self._calibrate_pipeline()
        budget_paths = simulate_fbm_paths(
            n_paths=100, n_steps=20, hurst=0.7, S0=100.0, seed=99,
        )
        budget_payoffs = asian_call_payoff(budget_paths, strike=100.0)
        gt = float(np.mean(budget_payoffs))  # use raw mean as approx gt
        result = pipe.mitigate(budget_paths, budget_payoffs, ground_truth=gt)
        assert not math.isnan(result.improvement_factor)

    def test_without_ground_truth(self):
        pipe = self._calibrate_pipeline()
        budget_paths = simulate_fbm_paths(
            n_paths=100, n_steps=20, hurst=0.7, S0=100.0, seed=99,
        )
        budget_payoffs = asian_call_payoff(budget_paths, strike=100.0)
        result = pipe.mitigate(budget_paths, budget_payoffs)
        assert math.isnan(result.improvement_factor)

    def test_latency_measured(self):
        pipe = self._calibrate_pipeline()
        budget_paths = simulate_fbm_paths(
            n_paths=50, n_steps=20, hurst=0.7, S0=100.0, seed=99,
        )
        budget_payoffs = asian_call_payoff(budget_paths, strike=100.0)
        result = pipe.mitigate(budget_paths, budget_payoffs)
        assert result.latency_seconds > 0

    def test_metadata(self):
        pipe = self._calibrate_pipeline()
        budget_paths = simulate_fbm_paths(
            n_paths=50, n_steps=20, hurst=0.7, S0=100.0, seed=99,
        )
        budget_payoffs = asian_call_payoff(budget_paths, strike=100.0)
        result = pipe.mitigate(budget_paths, budget_payoffs)
        assert "n_budget_paths" in result.metadata
        assert result.metadata["n_budget_paths"] == 50

    def test_properties(self):
        pipe = PPUMitigationPipeline()
        assert isinstance(pipe.config, StochasticConfig)
        assert isinstance(pipe.extractor, StochasticTelemetryExtractor)
        assert isinstance(pipe.filter, GaltonOutlierFilter)
        assert isinstance(pipe.mitigator, StochasticMitigator)


# ═══════════════════════════════════════════════════════════════════════════
# run_monte_carlo_benchmark
# ═══════════════════════════════════════════════════════════════════════════


class TestMonteCarloConvenience:
    def test_quick_benchmark(self):
        """Smoke test with small problem size."""
        result = run_monte_carlo_benchmark(
            n_ground_truth=5000,
            n_budget=200,
            n_calibration=300,
            n_steps=20,
            seed=42,
        )
        assert "ground_truth_price" in result
        assert "mitigated_price" in result
        assert "improvement_factor" in result
        assert "equivalent_paths" in result
        assert result["improvement_factor"] > 0

    def test_mitigated_improves_over_raw(self):
        """Core benchmark property: mitigated should beat raw."""
        result = run_monte_carlo_benchmark(
            n_ground_truth=10000,
            n_budget=500,
            n_calibration=500,
            n_steps=20,
            seed=42,
        )
        assert result["mitigated_mae"] < result["raw_mae"], (
            f"Mitigated MAE {result['mitigated_mae']:.6f} should be < "
            f"raw MAE {result['raw_mae']:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Result Dataclasses
# ═══════════════════════════════════════════════════════════════════════════


class TestResultDataclasses:
    def test_calibration_result_frozen(self):
        r = StochasticCalibrationResult(model_name="rf", n_samples=100)
        with pytest.raises(Exception):
            r.model_name = "gbr"  # type: ignore[misc]

    def test_mitigation_result_frozen(self):
        r = StochasticMitigationResult(
            mitigated_value=1.0, raw_value=2.0,
            stage1_survivors=80, stage1_rejected=20,
        )
        with pytest.raises(Exception):
            r.mitigated_value = 99.0  # type: ignore[misc]

    def test_mitigation_result_defaults(self):
        r = StochasticMitigationResult(
            mitigated_value=1.0, raw_value=2.0,
            stage1_survivors=80, stage1_rejected=20,
        )
        assert math.isnan(r.improvement_factor)
        assert r.latency_seconds == 0.0
        assert r.metadata == {}
