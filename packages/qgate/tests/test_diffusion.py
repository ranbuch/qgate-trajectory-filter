"""Unit tests for :mod:`qgate.diffusion`.

Covers the full PPU diffusion latent mitigation pipeline:
  - Mock diffusion latent simulator
  - Latent-space metric proxies (FID, CLIP, PSNR)
  - LatentTelemetryExtractor
  - GaltonLatentFilter (Stage 1 Galton rejection)
  - DiffusionMitigator (Stage 2 ML reconstruction)
  - DiffusionMitigationPipeline (end-to-end orchestrator)
  - run_diffusion_benchmark convenience function

Patent reference: US Provisional App. No. 64/XXX,XXX (April 2026), §22.
CIP — PPU generalization to generative AI diffusion models.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from qgate.diffusion import (
    DEFAULT_LATENT_CHANNELS,
    DEFAULT_LATENT_HEIGHT,
    DEFAULT_LATENT_WIDTH,
    LATENT_FEATURE_NAMES,
    DiffusionCalibrationResult,
    DiffusionConfig,
    DiffusionMitigationPipeline,
    DiffusionMitigationResult,
    DiffusionMitigator,
    GaltonLatentFilter,
    LatentTelemetryExtractor,
    _MAD_TO_SIGMA,
    _prompt_to_seed,
    compute_clip_score,
    compute_latent_fid,
    compute_psnr,
    run_diffusion_benchmark,
    simulate_diffusion_latents,
)


# Small latent shape for fast tests
_C = 4
_H = 16
_W = 16
_PROMPT = "macro gears ruby bearings photorealistic"


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def small_latents() -> np.ndarray:
    """12 latent trajectories at 10 steps — fast deterministic batch."""
    return simulate_diffusion_latents(
        prompt=_PROMPT, n_trajectories=12, num_steps=10,
        latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
    )


@pytest.fixture()
def gt_latent() -> np.ndarray:
    """Single ground truth latent at 50 steps."""
    return simulate_diffusion_latents(
        prompt=_PROMPT, n_trajectories=1, num_steps=50,
        latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
    )[0]


@pytest.fixture()
def default_config() -> DiffusionConfig:
    return DiffusionConfig(
        latent_channels=_C, latent_height=_H, latent_width=_W,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_mad_to_sigma(self):
        assert abs(_MAD_TO_SIGMA - 1.4826) < 1e-4

    def test_feature_names(self):
        assert len(LATENT_FEATURE_NAMES) == 6
        assert "spatial_energy" in LATENT_FEATURE_NAMES
        assert "channel_mean_std" in LATENT_FEATURE_NAMES
        assert "channel_var_mean" in LATENT_FEATURE_NAMES
        assert "high_freq_ratio" in LATENT_FEATURE_NAMES
        assert "spatial_autocorr" in LATENT_FEATURE_NAMES
        assert "kurtosis" in LATENT_FEATURE_NAMES

    def test_default_latent_shape(self):
        assert DEFAULT_LATENT_CHANNELS == 4
        assert DEFAULT_LATENT_HEIGHT == 64
        assert DEFAULT_LATENT_WIDTH == 64


# ═══════════════════════════════════════════════════════════════════════════
# DiffusionConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffusionConfig:
    def test_defaults(self, default_config):
        assert default_config.reject_fraction == 0.25
        assert default_config.model_name == "random_forest"
        assert default_config.scale_features is True
        assert default_config.random_state == 42

    def test_frozen(self, default_config):
        with pytest.raises(Exception):
            default_config.reject_fraction = 0.5  # type: ignore[misc]

    def test_custom_values(self):
        cfg = DiffusionConfig(
            reject_fraction=0.3,
            model_name="ridge",
            random_state=123,
            latent_channels=8,
        )
        assert cfg.reject_fraction == 0.3
        assert cfg.model_name == "ridge"
        assert cfg.latent_channels == 8

    def test_reject_fraction_bounds(self):
        with pytest.raises(Exception):
            DiffusionConfig(reject_fraction=0.0)
        with pytest.raises(Exception):
            DiffusionConfig(reject_fraction=1.0)
        with pytest.raises(Exception):
            DiffusionConfig(reject_fraction=-0.1)

    def test_latent_channels_positive(self):
        with pytest.raises(Exception):
            DiffusionConfig(latent_channels=0)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt-to-Seed Utility
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptToSeed:
    def test_deterministic(self):
        s1 = _prompt_to_seed("hello world")
        s2 = _prompt_to_seed("hello world")
        assert s1 == s2

    def test_different_prompts(self):
        s1 = _prompt_to_seed("macro gears")
        s2 = _prompt_to_seed("sunset ocean")
        assert s1 != s2

    def test_returns_int(self):
        s = _prompt_to_seed("test")
        assert isinstance(s, int)
        assert s >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion Latent Simulator
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffusionSimulator:
    def test_shape(self):
        latents = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=5, num_steps=20,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=1,
        )
        assert latents.shape == (5, _C, _H, _W)

    def test_single_trajectory(self):
        latents = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=1, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=1,
        )
        assert latents.shape == (1, _C, _H, _W)

    def test_reproducibility(self):
        l1 = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=3, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        l2 = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=3, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds(self):
        l1 = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=3, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=1,
        )
        l2 = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=3, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=2,
        )
        assert not np.allclose(l1, l2)

    def test_more_steps_less_noise(self):
        """Higher step count should produce latents closer to the signal."""
        lo = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=20, num_steps=5,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        hi = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=20, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        # Higher-step latents should have less inter-trajectory variance
        var_lo = float(np.var(lo, axis=0).mean())
        var_hi = float(np.var(hi, axis=0).mean())
        assert var_hi < var_lo, (
            f"50-step variance {var_hi:.4f} should be < 5-step {var_lo:.4f}"
        )

    def test_prompt_determines_signal(self):
        """Different prompts should produce different signal components."""
        l1 = simulate_diffusion_latents(
            prompt="macro gears", n_trajectories=1, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        l2 = simulate_diffusion_latents(
            prompt="sunset ocean", n_trajectories=1, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        assert not np.allclose(l1, l2)

    def test_dtype_float64(self):
        latents = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=2, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=1,
        )
        assert latents.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════
# Latent-Space Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestLatentFID:
    def test_perfect_match(self, gt_latent):
        fid = compute_latent_fid(gt_latent, gt_latent)
        assert fid < 1e-10, f"FID of identical latents should be ~0, got {fid}"

    def test_non_negative(self, gt_latent, small_latents):
        fid = compute_latent_fid(small_latents[0], gt_latent)
        assert fid >= 0

    def test_noisy_is_worse(self, gt_latent, rng):
        clean = gt_latent + 0.01 * rng.standard_normal(gt_latent.shape)
        noisy = gt_latent + 0.5 * rng.standard_normal(gt_latent.shape)
        fid_clean = compute_latent_fid(clean, gt_latent)
        fid_noisy = compute_latent_fid(noisy, gt_latent)
        assert fid_noisy > fid_clean


class TestCLIPScore:
    def test_perfect_match(self, gt_latent):
        score = compute_clip_score(gt_latent, _PROMPT, gt_latent)
        assert score > 0.9, f"CLIP of identical latents should be high, got {score}"

    def test_range(self, gt_latent, small_latents):
        score = compute_clip_score(small_latents[0], _PROMPT, gt_latent)
        assert 0.0 <= score <= 1.0

    def test_closer_is_higher(self, gt_latent, rng):
        close = gt_latent + 0.01 * rng.standard_normal(gt_latent.shape)
        far = gt_latent + 1.0 * rng.standard_normal(gt_latent.shape)
        clip_close = compute_clip_score(close, _PROMPT, gt_latent)
        clip_far = compute_clip_score(far, _PROMPT, gt_latent)
        assert clip_close > clip_far


class TestPSNR:
    def test_perfect_match(self, gt_latent):
        psnr = compute_psnr(gt_latent, gt_latent)
        assert psnr == 100.0  # our cap for perfect match

    def test_noisy_lower(self, gt_latent, rng):
        clean = gt_latent + 0.01 * rng.standard_normal(gt_latent.shape)
        noisy = gt_latent + 0.5 * rng.standard_normal(gt_latent.shape)
        psnr_clean = compute_psnr(clean, gt_latent)
        psnr_noisy = compute_psnr(noisy, gt_latent)
        assert psnr_clean > psnr_noisy

    def test_positive(self, gt_latent, small_latents):
        psnr = compute_psnr(small_latents[0], gt_latent)
        assert psnr > 0


# ═══════════════════════════════════════════════════════════════════════════
# LatentTelemetryExtractor
# ═══════════════════════════════════════════════════════════════════════════


class TestLatentTelemetryExtractor:
    def test_feature_names(self):
        ext = LatentTelemetryExtractor()
        assert ext.feature_names == LATENT_FEATURE_NAMES
        assert ext.n_features == 6

    def test_shape(self, small_latents):
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        assert features.shape == (12, 6)

    def test_no_nans(self, small_latents):
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        assert not np.any(np.isnan(features))

    def test_spatial_energy_positive(self, small_latents):
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        assert np.all(features[:, 0] >= 0)  # spatial_energy

    def test_channel_var_mean_positive(self, small_latents):
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        assert np.all(features[:, 2] >= 0)  # channel_var_mean

    def test_high_freq_ratio_range(self, small_latents):
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        hf = features[:, 3]  # high_freq_ratio
        assert np.all(hf >= 0.0)
        assert np.all(hf <= 1.0)

    def test_noisy_latents_have_more_energy(self):
        """Low-step (noisy) latents should have higher spatial energy."""
        lo = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=20, num_steps=5,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        hi = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=20, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=42,
        )
        ext = LatentTelemetryExtractor()
        feat_lo = ext.extract(lo)
        feat_hi = ext.extract(hi)
        # Low-step has more noise energy
        energy_lo = np.mean(feat_lo[:, 2])  # channel_var_mean
        energy_hi = np.mean(feat_hi[:, 2])
        assert energy_lo > energy_hi, (
            f"5-step variance {energy_lo:.4f} should be > 50-step {energy_hi:.4f}"
        )

    def test_single_latent(self, gt_latent):
        """Extract features from a batch of size 1."""
        ext = LatentTelemetryExtractor()
        features = ext.extract(gt_latent[np.newaxis, ...])
        assert features.shape == (1, 6)
        assert not np.any(np.isnan(features))


# ═══════════════════════════════════════════════════════════════════════════
# GaltonLatentFilter
# ═══════════════════════════════════════════════════════════════════════════


class TestGaltonLatentFilter:
    def test_reject_fraction(self, small_latents):
        filt = GaltonLatentFilter(reject_fraction=0.25)
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        mask = filt.filter(features)
        survival_rate = mask.sum() / len(mask)
        assert 0.65 <= survival_rate <= 0.85

    def test_viability_scores_shape(self, small_latents):
        filt = GaltonLatentFilter(reject_fraction=0.25)
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        scores = filt.compute_viability_scores(features)
        assert scores.shape == (12,)

    def test_viability_scores_non_negative(self, small_latents):
        filt = GaltonLatentFilter(reject_fraction=0.25)
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        scores = filt.compute_viability_scores(features)
        assert np.all(scores >= 0)

    def test_high_reject_fraction(self, small_latents):
        filt = GaltonLatentFilter(reject_fraction=0.8)
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        mask = filt.filter(features)
        survival_rate = mask.sum() / len(mask)
        assert survival_rate < 0.3

    def test_low_reject_fraction(self, small_latents):
        filt = GaltonLatentFilter(reject_fraction=0.05)
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        mask = filt.filter(features)
        survival_rate = mask.sum() / len(mask)
        assert survival_rate > 0.9

    def test_invalid_reject_fraction(self):
        with pytest.raises(ValueError):
            GaltonLatentFilter(reject_fraction=0.0)
        with pytest.raises(ValueError):
            GaltonLatentFilter(reject_fraction=1.0)

    def test_mask_is_boolean(self, small_latents):
        filt = GaltonLatentFilter(reject_fraction=0.25)
        ext = LatentTelemetryExtractor()
        features = ext.extract(small_latents)
        mask = filt.filter(features)
        assert mask.dtype == np.bool_


# ═══════════════════════════════════════════════════════════════════════════
# DiffusionMitigator
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffusionMitigator:
    def _make_calibration_data(
        self, n_cal=16, seed=42,
    ) -> tuple:
        """Helper: generate paired calibration latents and telemetry."""
        cal_low = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=n_cal, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W,
            seed=seed,
        )
        cal_high = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=n_cal, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W,
            seed=seed + 1000,
        )
        ext = LatentTelemetryExtractor()
        telemetry = ext.extract(cal_low)
        return telemetry, cal_low, cal_high

    def test_calibrate(self, default_config):
        mit = DiffusionMitigator(default_config)
        telemetry, low, high = self._make_calibration_data()
        result = mit.calibrate(telemetry, low, high)
        assert isinstance(result, DiffusionCalibrationResult)
        assert mit.is_calibrated

    def test_predict_raises_before_calibrate(self, default_config):
        mit = DiffusionMitigator(default_config)
        telemetry, low, _ = self._make_calibration_data()
        with pytest.raises(RuntimeError, match="not calibrated"):
            mit.predict(telemetry, low)

    def test_predict_shape(self, default_config):
        mit = DiffusionMitigator(default_config)
        telemetry, low, high = self._make_calibration_data()
        mit.calibrate(telemetry, low, high)
        result = mit.predict(telemetry, low)
        assert result.shape == (_C, _H, _W)

    def test_calibration_result_fields(self, default_config):
        mit = DiffusionMitigator(default_config)
        telemetry, low, high = self._make_calibration_data()
        result = mit.calibrate(telemetry, low, high)
        assert result.model_name == "random_forest"
        assert result.n_samples > 0
        assert result.train_mae >= 0
        assert result.train_rmse >= 0
        assert result.elapsed_seconds >= 0
        assert len(result.feature_names) == 7  # pixel_value + 6 telemetry

    def test_model_names(self):
        for name in ("random_forest", "gradient_boosting", "ridge"):
            cfg = DiffusionConfig(
                model_name=name,
                latent_channels=_C, latent_height=_H, latent_width=_W,
            )
            mit = DiffusionMitigator(cfg)
            telemetry, low, high = self._make_calibration_data()
            result = mit.calibrate(telemetry, low, high)
            assert result.model_name == name

    def test_unknown_model_raises(self):
        cfg = DiffusionConfig(
            model_name="xgboost",
            latent_channels=_C, latent_height=_H, latent_width=_W,
        )
        mit = DiffusionMitigator(cfg)
        telemetry, low, high = self._make_calibration_data()
        with pytest.raises(ValueError, match="Unknown model"):
            mit.calibrate(telemetry, low, high)


# ═══════════════════════════════════════════════════════════════════════════
# DiffusionMitigationPipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffusionMitigationPipeline:
    def _calibrate_pipeline(
        self, config=None, n_cal=16, seed=42,
    ) -> DiffusionMitigationPipeline:
        if config is None:
            config = DiffusionConfig(
                latent_channels=_C, latent_height=_H, latent_width=_W,
            )
        pipe = DiffusionMitigationPipeline(config=config)
        cal_low = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=n_cal, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W,
            seed=seed,
        )
        cal_high = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=n_cal, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W,
            seed=seed + 1000,
        )
        pipe.calibrate(cal_low, cal_high)
        return pipe

    def test_end_to_end(self):
        pipe = self._calibrate_pipeline()
        budget = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=8, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
        )
        gt = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=1, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=88,
        )[0]

        result = pipe.mitigate(budget, prompt=_PROMPT, ground_truth_latent=gt)
        assert isinstance(result, DiffusionMitigationResult)
        assert result.stage1_survivors > 0
        assert result.stage1_rejected >= 0
        assert result.stage1_survivors + result.stage1_rejected == 8

    def test_mitigated_latent_shape(self):
        pipe = self._calibrate_pipeline()
        budget = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=8, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
        )
        result = pipe.mitigate(budget, prompt=_PROMPT)
        assert result.mitigated_latent.shape == (_C, _H, _W)
        assert result.raw_mean_latent.shape == (_C, _H, _W)

    def test_with_ground_truth(self):
        pipe = self._calibrate_pipeline()
        budget = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=8, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
        )
        gt = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=1, num_steps=50,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=88,
        )[0]

        result = pipe.mitigate(budget, prompt=_PROMPT, ground_truth_latent=gt)
        assert result.fid_score >= 0
        assert 0.0 <= result.clip_score <= 1.0
        assert result.psnr > 0
        assert not math.isnan(result.improvement_factor)

    def test_without_ground_truth(self):
        pipe = self._calibrate_pipeline()
        budget = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=8, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
        )
        result = pipe.mitigate(budget, prompt=_PROMPT)
        assert result.fid_score == 0.0
        assert result.clip_score == 0.0
        assert math.isnan(result.improvement_factor)

    def test_latency_measured(self):
        pipe = self._calibrate_pipeline()
        budget = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=8, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
        )
        result = pipe.mitigate(budget, prompt=_PROMPT)
        assert result.latency_seconds > 0

    def test_metadata(self):
        pipe = self._calibrate_pipeline()
        budget = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=8, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
        )
        result = pipe.mitigate(budget, prompt=_PROMPT)
        assert "n_budget_latents" in result.metadata
        assert result.metadata["n_budget_latents"] == 8
        assert "reject_fraction" in result.metadata
        assert "model_name" in result.metadata

    def test_properties(self):
        pipe = DiffusionMitigationPipeline()
        assert isinstance(pipe.config, DiffusionConfig)
        assert isinstance(pipe.extractor, LatentTelemetryExtractor)
        assert isinstance(pipe.filter, GaltonLatentFilter)
        assert isinstance(pipe.mitigator, DiffusionMitigator)

    def test_ridge_model(self):
        cfg = DiffusionConfig(
            model_name="ridge",
            latent_channels=_C, latent_height=_H, latent_width=_W,
        )
        pipe = self._calibrate_pipeline(config=cfg)
        budget = simulate_diffusion_latents(
            prompt=_PROMPT, n_trajectories=8, num_steps=10,
            latent_channels=_C, latent_height=_H, latent_width=_W, seed=99,
        )
        result = pipe.mitigate(budget, prompt=_PROMPT)
        assert result.mitigated_latent.shape == (_C, _H, _W)


# ═══════════════════════════════════════════════════════════════════════════
# run_diffusion_benchmark (convenience function)
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffusionBenchmarkConvenience:
    def test_quick_benchmark(self):
        """Smoke test with small problem size."""
        result = run_diffusion_benchmark(
            prompt=_PROMPT,
            gt_steps=50,
            budget_steps=10,
            n_batch=8,
            n_calibration=16,
            latent_channels=_C,
            latent_height=_H,
            latent_width=_W,
            seed=42,
        )
        assert "ground_truth" in result
        assert "raw_budget" in result
        assert "qgate_mitigated" in result
        assert "improvement" in result
        assert "calibration" in result
        assert "timing" in result
        assert "params" in result

    def test_has_metrics(self):
        result = run_diffusion_benchmark(
            prompt=_PROMPT,
            gt_steps=50,
            budget_steps=10,
            n_batch=8,
            n_calibration=16,
            latent_channels=_C,
            latent_height=_H,
            latent_width=_W,
            seed=42,
        )
        gt = result["ground_truth"]
        assert gt["fid"] == 0.0
        assert gt["clip_score"] == 1.0

        raw = result["raw_budget"]
        assert raw["fid"] > 0
        assert 0.0 <= raw["clip_score"] <= 1.0

        mit = result["qgate_mitigated"]
        assert mit["fid"] >= 0
        assert 0.0 <= mit["clip_score"] <= 1.0

    def test_improvement_positive(self):
        result = run_diffusion_benchmark(
            prompt=_PROMPT,
            gt_steps=50,
            budget_steps=10,
            n_batch=8,
            n_calibration=16,
            latent_channels=_C,
            latent_height=_H,
            latent_width=_W,
            seed=42,
        )
        assert result["improvement"]["fid_improvement"] > 0

    def test_timing_keys(self):
        result = run_diffusion_benchmark(
            prompt=_PROMPT,
            gt_steps=50,
            budget_steps=10,
            n_batch=8,
            n_calibration=16,
            latent_channels=_C,
            latent_height=_H,
            latent_width=_W,
            seed=42,
        )
        t = result["timing"]
        assert "gt_wall_seconds" in t
        assert "raw_wall_seconds" in t
        assert "mitigate_wall_seconds" in t
        assert t["gt_wall_seconds"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Result Dataclasses
# ═══════════════════════════════════════════════════════════════════════════


class TestResultDataclasses:
    def test_calibration_result_frozen(self):
        r = DiffusionCalibrationResult(model_name="rf", n_samples=100)
        with pytest.raises(Exception):
            r.model_name = "gbr"  # type: ignore[misc]

    def test_mitigation_result_frozen(self):
        r = DiffusionMitigationResult(
            mitigated_latent=np.zeros((4, 16, 16)),
            raw_mean_latent=np.zeros((4, 16, 16)),
            stage1_survivors=6,
            stage1_rejected=2,
        )
        with pytest.raises(Exception):
            r.stage1_survivors = 99  # type: ignore[misc]

    def test_mitigation_result_defaults(self):
        r = DiffusionMitigationResult(
            mitigated_latent=np.zeros((4, 16, 16)),
            raw_mean_latent=np.zeros((4, 16, 16)),
            stage1_survivors=6,
            stage1_rejected=2,
        )
        assert r.fid_score == 0.0
        assert r.clip_score == 0.0
        assert r.psnr == 0.0
        assert math.isnan(r.improvement_factor)
        assert r.latency_seconds == 0.0
        assert r.metadata == {}

    def test_calibration_result_defaults(self):
        r = DiffusionCalibrationResult(model_name="rf", n_samples=100)
        assert r.feature_names == ()
        assert r.train_mae == 0.0
        assert r.train_rmse == 0.0
        assert r.elapsed_seconds == 0.0
        assert r.metadata == {}
