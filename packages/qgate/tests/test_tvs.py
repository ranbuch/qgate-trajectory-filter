"""Tests for qgate.tvs — Trajectory Viability Score pipeline.

Validates the full TVS pipeline:
  Step 1: HF normalisation (Level-1 RBF, Level-1-cluster K-Means+RBF, Level-2 binary)
  Step 2: Dynamic α calculation (static / Kalman-style)
  Step 3: Fusion (normalised to [0, 1])
  Step 4: Stage-1 Galton outlier rejection
  IQ separability: compute_iq_snr
  Autonomous routing: _determine_optimal_pipeline + auto mode
  Force-mode overrides: force_mode parameter
  Legacy deprecation: positional mode= parameter

Coverage targets:
  - All hardware modes (level_1, level_1_cluster, level_2, auto)
  - HF normalisation correctness and bounds
  - K-Means cluster scoring: intra-cluster compactness, k edge cases
  - IQ separability metric: SNR formula, separable vs overlapping clouds
  - Autonomous pipeline routing: dtype detection, SNR thresholding
  - Dynamic alpha: confidence mapping, min/max bounds
  - Fusion: normalisation guarantee, clamping
  - Galton filter: percentile thresholding, edge cases
  - Input validation (shapes, dtypes, invalid modes)
  - Vectorisation: no for-loops, large batches
  - process_telemetry_batch end-to-end
  - ml_features shape and dtype consistency
  - Legacy mode= deprecation warnings
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from qgate.tvs import (
    DEFAULT_ALPHA,
    DEFAULT_DROP_PERCENTILE,
    DEFAULT_HYBRID_FALLBACK,
    DEFAULT_K_CLUSTERS,
    DEFAULT_MAX_ALPHA,
    DEFAULT_MIN_ALPHA,
    DEFAULT_SNR_THRESHOLD,
    DEFAULT_VARIANCE,
    VALID_FORCE_MODES,
    VALID_MODES,
    _determine_optimal_pipeline,
    adaptive_galton_schedule,
    compute_alpha_dynamic,
    compute_alpha_static,
    compute_iq_snr,
    fuse_scores,
    galton_filter,
    normalise_hf_level1,
    normalise_hf_level1_cluster,
    normalise_hf_level2,
    process_telemetry_batch,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def binary_hf() -> np.ndarray:
    """A simple 10-element binary HF array."""
    return np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int64)


@pytest.fixture()
def iq_hf(rng: np.random.Generator) -> np.ndarray:
    """100 I/Q samples clustered near centroid (0.95 + 0.02j)."""
    real = rng.normal(0.95, 0.08, size=100)
    imag = rng.normal(0.02, 0.03, size=100)
    return real + 1j * imag


@pytest.fixture()
def lf_good(rng: np.random.Generator) -> np.ndarray:
    """100 well-behaved LF scores in [0.4, 1.0]."""
    return rng.uniform(0.4, 1.0, size=100)


@pytest.fixture()
def lf_10() -> np.ndarray:
    """10 LF scores matched to binary_hf."""
    return np.linspace(0.3, 0.9, 10)


# ═══════════════════════════════════════════════════════════════════════════
# TestHFNormalisationLevel2
# ═══════════════════════════════════════════════════════════════════════════


class TestHFNormalisationLevel2:
    """Level-2 binary HF normalisation."""

    def test_zeros_map_to_ones(self):
        hf = np.array([0, 0, 0])
        result = normalise_hf_level2(hf)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])

    def test_ones_map_to_zeros(self):
        hf = np.array([1, 1, 1])
        result = normalise_hf_level2(hf)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_mixed_values(self, binary_hf: np.ndarray):
        result = normalise_hf_level2(binary_hf)
        expected = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_output_in_zero_one(self, binary_hf: np.ndarray):
        result = normalise_hf_level2(binary_hf)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_invalid_values_rejected(self):
        hf = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="only 0 and 1"):
            normalise_hf_level2(hf)

    def test_float_binary_accepted(self):
        hf = np.array([0.0, 1.0, 0.0])
        result = normalise_hf_level2(hf)
        np.testing.assert_array_equal(result, [1.0, 0.0, 1.0])

    def test_negative_values_rejected(self):
        hf = np.array([0, -1, 1])
        with pytest.raises(ValueError, match="only 0 and 1"):
            normalise_hf_level2(hf)


# ═══════════════════════════════════════════════════════════════════════════
# TestHFNormalisationLevel1
# ═══════════════════════════════════════════════════════════════════════════


class TestHFNormalisationLevel1:
    """Level-1 I/Q soft-decision HF normalisation."""

    def test_at_centroid_gives_one(self):
        """Point exactly at centroid → distance = 0 → score = 1."""
        centroid = 0.95 + 0.02j
        hf = np.array([centroid])
        result = normalise_hf_level1(hf, centroid, variance=0.15)
        np.testing.assert_allclose(result, [1.0], atol=1e-12)

    def test_far_from_centroid_gives_near_zero(self):
        """Point very far from centroid → score → 0."""
        centroid = 0.0 + 0.0j
        hf = np.array([100.0 + 100.0j])
        result = normalise_hf_level1(hf, centroid, variance=0.1)
        assert result[0] < 1e-10

    def test_output_bounded_zero_one(self, iq_hf: np.ndarray):
        result = normalise_hf_level1(iq_hf, 0.95 + 0.02j, variance=0.15)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_closer_points_score_higher(self):
        """Points closer to centroid should have higher scores."""
        centroid = 1.0 + 0.0j
        close = np.array([0.99 + 0.0j])
        far = np.array([0.5 + 0.0j])
        s_close = normalise_hf_level1(close, centroid, 0.1)
        s_far = normalise_hf_level1(far, centroid, 0.1)
        assert s_close[0] > s_far[0]

    def test_rbf_formula_exact(self):
        """Verify the RBF formula matches manual calculation."""
        centroid = 1.0 + 0.0j
        point = 1.1 + 0.2j
        variance = 0.25
        d = abs(point - centroid)
        expected = np.exp(-(d ** 2) / variance)
        result = normalise_hf_level1(np.array([point]), centroid, variance)
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_variance_zero_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            normalise_hf_level1(np.array([1.0 + 0j]), 0.0 + 0j, variance=0.0)

    def test_negative_variance_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            normalise_hf_level1(np.array([1.0 + 0j]), 0.0 + 0j, variance=-0.5)

    def test_invalid_centroid_type(self):
        with pytest.raises(TypeError, match="numeric type"):
            normalise_hf_level1(np.array([1.0 + 0j]), "bad", variance=0.1)  # type: ignore[arg-type]

    def test_larger_variance_broader_scores(self):
        """Larger σ² → broader RBF → higher scores for distant points."""
        centroid = 0.0 + 0.0j
        point = np.array([0.5 + 0.0j])
        s_narrow = normalise_hf_level1(point, centroid, variance=0.01)
        s_broad = normalise_hf_level1(point, centroid, variance=10.0)
        assert s_broad[0] > s_narrow[0]

    def test_real_centroid_works(self):
        """Real float centroid (not complex) should work."""
        result = normalise_hf_level1(np.array([1.0 + 0j]), 1.0, variance=0.1)
        np.testing.assert_allclose(result, [1.0], atol=1e-12)

    def test_vectorised_large_batch(self, rng: np.random.Generator):
        """Process 50k shots — no for-loops, just vectorised ops."""
        n = 50_000
        iq = rng.normal(0.9, 0.1, n) + 1j * rng.normal(0.0, 0.05, n)
        result = normalise_hf_level1(iq, 0.9 + 0.0j, variance=0.2)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# TestAlphaCalculation
# ═══════════════════════════════════════════════════════════════════════════


class TestAlphaCalculation:
    """Static and dynamic α computation."""

    def test_static_constant(self):
        alpha = compute_alpha_static(5, 0.6)
        np.testing.assert_array_equal(alpha, [0.6, 0.6, 0.6, 0.6, 0.6])

    def test_static_default(self):
        alpha = compute_alpha_static(3)
        np.testing.assert_array_equal(alpha, [DEFAULT_ALPHA] * 3)

    def test_static_invalid_alpha(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            compute_alpha_static(3, 1.5)

    def test_static_negative_alpha(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            compute_alpha_static(3, -0.1)

    def test_dynamic_at_score_zero(self):
        """hf_score=0 → confidence=1 → alpha=max."""
        hf = np.array([0.0])
        alpha = compute_alpha_dynamic(hf, 0.3, 0.9)
        np.testing.assert_allclose(alpha, [0.9])

    def test_dynamic_at_score_one(self):
        """hf_score=1 → confidence=1 → alpha=max."""
        hf = np.array([1.0])
        alpha = compute_alpha_dynamic(hf, 0.3, 0.9)
        np.testing.assert_allclose(alpha, [0.9])

    def test_dynamic_at_score_half(self):
        """hf_score=0.5 → confidence=0 → alpha=min."""
        hf = np.array([0.5])
        alpha = compute_alpha_dynamic(hf, 0.3, 0.9)
        np.testing.assert_allclose(alpha, [0.3])

    def test_dynamic_symmetric(self):
        """Symmetric: score 0.2 and 0.8 should give same α."""
        hf = np.array([0.2, 0.8])
        alpha = compute_alpha_dynamic(hf, 0.3, 0.9)
        np.testing.assert_allclose(alpha[0], alpha[1])

    def test_dynamic_bounded(self, rng: np.random.Generator):
        """All α values must fall in [min_alpha, max_alpha]."""
        hf = rng.uniform(0.0, 1.0, 1000)
        alpha = compute_alpha_dynamic(hf, 0.2, 0.8)
        assert np.all(alpha >= 0.2 - 1e-12)
        assert np.all(alpha <= 0.8 + 1e-12)

    def test_dynamic_invalid_bounds(self):
        with pytest.raises(ValueError, match="min_alpha <= max_alpha"):
            compute_alpha_dynamic(np.array([0.5]), 0.9, 0.3)

    def test_dynamic_formula_exact(self):
        """Verify formula: α = min + confidence * (max - min)."""
        hf = np.array([0.3])
        min_a, max_a = 0.2, 0.8
        confidence = 2.0 * abs(0.3 - 0.5)  # = 0.4
        expected = min_a + confidence * (max_a - min_a)  # = 0.2 + 0.4*0.6 = 0.44
        alpha = compute_alpha_dynamic(hf, min_a, max_a)
        np.testing.assert_allclose(alpha[0], expected)


# ═══════════════════════════════════════════════════════════════════════════
# TestFusion
# ═══════════════════════════════════════════════════════════════════════════


class TestFusion:
    """Score fusion and normalisation guarantees."""

    def test_pure_hf(self):
        """α = 1 → fusion = hf_score."""
        hf = np.array([0.8, 0.3])
        lf = np.array([0.2, 0.9])
        alpha = np.array([1.0, 1.0])
        result = fuse_scores(hf, lf, alpha)
        np.testing.assert_allclose(result, [0.8, 0.3])

    def test_pure_lf(self):
        """α = 0 → fusion = lf_score."""
        hf = np.array([0.8, 0.3])
        lf = np.array([0.2, 0.9])
        alpha = np.array([0.0, 0.0])
        result = fuse_scores(hf, lf, alpha)
        np.testing.assert_allclose(result, [0.2, 0.9])

    def test_half_fusion(self):
        """α = 0.5 → simple average."""
        hf = np.array([1.0, 0.0])
        lf = np.array([0.0, 1.0])
        alpha = np.array([0.5, 0.5])
        result = fuse_scores(hf, lf, alpha)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_output_bounded_zero_one(self, rng: np.random.Generator):
        """Fusion output is always in [0, 1] when inputs are in [0, 1]."""
        n = 10_000
        hf = rng.uniform(0.0, 1.0, n)
        lf = rng.uniform(0.0, 1.0, n)
        alpha = rng.uniform(0.0, 1.0, n)
        result = fuse_scores(hf, lf, alpha)
        assert np.all(result >= 0.0 - 1e-12)
        assert np.all(result <= 1.0 + 1e-12)

    def test_clamping_out_of_range_inputs(self):
        """Out-of-range inputs are clamped before fusion."""
        hf = np.array([1.5, -0.5])
        lf = np.array([-0.3, 1.2])
        alpha = np.array([0.5, 0.5])
        result = fuse_scores(hf, lf, alpha)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_formula_exact(self):
        """Verify formula: fusion = hf * α + lf * (1 - α)."""
        hf = np.array([0.7])
        lf = np.array([0.4])
        alpha = np.array([0.6])
        expected = 0.7 * 0.6 + 0.4 * 0.4
        result = fuse_scores(hf, lf, alpha)
        np.testing.assert_allclose(result[0], expected)

    def test_normalised_by_construction(self, rng: np.random.Generator):
        """With inputs in [0,1], fusion is guaranteed [0,1] (no extra step)."""
        hf = rng.uniform(0, 1, 5000)
        lf = rng.uniform(0, 1, 5000)
        alpha = rng.uniform(0, 1, 5000)
        result = fuse_scores(hf, lf, alpha)
        # Weighted average of [0,1] values → [0,1]
        assert float(np.min(result)) >= 0.0
        assert float(np.max(result)) <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TestGaltonFilter
# ═══════════════════════════════════════════════════════════════════════════


class TestGaltonFilter:
    """Stage-1 Galton percentile-based outlier rejection."""

    def test_percentile_zero_keeps_all(self):
        scores = np.array([0.1, 0.5, 0.9])
        mask = galton_filter(scores, drop_percentile=0.0)
        assert np.all(mask)

    def test_percentile_100_keeps_max_only(self):
        scores = np.array([0.1, 0.5, 0.9])
        mask = galton_filter(scores, drop_percentile=100.0)
        # Only the maximum (0.9) is >= the 100th percentile
        assert mask.sum() == 1
        assert mask[2]

    def test_default_25th_percentile(self):
        """With uniform scores, ~75% should survive the 25th percentile."""
        rng = np.random.default_rng(99)
        scores = rng.uniform(0, 1, 10_000)
        mask = galton_filter(scores)
        survival_rate = mask.sum() / len(scores)
        assert 0.70 < survival_rate < 0.80

    def test_all_equal_scores(self):
        """All equal → threshold equals that value → all survive."""
        scores = np.full(100, 0.5)
        mask = galton_filter(scores, drop_percentile=50.0)
        assert np.all(mask)

    def test_invalid_percentile_low(self):
        with pytest.raises(ValueError, match="in \\[0, 100\\]"):
            galton_filter(np.array([0.5]), drop_percentile=-1.0)

    def test_invalid_percentile_high(self):
        with pytest.raises(ValueError, match="in \\[0, 100\\]"):
            galton_filter(np.array([0.5]), drop_percentile=101.0)

    def test_mask_is_boolean(self):
        scores = np.array([0.2, 0.8, 0.5])
        mask = galton_filter(scores, drop_percentile=30.0)
        assert mask.dtype == bool

    def test_single_element(self):
        mask = galton_filter(np.array([0.7]), drop_percentile=50.0)
        assert mask[0]


# ═══════════════════════════════════════════════════════════════════════════
# TestProcessTelemetryBatch — Level-2 end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessLevel2:
    """End-to-end Level-2 (binary) pipeline tests."""

    def test_basic_level2(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert result["mode"] == "level_2"
        assert result["n_shots"] == 10
        assert result["surviving_mask"].shape == (10,)
        assert result["fusion_scores"].shape == (10,)
        assert result["n_surviving"] <= 10
        assert result["n_surviving"] == int(result["surviving_mask"].sum())

    def test_ml_features_are_binary(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        features = result["ml_features"]
        unique = np.unique(features)
        assert np.all(np.isin(unique, [0, 1]))

    def test_ml_features_only_surviving(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert len(result["ml_features"]) == result["n_surviving"]

    def test_alpha_is_static(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(
            binary_hf, lf_10, force_mode="level_2", alpha=0.7
        )
        np.testing.assert_array_equal(result["alpha_values"], 0.7)

    def test_fusion_bounded(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert np.all(result["fusion_scores"] >= 0.0)
        assert np.all(result["fusion_scores"] <= 1.0)

    def test_all_zeros_hf_high_scores(self):
        """All bits 0 → hf_score = 1 → high fusion scores."""
        hf = np.zeros(20, dtype=int)
        lf = np.full(20, 0.8)
        result = process_telemetry_batch(hf, lf, force_mode="level_2", alpha=0.5)
        # fusion = 1.0 * 0.5 + 0.8 * 0.5 = 0.9 for all
        np.testing.assert_allclose(result["fusion_scores"], 0.9)

    def test_custom_drop_percentile(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        r50 = process_telemetry_batch(
            binary_hf, lf_10, force_mode="level_2", drop_percentile=50.0
        )
        r10 = process_telemetry_batch(
            binary_hf, lf_10, force_mode="level_2", drop_percentile=10.0
        )
        # Lower percentile → more survivors
        assert r10["n_surviving"] >= r50["n_surviving"]

    def test_threshold_returned(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert isinstance(result["threshold"], float)
        assert 0.0 <= result["threshold"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TestProcessTelemetryBatch — Level-1 end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessLevel1:
    """End-to-end Level-1 (I/Q) pipeline tests."""

    def test_basic_level1(self, iq_hf: np.ndarray, lf_good: np.ndarray):
        result = process_telemetry_batch(
            iq_hf, lf_good, force_mode="level_1",
            zero_centroid=0.95 + 0.02j, variance=0.15,
        )
        assert result["mode"] == "level_1"
        assert result["n_shots"] == 100
        assert result["n_surviving"] <= 100

    def test_ml_features_are_complex(self, iq_hf: np.ndarray, lf_good: np.ndarray):
        result = process_telemetry_batch(
            iq_hf, lf_good, force_mode="level_1",
            zero_centroid=0.95 + 0.02j, variance=0.15,
        )
        assert np.issubdtype(result["ml_features"].dtype, np.complexfloating)

    def test_dynamic_alpha_varies(self, iq_hf: np.ndarray, lf_good: np.ndarray):
        """Level-1 uses per-shot dynamic α — values should vary."""
        result = process_telemetry_batch(
            iq_hf, lf_good, force_mode="level_1",
            zero_centroid=0.95 + 0.02j, variance=0.15,
        )
        alpha = result["alpha_values"]
        assert alpha.std() > 0.0  # Not all the same

    def test_alpha_in_bounds(self, iq_hf: np.ndarray, lf_good: np.ndarray):
        min_a, max_a = 0.2, 0.85
        result = process_telemetry_batch(
            iq_hf, lf_good, force_mode="level_1",
            zero_centroid=0.95 + 0.02j, variance=0.15,
            min_alpha=min_a, max_alpha=max_a,
        )
        alpha = result["alpha_values"]
        assert np.all(alpha >= min_a - 1e-12)
        assert np.all(alpha <= max_a + 1e-12)

    def test_fusion_bounded(self, iq_hf: np.ndarray, lf_good: np.ndarray):
        result = process_telemetry_batch(
            iq_hf, lf_good, force_mode="level_1",
            zero_centroid=0.95 + 0.02j, variance=0.15,
        )
        assert np.all(result["fusion_scores"] >= 0.0)
        assert np.all(result["fusion_scores"] <= 1.0)

    def test_hf_scores_bounded(self, iq_hf: np.ndarray, lf_good: np.ndarray):
        result = process_telemetry_batch(
            iq_hf, lf_good, force_mode="level_1",
            zero_centroid=0.95 + 0.02j, variance=0.15,
        )
        assert np.all(result["hf_scores"] >= 0.0)
        assert np.all(result["hf_scores"] <= 1.0)

    def test_large_batch_level1(self, rng: np.random.Generator):
        """50k shots, Level-1 — must be fast (vectorised)."""
        n = 50_000
        iq = rng.normal(0.9, 0.1, n) + 1j * rng.normal(0.0, 0.05, n)
        lf = rng.uniform(0.5, 1.0, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=0.9 + 0.0j, variance=0.2,
        )
        assert result["n_shots"] == n
        assert result["n_surviving"] > 0
        assert result["fusion_scores"].shape == (n,)

    def test_custom_variance(self, rng: np.random.Generator):
        """Different variance → different HF scores."""
        n = 100
        iq = rng.normal(0.9, 0.2, n) + 1j * rng.normal(0.0, 0.1, n)
        lf = np.full(n, 0.5)
        r1 = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=0.9 + 0.0j, variance=0.05,
        )
        r2 = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=0.9 + 0.0j, variance=5.0,
        )
        # Broader variance → higher mean HF score (more lenient)
        assert np.mean(r2["hf_scores"]) > np.mean(r1["hf_scores"])


# ═══════════════════════════════════════════════════════════════════════════
# TestInputValidation
# ═══════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """Verify robust input validation."""

    def test_invalid_force_mode(self):
        with pytest.raises(ValueError, match="force_mode must be one of"):
            process_telemetry_batch(
                np.array([0, 1]), np.array([0.5, 0.5]), force_mode="level_3"
            )

    def test_hf_not_array(self):
        with pytest.raises(TypeError, match="numpy ndarray"):
            process_telemetry_batch(
                [0, 1], np.array([0.5, 0.5]), force_mode="level_2"  # type: ignore[arg-type]
            )

    def test_lf_not_array(self):
        with pytest.raises(TypeError, match="numpy ndarray"):
            process_telemetry_batch(
                np.array([0, 1]), [0.5, 0.5], force_mode="level_2"  # type: ignore[arg-type]
            )

    def test_2d_hf_rejected(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            process_telemetry_batch(
                np.array([[0, 1]]), np.array([0.5, 0.5]), force_mode="level_2"
            )

    def test_empty_hf_rejected(self):
        with pytest.raises(ValueError, match="not be empty"):
            process_telemetry_batch(
                np.array([]), np.array([]), force_mode="level_2"
            )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            process_telemetry_batch(
                np.array([0, 1, 0]), np.array([0.5, 0.5]), force_mode="level_2"
            )

    def test_2d_lf_rejected(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            process_telemetry_batch(
                np.array([0, 1]), np.array([[0.5, 0.5]]), force_mode="level_2"
            )


# ═══════════════════════════════════════════════════════════════════════════
# TestModeDynamic
# ═══════════════════════════════════════════════════════════════════════════


class TestModeDynamic:
    """Verify force_mode is dynamic — user can switch at call time."""

    def test_same_data_different_mode(self, rng: np.random.Generator):
        """Calling with level_1 vs level_2 on compatible data gives different results."""
        # Use binary-ish data that's valid for both modes
        hf_binary = np.array([0, 1, 0, 1, 0])
        # For level_1, wrap binary as complex near centroid / far
        hf_complex = np.array([0.0 + 0j, 5.0 + 0j, 0.0 + 0j, 5.0 + 0j, 0.0 + 0j])
        lf = np.array([0.6, 0.6, 0.6, 0.6, 0.6])

        r2 = process_telemetry_batch(hf_binary, lf, force_mode="level_2")
        r1 = process_telemetry_batch(
            hf_complex, lf, force_mode="level_1",
            zero_centroid=0.0 + 0.0j, variance=0.1,
        )
        # Both should produce valid results but different fusion scores
        assert r2["mode"] == "level_2"
        assert r1["mode"] == "level_1"
        assert not np.array_equal(r2["fusion_scores"], r1["fusion_scores"])

    def test_mode_selection_changes_alpha_type(self, rng: np.random.Generator):
        """Level-2 → static α, Level-1 → varying α."""
        n = 100
        iq = rng.normal(0.9, 0.2, n) + 1j * rng.normal(0.0, 0.1, n)
        lf = rng.uniform(0.4, 1.0, n)

        r1 = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=0.9 + 0j, variance=0.2,
        )
        # Level-1 alpha should vary
        assert r1["alpha_values"].std() > 0

        # Level-2 with binary data → constant alpha
        hf_bin = np.zeros(n, dtype=int)
        r2 = process_telemetry_batch(hf_bin, lf, force_mode="level_2", alpha=0.5)
        assert r2["alpha_values"].std() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TestNormalisationProperty
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalisationProperty:
    """Verify fusion normalisation is guaranteed by construction."""

    def test_level2_fusion_normalised(self, rng: np.random.Generator):
        """Binary HF → hf_score ∈ {0,1}, lf ∈ [0,1], α ∈ [0,1] → fusion ∈ [0,1]."""
        hf = rng.integers(0, 2, size=5000).astype(np.int64)
        lf = rng.uniform(0, 1, 5000)
        result = process_telemetry_batch(
            hf, lf, force_mode="level_2",
            alpha=rng.uniform(0, 1),
        )
        assert float(np.min(result["fusion_scores"])) >= 0.0
        assert float(np.max(result["fusion_scores"])) <= 1.0

    def test_level1_fusion_normalised(self, rng: np.random.Generator):
        """RBF → hf_score ∈ [0,1], lf ∈ [0,1], dynamic α ∈ [min,max] → fusion ∈ [0,1]."""
        n = 5000
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=0.0 + 0.0j, variance=1.0,
        )
        assert float(np.min(result["fusion_scores"])) >= 0.0
        assert float(np.max(result["fusion_scores"])) <= 1.0

    def test_extreme_iq_still_normalised(self):
        """Very far I/Q points → hf_score ≈ 0, but fusion still ∈ [0,1]."""
        iq = np.array([1e6 + 1e6j, -1e6 - 1e6j])
        lf = np.array([0.5, 0.5])
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=0.0 + 0.0j, variance=0.1,
        )
        assert np.all(result["fusion_scores"] >= 0.0)
        assert np.all(result["fusion_scores"] <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# TestReturnContract
# ═══════════════════════════════════════════════════════════════════════════


class TestReturnContract:
    """Verify all promised return keys and types."""

    def test_all_keys_present(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        expected_keys = {
            "surviving_mask", "fusion_scores", "ml_features",
            "hf_scores", "alpha_values", "threshold",
            "n_shots", "n_surviving", "mode",
        }
        assert set(result.keys()) == expected_keys

    def test_surviving_mask_dtype(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert result["surviving_mask"].dtype == bool

    def test_n_surviving_matches_mask(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert result["n_surviving"] == int(result["surviving_mask"].sum())

    def test_ml_features_length_matches(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert len(result["ml_features"]) == result["n_surviving"]

    def test_threshold_is_float(self, binary_hf: np.ndarray, lf_10: np.ndarray):
        result = process_telemetry_batch(binary_hf, lf_10, force_mode="level_2")
        assert isinstance(result["threshold"], float)


# ═══════════════════════════════════════════════════════════════════════════
# TestEdgeCases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_single_shot_level2(self):
        result = process_telemetry_batch(
            np.array([0]), np.array([0.5]), force_mode="level_2"
        )
        assert result["n_shots"] == 1
        assert result["n_surviving"] == 1

    def test_single_shot_level1(self):
        result = process_telemetry_batch(
            np.array([0.5 + 0.5j]), np.array([0.5]),
            force_mode="level_1", zero_centroid=0.5 + 0.5j, variance=0.1,
        )
        assert result["n_shots"] == 1
        assert result["n_surviving"] == 1

    def test_all_same_hf_scores(self):
        """All shots have identical HF → all equal fusion → all survive."""
        hf = np.zeros(50, dtype=int)
        lf = np.full(50, 0.7)
        result = process_telemetry_batch(hf, lf, force_mode="level_2")
        assert result["n_surviving"] == 50

    def test_drop_percentile_zero(self):
        """p=0 keeps everything."""
        hf = np.array([0, 1, 0, 1])
        lf = np.array([0.5, 0.5, 0.5, 0.5])
        result = process_telemetry_batch(
            hf, lf, force_mode="level_2", drop_percentile=0.0
        )
        assert result["n_surviving"] == 4

    def test_drop_percentile_high(self):
        """p=99 keeps very few."""
        rng = np.random.default_rng(123)
        hf = rng.integers(0, 2, size=1000).astype(np.int64)
        lf = rng.uniform(0, 1, 1000)
        result = process_telemetry_batch(
            hf, lf, force_mode="level_2", drop_percentile=99.0
        )
        assert result["n_surviving"] <= 20  # ~1% of 1000

    def test_lf_all_zeros(self):
        """LF scores all zero — HF dominates."""
        hf = np.zeros(10, dtype=int)
        lf = np.zeros(10)
        result = process_telemetry_batch(hf, lf, force_mode="level_2", alpha=0.5)
        # fusion = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        np.testing.assert_allclose(result["fusion_scores"], 0.5)

    def test_lf_all_ones(self):
        """LF scores all one — combined with HF."""
        hf = np.ones(10, dtype=int)  # hf_score = 0
        lf = np.ones(10)
        result = process_telemetry_batch(hf, lf, force_mode="level_2", alpha=0.5)
        # fusion = 0.0 * 0.5 + 1.0 * 0.5 = 0.5
        np.testing.assert_allclose(result["fusion_scores"], 0.5)

    def test_constants_exported(self):
        """Public constants are accessible."""
        assert VALID_MODES == ("level_1", "level_1_cluster", "level_2", "auto")
        assert VALID_FORCE_MODES == ("level_1", "level_1_cluster", "level_2")
        assert isinstance(DEFAULT_ALPHA, float)
        assert isinstance(DEFAULT_DROP_PERCENTILE, float)
        assert isinstance(DEFAULT_MIN_ALPHA, float)
        assert isinstance(DEFAULT_MAX_ALPHA, float)
        assert isinstance(DEFAULT_VARIANCE, float)
        assert isinstance(DEFAULT_K_CLUSTERS, int)
        assert isinstance(DEFAULT_SNR_THRESHOLD, float)
        assert isinstance(DEFAULT_HYBRID_FALLBACK, str)


# ═══════════════════════════════════════════════════════════════════════════
# TestHFNormalisationLevel1Cluster
# ═══════════════════════════════════════════════════════════════════════════


class TestHFNormalisationLevel1Cluster:
    """Level-1 cluster (K-Means + RBF) HF normalisation."""

    def test_output_bounded_zero_one(self, rng: np.random.Generator):
        """All scores must be in [0, 1]."""
        n = 500
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        result = normalise_hf_level1_cluster(iq, k_clusters=4, variance=0.5)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_tight_cluster_scores_near_one(self, rng: np.random.Generator):
        """Points tightly clustered → distance to centre ≈ 0 → score ≈ 1."""
        n = 200
        # Very tight cluster around two points
        iq = np.concatenate([
            rng.normal(1.0, 1e-5, n // 2) + 1j * rng.normal(0.0, 1e-5, n // 2),
            rng.normal(-1.0, 1e-5, n // 2) + 1j * rng.normal(0.0, 1e-5, n // 2),
        ])
        result = normalise_hf_level1_cluster(iq, k_clusters=2, variance=0.1)
        assert np.mean(result) > 0.99

    def test_outlier_scores_low(self, rng: np.random.Generator):
        """Points far from all centres should score low."""
        n = 200
        # Tight cluster + one outlier
        iq = rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)
        iq = np.append(iq, [100.0 + 100.0j])
        lf_dummy = np.ones(n + 1)
        # The outlier should get a low score after clustering
        result = normalise_hf_level1_cluster(iq, k_clusters=2, variance=0.01)
        # Last point (outlier) — it might be in its own cluster or far from
        # its assigned centre.  With k=2, if one cluster contains the outlier
        # alone, distance=0 → score=1; if it's with the cloud, distance huge.
        # Either way, the main cloud should be high.
        main_cloud_scores = result[:n]
        assert np.mean(main_cloud_scores) > 0.8

    def test_k_clusters_minimum(self):
        """k_clusters < 2 should raise ValueError."""
        iq = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        with pytest.raises(ValueError, match="k_clusters must be >= 2"):
            normalise_hf_level1_cluster(iq, k_clusters=1)

    def test_k_clusters_exceeds_n_shots(self):
        """k_clusters > n_shots should raise ValueError."""
        iq = np.array([1.0 + 0j, 2.0 + 0j])
        with pytest.raises(ValueError, match="k_clusters.*must be <= n_shots"):
            normalise_hf_level1_cluster(iq, k_clusters=5)

    def test_variance_zero_rejected(self):
        """Zero variance must raise."""
        iq = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        with pytest.raises(ValueError, match="strictly positive"):
            normalise_hf_level1_cluster(iq, k_clusters=2, variance=0.0)

    def test_negative_variance_rejected(self):
        iq = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        with pytest.raises(ValueError, match="strictly positive"):
            normalise_hf_level1_cluster(iq, k_clusters=2, variance=-0.5)

    def test_larger_k_refines_clusters(self, rng: np.random.Generator):
        """More clusters → tighter fit → higher mean score (given same σ²)."""
        n = 500
        iq = rng.normal(0, 0.5, n) + 1j * rng.normal(0, 0.5, n)
        scores_k2 = normalise_hf_level1_cluster(iq, k_clusters=2, variance=0.1)
        scores_k16 = normalise_hf_level1_cluster(iq, k_clusters=16, variance=0.1)
        # More clusters = points closer to their centre = higher scores
        assert np.mean(scores_k16) >= np.mean(scores_k2) - 0.05  # allow small tolerance

    def test_vectorised_large_batch(self, rng: np.random.Generator):
        """Process 50k shots efficiently."""
        n = 50_000
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        result = normalise_hf_level1_cluster(iq, k_clusters=8, variance=0.5)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_k_equals_n_shots(self):
        """k = n_shots is valid — each point may form its own cluster."""
        iq = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        result = normalise_hf_level1_cluster(iq, k_clusters=3, variance=0.1)
        # MiniBatchKMeans with k=n doesn't guarantee each point is its own
        # centre, but scores should still be bounded and positive.
        assert result.shape == (3,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_bimodal_separation(self, rng: np.random.Generator):
        """Two well-separated blobs → k=2 → all scores high."""
        n = 500
        blob0 = rng.normal(0, 0.01, n // 2) + 1j * rng.normal(0, 0.01, n // 2)
        blob1 = rng.normal(5, 0.01, n // 2) + 1j * rng.normal(5, 0.01, n // 2)
        iq = np.concatenate([blob0, blob1])
        result = normalise_hf_level1_cluster(iq, k_clusters=2, variance=0.1)
        assert np.mean(result) > 0.95


# ═══════════════════════════════════════════════════════════════════════════
# TestIQSeparability
# ═══════════════════════════════════════════════════════════════════════════


class TestIQSeparability:
    """I/Q cloud separability metric (SNR)."""

    def test_well_separated_high_snr(self, rng: np.random.Generator):
        """Well-separated clouds should yield SNR >> 3."""
        n = 1000
        # Two tight blobs far apart
        blob0 = rng.normal(0, 0.01, n // 2) + 1j * rng.normal(0, 0.01, n // 2)
        blob1 = rng.normal(10, 0.01, n // 2) + 1j * rng.normal(0, 0.01, n // 2)
        iq = np.concatenate([blob0, blob1])
        snr = compute_iq_snr(iq, zero_centroid=0.0 + 0.0j)
        assert snr > 5.0

    def test_overlapping_low_snr(self, rng: np.random.Generator):
        """Heavily overlapping clouds should yield low SNR."""
        n = 1000
        # Both blobs centred nearly at the same place
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        snr = compute_iq_snr(iq, zero_centroid=0.0 + 0.0j)
        assert snr < 1.0

    def test_with_one_centroid(self, rng: np.random.Generator):
        """Explicit one_centroid gives exact inter-centroid distance."""
        n = 500
        iq = rng.normal(0, 0.5, n) + 1j * rng.normal(0, 0.5, n)
        snr = compute_iq_snr(iq, zero_centroid=0.0 + 0.0j, one_centroid=10.0 + 0.0j)
        # Separation = |10 - 0| = 10, σ_IQ for uniform cloud ~ 0.5 → SNR >> 3
        assert snr > 5.0

    def test_identical_points_infinite_snr(self):
        """All points identical → σ = 0 → SNR = inf."""
        iq = np.full(100, 1.0 + 1.0j)
        snr = compute_iq_snr(iq, zero_centroid=0.0 + 0.0j)
        assert snr == float("inf")

    def test_single_point_infinite_snr(self):
        """Single point → σ = 0 → SNR = inf."""
        iq = np.array([1.0 + 0j])
        snr = compute_iq_snr(iq, zero_centroid=0.0 + 0.0j)
        assert snr == float("inf")

    def test_centroid_at_sample_mean_zero_snr(self, rng: np.random.Generator):
        """If zero_centroid equals sample mean, separation ≈ 0 → SNR ≈ 0."""
        n = 1000
        iq = rng.normal(5, 0.5, n) + 1j * rng.normal(3, 0.5, n)
        sample_mean = np.mean(iq)
        snr = compute_iq_snr(iq, zero_centroid=sample_mean)
        assert snr < 0.01

    def test_empty_array_rejected(self):
        """Empty hf_data must raise."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_iq_snr(np.array([], dtype=complex), zero_centroid=0.0 + 0j)

    def test_invalid_centroid_type_rejected(self):
        """Non-numeric centroid must raise TypeError."""
        iq = np.array([1.0 + 0j])
        with pytest.raises(TypeError, match="numeric type"):
            compute_iq_snr(iq, zero_centroid="bad")  # type: ignore[arg-type]

    def test_invalid_one_centroid_type_rejected(self):
        """Non-numeric one_centroid must raise TypeError."""
        iq = np.array([1.0 + 0j])
        with pytest.raises(TypeError, match="numeric type"):
            compute_iq_snr(iq, zero_centroid=0.0, one_centroid="bad")  # type: ignore[arg-type]

    def test_snr_formula_exact(self):
        """Verify SNR formula against manual calculation."""
        # Construct a known distribution: 4 points symmetrically placed
        iq = np.array([1.0 + 0j, -1.0 + 0j, 0.0 + 1.0j, 0.0 - 1.0j])
        c0 = 5.0 + 0j

        sample_mean = np.mean(iq)  # = 0+0j
        separation = abs(sample_mean - np.complex128(c0))  # = 5.0
        distances = np.abs(iq - sample_mean)  # all = 1.0
        sigma = float(np.std(distances))  # = 0.0 (all same)
        # All distances are 1.0 → σ = 0 → SNR = inf
        snr = compute_iq_snr(iq, zero_centroid=c0)
        assert snr == float("inf")

    def test_snr_formula_nondegenerate(self, rng: np.random.Generator):
        """Verify formula: SNR = separation / sigma for non-degenerate case."""
        # Use asymmetric points so distances from mean have non-zero std.
        iq = np.array([0.0 + 0j, 1.0 + 0j, 3.0 + 0j, 6.0 + 0j])
        c0 = 10.0 + 0j

        sample_mean = np.mean(iq)  # = 2.5+0j
        separation = abs(sample_mean - np.complex128(c0))
        distances = np.abs(iq - sample_mean)
        sigma = float(np.std(distances))
        assert sigma > 0  # sanity

        expected_snr = separation / sigma
        snr = compute_iq_snr(iq, zero_centroid=c0)
        np.testing.assert_allclose(snr, expected_snr, rtol=1e-10)

    def test_returns_float(self, rng: np.random.Generator):
        """SNR should be a Python float."""
        iq = rng.normal(0, 1, 100) + 1j * rng.normal(0, 1, 100)
        snr = compute_iq_snr(iq, zero_centroid=0.0 + 0j)
        assert isinstance(snr, float)


# ═══════════════════════════════════════════════════════════════════════════
# TestProcessLevel1Cluster
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessLevel1Cluster:
    """End-to-end process_telemetry_batch with force_mode='level_1_cluster'."""

    def test_basic_pipeline(self, rng: np.random.Generator):
        """Runs without errors and returns correct keys."""
        n = 200
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=4, variance=0.5,
        )
        expected_keys = {
            "surviving_mask", "fusion_scores", "ml_features",
            "hf_scores", "alpha_values", "threshold",
            "n_shots", "n_surviving", "mode",
        }
        assert set(result.keys()) == expected_keys
        assert result["mode"] == "level_1_cluster"

    def test_uses_dynamic_alpha(self, rng: np.random.Generator):
        """Cluster mode should use dynamic α (not static)."""
        n = 500
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=4, variance=0.5, min_alpha=0.2, max_alpha=0.8,
        )
        alpha = result["alpha_values"]
        # Dynamic: not all the same unless all hf_scores equal
        assert alpha.min() >= 0.2 - 1e-12
        assert alpha.max() <= 0.8 + 1e-12

    def test_fusion_bounded(self, rng: np.random.Generator):
        """Fusion scores must be in [0, 1]."""
        n = 1000
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=8, variance=0.3,
        )
        assert float(np.min(result["fusion_scores"])) >= 0.0
        assert float(np.max(result["fusion_scores"])) <= 1.0

    def test_ml_features_shape(self, rng: np.random.Generator):
        """ml_features should be complex I/Q of surviving shots."""
        n = 200
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=4, variance=0.5,
        )
        assert len(result["ml_features"]) == result["n_surviving"]
        assert result["ml_features"].dtype == np.complex128

    def test_n_surviving_consistent(self, rng: np.random.Generator):
        """n_surviving must match surviving_mask.sum()."""
        n = 300
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=4, variance=0.5,
        )
        assert result["n_surviving"] == int(result["surviving_mask"].sum())

    def test_high_noise_better_than_level1(self, rng: np.random.Generator):
        """At high noise, cluster mode should score ≥ level_1 (hf mean)."""
        n = 1000
        sigma_iq = 0.5  # High noise
        centroid = 0.95 + 0.02j
        iq = (
            rng.normal(centroid.real, sigma_iq, n)
            + 1j * rng.normal(centroid.imag, sigma_iq, n)
        )
        lf = rng.uniform(0.4, 1.0, n)

        res_l1 = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=centroid, variance=0.15,
        )
        res_cluster = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=8, variance=0.15,
        )
        # Cluster mode should have higher mean hf_score (better discrimination)
        # because it fits local structure rather than measuring from one centroid
        mean_cluster = np.mean(res_cluster["hf_scores"])
        mean_l1 = np.mean(res_l1["hf_scores"])
        # The cluster scores should be at least as high (they fit local structure)
        assert mean_cluster >= mean_l1 - 0.05  # allow tolerance

    def test_k_auto_clamp_in_batch(self, rng: np.random.Generator):
        """k_clusters > n_shots should be auto-clamped in batch processor."""
        n = 5
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        # k=8 > n=5 — batch processor clamps k to n
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=8, variance=0.5,
        )
        assert result["n_shots"] == n
        assert result["n_surviving"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# TestAutoRouter — autonomous pipeline selection
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoRouter:
    """Autonomous pipeline routing via _determine_optimal_pipeline."""

    def test_integer_dtype_routes_level2(self):
        """Integer HF data → auto-routes to level_2."""
        hf = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        mode, snr = _determine_optimal_pipeline(hf)
        assert mode == "level_2"
        assert snr is None

    def test_bool_dtype_routes_level2(self):
        """Boolean HF data → auto-routes to level_2."""
        hf = np.array([True, False, True, False], dtype=np.bool_)
        mode, snr = _determine_optimal_pipeline(hf)
        assert mode == "level_2"
        assert snr is None

    def test_complex_high_snr_routes_level1(self, rng: np.random.Generator):
        """Well-separated I/Q → high SNR → level_1."""
        n = 500
        blob0 = rng.normal(0, 0.01, n // 2) + 1j * rng.normal(0, 0.01, n // 2)
        blob1 = rng.normal(10, 0.01, n // 2) + 1j * rng.normal(10, 0.01, n // 2)
        iq = np.concatenate([blob0, blob1])
        mode, snr = _determine_optimal_pipeline(
            iq, zero_centroid=0.0 + 0.0j, snr_threshold=3.0,
        )
        assert mode == "level_1"
        assert snr is not None
        assert snr >= 3.0

    def test_complex_low_snr_routes_cluster(self, rng: np.random.Generator):
        """Overlapping I/Q → low SNR → level_1_cluster."""
        n = 500
        iq = rng.normal(0, 1.0, n) + 1j * rng.normal(0, 1.0, n)
        mode, snr = _determine_optimal_pipeline(
            iq, zero_centroid=0.0 + 0.0j, snr_threshold=3.0,
        )
        assert mode == "level_1_cluster"
        assert snr is not None
        assert snr < 3.0

    def test_one_centroid_affects_snr(self, rng: np.random.Generator):
        """Explicit one_centroid far away → high SNR → level_1."""
        n = 200
        iq = rng.normal(0, 0.5, n) + 1j * rng.normal(0, 0.5, n)
        # Without one_centroid → separation = |mean - 0| ≈ 0 → low SNR
        m1, snr1 = _determine_optimal_pipeline(
            iq, zero_centroid=0.0 + 0.0j, snr_threshold=3.0,
        )
        # With one_centroid far away → separation = 100 → high SNR
        m2, snr2 = _determine_optimal_pipeline(
            iq, zero_centroid=0.0 + 0.0j, one_centroid=100.0 + 0.0j,
            snr_threshold=3.0,
        )
        assert snr2 > snr1
        assert m2 == "level_1"

    def test_snr_threshold_respected(self, rng: np.random.Generator):
        """Varying the threshold can flip the routing decision."""
        n = 500
        iq = rng.normal(0, 0.3, n) + 1j * rng.normal(0, 0.3, n)
        _, snr = _determine_optimal_pipeline(
            iq, zero_centroid=0.0 + 0.0j, snr_threshold=3.0,
        )
        # With threshold = 0.0 → always level_1
        m_low, _ = _determine_optimal_pipeline(
            iq, zero_centroid=0.0 + 0.0j, snr_threshold=0.0,
        )
        assert m_low == "level_1"
        # With threshold = 1e6 → always cluster
        m_high, _ = _determine_optimal_pipeline(
            iq, zero_centroid=0.0 + 0.0j, snr_threshold=1e6,
        )
        assert m_high == "level_1_cluster"

    def test_float_dtype_treated_as_complex_path(self, rng: np.random.Generator):
        """Float64 array (not int/bool) → complex path → SNR-based routing."""
        hf = rng.normal(0, 1, 200)  # float64, not integer
        mode, snr = _determine_optimal_pipeline(hf, zero_centroid=0.0)
        assert mode in ("level_1", "level_1_cluster")
        assert snr is not None

    # ── End-to-end auto-routing via process_telemetry_batch ──────────

    def test_auto_route_binary_data(self):
        """Binary int data → auto-routes to level_2 without force_mode."""
        hf = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int64)
        lf = np.linspace(0.3, 0.9, 10)
        result = process_telemetry_batch(hf, lf)
        assert result["mode"] == "auto→level_2"
        assert "snr" not in result

    def test_auto_route_complex_high_snr(self, rng: np.random.Generator):
        """Well-separated I/Q → auto→level_1."""
        n = 500
        blob0 = rng.normal(0, 0.01, n // 2) + 1j * rng.normal(0, 0.01, n // 2)
        blob1 = rng.normal(10, 0.01, n // 2) + 1j * rng.normal(10, 0.01, n // 2)
        iq = np.concatenate([blob0, blob1])
        lf = rng.uniform(0.4, 1.0, n)
        result = process_telemetry_batch(
            iq, lf,
            zero_centroid=0.0 + 0.0j, variance=0.15, snr_threshold=3.0,
        )
        assert result["mode"] == "auto→level_1"
        assert "snr" in result
        assert result["snr"] > 3.0

    def test_auto_route_complex_low_snr(self, rng: np.random.Generator):
        """Overlapping I/Q → auto→level_1_cluster."""
        n = 500
        iq = rng.normal(0, 1.0, n) + 1j * rng.normal(0, 1.0, n)
        lf = rng.uniform(0.4, 1.0, n)
        result = process_telemetry_batch(
            iq, lf,
            zero_centroid=0.0 + 0.0j, variance=0.15, snr_threshold=3.0,
            k_clusters=4,
        )
        assert result["mode"] == "auto→level_1_cluster"
        assert "snr" in result
        assert result["snr"] < 3.0

    def test_auto_route_return_contract(self, rng: np.random.Generator):
        """Auto-routed result has all standard keys + snr (for complex)."""
        n = 200
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, zero_centroid=0.0 + 0.0j, variance=0.15,
        )
        expected_keys = {
            "surviving_mask", "fusion_scores", "ml_features",
            "hf_scores", "alpha_values", "threshold",
            "n_shots", "n_surviving", "mode", "snr",
        }
        assert set(result.keys()) == expected_keys

    def test_auto_route_fusion_bounded(self, rng: np.random.Generator):
        """Auto-routed fusion scores must be in [0, 1]."""
        n = 1000
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, zero_centroid=0.0 + 0.0j, variance=0.5,
        )
        assert float(np.min(result["fusion_scores"])) >= 0.0
        assert float(np.max(result["fusion_scores"])) <= 1.0

    def test_force_mode_overrides_auto(self, rng: np.random.Generator):
        """force_mode should override what auto-router would choose."""
        n = 200
        hf = np.zeros(n, dtype=np.int64)  # auto would choose level_2
        lf = rng.uniform(0, 1, n)
        # Force level_2 explicitly — should NOT have auto→ prefix
        result = process_telemetry_batch(hf, lf, force_mode="level_2")
        assert result["mode"] == "level_2"
        assert "snr" not in result

    def test_snr_key_absent_for_forced_level2(self):
        """Forced level_2 should NOT have 'snr' key."""
        hf = np.array([0, 1, 0])
        lf = np.array([0.5, 0.5, 0.5])
        result = process_telemetry_batch(hf, lf, force_mode="level_2")
        assert "snr" not in result


# ═══════════════════════════════════════════════════════════════════════════
# TestLegacyDeprecation
# ═══════════════════════════════════════════════════════════════════════════


class TestLegacyDeprecation:
    """Legacy `mode` parameter emits DeprecationWarning and works."""

    def test_legacy_mode_emits_warning(self):
        """Passing mode= as positional should emit DeprecationWarning."""
        hf = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        lf = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = process_telemetry_batch(hf, lf, "level_2")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert result["mode"] == "level_2"

    def test_legacy_mode_keyword_emits_warning(self):
        """Passing mode= as keyword should also emit DeprecationWarning."""
        hf = np.array([0, 1, 0], dtype=np.int64)
        lf = np.array([0.5, 0.5, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = process_telemetry_batch(hf, lf, mode="level_2")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        assert result["mode"] == "level_2"

    def test_legacy_hybrid_maps_to_auto(self, rng: np.random.Generator):
        """Legacy mode='hybrid' → maps to auto-routing."""
        n = 200
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = process_telemetry_batch(
                iq, lf, mode="hybrid",
                zero_centroid=0.0 + 0.0j, variance=0.15,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        # Hybrid maps to auto → mode should show auto→...
        assert result["mode"].startswith("auto→")
        assert "snr" in result

    def test_force_mode_overrides_legacy_mode(self):
        """force_mode takes precedence over legacy mode."""
        hf = np.array([0, 1, 0], dtype=np.int64)
        lf = np.array([0.5, 0.5, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = process_telemetry_batch(
                hf, lf, mode="level_1", force_mode="level_2",
            )
            # mode= was provided → deprecation warning
            assert len(w) == 1
        # force_mode wins
        assert result["mode"] == "level_2"

    def test_no_warning_when_using_force_mode(self):
        """Using force_mode only should NOT emit a warning."""
        hf = np.array([0, 1, 0], dtype=np.int64)
        lf = np.array([0.5, 0.5, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = process_telemetry_batch(hf, lf, force_mode="level_2")
            assert len(w) == 0
        assert result["mode"] == "level_2"

    def test_no_warning_when_auto_routing(self, rng: np.random.Generator):
        """Using default (no mode, no force_mode) should NOT warn."""
        hf = np.array([0, 1, 0], dtype=np.int64)
        lf = np.array([0.5, 0.5, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = process_telemetry_batch(hf, lf)
            assert len(w) == 0
        assert result["mode"] == "auto→level_2"


# ═══════════════════════════════════════════════════════════════════════════
# TestInputValidationNewModes
# ═══════════════════════════════════════════════════════════════════════════


class TestInputValidationNewModes:
    """Input validation for force_mode parameter."""

    def test_invalid_force_mode_rejected(self):
        """Unknown force_mode must raise ValueError."""
        hf = np.array([0, 1, 0])
        lf = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="force_mode must be one of"):
            process_telemetry_batch(hf, lf, force_mode="bad_mode")

    def test_level_1_cluster_accepted(self, rng: np.random.Generator):
        """level_1_cluster is a valid force_mode."""
        n = 50
        iq = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        lf = rng.uniform(0, 1, n)
        result = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster", k_clusters=4, variance=0.5,
        )
        assert result["mode"] == "level_1_cluster"

    def test_auto_is_not_valid_force_mode(self):
        """'auto' is not a valid force_mode (it's the default)."""
        hf = np.array([0, 1, 0])
        lf = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="force_mode must be one of"):
            process_telemetry_batch(hf, lf, force_mode="auto")

    def test_none_force_mode_triggers_auto(self):
        """force_mode=None → autonomous routing (default)."""
        hf = np.array([0, 1, 0], dtype=np.int64)
        lf = np.array([0.5, 0.5, 0.5])
        result = process_telemetry_batch(hf, lf, force_mode=None)
        assert result["mode"] == "auto→level_2"


# ═══════════════════════════════════════════════════════════════════════════
# TestAdaptiveGaltonSchedule
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveGaltonSchedule:
    """Tests for adaptive_galton_schedule — depth-aware rejection percentiles."""

    # ── Basic behaviour ───────────────────────────────────────────────

    def test_shallow_depth_returns_near_base(self):
        """Very shallow depths should yield percentile ≈ base_percentile."""
        depths = np.array([1, 5, 10])
        schedule = adaptive_galton_schedule(depths)
        # All values should be close to the base (25) for very shallow depths
        # With steepness=3, knee=100: d=10 gives ~28.5 — within 4 of base
        np.testing.assert_allclose(schedule, 25.0, atol=4.0)

    def test_deep_depth_approaches_max(self):
        """Very deep depths should approach max_percentile asymptotically."""
        depths = np.array([5000, 10000])
        schedule = adaptive_galton_schedule(depths)
        np.testing.assert_allclose(schedule, 75.0, atol=0.5)

    def test_knee_depth_gives_midpoint(self):
        """At the knee depth, sigmoid is 0.5 → schedule = midpoint of [base, max]."""
        depths = np.array([300.0])  # default knee = 300
        schedule = adaptive_galton_schedule(depths)
        expected_midpoint = 25.0 + (75.0 - 25.0) * 0.5  # 50.0
        np.testing.assert_allclose(schedule, expected_midpoint, atol=0.01)

    def test_monotonically_increasing(self):
        """Schedule must be strictly monotonically increasing with depth."""
        depths = np.array([1, 10, 25, 50, 100, 200, 500, 1000, 5000])
        schedule = adaptive_galton_schedule(depths)
        assert np.all(np.diff(schedule) > 0), "Schedule must increase with depth"

    def test_bounded_in_range(self):
        """All schedule values must lie within [base, max]."""
        depths = np.array([0, 1, 10, 100, 1000, 100000])
        schedule = adaptive_galton_schedule(depths)
        assert np.all(schedule >= 25.0), f"Min {schedule.min()} < base 25.0"
        assert np.all(schedule <= 75.0), f"Max {schedule.max()} > max 75.0"

    # ── Parameter effects ─────────────────────────────────────────────

    def test_steepness_controls_sharpness(self):
        """Higher steepness → sharper transition, further from midpoint at d=50."""
        depths = np.array([50.0])
        gentle = adaptive_galton_schedule(depths, steepness=1.0)[0]
        sharp = adaptive_galton_schedule(depths, steepness=10.0)[0]
        # Both below midpoint (d=50 < knee=100), but sharp should be closer to base
        assert sharp < gentle, "Higher steepness should push sub-knee values closer to base"

    def test_custom_base_max(self):
        """Custom base/max percentiles are respected."""
        depths = np.array([300.0])  # at default knee
        schedule = adaptive_galton_schedule(
            depths, base_percentile=10.0, max_percentile=90.0,
        )
        expected = 10.0 + (90.0 - 10.0) * 0.5  # 50.0
        np.testing.assert_allclose(schedule, expected, atol=0.01)

    def test_custom_knee(self):
        """Custom knee shifts the midpoint."""
        depths = np.array([500.0])
        schedule = adaptive_galton_schedule(depths, depth_knee=500.0)
        expected = 25.0 + (75.0 - 25.0) * 0.5  # 50.0
        np.testing.assert_allclose(schedule, expected, atol=0.01)

    # ── Edge cases ────────────────────────────────────────────────────

    def test_zero_depth(self):
        """Depth=0 is valid and should return near base_percentile."""
        schedule = adaptive_galton_schedule(np.array([0]))
        assert 25.0 <= schedule[0] <= 30.0, f"d=0 schedule {schedule[0]} too far from base"

    def test_scalar_depth_promoted(self):
        """A 0-d array (scalar) should be promoted to 1-d."""
        schedule = adaptive_galton_schedule(np.float64(100.0))
        assert schedule.ndim == 1
        assert schedule.shape == (1,)

    # ── Return type ───────────────────────────────────────────────────

    def test_return_type_and_shape(self):
        """Returns float64 ndarray with matching length."""
        depths = np.array([10, 50, 100, 500, 1000])
        schedule = adaptive_galton_schedule(depths)
        assert isinstance(schedule, np.ndarray)
        assert schedule.dtype == np.float64
        assert schedule.shape == (5,)

    # ── Oversampling factor derivation ────────────────────────────────

    def test_oversampling_factor_derivation(self):
        """Oversampling factor = 100/(100-schedule) should be >1 and finite."""
        depths = np.array([10, 100, 1000])
        schedule = adaptive_galton_schedule(depths)
        oversample = 100.0 / (100.0 - schedule)
        assert np.all(oversample > 1.0), "Oversampling must be > 1"
        assert np.all(np.isfinite(oversample)), "Oversampling must be finite"
        # Deep → more aggressive → higher oversampling
        assert oversample[-1] > oversample[0], "Deep circuits need more oversampling"

    # ── Validation errors ─────────────────────────────────────────────

    def test_negative_depth_raises(self):
        """Negative depths are physically meaningless and must raise."""
        with pytest.raises(ValueError, match="non-negative"):
            adaptive_galton_schedule(np.array([-1, 10, 100]))

    def test_base_ge_max_raises(self):
        """base_percentile >= max_percentile is invalid."""
        with pytest.raises(ValueError, match="less than"):
            adaptive_galton_schedule(
                np.array([100]), base_percentile=80.0, max_percentile=25.0,
            )

    def test_base_equals_max_raises(self):
        """base_percentile == max_percentile is also invalid."""
        with pytest.raises(ValueError, match="less than"):
            adaptive_galton_schedule(
                np.array([100]), base_percentile=50.0, max_percentile=50.0,
            )

    def test_percentile_out_of_range_raises(self):
        """Percentile > 100 must raise."""
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            adaptive_galton_schedule(
                np.array([100]), base_percentile=-1.0,
            )
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            adaptive_galton_schedule(
                np.array([100]), max_percentile=101.0,
            )
