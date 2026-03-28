"""Tests for qgate.compressor — TelemetryCompressor.

Validates the two-stage telemetry compression pipeline:
  Stage 1: Spatial / topological pooling
  Stage 2: Tree-based Gini pruning

Coverage targets:
  - Construction & parameter validation
  - Subsystem map validation (types, duplicates, out-of-range)
  - Spatial pooling correctness (mean, median, max aggregation)
  - Uncovered columns become singletons
  - fit() / transform() contract
  - Feature-count mismatches rejected
  - Edge cases (1 sample, 1 feature, retain_ratio=1.0, all same importance)
  - Compression summary
  - sklearn Pipeline compatibility (fit → predict)
  - clone() compatibility (get_params / set_params)
  - NotFittedError before transform()
  - Deterministic output with fixed random_state
"""

from __future__ import annotations

import pickle
from typing import Dict, List

import numpy as np
import pytest
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]

from qgate.compressor import (
    SubsystemMap,
    TelemetryCompressor,
    _validate_subsystem_map,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def simple_map() -> SubsystemMap:
    """3-neighbourhood map covering 9 columns."""
    return {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
    }


@pytest.fixture()
def partial_map() -> SubsystemMap:
    """Map that covers only columns 0-5 of a 10-column matrix."""
    return {
        0: [0, 1, 2],
        1: [3, 4, 5],
    }


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_data(simple_map: SubsystemMap, rng: np.random.Generator):
    """100 samples × 9 features, with a structured target."""
    n_samples = 100
    n_features = 9
    X = rng.standard_normal((n_samples, n_features))
    # Target correlates with mean of group 0 — gives group 0 high importance
    y = X[:, 0:3].mean(axis=1) + 0.1 * rng.standard_normal(n_samples)
    return X, y


@pytest.fixture()
def large_data(rng: np.random.Generator):
    """200 samples × 30 features, simulating a 30-qubit IQ telemetry matrix."""
    n_samples = 200
    n_features = 30
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, :5].mean(axis=1) + 0.05 * rng.standard_normal(n_samples)
    subsystem_map: SubsystemMap = {
        i: list(range(i * 5, (i + 1) * 5))
        for i in range(6)  # 6 groups of 5 = 30 columns
    }
    return X, y, subsystem_map


# ═══════════════════════════════════════════════════════════════════════════
# Test: subsystem map validation
# ═══════════════════════════════════════════════════════════════════════════


class TestSubsystemMapValidation:
    """Tests for _validate_subsystem_map()."""

    def test_valid_map_passes(self, simple_map: SubsystemMap) -> None:
        _validate_subsystem_map(simple_map)  # should not raise

    def test_valid_map_with_n_features(self, simple_map: SubsystemMap) -> None:
        _validate_subsystem_map(simple_map, n_features=9)

    def test_empty_map_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one neighbourhood"):
            _validate_subsystem_map({})

    def test_non_dict_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            _validate_subsystem_map([(0, [1, 2])])  # type: ignore[arg-type]

    def test_non_list_values_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a list"):
            _validate_subsystem_map({0: "abc"})  # type: ignore[dict-item]

    def test_empty_neighbourhood_rejected(self) -> None:
        with pytest.raises(ValueError, match="is empty"):
            _validate_subsystem_map({0: []})

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _validate_subsystem_map({0: [-1, 0]})

    def test_non_integer_index_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be an integer"):
            _validate_subsystem_map({0: [0.5, 1]})  # type: ignore[list-item]

    def test_duplicate_across_groups_rejected(self) -> None:
        with pytest.raises(ValueError, match="appears in both"):
            _validate_subsystem_map({0: [0, 1], 1: [1, 2]})

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds the feature count"):
            _validate_subsystem_map({0: [0, 10]}, n_features=5)


# ═══════════════════════════════════════════════════════════════════════════
# Test: construction
# ═══════════════════════════════════════════════════════════════════════════


class TestConstruction:
    """Test TelemetryCompressor __init__ parameter validation."""

    def test_valid_construction(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map, retain_ratio=0.30)
        assert tc.retain_ratio == 0.30
        assert tc.n_estimators == 200
        assert tc.aggregation == "mean"

    def test_custom_params(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(
            simple_map,
            retain_ratio=0.50,
            n_estimators=100,
            random_state=123,
            aggregation="median",
        )
        assert tc.retain_ratio == 0.50
        assert tc.n_estimators == 100
        assert tc.random_state == 123
        assert tc.aggregation == "median"

    def test_retain_ratio_zero_rejected(self, simple_map: SubsystemMap) -> None:
        with pytest.raises(ValueError, match="retain_ratio"):
            TelemetryCompressor(simple_map, retain_ratio=0.0)

    def test_retain_ratio_negative_rejected(self, simple_map: SubsystemMap) -> None:
        with pytest.raises(ValueError, match="retain_ratio"):
            TelemetryCompressor(simple_map, retain_ratio=-0.1)

    def test_retain_ratio_above_one_rejected(self, simple_map: SubsystemMap) -> None:
        with pytest.raises(ValueError, match="retain_ratio"):
            TelemetryCompressor(simple_map, retain_ratio=1.5)

    def test_retain_ratio_one_accepted(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map, retain_ratio=1.0)
        assert tc.retain_ratio == 1.0

    def test_retain_ratio_non_float_rejected(self, simple_map: SubsystemMap) -> None:
        with pytest.raises(TypeError, match="retain_ratio"):
            TelemetryCompressor(simple_map, retain_ratio="high")  # type: ignore[arg-type]

    def test_invalid_n_estimators(self, simple_map: SubsystemMap) -> None:
        with pytest.raises(ValueError, match="n_estimators"):
            TelemetryCompressor(simple_map, n_estimators=0)

    def test_invalid_aggregation(self, simple_map: SubsystemMap) -> None:
        with pytest.raises(ValueError, match="aggregation"):
            TelemetryCompressor(simple_map, aggregation="sum")


# ═══════════════════════════════════════════════════════════════════════════
# Test: spatial pooling
# ═══════════════════════════════════════════════════════════════════════════


class TestSpatialPooling:
    """Test Stage 1 — spatial/topological pooling."""

    def test_mean_pooling_exact(self) -> None:
        """Verify mean aggregation produces correct values."""
        smap: SubsystemMap = {0: [0, 1], 1: [2, 3]}
        tc = TelemetryCompressor(smap, aggregation="mean")
        X = np.array([[1.0, 3.0, 5.0, 7.0]])
        pooled = tc._spatial_pool(X)
        assert pooled.shape == (1, 2)
        np.testing.assert_allclose(pooled[0, 0], 2.0)  # mean(1, 3)
        np.testing.assert_allclose(pooled[0, 1], 6.0)  # mean(5, 7)

    def test_median_pooling_exact(self) -> None:
        smap: SubsystemMap = {0: [0, 1, 2]}
        tc = TelemetryCompressor(smap, aggregation="median")
        X = np.array([[1.0, 5.0, 3.0]])
        pooled = tc._spatial_pool(X)
        np.testing.assert_allclose(pooled[0, 0], 3.0)  # median(1, 5, 3)

    def test_max_pooling_exact(self) -> None:
        smap: SubsystemMap = {0: [0, 1, 2]}
        tc = TelemetryCompressor(smap, aggregation="max")
        X = np.array([[1.0, 5.0, 3.0]])
        pooled = tc._spatial_pool(X)
        np.testing.assert_allclose(pooled[0, 0], 5.0)

    def test_pooling_reduces_columns(self, simple_map: SubsystemMap) -> None:
        """9 columns → 3 pooled features."""
        tc = TelemetryCompressor(simple_map)
        X = np.ones((10, 9))
        pooled = tc._spatial_pool(X)
        assert pooled.shape == (10, 3)

    def test_uncovered_columns_become_singletons(
        self, partial_map: SubsystemMap
    ) -> None:
        """Partial map covering 6 of 10 columns: 2 groups + 4 singletons = 6."""
        tc = TelemetryCompressor(partial_map)
        X = np.ones((5, 10))
        pooled = tc._spatial_pool(X)
        # 2 neighbourhoods + 4 uncovered singletons (cols 6, 7, 8, 9)
        assert pooled.shape == (5, 6)

    def test_singleton_columns_preserve_values(self) -> None:
        """Uncovered columns pass through unchanged."""
        smap: SubsystemMap = {0: [0, 1]}
        tc = TelemetryCompressor(smap)
        X = np.array([[10.0, 20.0, 99.0]])  # col 2 is uncovered
        pooled = tc._spatial_pool(X)
        assert pooled.shape == (1, 2)  # 1 group + 1 singleton
        np.testing.assert_allclose(pooled[0, 0], 15.0)  # mean(10, 20)
        np.testing.assert_allclose(pooled[0, 1], 99.0)  # singleton passthrough

    def test_pooling_deterministic(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map)
        X = np.arange(27.0).reshape(3, 9)
        p1 = tc._spatial_pool(X)
        p2 = tc._spatial_pool(X)
        np.testing.assert_array_equal(p1, p2)

    def test_pooling_multiple_samples(self) -> None:
        """Verify pooling operates row-wise."""
        smap: SubsystemMap = {0: [0, 1]}
        tc = TelemetryCompressor(smap)
        X = np.array([[2.0, 4.0], [10.0, 20.0], [0.0, 0.0]])
        pooled = tc._spatial_pool(X)
        assert pooled.shape == (3, 1)
        np.testing.assert_allclose(pooled[:, 0], [3.0, 15.0, 0.0])


# ═══════════════════════════════════════════════════════════════════════════
# Test: fit + transform
# ═══════════════════════════════════════════════════════════════════════════


class TestFitTransform:
    """Test full fit → transform pipeline."""

    def test_basic_fit_transform(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map, retain_ratio=0.50)
        tc.fit(X, y)

        assert tc.n_features_in_ == 9
        assert tc.n_pooled_features_ == 3
        # retain 0.50 of 3 = ceil(1.5) = 2
        assert tc.n_compressed_features_ == 2

        X_out = tc.transform(X)
        assert X_out.shape == (100, 2)

    def test_fit_transform_shortcut(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        """fit_transform() inherited from TransformerMixin works."""
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map, retain_ratio=0.50)
        X_out = tc.fit_transform(X, y)
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] == tc.n_compressed_features_

    def test_retain_ratio_one_keeps_all(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map, retain_ratio=1.0)
        tc.fit(X, y)
        assert tc.n_compressed_features_ == tc.n_pooled_features_

    def test_very_small_retain_keeps_at_least_one(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map, retain_ratio=0.01)
        tc.fit(X, y)
        assert tc.n_compressed_features_ >= 1

    def test_transform_wrong_feature_count_rejected(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map)
        tc.fit(X, y)
        with pytest.raises(ValueError, match="features"):
            tc.transform(np.ones((5, 20)))

    def test_transform_before_fit_raises(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map)
        with pytest.raises(Exception):  # NotFittedError
            tc.transform(np.ones((5, 9)))

    def test_1d_X_rejected(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map)
        with pytest.raises(ValueError, match="2-dimensional"):
            tc.fit(np.ones(9), np.ones(9))

    def test_empty_X_rejected(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map)
        with pytest.raises(ValueError, match="at least one sample"):
            tc.fit(np.ones((0, 9)), np.ones(0))

    def test_X_y_length_mismatch(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map)
        with pytest.raises(ValueError, match="same number of samples"):
            tc.fit(np.ones((10, 9)), np.ones(5))

    def test_feature_importances_stored(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map)
        tc.fit(X, y)
        assert hasattr(tc, "feature_importances_")
        assert tc.feature_importances_.shape == (tc.n_pooled_features_,)
        assert np.all(tc.feature_importances_ >= 0)
        np.testing.assert_allclose(tc.feature_importances_.sum(), 1.0, atol=0.01)

    def test_support_mask_is_boolean(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map)
        tc.fit(X, y)
        assert tc.support_mask_.dtype == bool
        assert tc.support_mask_.sum() == tc.n_compressed_features_

    def test_deterministic_with_fixed_seed(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc1 = TelemetryCompressor(simple_map, random_state=42)
        tc2 = TelemetryCompressor(simple_map, random_state=42)
        out1 = tc1.fit_transform(X, y)
        out2 = tc2.fit_transform(X, y)
        np.testing.assert_array_equal(out1, out2)


# ═══════════════════════════════════════════════════════════════════════════
# Test: large-scale compression
# ═══════════════════════════════════════════════════════════════════════════


class TestLargeScale:
    """Test on larger data resembling real 30+ qubit devices."""

    def test_30_feature_compression(self, large_data: tuple) -> None:
        X, y, smap = large_data
        tc = TelemetryCompressor(smap, retain_ratio=0.20)
        tc.fit(X, y)

        # 30 raw → 6 pooled → ceil(0.20 × 6) = 2 retained
        assert tc.n_features_in_ == 30
        assert tc.n_pooled_features_ == 6
        assert tc.n_compressed_features_ == 2

        X_out = tc.transform(X)
        assert X_out.shape == (200, 2)

    def test_group0_has_highest_importance(self, large_data: tuple) -> None:
        """Target correlates with group 0 → group 0 should dominate importance."""
        X, y, smap = large_data
        tc = TelemetryCompressor(smap, retain_ratio=0.50)
        tc.fit(X, y)
        # Group 0 (pooled feature index 0) should have top importance
        assert tc.feature_importances_[0] == tc.feature_importances_.max()

    def test_compression_ratio_reported(self, large_data: tuple) -> None:
        X, y, smap = large_data
        tc = TelemetryCompressor(smap, retain_ratio=0.20)
        tc.fit(X, y)
        summary = tc.get_compression_summary()
        assert summary["n_raw"] == 30
        assert summary["n_pooled"] == 6
        assert summary["n_compressed"] == 2
        assert summary["compression_ratio"] == 30 / 2


# ═══════════════════════════════════════════════════════════════════════════
# Test: sklearn integration
# ═══════════════════════════════════════════════════════════════════════════


class TestSklearnIntegration:
    """Test compatibility with sklearn Pipeline, clone, pickle."""

    def test_pipeline_fit_predict(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        pipe = Pipeline([
            ("compress", TelemetryCompressor(simple_map, retain_ratio=0.50)),
            ("regress", GradientBoostingRegressor(n_estimators=10, random_state=42)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (100,)
        # Pipeline should actually learn something (R² > 0)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.0

    def test_clone_preserves_params(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(
            simple_map, retain_ratio=0.35, n_estimators=100, aggregation="median"
        )
        cloned = clone(tc)
        assert cloned.retain_ratio == 0.35
        assert cloned.n_estimators == 100
        assert cloned.aggregation == "median"
        assert cloned.subsystem_map == simple_map

    def test_get_params(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map, retain_ratio=0.25)
        params = tc.get_params()
        assert params["retain_ratio"] == 0.25
        assert params["subsystem_map"] is simple_map

    def test_set_params(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map)
        tc.set_params(retain_ratio=0.60)
        assert tc.retain_ratio == 0.60

    def test_pickle_roundtrip(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map, retain_ratio=0.50)
        tc.fit(X, y)

        blob = pickle.dumps(tc)
        tc2 = pickle.loads(blob)

        X_out1 = tc.transform(X)
        X_out2 = tc2.transform(X)
        np.testing.assert_array_equal(X_out1, X_out2)


# ═══════════════════════════════════════════════════════════════════════════
# Test: edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge-case and boundary-condition tests."""

    def test_single_sample(self) -> None:
        smap: SubsystemMap = {0: [0, 1]}
        tc = TelemetryCompressor(smap, retain_ratio=1.0)
        X = np.array([[3.0, 7.0]])
        y = np.array([5.0])
        tc.fit(X, y)
        out = tc.transform(X)
        assert out.shape == (1, 1)  # 1 pooled feature retained

    def test_single_feature(self) -> None:
        smap: SubsystemMap = {0: [0]}
        tc = TelemetryCompressor(smap, retain_ratio=1.0)
        X = np.arange(10.0).reshape(-1, 1)
        y = X.ravel() * 2
        tc.fit(X, y)
        out = tc.transform(X)
        assert out.shape == (10, 1)

    def test_all_features_singleton(self) -> None:
        """Each column is its own neighbourhood — no aggregation."""
        smap: SubsystemMap = {i: [i] for i in range(5)}
        tc = TelemetryCompressor(smap, retain_ratio=0.40)
        rng = np.random.default_rng(99)
        X = rng.standard_normal((50, 5))
        y = X[:, 0] + 0.1 * rng.standard_normal(50)
        tc.fit(X, y)
        # 5 pooled → ceil(0.40 × 5) = 2 retained
        assert tc.n_compressed_features_ == 2

    def test_constant_target_still_works(self, simple_map: SubsystemMap) -> None:
        """Even with constant y, fit should not crash."""
        X = np.ones((20, 9))
        y = np.zeros(20)
        tc = TelemetryCompressor(simple_map)
        tc.fit(X, y)
        out = tc.transform(X)
        assert out.shape[0] == 20

    def test_transform_single_row(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map, retain_ratio=0.50)
        tc.fit(X, y)
        out = tc.transform(X[:1])
        assert out.shape == (1, tc.n_compressed_features_)

    def test_repr_unfitted(self, simple_map: SubsystemMap) -> None:
        tc = TelemetryCompressor(simple_map, retain_ratio=0.30)
        r = repr(tc)
        assert "TelemetryCompressor" in r
        assert "retain_ratio=0.3" in r

    def test_repr_fitted(
        self,
        simple_map: SubsystemMap,
        synthetic_data: tuple,
    ) -> None:
        X, y = synthetic_data
        tc = TelemetryCompressor(simple_map, retain_ratio=0.50)
        tc.fit(X, y)
        r = repr(tc)
        assert "fitted=" in r

    def test_compression_summary_before_fit_raises(
        self, simple_map: SubsystemMap
    ) -> None:
        tc = TelemetryCompressor(simple_map)
        with pytest.raises(Exception):  # NotFittedError
            tc.get_compression_summary()


# ═══════════════════════════════════════════════════════════════════════════
# Test: aggregation modes
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregationModes:
    """Verify all three aggregation modes produce correct pooling."""

    @pytest.fixture()
    def setup(self):
        smap: SubsystemMap = {0: [0, 1, 2]}
        X = np.array([[1.0, 2.0, 6.0], [10.0, 20.0, 30.0]])
        y = np.array([3.0, 20.0])
        return smap, X, y

    def test_mean_mode(self, setup: tuple) -> None:
        smap, X, y = setup
        tc = TelemetryCompressor(smap, aggregation="mean")
        tc.fit(X, y)
        pooled = tc._spatial_pool(X)
        np.testing.assert_allclose(pooled[:, 0], [3.0, 20.0])

    def test_median_mode(self, setup: tuple) -> None:
        smap, X, y = setup
        tc = TelemetryCompressor(smap, aggregation="median")
        tc.fit(X, y)
        pooled = tc._spatial_pool(X)
        np.testing.assert_allclose(pooled[:, 0], [2.0, 20.0])

    def test_max_mode(self, setup: tuple) -> None:
        smap, X, y = setup
        tc = TelemetryCompressor(smap, aggregation="max")
        tc.fit(X, y)
        pooled = tc._spatial_pool(X)
        np.testing.assert_allclose(pooled[:, 0], [6.0, 30.0])
