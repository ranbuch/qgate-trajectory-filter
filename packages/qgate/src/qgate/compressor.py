"""
compressor.py — TelemetryCompressor: two-stage dimensionality reduction.

Implements a scikit-learn compatible transformer that combines **Spatial
Topological Pooling** with **Tree-Based Feature Pruning** (Gini
importance) to compress raw telemetry matrices from 50+ qubit devices
into dense, low-dimensional latent vectors suitable for the Stage-2 ML
regression model.

Why it's needed
---------------

At ≤ 20 qubits the raw telemetry vectors (6 features from
:mod:`~qgate.mitigation`, 5 IQ features from
:mod:`~qgate.pulse_mitigator`, plus spectator metrics and drift scores)
are manageable.  At 50–156+ qubits the feature count explodes —
e.g. 156 qubits × 5 IQ features = 780-dimensional raw input — and
standard ML regressors overfit rapidly on noisy calibration data.

Two-stage compression pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Stage 1 — Spatial / Topological Pooling:**

The heavy-hex (or arbitrary) coupling map is partitioned into local DAG
neighborhoods (subsystem groups).  Features belonging to qubits in the
same neighborhood are averaged, collapsing the per-qubit dimension down
to per-neighborhood dimension.  This step is *physics-informed*: it
exploits the locality of CNOT cross-talk and correlated dephasing.

**Stage 2 — Tree-Based Gini Pruning:**

A ``RandomForestRegressor`` is trained on the pooled features with
the exact calibration observable as the target.  The Gini impurity-
based feature importances are extracted, and only the top
``retain_ratio`` fraction of features are kept.  Everything else is
discarded.

The result is a highly compressed latent vector that preserves
the predictive signal while eliminating redundant and noisy
dimensions.

Usage::

    from qgate.compressor import TelemetryCompressor

    # Define qubit neighborhoods (from coupling map or manual)
    subsystem_map = {
        0: [0, 1, 2],       # neighborhood A
        1: [3, 4, 5],       # neighborhood B
        2: [6, 7, 8, 9],    # neighborhood C
    }

    compressor = TelemetryCompressor(
        subsystem_map=subsystem_map,
        retain_ratio=0.20,
    )

    # Calibrate (learn which pooled features matter)
    compressor.fit(X_telemetry, y_calibration)

    # Compress new data
    X_compressed = compressor.transform(X_telemetry_new)
    # X_compressed.shape[1] << X_telemetry.shape[1]

Integration with scikit-learn pipelines::

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingRegressor

    pipe = Pipeline([
        ("compress", TelemetryCompressor(subsystem_map, retain_ratio=0.25)),
        ("regress",  GradientBoostingRegressor()),
    ])
    pipe.fit(X_raw, y_ideal)
    predictions = pipe.predict(X_new)

Patent reference
----------------
US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
CIP addendum — Telemetry compression for utility-scale ML mitigation.

Licensed under the QGATE Source Available Evaluation License v1.2.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license.

.. warning::
   CONFIDENTIAL — DO NOT PUSH / DO NOT PUBLISH.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("qgate.compressor")

# ---------------------------------------------------------------------------
# Lazy scikit-learn imports — fail gracefully
# ---------------------------------------------------------------------------

try:
    from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
    from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
    from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    BaseEstimator = object  # type: ignore[assignment,misc]
    TransformerMixin = object  # type: ignore[assignment,misc]


def _require_sklearn() -> None:
    """Raise a helpful ``ImportError`` when scikit-learn is missing."""
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for TelemetryCompressor.  "
            "Install with:  pip install qgate[ml]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Subsystem map helpers
# ═══════════════════════════════════════════════════════════════════════════

SubsystemMap = Dict[int, List[int]]
"""Type alias for subsystem neighbourhood definitions.

Maps **neighbourhood ID** (arbitrary integer key) →  **list of qubit
column indices** in the raw telemetry matrix that belong to that
neighbourhood.  The column indices are *0-based* and refer to columns
in the input ``X`` matrix — **not** physical qubit IDs, unless the
telemetry matrix was constructed with one column per qubit.

Example::

    # 10-qubit device split into 3 local DAG clusters
    subsystem_map: SubsystemMap = {
        0: [0, 1, 2, 3],   # columns 0-3 → neighbourhood A
        1: [4, 5, 6],       # columns 4-6 → neighbourhood B
        2: [7, 8, 9],       # columns 7-9 → neighbourhood C
    }
"""


def _validate_subsystem_map(
    subsystem_map: SubsystemMap,
    *,
    n_features: Optional[int] = None,
) -> None:
    """Validate the subsystem map structure and column indices.

    Args:
        subsystem_map: Neighbourhood → column-index mapping.
        n_features:    If given, validate that every column index is
                       within ``[0, n_features)``.

    Raises:
        TypeError:  If *subsystem_map* is not a dict or contains
                    non-list values.
        ValueError: If any neighbourhood is empty, contains duplicate
                    column indices, or references out-of-range columns.
    """
    if not isinstance(subsystem_map, dict):
        raise TypeError(
            f"subsystem_map must be a dict, got {type(subsystem_map).__name__}"
        )
    if len(subsystem_map) == 0:
        raise ValueError("subsystem_map must have at least one neighbourhood")

    all_indices: List[int] = []
    for key, cols in subsystem_map.items():
        if not isinstance(cols, (list, tuple)):
            raise TypeError(
                f"subsystem_map[{key!r}] must be a list of column indices, "
                f"got {type(cols).__name__}"
            )
        if len(cols) == 0:
            raise ValueError(
                f"subsystem_map[{key!r}] is empty — every neighbourhood "
                f"must contain at least one column index"
            )
        for idx in cols:
            if not isinstance(idx, (int, np.integer)):
                raise TypeError(
                    f"Column index {idx!r} in subsystem_map[{key!r}] "
                    f"must be an integer"
                )
            if idx < 0:
                raise ValueError(
                    f"Column index {idx} in subsystem_map[{key!r}] "
                    f"must be non-negative"
                )
            if n_features is not None and idx >= n_features:
                raise ValueError(
                    f"Column index {idx} in subsystem_map[{key!r}] "
                    f"exceeds the feature count ({n_features}). "
                    f"Valid range: [0, {n_features - 1}]"
                )
        all_indices.extend(cols)

    # Check for duplicates across neighbourhoods
    if len(all_indices) != len(set(all_indices)):
        seen: Dict[int, int] = {}
        for key, cols in subsystem_map.items():
            for idx in cols:
                if idx in seen:
                    raise ValueError(
                        f"Column index {idx} appears in both "
                        f"subsystem_map[{seen[idx]!r}] and "
                        f"subsystem_map[{key!r}] — each column must "
                        f"belong to exactly one neighbourhood"
                    )
                seen[idx] = key


# ═══════════════════════════════════════════════════════════════════════════
# TelemetryCompressor
# ═══════════════════════════════════════════════════════════════════════════


class TelemetryCompressor(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Two-stage telemetry compression: spatial pooling + Gini pruning.

    Compresses high-dimensional raw telemetry matrices from 50–156+
    qubit devices into dense latent vectors suitable for the Stage-2
    ML regression model (:class:`~qgate.mitigation.TelemetryMitigator`).

    **Stage 1 — Spatial Topological Pooling:**
    Features are grouped by local DAG neighbourhoods (defined in
    ``subsystem_map``) and averaged within each group.  This exploits
    the locality of cross-talk and correlated dephasing to reduce
    baseline dimensionality.

    **Stage 2 — Tree-Based Gini Pruning:**
    A ``RandomForestRegressor`` is fitted on the pooled features.
    Gini impurity-based feature importances identify the most
    predictive pooled columns.  Only the top ``retain_ratio``
    fraction are kept; the rest are discarded.

    The class inherits from :class:`sklearn.base.BaseEstimator` and
    :class:`sklearn.base.TransformerMixin`, making it fully compatible
    with :class:`sklearn.pipeline.Pipeline`.

    Parameters
    ----------
    subsystem_map : dict[int, list[int]]
        Maps neighbourhood IDs to lists of column indices in the raw
        telemetry matrix ``X``.  Every column referenced must exist
        in ``X``; no column may appear in more than one neighbourhood.
        Columns *not* covered by the map are preserved as-is (each
        becomes its own 1-column "neighbourhood").

    retain_ratio : float, default 0.20
        Fraction of pooled features to retain after Gini pruning.
        Must be in ``(0.0, 1.0]``.

    n_estimators : int, default 200
        Number of trees in the internal ``RandomForestRegressor``
        used for importance estimation.

    random_state : int or None, default 42
        RNG seed for the internal forest (reproducibility).

    aggregation : {'mean', 'median', 'max'}, default 'mean'
        How to aggregate features within each neighbourhood.

    Attributes
    ----------
    n_features_in_ : int
        Number of raw features seen during ``fit()``.

    n_pooled_features_ : int
        Number of features after spatial pooling.

    n_compressed_features_ : int
        Number of features after Gini pruning (final output width).

    feature_importances_ : ndarray of shape (n_pooled_features_,)
        Gini importances from the internal forest.

    support_mask_ : ndarray of shape (n_pooled_features_,)
        Boolean mask — ``True`` for retained pooled features.

    importance_forest_ : RandomForestRegressor
        The fitted internal forest (exposed for inspection /
        debugging, not intended for direct use).

    Examples
    --------
    Basic usage::

        from qgate.compressor import TelemetryCompressor

        subsystem_map = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}
        tc = TelemetryCompressor(subsystem_map, retain_ratio=0.50)
        tc.fit(X_train, y_train)
        X_compressed = tc.transform(X_test)

    Inside an sklearn pipeline::

        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import GradientBoostingRegressor

        pipe = Pipeline([
            ("compress", TelemetryCompressor(subsystem_map)),
            ("regress",  GradientBoostingRegressor()),
        ])
        pipe.fit(X, y)

    Patent reference:
        US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
        CIP addendum — telemetry compression for utility-scale ML mitigation.
    """

    # scikit-learn convention: all __init__ params stored as attributes
    # with the same name, no computation in __init__.

    _VALID_AGGREGATIONS = ("mean", "median", "max")

    def __init__(
        self,
        subsystem_map: SubsystemMap,
        retain_ratio: float = 0.20,
        n_estimators: int = 200,
        random_state: Optional[int] = 42,
        aggregation: str = "mean",
    ) -> None:
        _require_sklearn()

        # ── Validate immutable params ─────────────────────────────────
        _validate_subsystem_map(subsystem_map)

        if not isinstance(retain_ratio, (int, float)):
            raise TypeError(
                f"retain_ratio must be a float, got {type(retain_ratio).__name__}"
            )
        if not (0.0 < retain_ratio <= 1.0):
            raise ValueError(
                f"retain_ratio must be in (0.0, 1.0], got {retain_ratio}"
            )

        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError(
                f"n_estimators must be a positive integer, got {n_estimators!r}"
            )

        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {self._VALID_AGGREGATIONS}, "
                f"got {aggregation!r}"
            )

        self.subsystem_map = subsystem_map
        self.retain_ratio = retain_ratio
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.aggregation = aggregation

    # ------------------------------------------------------------------
    # Internal — spatial pooling
    # ------------------------------------------------------------------

    def _build_pool_plan(self, n_features: int) -> List[List[int]]:
        """Build the ordered list of column-groups for spatial pooling.

        Groups are ordered by neighbourhood key.  Columns **not**
        covered by ``subsystem_map`` are appended as singletons (each
        uncovered column becomes its own 1-element group).

        Args:
            n_features: Total number of columns in the raw matrix.

        Returns:
            List of column-index lists, one per pooled feature.
        """
        _validate_subsystem_map(self.subsystem_map, n_features=n_features)

        covered: set = set()
        groups: List[List[int]] = []

        # Sorted neighbourhood keys for deterministic ordering
        for key in sorted(self.subsystem_map.keys()):
            cols = list(self.subsystem_map[key])
            groups.append(cols)
            covered.update(cols)

        # Uncovered columns become singleton groups
        for col in range(n_features):
            if col not in covered:
                groups.append([col])

        return groups

    def _spatial_pool(self, X: np.ndarray) -> np.ndarray:
        """Apply Stage 1 spatial/topological pooling.

        Groups columns according to ``subsystem_map`` and aggregates
        each group into a single feature using the configured
        ``aggregation`` function.

        Args:
            X: Raw telemetry matrix of shape ``(n_samples, n_features)``.

        Returns:
            Pooled matrix of shape ``(n_samples, n_pooled_features)``.
        """
        n_samples, n_features = X.shape
        groups = self._build_pool_plan(n_features)

        agg_func = {
            "mean": np.mean,
            "median": np.median,
            "max": np.max,
        }[self.aggregation]

        pooled_cols: List[np.ndarray] = []
        for group in groups:
            # Extract columns → shape (n_samples, group_size)
            block = X[:, group]
            if block.ndim == 1:
                block = block.reshape(-1, 1)
            # Aggregate across columns → shape (n_samples,)
            pooled_cols.append(agg_func(block, axis=1))

        return np.column_stack(pooled_cols)

    # ------------------------------------------------------------------
    # Public API — fit / transform
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "TelemetryCompressor":
        """Fit the two-stage compression pipeline.

        1. Apply spatial pooling to *X*.
        2. Train the internal ``RandomForestRegressor`` on pooled *X*
           with target *y*.
        3. Extract Gini importances and compute the boolean mask
           retaining the top ``retain_ratio`` features.

        Args:
            X: Raw telemetry matrix of shape ``(n_samples, n_features)``.
            y: Calibration target values of shape ``(n_samples,)`` —
               e.g. exact expectation values from near-Clifford circuits.

        Returns:
            ``self`` (for method chaining / pipeline compatibility).

        Raises:
            ImportError: If scikit-learn is not installed.
            ValueError:  If shapes are inconsistent or ``X`` has
                         fewer samples than features (degenerate fit).
        """
        _require_sklearn()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, got shape {X.shape}"
            )
        if y.ndim != 1:
            raise ValueError(
                f"y must be 1-dimensional, got shape {y.shape}"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples: "
                f"X has {X.shape[0]}, y has {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must have at least one sample")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # ── Stage 1: spatial pooling ──────────────────────────────────
        X_pooled = self._spatial_pool(X)
        self.n_pooled_features_ = X_pooled.shape[1]

        logger.info(
            "Stage 1 — spatial pooling: %d raw features → %d pooled "
            "(%d neighbourhoods + %d singletons)",
            n_features,
            self.n_pooled_features_,
            len(self.subsystem_map),
            self.n_pooled_features_ - len(self.subsystem_map),
        )

        # ── Stage 2: tree-based Gini pruning ─────────────────────────
        self.importance_forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.importance_forest_.fit(X_pooled, y)

        importances = self.importance_forest_.feature_importances_
        self.feature_importances_ = importances

        # Number of features to retain (at least 1, at most all)
        n_retain = max(1, int(np.ceil(self.retain_ratio * self.n_pooled_features_)))
        n_retain = min(n_retain, self.n_pooled_features_)

        # Boolean mask: True for top-importance features
        threshold_idx = np.argsort(importances)[-n_retain:]
        mask = np.zeros(self.n_pooled_features_, dtype=bool)
        mask[threshold_idx] = True
        self.support_mask_ = mask
        self.n_compressed_features_ = int(mask.sum())

        logger.info(
            "Stage 2 — Gini pruning: %d pooled → %d retained "
            "(retain_ratio=%.2f, actual=%.2f)",
            self.n_pooled_features_,
            self.n_compressed_features_,
            self.retain_ratio,
            self.n_compressed_features_ / self.n_pooled_features_,
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compress a raw telemetry matrix through the fitted pipeline.

        1. Apply the **same** spatial pooling logic used during ``fit()``.
        2. Slice the pooled matrix with the boolean ``support_mask_``
           learned during ``fit()``.

        Args:
            X: Raw telemetry matrix of shape ``(n_samples, n_features)``.
               Must have the same number of features as the ``X``
               passed to ``fit()``.

        Returns:
            Compressed matrix of shape
            ``(n_samples, n_compressed_features_)`` — a dense, low-
            dimensional latent vector ready for the Stage-2 regressor.

        Raises:
            NotFittedError: If ``fit()`` has not been called.
            ValueError:     If ``X`` has a different number of features
                            than the training data.
        """
        _require_sklearn()
        check_is_fitted(self, ["support_mask_", "n_features_in_"])

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, got shape {X.shape}"
            )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but TelemetryCompressor "
                f"was fitted with {self.n_features_in_} features"
            )

        X_pooled = self._spatial_pool(X)
        return X_pooled[:, self.support_mask_]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_compression_summary(self) -> Dict[str, Any]:
        """Return a summary dict of the compression pipeline.

        Useful for logging, patent evidence, and debugging.

        Returns:
            Dict with keys: ``n_raw``, ``n_pooled``, ``n_compressed``,
            ``compression_ratio``, ``retain_ratio``, ``aggregation``,
            ``n_neighbourhoods``, ``top_features`` (indices + importances).

        Raises:
            NotFittedError: If ``fit()`` has not been called.
        """
        check_is_fitted(self, ["support_mask_", "n_features_in_"])

        retained_indices = np.where(self.support_mask_)[0].tolist()
        retained_importances = self.feature_importances_[self.support_mask_].tolist()

        return {
            "n_raw": self.n_features_in_,
            "n_pooled": self.n_pooled_features_,
            "n_compressed": self.n_compressed_features_,
            "compression_ratio": float(
                self.n_features_in_ / max(self.n_compressed_features_, 1)
            ),
            "retain_ratio": self.retain_ratio,
            "aggregation": self.aggregation,
            "n_neighbourhoods": len(self.subsystem_map),
            "top_features": [
                {"pooled_index": idx, "importance": imp}
                for idx, imp in zip(retained_indices, retained_importances)
            ],
        }

    def __repr__(self) -> str:
        """Concise repr matching sklearn conventions."""
        fitted = hasattr(self, "support_mask_")
        if fitted:
            return (
                f"TelemetryCompressor("
                f"n_neighbourhoods={len(self.subsystem_map)}, "
                f"retain_ratio={self.retain_ratio}, "
                f"aggregation={self.aggregation!r}, "
                f"fitted={self.n_features_in_}→{self.n_compressed_features_})"
            )
        return (
            f"TelemetryCompressor("
            f"n_neighbourhoods={len(self.subsystem_map)}, "
            f"retain_ratio={self.retain_ratio}, "
            f"aggregation={self.aggregation!r})"
        )
