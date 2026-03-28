"""
tvs.py — Trajectory Viability Score: HF/LF fusion and Stage-1 filtering.

Calculates a per-shot **Trajectory Viability Score** (TVS) that fuses
high-frequency (HF) and low-frequency (LF) telemetry, then applies
Stage-1 statistical filtering (Galton percentile rejection) to produce
a surviving-shot mask for downstream Stage-2 ML regressors.

Hardware modes
--------------

The module supports autonomous pipeline selection.  By default, the
optimal readout pipeline is determined automatically from the data
dtype and I/Q cloud separability.  For debugging or benchmarking,
a ``force_mode`` parameter overrides the auto-router.

**Level-2** — *hard-decision decoding*
    HF data is a binary array of 0s and 1s (bit-string readout).
    Fusion uses a static α weight.  Selected automatically when HF
    data has integer or boolean dtype.

**Level-1** — *soft-decision decoding*
    HF data is a complex-valued array of raw I + iQ microwave baseband
    samples.  Distances to a calibrated ``zero_centroid`` are converted
    to probabilities via a Gaussian Radial Basis Function (RBF), and
    fusion uses a per-shot *dynamic* α derived from HF confidence.
    Selected automatically when I/Q clouds are well-separated (SNR ≥ threshold).

**Level-1-cluster** — *multi-centroid decoding*
    For high-noise regimes where the I/Q clouds overlap, a single
    centroid RBF collapses.  This mode clusters I/Q data with
    ``MiniBatchKMeans`` into *k* Voronoi regions and scores each shot
    by its RBF distance to the assigned cluster centre.  Intra-cluster
    compactness becomes the fidelity proxy.  Selected automatically
    when I/Q clouds overlap heavily (SNR < threshold).

**Auto (default)** — *autonomous routing*
    The ``_determine_optimal_pipeline`` function inspects the data dtype
    and, for complex data, computes the I/Q SNR to route to the optimal
    sub-pipeline.  This is the default behaviour when ``force_mode`` is
    ``None``.

Pipeline
--------

1. **HF normalisation** — convert raw HF data into ``hf_score ∈ [0, 1]``.
2. **Dynamic α calculation** — Kalman-style: static for Level-2,
   per-shot confidence-weighted for Level-1.
3. **Fusion** — ``fusion_score = hf_score · α + lf_score · (1 − α)``
   (normalised to [0, 1] by construction).
4. **Stage-1 Galton filter** — reject shots below a configurable
   percentile of the fusion-score distribution.

All operations are **fully vectorised** (NumPy) — no Python ``for``
loops — and process tens of thousands of shots with minimal latency.

Usage::

    from qgate.tvs import process_telemetry_batch

    # Auto-routing (default) — pipeline chosen from data dtype + SNR:
    result = process_telemetry_batch(
        hf_data=iq_samples,           # complex128, shape (n_shots,)
        lf_scores=lf_drift_scores,    # float64,    shape (n_shots,)
        zero_centroid=0.95 + 0.02j,
        variance=0.15,
    )

    # Force a specific mode for benchmarking / debugging:
    result = process_telemetry_batch(
        hf_data=binary_bits,
        lf_scores=lf_drift_scores,
        force_mode="level_2",
        alpha=0.6,
    )

    surviving_mask = result["surviving_mask"]
    features       = result["ml_features"]

Integration
~~~~~~~~~~~

The ``surviving_mask`` and ``ml_features`` feed directly into the
Stage-2 :class:`~qgate.mitigation.TelemetryMitigator` regression
pipeline, or into :class:`~qgate.compressor.TelemetryCompressor` for
utility-scale (50+ qubit) dimensionality reduction before ML.

Patent reference
----------------
US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
CIP addendum — ML-augmented TSVF trajectory mitigation.

Licensed under the QGATE Source Available Evaluation License v1.2.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license.

.. warning::
   CONFIDENTIAL — DO NOT PUSH / DO NOT PUBLISH.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("qgate.tvs")

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

#: Supported hardware modes.
VALID_MODES = ("level_1", "level_1_cluster", "level_2", "auto")

#: Modes accepted by ``force_mode`` (excludes "auto" — that's the default).
VALID_FORCE_MODES = ("level_1", "level_1_cluster", "level_2")

#: Legacy modes kept for backward-compatible deprecation shim.
_LEGACY_MODES = ("level_1", "level_1_cluster", "level_2", "hybrid")

#: Default fusion weight for Level-2 mode.
DEFAULT_ALPHA: float = 0.5

#: Default percentile for Galton outlier rejection.
DEFAULT_DROP_PERCENTILE: float = 25.0

#: Default dynamic-alpha range for Level-1 mode.
DEFAULT_MIN_ALPHA: float = 0.3
DEFAULT_MAX_ALPHA: float = 0.9

#: Default RBF variance for Level-1 I/Q → probability conversion.
DEFAULT_VARIANCE: float = 0.15

#: Default number of MiniBatchKMeans clusters for Level-1 cluster mode.
DEFAULT_K_CLUSTERS: int = 8

#: Default SNR threshold for hybrid-mode routing decision.
#: Empirically, SNR ≥ 3.0 implies well-separated I/Q clouds where
#: single-centroid RBF (Level-1) suffices.  Below this, the overlap
#: warrants multi-centroid K-Means clustering.
DEFAULT_SNR_THRESHOLD: float = 3.0

#: Default fallback mode when hybrid SNR is below threshold.
DEFAULT_HYBRID_FALLBACK: str = "level_1_cluster"


# ═══════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════


def _validate_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a supported hardware level."""
    if mode not in VALID_MODES and mode not in _LEGACY_MODES:
        raise ValueError(
            f"mode must be one of {VALID_MODES}, got {mode!r}"
        )


def _validate_force_mode(force_mode: Optional[str]) -> None:
    """Raise ``ValueError`` if *force_mode* is not valid."""
    if force_mode is not None and force_mode not in VALID_FORCE_MODES:
        raise ValueError(
            f"force_mode must be one of {VALID_FORCE_MODES} or None, "
            f"got {force_mode!r}"
        )


def _determine_optimal_pipeline(
    hf_data: np.ndarray,
    zero_centroid: complex = 0.0 + 0.0j,
    one_centroid: Optional[complex] = None,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
) -> Tuple[str, Optional[float]]:
    """Autonomously select the best TVS sub-pipeline.

    Three-step decision logic:

    **Step A — Hardware dtype check:**
      If ``hf_data`` has an integer or boolean dtype, the data comes
      from a Level-2 (hard-decision) backend → return ``"level_2"``.

    **Step B — SNR computation:**
      For complex-valued I/Q data, compute the I/Q cloud separability
      metric ``compute_iq_snr(hf_data, zero_centroid, one_centroid)``.

    **Step C — Physics switch:**
      If SNR ≥ ``snr_threshold``, the clouds are well-separated and
      a single-centroid RBF suffices → return ``"level_1"``.
      Otherwise the overlap warrants K-Means clustering →
      return ``"level_1_cluster"``.

    Args:
        hf_data:        Raw HF telemetry, shape ``(n_shots,)``.
        zero_centroid:  Calibrated |0⟩ centroid for SNR computation.
        one_centroid:   Calibrated |1⟩ centroid (optional).
        snr_threshold:  SNR boundary for the level_1 / level_1_cluster
                        decision.  Default: 3.0.

    Returns:
        Tuple of ``(resolved_mode, snr_value)``.  ``snr_value`` is
        ``None`` for Level-2 data and a float for complex data.

    .. note::
       This is an internal function — not part of the public API.
       It is called automatically by ``process_telemetry_batch`` when
       ``force_mode`` is ``None`` (the default).
    """
    # ── Step A: dtype check ──────────────────────────────────────────
    if np.issubdtype(hf_data.dtype, np.integer) or np.issubdtype(hf_data.dtype, np.bool_):
        logger.info(
            "Auto-router: integer/bool dtype (%s) → level_2",
            hf_data.dtype,
        )
        return "level_2", None

    # ── Step B: SNR computation for complex / float data ─────────────
    snr_value = compute_iq_snr(hf_data, zero_centroid, one_centroid)

    # ── Step C: physics switch ───────────────────────────────────────
    if snr_value >= snr_threshold:
        logger.info(
            "Auto-router: SNR=%.3f ≥ %.3f → level_1 (RBF)",
            snr_value, snr_threshold,
        )
        return "level_1", snr_value

    logger.info(
        "Auto-router: SNR=%.3f < %.3f → level_1_cluster (K-Means+RBF)",
        snr_value, snr_threshold,
    )
    return "level_1_cluster", snr_value


def _validate_arrays(
    hf_data: np.ndarray,
    lf_scores: np.ndarray,
) -> None:
    """Validate shape and dtype compatibility of input arrays.

    Raises:
        ValueError: On empty, mismatched, or wrong-dimension arrays.
        TypeError:  On non-ndarray inputs.
    """
    if not isinstance(hf_data, np.ndarray):
        raise TypeError(
            f"hf_data must be a numpy ndarray, got {type(hf_data).__name__}"
        )
    if not isinstance(lf_scores, np.ndarray):
        raise TypeError(
            f"lf_scores must be a numpy ndarray, got {type(lf_scores).__name__}"
        )
    if hf_data.ndim != 1:
        raise ValueError(
            f"hf_data must be 1-dimensional, got shape {hf_data.shape}"
        )
    if lf_scores.ndim != 1:
        raise ValueError(
            f"lf_scores must be 1-dimensional, got shape {lf_scores.shape}"
        )
    if hf_data.shape[0] == 0:
        raise ValueError("hf_data must not be empty")
    if hf_data.shape[0] != lf_scores.shape[0]:
        raise ValueError(
            f"hf_data and lf_scores must have the same length: "
            f"hf_data has {hf_data.shape[0]}, lf_scores has {lf_scores.shape[0]}"
        )


def _validate_lf_bounds(lf_scores: np.ndarray) -> None:
    """Warn (but do not fail) if LF scores are outside [0, 1]."""
    mn, mx = float(np.min(lf_scores)), float(np.max(lf_scores))
    if mn < 0.0 or mx > 1.0:
        logger.warning(
            "lf_scores outside [0, 1] range (min=%.4f, max=%.4f). "
            "Clamping to [0, 1].",
            mn, mx,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — HF Normalisation
# ═══════════════════════════════════════════════════════════════════════════


def normalise_hf_level2(hf_data: np.ndarray) -> np.ndarray:
    """Convert Level-2 binary HF data to viability scores.

    Hard-decision decoding: bit 0 → 1.0 (ideal), bit 1 → 0.0 (error).

    Args:
        hf_data: Integer array of 0s and 1s, shape ``(n_shots,)``.

    Returns:
        ``hf_score`` array of shape ``(n_shots,)`` with values in {0.0, 1.0}.

    Raises:
        ValueError: If *hf_data* contains values other than 0 and 1.
    """
    hf = np.asarray(hf_data, dtype=np.float64)
    unique_vals = np.unique(hf)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(
            f"Level-2 hf_data must contain only 0 and 1, "
            f"got unique values: {unique_vals.tolist()}"
        )
    # Flip: 0 → 1.0 (good), 1 → 0.0 (error)
    return 1.0 - hf


def normalise_hf_level1(
    hf_data: np.ndarray,
    zero_centroid: complex,
    variance: float,
) -> np.ndarray:
    """Convert Level-1 I/Q complex samples to viability scores via RBF.

    Soft-decision decoding: computes the Euclidean distance from each
    I/Q sample to the calibrated ``zero_centroid``, then maps to
    ``[0, 1]`` via a Gaussian Radial Basis Function:

    .. math::

        d_i = |\\text{hf}[i] - \\text{centroid}|
        \\\\
        \\text{hf\\_score}[i] = \\exp\\!\\left(-\\frac{d_i^2}{\\sigma^2}\\right)

    Points close to the centroid yield scores near 1.0 (high fidelity);
    distant points yield scores near 0.0 (likely error).

    Args:
        hf_data:        Complex-valued array of I + iQ samples, shape ``(n_shots,)``.
        zero_centroid:  Calibration centroid for the |0⟩ state (complex float).
        variance:       Gaussian RBF bandwidth σ² — controls the decay rate.
                        Must be strictly positive.

    Returns:
        ``hf_score`` array of shape ``(n_shots,)`` with values in ``[0, 1]``.

    Raises:
        ValueError: If *variance* is not strictly positive.
        TypeError:  If *zero_centroid* is not a complex or real number.
    """
    if not isinstance(zero_centroid, (complex, float, int, np.complexfloating, np.floating, np.integer)):
        raise TypeError(
            f"zero_centroid must be a numeric type, got {type(zero_centroid).__name__}"
        )
    if variance <= 0:
        raise ValueError(
            f"variance must be strictly positive, got {variance}"
        )

    iq = np.asarray(hf_data, dtype=np.complex128)
    centroid = np.complex128(zero_centroid)

    # Euclidean distance in the I/Q plane
    d = np.abs(iq - centroid)

    # Gaussian RBF: exp(-d² / σ²)
    hf_score = np.exp(-(d ** 2) / variance)

    return hf_score


def normalise_hf_level1_cluster(
    hf_data: np.ndarray,
    k_clusters: int = DEFAULT_K_CLUSTERS,
    variance: float = DEFAULT_VARIANCE,
) -> np.ndarray:
    """Convert Level-1 I/Q samples to viability scores via K-Means + RBF.

    **Physics rationale:** When qubit readout noise is high (large σ_IQ),
    the |0⟩ and |1⟩ I/Q clouds overlap substantially and a single-centroid
    RBF (``normalise_hf_level1``) cannot discriminate — all distances
    become similar, collapsing the score distribution.

    Multi-centroid clustering recovers structure by partitioning the I/Q
    plane into *k* Voronoi regions via ``MiniBatchKMeans``.  Each shot is
    assigned to its nearest cluster centre, and the RBF is computed from
    the shot's distance to *that* centre rather than a single global
    centroid.  Shots tightly grouped around any cluster centre receive
    high scores (consistent readout), while outliers far from all
    centres score low (likely noise-corrupted).

    The key insight is that **intra-cluster compactness**, not distance
    from a fixed |0⟩ reference, becomes the fidelity proxy.

    .. math::

        c_i = \\text{argmin}_{j \\in 1..k} \\;|\\text{iq}[i] - \\mu_j|
        \\\\
        d_i = |\\text{iq}[i] - \\mu_{c_i}|
        \\\\
        \\text{hf\\_score}[i] = \\exp\\!\\left(-\\frac{d_i^2}{\\sigma^2}\\right)

    Args:
        hf_data:      Complex-valued I/Q array, shape ``(n_shots,)``.
        k_clusters:   Number of MiniBatchKMeans clusters (default 8).
                      Must be ≥ 2 and ≤ ``n_shots``.
        variance:     Gaussian RBF bandwidth σ² — controls the decay rate.
                      Must be strictly positive.

    Returns:
        ``hf_score`` array of shape ``(n_shots,)`` with values in ``[0, 1]``.

    Raises:
        ValueError: If *k_clusters* < 2, *k_clusters* > n_shots,
                    or *variance* is not strictly positive.
    """
    from sklearn.cluster import MiniBatchKMeans

    n = hf_data.shape[0]

    if k_clusters < 2:
        raise ValueError(
            f"k_clusters must be >= 2, got {k_clusters}"
        )
    if k_clusters > n:
        raise ValueError(
            f"k_clusters ({k_clusters}) must be <= n_shots ({n})"
        )
    if variance <= 0:
        raise ValueError(
            f"variance must be strictly positive, got {variance}"
        )

    iq = np.asarray(hf_data, dtype=np.complex128)

    # Stack real (I) and imag (Q) as a 2D feature matrix for K-Means.
    X = np.column_stack([iq.real, iq.imag])  # (n_shots, 2)

    kmeans = MiniBatchKMeans(
        n_clusters=k_clusters,
        random_state=42,
        batch_size=min(1024, n),
        n_init=3,
    )
    kmeans.fit(X)

    # Compute distance from each shot to its assigned cluster centre.
    centres = kmeans.cluster_centers_  # (k, 2)
    labels = kmeans.labels_            # (n_shots,)
    assigned_centres = centres[labels]  # (n_shots, 2)

    dx = X[:, 0] - assigned_centres[:, 0]
    dy = X[:, 1] - assigned_centres[:, 1]
    d_sq = dx * dx + dy * dy

    # Gaussian RBF: exp(-d² / σ²)
    hf_score = np.exp(-d_sq / variance)

    return hf_score


def compute_iq_snr(
    hf_data: np.ndarray,
    zero_centroid: complex,
    one_centroid: Optional[complex] = None,
) -> float:
    """Estimate I/Q cloud separability via a Signal-to-Noise Ratio.

    **Physics rationale:** The readout signal of a transmon qubit maps
    the computational basis states |0⟩ and |1⟩ to two distinct
    "blobs" in the I/Q plane.  When these blobs are well-separated
    relative to their spread, single-centroid RBF scoring works well.
    When they overlap heavily, multi-centroid clustering is needed.

    This metric quantifies that separation using a simplified
    Bhattacharyya-like measure:

    .. math::

        \\text{SNR} = \\frac{|\\mu_1 - \\mu_0|}{\\sigma_{\\text{IQ}}}

    where σ_IQ is the standard deviation of the Euclidean distances
    from each shot to the overall mean of the I/Q cloud.

    If ``one_centroid`` is not supplied, the function uses the sample
    mean of all shots as an empirical centroid, making the metric a
    *spread ratio* rather than a true two-state SNR.  When both
    centroids are given, the numerator is the exact inter-centroid
    distance.

    Args:
        hf_data:         Complex-valued I/Q array, shape ``(n_shots,)``.
        zero_centroid:   Calibrated |0⟩ centroid.
        one_centroid:    Calibrated |1⟩ centroid (optional).  If ``None``,
                         uses the sample mean.

    Returns:
        Non-negative float SNR.  Values ≥ 3.0 typically indicate
        well-separated clouds suitable for single-centroid RBF.

    Raises:
        TypeError:  If centroids are not numeric.
        ValueError: If *hf_data* is empty.
    """
    for name, val in [("zero_centroid", zero_centroid), ("one_centroid", one_centroid)]:
        if val is not None and not isinstance(
            val, (complex, float, int, np.complexfloating, np.floating, np.integer)
        ):
            raise TypeError(
                f"{name} must be a numeric type, got {type(val).__name__}"
            )

    iq = np.asarray(hf_data, dtype=np.complex128)
    if iq.shape[0] == 0:
        raise ValueError("hf_data must not be empty")

    c0 = np.complex128(zero_centroid)

    if one_centroid is not None:
        c1 = np.complex128(one_centroid)
        separation = float(np.abs(c1 - c0))
    else:
        # Use sample mean as surrogate for the second centroid.
        sample_mean = np.mean(iq)
        separation = float(np.abs(sample_mean - c0))

    # Spread: std of Euclidean distances from each shot to sample mean.
    sample_mean_all = np.mean(iq)
    distances = np.abs(iq - sample_mean_all)
    sigma = float(np.std(distances))

    if sigma < 1e-15:
        # All points identical — effectively infinite SNR.
        return float("inf")

    return separation / sigma


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Dynamic Alpha Calculation (Kalman-style)
# ═══════════════════════════════════════════════════════════════════════════


def compute_alpha_static(n_shots: int, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Return a constant α array for Level-2 fusion.

    Args:
        n_shots: Number of shots.
        alpha:   Static weight in ``[0, 1]``.

    Returns:
        Array of shape ``(n_shots,)`` filled with *alpha*.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return np.full(n_shots, alpha, dtype=np.float64)


def compute_alpha_dynamic(
    hf_score: np.ndarray,
    min_alpha: float = DEFAULT_MIN_ALPHA,
    max_alpha: float = DEFAULT_MAX_ALPHA,
) -> np.ndarray:
    """Compute per-shot dynamic α based on HF confidence.

    When ``hf_score`` is near 0.0 or 1.0 the measurement is
    *confident* (unambiguous discrimination) so we trust HF more
    (higher α).  When ``hf_score`` is near 0.5 the measurement is
    *ambiguous* so we fall back to LF (lower α).

    .. math::

        \\text{confidence}[i] = 2 \\cdot |\\text{hf\\_score}[i] - 0.5|
        \\\\
        \\alpha[i] = \\alpha_{\\min} + \\text{confidence}[i] \\cdot
                     (\\alpha_{\\max} - \\alpha_{\\min})

    Args:
        hf_score:   Array of HF viability scores in [0, 1].
        min_alpha:  α floor when confidence is 0 (most ambiguous).
        max_alpha:  α ceiling when confidence is 1 (most certain).

    Returns:
        Per-shot α array of shape ``(n_shots,)``, bounded ``[min_alpha, max_alpha]``.

    Raises:
        ValueError: If alpha bounds are invalid.
    """
    if not 0.0 <= min_alpha <= max_alpha <= 1.0:
        raise ValueError(
            f"Require 0 <= min_alpha <= max_alpha <= 1, "
            f"got min_alpha={min_alpha}, max_alpha={max_alpha}"
        )

    # Confidence: 0.0 at hf_score=0.5 → 1.0 at hf_score ∈ {0, 1}
    confidence = 2.0 * np.abs(hf_score - 0.5)

    # Linear mapping confidence → [min_alpha, max_alpha]
    alpha = min_alpha + confidence * (max_alpha - min_alpha)

    return alpha


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Fusion
# ═══════════════════════════════════════════════════════════════════════════


def fuse_scores(
    hf_score: np.ndarray,
    lf_score: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Compute the fused Trajectory Viability Score.

    .. math::

        \\text{fusion}[i] = \\text{hf\\_score}[i] \\cdot \\alpha[i]
                           + \\text{lf\\_score}[i] \\cdot (1 - \\alpha[i])

    Both ``hf_score`` and ``lf_score`` are clamped to [0, 1] before
    fusion so the output is *guaranteed* to lie in [0, 1] (normalised).

    Args:
        hf_score: HF viability scores, shape ``(n_shots,)``.
        lf_score: LF drift scores, shape ``(n_shots,)``.
        alpha:    Per-shot fusion weights, shape ``(n_shots,)``.

    Returns:
        ``fusion_score`` array of shape ``(n_shots,)``, bounded ``[0, 1]``.
    """
    hf_clamped = np.clip(hf_score, 0.0, 1.0)
    lf_clamped = np.clip(lf_score, 0.0, 1.0)
    alpha_clamped = np.clip(alpha, 0.0, 1.0)

    fusion = hf_clamped * alpha_clamped + lf_clamped * (1.0 - alpha_clamped)
    return fusion


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Stage-1 Galton Outlier Rejection
# ═══════════════════════════════════════════════════════════════════════════


def galton_filter(
    fusion_scores: np.ndarray,
    drop_percentile: float = DEFAULT_DROP_PERCENTILE,
) -> np.ndarray:
    """Apply Stage-1 Galton percentile-based outlier rejection.

    Computes a dynamic threshold at the given percentile of the
    ``fusion_scores`` distribution and returns a boolean mask marking
    surviving trajectories.

    Args:
        fusion_scores:   Array of fused viability scores, shape ``(n_shots,)``.
        drop_percentile: Percentile for the rejection threshold (0–100).
                         Shots *below* this percentile are rejected.

    Returns:
        Boolean mask of shape ``(n_shots,)`` — ``True`` for surviving
        shots, ``False`` for rejected.

    Raises:
        ValueError: If *drop_percentile* is outside ``[0, 100]``.
    """
    if not 0.0 <= drop_percentile <= 100.0:
        raise ValueError(
            f"drop_percentile must be in [0, 100], got {drop_percentile}"
        )

    threshold = np.percentile(fusion_scores, drop_percentile)
    mask = fusion_scores >= threshold
    return mask


def adaptive_galton_schedule(
    depths: np.ndarray,
    *,
    base_percentile: float = 25.0,
    max_percentile: float = 75.0,
    depth_knee: float = 300.0,
    steepness: float = 3.0,
) -> np.ndarray:
    """Compute depth-adaptive Galton rejection percentiles.

    At shallow depths the fixed ``base_percentile`` suffices, but deeper
    circuits accumulate depolarisation noise so most shots thermalise.
    This helper returns a **per-depth rejection percentile** following a
    sigmoid schedule:

    .. math::

        p(d) = \\text{base} + (\\text{max} - \\text{base})
               \\cdot \\sigma\\bigl(\\text{steepness} \\cdot (d / \\text{knee} - 1)\\bigr)

    where :math:`\\sigma(x) = 1/(1+e^{-x})`.

    An optional oversampling factor can be derived from the schedule to
    compensate for the more aggressive rejection::

        oversample_factor = 100.0 / (100.0 - schedule)

    The default knee depth of **300** is chosen so that the transition
    from gentle to aggressive rejection occurs *well into the
    extrapolation regime* (training typically uses d ≤ 100), avoiding
    interference with the ML model's training distribution.

    Args:
        depths:          1-D array of integer circuit depths (must be ≥ 0).
        base_percentile: Rejection percentile for very shallow circuits
                         (default 25).
        max_percentile:  Asymptotic rejection percentile for very deep
                         circuits (default 75).
        depth_knee:      Circuit depth at which the schedule crosses the
                         midpoint of ``[base, max]`` (default 300).
        steepness:       Controls transition sharpness around the knee
                         (default 3.0; higher → sharper).

    Returns:
        1-D float array of per-depth rejection percentiles, same length
        as *depths*, each value in ``[base_percentile, max_percentile]``.

    Raises:
        ValueError: If any depth is negative, or
                    ``base_percentile >= max_percentile``, or either
                    percentile is outside ``[0, 100]``.

    Example::

        >>> import numpy as np
        >>> from qgate.tvs import adaptive_galton_schedule
        >>> depths = np.array([10, 50, 100, 500, 1000])
        >>> schedule = adaptive_galton_schedule(depths)
        >>> schedule  # doctest: +SKIP
        array([27.6, 28.8, 31.0, 69.0, 75.0])
        >>> oversample = 100.0 / (100.0 - schedule)
        >>> oversample  # doctest: +SKIP
        array([1.38, 1.40, 1.45, 3.23, 3.99])

    Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
    """
    depths = np.asarray(depths, dtype=np.float64)

    # ── Validation ────────────────────────────────────────────────────
    if depths.ndim == 0:
        depths = depths.reshape(1)
    if np.any(depths < 0):
        raise ValueError("All depths must be non-negative.")
    if not 0.0 <= base_percentile <= 100.0:
        raise ValueError(
            f"base_percentile must be in [0, 100], got {base_percentile}"
        )
    if not 0.0 <= max_percentile <= 100.0:
        raise ValueError(
            f"max_percentile must be in [0, 100], got {max_percentile}"
        )
    if base_percentile >= max_percentile:
        raise ValueError(
            f"base_percentile ({base_percentile}) must be less than "
            f"max_percentile ({max_percentile})"
        )

    # ── Sigmoid schedule ──────────────────────────────────────────────
    x = steepness * (depths / depth_knee - 1.0)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    schedule = base_percentile + (max_percentile - base_percentile) * sigmoid

    return schedule


# ═══════════════════════════════════════════════════════════════════════════
# Top-level batch processor
# ═══════════════════════════════════════════════════════════════════════════


def process_telemetry_batch(
    hf_data: np.ndarray,
    lf_scores: np.ndarray,
    mode: Optional[str] = None,
    *,
    force_mode: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    zero_centroid: complex = 0.0 + 0.0j,
    variance: float = DEFAULT_VARIANCE,
    min_alpha: float = DEFAULT_MIN_ALPHA,
    max_alpha: float = DEFAULT_MAX_ALPHA,
    drop_percentile: float = DEFAULT_DROP_PERCENTILE,
    k_clusters: int = DEFAULT_K_CLUSTERS,
    one_centroid: Optional[complex] = None,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    hybrid_fallback: str = DEFAULT_HYBRID_FALLBACK,
) -> Dict[str, Any]:
    """Process a full telemetry batch: normalise → fuse → filter.

    This is the main entry point for the TVS pipeline.  It accepts raw
    HF data (binary or complex I/Q), LF drift scores, and calibration
    parameters, then returns the surviving-shot mask, fusion scores,
    and ML feature matrix for Stage-2 processing.

    **Autonomous routing (default):**  When neither ``force_mode`` nor
    the legacy ``mode`` parameter is specified, the pipeline is chosen
    automatically via ``_determine_optimal_pipeline``:

    * Integer/bool HF data → ``level_2``
    * Complex HF data, SNR ≥ ``snr_threshold`` → ``level_1``
    * Complex HF data, SNR < ``snr_threshold`` → ``level_1_cluster``

    **Forced mode:**  Pass ``force_mode`` to override auto-routing for
    benchmarking or debugging.

    .. deprecated:: 0.7.0
       The positional ``mode`` parameter is deprecated.  Use the
       keyword-only ``force_mode`` parameter instead.  The legacy
       ``mode='hybrid'`` is equivalent to omitting both (auto-routing).

    The function is **fully vectorised** — no Python ``for`` loops —
    and handles tens of thousands of shots with sub-millisecond latency
    on typical hardware.

    Args:
        hf_data:
            Raw HF telemetry array, shape ``(n_shots,)``.

            * Integer/bool dtype → auto-routes to Level-2.
            * Complex dtype → auto-routes to Level-1 or Level-1-cluster
              based on I/Q separability.

        lf_scores:
            LF drift-tracking scores, shape ``(n_shots,)``, ideally in
            ``[0, 1]``.  Values outside this range are clamped with a
            warning.

        mode:
            **Deprecated.**  Legacy positional parameter.  Use
            ``force_mode`` instead.  If both are provided, ``force_mode``
            takes precedence.  ``mode='hybrid'`` is mapped to auto-routing.

        force_mode:
            Override the autonomous router.  One of:

            * ``None`` — autonomous selection (default).
            * ``'level_2'`` — binary hard-decision readout.
            * ``'level_1'`` — single-centroid RBF soft-decision.
            * ``'level_1_cluster'`` — multi-centroid K-Means + RBF.

        alpha:
            Static fusion weight for Level-2 mode (ignored in Level-1).
            Default: 0.5.

        zero_centroid:
            Calibration centroid for the |0⟩ state in the I/Q plane.
            Used in Level-1 and as reference in SNR computation.
            Default: ``0+0j``.

        variance:
            Gaussian RBF bandwidth σ² for Level-1 I/Q → probability
            conversion.  Must be > 0.  Default: 0.15.

        min_alpha:
            Dynamic-α floor for Level-1 confidence-based fusion.
            Default: 0.3.

        max_alpha:
            Dynamic-α ceiling for Level-1 confidence-based fusion.
            Default: 0.9.

        drop_percentile:
            Galton rejection percentile (0–100).  Shots below this
            percentile of the fusion-score distribution are discarded.
            Default: 25.0 (reject bottom 25%).

        k_clusters:
            Number of MiniBatchKMeans clusters for ``level_1_cluster``
            mode.  Must be ≥ 2.  Default: 8.

        one_centroid:
            Calibrated |1⟩ centroid for SNR computation in auto-routing.
            Optional — if ``None``, SNR uses the sample mean.

        snr_threshold:
            SNR threshold for auto-routing.  If SNR ≥ threshold,
            uses ``level_1``; otherwise falls back to ``level_1_cluster``.
            Default: 3.0.

        hybrid_fallback:
            **Deprecated.**  Previously used for hybrid-mode fallback.
            Retained for backward compatibility but has no effect when
            using the new auto-router.  Default: ``'level_1_cluster'``.

    Returns:
        Dict with keys:

        ``surviving_mask``
            Boolean array of shape ``(n_shots,)`` — ``True`` for shots
            that passed Stage-1 filtering.

        ``fusion_scores``
            Float array of shape ``(n_shots,)`` — per-shot fused TVS
            values in ``[0, 1]``.

        ``ml_features``
            Feature array for Stage-2 ML.

            * Level-1 / Level-1-cluster: complex I/Q coordinates of
              surviving shots (shape ``(n_surviving,)``).
            * Level-2: binary bits of surviving shots
              (shape ``(n_surviving,)``).

        ``hf_scores``
            Float array of normalised HF scores, shape ``(n_shots,)``.

        ``alpha_values``
            Per-shot α values used in fusion, shape ``(n_shots,)``.

        ``threshold``
            The Galton percentile threshold value used for filtering.

        ``n_shots``
            Total number of input shots.

        ``n_surviving``
            Number of shots that passed the filter.

        ``mode``
            The pipeline that was used.  For auto-routing this shows
            the resolved sub-mode (e.g. ``'auto→level_1'``).

        ``snr``  *(present when auto-routing or force_mode is complex)*
            Computed I/Q cloud SNR that determined routing.

    Raises:
        ValueError: On invalid mode, empty arrays, or mismatched lengths.
        TypeError:  On non-ndarray inputs or invalid centroid type.

    Examples::

        # Auto-routing (recommended — pipeline chosen automatically):
        import numpy as np
        from qgate.tvs import process_telemetry_batch

        rng = np.random.default_rng(42)
        iq = rng.normal(0.9, 0.1, 1000) + 1j * rng.normal(0.02, 0.05, 1000)
        lf = rng.uniform(0.5, 1.0, 1000)
        res = process_telemetry_batch(
            iq, lf,
            zero_centroid=0.95 + 0.02j,
            variance=0.15,
        )
        print(f"Auto-routed to: {res['mode']}, SNR: {res['snr']:.2f}")

        # Force Level-2 (binary readout):
        hf = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        lf = rng.uniform(0.4, 1.0, size=10)
        res = process_telemetry_batch(hf, lf, force_mode="level_2", alpha=0.6)
        print(res["n_surviving"], "shots survived")

        # Force Level-1-cluster (K-Means + RBF, for benchmarking):
        res = process_telemetry_batch(
            iq, lf,
            force_mode="level_1_cluster",
            k_clusters=8,
            variance=0.15,
        )
    """
    import warnings

    # ── Resolve force_mode vs legacy mode ─────────────────────────────
    effective_mode: Optional[str] = force_mode

    if mode is not None:
        # Legacy positional `mode` was provided.
        warnings.warn(
            "The positional 'mode' parameter is deprecated since v0.7.0. "
            "Use the keyword-only 'force_mode' parameter instead. "
            "Auto-routing (force_mode=None) is now the default.",
            DeprecationWarning,
            stacklevel=2,
        )
        if effective_mode is None:
            # Legacy mode is the only signal — translate it.
            if mode == "hybrid":
                # hybrid → auto-routing (None)
                effective_mode = None
            else:
                _validate_mode(mode)
                effective_mode = mode

    # Validate force_mode if provided.
    _validate_force_mode(effective_mode)

    # ── Input validation ──────────────────────────────────────────────
    _validate_arrays(hf_data, lf_scores)
    _validate_lf_bounds(lf_scores)

    n_shots = hf_data.shape[0]
    lf = np.asarray(lf_scores, dtype=np.float64)

    # ── Pipeline routing ─────────────────────────────────────────────
    snr_value: Optional[float] = None

    if effective_mode is not None:
        # Forced / legacy mode — skip auto-routing.
        resolved_mode = effective_mode
        logger.info("Pipeline forced to %s", resolved_mode)
    else:
        # Autonomous routing.
        resolved_mode, snr_value = _determine_optimal_pipeline(
            hf_data, zero_centroid, one_centroid, snr_threshold,
        )

    # ── Step 1: HF normalisation ─────────────────────────────────────
    if resolved_mode == "level_2":
        hf_score = normalise_hf_level2(hf_data)
    elif resolved_mode == "level_1":
        hf_score = normalise_hf_level1(hf_data, zero_centroid, variance)
    elif resolved_mode == "level_1_cluster":
        # Clamp k_clusters to n_shots if needed for safety.
        k = min(k_clusters, n_shots) if n_shots >= 2 else 2
        hf_score = normalise_hf_level1_cluster(hf_data, k_clusters=k, variance=variance)
    else:
        raise ValueError(f"Unhandled resolved mode: {resolved_mode!r}")

    logger.debug(
        "HF normalisation (%s): min=%.4f, max=%.4f, mean=%.4f",
        resolved_mode, float(np.min(hf_score)), float(np.max(hf_score)),
        float(np.mean(hf_score)),
    )

    # ── Step 2: alpha calculation ────────────────────────────────────
    if resolved_mode == "level_2":
        alpha_arr = compute_alpha_static(n_shots, alpha)
    else:
        # Level-1, level_1_cluster — all use dynamic α based on HF confidence.
        alpha_arr = compute_alpha_dynamic(hf_score, min_alpha, max_alpha)

    # ── Step 3: fusion ───────────────────────────────────────────────
    fusion = fuse_scores(hf_score, lf, alpha_arr)

    logger.debug(
        "Fusion scores: min=%.4f, max=%.4f, mean=%.4f",
        float(np.min(fusion)), float(np.max(fusion)),
        float(np.mean(fusion)),
    )

    # ── Step 4: Galton outlier rejection ─────────────────────────────
    mask = galton_filter(fusion, drop_percentile)
    threshold = float(np.percentile(fusion, drop_percentile))
    n_surviving = int(np.sum(mask))

    logger.info(
        "Galton filter (p=%.0f%%): threshold=%.4f, surviving=%d/%d (%.1f%%)",
        drop_percentile, threshold, n_surviving, n_shots,
        100.0 * n_surviving / max(n_shots, 1),
    )

    # ── ML features for Stage 2 ─────────────────────────────────────
    ml_features = hf_data[mask]

    # ── Build return dict ────────────────────────────────────────────
    if effective_mode is not None:
        mode_label = resolved_mode
    else:
        mode_label = f"auto→{resolved_mode}"

    result: Dict[str, Any] = {
        "surviving_mask": mask,
        "fusion_scores": fusion,
        "ml_features": ml_features,
        "hf_scores": hf_score,
        "alpha_values": alpha_arr,
        "threshold": threshold,
        "n_shots": n_shots,
        "n_surviving": n_surviving,
        "mode": mode_label,
    }

    if snr_value is not None:
        result["snr"] = snr_value

    return result
