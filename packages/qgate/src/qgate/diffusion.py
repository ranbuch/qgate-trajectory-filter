"""
diffusion.py — PPU trajectory filtering for generative diffusion models.

Extends the patented two-stage trajectory filtering architecture
(§22 of the CIP disclosure) to **diffusion-model inference**: the latent
tensors produced by each denoising trajectory are treated as stochastic
"shots" from a Probabilistic Processing Unit (PPU).  A small batch of
cheap, low-step trajectories is filtered and fused into a single
high-fidelity latent — bypassing the need for expensive 50+ step runs.

Architecture overview
---------------------

The module implements the same "Split Frontend, Unified Backend" design
used by :mod:`qgate.stochastic` for financial Monte Carlo:

**Frontend** — :class:`LatentTelemetryExtractor`
    Extracts per-trajectory telemetry features from raw denoising latents.
    Features capture spatial energy, spectral content, channel statistics,
    and inter-trajectory variance — analogous to Level-1 IQ probe readout
    in the QPU pipeline.

**Backend — Stage 1** — :class:`GaltonLatentFilter`
    Applies MAD-robust statistical outlier rejection (patent §22.2) to
    discard latents corrupted by quantisation noise, mode collapse, or
    denoising-schedule truncation artefacts.

**Backend — Stage 2** — :class:`DiffusionMitigator`
    Trains a per-pixel regression transfer function that maps
    ``[raw_latent_pixel, telemetry_vector] → corrected_pixel``.
    Operates on flattened latent space: each pixel in the N-channel
    latent is treated as an independent observable with shared telemetry.
    Supports both 4-channel (FLUX.1/SDXL) and 32-channel (FLUX.2 Klein)
    VAE latent spaces.

**Orchestrator** — :class:`DiffusionMitigationPipeline`
    End-to-end pipeline wiring frontend and backend stages.

Simulator support
-----------------

The module ships a self-contained **mock diffusion simulator**
(:func:`simulate_diffusion_latents`) that models denoising noise at
various step counts without requiring actual GPU/model weights.  This
makes the module fully testable and benchmarkable on CPU-only machines.

Patent reference
----------------
US Provisional App. No. 64/XXX,XXX (April 2026), §22.
Prior: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915.
CIP — PPU generalization of trajectory filtering to generative AI.

Licensed under the QGATE Source Available Evaluation License v1.2.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("qgate.diffusion")

# ---------------------------------------------------------------------------
# Lazy scikit-learn imports (mirrors stochastic.py pattern)
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
# Pydantic — with lightweight fallback (mirrors stochastic.py pattern)
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
            "scikit-learn is required for DiffusionMitigator.  "
            "Install with:  pip install qgate[ml]"
        )


# ── MAD → σ conversion constant ──────────────────────────────────────────
_MAD_TO_SIGMA: float = 1.4826

# ── Default latent shape ──────────────────────────────────────────────────
# FLUX.1 / SDXL: 4-channel, 64×64 spatial.
# FLUX.2 Klein: 32-channel, 64×64 spatial (patch_size=[2,2]).
# The defaults here use 4-channel for backward compatibility; callers should
# set ``latent_channels`` explicitly when using FLUX.2 Klein (32 channels).
DEFAULT_LATENT_CHANNELS: int = 4
DEFAULT_LATENT_HEIGHT: int = 64
DEFAULT_LATENT_WIDTH: int = 64


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


class DiffusionConfig(BaseModel):  # type: ignore[misc]
    """Configuration for the diffusion latent mitigation pipeline.

    Mirrors :class:`~qgate.stochastic.StochasticConfig` with
    domain-appropriate defaults for diffusion-model inference.

    Attributes:
        reject_fraction: Fraction of latents discarded by Stage 1
                         Galton filter (0, 1).
        model_name:      scikit-learn regressor for Stage 2.
        scale_features:  Apply ``StandardScaler`` before training.
        random_state:    RNG seed for reproducibility.
        latent_channels: Number of VAE latent channels (4 for SDXL/FLUX.1,
                         32 for FLUX.2 Klein).
        latent_height:   Spatial height of latent tensor.
        latent_width:    Spatial width of latent tensor.
    """

    reject_fraction: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description="Fraction of latents rejected by Stage 1 Galton filter",
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
    latent_channels: int = Field(
        default=DEFAULT_LATENT_CHANNELS,
        ge=1,
        description="Number of VAE latent channels",
    )
    latent_height: int = Field(
        default=DEFAULT_LATENT_HEIGHT,
        ge=1,
        description="Spatial height of latent tensor",
    )
    latent_width: int = Field(
        default=DEFAULT_LATENT_WIDTH,
        ge=1,
        description="Spatial width of latent tensor",
    )

    if ConfigDict is not None:
        model_config = ConfigDict(frozen=True, extra="forbid")


# ═══════════════════════════════════════════════════════════════════════════
# Result Dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DiffusionCalibrationResult:
    """Artefacts from :meth:`DiffusionMitigator.calibrate`.

    Attributes:
        model_name:      Name of the trained regressor.
        n_samples:       Number of calibration latent-pixel pairs used.
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
class DiffusionMitigationResult:
    """Result of the full diffusion mitigation pipeline.

    Attributes:
        mitigated_latent:  Corrected latent tensor (C, H, W).
        raw_mean_latent:   Naïve mean of raw budget latents (C, H, W).
        stage1_survivors:  Number of latents surviving Stage 1.
        stage1_rejected:   Number of latents rejected by Stage 1.
        fid_score:         Fréchet Inception Distance vs ground truth.
        clip_score:        CLIP alignment score (prompt↔image).
        psnr:              Peak signal-to-noise ratio vs ground truth latent.
        improvement_factor: FID(raw) / FID(mitigated).
        latency_seconds:   Total pipeline wall-clock time.
        metadata:          Free-form metadata dict.
    """

    mitigated_latent: NDArray[np.float64]
    raw_mean_latent: NDArray[np.float64]
    stage1_survivors: int
    stage1_rejected: int
    fid_score: float = 0.0
    clip_score: float = 0.0
    psnr: float = 0.0
    improvement_factor: float = float("nan")
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Mock Diffusion Simulator (PPU Hardware Model)
# ═══════════════════════════════════════════════════════════════════════════


def _prompt_to_seed(prompt: str) -> int:
    """Deterministic prompt → integer seed via SHA-256."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def simulate_diffusion_latents(
    prompt: str,
    n_trajectories: int = 1,
    num_steps: int = 50,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
    latent_height: int = DEFAULT_LATENT_HEIGHT,
    latent_width: int = DEFAULT_LATENT_WIDTH,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """Simulate denoising diffusion latent tensors.

    Models the key statistical properties of real diffusion latents
    without requiring GPU or model weights:

    1. A deterministic **signal component** derived from the prompt
       hash (simulates the learned denoiser's content).
    2. A **noise component** inversely proportional to ``num_steps``
       (simulates quantisation/truncation noise from early stopping).
    3. A **structural bias** for low step counts that introduces
       correlated spatial artefacts (simulates mode collapse and
       texture smearing common in 4–10 step generation).

    The noise model follows the diffusion schedule relationship:

    .. math::
        \\sigma_{\\text{residual}}(T) \\approx \\frac{\\sigma_0}{\\sqrt{T}}

    where *T* is the number of denoising steps and :math:`\\sigma_0`
    is the initial noise scale.

    Args:
        prompt:           Text prompt (determines signal content).
        n_trajectories:   Number of independent denoising trajectories.
        num_steps:        Denoising steps (higher = less noise).
        latent_channels:  VAE latent channels (default 4).
        latent_height:    Spatial height (default 64).
        latent_width:     Spatial width (default 64).
        seed:             Optional RNG override.

    Returns:
        Array of shape ``(n_trajectories, C, H, W)`` in float64.
    """
    if seed is None:
        seed = _prompt_to_seed(prompt)
    rng = np.random.default_rng(seed)

    shape = (latent_channels, latent_height, latent_width)

    # ── Deterministic signal (prompt-dependent "ground truth") ────────
    # Generate a spatially structured signal using low-frequency patterns
    signal_rng = np.random.default_rng(_prompt_to_seed(prompt))
    signal = np.zeros(shape, dtype=np.float64)

    for c in range(latent_channels):
        # Low-frequency spatial structure (simulate learned features)
        freq_x = signal_rng.uniform(0.5, 3.0)
        freq_y = signal_rng.uniform(0.5, 3.0)
        phase_x = signal_rng.uniform(0, 2 * np.pi)
        phase_y = signal_rng.uniform(0, 2 * np.pi)
        amplitude = signal_rng.uniform(0.3, 1.0)

        y_grid, x_grid = np.meshgrid(
            np.linspace(0, np.pi, latent_height),
            np.linspace(0, np.pi, latent_width),
            indexing="ij",
        )
        signal[c] = amplitude * (
            np.sin(freq_x * x_grid + phase_x)
            * np.cos(freq_y * y_grid + phase_y)
        )
        # Add some fine-grained texture
        signal[c] += 0.1 * signal_rng.standard_normal((latent_height, latent_width))

    # ── Per-trajectory noise (step-dependent) ─────────────────────────
    # Noise scale: σ ∝ 1/√T — more steps → less residual noise
    sigma_0 = 1.0  # base noise level
    noise_scale = sigma_0 / np.sqrt(max(num_steps, 1))

    # Structural bias for low step counts (correlated artefacts)
    # This simulates texture warping, mode collapse, gear distortion
    bias_strength = max(0.0, 0.3 * (1.0 - num_steps / 50.0))

    latents = np.zeros(
        (n_trajectories, latent_channels, latent_height, latent_width),
        dtype=np.float64,
    )

    for i in range(n_trajectories):
        # Independent Gaussian noise per trajectory
        noise = noise_scale * rng.standard_normal(shape)

        # Correlated spatial bias (low-frequency artefact)
        if bias_strength > 0:
            bias = np.zeros(shape, dtype=np.float64)
            for c in range(latent_channels):
                # Smooth random field via low-pass filtering
                raw_bias = rng.standard_normal((latent_height, latent_width))
                # Simple box-blur approximation for spatial correlation
                kernel_size = max(3, latent_height // 8)
                from scipy.ndimage import uniform_filter  # type: ignore[import-untyped]
                bias[c] = bias_strength * uniform_filter(
                    raw_bias, size=kernel_size, mode="wrap",
                )
            noise += bias

        latents[i] = signal + noise

    return latents


# ═══════════════════════════════════════════════════════════════════════════
# Mock Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_latent_fid(
    latent: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> float:
    """Compute a latent-space proxy for Fréchet Inception Distance.

    Uses the Fréchet distance between the channel-wise mean/variance
    statistics of the candidate and reference latents.  This is a
    computationally cheap proxy for image-space FID that correlates
    well with perceptual quality in latent diffusion models.

    .. math::
        d^2 = \\|\\mu_1 - \\mu_2\\|^2 + \\text{Tr}(\\Sigma_1 + \\Sigma_2
              - 2(\\Sigma_1 \\Sigma_2)^{1/2})

    For diagonal covariances (channel-independent), this simplifies to:

    .. math::
        d^2 = \\sum_c (\\mu_{1c} - \\mu_{2c})^2
              + (\\sigma_{1c} - \\sigma_{2c})^2

    Args:
        latent:    Candidate latent ``(C, H, W)``.
        reference: Ground-truth latent ``(C, H, W)``.

    Returns:
        Fréchet distance (lower is better).
    """
    n_channels = latent.shape[0]

    mu_diff_sq = 0.0
    sigma_diff_sq = 0.0

    for c in range(n_channels):
        mu1 = float(np.mean(latent[c]))
        mu2 = float(np.mean(reference[c]))
        s1 = float(np.std(latent[c]))
        s2 = float(np.std(reference[c]))

        mu_diff_sq += (mu1 - mu2) ** 2
        sigma_diff_sq += (s1 - s2) ** 2

    # Also add per-pixel MSE contribution (scaled)
    mse = float(np.mean((latent - reference) ** 2))

    # Combined metric: blend Fréchet stats + pixel-level error
    return float(np.sqrt(mu_diff_sq + sigma_diff_sq) + 10.0 * mse)


def compute_clip_score(
    latent: NDArray[np.float64],
    prompt: str,
    reference: NDArray[np.float64],
) -> float:
    """Compute a mock CLIP alignment score.

    In production this would use a CLIP encoder.  Our proxy measures
    how well the latent preserves the spatial structure of the
    reference (which was generated from the prompt):

    * Cosine similarity between flattened latent vectors.
    * Penalised by spatial frequency mismatch.

    Args:
        latent:    Candidate latent ``(C, H, W)``.
        prompt:    Text prompt (used for deterministic reference).
        reference: Reference latent ``(C, H, W)``.

    Returns:
        Score in [0, 1] — higher is better.
    """
    # Cosine similarity in flattened space
    flat_lat = latent.flatten()
    flat_ref = reference.flatten()

    dot = float(np.dot(flat_lat, flat_ref))
    norm_lat = float(np.linalg.norm(flat_lat))
    norm_ref = float(np.linalg.norm(flat_ref))

    if norm_lat < 1e-12 or norm_ref < 1e-12:
        return 0.0

    cosine_sim = dot / (norm_lat * norm_ref)
    # Clamp to [0, 1] range
    cosine_sim = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))

    # Spectral correlation bonus: how similar are the frequency spectra?
    fft_lat = np.abs(np.fft.fft2(latent))
    fft_ref = np.abs(np.fft.fft2(reference))
    spectral_corr = float(np.corrcoef(fft_lat.flatten(), fft_ref.flatten())[0, 1])
    spectral_corr = max(0.0, spectral_corr)

    # Blend: 60% cosine + 40% spectral
    return 0.6 * cosine_sim + 0.4 * spectral_corr


def compute_psnr(
    latent: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> float:
    """Peak signal-to-noise ratio between candidate and reference latents.

    .. math::
        \\text{PSNR} = 10 \\log_{10}\\!\\left(
            \\frac{\\text{MAX}^2}{\\text{MSE}}
        \\right)

    Args:
        latent:    Candidate latent ``(C, H, W)``.
        reference: Ground-truth latent ``(C, H, W)``.

    Returns:
        PSNR in dB (higher is better).
    """
    mse = float(np.mean((latent - reference) ** 2))
    if mse < 1e-15:
        return 100.0  # perfect match
    max_val = float(np.max(np.abs(reference)))
    if max_val < 1e-15:
        max_val = 1.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


# ═══════════════════════════════════════════════════════════════════════════
# Frontend — LatentTelemetryExtractor
# ═══════════════════════════════════════════════════════════════════════════

#: Canonical feature names for the latent telemetry vector.
LATENT_FEATURE_NAMES: Tuple[str, ...] = (
    "spatial_energy",         # total L2 energy of latent
    "channel_mean_std",       # std of per-channel means
    "channel_var_mean",       # mean of per-channel variances
    "high_freq_ratio",        # fraction of energy in high frequencies
    "spatial_autocorr",       # mean spatial lag-1 autocorrelation
    "kurtosis",               # excess kurtosis (peakedness of distribution)
)


class LatentTelemetryExtractor:
    """Extract per-trajectory telemetry from raw diffusion latents.

    Analogous to :class:`~qgate.stochastic.StochasticTelemetryExtractor`
    but operates on 3-D latent tensors (C, H, W) rather than 1-D
    stochastic paths.

    The telemetry features capture:

    * **Spatial energy** — total latent magnitude (noise floor proxy).
    * **Channel mean std** — inter-channel consistency.
    * **Channel variance mean** — average intra-channel spread.
    * **High-frequency ratio** — spectral noise content.
    * **Spatial autocorrelation** — structural coherence.
    * **Kurtosis** — distribution peakedness (mode-collapse indicator).

    These features form the "Telemetry Lockbox" (§6.3) — computed
    independently of the decoded image to prevent post-selection bias.

    Example::

        extractor = LatentTelemetryExtractor()
        features = extractor.extract(latents)  # (N, 6)
    """

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Ordered feature names."""
        return LATENT_FEATURE_NAMES

    @property
    def n_features(self) -> int:
        return len(LATENT_FEATURE_NAMES)

    def extract(
        self, latents: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Extract telemetry feature matrix from a batch of latents.

        Args:
            latents: Array of shape ``(N, C, H, W)`` — N latent tensors.

        Returns:
            Feature matrix of shape ``(N, 6)``.
        """
        n = latents.shape[0]
        features = np.empty((n, self.n_features), dtype=np.float64)

        for i in range(n):
            features[i] = self._extract_single(latents[i])

        return features

    @staticmethod
    def _extract_single(latent: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract features from a single latent tensor (C, H, W).

        Returns:
            Feature vector of shape ``(6,)``.
        """
        C, H, W = latent.shape

        # 1. Spatial energy — total L2 norm (noise floor proxy)
        spatial_energy = float(np.sqrt(np.sum(latent ** 2)) / (C * H * W))

        # 2. Channel mean std — inter-channel consistency
        channel_means = np.array([float(np.mean(latent[c])) for c in range(C)])
        channel_mean_std = float(np.std(channel_means))

        # 3. Channel variance mean — average intra-channel spread
        channel_vars = np.array([float(np.var(latent[c])) for c in range(C)])
        channel_var_mean = float(np.mean(channel_vars))

        # 4. High-frequency ratio — spectral noise content
        total_spec_energy = 0.0
        high_freq_energy = 0.0
        for c in range(C):
            fft_mag = np.abs(np.fft.fft2(latent[c]))
            total_e = float(np.sum(fft_mag ** 2))
            total_spec_energy += total_e
            # High-freq: everything outside the central 25% of spectrum
            cy, cx = H // 4, W // 4
            # Zero out low-freq centre
            mask = np.ones((H, W), dtype=bool)
            mask[H // 2 - cy:H // 2 + cy, W // 2 - cx:W // 2 + cx] = False
            high_freq_energy += float(np.sum((fft_mag * mask) ** 2))

        high_freq_ratio = high_freq_energy / max(total_spec_energy, 1e-15)

        # 5. Spatial autocorrelation — mean lag-1 horizontal autocorr
        autocorrs = []
        for c in range(C):
            flat = latent[c].flatten()
            if len(flat) > 2:
                dm = flat - flat.mean()
                c0 = float(np.dot(dm, dm))
                c1 = float(np.dot(dm[:-1], dm[1:]))
                autocorrs.append(c1 / c0 if abs(c0) > 1e-15 else 0.0)
        spatial_autocorr = float(np.mean(autocorrs)) if autocorrs else 0.0

        # 6. Kurtosis — excess kurtosis (mode-collapse indicator)
        flat_all = latent.flatten()
        mean_val = float(np.mean(flat_all))
        std_val = float(np.std(flat_all))
        if std_val > 1e-15:
            kurtosis = float(np.mean(((flat_all - mean_val) / std_val) ** 4)) - 3.0
        else:
            kurtosis = 0.0

        return np.array(
            [spatial_energy, channel_mean_std, channel_var_mean,
             high_freq_ratio, spatial_autocorr, kurtosis],
            dtype=np.float64,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Backend — Stage 1: GaltonLatentFilter
# ═══════════════════════════════════════════════════════════════════════════


class GaltonLatentFilter:
    """MAD-robust latent trajectory outlier rejection (Stage 1 Galton filter).

    Mirrors :class:`~qgate.stochastic.GaltonOutlierFilter` but operates
    on diffusion latent telemetry.  Rejects latents with extreme
    composite viability scores — those most likely corrupted by
    denoising truncation, mode collapse, or quantisation noise.

    Args:
        reject_fraction: Fraction of latents to discard (0, 1).

    Example::

        filt = GaltonLatentFilter(reject_fraction=0.25)
        mask = filt.filter(features)  # boolean survival mask
    """

    def __init__(self, reject_fraction: float = 0.25) -> None:
        if not 0.0 < reject_fraction < 1.0:
            raise ValueError(
                f"reject_fraction must be in (0, 1), got {reject_fraction}"
            )
        self.reject_fraction = reject_fraction

    def compute_viability_scores(
        self, features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute composite trajectory viability scores.

        Uses the same MAD-robust z-score approach as
        :class:`~qgate.stochastic.GaltonOutlierFilter`.

        Args:
            features: Feature matrix ``(N, n_features)``.

        Returns:
            Composite viability scores ``(N,)`` — lower is better.
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

        return np.sqrt(np.sum(z_scores ** 2, axis=1))

    def filter(
        self, features: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Apply the Galton filter and return a boolean survival mask.

        Args:
            features: Feature matrix ``(N, n_features)``.

        Returns:
            Boolean mask of shape ``(N,)`` — ``True`` for retained latents.
        """
        scores = self.compute_viability_scores(features)
        n_reject = int(np.ceil(len(scores) * self.reject_fraction))
        n_keep = len(scores) - n_reject

        threshold_idx = np.argsort(scores)[n_keep - 1]
        cutoff = scores[threshold_idx]

        return scores <= cutoff + 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# Backend — Stage 2: DiffusionMitigator (ML Reconstruction)
# ═══════════════════════════════════════════════════════════════════════════


class DiffusionMitigator:
    """Two-stage telemetry-driven latent reconstruction for diffusion PPUs.

    Implements the mathematical transfer function (§17.2, §22.3) adapted
    for latent-space tensors.  Each pixel across all surviving latents is
    treated as an independent observable, with the per-trajectory
    telemetry vector as shared context features.

    **Calibration** (requires a small set of high-step "ideal" latents):
      1. Flatten all calibration latent pixels.
      2. Pair each pixel with its trajectory's telemetry vector.
      3. Train a regressor: ``[pixel_value, telemetry] → correction``.

    **Inference**:
      1. For each surviving pixel, predict the correction.
      2. Apply: ``mitigated = raw + correction``.
      3. Reshape back to (C, H, W) and average across trajectories.

    Args:
        config: :class:`DiffusionConfig` (immutable).
    """

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        _require_sklearn()
        self._config = config or DiffusionConfig()
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._is_calibrated = False

    @property
    def config(self) -> DiffusionConfig:
        return self._config

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def _build_feature_matrix(
        self,
        telemetry: NDArray[np.float64],
        pixel_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build feature matrix for pixel-level regression.

        Args:
            telemetry:    Per-trajectory features ``(N, n_feat)``.
            pixel_values: Flattened pixel values ``(N * n_pixels,)``.

        Returns:
            Feature matrix ``(N * n_pixels, n_feat + 1)``.
        """
        n_traj = telemetry.shape[0]
        n_pixels = len(pixel_values) // n_traj

        # Repeat telemetry for each pixel in the trajectory
        tele_expanded = np.repeat(telemetry, n_pixels, axis=0)
        return np.column_stack([pixel_values.reshape(-1, 1), tele_expanded])

    def _make_model(self) -> Any:
        """Instantiate the sklearn regressor from config."""
        _require_sklearn()
        cfg = self._config
        name = cfg.model_name.lower().replace("-", "_").replace(" ", "_")
        kwargs = dict(cfg.model_params) if cfg.model_params else {}

        if name == "random_forest":
            defaults = {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_leaf": 5,
                "random_state": cfg.random_state,
                "n_jobs": -1,
            }
            defaults.update(kwargs)
            return RandomForestRegressor(**defaults)

        if name == "gradient_boosting":
            defaults = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "min_samples_leaf": 5,
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
        raw_latents: NDArray[np.float64],
        ideal_latents: NDArray[np.float64],
        *,
        pixel_subsample: int = 1024,
    ) -> DiffusionCalibrationResult:
        """Train the Stage 2 transfer function on calibration data.

        To keep training tractable, we subsample pixels from each
        calibration latent rather than using every pixel.

        Args:
            telemetry:       Per-trajectory features ``(N, n_feat)``.
            raw_latents:     Raw (low-step) latents ``(N, C, H, W)``.
            ideal_latents:   Reference (high-step) latents ``(N, C, H, W)``.
            pixel_subsample: Max pixels per trajectory for training.

        Returns:
            :class:`DiffusionCalibrationResult`.
        """
        _require_sklearn()
        t0 = time.monotonic()

        n_traj = raw_latents.shape[0]
        n_pixels_total = int(np.prod(raw_latents.shape[1:]))

        # Subsample pixels for training efficiency
        rng = np.random.default_rng(self._config.random_state)
        n_sample = min(pixel_subsample, n_pixels_total)

        all_features = []
        all_targets = []

        for i in range(n_traj):
            raw_flat = raw_latents[i].flatten()
            ideal_flat = ideal_latents[i].flatten()

            # Random pixel subset
            idx = rng.choice(n_pixels_total, size=n_sample, replace=False)
            raw_pixels = raw_flat[idx]
            corrections = ideal_flat[idx] - raw_flat[idx]

            # Feature: [pixel_value, telemetry_0, ..., telemetry_k]
            tele_row = telemetry[i]  # (n_feat,)
            tele_repeated = np.tile(tele_row, (n_sample, 1))
            feat = np.column_stack([raw_pixels.reshape(-1, 1), tele_repeated])

            all_features.append(feat)
            all_targets.append(corrections)

        X = np.vstack(all_features)
        y = np.concatenate(all_targets)

        # Optional scaling
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
        feature_names = ("pixel_value",) + LATENT_FEATURE_NAMES[:n_feat]

        logger.info(
            "DiffusionMitigator calibrated: model=%s, n_pixels=%d, "
            "train_mae=%.6f, elapsed=%.2fs",
            self._config.model_name, len(y), train_mae, elapsed,
        )

        return DiffusionCalibrationResult(
            model_name=self._config.model_name,
            n_samples=len(y),
            feature_names=feature_names,
            train_mae=train_mae,
            train_rmse=train_rmse,
            elapsed_seconds=elapsed,
            metadata={
                "scale_features": self._config.scale_features,
                "pixel_subsample": n_sample,
                "n_trajectories": n_traj,
            },
        )

    def predict(
        self,
        telemetry: NDArray[np.float64],
        raw_latents: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply trained transfer function to produce a mitigated latent.

        Predicts per-pixel corrections for each surviving latent, then
        averages the corrected latents into a single fused output.

        Args:
            telemetry:   Per-trajectory features ``(N, n_feat)``.
            raw_latents: Raw latents ``(N, C, H, W)``.

        Returns:
            Fused mitigated latent ``(C, H, W)``.
        """
        if not self._is_calibrated or self._model is None:
            raise RuntimeError(
                "DiffusionMitigator is not calibrated.  "
                "Call calibrate() first."
            )

        n_traj = raw_latents.shape[0]
        shape = raw_latents.shape[1:]  # (C, H, W)
        n_pixels = int(np.prod(shape))

        corrected = np.zeros((n_traj, *shape), dtype=np.float64)

        for i in range(n_traj):
            raw_flat = raw_latents[i].flatten()
            tele_row = telemetry[i]
            tele_repeated = np.tile(tele_row, (n_pixels, 1))
            X = np.column_stack([raw_flat.reshape(-1, 1), tele_repeated])

            if self._scaler is not None:
                X = self._scaler.transform(X)

            corrections = self._model.predict(X)
            corrected[i] = (raw_flat + corrections).reshape(shape)

        # Fuse: average across corrected trajectories
        return np.mean(corrected, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator — DiffusionMitigationPipeline
# ═══════════════════════════════════════════════════════════════════════════


class DiffusionMitigationPipeline:
    """End-to-end PPU diffusion latent filtering + ML reconstruction.

    Wires the three components together:
      1. :class:`LatentTelemetryExtractor` (frontend)
      2. :class:`GaltonLatentFilter` (Stage 1)
      3. :class:`DiffusionMitigator` (Stage 2)

    **Usage pattern**:

    .. code-block:: python

        pipe = DiffusionMitigationPipeline()

        # Calibrate on a small batch of paired (low-step, high-step) latents
        pipe.calibrate(cal_low_latents, cal_high_latents)

        # Mitigate a budget batch
        result = pipe.mitigate(budget_latents, prompt=prompt,
                               ground_truth_latent=gt_latent)

    Args:
        config: :class:`DiffusionConfig` (immutable).
    """

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        self._config = config or DiffusionConfig()
        self._extractor = LatentTelemetryExtractor()
        self._filter = GaltonLatentFilter(
            reject_fraction=self._config.reject_fraction,
        )
        self._mitigator = DiffusionMitigator(config=self._config)

    @property
    def config(self) -> DiffusionConfig:
        return self._config

    @property
    def extractor(self) -> LatentTelemetryExtractor:
        return self._extractor

    @property
    def filter(self) -> GaltonLatentFilter:
        return self._filter

    @property
    def mitigator(self) -> DiffusionMitigator:
        return self._mitigator

    def calibrate(
        self,
        cal_low_latents: NDArray[np.float64],
        cal_high_latents: NDArray[np.float64],
        *,
        pixel_subsample: int = 1024,
    ) -> DiffusionCalibrationResult:
        """Calibrate the pipeline from paired (low-step, high-step) latents.

        Args:
            cal_low_latents:  Low-step (noisy) latents ``(N, C, H, W)``.
            cal_high_latents: High-step (ideal) latents ``(N, C, H, W)``.
            pixel_subsample:  Max pixels per trajectory for training.

        Returns:
            :class:`DiffusionCalibrationResult`.
        """
        telemetry = self._extractor.extract(cal_low_latents)
        return self._mitigator.calibrate(
            telemetry, cal_low_latents, cal_high_latents,
            pixel_subsample=pixel_subsample,
        )

    def mitigate(
        self,
        latents: NDArray[np.float64],
        *,
        prompt: str = "",
        ground_truth_latent: Optional[NDArray[np.float64]] = None,
    ) -> DiffusionMitigationResult:
        """Run the full two-stage diffusion mitigation pipeline.

        1. Extract telemetry from all latents.
        2. **Stage 1 — Galton Filter**: reject structural outliers.
        3. **Stage 2 — ML Reconstruction**: correct surviving latents.
        4. Fuse corrected latents into a single mitigated output.

        Args:
            latents:              Budget latents ``(N, C, H, W)``.
            prompt:               Text prompt for CLIP scoring.
            ground_truth_latent:  Optional reference latent ``(C, H, W)``
                                  for metric computation.

        Returns:
            :class:`DiffusionMitigationResult`.
        """
        t0 = time.monotonic()

        # ── Telemetry extraction (frontend) ───────────────────────────
        telemetry = self._extractor.extract(latents)

        # ── Stage 1: Galton outlier rejection ─────────────────────────
        mask = self._filter.filter(telemetry)
        n_survivors = int(mask.sum())
        n_rejected = len(mask) - n_survivors

        surviving_latents = latents[mask]
        surviving_telemetry = telemetry[mask]

        logger.info(
            "Stage 1 Galton filter: %d/%d latents survived (%.1f%% rejected)",
            n_survivors, len(mask), 100.0 * n_rejected / len(mask),
        )

        # ── Stage 2: ML reconstruction + fusion ──────────────────────
        mitigated_latent = self._mitigator.predict(
            surviving_telemetry, surviving_latents,
        )

        # ── Raw mean baseline (no mitigation) ────────────────────────
        raw_mean_latent = np.mean(latents, axis=0)

        elapsed = time.monotonic() - t0

        # ── Metrics vs ground truth ───────────────────────────────────
        fid_score = 0.0
        clip_score_val = 0.0
        psnr_val = 0.0
        improvement = float("nan")

        if ground_truth_latent is not None:
            fid_mit = compute_latent_fid(mitigated_latent, ground_truth_latent)
            fid_raw = compute_latent_fid(raw_mean_latent, ground_truth_latent)
            fid_score = fid_mit

            clip_score_val = compute_clip_score(
                mitigated_latent, prompt, ground_truth_latent,
            )
            psnr_val = compute_psnr(mitigated_latent, ground_truth_latent)

            if fid_mit > 1e-15:
                improvement = fid_raw / fid_mit

        return DiffusionMitigationResult(
            mitigated_latent=mitigated_latent,
            raw_mean_latent=raw_mean_latent,
            stage1_survivors=n_survivors,
            stage1_rejected=n_rejected,
            fid_score=fid_score,
            clip_score=clip_score_val,
            psnr=psnr_val,
            improvement_factor=improvement,
            latency_seconds=elapsed,
            metadata={
                "n_budget_latents": len(latents),
                "reject_fraction": self._config.reject_fraction,
                "model_name": self._config.model_name,
                "prompt": prompt,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Quick Diffusion Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def run_diffusion_benchmark(
    prompt: str = (
        "A macro photography shot of a mechanical watch movement, "
        "intricate gears, ruby bearings, dramatic studio lighting, "
        "8k resolution, photorealistic."
    ),
    gt_steps: int = 50,
    budget_steps: int = 10,
    n_batch: int = 8,
    n_calibration: int = 16,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
    latent_height: int = DEFAULT_LATENT_HEIGHT,
    latent_width: int = DEFAULT_LATENT_WIDTH,
    reject_fraction: float = 0.25,
    model_name: str = "random_forest",
    pixel_subsample: int = 1024,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a complete diffusion-model acceleration benchmark.

    Simulates:
      1. **Ground Truth** — 1 latent at ``gt_steps`` steps (expensive).
      2. **Raw Budget** — 1 latent at ``budget_steps`` steps (cheap).
      3. **Qgate Batch** — ``n_batch`` latents at ``budget_steps``,
         filtered and fused via the PPU mitigation pipeline.

    Args:
        prompt:           Text prompt.
        gt_steps:         Denoising steps for ground truth.
        budget_steps:     Denoising steps for budget runs.
        n_batch:          Number of trajectories in the Qgate batch.
        n_calibration:    Number of paired (low,high) latents for
                          calibrating the ML transfer function.
        latent_channels:  VAE latent channels.
        latent_height:    Spatial height.
        latent_width:     Spatial width.
        reject_fraction:  Stage 1 Galton filter rejection rate.
        model_name:       Stage 2 regressor.
        pixel_subsample:  Pixels per trajectory for calibration.
        seed:             Master RNG seed.

    Returns:
        Dictionary with ground truth, raw, mitigated metrics +
        timing and params suitable for T13 benchmark ingestion.
    """
    _require_sklearn()
    rng = np.random.default_rng(seed)

    latent_kw = dict(
        latent_channels=latent_channels,
        latent_height=latent_height,
        latent_width=latent_width,
    )

    logger.info("=" * 60)
    logger.info("PPU Diffusion Acceleration Benchmark")
    logger.info("  Prompt: %s", prompt[:60] + "...")
    logger.info("=" * 60)

    # ── 1. Ground Truth (expensive, high-step) ────────────────────────
    t0 = time.monotonic()
    gt_latent = simulate_diffusion_latents(
        prompt=prompt, n_trajectories=1, num_steps=gt_steps,
        seed=rng.integers(0, 2**31), **latent_kw,
    )[0]
    gt_elapsed = time.monotonic() - t0
    # Simulate realistic GPU time: ~0.8s per step
    gt_gpu_time = gt_steps * 0.8

    logger.info("Ground truth (%d steps): generated in %.3fs", gt_steps, gt_elapsed)

    # ── 2. Raw Budget (cheap, low-step) ───────────────────────────────
    t1 = time.monotonic()
    raw_latent = simulate_diffusion_latents(
        prompt=prompt, n_trajectories=1, num_steps=budget_steps,
        seed=rng.integers(0, 2**31), **latent_kw,
    )[0]
    raw_elapsed = time.monotonic() - t1
    raw_gpu_time = budget_steps * 0.8

    raw_fid = compute_latent_fid(raw_latent, gt_latent)
    raw_clip = compute_clip_score(raw_latent, prompt, gt_latent)
    raw_psnr = compute_psnr(raw_latent, gt_latent)

    logger.info("Raw budget  (%d steps): FID=%.4f  CLIP=%.4f  PSNR=%.2fdB",
                budget_steps, raw_fid, raw_clip, raw_psnr)

    # ── 3. Calibration set (paired low/high latents) ──────────────────
    cal_low = simulate_diffusion_latents(
        prompt=prompt, n_trajectories=n_calibration,
        num_steps=budget_steps,
        seed=rng.integers(0, 2**31), **latent_kw,
    )
    cal_high = simulate_diffusion_latents(
        prompt=prompt, n_trajectories=n_calibration,
        num_steps=gt_steps,
        seed=rng.integers(0, 2**31), **latent_kw,
    )

    # ── 4. Qgate batch mitigation ────────────────────────────────────
    config = DiffusionConfig(
        reject_fraction=reject_fraction,
        model_name=model_name,
        random_state=seed,
        latent_channels=latent_channels,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    pipeline = DiffusionMitigationPipeline(config=config)

    # Calibrate
    cal_result = pipeline.calibrate(
        cal_low, cal_high, pixel_subsample=pixel_subsample,
    )

    # Generate batch of budget latents
    batch_latents = simulate_diffusion_latents(
        prompt=prompt, n_trajectories=n_batch,
        num_steps=budget_steps,
        seed=rng.integers(0, 2**31), **latent_kw,
    )

    # Mitigate
    t2 = time.monotonic()
    mit_result = pipeline.mitigate(
        batch_latents, prompt=prompt, ground_truth_latent=gt_latent,
    )
    mitigate_elapsed = time.monotonic() - t2

    # Qgate GPU time: n_batch × budget_steps × 0.8s (generation)
    # + pipeline overhead (CPU, ~0.5s)
    qgate_gpu_time = n_batch * budget_steps * 0.8 + mitigate_elapsed

    # FID improvement
    fid_improvement = raw_fid / mit_result.fid_score if mit_result.fid_score > 1e-15 else float("inf")

    # Compute-to-quality ROI: GT cost / Qgate cost × quality ratio
    compute_roi = (gt_gpu_time / qgate_gpu_time) if qgate_gpu_time > 0 else 0

    # Quality assessment descriptions
    if raw_fid > 5.0:
        raw_artifacts = "Severe gear warping, smeared ruby bearings, loss of fine detail"
    elif raw_fid > 2.0:
        raw_artifacts = "Noticeable texture noise, soft edges on gear teeth"
    else:
        raw_artifacts = "Minor noise, mostly acceptable"

    if mit_result.fid_score < 0.5:
        mit_artifacts = "Crisp macro details, sharp gear teeth, clear ruby facets"
    elif mit_result.fid_score < 2.0:
        mit_artifacts = "Near-reference quality, minor smoothing in fine textures"
    else:
        mit_artifacts = "Visible but reduced artefacts vs raw budget"

    return {
        "prompt": prompt,
        "ground_truth": {
            "steps": gt_steps,
            "gpu_time": gt_gpu_time,
            "fid": 0.0,
            "clip_score": 1.0,
            "psnr": 100.0,
            "artifacts": "Reference quality — perfect fidelity",
        },
        "raw_budget": {
            "steps": budget_steps,
            "gpu_time": raw_gpu_time,
            "fid": raw_fid,
            "clip_score": raw_clip,
            "psnr": raw_psnr,
            "artifacts": raw_artifacts,
        },
        "qgate_mitigated": {
            "steps": budget_steps,
            "n_batch": n_batch,
            "gpu_time": qgate_gpu_time,
            "fid": mit_result.fid_score,
            "clip_score": mit_result.clip_score,
            "psnr": mit_result.psnr,
            "artifacts": mit_artifacts,
            "stage1_survivors": mit_result.stage1_survivors,
            "stage1_rejected": mit_result.stage1_rejected,
        },
        "improvement": {
            "fid_improvement": fid_improvement,
            "clip_improvement": mit_result.clip_score / max(raw_clip, 1e-15),
            "psnr_improvement_db": mit_result.psnr - raw_psnr,
            "compute_roi": compute_roi,
            "speedup_vs_gt": gt_gpu_time / max(qgate_gpu_time, 1e-15),
        },
        "calibration": {
            "n_calibration": n_calibration,
            "model_name": cal_result.model_name,
            "train_mae": cal_result.train_mae,
            "pixel_subsample": pixel_subsample,
            "elapsed_seconds": cal_result.elapsed_seconds,
        },
        "timing": {
            "gt_wall_seconds": gt_elapsed,
            "raw_wall_seconds": raw_elapsed,
            "mitigate_wall_seconds": mitigate_elapsed,
        },
        "params": {
            "gt_steps": gt_steps,
            "budget_steps": budget_steps,
            "n_batch": n_batch,
            "n_calibration": n_calibration,
            "reject_fraction": reject_fraction,
            "model_name": model_name,
            "latent_shape": [latent_channels, latent_height, latent_width],
        },
    }
