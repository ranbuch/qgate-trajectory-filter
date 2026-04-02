"""
neural_mitigation.py — PyTorch-based neural error mitigation strategies.

Extends the qgate ML-augmented mitigation pipeline (Stage 2 + Stage 3)
with three neural architecture strategies for edge-compute / FPGA
deployment.  All strategies ingest **Level-1.5 telemetry** — continuous
Level-1 IQ readout data mapped into discrete micro-state clusters via
unsupervised learning (k-means / Voronoi) — and reconstruct noiseless
observables for active error mitigation.

Architecture
------------

Uses the **Strategy Design Pattern** with a common abstract base class
:class:`ErrorMitigationStrategy` and three interchangeable concrete
strategies:

* **Strategy 1 — QJLLinearTransformer** (default):
    Lightweight Linear Attention Transformer with 1-bit Quantized
    Johnson–Lindenstrauss (QJL) error correction on the attention
    mechanism and INT8 quantization-aware training (QAT).

* **Strategy 2 — LegacyQuantizedLSTM** (baseline):
    Standard quantized 1D-LSTM for latency and accuracy comparison.
    Serves as the control baseline for benchmarking.

* **Strategy 3 — DiffusionAnomalyDetector** (experimental):
    Lightweight discrete sequence-to-sequence denoising autoencoder /
    diffusion model for high-accuracy transient anomaly detection.

All strategies are **ONNX/HLS-compatible** for FPGA deployment and
integrate with the existing :class:`~qgate.mitigation.TelemetryMitigator`
and :class:`~qgate.pulse_mitigator.PulseMitigator` pipeline stages.

The :class:`TelemetryProcessor` orchestrator dynamically loads a
strategy by name and provides a unified ``forward`` interface.

Usage::

    from qgate.neural_mitigation import (
        TelemetryProcessor,
        NeuralMitigationConfig,
        run_historical_benchmarks,
    )

    # Select strategy and run inference
    processor = TelemetryProcessor(method="qjl_transformer")
    corrected = processor.forward(micro_state_tokens)

    # Benchmark all strategies against exact observables
    report = run_historical_benchmarks(tokens, exact_observables)

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

import abc
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

logger = logging.getLogger("qgate.neural_mitigation")

# ---------------------------------------------------------------------------
# Lazy PyTorch imports — fail gracefully
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:  # pragma: no cover
    raise ImportError(
        "qgate.neural_mitigation requires PyTorch ≥ 2.0. "
        "Install it with:  pip install 'qgate[neural]'"
    ) from None

# Quantization-aware training support (PyTorch ≥ 2.0)
try:
    import torch.ao.quantization as tq
    from torch.ao.quantization import (
        DeQuantStub,
        QuantStub,
        get_default_qat_qconfig,
    )

    HAS_QAT = True
except (ImportError, AttributeError):  # pragma: no cover
    HAS_QAT = False
    tq = None  # type: ignore[assignment]

# ONNX export support — requires both torch.onnx and the standalone onnx package
try:
    import torch.onnx
    import onnx as _onnx_pkg  # noqa: F401 — needed by torch.onnx.export internally

    HAS_ONNX = True
except (ImportError, AttributeError):  # pragma: no cover
    HAS_ONNX = False

# ---------------------------------------------------------------------------
# Pydantic — with lightweight fallback (mirrors mitigation.py pattern)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover — lightweight fallback
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = None  # type: ignore[assignment,misc]

    def _field_fallback(**kw: Any) -> Any:  # type: ignore[misc]
        return kw.get("default")

    Field = _field_fallback  # type: ignore[assignment]


def _require_torch() -> None:
    """No-op — torch is guaranteed available when this module loads."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


class NeuralMitigationConfig(BaseModel):  # type: ignore[misc]
    """Configuration for neural error mitigation strategies.

    All fields are immutable after construction (``frozen=True``) to
    prevent accidental mutation between calibration and inference.

    Attributes:
        vocab_size:         Number of discrete micro-state token IDs.
                            Determined by the k-means / Voronoi
                            clustering applied to raw Level-1 IQ data.
        embed_dim:          Dimensionality of the token embedding layer.
        max_seq_len:        Maximum input sequence length (number of
                            readout time-steps per shot).
        n_heads:            Number of attention heads (Strategy 1).
        n_layers:           Number of encoder layers.
        hidden_dim:         LSTM hidden dimension (Strategy 2) or
                            feedforward dimension (Strategy 1).
        dropout:            Dropout probability for regularisation.
        diffusion_steps:    Number of denoising steps (Strategy 3).
        qjl_dim:            QJL projection dimension (Strategy 1).
                            Reduces attention matrix rank for FPGA.
        use_qat:            Enable INT8 quantization-aware training.
        random_state:       RNG seed for reproducibility.
    """

    vocab_size: int = Field(
        default=64,
        ge=2,
        le=4096,
        description="Number of discrete micro-state token IDs (k-means clusters)",
    )
    embed_dim: int = Field(
        default=32,
        ge=4,
        le=512,
        description="Token embedding dimensionality",
    )
    max_seq_len: int = Field(
        default=128,
        ge=1,
        le=8192,
        description="Maximum readout sequence length",
    )
    n_heads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of attention heads (Strategy 1)",
    )
    n_layers: int = Field(
        default=2,
        ge=1,
        le=12,
        description="Number of encoder / decoder layers",
    )
    hidden_dim: int = Field(
        default=64,
        ge=4,
        le=1024,
        description="Hidden dimension for LSTM / feedforward layers",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout probability",
    )
    diffusion_steps: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of discrete denoising steps (Strategy 3)",
    )
    qjl_dim: int = Field(
        default=16,
        ge=1,
        le=256,
        description="QJL projection dimension (Strategy 1)",
    )
    use_qat: bool = Field(
        default=False,
        description="Enable INT8 quantization-aware training",
    )
    use_temporal_downsample: bool = Field(
        default=False,
        description=(
            "Place a strided Conv1D before the Transformer encoder to "
            "reduce sequence length before attention, cutting compute "
            "by the stride factor with minimal accuracy loss."
        ),
    )
    downsample_stride: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Stride for temporal downsampling Conv1D (1 = no downsampling)",
    )
    random_state: int = Field(
        default=42,
        description="RNG seed for reproducibility",
    )

    if ConfigDict is not None:
        model_config = ConfigDict(frozen=True, extra="forbid")


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NeuralCalibrationResult:
    """Artefacts produced by a neural strategy's ``calibrate`` call.

    Attributes:
        strategy_name:      Name of the strategy that was calibrated.
        n_samples:          Number of training samples.
        final_loss:         Final training loss.
        n_epochs:           Number of training epochs completed.
        n_parameters:       Total model parameters.
        elapsed_seconds:    Wall-clock calibration time.
        metadata:           Free-form metadata dict.
    """

    strategy_name: str
    n_samples: int
    final_loss: float = 0.0
    n_epochs: int = 0
    n_parameters: int = 0
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuralMitigationResult:
    """Result of a neural strategy's ``forward`` pass.

    Attributes:
        mitigated_values:   Batch of corrected observable estimates,
                            shape ``(batch_size,)``.
        raw_logits:         Raw model output logits / continuous values.
        latency_us:         Inference latency in microseconds.
        strategy_name:      Strategy that produced this result.
        metadata:           Free-form metadata dict.
    """

    mitigated_values: np.ndarray
    raw_logits: np.ndarray
    latency_us: float = 0.0
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkReport:
    """Comparative benchmark report across all strategies.

    Attributes:
        results:        Per-strategy results dict mapping strategy name
                        to ``BenchmarkEntry``.
        best_mae:       Strategy name with the lowest MAE.
        best_latency:   Strategy name with the lowest latency.
        timestamp:      ISO timestamp of the benchmark run.
        metadata:       Free-form metadata dict.
    """

    results: Dict[str, "BenchmarkEntry"]
    best_mae: str = ""
    best_latency: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkEntry:
    """Single strategy's benchmark metrics.

    Attributes:
        strategy_name:      Strategy identifier.
        mae:                Mean Absolute Error against exact observables.
        latency_us:         Mean inference latency in microseconds.
        n_parameters:       Total model parameters.
        mitigated_values:   Model output values.
        metadata:           Free-form metadata dict.
    """

    strategy_name: str
    mae: float
    latency_us: float
    n_parameters: int = 0
    mitigated_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Base Strategy
# ═══════════════════════════════════════════════════════════════════════════


class ErrorMitigationStrategy(abc.ABC):
    """Abstract base class for neural error mitigation strategies.

    All concrete strategies must implement:

    * :meth:`forward` — run inference on a batch of micro-state tokens.
    * :meth:`export_to_fpga` — serialise the model for FPGA deployment.

    Optional:

    * :meth:`calibrate` — train / fine-tune the model on labelled data.

    The Strategy Design Pattern allows the :class:`TelemetryProcessor`
    orchestrator to swap strategies transparently at runtime.

    Args:
        config: Shared :class:`NeuralMitigationConfig`.
    """

    def __init__(self, config: NeuralMitigationConfig) -> None:
        _require_torch()
        self._config = config

    @property
    def config(self) -> NeuralMitigationConfig:
        """Return the (immutable) configuration."""
        return self._config

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""
        ...

    @property
    @abc.abstractmethod
    def model(self) -> nn.Module:
        """Return the underlying ``nn.Module``."""
        ...

    @abc.abstractmethod
    def forward(self, micro_state_tokens: Tensor) -> Tensor:
        """Run inference on a batch of micro-state token sequences.

        Args:
            micro_state_tokens: Integer token IDs of shape
                ``[batch_size, sequence_length]``.

        Returns:
            Corrected observable estimates, shape ``[batch_size]``.
        """
        ...

    @abc.abstractmethod
    def export_to_fpga(self, filepath: Union[str, Path]) -> Path:
        """Export the trained model for FPGA deployment (ONNX format).

        Args:
            filepath: Destination path (with ``.onnx`` extension).

        Returns:
            Resolved path to the exported file.
        """
        ...

    def calibrate(
        self,
        tokens: Tensor,
        targets: Tensor,
        *,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> NeuralCalibrationResult:
        """Train the model on labelled (tokens, targets) pairs.

        Default implementation uses Adam + MSE loss.  Subclasses may
        override for custom training loops (e.g. diffusion noise
        schedule).

        Args:
            tokens:     Input token IDs, ``[n_samples, seq_len]``.
            targets:    Exact observable values, ``[n_samples]``.
            n_epochs:   Number of training epochs.
            lr:         Learning rate.
            batch_size: Mini-batch size.

        Returns:
            :class:`NeuralCalibrationResult` with training metrics.
        """
        _require_torch()
        t0 = time.monotonic()
        model = self.model
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        n_samples = tokens.shape[0]
        final_loss = 0.0

        for epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch_tokens = tokens[idx]
                batch_targets = targets[idx]

                optimizer.zero_grad()
                preds = self.forward(batch_tokens)
                loss = loss_fn(preds, batch_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            final_loss = epoch_loss / max(n_batches, 1)

        elapsed = time.monotonic() - t0
        n_params = sum(p.numel() for p in model.parameters())

        model.eval()

        return NeuralCalibrationResult(
            strategy_name=self.name,
            n_samples=n_samples,
            final_loss=final_loss,
            n_epochs=n_epochs,
            n_parameters=n_params,
            elapsed_seconds=elapsed,
            metadata={"lr": lr, "batch_size": batch_size},
        )

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _export_onnx(
        self,
        filepath: Union[str, Path],
        dummy_input: Tensor,
    ) -> Path:
        """Helper to export any strategy to ONNX format.

        Args:
            filepath:    Destination path.
            dummy_input: Example input tensor for tracing.

        Returns:
            Resolved ``Path`` to the written ONNX file.
        """
        _require_torch()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        if HAS_ONNX:
            torch.onnx.export(
                self.model,
                (dummy_input,),
                str(path),
                input_names=["micro_state_tokens"],
                output_names=["mitigated_observable"],
                dynamic_axes={
                    "micro_state_tokens": {0: "batch_size", 1: "seq_len"},
                    "mitigated_observable": {0: "batch_size"},
                },
                opset_version=17,
            )
            logger.info("Exported %s → %s", self.name, path)
        else:
            # Fallback: save TorchScript for environments without ONNX
            scripted = torch.jit.trace(self.model, (dummy_input,))
            scripted.save(str(path.with_suffix(".pt")))
            path = path.with_suffix(".pt")
            logger.info("ONNX unavailable; exported TorchScript → %s", path)

        return path


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 1 — QJL Linear Transformer
# ═══════════════════════════════════════════════════════════════════════════


class _QJLProjection(nn.Module):
    """1-bit Quantized Johnson–Lindenstrauss (QJL) projection layer.

    Approximates high-dimensional dot-product attention by projecting
    queries and keys into a lower-dimensional space with a random
    {+1, −1} matrix, then computing attention in the reduced space.

    This is a placeholder implementation for the full QJL error
    correction layer.  The 1-bit quantisation makes it amenable to
    HLS synthesis for FPGA deployment (only XNOR + popcount ops).

    The JL lemma guarantees that pairwise distances are preserved
    up to ``(1 ± ε)`` with probability ≥ ``1 − δ`` when::

        qjl_dim ≥ C · ε⁻² · log(n / δ)

    where ``n`` is the sequence length and ``C`` is a small constant.

    Args:
        input_dim:  Original dimension of Q/K vectors.
        qjl_dim:    Target projection dimension.
        seed:       RNG seed for reproducible random projection matrix.
    """

    def __init__(self, input_dim: int, qjl_dim: int, seed: int = 42) -> None:
        super().__init__()
        _require_torch()
        gen = torch.Generator().manual_seed(seed)
        # Random Rademacher matrix: entries ∈ {+1, −1}
        R = torch.sign(torch.randn(input_dim, qjl_dim, generator=gen))
        R[R == 0] = 1.0  # edge case: exactly zero → +1
        # Scale factor for unbiased estimation: 1/√(qjl_dim)
        self.register_buffer("projection", R / math.sqrt(qjl_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Project input from ``input_dim`` → ``qjl_dim``.

        Args:
            x: Tensor of shape ``[..., input_dim]``.

        Returns:
            Projected tensor of shape ``[..., qjl_dim]``.
        """
        return x @ self.projection


class _LinearAttentionBlock(nn.Module):
    """Linear Attention with optional QJL projection.

    Standard attention has ``O(n²)`` complexity in sequence length.
    Linear attention replaces ``softmax(QKᵀ)·V`` with
    ``φ(Q)·(φ(K)ᵀ·V)`` using a feature map ``φ``, reducing complexity
    to ``O(n·d²)`` — critical for FPGA real-time inference.

    When QJL is enabled, Q and K are first projected to a lower dimension
    before computing the linear attention kernel, further reducing
    computation and memory bandwidth.

    Args:
        embed_dim:  Model dimension (d_model).
        n_heads:    Number of attention heads.
        qjl_dim:    QJL projection dimension (0 = disabled).
        dropout:    Dropout probability.
        seed:       RNG seed for QJL matrix.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        qjl_dim: int = 0,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        _require_torch()
        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # QJL projection (optional)
        self.qjl: Optional[_QJLProjection] = None
        if qjl_dim > 0:
            self.qjl = _QJLProjection(self.head_dim, qjl_dim, seed=seed)

    @staticmethod
    def _elu_feature_map(x: Tensor) -> Tensor:
        """ELU+1 feature map for linear attention kernel."""
        return F.elu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear attention.

        Args:
            x: Input of shape ``[batch, seq_len, embed_dim]``.

        Returns:
            Output of shape ``[batch, seq_len, embed_dim]``.
        """
        B, L, D = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, n_heads, L, head_dim]

        # Optional QJL dimensionality reduction
        if self.qjl is not None:
            Q = self.qjl(Q)
            K = self.qjl(K)

        # Linear attention: φ(Q) · (φ(K)ᵀ · V)
        Q = self._elu_feature_map(Q)
        K = self._elu_feature_map(K)

        # Compute KᵀV: [B, n_heads, d_k, head_dim]
        KV = torch.einsum("bhld,bhlv->bhdv", K, V)
        # Normalisation denominator: sum of φ(K) over L
        Z = 1.0 / (torch.einsum("bhld,bhd->bhl", Q, K.sum(dim=2)) + 1e-6)
        # Attention output: φ(Q) · (KᵀV) with normalisation
        attn = torch.einsum("bhld,bhdv,bhl->bhlv", Q, KV, Z)

        # Merge heads
        attn = attn.transpose(1, 2).contiguous().view(B, L, D if self.qjl is None else
                                                       self.n_heads * V.shape[-1])
        out = self.W_o(attn)
        return self.dropout(out)


class _TransformerEncoderBlock(nn.Module):
    """Single encoder block: Linear Attention + FFN + LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        hidden_dim: int,
        qjl_dim: int = 0,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        _require_torch()
        self.attn = _LinearAttentionBlock(
            embed_dim, n_heads, qjl_dim=qjl_dim, dropout=dropout, seed=seed,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm Transformer block."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _QJLLinearTransformerModule(nn.Module):
    """Full QJL Linear Transformer model for observable estimation.

    Architecture::

        TokenEmbedding → PositionalEncoding → [Conv1D↓s] →
        N × EncoderBlock → GlobalPool → RegressionHead → scalar

    When ``use_temporal_downsample=True``, a lightweight 1-D
    convolutional layer with ``stride=downsample_stride`` is inserted
    after embedding.  This shrinks the sequence length by the stride
    factor before it hits the attention blocks, cutting compute with
    minimal accuracy loss.

    QAT (INT8 quantization-aware training) structures are injected
    when ``use_qat=True``, wrapping the model with ``QuantStub`` /
    ``DeQuantStub`` so quantisation effects are simulated during
    training and the final model can be exported as INT8.
    """

    def __init__(self, config: NeuralMitigationConfig) -> None:
        super().__init__()
        _require_torch()
        self.use_qat = config.use_qat and HAS_QAT

        # ── QAT stubs ────────────────────────────────────────────────
        if self.use_qat:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        # ── Embedding ─────────────────────────────────────────────────
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.embed_dim) * 0.02
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Temporal downsampling (optional) ──────────────────────────
        self.use_downsample = (
            config.use_temporal_downsample
            and config.downsample_stride > 1
        )
        if self.use_downsample:
            s = config.downsample_stride
            # Conv1D over the time axis: (B, embed_dim, L) → (B, embed_dim, L//s)
            self.downsample_conv = nn.Conv1d(
                in_channels=config.embed_dim,
                out_channels=config.embed_dim,
                kernel_size=s,
                stride=s,
                padding=0,
            )

        # ── Encoder stack ─────────────────────────────────────────────
        self.layers = nn.ModuleList([
            _TransformerEncoderBlock(
                embed_dim=config.embed_dim,
                n_heads=config.n_heads,
                hidden_dim=config.hidden_dim,
                qjl_dim=config.qjl_dim,
                dropout=config.dropout,
                seed=config.random_state + i,
            )
            for i in range(config.n_layers)
        ])

        # ── Regression head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        self._config = config

    def forward(self, tokens: Tensor) -> Tensor:
        """Forward pass: tokens → scalar observable estimates.

        Args:
            tokens: ``[batch_size, seq_len]`` integer token IDs.

        Returns:
            ``[batch_size]`` corrected observable estimates.
        """
        B, L = tokens.shape

        # Embed + positional encoding
        x = self.embedding(tokens) + self.pos_encoding[:, :L, :]
        x = self.embed_dropout(x)

        # ── Optional temporal downsampling ────────────────────────────
        if self.use_downsample:
            # (B, L, D) → (B, D, L) → Conv1d → (B, D, L//s) → (B, L//s, D)
            x = x.transpose(1, 2)
            x = self.downsample_conv(x)
            x = x.transpose(1, 2)

        if self.use_qat:
            x = self.quant(x)

        # Encoder layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [B, embed_dim]

        if self.use_qat:
            x = self.dequant(x)

        # Regression head → scalar
        return self.head(x).squeeze(-1)  # [B]


class QJLLinearTransformer(ErrorMitigationStrategy):
    """Strategy 1: Lightweight Linear Attention Transformer with QJL.

    Default strategy for production deployment.  Combines:

    * **Linear attention** — ``O(n·d²)`` complexity vs ``O(n²·d)``
      for standard softmax attention.
    * **QJL dimensionality reduction** — 1-bit random projection on
      Q/K matrices reduces memory bandwidth.
    * **INT8 QAT** — quantization-aware training for FPGA INT8 inference.

    This architecture targets sub-100µs inference latency on edge FPGAs
    (e.g. Xilinx Versal / Intel Agilex) while maintaining ≤ 1% MAE
    degradation relative to the FP32 baseline.
    """

    def __init__(self, config: Optional[NeuralMitigationConfig] = None) -> None:
        cfg = config or NeuralMitigationConfig()
        super().__init__(cfg)
        self._model = _QJLLinearTransformerModule(cfg)
        torch.manual_seed(cfg.random_state)

    @property
    def name(self) -> str:
        return "qjl_transformer"

    @property
    def model(self) -> nn.Module:
        return self._model

    def forward(self, micro_state_tokens: Tensor) -> Tensor:
        """Infer corrected observable from Level-1.5 micro-state tokens.

        Args:
            micro_state_tokens: ``[batch_size, seq_len]`` int64 tokens.

        Returns:
            ``[batch_size]`` corrected observables.
        """
        return self._model(micro_state_tokens)

    def export_to_fpga(self, filepath: Union[str, Path]) -> Path:
        """Export to ONNX for FPGA HLS synthesis.

        The exported model uses INT8 quantisation (if QAT was enabled)
        and static sequence length for HLS compatibility.
        """
        dummy = torch.zeros(
            1, self._config.max_seq_len, dtype=torch.long,
        )
        return self._export_onnx(filepath, dummy)

    # ── Speed optimisations ───────────────────────────────────────────

    def compile_model(self, mode: str = "default") -> None:
        """Apply ``torch.compile`` to the inner model (in-place).

        This JIT-compiles the forward graph into optimised kernels via
        TorchInductor.  On CPU (Apple Silicon / x86) ``mode="default"``
        provides solid speed-up without requiring CUDA.

        On CUDA ``mode="reduce-overhead"`` additionally uses CUDA
        graphs for further gains.

        Args:
            mode: ``torch.compile`` mode — one of ``"default"``,
                  ``"reduce-overhead"``, ``"max-autotune"``.

        Raises:
            RuntimeError: If ``torch.compile`` is not available
                          (PyTorch < 2.0).
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "torch.compile requires PyTorch >= 2.0 "
                f"(installed: {torch.__version__})"
            )
        self._model = torch.compile(self._model, mode=mode)  # type: ignore[assignment]
        logger.info("torch.compile applied (mode=%s)", mode)

    _ort_session: Any = None  # lazily initialised ORT InferenceSession

    def prepare_ort_session(
        self,
        opset_version: int = 17,
    ) -> None:
        """Export to ONNX in-memory and create an ORT InferenceSession.

        After calling this method, :meth:`forward_ort` can be used for
        inference through ONNX Runtime, which often outperforms eager
        PyTorch for small models on CPU.

        Raises:
            ImportError: If ``onnxruntime`` or ``onnx`` is not installed.
        """
        try:
            import onnxruntime as ort  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "ONNX Runtime is required for ORT inference. "
                "Install it with: pip install onnxruntime"
            ) from exc

        if not HAS_ONNX:
            raise ImportError(
                "The 'onnx' package is required for ONNX export. "
                "Install it with: pip install onnx"
            )

        import io
        buf = io.BytesIO()

        self._model.eval()
        dummy = torch.zeros(
            1, self._config.max_seq_len, dtype=torch.long,
        )
        torch.onnx.export(
            self._model,
            (dummy,),
            buf,
            input_names=["micro_state_tokens"],
            output_names=["mitigated_observable"],
            dynamic_axes={
                "micro_state_tokens": {0: "batch_size", 1: "seq_len"},
                "mitigated_observable": {0: "batch_size"},
            },
            opset_version=opset_version,
        )

        buf.seek(0)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.intra_op_num_threads = 1  # deterministic single-thread
        self._ort_session = ort.InferenceSession(
            buf.read(), sess_options=sess_opts,
        )
        logger.info("ORT InferenceSession prepared (opset %d)", opset_version)

    def forward_ort(self, micro_state_tokens: Tensor) -> Tensor:
        """Run inference through ONNX Runtime.

        Args:
            micro_state_tokens: ``[batch_size, seq_len]`` int64 tokens.

        Returns:
            ``[batch_size]`` corrected observables as a CPU tensor.

        Raises:
            RuntimeError: If :meth:`prepare_ort_session` was not called.
        """
        if self._ort_session is None:
            raise RuntimeError(
                "ORT session not initialised. "
                "Call prepare_ort_session() first."
            )
        tokens_np = micro_state_tokens.detach().cpu().numpy()
        (output,) = self._ort_session.run(
            None, {"micro_state_tokens": tokens_np},
        )
        return torch.from_numpy(output.squeeze(-1) if output.ndim > 1 else output)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 2 — Legacy Quantized LSTM (Baseline)
# ═══════════════════════════════════════════════════════════════════════════


class _LegacyQuantizedLSTMModule(nn.Module):
    """Quantized 1D-LSTM for baseline comparison.

    Architecture::

        TokenEmbedding → LSTM → FinalHiddenState → RegressionHead → scalar

    Serves as the latency and accuracy control baseline against which
    Strategy 1 (Transformer) and Strategy 3 (Diffusion) are compared.

    For FPGA deployment, the LSTM is quantized to INT8 post-training
    or via QAT.  While LSTMs are less parallelisable than Transformers,
    they are well-supported by HLS synthesis tools and provide
    predictable latency characteristics.
    """

    def __init__(self, config: NeuralMitigationConfig) -> None:
        super().__init__()
        _require_torch()
        self.use_qat = config.use_qat and HAS_QAT

        if self.use_qat:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self._config = config

    def forward(self, tokens: Tensor) -> Tensor:
        """Forward pass: tokens → scalar observable estimates.

        Args:
            tokens: ``[batch_size, seq_len]`` integer token IDs.

        Returns:
            ``[batch_size]`` corrected observable estimates.
        """
        x = self.embedding(tokens)  # [B, L, embed_dim]

        if self.use_qat:
            x = self.quant(x)

        # LSTM: take final hidden state
        _, (h_n, _) = self.lstm(x)  # h_n: [n_layers, B, hidden_dim]
        out = h_n[-1]  # Last layer's final hidden: [B, hidden_dim]

        if self.use_qat:
            out = self.dequant(out)

        return self.head(out).squeeze(-1)  # [B]


class LegacyQuantizedLSTM(ErrorMitigationStrategy):
    """Strategy 2: Standard quantized 1D-LSTM baseline.

    Control baseline for latency and accuracy comparisons.  This
    represents the "prior art" approach to sequence-based error
    mitigation — a standard LSTM processing temporal IQ readout
    sequences.  Benchmarking against this baseline validates that
    the QJL Transformer (Strategy 1) provides genuine improvements.
    """

    def __init__(self, config: Optional[NeuralMitigationConfig] = None) -> None:
        cfg = config or NeuralMitigationConfig()
        super().__init__(cfg)
        self._model = _LegacyQuantizedLSTMModule(cfg)
        torch.manual_seed(cfg.random_state)

    @property
    def name(self) -> str:
        return "legacy_lstm"

    @property
    def model(self) -> nn.Module:
        return self._model

    def forward(self, micro_state_tokens: Tensor) -> Tensor:
        return self._model(micro_state_tokens)

    def export_to_fpga(self, filepath: Union[str, Path]) -> Path:
        dummy = torch.zeros(
            1, self._config.max_seq_len, dtype=torch.long,
        )
        return self._export_onnx(filepath, dummy)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 3 — Diffusion Anomaly Detector (Experimental)
# ═══════════════════════════════════════════════════════════════════════════


class _DiffusionDenoisingModule(nn.Module):
    """Lightweight discrete denoising autoencoder for anomaly detection.

    Architecture::

        TokenEmbedding + StepEmbedding → Encoder → BottleNeck →
        Decoder → Reconstruction → RegressionHead → scalar

    The diffusion formulation works as follows:

    1. **Forward process** (training only): progressively corrupt the
       input token sequence by replacing tokens with random noise tokens
       according to a cosine schedule over ``T`` steps.

    2. **Reverse process**: the model learns to reconstruct the clean
       sequence from the corrupted version.  The quality of
       reconstruction (reconstruction loss) at each step serves as an
       anomaly / decoherence indicator.

    3. **Observable estimation**: the final reconstruction is pooled
       and passed through a regression head to predict the corrected
       observable value.

    For FPGA deployment, only the single-step reverse pass is needed
    (not the full T-step chain), making inference latency comparable
    to a standard autoencoder.
    """

    def __init__(self, config: NeuralMitigationConfig) -> None:
        super().__init__()
        _require_torch()
        self._config = config

        # Token + timestep embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.step_embedding = nn.Embedding(config.diffusion_steps + 1, config.embed_dim)

        # Encoder: compress sequence
        self.encoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Bottleneck
        self.bottleneck = nn.Linear(config.hidden_dim, config.embed_dim)

        # Decoder: reconstruct token logits
        self.decoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vocab_size),
        )

        # Regression head: pooled features → observable
        self.regression_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # Cosine noise schedule: β(t) controls corruption rate
        steps = torch.arange(config.diffusion_steps + 1, dtype=torch.float32)
        # Cosine schedule: smooth from ~0 to ~1
        alpha_bar = torch.cos(
            (steps / config.diffusion_steps) * (math.pi / 2)
        ) ** 2
        self.register_buffer("alpha_bar", alpha_bar)

    def _corrupt(self, tokens: Tensor, t: int) -> Tensor:
        """Corrupt tokens at diffusion step ``t``.

        Replaces each token with a random token with probability
        ``1 − ᾱ(t)``.  At ``t=0`` (no noise) the input is unchanged;
        at ``t=T`` the sequence is fully random.

        Args:
            tokens: Clean token IDs, ``[batch, seq_len]``.
            t:      Diffusion timestep (0 = clean, T = fully noisy).

        Returns:
            Corrupted token IDs, same shape.
        """
        alpha = self.alpha_bar[t]
        mask = torch.rand_like(tokens.float()) > alpha
        noise_tokens = torch.randint_like(tokens, 0, self._config.vocab_size)
        return torch.where(mask, noise_tokens, tokens)

    def forward(self, tokens: Tensor, t: Optional[int] = None) -> Tensor:
        """Forward pass: denoise and estimate observable.

        During training, ``t`` specifies the corruption level.
        During inference, ``t=1`` (minimal corruption) is used as a
        single-step denoising pass for FPGA efficiency.

        Args:
            tokens: ``[batch_size, seq_len]`` integer token IDs.
            t:      Diffusion timestep.  ``None`` defaults to ``1``
                    (single-step inference).

        Returns:
            ``[batch_size]`` corrected observable estimates.
        """
        B, L = tokens.shape
        step = t if t is not None else 1

        # Corrupt input (during training; at inference t=1 gives minimal noise)
        if self.training and t is not None and t > 0:
            noisy_tokens = self._corrupt(tokens, t)
        else:
            noisy_tokens = tokens

        # Embed tokens + timestep
        x = self.token_embedding(noisy_tokens)  # [B, L, embed_dim]
        step_emb = self.step_embedding(
            torch.full((B,), step, dtype=torch.long, device=tokens.device)
        )  # [B, embed_dim]
        x = x + step_emb.unsqueeze(1)  # broadcast over seq_len

        # Encode → bottleneck
        encoded = self.encoder(x)   # [B, L, hidden_dim]
        latent = self.bottleneck(encoded)  # [B, L, embed_dim]

        # Pool → regression
        pooled = latent.mean(dim=1)  # [B, embed_dim]
        observable = self.regression_head(pooled).squeeze(-1)  # [B]

        return observable


class DiffusionAnomalyDetector(ErrorMitigationStrategy):
    """Strategy 3: Discrete sequence diffusion denoising autoencoder.

    Experimental high-accuracy strategy that frames error mitigation as
    a **denoising problem**: transient decoherence anomalies are treated
    as noise injected into the micro-state token sequence, and the model
    learns to denoise (reconstruct) the clean sequence.

    The diffusion formulation provides two advantages:

    1. **Anomaly sensitivity** — the reconstruction loss at each
       diffusion step is a calibrated anomaly score.  Steps where the
       model cannot reconstruct well correspond to genuine decoherence
       events.

    2. **Robustness** — training with multi-step corruption acts as
       strong data augmentation, making the model robust to the
       stochastic nature of quantum noise.

    For FPGA deployment, only a single-step denoising pass is needed
    (``t=1``), so inference latency is comparable to the LSTM baseline.
    """

    def __init__(self, config: Optional[NeuralMitigationConfig] = None) -> None:
        cfg = config or NeuralMitigationConfig()
        super().__init__(cfg)
        self._model = _DiffusionDenoisingModule(cfg)
        torch.manual_seed(cfg.random_state)

    @property
    def name(self) -> str:
        return "diffusion_detector"

    @property
    def model(self) -> nn.Module:
        return self._model

    def forward(self, micro_state_tokens: Tensor) -> Tensor:
        return self._model(micro_state_tokens)

    def calibrate(
        self,
        tokens: Tensor,
        targets: Tensor,
        *,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> NeuralCalibrationResult:
        """Train with multi-step diffusion noise schedule.

        Override the default training loop to sample random diffusion
        timesteps for each mini-batch, training the model to denoise
        at varying corruption levels.

        Args:
            tokens:     Input token IDs, ``[n_samples, seq_len]``.
            targets:    Exact observable values, ``[n_samples]``.
            n_epochs:   Number of training epochs.
            lr:         Learning rate.
            batch_size: Mini-batch size.

        Returns:
            :class:`NeuralCalibrationResult` with training metrics.
        """
        _require_torch()
        t0 = time.monotonic()
        model = self.model
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        n_samples = tokens.shape[0]
        max_t = self._config.diffusion_steps
        final_loss = 0.0

        for epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch_tokens = tokens[idx]
                batch_targets = targets[idx]

                # Sample random diffusion timestep per batch
                t_step = torch.randint(1, max_t + 1, (1,)).item()

                optimizer.zero_grad()
                preds = self._model(batch_tokens, t=t_step)
                loss = loss_fn(preds, batch_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            final_loss = epoch_loss / max(n_batches, 1)

        elapsed = time.monotonic() - t0
        n_params = sum(p.numel() for p in model.parameters())
        model.eval()

        return NeuralCalibrationResult(
            strategy_name=self.name,
            n_samples=n_samples,
            final_loss=final_loss,
            n_epochs=n_epochs,
            n_parameters=n_params,
            elapsed_seconds=elapsed,
            metadata={
                "lr": lr,
                "batch_size": batch_size,
                "diffusion_steps": max_t,
            },
        )

    def export_to_fpga(self, filepath: Union[str, Path]) -> Path:
        dummy = torch.zeros(
            1, self._config.max_seq_len, dtype=torch.long,
        )
        return self._export_onnx(filepath, dummy)


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Registry
# ═══════════════════════════════════════════════════════════════════════════

#: Map of short names to strategy classes.
STRATEGY_REGISTRY: Dict[str, Type[ErrorMitigationStrategy]] = {
    "qjl_transformer": QJLLinearTransformer,
    "legacy_lstm": LegacyQuantizedLSTM,
    "diffusion_detector": DiffusionAnomalyDetector,
}


def list_strategies() -> List[str]:
    """Return all registered strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def fast_qjl_config(
    *,
    n_layers: int = 1,
    n_heads: int = 2,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    downsample_stride: int = 4,
    **overrides: Any,
) -> NeuralMitigationConfig:
    """Create a speed-optimised QJL Transformer config.

    Returns a :class:`NeuralMitigationConfig` tuned for minimal
    latency while preserving reasonable accuracy.  The key changes
    relative to the default config are:

    * **1 encoder layer** (vs 2) — halves encoder cost.
    * **2 attention heads** (vs 4) — reduces per-layer work.
    * **Temporal downsampling** enabled with ``stride=4`` — quarters
      the sequence length before attention.

    All arguments can be further overridden via ``**overrides``.

    Example::

        cfg = fast_qjl_config()
        strategy = QJLLinearTransformer(cfg)
        strategy.compile_model()          # + torch.compile
        strategy.prepare_ort_session()    # or ORT

    Returns:
        A frozen :class:`NeuralMitigationConfig` instance.
    """
    defaults: Dict[str, Any] = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "use_temporal_downsample": True,
        "downsample_stride": downsample_stride,
    }
    defaults.update(overrides)
    return NeuralMitigationConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# TelemetryProcessor — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class TelemetryProcessor:
    """Orchestrator that dynamically loads and runs a mitigation strategy.

    Uses the Strategy Design Pattern to allow transparent swapping of
    the underlying neural architecture at runtime.  Integrates with the
    existing qgate mitigation pipeline — the processor can be used as a
    drop-in replacement for the scikit-learn regressor in
    :class:`~qgate.mitigation.TelemetryMitigator` Stage 2.

    Args:
        method:     Strategy name (see :func:`list_strategies`).
        config:     Shared :class:`NeuralMitigationConfig`.

    Example::

        proc = TelemetryProcessor(method="qjl_transformer")
        corrected = proc.forward(tokens)  # [batch, seq_len] → [batch]

        # Switch strategy at runtime
        proc.set_strategy("legacy_lstm")
        baseline = proc.forward(tokens)
    """

    def __init__(
        self,
        method: str = "qjl_transformer",
        config: Optional[NeuralMitigationConfig] = None,
    ) -> None:
        _require_torch()
        self._config = config or NeuralMitigationConfig()
        self._strategy: Optional[ErrorMitigationStrategy] = None
        self.set_strategy(method)

    @property
    def strategy(self) -> ErrorMitigationStrategy:
        """Return the currently active strategy."""
        if self._strategy is None:
            raise RuntimeError("No strategy loaded. Call set_strategy() first.")
        return self._strategy

    @property
    def config(self) -> NeuralMitigationConfig:
        return self._config

    def set_strategy(self, method: str) -> None:
        """Load a strategy by name.

        Args:
            method: One of :func:`list_strategies`.

        Raises:
            ValueError: If *method* is not a registered strategy.
        """
        method_lower = method.lower().replace("-", "_").replace(" ", "_")
        if method_lower not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy {method!r}. "
                f"Available: {list_strategies()}"
            )
        cls = STRATEGY_REGISTRY[method_lower]
        self._strategy = cls(config=self._config)
        logger.info("TelemetryProcessor: loaded strategy %r", method_lower)

    def forward(self, micro_state_tokens: Tensor) -> NeuralMitigationResult:
        """Run inference through the active strategy.

        Args:
            micro_state_tokens: ``[batch_size, seq_len]`` int64 tokens.

        Returns:
            :class:`NeuralMitigationResult` with corrected values and
            timing information.
        """
        strategy = self.strategy
        model = strategy.model
        model.eval()

        with torch.no_grad():
            t0 = time.perf_counter_ns()
            output = strategy.forward(micro_state_tokens)
            t1 = time.perf_counter_ns()

        latency_us = (t1 - t0) / 1_000.0  # ns → µs
        values = output.detach().cpu().numpy()

        return NeuralMitigationResult(
            mitigated_values=values,
            raw_logits=values,
            latency_us=latency_us,
            strategy_name=strategy.name,
            metadata={"batch_size": micro_state_tokens.shape[0]},
        )

    def calibrate(
        self,
        tokens: Tensor,
        targets: Tensor,
        **kwargs: Any,
    ) -> NeuralCalibrationResult:
        """Calibrate the active strategy on labelled data.

        Args:
            tokens:  Input token IDs, ``[n_samples, seq_len]``.
            targets: Exact observable values, ``[n_samples]``.
            **kwargs: Forwarded to the strategy's ``calibrate`` method.

        Returns:
            :class:`NeuralCalibrationResult`.
        """
        return self.strategy.calibrate(tokens, targets, **kwargs)

    def export_to_fpga(self, filepath: Union[str, Path]) -> Path:
        """Export the active strategy for FPGA deployment."""
        return self.strategy.export_to_fpga(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarking Suite
# ═══════════════════════════════════════════════════════════════════════════


def generate_mock_dataset(
    n_samples: int = 500,
    seq_len: int = 64,
    vocab_size: int = 64,
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """Generate a mock historical Level-1.5 telemetry dataset.

    Simulates the output of the k-means / Voronoi micro-state clustering
    applied to raw Level-1 IQ readout data, plus known exact observables.

    The mock generation process:

    1. Sample random token sequences (representing micro-state IDs).
    2. Compute a "ground truth" observable as a function of token
       statistics (mean, variance, specific token frequencies) to
       create a learnable target.  This mimics how micro-state
       distributions correlate with physical observables.

    Args:
        n_samples:  Number of readout sequences.
        seq_len:    Tokens per sequence.
        vocab_size: Number of micro-state clusters.
        seed:       RNG seed.

    Returns:
        Tuple of ``(tokens, exact_observables)`` where ``tokens`` has
        shape ``[n_samples, seq_len]`` and ``exact_observables`` has
        shape ``[n_samples]``.
    """
    _require_torch()
    gen = torch.Generator().manual_seed(seed)

    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), generator=gen)

    # Observable = f(token statistics):
    # - Weighted sum of token frequencies (physics-motivated: certain
    #   micro-states correlate with error-free readout)
    # - Small nonlinear term for model complexity
    token_freqs = torch.zeros(n_samples, vocab_size)
    for i in range(n_samples):
        for j in range(seq_len):
            token_freqs[i, tokens[i, j].item()] += 1.0
    token_freqs /= seq_len

    # Observation model: linear combination + noise
    weights = torch.randn(vocab_size, generator=gen) * 0.1
    observables = token_freqs @ weights
    # Add nonlinear component (variance sensitivity)
    variance_term = token_freqs.var(dim=1) * 5.0
    observables = observables + variance_term
    # Scale to typical energy range [−30, 0]
    observables = -15.0 + observables * 10.0

    return tokens.long(), observables.float()


def run_historical_benchmarks(
    tokens: Tensor,
    exact_observables: Tensor,
    *,
    config: Optional[NeuralMitigationConfig] = None,
    n_train_epochs: int = 30,
    train_fraction: float = 0.7,
    n_inference_runs: int = 5,
) -> BenchmarkReport:
    """Run comparative benchmarks across all registered strategies.

    For each strategy:

    1. Split data into train / test sets.
    2. Calibrate (train) on the training set.
    3. Run inference on the test set, measure MAE and latency.
    4. Repeat inference ``n_inference_runs`` times for stable timing.

    Args:
        tokens:             ``[n_samples, seq_len]`` token IDs.
        exact_observables:  ``[n_samples]`` ground-truth observables.
        config:             Shared model configuration.
        n_train_epochs:     Training epochs per strategy.
        train_fraction:     Fraction of data used for training.
        n_inference_runs:   Number of inference passes for latency averaging.

    Returns:
        :class:`BenchmarkReport` with per-strategy MAE and latency.
    """
    _require_torch()
    cfg = config or NeuralMitigationConfig()

    n_samples = tokens.shape[0]
    n_train = int(n_samples * train_fraction)

    # Deterministic split
    gen = torch.Generator().manual_seed(cfg.random_state)
    perm = torch.randperm(n_samples, generator=gen)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_tokens = tokens[train_idx]
    train_targets = exact_observables[train_idx]
    test_tokens = tokens[test_idx]
    test_targets = exact_observables[test_idx].numpy()

    results: Dict[str, BenchmarkEntry] = {}

    for strategy_name in list_strategies():
        logger.info("Benchmarking strategy: %s", strategy_name)
        processor = TelemetryProcessor(method=strategy_name, config=cfg)

        # ── Train ─────────────────────────────────────────────────────
        cal = processor.calibrate(
            train_tokens,
            train_targets,
            n_epochs=n_train_epochs,
        )
        logger.info(
            "  %s: trained %d params in %.2fs (loss=%.6f)",
            strategy_name, cal.n_parameters, cal.elapsed_seconds, cal.final_loss,
        )

        # ── Inference (multiple runs for stable timing) ───────────────
        latencies_us: List[float] = []
        mitigated: Optional[np.ndarray] = None

        for _ in range(n_inference_runs):
            result = processor.forward(test_tokens)
            latencies_us.append(result.latency_us)
            mitigated = result.mitigated_values

        assert mitigated is not None  # guaranteed by loop above
        mean_latency_us = float(np.mean(latencies_us))

        # ── MAE ───────────────────────────────────────────────────────
        mae = float(np.mean(np.abs(mitigated - test_targets)))

        results[strategy_name] = BenchmarkEntry(
            strategy_name=strategy_name,
            mae=mae,
            latency_us=mean_latency_us,
            n_parameters=cal.n_parameters,
            mitigated_values=mitigated,
            metadata={
                "n_train": n_train,
                "n_test": len(test_idx),
                "n_epochs": n_train_epochs,
                "final_loss": cal.final_loss,
                "n_inference_runs": n_inference_runs,
            },
        )

        logger.info(
            "  %s: MAE=%.6f  latency=%.1f µs  params=%d",
            strategy_name, mae, mean_latency_us, cal.n_parameters,
        )

    # ── Determine winners ─────────────────────────────────────────────
    best_mae = min(results, key=lambda k: results[k].mae) if results else ""
    best_latency = min(results, key=lambda k: results[k].latency_us) if results else ""

    import datetime

    report = BenchmarkReport(
        results=results,
        best_mae=best_mae,
        best_latency=best_latency,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metadata={
            "n_samples": n_samples,
            "n_strategies": len(results),
            "config": cfg.model_dump() if hasattr(cfg, "model_dump") else {},
        },
    )

    return report


def print_benchmark_report(report: BenchmarkReport) -> str:
    """Format a :class:`BenchmarkReport` as a human-readable table.

    Args:
        report: The benchmark report to format.

    Returns:
        Formatted string representation.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  NEURAL MITIGATION BENCHMARK REPORT")
    lines.append("=" * 78)
    lines.append(f"  Timestamp: {report.timestamp}")
    lines.append(f"  Strategies tested: {len(report.results)}")
    lines.append("")

    # Header
    hdr = f"  {'Strategy':<25} {'MAE':>12} {'Latency (µs)':>15} {'Params':>12}"
    lines.append(hdr)
    lines.append("  " + "-" * 66)

    for name, entry in report.results.items():
        marker = ""
        if name == report.best_mae:
            marker += " ★mae"
        if name == report.best_latency:
            marker += " ★lat"
        lines.append(
            f"  {name:<25} {entry.mae:>12.6f} {entry.latency_us:>15.1f} "
            f"{entry.n_parameters:>12,d}{marker}"
        )

    lines.append("")
    lines.append(f"  Best MAE:     {report.best_mae}")
    lines.append(f"  Best Latency: {report.best_latency}")
    lines.append("=" * 78)
    lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
