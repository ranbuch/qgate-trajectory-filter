"""
qgate — Quantum Trajectory Filter.

Runtime post-selection conditioning for quantum circuits.  This package
explores trajectory filtering concepts from US Patent Application
Nos. 63/983,831 & 63/989,632 and Israeli Patent Application No. 326915.
The underlying invention is patent pending.

Quick start::

    from qgate import TrajectoryFilter, GateConfig
    from qgate.adapters import MockAdapter

    config = GateConfig(n_subsystems=4, n_cycles=2, shots=1024)
    adapter = MockAdapter(error_rate=0.05, seed=42)
    tf = TrajectoryFilter(config, adapter)
    result = tf.run()
    print(result.acceptance_probability)

Install extras for specific backends::

    pip install qgate[qiskit]       # IBM Qiskit
    pip install qgate[cirq]         # Google Cirq (stub)
    pip install qgate[pennylane]    # PennyLane (stub)
    pip install qgate[all]          # Everything

License: QGATE SOURCE AVAILABLE EVALUATION LICENSE v1.2 — see LICENSE
"""

__version__ = "0.6.0"

# ── Primary public API ────────────────────────────────────────────────────
from qgate.adapters.base import BaseAdapter
from qgate.adapters.grover_adapter import GroverTSVFAdapter
from qgate.adapters.qaoa_adapter import QAOATSVFAdapter
from qgate.adapters.qpe_adapter import QPETSVFAdapter
from qgate.adapters.registry import list_adapters, load_adapter
from qgate.adapters.vqe_adapter import VQETSVFAdapter

# ── Legacy / backward-compatible re-exports ──────────────────────────────
# These symbols were available in qgate <= 0.2; kept for compatibility.
from qgate.conditioning import (
    ConditioningStats,
    ParityOutcome,
    apply_rule_to_batch,
    decide_global,
    decide_hierarchical,
    decide_score_fusion,
)
from qgate.config import (
    AdapterKind,
    ConditioningVariant,
    DynamicThresholdConfig,
    FusionConfig,
    GateConfig,
    ProbeConfig,
    ThresholdMode,
)
from qgate.filter import TrajectoryFilter
from qgate.monitors import (
    MultiRateMonitor,
    compute_window_metric,
    score_fusion,
)
from qgate.run_logging import FilterResult, RunLogger, compute_run_id

# ── TelemetryMitigator — two-stage ML-augmented error mitigation ──────────
# Stage 1 Galton trajectory filtering + Stage 2 ML regression correction.
# Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
# CIP addendum — ML-augmented TSVF trajectory mitigation.
from qgate.mitigation import (
    CalibrationResult,
    MitigationResult,
    MitigatorConfig,
    TelemetryMitigator,
)

# ── PulseMitigator — firmware-level ML-driven TLS drift cancellation ──────
# Operates on Level-1 (analog) IQ telemetry, not Level-2 (binary) data.
# Predicts TLS drift and injects inverse frequency shift into drive pulses.
# Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
# CIP addendum — Pulse-level ML-augmented TSVF firmware mitigation.
from qgate.pulse_mitigator import (
    ActiveCancellationResult,
    DriftPrediction,
    PulseCalibrationResult,
    PulseMitigator,
    PulseMitigatorConfig,
    SimulatedPulseSchedule,
    extract_iq_features,
)

# ── QgateTranspiler — ML-aware quantum circuit compiler ───────────────────
# Auto-configures circuit padding and shot oversampling based on active
# mitigation mode.  Disables aggressive chaotic Hamiltonian mixing and
# reduces shot oversampling when ML mitigators replace legacy binary
# filtering, cutting QPU cost by up to 8×.
# Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
# CIP addendum — ML-aware transpilation and shot optimisation.
from qgate.transpiler import (
    CompilationResult,
    MitigationMode,
    QgateTranspiler,
    QgateTranspilerConfig,
    apply_uzdin_unitary_folding,
    validate_noise_scale_factor,
)

# ── TelemetryCompressor — two-stage dimensionality reduction ──────────────
# Spatial topological pooling + tree-based Gini pruning for utility-scale
# (50–156+ qubit) telemetry compression.  Reduces high-dimensional IQ /
# telemetry vectors to dense latent vectors for Stage-2 ML regressors.
# Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
# CIP addendum — telemetry compression for utility-scale ML mitigation.
from qgate.compressor import TelemetryCompressor

# ── TVS — Trajectory Viability Score: HF/LF fusion + Stage-1 filtering ───
# Fuses Level-1 (I/Q soft-decision) or Level-2 (binary hard-decision) HF
# telemetry with LF drift scores via Kalman-style dynamic alpha weighting.
# Normalised fusion scores feed into Galton percentile-based outlier
# rejection (Stage 1), producing a surviving-shot mask + ML features for
# Stage-2 regressors.  Fully vectorised NumPy — no Python for-loops.
# Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
# CIP addendum — Level-1 I/Q trajectory viability scoring.
from qgate.tvs import (
    VALID_FORCE_MODES,
    adaptive_galton_schedule,
    compute_iq_snr,
    normalise_hf_level1_cluster,
    process_telemetry_batch,
)

# ── Neural Mitigation — PyTorch strategy-based error mitigation ───────────
# Three interchangeable neural strategies (QJL Linear Transformer, LSTM
# baseline, Diffusion anomaly detector) for Level-1.5 micro-state telemetry.
# ONNX/HLS compatible for FPGA edge deployment.
# Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
# CIP addendum — Neural ML-augmented TSVF trajectory mitigation.
try:
    from qgate.neural_mitigation import (
        BenchmarkEntry,
        BenchmarkReport,
        DiffusionAnomalyDetector,
        ErrorMitigationStrategy,
        LegacyQuantizedLSTM,
        NeuralCalibrationResult,
        NeuralMitigationConfig,
        NeuralMitigationResult,
        QJLLinearTransformer,
        STRATEGY_REGISTRY,
        TelemetryProcessor,
        fast_qjl_config,
        generate_mock_dataset,
        list_strategies,
        print_benchmark_report,
        run_historical_benchmarks,
    )

    HAS_NEURAL = True
except ImportError:  # pragma: no cover — torch not installed
    HAS_NEURAL = False

# ── QgateSampler OS layer ─────────────────────────────────────────────────
# Transparent drop-in SamplerV2 replacement with autonomous probe injection
# and Galton-filtered result reconstruction.
# Patent pending — US App. Nos. 63/983,831 & 63/989,632, IL 326915.
from qgate.sampler import QgateSampler, SamplerConfig
from qgate.scoring import (
    fuse_scores,
    score_batch,
    score_outcome,
)
from qgate.threshold import (
    GaltonAdaptiveThreshold,
    GaltonSnapshot,
    estimate_diffusion_width,
)

__all__ = [
    # PulseMitigator — firmware-level ML drift cancellation
    "ActiveCancellationResult",
    "AdapterKind",
    # Primary API
    "BaseAdapter",
    # Calibration / Mitigation results
    "CalibrationResult",
    # QgateTranspiler — ML-aware circuit compiler
    "CompilationResult",
    # Legacy (backward compat)
    "ConditioningStats",
    "ConditioningVariant",
    # PulseMitigator result
    "DriftPrediction",
    "DynamicThresholdConfig",
    "FilterResult",
    "FusionConfig",
    # Galton adaptive threshold
    "GaltonAdaptiveThreshold",
    "GaltonSnapshot",
    "GateConfig",
    # Grover/TSVF adapter
    "GroverTSVFAdapter",
    # TelemetryMitigator — two-stage ML error mitigation
    "MitigationResult",
    # Mitigation mode (transpiler)
    "MitigationMode",
    "MitigatorConfig",
    "MultiRateMonitor",
    "ParityOutcome",
    "ProbeConfig",
    # PulseMitigator — pulse-level IQ telemetry + ML
    "PulseCalibrationResult",
    "PulseMitigator",
    "PulseMitigatorConfig",
    # TVS — Trajectory Viability Score (HF/LF fusion + Galton filter)
    "VALID_FORCE_MODES",
    "adaptive_galton_schedule",
    "compute_iq_snr",
    "normalise_hf_level1_cluster",
    "process_telemetry_batch",
    # QAOA/TSVF adapter
    "QAOATSVFAdapter",
    # QPE/TSVF adapter
    "QPETSVFAdapter",
    # QgateSampler OS
    "QgateSampler",
    # QgateTranspiler — ML-aware compiler
    "QgateTranspiler",
    "QgateTranspilerConfig",
    "RunLogger",
    "SamplerConfig",
    # Simulated pulse schedule (Qiskit 2.x fallback)
    "SimulatedPulseSchedule",
    # TelemetryMitigator
    "TelemetryMitigator",
    # TelemetryCompressor — utility-scale telemetry compression
    "TelemetryCompressor",
    "ThresholdMode",
    "TrajectoryFilter",
    # Uzdin unitary folding — noise amplification utilities
    "apply_uzdin_unitary_folding",
    # VQE/TSVF adapter
    "VQETSVFAdapter",
    "apply_rule_to_batch",
    "compute_run_id",
    "compute_window_metric",
    "decide_global",
    "decide_hierarchical",
    "decide_score_fusion",
    "estimate_diffusion_width",
    # IQ feature extraction (public helper)
    "extract_iq_features",
    "fuse_scores",
    "list_adapters",
    "load_adapter",
    "score_batch",
    "score_fusion",
    "score_outcome",
    # Uzdin scale-factor validator
    "validate_noise_scale_factor",
]

# ── Conditionally extend __all__ with neural mitigation symbols ───────────
if HAS_NEURAL:
    __all__.extend([
        # Neural Mitigation — PyTorch strategy-based pipeline
        "BenchmarkEntry",
        "BenchmarkReport",
        "DiffusionAnomalyDetector",
        "ErrorMitigationStrategy",
        "LegacyQuantizedLSTM",
        "NeuralCalibrationResult",
        "NeuralMitigationConfig",
        "NeuralMitigationResult",
        "QJLLinearTransformer",
        "STRATEGY_REGISTRY",
        "TelemetryProcessor",
        "fast_qjl_config",
        "generate_mock_dataset",
        "list_strategies",
        "print_benchmark_report",
        "run_historical_benchmarks",
    ])
