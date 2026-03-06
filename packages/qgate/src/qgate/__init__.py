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

__version__ = "0.5.0"

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

# ── QgateSampler OS layer ─────────────────────────────────────────────────
# Transparent drop-in SamplerV2 replacement with autonomous probe injection
# and Galton-filtered result reconstruction.
# NOTICE: Pre-patent proprietary code — do NOT push to public repositories.
from qgate.sampler import QgateSampler, SamplerConfig

__all__ = [
    "AdapterKind",
    # Primary API
    "BaseAdapter",
    # Legacy (backward compat)
    "ConditioningStats",
    "ConditioningVariant",
    "DynamicThresholdConfig",
    "FilterResult",
    "FusionConfig",
    # Galton adaptive threshold
    "GaltonAdaptiveThreshold",
    "GaltonSnapshot",
    "GateConfig",
    # Grover/TSVF adapter
    "GroverTSVFAdapter",
    "MultiRateMonitor",
    "ParityOutcome",
    "ProbeConfig",
    # QAOA/TSVF adapter
    "QAOATSVFAdapter",
    # QPE/TSVF adapter
    "QPETSVFAdapter",
    # QgateSampler OS
    "QgateSampler",
    "RunLogger",
    "SamplerConfig",
    "ThresholdMode",
    "TrajectoryFilter",
    # VQE/TSVF adapter
    "VQETSVFAdapter",
    "apply_rule_to_batch",
    "compute_run_id",
    "compute_window_metric",
    "decide_global",
    "decide_hierarchical",
    "decide_score_fusion",
    "estimate_diffusion_width",
    "fuse_scores",
    "list_adapters",
    "load_adapter",
    "score_batch",
    "score_fusion",
    "score_outcome",
]
