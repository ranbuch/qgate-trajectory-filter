"""
sampler.py — QgateSampler: transparent drop-in replacement for Qiskit SamplerV2.

The QgateSampler intercepts submitted circuits, injects energy-probe
ancilla qubits, executes via the underlying SamplerV2, applies Galton
adaptive thresholding to filter low-quality trajectories, and returns
a standard ``PrimitiveResult`` with the filtered (higher-fidelity)
measurement outcomes.

From the caller's perspective QgateSampler behaves identically to
``SamplerV2`` — the probe injection, filtering, and result
reconstruction are entirely transparent.

Usage::

    from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
    from qgate.sampler import QgateSampler, SamplerConfig

    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")

    # Wrap the real sampler — everything else is transparent
    sampler = QgateSampler(backend=backend)
    job = sampler.run([pub])          # identical to SamplerV2.run()
    result = job.result()             # standard PrimitiveResult

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915

Licensed under the QGATE Source Available Evaluation License v1.2.
Academic research, internal evaluation, and peer review are freely permitted.
Commercial deployment requires a separate license.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("qgate.sampler")

# ---------------------------------------------------------------------------
# Lazy Qiskit imports — fail gracefully
# ---------------------------------------------------------------------------

try:
    from qiskit import QuantumCircuit  # type: ignore[import-untyped]
    from qiskit.circuit import ClassicalRegister  # type: ignore[import-untyped]
    from qiskit.circuit.library import RYGate  # type: ignore[import-untyped]
    from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import-untyped]
        generate_preset_pass_manager,
    )

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    from qiskit_ibm_runtime import SamplerV2 as _BaseSampler  # type: ignore[import-untyped]

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False
    _BaseSampler = None  # type: ignore[assignment,misc]


def _require_deps() -> None:
    if not HAS_QISKIT:
        raise ImportError(
            "Qiskit is required for QgateSampler.  "
            "Install with:  pip install qgate[qiskit]"
        )
    if not HAS_RUNTIME:
        raise ImportError(
            "qiskit-ibm-runtime is required for QgateSampler.  "
            "Install with:  pip install qiskit-ibm-runtime"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover — lightweight fallback
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = None  # type: ignore[assignment,misc]

    def _field_fallback(**kw: Any) -> Any:  # type: ignore[misc]
        return kw.get("default")

    Field = _field_fallback  # type: ignore[assignment]


class SamplerConfig(BaseModel):  # type: ignore[misc]
    """Configuration for the QgateSampler probe-and-filter pipeline.

    All fields are immutable after construction to prevent accidental
    mutation between runs.

    Attributes:
        probe_angle:        Total weak-rotation angle (radians) distributed
                            across nearest-neighbour probe gates.  Smaller
                            angles ≈ weaker measurement back-action on the
                            system register.  Default π/6 ≈ 30°.
        target_acceptance:  Target fraction of shots to accept (0, 1).
                            The Galton threshold adapts to maintain this
                            acceptance rate.  Default 0.05 (top 5 %).
        window_size:        Rolling window capacity for the Galton adaptive
                            threshold (number of individual shot scores).
        min_window_size:    Minimum observations before adaptation kicks in
                            (warmup phase uses ``baseline_threshold``).
        baseline_threshold: Fallback threshold used during warmup or when
                            the adaptive window has insufficient data.
        min_threshold:      Floor — threshold never drops below this.
        max_threshold:      Ceiling — threshold never exceeds this.
        use_quantile:       If True (default), use empirical quantile for
                            threshold placement.  If False, use z-score.
        robust_stats:       If True (default) and ``use_quantile=False``,
                            use median + MAD instead of mean + std.
        z_sigma:            Number of σ above centre for z-score mode.
        optimization_level: Qiskit transpiler optimization level (0–3).
        oversample_factor:  Multiplicative factor to request extra shots
                            from the backend so that post-filtering still
                            yields ≈ the originally-requested shot count.
                            Set to 1.0 to disable oversampling.
    """

    probe_angle: float = Field(
        default=math.pi / 6,
        gt=0.0,
        le=math.pi,
        description="Total weak-rotation angle for energy probe (radians)",
    )
    target_acceptance: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Target fraction of shots to accept",
    )
    window_size: int = Field(default=4096, ge=64, description="Galton rolling window size")
    min_window_size: int = Field(
        default=100,
        ge=1,
        description="Minimum observations before adaptation (warmup)",
    )
    baseline_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Fallback threshold during warmup"
    )
    min_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Threshold floor")
    max_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Threshold ceiling")
    use_quantile: bool = Field(default=True, description="Use empirical quantile (recommended)")
    robust_stats: bool = Field(
        default=True, description="Use median + MAD for z-score mode"
    )
    z_sigma: float = Field(default=1.645, ge=0.0, description="Std-dev multiplier (z-score mode)")
    optimization_level: int = Field(
        default=1, ge=0, le=3, description="Qiskit transpiler optimization level"
    )
    oversample_factor: float = Field(
        default=1.0,
        ge=1.0,
        le=20.0,
        description="Oversampling factor (1.0 = no oversampling)",
    )

    if ConfigDict is not None:
        model_config = ConfigDict(frozen=True, extra="forbid")


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight Galton threshold (self-contained — no dependency on qgate.threshold)
# ═══════════════════════════════════════════════════════════════════════════

_MAD_TO_SIGMA: float = 1.4826


@dataclass
class _GaltonState:
    """Internal mutable state for the adaptive Galton threshold."""

    window: deque = field(default_factory=lambda: deque(maxlen=4096))
    current_threshold: float = 0.65
    in_warmup: bool = True


class _SamplerGaltonThreshold:
    """Lightweight Galton adaptive threshold tailored for QgateSampler.

    This is a self-contained re-implementation so that ``sampler.py``
    does not pull in the full ``qgate.threshold`` module — keeping the
    OS layer decoupled and independently testable.
    """

    def __init__(self, cfg: SamplerConfig) -> None:
        self._cfg = cfg
        self._state = _GaltonState(
            window=deque(maxlen=cfg.window_size),
            current_threshold=cfg.baseline_threshold,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_threshold(self) -> float:
        return self._state.current_threshold

    @property
    def in_warmup(self) -> bool:
        return self._state.in_warmup

    # ------------------------------------------------------------------
    # Observe + adapt
    # ------------------------------------------------------------------

    def observe_batch(self, scores: Sequence[float]) -> float:
        """Ingest a batch of per-shot scores and return the new threshold.

        During warmup (fewer than ``min_window_size`` observations) the
        baseline threshold is used.  After warmup the threshold is set
        via empirical quantile or z-score depending on config.
        """
        self._state.window.extend(scores)
        n = len(self._state.window)
        if n < self._cfg.min_window_size:
            self._state.in_warmup = True
            self._state.current_threshold = self._cfg.baseline_threshold
            return self._state.current_threshold

        self._state.in_warmup = False
        arr = np.asarray(self._state.window, dtype=np.float64)

        if self._cfg.use_quantile:
            q = 1.0 - self._cfg.target_acceptance
            raw = float(np.quantile(arr, q))
        else:
            if self._cfg.robust_stats:
                centre = float(np.median(arr))
                mad = float(np.median(np.abs(arr - centre)))
                sigma = mad * _MAD_TO_SIGMA
            else:
                centre = float(np.mean(arr))
                sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            raw = centre + self._cfg.z_sigma * sigma

        clamped = max(self._cfg.min_threshold, min(raw, self._cfg.max_threshold))
        self._state.current_threshold = clamped
        return clamped

    def reset(self) -> None:
        """Clear the rolling window and reset to warmup."""
        self._state.window.clear()
        self._state.current_threshold = self._cfg.baseline_threshold
        self._state.in_warmup = True


# ═══════════════════════════════════════════════════════════════════════════
# Result container for the lazy-evaluation pattern
# ═══════════════════════════════════════════════════════════════════════════


class QgateSamplerResult:
    """Wraps a raw ``PrimitiveResult`` and lazily applies Galton filtering.

    Behaves like a normal ``PrimitiveResult`` — calling ``.result()``
    triggers the filter pass and returns a standard result object with
    only the accepted (high-quality) shots.
    """

    def __init__(
        self,
        raw_result: Any,
        sampler_ref: QgateSampler,
        probe_metadata: list[dict[str, Any]],
    ) -> None:
        self._raw = raw_result
        self._sampler = sampler_ref
        self._probe_meta = probe_metadata
        self._filtered: Any | None = None

    def result(self) -> Any:
        """Return the Galton-filtered ``PrimitiveResult``."""
        if self._filtered is None:
            self._filtered = self._sampler._apply_galton_filter(
                self._raw, self._probe_meta
            )
        return self._filtered

    # Forward all other attribute access to the raw result (transparency)
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.result(), name)


# ═══════════════════════════════════════════════════════════════════════════
# QgateSampler — the main OS entry-point
# ═══════════════════════════════════════════════════════════════════════════


class QgateSampler:
    """Transparent drop-in replacement for Qiskit ``SamplerV2``.

    Intercepts submitted circuits, injects energy-probe ancilla qubits,
    delegates execution to the real ``SamplerV2``, applies Galton
    adaptive thresholding to filter low-quality trajectories, and
    returns a standard ``PrimitiveResult`` with filtered outcomes.

    Example::

        from qiskit_ibm_runtime import QiskitRuntimeService
        from qgate.sampler import QgateSampler

        service = QiskitRuntimeService()
        backend = service.backend("ibm_torino")

        sampler = QgateSampler(backend=backend)
        job = sampler.run([pub])
        result = job.result()        # standard PrimitiveResult, filtered

    Args:
        backend:   IBM Runtime backend (or Aer simulator).
        config:    Optional :class:`SamplerConfig`.  Defaults are sensible.
        sampler:   Optional pre-initialised ``SamplerV2``.  If *None* one
                   is created from *backend*.
    """

    def __init__(
        self,
        backend: Any,
        config: SamplerConfig | None = None,
        sampler: Any | None = None,
    ) -> None:
        _require_deps()
        self._backend = backend
        self._config = config or SamplerConfig()
        self._inner_sampler = sampler or _BaseSampler(mode=backend)
        self._galton = _SamplerGaltonThreshold(self._config)
        self._pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=self._config.optimization_level,
        )
        logger.info(
            "QgateSampler initialised — probe_angle=%.3f, "
            "target_acceptance=%.3f, oversample=%.1f×",
            self._config.probe_angle,
            self._config.target_acceptance,
            self._config.oversample_factor,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SamplerConfig:
        """The active sampler configuration (immutable)."""
        return self._config

    @property
    def backend(self) -> Any:
        """The underlying backend."""
        return self._backend

    @property
    def current_threshold(self) -> float:
        """Current Galton adaptive threshold."""
        return self._galton.current_threshold

    @property
    def in_warmup(self) -> bool:
        """True if the Galton window is still warming up."""
        return self._galton.in_warmup

    # ------------------------------------------------------------------
    # Public API — mirrors SamplerV2
    # ------------------------------------------------------------------

    def run(
        self,
        pubs: Sequence[Any],
        *,
        shots: int | None = None,
        **kwargs: Any,
    ) -> QgateSamplerResult:
        """Submit PUBs for execution with transparent probe injection.

        This method mirrors ``SamplerV2.run()`` — accepts the same
        arguments, performs identical transpilation, but silently injects
        energy-probe ancilla qubits into each circuit.

        Args:
            pubs:   Sequence of Primitive Unified Blocs (PUBs).  Each PUB
                    is a ``(circuit,)`` or ``(circuit, param_values, shots)``
                    tuple, or a ``SamplerPub`` object.
            shots:  Global shot override (applied to all PUBs).
            **kwargs: Forwarded to the inner ``SamplerV2.run()``.

        Returns:
            :class:`QgateSamplerResult` — call ``.result()`` to get the
            filtered ``PrimitiveResult``.
        """
        modified_pubs = []
        probe_metadata: list[dict[str, Any]] = []

        for pub in pubs:
            circuit, params, pub_shots = self._unpack_pub(pub)
            effective_shots = shots or pub_shots

            # --- Inject energy probe ---
            probed_circuit, meta = self._inject_probes(circuit)

            # --- Oversample to compensate for filtering ---
            if effective_shots is not None and self._config.oversample_factor > 1.0:
                effective_shots = math.ceil(
                    effective_shots * self._config.oversample_factor
                )

            # --- Transpile for backend ---
            isa_circuit = self._pass_manager.run(probed_circuit)
            meta["isa_num_qubits"] = isa_circuit.num_qubits

            # --- Reassemble PUB ---
            new_pub = self._repack_pub(isa_circuit, params, effective_shots)
            modified_pubs.append(new_pub)
            probe_metadata.append(meta)

        # --- Execute via inner SamplerV2 ---
        logger.info("Submitting %d PUBs to inner SamplerV2", len(modified_pubs))
        raw_job = self._inner_sampler.run(modified_pubs, **kwargs)
        raw_result = raw_job.result()

        return QgateSamplerResult(raw_result, self, probe_metadata)

    def reset_threshold(self) -> None:
        """Reset the Galton adaptive threshold to its warmup state."""
        self._galton.reset()
        logger.info("Galton threshold reset to warmup state")

    # ------------------------------------------------------------------
    # Probe injection
    # ------------------------------------------------------------------

    def _inject_probes(
        self,
        circuit: QuantumCircuit,
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
        """Inject energy-probe ancilla into a circuit.

        Adds one ancilla qubit + one classical bit.  Entangles the
        ancilla with nearest-neighbour pairs via controlled-RY gates
        conditioned on spin alignment (low-energy proxy).

        Returns:
            (probed_circuit, metadata_dict)
        """
        n_system = circuit.num_qubits

        # --- Build augmented circuit with ancilla register ---
        from qiskit.circuit import QuantumRegister

        anc_reg = QuantumRegister(1, name="qgate_anc")
        probe_creg = ClassicalRegister(1, name="qgate_probe")

        probed_new = QuantumCircuit(
            *circuit.qregs,
            anc_reg,
            name=circuit.name,
        )

        # Copy existing classical registers
        for creg in circuit.cregs:
            probed_new.add_register(creg)

        # Add probe classical register
        probed_new.add_register(probe_creg)

        # Copy all existing gates
        for instruction in circuit.data:
            probed_new.append(instruction)

        # --- Energy probe entanglement ---
        ancilla_qubit = probed_new.qubits[n_system]  # the added ancilla
        n_pairs = max(n_system - 1, 1)
        per_pair_angle = self._config.probe_angle / n_pairs

        for i in range(n_system - 1):
            qi = probed_new.qubits[i]
            qj = probed_new.qubits[i + 1]

            # Path A: reward |00⟩ alignment (flip → 2-CRY → un-flip)
            probed_new.x(qi)
            probed_new.x(qj)
            cry_00 = RYGate(per_pair_angle).control(2)
            probed_new.append(cry_00, [qi, qj, ancilla_qubit])
            probed_new.x(qj)
            probed_new.x(qi)

            # Path B: reward |11⟩ alignment (2-CRY directly)
            cry_11 = RYGate(per_pair_angle).control(2)
            probed_new.append(cry_11, [qi, qj, ancilla_qubit])

        # Measure ancilla
        probed_new.measure(ancilla_qubit, probe_creg[0])

        metadata = {
            "n_system_qubits": n_system,
            "n_total_qubits": n_system + 1,
            "ancilla_index": n_system,
            "probe_creg_name": "qgate_probe",
            "probe_angle": self._config.probe_angle,
        }

        logger.debug(
            "Injected probe: %d system qubits → %d total, angle=%.3f",
            n_system,
            n_system + 1,
            self._config.probe_angle,
        )

        return probed_new, metadata

    # ------------------------------------------------------------------
    # Galton filtering + result reconstruction
    # ------------------------------------------------------------------

    def _apply_galton_filter(
        self,
        raw_result: Any,
        probe_metadata: list[dict[str, Any]],
    ) -> Any:
        """Filter raw SamplerV2 results via Galton adaptive thresholding.

        For each PUB result:
        1. Extract the probe (ancilla) bit from each shot
        2. Compute per-shot quality scores from probe measurements
        3. Apply Galton adaptive threshold
        4. Reconstruct a ``PrimitiveResult`` with only accepted shots

        Returns:
            A ``PrimitiveResult``-compatible object with filtered data.
        """
        from qiskit.primitives.containers import (  # type: ignore[import-untyped]
            BitArray,
            PrimitiveResult,
            PubResult,
        )
        from qiskit.primitives.containers.data_bin import (  # type: ignore[import-untyped]
            DataBin,
        )

        filtered_pub_results = []

        for pub_idx, pub_result in enumerate(raw_result):
            meta = probe_metadata[pub_idx]
            probe_creg_name = meta["probe_creg_name"]

            # --- Extract measurement data ---
            data = pub_result.data

            # Get all classical register names
            creg_names = [
                attr for attr in dir(data) if not attr.startswith("_")
            ]

            # Separate probe bits from system bits
            probe_bitarray = getattr(data, probe_creg_name, None)
            system_creg_names = [
                name for name in creg_names
                if name != probe_creg_name
                and isinstance(getattr(data, name, None), BitArray)
            ]

            if probe_bitarray is None:
                warnings.warn(
                    f"PUB {pub_idx}: probe register '{probe_creg_name}' not found "
                    f"— passing through unfiltered",
                    stacklevel=2,
                )
                filtered_pub_results.append(pub_result)
                continue

            # --- Extract per-shot probe values ---
            n_shots = probe_bitarray.num_shots
            # BitArray → numpy: shape (num_shots, num_bits)
            probe_array = probe_bitarray.array  # uint8 packed array
            # For a single-bit register, each byte has the bit in the LSB
            probe_bits = (probe_array.flatten() & 1).astype(np.float64)

            # --- Compute quality scores ---
            # Score = probe_bit value (1 = low-energy trajectory, 0 = high-energy)
            # The ancilla rotation is designed so P(ancilla=1) correlates with
            # the number of aligned nearest-neighbour pairs (low energy).
            scores = probe_bits

            # --- Apply Galton threshold ---
            threshold = self._galton.observe_batch(scores.tolist())
            accepted_mask = scores >= threshold

            # If threshold is too aggressive (nothing passes), use all shots
            n_accepted = int(accepted_mask.sum())
            if n_accepted == 0:
                logger.warning(
                    "PUB %d: zero shots accepted (threshold=%.4f) — "
                    "falling back to ancilla=1 post-selection",
                    pub_idx,
                    threshold,
                )
                # Fallback: accept only ancilla=1 shots
                accepted_mask = scores > 0.5
                n_accepted = int(accepted_mask.sum())
                if n_accepted == 0:
                    logger.warning(
                        "PUB %d: no ancilla=1 shots either — "
                        "returning all shots unfiltered",
                        pub_idx,
                    )
                    accepted_mask = np.ones(n_shots, dtype=bool)
                    n_accepted = n_shots

            logger.info(
                "PUB %d: %d/%d shots accepted (threshold=%.4f, warmup=%s)",
                pub_idx,
                n_accepted,
                n_shots,
                threshold,
                self._galton.in_warmup,
            )

            # --- Reconstruct filtered BitArrays (system registers only) ---
            accepted_indices = np.where(accepted_mask)[0]
            filtered_data_dict: dict[str, Any] = {}

            for creg_name in system_creg_names:
                original_ba: BitArray = getattr(data, creg_name)
                # Extract accepted shots from the packed array
                original_array = original_ba.array  # shape (num_shots, num_bytes)
                filtered_array = original_array[accepted_indices]
                filtered_ba = BitArray(filtered_array, num_bits=original_ba.num_bits)
                filtered_data_dict[creg_name] = filtered_ba

            # --- Build filtered DataBin + PubResult ---
            # DataBin shape must match the BitArray.shape (which is the
            # *parameter broadcast* shape, NOT the shot count).  For
            # standard sampler usage this is always ().
            original_shape = getattr(data, "_shape", ())
            filtered_data = DataBin(**filtered_data_dict, shape=original_shape)

            # Preserve metadata from original result
            original_metadata = getattr(pub_result, "metadata", {})
            filtered_metadata = dict(original_metadata) if original_metadata else {}
            filtered_metadata["qgate_filter"] = {
                "total_shots": n_shots,
                "accepted_shots": n_accepted,
                "acceptance_rate": n_accepted / n_shots if n_shots > 0 else 0.0,
                "threshold": threshold,
                "in_warmup": self._galton.in_warmup,
                "probe_angle": meta["probe_angle"],
            }

            filtered_pub = PubResult(filtered_data, metadata=filtered_metadata)
            filtered_pub_results.append(filtered_pub)

        return PrimitiveResult(filtered_pub_results, metadata=raw_result.metadata)

    # ------------------------------------------------------------------
    # PUB packing / unpacking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_pub(pub: Any) -> tuple[QuantumCircuit, Any, int | None]:
        """Unpack a PUB into (circuit, parameter_values, shots).

        Handles:
          - Plain ``QuantumCircuit`` (no params, no shots)
          - Tuple of 1–3 elements: ``(circuit,)`` / ``(circuit, params)``
            / ``(circuit, params, shots)``
          - ``SamplerPub`` objects with ``.circuit``, ``.parameter_values``,
            ``.shots`` attributes
        """
        if isinstance(pub, QuantumCircuit):
            return pub, None, None

        # SamplerPub or namedtuple-like object
        if hasattr(pub, "circuit"):
            circuit = pub.circuit
            params = getattr(pub, "parameter_values", None)
            shots = getattr(pub, "shots", None)
            return circuit, params, shots

        # Tuple / list
        if isinstance(pub, (tuple, list)):
            circuit = pub[0]
            params = pub[1] if len(pub) > 1 else None
            shots = pub[2] if len(pub) > 2 else None
            return circuit, params, shots

        raise TypeError(
            f"Cannot unpack PUB of type {type(pub).__name__}. "
            f"Expected QuantumCircuit, SamplerPub, or tuple."
        )

    @staticmethod
    def _repack_pub(
        circuit: QuantumCircuit,
        params: Any,
        shots: int | None,
    ) -> tuple:
        """Repack into a PUB tuple suitable for ``SamplerV2.run()``.

        Returns the simplest tuple form that SamplerV2 accepts.
        """
        if params is not None and shots is not None:
            return (circuit, params, shots)
        if params is not None:
            return (circuit, params)
        if shots is not None:
            return (circuit, [], shots)
        return (circuit,)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        backend_name = getattr(self._backend, "name", str(self._backend))
        return (
            f"QgateSampler(backend={backend_name!r}, "
            f"probe_angle={self._config.probe_angle:.3f}, "
            f"target_acceptance={self._config.target_acceptance:.3f}, "
            f"threshold={self._galton.current_threshold:.4f})"
        )
