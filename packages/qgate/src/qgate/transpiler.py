"""
transpiler.py — QgateTranspiler: ML-aware quantum circuit compiler.

When non-parametric ML mitigators (:class:`~qgate.mitigation.TelemetryMitigator`
or :class:`~qgate.pulse_mitigator.PulseMitigator`) are active, the transpiler
**automatically disables** the aggressive Hamiltonian mixing (chaotic padding)
and high-factor shot oversampling that were previously required by the legacy
binary trajectory filter.

Cost rationale
--------------
The legacy ``TrajectoryFilter`` relies on a *binary* accept/reject decision
derived from parity-check scores.  To obtain statistically sufficient accepted
shots, the filter must request up to 10× the target shot count (oversampling)
**and** pad circuits with pseudo-random mixing gates to decorrelate successive
shots.  Both techniques increase QPU compute cost and circuit depth.

ML-based mitigators (``ml_extrapolation``, ``pulse_active``) replace the binary
decision with a *continuous* error-correction prediction.  Because the ML model
extrapolates the correction from telemetry features, it does not discard any
shots — only a small fraction (≈30 %) is truncated by the Galton filter in
Stage 1.  Consequently:

* **Aggressive mixing is unnecessary** — shallow circuits preserve gate
  fidelity and avoid depth amplification.
* **Oversampling can be reduced from 10× to ≈1.2×** — just enough to
  compensate for the Galton truncation.

Pipeline overview
-----------------
1. **Telemetry probe injection** — adds weak-measurement ancilla qubits
   (identical across all modes).
2. **Conditional chaotic padding** — appended only in ``legacy_filter``
   mode.  Skipped entirely when ML mitigators are active.
3. **Shot calculation** — ``optimized_shots = base_shots × oversampling_factor``.

Usage::

    from qiskit import QuantumCircuit
    from qgate.transpiler import QgateTranspiler, QgateTranspilerConfig

    # Automatic ML-aware configuration
    config = QgateTranspilerConfig(mitigation_mode="ml_extrapolation")
    assert config.aggressive_mixing is False
    assert config.oversampling_factor == 1.2

    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.measure_all()

    transpiler = QgateTranspiler(config=config)
    compiled, shots = transpiler.compile(qc, base_shots=4096)

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
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger("qgate.transpiler")

# ---------------------------------------------------------------------------
# Lazy Qiskit imports — fail gracefully
# ---------------------------------------------------------------------------

try:
    from qiskit import QuantumCircuit  # type: ignore[import-untyped]
    from qiskit.circuit import ClassicalRegister, QuantumRegister  # type: ignore[import-untyped]
    from qiskit.circuit.library import RYGate  # type: ignore[import-untyped]

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    QuantumCircuit = None  # type: ignore[assignment,misc]


def _require_qiskit() -> None:
    """Raise ``ImportError`` if Qiskit is not available."""
    if not HAS_QISKIT:
        raise ImportError(
            "Qiskit is required for QgateTranspiler.  "
            "Install with:  pip install qgate[qiskit]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Uzdin Odd-Factor Unitary Folding
# ═══════════════════════════════════════════════════════════════════════════


def validate_noise_scale_factor(scale_factor: int) -> None:
    """Validate that a noise-amplification scale factor is a positive odd integer.

    Per Prof. Raam Uzdin's KIK / Layered mitigation theory, reliable
    noise amplification on physical hardware **must** use odd integer
    scale factors (1, 3, 5, 7, …).  Non-integer factors such as 1.5 or
    2.0 require "digital folding" (inserting pairs of inverse gates or
    identity gates) which introduces coherent errors that scale
    non-linearly — making the amplified noise channel *qualitatively
    different* from the original, not merely stronger.

    The correct procedure is **strict unitary folding**::

        U → U · U† · U          (scale factor 3)
        U → U · U† · U · U† · U  (scale factor 5)

    which preserves the exact logical identity of each gate while
    tripling / quintupling the noise exposure.

    Args:
        scale_factor: The integer noise-amplification factor.

    Raises:
        ValueError: If *scale_factor* is not a positive odd integer.

    Example::

        validate_noise_scale_factor(3)  # OK
        validate_noise_scale_factor(5)  # OK
        validate_noise_scale_factor(2)  # raises ValueError
    """
    if not isinstance(scale_factor, int) or scale_factor < 1:
        raise ValueError(
            f"scale_factor must be a positive integer, got {scale_factor!r}. "
            f"Per Uzdin KIK rules, use odd integers: 1, 3, 5, 7, …"
        )
    if scale_factor % 2 == 0:
        raise ValueError(
            f"scale_factor={scale_factor} is even — this is forbidden by the "
            f"Uzdin odd-factor rule.  Non-odd scaling requires 'digital folding' "
            f"(e.g. inserting CNOT·CNOT or Identity gates), which introduces "
            f"coherent errors that scale non-linearly on physical hardware.  "
            f"Use odd integers only: 1, 3, 5, 7, …"
        )


def apply_uzdin_unitary_folding(circuit: Any, scale_factor: int) -> Any:
    """Apply noise amplification via strict odd-integer unitary folding.

    Implements the gate-level folding rule:

    .. math::

        U \\to U \\cdot (U^\\dagger \\cdot U)^{(k-1)/2}

    where *k* is the odd-integer ``scale_factor``.  Each gate in the
    original circuit is replaced by a sequence that is **logically
    identical** to the original gate but exposes the physical qubit to
    *k*× the hardware noise.

    +---------+------------------------------+
    | k       | Gate sequence                |
    +=========+==============================+
    | 1       | U  (identity, no folding)    |
    +---------+------------------------------+
    | 3       | U · U† · U                   |
    +---------+------------------------------+
    | 5       | U · U† · U · U† · U          |
    +---------+------------------------------+
    | 7       | U · U† · U · U† · U · U† · U |
    +---------+------------------------------+

    This is the *only* physically correct method for noise amplification
    on real hardware.  **Do not use** digital folding (inserting pairs of
    inverse gates or Identity gates at arbitrary points), as it
    introduces coherent error terms that break the depolarising-channel
    assumption required by Zero-Noise Extrapolation.

    Reference:
        Uzdin, R. — Layered KIK mitigation theory.  Gate folding must
        use odd factors to preserve the noise channel structure.

    Args:
        circuit: A Qiskit :class:`QuantumCircuit`.
        scale_factor: A positive odd integer (1, 3, 5, 7, …).

    Returns:
        A new :class:`QuantumCircuit` with each gate folded *k* times.

    Raises:
        ImportError: If Qiskit is not installed.
        ValueError: If *scale_factor* is not a positive odd integer.
        TypeError: If *circuit* is not a ``QuantumCircuit``.

    Example::

        from qiskit import QuantumCircuit
        from qgate.transpiler import apply_uzdin_unitary_folding

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        # folded contains: H·H†·H  then  CX·CX†·CX
        assert folded.size() == 6  # 2 gates × 3-fold
    """
    _require_qiskit()
    validate_noise_scale_factor(scale_factor)

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"Expected a Qiskit QuantumCircuit, got {type(circuit).__name__}"
        )

    if scale_factor == 1:
        return circuit.copy()

    # Number of extra U†·U pairs to append after each gate
    num_pairs = (scale_factor - 1) // 2

    # Build a fresh circuit with the same registers
    folded = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)

    # Non-invertible operations are passed through without folding
    _NON_FOLDABLE = {"measure", "barrier", "reset", "delay"}

    for instruction in circuit.data:
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        if op.name in _NON_FOLDABLE:
            # Measurements, barriers, resets cannot be inverted —
            # pass them through as-is.
            folded.append(op, qargs, cargs)
            continue

        # 1. Original gate (forward)
        folded.append(op, qargs, cargs)

        # 2. Append (U† · U) pairs to reach the target scale factor
        for _ in range(num_pairs):
            folded.append(op.inverse(), qargs, cargs)  # backward
            folded.append(op, qargs, cargs)             # forward

    logger.info(
        "Uzdin unitary folding: scale_factor=%d, "
        "original_gates=%d → folded_gates=%d",
        scale_factor,
        circuit.size(),
        folded.size(),
    )

    return folded


# ═══════════════════════════════════════════════════════════════════════════
# Supported mitigation modes
# ═══════════════════════════════════════════════════════════════════════════

MitigationMode = Literal["legacy_filter", "ml_extrapolation", "pulse_active"]
"""Supported mitigation pipeline modes.

``"legacy_filter"``
    Binary accept/reject trajectory filtering (original qgate pipeline).
    Requires aggressive chaotic padding and high shot oversampling.

``"ml_extrapolation"``
    Two-stage ML error mitigation via :class:`~qgate.mitigation.TelemetryMitigator`.
    Stage 1 Galton filtering retains ≈70 % of shots; Stage 2 ML regression
    predicts the residual correction.  Chaotic padding is unnecessary.

``"pulse_active"``
    Firmware-level active cancellation via :class:`~qgate.pulse_mitigator.PulseMitigator`.
    IQ-level drift prediction feeds back into drive pulses in real time.
    Chaotic padding is bypassed to preserve shallow circuit depth.
"""

# Mode → (aggressive_mixing, oversampling_factor)
_MODE_DEFAULTS: Dict[str, Tuple[bool, float]] = {
    "legacy_filter": (True, 10.0),
    "ml_extrapolation": (False, 1.2),
    "pulse_active": (False, 1.2),
}


# ═══════════════════════════════════════════════════════════════════════════
# QgateTranspilerConfig
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class QgateTranspilerConfig:
    """Configuration for the ML-aware qgate transpiler.

    Automatically tunes ``aggressive_mixing`` and ``oversampling_factor``
    based on the selected ``mitigation_mode``.  When non-parametric ML
    models (:class:`~qgate.mitigation.TelemetryMitigator` or
    :class:`~qgate.pulse_mitigator.PulseMitigator`) handle error
    extrapolation, chaotic Hamiltonian padding is counter-productive:
    it inflates circuit depth without improving the ML correction signal.
    Likewise, shot oversampling can be reduced from 10× to ≈1.2× because
    only ≈30 % of shots are discarded by the Galton truncation.

    Attributes:
        mitigation_mode:      Active error-mitigation strategy.
        aggressive_mixing:    Whether to inject pseudo-random chaotic
                              padding gates.  Auto-set to ``False`` for
                              ML modes, ``True`` for ``legacy_filter``.
        oversampling_factor:  Multiplicative shot inflation factor.
                              Auto-set to 1.2 for ML modes, 10.0 for
                              ``legacy_filter``.
        probe_angle:          Total weak-rotation angle (rad) for the
                              energy-probe ancilla entanglement.
        mixing_depth:         Number of chaotic padding layers when
                              ``aggressive_mixing`` is enabled.
        mixing_seed:          RNG seed for reproducible mixing patterns.

    Examples::

        # ML mitigator — shallow depth, minimal oversampling
        cfg = QgateTranspilerConfig(mitigation_mode="ml_extrapolation")
        assert cfg.aggressive_mixing is False
        assert cfg.oversampling_factor == 1.2

        # Legacy filter — full padding, 10× oversampling
        cfg = QgateTranspilerConfig(mitigation_mode="legacy_filter")
        assert cfg.aggressive_mixing is True
        assert cfg.oversampling_factor == 10.0

        # Explicit override — ML mode but custom oversampling
        cfg = QgateTranspilerConfig(
            mitigation_mode="ml_extrapolation",
            oversampling_factor=2.0,
        )
    """

    mitigation_mode: MitigationMode = "legacy_filter"
    aggressive_mixing: bool = True
    oversampling_factor: float = 10.0
    probe_angle: float = math.pi / 6
    mixing_depth: int = 3
    mixing_seed: int = 42

    def __post_init__(self) -> None:
        """Auto-configure flags from mitigation_mode.

        When ``mitigation_mode`` is ``'ml_extrapolation'`` or
        ``'pulse_active'``, force ``aggressive_mixing = False`` and
        reduce ``oversampling_factor`` to 1.2 (sufficient for the
        ≈30 % truncated Galton filter drop).

        When ``mitigation_mode`` is ``'legacy_filter'``, enforce
        ``aggressive_mixing = True`` and ``oversampling_factor = 10.0``
        to ensure enough accepted shots after binary filtering.

        Explicit user overrides for ``oversampling_factor`` are
        preserved when they differ from the class default.
        """
        mode = self.mitigation_mode
        if mode not in _MODE_DEFAULTS:
            raise ValueError(
                f"Unknown mitigation_mode {mode!r}.  "
                f"Choose from: {', '.join(sorted(_MODE_DEFAULTS))}"
            )

        default_mixing, default_oversample = _MODE_DEFAULTS[mode]

        # frozen dataclass → use object.__setattr__
        object.__setattr__(self, "aggressive_mixing", default_mixing)

        # Preserve explicit overrides: if oversampling_factor was left
        # at the class default (10.0), apply the mode-specific default.
        # If the user explicitly set a different value, keep it.
        if self.oversampling_factor == 10.0:
            object.__setattr__(self, "oversampling_factor", default_oversample)

        # Validate ranges
        if self.oversampling_factor < 1.0:
            raise ValueError(
                f"oversampling_factor must be ≥ 1.0, got {self.oversampling_factor}"
            )
        if self.probe_angle <= 0 or self.probe_angle > math.pi:
            raise ValueError(
                f"probe_angle must be in (0, π], got {self.probe_angle}"
            )
        if self.mixing_depth < 1:
            raise ValueError(
                f"mixing_depth must be ≥ 1, got {self.mixing_depth}"
            )

    @classmethod
    def for_mode(cls, mode: MitigationMode, **overrides: Any) -> QgateTranspilerConfig:
        """Factory method — create a config pre-tuned for a mitigation mode.

        This is the recommended way to instantiate the config.  All
        mode-specific defaults are applied automatically, and any
        additional keyword arguments override individual fields.

        Args:
            mode:       One of ``"legacy_filter"``, ``"ml_extrapolation"``,
                        or ``"pulse_active"``.
            **overrides: Keyword arguments forwarded to the constructor.

        Returns:
            A new :class:`QgateTranspilerConfig` instance.

        Examples::

            cfg = QgateTranspilerConfig.for_mode("ml_extrapolation")
            cfg = QgateTranspilerConfig.for_mode("pulse_active", probe_angle=0.3)
        """
        return cls(mitigation_mode=mode, **overrides)


# ═══════════════════════════════════════════════════════════════════════════
# CompilationResult
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CompilationResult:
    """Artefacts returned by :meth:`QgateTranspiler.compile`.

    Attributes:
        circuit:              The transpiled :class:`QuantumCircuit` with
                              probes and (optionally) chaotic padding.
        optimized_shots:      Shot count after applying the oversampling
                              factor to the caller's ``base_shots``.
        probes_injected:      Whether telemetry probes were added.
        chaotic_padding_applied: Whether chaotic mixing was injected.
        metadata:             Free-form metadata dict with compiler
                              telemetry.
    """

    circuit: Any  # QuantumCircuit (typed as Any for import safety)
    optimized_shots: int
    probes_injected: bool = True
    chaotic_padding_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# QgateTranspiler
# ═══════════════════════════════════════════════════════════════════════════


class QgateTranspiler:
    """ML-aware quantum circuit compiler for the qgate pipeline.

    The transpiler prepares circuits for execution by injecting
    weak-measurement telemetry probes and — only when required —
    pseudo-random chaotic Hamiltonian mixing gates.

    Disabling aggressive mixing when ML mitigators are active yields
    two concrete benefits:

    1. **Reduced QPU compute cost.**  Chaotic padding adds 2-qubit gates
       proportional to ``(n_qubits − 1) × mixing_depth``.  On current
       NISQ hardware each additional CX / CZ gate increases execution
       time and reduces overall fidelity budget.

    2. **Preserved shallow circuit depth.**  Non-parametric ML models
       (RandomForest, GradientBoosting) learn the correction from
       telemetry *features* — they do not require the shot distribution
       to be decorrelated.  Keeping circuits shallow maximises the
       fidelity of the raw data that feeds the ML regressor.

    The transpiler operates in three modes:

    ``legacy_filter``
        Full padding + 10× oversampling (original pipeline).

    ``ml_extrapolation``
        Probes only + 1.2× oversampling (TelemetryMitigator active).

    ``pulse_active``
        Probes only + 1.2× oversampling (PulseMitigator active).

    Args:
        config: Transpiler configuration (immutable after construction).

    Example::

        from qiskit import QuantumCircuit
        from qgate.transpiler import QgateTranspiler, QgateTranspilerConfig

        config = QgateTranspilerConfig.for_mode("ml_extrapolation")
        transpiler = QgateTranspiler(config=config)

        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.measure_all()

        result = transpiler.compile(qc, base_shots=4096)
        print(result.circuit.num_qubits)   # 5 (4 system + 1 probe)
        print(result.optimized_shots)       # 4916 (4096 × 1.2)

    Patent reference:
        US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
        CIP addendum — ML-augmented TSVF trajectory mitigation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[QgateTranspilerConfig] = None) -> None:
        _require_qiskit()
        self._config: QgateTranspilerConfig = config or QgateTranspilerConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> QgateTranspilerConfig:
        """Return the (immutable) transpiler configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Public API — compile
    # ------------------------------------------------------------------

    def compile(
        self,
        circuit: Any,  # QuantumCircuit
        base_shots: int = 4096,
    ) -> CompilationResult:
        """Transpile a circuit through the qgate compiler pipeline.

        Pipeline:

        1. **Telemetry probe injection** — adds one ancilla qubit with
           weak-measurement entanglement (all modes).
        2. **Chaotic padding** — pseudo-random mixing gates (``legacy_filter``
           only).  Completely bypassed for ML modes to preserve shallow
           circuit depth and avoid unnecessary depth amplification.
        3. **Shot optimisation** — ``optimized_shots = ceil(base_shots ×
           oversampling_factor)``.

        Args:
            circuit:    The Qiskit :class:`QuantumCircuit` to compile.
            base_shots: Desired number of *useful* shots after filtering.

        Returns:
            :class:`CompilationResult` with the compiled circuit and
            optimised shot count.

        Raises:
            ImportError: If Qiskit is not installed.
            TypeError:   If *circuit* is not a ``QuantumCircuit``.
            ValueError:  If *base_shots* is not a positive integer.
        """
        _require_qiskit()

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                f"Expected a Qiskit QuantumCircuit, got {type(circuit).__name__}"
            )
        if base_shots < 1:
            raise ValueError(f"base_shots must be ≥ 1, got {base_shots}")

        logger.info(
            "QgateTranspiler.compile: mode=%s, base_shots=%d, "
            "mixing=%s, oversample=%.1f×",
            self._config.mitigation_mode,
            base_shots,
            self._config.aggressive_mixing,
            self._config.oversampling_factor,
        )

        # ── Step 1: inject telemetry probes ───────────────────────────
        probed = self._inject_telemetry_probes(circuit)

        # ── Step 2: conditional chaotic padding ───────────────────────
        padding_applied = False
        if self._config.aggressive_mixing:
            probed = self._inject_chaotic_padding(probed)
            padding_applied = True
            logger.info(
                "Chaotic padding applied: %d layers on %d qubits",
                self._config.mixing_depth,
                probed.num_qubits,
            )
        else:
            logger.info(
                "Chaotic padding SKIPPED — ML mitigator active (mode=%s). "
                "Preserving shallow circuit depth to maximise raw fidelity "
                "for the regression model.",
                self._config.mitigation_mode,
            )

        # ── Step 3: shot optimisation ─────────────────────────────────
        optimized_shots = int(math.ceil(base_shots * self._config.oversampling_factor))

        logger.info(
            "Shot budget: %d base → %d optimised (%.1f× factor)",
            base_shots,
            optimized_shots,
            self._config.oversampling_factor,
        )

        return CompilationResult(
            circuit=probed,
            optimized_shots=optimized_shots,
            probes_injected=True,
            chaotic_padding_applied=padding_applied,
            metadata={
                "mitigation_mode": self._config.mitigation_mode,
                "aggressive_mixing": self._config.aggressive_mixing,
                "oversampling_factor": self._config.oversampling_factor,
                "base_shots": base_shots,
                "n_system_qubits": circuit.num_qubits,
                "n_total_qubits": probed.num_qubits,
                "mixing_depth": self._config.mixing_depth if padding_applied else 0,
                "probe_angle": self._config.probe_angle,
            },
        )

    # ------------------------------------------------------------------
    # Internal — telemetry probe injection
    # ------------------------------------------------------------------

    def _inject_telemetry_probes(self, circuit: Any) -> Any:
        """Inject weak-measurement ancilla probes into *circuit*.

        Adds one ancilla qubit and one classical bit.  The ancilla is
        entangled with nearest-neighbour system qubit pairs via
        controlled-RY gates conditioned on spin alignment.  This
        implements the standard qgate energy-probe protocol identical
        to :meth:`QgateSampler._inject_probes`.

        The rotation angle per pair is::

            per_pair_angle = config.probe_angle / max(n_system - 1, 1)

        Two controlled-RY paths reward |00⟩ and |11⟩ alignments:

        * **Path A (|00⟩):** X-flip both neighbours → 2-CRY → un-flip.
        * **Path B (|11⟩):** 2-CRY directly (both qubits high = ferromagnetic).

        Args:
            circuit: The system :class:`QuantumCircuit`.

        Returns:
            A new :class:`QuantumCircuit` with the ancilla register
            and probe measurement appended.
        """
        _require_qiskit()

        n_system = circuit.num_qubits

        # Build augmented circuit with ancilla register
        anc_reg = QuantumRegister(1, name="qgate_anc")
        probe_creg = ClassicalRegister(1, name="qgate_probe")

        probed = QuantumCircuit(
            *circuit.qregs,
            anc_reg,
            name=circuit.name,
        )

        # Copy existing classical registers
        for creg in circuit.cregs:
            probed.add_register(creg)
        probed.add_register(probe_creg)

        # Copy all existing gates
        for instruction in circuit.data:
            probed.append(instruction)

        # Entangle ancilla with nearest-neighbour pairs
        ancilla_qubit = probed.qubits[n_system]
        n_pairs = max(n_system - 1, 1)
        per_pair_angle = self._config.probe_angle / n_pairs

        for i in range(n_system - 1):
            qi = probed.qubits[i]
            qj = probed.qubits[i + 1]

            # Path A: reward |00⟩ alignment (flip → 2-CRY → un-flip)
            probed.x(qi)
            probed.x(qj)
            cry_00 = RYGate(per_pair_angle).control(2)
            probed.append(cry_00, [qi, qj, ancilla_qubit])
            probed.x(qj)
            probed.x(qi)

            # Path B: reward |11⟩ alignment (2-CRY directly)
            cry_11 = RYGate(per_pair_angle).control(2)
            probed.append(cry_11, [qi, qj, ancilla_qubit])

        # Measure ancilla
        probed.measure(ancilla_qubit, probe_creg[0])

        logger.debug(
            "Telemetry probes injected: %d system qubits → %d total, "
            "angle=%.4f rad",
            n_system,
            probed.num_qubits,
            self._config.probe_angle,
        )

        return probed

    # ------------------------------------------------------------------
    # Internal — chaotic Hamiltonian padding
    # ------------------------------------------------------------------

    def _inject_chaotic_padding(self, circuit: Any) -> Any:
        """Inject pseudo-random mixing gates into *circuit*.

        Appends ``mixing_depth`` layers of:

        * Random single-qubit rotations Rz(θ_i) on every system qubit.
        * Nearest-neighbour CX (CNOT) entangling gates in a brick-wall
          pattern (even pairs on odd layers, odd pairs on even layers).

        This decorrelates successive shots and approximates the chaotic
        Hamiltonian mixing required by the legacy binary trajectory filter.
        The mixing gates are **not needed** when ML mitigators extrapolate
        corrections from telemetry features.

        .. note::
           When :attr:`QgateTranspilerConfig.aggressive_mixing` is ``False``
           (i.e. ML mode), this method is **never called**.  This avoids
           unnecessary depth amplification and preserves the shallow circuit
           profile that maximises raw measurement fidelity for the
           regression model.

        Args:
            circuit: The circuit (already probe-augmented).

        Returns:
            The same circuit with chaotic padding layers appended
            (modified in-place and returned for chaining).
        """
        _require_qiskit()

        rng = np.random.default_rng(self._config.mixing_seed)

        # Operate on all qubits except the probe ancilla (last qubit)
        n_total = circuit.num_qubits
        # The last qubit is the probe ancilla — don't pad it
        n_system = n_total - 1
        if n_system < 1:
            return circuit

        system_qubits = circuit.qubits[:n_system]

        for layer in range(self._config.mixing_depth):
            # Single-qubit random Rz rotations
            for q in system_qubits:
                angle = float(rng.uniform(0, 2 * np.pi))
                circuit.rz(angle, q)

            # Brick-wall CX entangling pattern
            # Even layers: pairs (0,1), (2,3), (4,5), ...
            # Odd layers:  pairs (1,2), (3,4), (5,6), ...
            start = layer % 2
            for i in range(start, n_system - 1, 2):
                circuit.cx(system_qubits[i], system_qubits[i + 1])

        logger.debug(
            "Chaotic padding: %d layers, %d system qubits, seed=%d",
            self._config.mixing_depth,
            n_system,
            self._config.mixing_seed,
        )

        return circuit

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QgateTranspiler("
            f"mode={self._config.mitigation_mode!r}, "
            f"mixing={self._config.aggressive_mixing}, "
            f"oversample={self._config.oversampling_factor}×)"
        )
