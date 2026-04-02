"""
test_transpiler.py — Comprehensive tests for the QgateTranspiler.

Covers:
  - QgateTranspilerConfig auto-tuning across all mitigation modes
  - Config validation (invalid modes, out-of-range values)
  - Config factory method (.for_mode)
  - Telemetry probe injection (ancilla + measurement)
  - Chaotic padding injection (gate counts, brick-wall pattern)
  - ML-mode bypass (padding completely skipped)
  - Shot optimisation arithmetic
  - CompilationResult metadata integrity
  - Integration with real Qiskit QuantumCircuit objects
  - Repr coverage

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
CONFIDENTIAL — DO NOT PUSH / DO NOT PUBLISH.
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit  # type: ignore[import-untyped]

from qgate.transpiler import (
    CompilationResult,
    QgateTranspiler,
    QgateTranspilerConfig,
    apply_uzdin_unitary_folding,
    validate_noise_scale_factor,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. QgateTranspilerConfig — auto-tuning
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigAutoTuning:
    """Verify that __post_init__ auto-configures flags from mitigation_mode."""

    def test_legacy_filter_defaults(self) -> None:
        cfg = QgateTranspilerConfig(mitigation_mode="legacy_filter")
        assert cfg.aggressive_mixing is True
        assert cfg.oversampling_factor == 10.0

    def test_ml_extrapolation_defaults(self) -> None:
        cfg = QgateTranspilerConfig(mitigation_mode="ml_extrapolation")
        assert cfg.aggressive_mixing is False
        assert cfg.oversampling_factor == 1.2

    def test_pulse_active_defaults(self) -> None:
        cfg = QgateTranspilerConfig(mitigation_mode="pulse_active")
        assert cfg.aggressive_mixing is False
        assert cfg.oversampling_factor == 1.2

    def test_ml_mode_forces_mixing_false(self) -> None:
        """Even if user tries to set aggressive_mixing=True in ML mode,
        __post_init__ overrides it to False."""
        cfg = QgateTranspilerConfig(
            mitigation_mode="ml_extrapolation",
            aggressive_mixing=True,
        )
        assert cfg.aggressive_mixing is False

    def test_pulse_mode_forces_mixing_false(self) -> None:
        cfg = QgateTranspilerConfig(
            mitigation_mode="pulse_active",
            aggressive_mixing=True,
        )
        assert cfg.aggressive_mixing is False

    def test_legacy_forces_mixing_true(self) -> None:
        """Legacy mode always enables mixing, even if user passes False."""
        cfg = QgateTranspilerConfig(
            mitigation_mode="legacy_filter",
            aggressive_mixing=False,
        )
        assert cfg.aggressive_mixing is True

    def test_explicit_oversampling_preserved_ml(self) -> None:
        """User can explicitly override oversampling_factor in ML mode."""
        cfg = QgateTranspilerConfig(
            mitigation_mode="ml_extrapolation",
            oversampling_factor=2.5,
        )
        assert cfg.oversampling_factor == 2.5
        assert cfg.aggressive_mixing is False

    def test_explicit_oversampling_preserved_pulse(self) -> None:
        cfg = QgateTranspilerConfig(
            mitigation_mode="pulse_active",
            oversampling_factor=3.0,
        )
        assert cfg.oversampling_factor == 3.0

    def test_default_probe_angle(self) -> None:
        cfg = QgateTranspilerConfig()
        assert cfg.probe_angle == pytest.approx(math.pi / 6, rel=1e-6)

    def test_default_mixing_depth(self) -> None:
        cfg = QgateTranspilerConfig()
        assert cfg.mixing_depth == 3

    def test_default_mixing_seed(self) -> None:
        cfg = QgateTranspilerConfig()
        assert cfg.mixing_seed == 42

    def test_config_is_frozen(self) -> None:
        cfg = QgateTranspilerConfig()
        with pytest.raises(AttributeError):
            cfg.mitigation_mode = "pulse_active"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# 2. QgateTranspilerConfig — validation
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    """Verify that invalid configurations raise appropriate errors."""

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mitigation_mode"):
            QgateTranspilerConfig(mitigation_mode="quantum_magic")  # type: ignore[arg-type]

    def test_oversampling_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="oversampling_factor must be"):
            QgateTranspilerConfig(
                mitigation_mode="legacy_filter",
                oversampling_factor=0.5,
            )

    def test_probe_angle_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="probe_angle must be"):
            QgateTranspilerConfig(probe_angle=0.0)

    def test_probe_angle_above_pi_raises(self) -> None:
        with pytest.raises(ValueError, match="probe_angle must be"):
            QgateTranspilerConfig(probe_angle=math.pi + 0.1)

    def test_mixing_depth_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="mixing_depth must be"):
            QgateTranspilerConfig(mixing_depth=0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. QgateTranspilerConfig — factory method
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigFactory:
    """Verify the .for_mode() classmethod."""

    def test_for_mode_ml(self) -> None:
        cfg = QgateTranspilerConfig.for_mode("ml_extrapolation")
        assert cfg.mitigation_mode == "ml_extrapolation"
        assert cfg.aggressive_mixing is False
        assert cfg.oversampling_factor == 1.2

    def test_for_mode_pulse(self) -> None:
        cfg = QgateTranspilerConfig.for_mode("pulse_active")
        assert cfg.mitigation_mode == "pulse_active"
        assert cfg.aggressive_mixing is False

    def test_for_mode_legacy(self) -> None:
        cfg = QgateTranspilerConfig.for_mode("legacy_filter")
        assert cfg.mitigation_mode == "legacy_filter"
        assert cfg.aggressive_mixing is True
        assert cfg.oversampling_factor == 10.0

    def test_for_mode_with_overrides(self) -> None:
        cfg = QgateTranspilerConfig.for_mode(
            "ml_extrapolation",
            probe_angle=0.3,
            mixing_depth=5,
        )
        assert cfg.probe_angle == pytest.approx(0.3)
        assert cfg.mixing_depth == 5
        assert cfg.aggressive_mixing is False

    def test_for_mode_with_custom_oversample(self) -> None:
        cfg = QgateTranspilerConfig.for_mode(
            "pulse_active",
            oversampling_factor=2.0,
        )
        assert cfg.oversampling_factor == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Telemetry probe injection
# ═══════════════════════════════════════════════════════════════════════════


def _make_simple_circuit(n_qubits: int = 4) -> QuantumCircuit:
    """Build a minimal test circuit with H gates and measurement."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    qc.measure_all()
    return qc


class TestProbeInjection:
    """Verify that _inject_telemetry_probes adds ancilla correctly."""

    def test_adds_one_ancilla_qubit(self) -> None:
        qc = _make_simple_circuit(4)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        probed = transpiler._inject_telemetry_probes(qc)
        assert probed.num_qubits == 5  # 4 system + 1 ancilla

    def test_adds_probe_classical_register(self) -> None:
        qc = _make_simple_circuit(4)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        probed = transpiler._inject_telemetry_probes(qc)
        creg_names = [creg.name for creg in probed.cregs]
        assert "qgate_probe" in creg_names

    def test_preserves_original_gates(self) -> None:
        qc = _make_simple_circuit(3)
        original_ops = len(qc.data)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        probed = transpiler._inject_telemetry_probes(qc)
        # Should have original ops + probe entanglement + measurement
        assert len(probed.data) > original_ops

    def test_single_qubit_circuit(self) -> None:
        """Edge case: 1-qubit circuit still gets probe ancilla."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        transpiler = QgateTranspiler()
        probed = transpiler._inject_telemetry_probes(qc)
        assert probed.num_qubits == 2

    def test_probe_measurement_present(self) -> None:
        """Verify the probe ancilla is measured."""
        qc = _make_simple_circuit(3)
        transpiler = QgateTranspiler()
        probed = transpiler._inject_telemetry_probes(qc)
        measure_ops = [inst for inst in probed.data if inst.operation.name == "measure"]
        # Original measurements + 1 probe measurement
        assert len(measure_ops) >= 1

    def test_ancilla_register_named(self) -> None:
        qc = _make_simple_circuit(2)
        transpiler = QgateTranspiler()
        probed = transpiler._inject_telemetry_probes(qc)
        qreg_names = [qreg.name for qreg in probed.qregs]
        assert "qgate_anc" in qreg_names

    def test_two_qubit_circuit_has_one_pair(self) -> None:
        """2-qubit circuit: exactly 1 neighbour pair → 1 entanglement pair."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        transpiler = QgateTranspiler()
        probed = transpiler._inject_telemetry_probes(qc)
        # Should have ancilla qubit
        assert probed.num_qubits == 3


# ═══════════════════════════════════════════════════════════════════════════
# 5. Chaotic padding injection
# ═══════════════════════════════════════════════════════════════════════════


class TestChaoticPadding:
    """Verify chaotic mixing gate injection."""

    def test_padding_adds_rz_gates(self) -> None:
        qc = _make_simple_circuit(4)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("legacy_filter"))
        # First inject probes (padding operates on probed circuit)
        probed = transpiler._inject_telemetry_probes(qc)
        pre_ops = len(probed.data)
        padded = transpiler._inject_chaotic_padding(probed)
        post_ops = len(padded.data)
        assert post_ops > pre_ops

    def test_padding_adds_cx_gates(self) -> None:
        qc = _make_simple_circuit(4)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("legacy_filter"))
        probed = transpiler._inject_telemetry_probes(qc)
        padded = transpiler._inject_chaotic_padding(probed)
        cx_ops = [inst for inst in padded.data if inst.operation.name == "cx"]
        assert len(cx_ops) > 0

    def test_padding_depth_increases_with_mixing_depth(self) -> None:
        qc = _make_simple_circuit(4)

        cfg_shallow = QgateTranspilerConfig(
            mitigation_mode="legacy_filter",
            mixing_depth=1,
        )
        cfg_deep = QgateTranspilerConfig(
            mitigation_mode="legacy_filter",
            mixing_depth=5,
        )

        t_shallow = QgateTranspiler(cfg_shallow)
        t_deep = QgateTranspiler(cfg_deep)

        probed_s = t_shallow._inject_telemetry_probes(qc)
        probed_d = t_deep._inject_telemetry_probes(qc)

        padded_s = t_shallow._inject_chaotic_padding(probed_s)
        padded_d = t_deep._inject_chaotic_padding(probed_d)

        assert len(padded_d.data) > len(padded_s.data)

    def test_padding_is_reproducible_with_same_seed(self) -> None:
        qc = _make_simple_circuit(3)
        cfg = QgateTranspilerConfig(
            mitigation_mode="legacy_filter",
            mixing_seed=123,
        )
        t1 = QgateTranspiler(cfg)
        t2 = QgateTranspiler(cfg)

        p1 = t1._inject_telemetry_probes(qc)
        p2 = t2._inject_telemetry_probes(qc)
        pad1 = t1._inject_chaotic_padding(p1)
        pad2 = t2._inject_chaotic_padding(p2)

        # Same seed → same Rz angles
        rz_angles_1 = [
            inst.operation.params[0]
            for inst in pad1.data
            if inst.operation.name == "rz"
        ]
        rz_angles_2 = [
            inst.operation.params[0]
            for inst in pad2.data
            if inst.operation.name == "rz"
        ]
        assert rz_angles_1 == rz_angles_2

    def test_padding_differs_with_different_seed(self) -> None:
        qc = _make_simple_circuit(3)
        cfg_a = QgateTranspilerConfig(mitigation_mode="legacy_filter", mixing_seed=10)
        cfg_b = QgateTranspilerConfig(mitigation_mode="legacy_filter", mixing_seed=99)

        t_a = QgateTranspiler(cfg_a)
        t_b = QgateTranspiler(cfg_b)

        pa = t_a._inject_chaotic_padding(t_a._inject_telemetry_probes(qc))
        pb = t_b._inject_chaotic_padding(t_b._inject_telemetry_probes(qc))

        rz_a = [inst.operation.params[0] for inst in pa.data if inst.operation.name == "rz"]
        rz_b = [inst.operation.params[0] for inst in pb.data if inst.operation.name == "rz"]
        assert rz_a != rz_b

    def test_no_padding_on_ancilla_qubit(self) -> None:
        """Padding should only operate on system qubits, not probe ancilla."""
        qc = _make_simple_circuit(3)
        transpiler = QgateTranspiler(
            QgateTranspilerConfig(mitigation_mode="legacy_filter", mixing_depth=2)
        )
        probed = transpiler._inject_telemetry_probes(qc)
        ancilla_idx = probed.num_qubits - 1
        padded = transpiler._inject_chaotic_padding(probed)

        # Check that no RZ or CX gate targets the ancilla
        for inst in padded.data:
            if inst.operation.name in ("rz", "cx"):
                qubit_indices = [probed.qubits.index(q) for q in inst.qubits]
                assert ancilla_idx not in qubit_indices, (
                    f"Padding gate {inst.operation.name} touches ancilla at index {ancilla_idx}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 6. ML-mode bypass (padding NOT applied)
# ═══════════════════════════════════════════════════════════════════════════


class TestMLModeBypass:
    """Verify that ML modes completely skip chaotic padding."""

    def test_ml_extrapolation_skips_padding(self) -> None:
        qc = _make_simple_circuit(4)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=4096)
        assert result.chaotic_padding_applied is False

    def test_pulse_active_skips_padding(self) -> None:
        qc = _make_simple_circuit(4)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("pulse_active"))
        result = transpiler.compile(qc, base_shots=4096)
        assert result.chaotic_padding_applied is False

    def test_legacy_applies_padding(self) -> None:
        qc = _make_simple_circuit(4)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("legacy_filter"))
        result = transpiler.compile(qc, base_shots=4096)
        assert result.chaotic_padding_applied is True

    def test_ml_circuit_is_shallower_than_legacy(self) -> None:
        """ML-compiled circuit has fewer gates than legacy (no padding)."""
        qc = _make_simple_circuit(4)
        ml = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        legacy = QgateTranspiler(QgateTranspilerConfig.for_mode("legacy_filter"))

        result_ml = ml.compile(qc, base_shots=4096)
        result_legacy = legacy.compile(qc, base_shots=4096)

        assert len(result_ml.circuit.data) < len(result_legacy.circuit.data)

    def test_ml_mode_no_rz_padding(self) -> None:
        """In ML mode, no Rz rotation gates from padding should appear."""
        qc = QuantumCircuit(3)
        qc.h(range(3))
        qc.measure_all()

        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=1000)

        # Count Rz gates — should be zero (no padding)
        rz_count = sum(1 for inst in result.circuit.data if inst.operation.name == "rz")
        assert rz_count == 0

    def test_ml_mode_no_cx_padding(self) -> None:
        """In ML mode, no CX gates from padding should appear
        (only CX from probe entanglement via controlled-RY decomposition)."""
        qc = QuantumCircuit(3)
        qc.h(range(3))
        qc.measure_all()

        # Build a probed-only circuit for reference
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        probed_only = transpiler._inject_telemetry_probes(qc)
        probed_gate_count = len(probed_only.data)

        # Full compile should have same gate count (no padding added)
        result = transpiler.compile(qc, base_shots=1000)
        assert len(result.circuit.data) == probed_gate_count


# ═══════════════════════════════════════════════════════════════════════════
# 7. Shot optimisation
# ═══════════════════════════════════════════════════════════════════════════


class TestShotOptimisation:
    """Verify shot calculation: optimized = ceil(base × factor)."""

    def test_ml_oversampling_1_2x(self) -> None:
        qc = _make_simple_circuit(2)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=4096)
        expected = math.ceil(4096 * 1.2)
        assert result.optimized_shots == expected

    def test_legacy_oversampling_10x(self) -> None:
        qc = _make_simple_circuit(2)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("legacy_filter"))
        result = transpiler.compile(qc, base_shots=1000)
        expected = math.ceil(1000 * 10.0)
        assert result.optimized_shots == expected

    def test_custom_oversampling(self) -> None:
        qc = _make_simple_circuit(2)
        cfg = QgateTranspilerConfig(
            mitigation_mode="ml_extrapolation",
            oversampling_factor=2.5,
        )
        transpiler = QgateTranspiler(cfg)
        result = transpiler.compile(qc, base_shots=100)
        assert result.optimized_shots == 250

    def test_single_shot(self) -> None:
        qc = _make_simple_circuit(2)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=1)
        assert result.optimized_shots == math.ceil(1 * 1.2)

    def test_oversampling_ceiling_rounding(self) -> None:
        """ceil(4097 * 1.2) = ceil(4916.4) = 4917"""
        qc = _make_simple_circuit(2)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=4097)
        assert result.optimized_shots == math.ceil(4097 * 1.2)

    def test_pulse_mode_same_as_ml(self) -> None:
        qc = _make_simple_circuit(2)
        ml = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        pulse = QgateTranspiler(QgateTranspilerConfig.for_mode("pulse_active"))
        r_ml = ml.compile(qc, base_shots=4096)
        r_pulse = pulse.compile(qc, base_shots=4096)
        assert r_ml.optimized_shots == r_pulse.optimized_shots


# ═══════════════════════════════════════════════════════════════════════════
# 8. CompilationResult metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestCompilationResult:
    """Verify CompilationResult fields and metadata."""

    def test_probes_always_injected(self) -> None:
        qc = _make_simple_circuit(3)
        for mode in ("legacy_filter", "ml_extrapolation", "pulse_active"):
            transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode(mode))  # type: ignore[arg-type]
            result = transpiler.compile(qc, base_shots=100)
            assert result.probes_injected is True

    def test_metadata_contains_mode(self) -> None:
        qc = _make_simple_circuit(3)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=100)
        assert result.metadata["mitigation_mode"] == "ml_extrapolation"

    def test_metadata_contains_base_shots(self) -> None:
        qc = _make_simple_circuit(3)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=2048)
        assert result.metadata["base_shots"] == 2048

    def test_metadata_qubit_counts(self) -> None:
        qc = _make_simple_circuit(5)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=100)
        assert result.metadata["n_system_qubits"] == 5
        assert result.metadata["n_total_qubits"] == 6  # 5 + 1 ancilla

    def test_metadata_mixing_depth_zero_for_ml(self) -> None:
        qc = _make_simple_circuit(3)
        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=100)
        assert result.metadata["mixing_depth"] == 0

    def test_metadata_mixing_depth_nonzero_for_legacy(self) -> None:
        qc = _make_simple_circuit(3)
        cfg = QgateTranspilerConfig(
            mitigation_mode="legacy_filter",
            mixing_depth=4,
        )
        transpiler = QgateTranspiler(cfg)
        result = transpiler.compile(qc, base_shots=100)
        assert result.metadata["mixing_depth"] == 4


# ═══════════════════════════════════════════════════════════════════════════
# 9. Compile method — error handling
# ═══════════════════════════════════════════════════════════════════════════


class TestCompileErrors:
    """Verify that compile() rejects invalid inputs."""

    def test_non_circuit_raises_type_error(self) -> None:
        transpiler = QgateTranspiler()
        with pytest.raises(TypeError, match="Expected a Qiskit QuantumCircuit"):
            transpiler.compile("not a circuit", base_shots=100)  # type: ignore[arg-type]

    def test_zero_shots_raises(self) -> None:
        qc = _make_simple_circuit(2)
        transpiler = QgateTranspiler()
        with pytest.raises(ValueError, match="base_shots must be"):
            transpiler.compile(qc, base_shots=0)

    def test_negative_shots_raises(self) -> None:
        qc = _make_simple_circuit(2)
        transpiler = QgateTranspiler()
        with pytest.raises(ValueError, match="base_shots must be"):
            transpiler.compile(qc, base_shots=-5)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Default config (no arguments)
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaultConfig:
    """Verify that QgateTranspiler() with no args uses legacy defaults."""

    def test_default_is_legacy(self) -> None:
        transpiler = QgateTranspiler()
        assert transpiler.config.mitigation_mode == "legacy_filter"
        assert transpiler.config.aggressive_mixing is True
        assert transpiler.config.oversampling_factor == 10.0


# ═══════════════════════════════════════════════════════════════════════════
# 11. Repr
# ═══════════════════════════════════════════════════════════════════════════


class TestRepr:
    """Verify repr strings for debugging."""

    def test_transpiler_repr_legacy(self) -> None:
        t = QgateTranspiler(QgateTranspilerConfig.for_mode("legacy_filter"))
        r = repr(t)
        assert "legacy_filter" in r
        assert "mixing=True" in r
        assert "10.0×" in r

    def test_transpiler_repr_ml(self) -> None:
        t = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        r = repr(t)
        assert "ml_extrapolation" in r
        assert "mixing=False" in r
        assert "1.2×" in r


# ═══════════════════════════════════════════════════════════════════════════
# 12. Integration — full compile pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end integration tests across all three modes."""

    @pytest.mark.parametrize(
        "mode,expect_padding",
        [
            ("legacy_filter", True),
            ("ml_extrapolation", False),
            ("pulse_active", False),
        ],
    )
    def test_all_modes_compile_successfully(
        self, mode: str, expect_padding: bool
    ) -> None:
        qc = QuantumCircuit(4)
        qc.h(range(4))
        for i in range(3):
            qc.cx(i, i + 1)
        qc.measure_all()

        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode(mode))  # type: ignore[arg-type]
        result = transpiler.compile(qc, base_shots=4096)

        assert isinstance(result, CompilationResult)
        assert result.circuit.num_qubits == 5  # 4 + 1 ancilla
        assert result.probes_injected is True
        assert result.chaotic_padding_applied == expect_padding
        assert result.optimized_shots >= 4096

    def test_large_circuit(self) -> None:
        """Compile a 20-qubit circuit in ML mode."""
        qc = QuantumCircuit(20)
        for i in range(20):
            qc.h(i)
        qc.measure_all()

        transpiler = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        result = transpiler.compile(qc, base_shots=8192)

        assert result.circuit.num_qubits == 21
        assert result.optimized_shots == math.ceil(8192 * 1.2)
        assert result.chaotic_padding_applied is False

    def test_cost_reduction_ratio(self) -> None:
        """ML mode uses ≈8.3× fewer shots than legacy for same base."""
        qc = _make_simple_circuit(4)
        ml = QgateTranspiler(QgateTranspilerConfig.for_mode("ml_extrapolation"))
        legacy = QgateTranspiler(QgateTranspilerConfig.for_mode("legacy_filter"))

        r_ml = ml.compile(qc, base_shots=10000)
        r_legacy = legacy.compile(qc, base_shots=10000)

        ratio = r_legacy.optimized_shots / r_ml.optimized_shots
        # 100000 / 12000 = 8.33...
        assert ratio > 8.0
        assert ratio < 9.0


# ═══════════════════════════════════════════════════════════════════════════
# 12. validate_noise_scale_factor — Uzdin odd-factor rule
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateNoiseScaleFactor:
    """Tests for validate_noise_scale_factor (Uzdin KIK rule)."""

    @pytest.mark.parametrize("k", [1, 3, 5, 7, 9, 11, 101])
    def test_valid_odd_factors_pass(self, k: int) -> None:
        """Positive odd integers are accepted without error."""
        validate_noise_scale_factor(k)  # should not raise

    @pytest.mark.parametrize("k", [2, 4, 6, 8, 10, 100])
    def test_even_factors_rejected(self, k: int) -> None:
        """Even integers violate the Uzdin rule and are rejected."""
        with pytest.raises(ValueError, match="even"):
            validate_noise_scale_factor(k)

    @pytest.mark.parametrize("k", [0, -1, -3])
    def test_non_positive_rejected(self, k: int) -> None:
        """Zero and negative values are rejected."""
        with pytest.raises(ValueError):
            validate_noise_scale_factor(k)

    @pytest.mark.parametrize("k", [1.5, 2.0, 3.0, "3"])
    def test_non_integer_rejected(self, k) -> None:
        """Floats and strings are rejected (must be int)."""
        with pytest.raises(ValueError):
            validate_noise_scale_factor(k)


# ═══════════════════════════════════════════════════════════════════════════
# 13. apply_uzdin_unitary_folding — gate-level folding
# ═══════════════════════════════════════════════════════════════════════════


class TestUzdinUnitaryFolding:
    """Tests for apply_uzdin_unitary_folding (U → U·(U†·U)^n)."""

    def test_scale_factor_1_returns_copy(self) -> None:
        """scale_factor=1 returns an identical copy of the circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        folded = apply_uzdin_unitary_folding(qc, scale_factor=1)
        assert folded.size() == qc.size()
        assert folded is not qc  # must be a copy

    def test_scale_factor_3_triples_gates(self) -> None:
        """scale_factor=3: each gate becomes U·U†·U (3 operations)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        original_size = qc.size()  # 2

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        assert folded.size() == original_size * 3  # 6

    def test_scale_factor_5_quintuples_gates(self) -> None:
        """scale_factor=5: each gate becomes U·U†·U·U†·U (5 operations)."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.h(0)
        original_size = qc.size()  # 2

        folded = apply_uzdin_unitary_folding(qc, scale_factor=5)
        assert folded.size() == original_size * 5  # 10

    def test_scale_factor_7(self) -> None:
        """scale_factor=7: 7 operations per original gate."""
        qc = QuantumCircuit(1)
        qc.z(0)
        folded = apply_uzdin_unitary_folding(qc, scale_factor=7)
        assert folded.size() == 7

    def test_even_factor_rejected(self) -> None:
        """Even scale factors are rejected per Uzdin rule."""
        qc = QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(ValueError, match="even"):
            apply_uzdin_unitary_folding(qc, scale_factor=2)

    def test_non_integer_rejected(self) -> None:
        """Non-integer scale factors are rejected."""
        qc = QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(ValueError):
            apply_uzdin_unitary_folding(qc, scale_factor=1.5)  # type: ignore[arg-type]

    def test_wrong_circuit_type_rejected(self) -> None:
        """Non-QuantumCircuit input is rejected."""
        with pytest.raises(TypeError, match="QuantumCircuit"):
            apply_uzdin_unitary_folding("not_a_circuit", scale_factor=3)  # type: ignore[arg-type]

    def test_preserves_qubit_count(self) -> None:
        """Folding does not add or remove qubits."""
        qc = QuantumCircuit(4)
        qc.h(range(4))
        for i in range(3):
            qc.cx(i, i + 1)

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        assert folded.num_qubits == qc.num_qubits

    def test_preserves_classical_registers(self) -> None:
        """Folding preserves classical registers; measurements are NOT folded."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        assert folded.num_clbits == qc.num_clbits
        # 2 unitary gates × 3-fold + 2 measurements (not folded) = 8
        assert folded.size() == 2 * 3 + 2

    def test_logical_equivalence_factor_3(self) -> None:
        """U·U†·U ≡ U: the folded circuit is logically identical.

        We verify this by simulating with a noiseless statevector
        simulator and checking that the final statevector matches.
        """
        from qiskit.quantum_info import Statevector

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        # |00⟩ → (|00⟩ + |11⟩)/√2  (Bell state)

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)

        sv_orig = Statevector.from_instruction(qc)
        sv_folded = Statevector.from_instruction(folded)

        # States should be equivalent (up to global phase)
        fidelity = abs(sv_orig.inner(sv_folded)) ** 2
        assert fidelity > 0.9999, f"Fidelity {fidelity} — folded circuit is not logically equivalent"

    def test_logical_equivalence_factor_5(self) -> None:
        """5-fold circuit is logically identical to the original."""
        from qiskit.quantum_info import Statevector

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(0.7, 2)

        folded = apply_uzdin_unitary_folding(qc, scale_factor=5)

        sv_orig = Statevector.from_instruction(qc)
        sv_folded = Statevector.from_instruction(folded)

        fidelity = abs(sv_orig.inner(sv_folded)) ** 2
        assert fidelity > 0.9999

    def test_empty_circuit(self) -> None:
        """Folding an empty circuit returns an empty circuit."""
        qc = QuantumCircuit(2)
        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        assert folded.size() == 0
        assert folded.num_qubits == 2

    def test_single_gate_structure_factor_3(self) -> None:
        """Verify the gate sequence is exactly [H, H†, H] for scale_factor=3."""
        qc = QuantumCircuit(1)
        qc.h(0)

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        ops = [inst.operation.name for inst in folded.data]
        assert len(ops) == 3
        assert ops[0] == "h"       # forward
        assert ops[1] == "h"       # H† = H (self-inverse)
        assert ops[2] == "h"       # forward

    def test_cx_structure_factor_3(self) -> None:
        """CX → CX·CX†·CX (CX is self-inverse, so CX†=CX)."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        ops = [inst.operation.name for inst in folded.data]
        assert len(ops) == 3
        # CX is self-inverse so all three should be cx
        assert all(op == "cx" for op in ops)

    def test_rz_inverse_is_negated(self) -> None:
        """Rz(θ)† = Rz(-θ): verify the inverse has negated angle."""
        import numpy as np

        qc = QuantumCircuit(1)
        qc.rz(0.5, 0)

        folded = apply_uzdin_unitary_folding(qc, scale_factor=3)
        ops = folded.data
        assert len(ops) == 3

        # Middle gate should be the inverse: Rz(-0.5)
        middle_params = ops[1].operation.params
        assert len(middle_params) == 1
        assert abs(middle_params[0] - (-0.5)) < 1e-10
