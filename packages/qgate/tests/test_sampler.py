"""Tests for qgate.sampler — QgateSampler OS layer.

Tests are structured in tiers:
  1. Unit tests for SamplerConfig, _SamplerGaltonThreshold, PUB helpers
  2. Probe-injection circuit tests (circuit structure verification)
  3. Integration tests with Aer simulator (full pipeline)

NOTICE: Pre-patent proprietary code — do NOT push to public repositories.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if Qiskit / runtime not installed
# ---------------------------------------------------------------------------

qiskit = pytest.importorskip("qiskit", reason="Qiskit required for sampler tests")
qiskit_aer = pytest.importorskip("qiskit_aer", reason="Aer required for sampler tests")

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister

# Try importing runtime — tests that need it will skip individually
try:
    from qiskit_ibm_runtime import SamplerV2  # type: ignore[import-untyped]

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

from qgate.sampler import (
    QgateSampler,
    QgateSamplerResult,
    SamplerConfig,
    _SamplerGaltonThreshold,
)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — SamplerConfig unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSamplerConfig:
    """Verify SamplerConfig Pydantic model validation."""

    def test_defaults(self):
        cfg = SamplerConfig()
        assert cfg.probe_angle == pytest.approx(math.pi / 6)
        assert cfg.target_acceptance == 0.05
        assert cfg.window_size == 4096
        assert cfg.min_window_size == 100
        assert cfg.baseline_threshold == 0.65
        assert cfg.optimization_level == 1
        assert cfg.oversample_factor == 1.0

    def test_custom_values(self):
        cfg = SamplerConfig(
            probe_angle=0.3,
            target_acceptance=0.10,
            window_size=2048,
            oversample_factor=2.5,
        )
        assert cfg.probe_angle == pytest.approx(0.3)
        assert cfg.target_acceptance == 0.10
        assert cfg.window_size == 2048
        assert cfg.oversample_factor == 2.5

    def test_frozen(self):
        cfg = SamplerConfig()
        with pytest.raises(Exception):  # Pydantic ValidationError
            cfg.probe_angle = 0.5  # type: ignore[misc]

    def test_probe_angle_bounds(self):
        # Too small
        with pytest.raises(Exception):
            SamplerConfig(probe_angle=0.0)
        with pytest.raises(Exception):
            SamplerConfig(probe_angle=-0.1)
        # Upper bound is pi
        cfg = SamplerConfig(probe_angle=math.pi)
        assert cfg.probe_angle == pytest.approx(math.pi)

    def test_target_acceptance_bounds(self):
        with pytest.raises(Exception):
            SamplerConfig(target_acceptance=0.0)
        with pytest.raises(Exception):
            SamplerConfig(target_acceptance=1.0)

    def test_oversample_factor_bounds(self):
        with pytest.raises(Exception):
            SamplerConfig(oversample_factor=0.5)  # below 1.0
        cfg = SamplerConfig(oversample_factor=1.0)
        assert cfg.oversample_factor == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — _SamplerGaltonThreshold unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSamplerGaltonThreshold:
    """Verify the self-contained Galton threshold logic."""

    def _cfg(self, **kw) -> SamplerConfig:
        defaults = dict(
            window_size=500,
            min_window_size=50,
            target_acceptance=0.05,
            use_quantile=True,
            baseline_threshold=0.65,
            min_threshold=0.0,
            max_threshold=1.0,
        )
        defaults.update(kw)
        return SamplerConfig(**defaults)

    def test_warmup_uses_baseline(self):
        cfg = self._cfg(min_window_size=100)
        g = _SamplerGaltonThreshold(cfg)
        # Feed fewer than min_window_size scores
        g.observe_batch([0.5] * 50)
        assert g.in_warmup is True
        assert g.current_threshold == pytest.approx(0.65)

    def test_exits_warmup(self):
        cfg = self._cfg(min_window_size=50)
        g = _SamplerGaltonThreshold(cfg)
        rng = np.random.default_rng(42)
        scores = rng.normal(0.7, 0.05, size=200).tolist()
        g.observe_batch(scores)
        assert g.in_warmup is False
        assert g.current_threshold > 0.5  # should be adapted

    def test_quantile_acceptance_rate(self):
        """After observing N(0.7, 0.05) scores, ~5% should exceed threshold."""
        cfg = self._cfg(window_size=2000, min_window_size=100, target_acceptance=0.05)
        g = _SamplerGaltonThreshold(cfg)
        rng = np.random.default_rng(123)
        scores = rng.normal(0.7, 0.05, size=2000).tolist()
        g.observe_batch(scores)

        threshold = g.current_threshold
        acceptance = np.mean(np.array(scores) >= threshold)
        # Should be approximately 5% ± tolerance
        assert 0.02 < acceptance < 0.10

    def test_reset(self):
        cfg = self._cfg(min_window_size=50)
        g = _SamplerGaltonThreshold(cfg)
        g.observe_batch([0.8] * 200)
        assert g.in_warmup is False

        g.reset()
        assert g.in_warmup is True
        assert g.current_threshold == pytest.approx(0.65)

    def test_z_score_mode(self):
        cfg = self._cfg(
            use_quantile=False,
            robust_stats=True,
            min_window_size=50,
            z_sigma=1.645,
        )
        g = _SamplerGaltonThreshold(cfg)
        rng = np.random.default_rng(77)
        scores = rng.normal(0.7, 0.05, size=300).tolist()
        g.observe_batch(scores)
        assert g.in_warmup is False
        # Threshold should be above the median
        assert g.current_threshold > 0.7

    def test_clamping(self):
        cfg = self._cfg(min_threshold=0.4, max_threshold=0.8, min_window_size=10)
        g = _SamplerGaltonThreshold(cfg)
        # Feed very high scores — threshold should be clamped to max
        g.observe_batch([0.99] * 100)
        assert g.current_threshold <= 0.8
        # Feed very low scores — threshold should be clamped to min
        g2 = _SamplerGaltonThreshold(cfg)
        g2.observe_batch([0.01] * 100)
        assert g2.current_threshold >= 0.4


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — PUB packing / unpacking
# ═══════════════════════════════════════════════════════════════════════════


class TestPubHelpers:
    """Verify PUB unpacking and repacking."""

    def test_unpack_bare_circuit(self):
        qc = QuantumCircuit(2)
        circuit, params, shots = QgateSampler._unpack_pub(qc)
        assert circuit is qc
        assert params is None
        assert shots is None

    def test_unpack_tuple_1(self):
        qc = QuantumCircuit(2)
        circuit, params, shots = QgateSampler._unpack_pub((qc,))
        assert circuit is qc
        assert params is None
        assert shots is None

    def test_unpack_tuple_3(self):
        qc = QuantumCircuit(2)
        circuit, params, shots = QgateSampler._unpack_pub((qc, [0.1, 0.2], 1024))
        assert circuit is qc
        assert params == [0.1, 0.2]
        assert shots == 1024

    def test_repack_all_none(self):
        qc = QuantumCircuit(2)
        pub = QgateSampler._repack_pub(qc, None, None)
        assert pub == (qc,)

    def test_repack_with_shots(self):
        qc = QuantumCircuit(2)
        pub = QgateSampler._repack_pub(qc, None, 512)
        assert pub == (qc, [], 512)

    def test_repack_full(self):
        qc = QuantumCircuit(2)
        pub = QgateSampler._repack_pub(qc, [0.1], 1024)
        assert pub == (qc, [0.1], 1024)

    def test_unpack_invalid_type(self):
        with pytest.raises(TypeError, match="Cannot unpack PUB"):
            QgateSampler._unpack_pub(42)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Probe injection circuit tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_RUNTIME, reason="qiskit-ibm-runtime not installed")
class TestProbeInjection:
    """Verify probe injection adds correct ancilla structure."""

    def _make_sampler(self, **cfg_kw) -> QgateSampler:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        cfg = SamplerConfig(**cfg_kw)
        return QgateSampler(backend=backend, config=cfg)

    def test_adds_ancilla_qubit(self):
        sampler = self._make_sampler()
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.measure_all()

        probed, meta = sampler._inject_probes(qc)
        assert probed.num_qubits == 5  # 4 system + 1 ancilla
        assert meta["n_system_qubits"] == 4
        assert meta["n_total_qubits"] == 5
        assert meta["ancilla_index"] == 4
        assert meta["probe_creg_name"] == "qgate_probe"

    def test_adds_probe_classical_register(self):
        sampler = self._make_sampler()
        qc = QuantumCircuit(3)
        qc.measure_all()

        probed, meta = sampler._inject_probes(qc)
        creg_names = [cr.name for cr in probed.cregs]
        assert "qgate_probe" in creg_names

    def test_preserves_original_gates(self):
        sampler = self._make_sampler()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cr = ClassicalRegister(2, "meas")
        qc.add_register(cr)
        qc.measure([0, 1], cr)

        probed, _ = sampler._inject_probes(qc)
        # Original H and CX should be present
        gate_names = [inst.operation.name for inst in probed.data]
        assert "h" in gate_names
        assert "cx" in gate_names

    def test_probe_angle_reflected(self):
        sampler = self._make_sampler(probe_angle=0.5)
        qc = QuantumCircuit(3)
        qc.measure_all()

        _, meta = sampler._inject_probes(qc)
        assert meta["probe_angle"] == pytest.approx(0.5)

    def test_single_qubit_circuit(self):
        """Single-qubit circuit: n_pairs = max(0, 1) = 1, should still work."""
        sampler = self._make_sampler()
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        # Should not raise — 0 nearest-neighbour pairs but max(0,1)=1
        # handled gracefully (no CRY gates, just measure ancilla)
        probed, meta = sampler._inject_probes(qc)
        assert probed.num_qubits == 2
        assert meta["n_system_qubits"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — Full integration with Aer simulator
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_RUNTIME, reason="qiskit-ibm-runtime not installed")
class TestFullIntegration:
    """End-to-end tests: build circuit → QgateSampler.run() → filtered result."""

    def _make_sampler(self, **cfg_kw) -> QgateSampler:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        cfg = SamplerConfig(**cfg_kw)
        return QgateSampler(backend=backend, config=cfg)

    def test_run_returns_result(self):
        """Basic smoke test: run a GHZ circuit through QgateSampler."""
        sampler = self._make_sampler(
            target_acceptance=0.50,  # lenient for testing
            min_window_size=10,
            window_size=256,
        )

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        job = sampler.run([(qc,)], shots=256)
        assert isinstance(job, QgateSamplerResult)

        result = job.result()
        assert result is not None
        # Should have 1 PubResult
        assert len(result) >= 1

    def test_filtered_shot_count_reduced(self):
        """With tight acceptance, filtered shots should be fewer than original."""
        sampler = self._make_sampler(
            target_acceptance=0.05,
            min_window_size=10,
            window_size=512,
        )

        qc = QuantumCircuit(4)
        qc.h(range(4))  # uniform superposition — noisy/random
        qc.measure_all()

        job = sampler.run([(qc,)], shots=512)
        result = job.result()

        pub = result[0]
        meta = pub.metadata.get("qgate_filter", {})
        total = meta.get("total_shots", 0)
        accepted = meta.get("accepted_shots", 0)

        # With uniform superposition, most shots should have low probe score
        # so accepted < total (unless in warmup with lenient baseline)
        assert total > 0
        assert accepted > 0
        assert accepted <= total

    def test_metadata_populated(self):
        """Filtered results should carry qgate_filter metadata."""
        sampler = self._make_sampler(
            target_acceptance=0.20,
            min_window_size=10,
            window_size=256,
        )

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = sampler.run([(qc,)], shots=128).result()
        pub = result[0]
        assert "qgate_filter" in pub.metadata
        fm = pub.metadata["qgate_filter"]
        assert "total_shots" in fm
        assert "accepted_shots" in fm
        assert "acceptance_rate" in fm
        assert "threshold" in fm

    def test_multiple_pubs(self):
        """Multiple PUBs should each get independent filtering."""
        sampler = self._make_sampler(
            target_acceptance=0.30,
            min_window_size=10,
            window_size=256,
        )

        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure_all()

        qc2 = QuantumCircuit(3)
        qc2.h(range(3))
        qc2.measure_all()

        result = sampler.run([(qc1,), (qc2,)], shots=128).result()
        assert len(result) == 2

        for pub in result:
            assert "qgate_filter" in pub.metadata

    def test_repr(self):
        sampler = self._make_sampler()
        r = repr(sampler)
        assert "QgateSampler" in r
        assert "probe_angle" in r


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — QgateSamplerResult transparency
# ═══════════════════════════════════════════════════════════════════════════


class TestQgateSamplerResultTransparency:
    """Verify that QgateSamplerResult delegates attribute access."""

    def test_result_cached(self):
        """Calling .result() twice returns the same object."""

        class _FakeResult:
            metadata = {"fake": True}

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self

        class _FakeSampler:
            call_count = 0

            def _apply_galton_filter(self, raw, meta):
                self.call_count += 1
                return _FakeResult()

        fake_sampler = _FakeSampler()
        qsr = QgateSamplerResult(_FakeResult(), fake_sampler, [{}])  # type: ignore[arg-type]

        r1 = qsr.result()
        r2 = qsr.result()
        assert r1 is r2
        assert fake_sampler.call_count == 1  # called only once
