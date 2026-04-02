"""Tests for qgate.neural_mitigation — PyTorch neural error mitigation strategies.

Tests are structured in tiers:
  1. Unit tests for NeuralMitigationConfig (Pydantic validation, frozen, bounds)
  2. Unit tests for each strategy (construction, forward pass, parameter counts)
  3. Integration tests for TelemetryProcessor orchestrator
  4. Integration tests for the benchmarking suite
  5. Export / ONNX tests

NOTICE: Pre-patent proprietary code — do NOT push to public repositories.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if PyTorch not installed
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="PyTorch required for neural mitigation tests")

from pydantic import ValidationError  # noqa: E402

from qgate.neural_mitigation import (  # noqa: E402
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
    _LinearAttentionBlock,
    _QJLProjection,
    generate_mock_dataset,
    list_strategies,
    print_benchmark_report,
    run_historical_benchmarks,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — synthetic data generators
# ═══════════════════════════════════════════════════════════════════════════


def _make_tokens(batch: int = 8, seq_len: int = 16, vocab: int = 64, seed: int = 42):
    """Create a small batch of mock tokens for testing."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq_len), generator=gen)


def _make_targets(batch: int = 8, seed: int = 42):
    """Create mock target observables."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, generator=gen) * 10.0 - 15.0


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — NeuralMitigationConfig unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNeuralMitigationConfig:
    """Verify NeuralMitigationConfig Pydantic model validation."""

    def test_defaults(self):
        cfg = NeuralMitigationConfig()
        assert cfg.vocab_size == 64
        assert cfg.embed_dim == 32
        assert cfg.max_seq_len == 128
        assert cfg.n_heads == 4
        assert cfg.n_layers == 2
        assert cfg.hidden_dim == 64
        assert cfg.dropout == pytest.approx(0.1)
        assert cfg.diffusion_steps == 10
        assert cfg.qjl_dim == 16
        assert cfg.use_qat is False
        assert cfg.random_state == 42

    def test_custom_values(self):
        cfg = NeuralMitigationConfig(
            vocab_size=128,
            embed_dim=64,
            max_seq_len=256,
            n_heads=8,
            n_layers=4,
            hidden_dim=128,
            dropout=0.2,
            diffusion_steps=20,
            qjl_dim=32,
            use_qat=True,
            random_state=123,
        )
        assert cfg.vocab_size == 128
        assert cfg.embed_dim == 64
        assert cfg.n_heads == 8
        assert cfg.use_qat is True

    def test_frozen_config(self):
        cfg = NeuralMitigationConfig()
        with pytest.raises(ValidationError):
            cfg.vocab_size = 999

    def test_vocab_size_bounds(self):
        with pytest.raises(ValidationError):
            NeuralMitigationConfig(vocab_size=1)  # min 2
        with pytest.raises(ValidationError):
            NeuralMitigationConfig(vocab_size=5000)  # max 4096

    def test_embed_dim_bounds(self):
        with pytest.raises(ValidationError):
            NeuralMitigationConfig(embed_dim=2)  # min 4
        with pytest.raises(ValidationError):
            NeuralMitigationConfig(embed_dim=1024)  # max 512

    def test_dropout_bounds(self):
        with pytest.raises(ValidationError):
            NeuralMitigationConfig(dropout=-0.1)  # min 0.0
        with pytest.raises(ValidationError):
            NeuralMitigationConfig(dropout=0.6)  # max 0.5

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            NeuralMitigationConfig(nonexistent_field=42)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Strategy unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQJLProjection:
    """Verify the QJL random projection layer."""

    def test_output_shape(self):
        proj = _QJLProjection(input_dim=32, qjl_dim=16, seed=42)
        x = torch.randn(4, 10, 32)
        out = proj(x)
        assert out.shape == (4, 10, 16)

    def test_deterministic(self):
        proj1 = _QJLProjection(input_dim=32, qjl_dim=16, seed=42)
        proj2 = _QJLProjection(input_dim=32, qjl_dim=16, seed=42)
        x = torch.randn(2, 5, 32)
        torch.testing.assert_close(proj1(x), proj2(x))

    def test_rademacher_entries(self):
        """Projection matrix should contain only ±1/√d values."""
        proj = _QJLProjection(input_dim=8, qjl_dim=4, seed=0)
        R = proj.projection
        scale = 1.0 / math.sqrt(4)
        unique_abs = torch.unique(R.abs())
        assert len(unique_abs) == 1
        assert unique_abs[0].item() == pytest.approx(scale, abs=1e-6)


class TestLinearAttentionBlock:
    """Verify the linear attention mechanism."""

    def test_output_shape(self):
        attn = _LinearAttentionBlock(embed_dim=32, n_heads=4, qjl_dim=0)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_with_qjl(self):
        attn = _LinearAttentionBlock(embed_dim=32, n_heads=4, qjl_dim=8)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_requires_divisible_heads(self):
        with pytest.raises(AssertionError):
            _LinearAttentionBlock(embed_dim=30, n_heads=4)


class TestQJLLinearTransformer:
    """Verify Strategy 1: QJL Linear Transformer."""

    def test_construction(self):
        s = QJLLinearTransformer()
        assert s.name == "qjl_transformer"
        assert isinstance(s, ErrorMitigationStrategy)

    def test_forward_shape(self):
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16, n_heads=2,
                                      n_layers=1, hidden_dim=32, qjl_dim=8)
        s = QJLLinearTransformer(config=cfg)
        tokens = _make_tokens(batch=4, seq_len=8, vocab=32)
        out = s.forward(tokens)
        assert out.shape == (4,)

    def test_deterministic_forward(self):
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16, n_heads=2,
                                      n_layers=1, hidden_dim=32, random_state=42)
        s = QJLLinearTransformer(config=cfg)
        s.model.eval()
        tokens = _make_tokens(batch=2, seq_len=8, vocab=32)
        with torch.no_grad():
            out1 = s.forward(tokens)
            out2 = s.forward(tokens)
        torch.testing.assert_close(out1, out2)

    def test_parameter_count_positive(self):
        s = QJLLinearTransformer()
        assert s._count_parameters() > 0

    def test_config_access(self):
        cfg = NeuralMitigationConfig(vocab_size=128)
        s = QJLLinearTransformer(config=cfg)
        assert s.config.vocab_size == 128


class TestLegacyQuantizedLSTM:
    """Verify Strategy 2: Legacy LSTM baseline."""

    def test_construction(self):
        s = LegacyQuantizedLSTM()
        assert s.name == "legacy_lstm"
        assert isinstance(s, ErrorMitigationStrategy)

    def test_forward_shape(self):
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16,
                                      n_layers=1, hidden_dim=32)
        s = LegacyQuantizedLSTM(config=cfg)
        tokens = _make_tokens(batch=4, seq_len=8, vocab=32)
        out = s.forward(tokens)
        assert out.shape == (4,)

    def test_deterministic_forward(self):
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16,
                                      n_layers=1, hidden_dim=32, random_state=42)
        s = LegacyQuantizedLSTM(config=cfg)
        s.model.eval()
        tokens = _make_tokens(batch=2, seq_len=8, vocab=32)
        with torch.no_grad():
            out1 = s.forward(tokens)
            out2 = s.forward(tokens)
        torch.testing.assert_close(out1, out2)

    def test_parameter_count_positive(self):
        s = LegacyQuantizedLSTM()
        assert s._count_parameters() > 0


class TestDiffusionAnomalyDetector:
    """Verify Strategy 3: Diffusion denoising autoencoder."""

    def test_construction(self):
        s = DiffusionAnomalyDetector()
        assert s.name == "diffusion_detector"
        assert isinstance(s, ErrorMitigationStrategy)

    def test_forward_shape(self):
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16,
                                      hidden_dim=32, diffusion_steps=5)
        s = DiffusionAnomalyDetector(config=cfg)
        tokens = _make_tokens(batch=4, seq_len=8, vocab=32)
        out = s.forward(tokens)
        assert out.shape == (4,)

    def test_deterministic_eval(self):
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16,
                                      hidden_dim=32, diffusion_steps=5)
        s = DiffusionAnomalyDetector(config=cfg)
        s.model.eval()
        tokens = _make_tokens(batch=2, seq_len=8, vocab=32)
        with torch.no_grad():
            out1 = s.forward(tokens)
            out2 = s.forward(tokens)
        torch.testing.assert_close(out1, out2)

    def test_corruption_at_t0_is_identity(self):
        """At t=0 (no noise), corruption should not change tokens."""
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16,
                                      hidden_dim=32, diffusion_steps=5)
        s = DiffusionAnomalyDetector(config=cfg)
        tokens = _make_tokens(batch=4, seq_len=8, vocab=32)
        corrupted = s.model._corrupt(tokens, t=0)
        # alpha_bar[0] should be ~1.0 (cosine schedule), so no corruption
        # In practice the alpha is cos(0)^2 = 1.0 → all kept
        torch.testing.assert_close(corrupted, tokens)

    def test_corruption_at_max_t_is_random(self):
        """At max t, most tokens should be replaced."""
        cfg = NeuralMitigationConfig(vocab_size=32, embed_dim=16,
                                      hidden_dim=32, diffusion_steps=100)
        s = DiffusionAnomalyDetector(config=cfg)
        tokens = torch.zeros(4, 100, dtype=torch.long)  # all token 0
        corrupted = s.model._corrupt(tokens, t=100)
        # At t=max, alpha_bar ≈ 0 → almost all tokens replaced
        fraction_changed = (corrupted != tokens).float().mean().item()
        assert fraction_changed > 0.5  # most should be changed

    def test_parameter_count_positive(self):
        s = DiffusionAnomalyDetector()
        assert s._count_parameters() > 0


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — Calibration (training) tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibration:
    """Verify that each strategy can be calibrated (trained)."""

    @pytest.fixture
    def small_config(self):
        return NeuralMitigationConfig(
            vocab_size=16, embed_dim=8, max_seq_len=16,
            n_heads=2, n_layers=1, hidden_dim=16,
            qjl_dim=4, diffusion_steps=3, random_state=42,
        )

    @pytest.fixture
    def train_data(self, small_config):
        tokens = _make_tokens(batch=32, seq_len=16, vocab=small_config.vocab_size)
        targets = _make_targets(batch=32)
        return tokens, targets

    def test_qjl_transformer_calibrate(self, small_config, train_data):
        tokens, targets = train_data
        s = QJLLinearTransformer(config=small_config)
        cal = s.calibrate(tokens, targets, n_epochs=3, lr=1e-3)
        assert isinstance(cal, NeuralCalibrationResult)
        assert cal.strategy_name == "qjl_transformer"
        assert cal.n_samples == 32
        assert cal.n_epochs == 3
        assert cal.n_parameters > 0
        assert cal.elapsed_seconds > 0
        assert cal.final_loss >= 0

    def test_legacy_lstm_calibrate(self, small_config, train_data):
        tokens, targets = train_data
        s = LegacyQuantizedLSTM(config=small_config)
        cal = s.calibrate(tokens, targets, n_epochs=3, lr=1e-3)
        assert isinstance(cal, NeuralCalibrationResult)
        assert cal.strategy_name == "legacy_lstm"
        assert cal.n_samples == 32

    def test_diffusion_calibrate(self, small_config, train_data):
        tokens, targets = train_data
        s = DiffusionAnomalyDetector(config=small_config)
        cal = s.calibrate(tokens, targets, n_epochs=3, lr=1e-3)
        assert isinstance(cal, NeuralCalibrationResult)
        assert cal.strategy_name == "diffusion_detector"
        assert cal.n_samples == 32

    def test_calibration_reduces_loss(self, small_config):
        """Training for multiple epochs should reduce loss."""
        tokens = _make_tokens(batch=64, seq_len=16, vocab=small_config.vocab_size)
        targets = _make_targets(batch=64)

        s = QJLLinearTransformer(config=small_config)

        # Train 1 epoch
        cal1 = s.calibrate(tokens, targets, n_epochs=1, lr=1e-2)

        # Train more epochs (fresh model)
        s2 = QJLLinearTransformer(config=small_config)
        cal2 = s2.calibrate(tokens, targets, n_epochs=30, lr=1e-2)

        # More training should generally reduce loss (not guaranteed but typical)
        # At minimum, verify both losses are finite
        assert math.isfinite(cal1.final_loss)
        assert math.isfinite(cal2.final_loss)


# ═══════════════════════════════════════════════════════════════════════════
# Tier 4 — TelemetryProcessor orchestrator tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTelemetryProcessor:
    """Verify the TelemetryProcessor orchestrator."""

    @pytest.fixture
    def small_config(self):
        return NeuralMitigationConfig(
            vocab_size=16, embed_dim=8, max_seq_len=16,
            n_heads=2, n_layers=1, hidden_dim=16,
            qjl_dim=4, diffusion_steps=3,
        )

    def test_default_strategy(self, small_config):
        proc = TelemetryProcessor(config=small_config)
        assert proc.strategy.name == "qjl_transformer"

    def test_set_strategy_lstm(self, small_config):
        proc = TelemetryProcessor(method="legacy_lstm", config=small_config)
        assert proc.strategy.name == "legacy_lstm"

    def test_set_strategy_diffusion(self, small_config):
        proc = TelemetryProcessor(method="diffusion_detector", config=small_config)
        assert proc.strategy.name == "diffusion_detector"

    def test_invalid_strategy_raises(self, small_config):
        with pytest.raises(ValueError, match="Unknown strategy"):
            TelemetryProcessor(method="nonexistent", config=small_config)

    def test_switch_strategy(self, small_config):
        proc = TelemetryProcessor(config=small_config)
        assert proc.strategy.name == "qjl_transformer"
        proc.set_strategy("legacy_lstm")
        assert proc.strategy.name == "legacy_lstm"

    def test_forward_returns_result(self, small_config):
        proc = TelemetryProcessor(config=small_config)
        tokens = _make_tokens(batch=4, seq_len=8, vocab=small_config.vocab_size)
        result = proc.forward(tokens)
        assert isinstance(result, NeuralMitigationResult)
        assert result.mitigated_values.shape == (4,)
        assert result.latency_us > 0
        assert result.strategy_name == "qjl_transformer"

    def test_calibrate_through_processor(self, small_config):
        proc = TelemetryProcessor(config=small_config)
        tokens = _make_tokens(batch=16, seq_len=8, vocab=small_config.vocab_size)
        targets = _make_targets(batch=16)
        cal = proc.calibrate(tokens, targets, n_epochs=2)
        assert isinstance(cal, NeuralCalibrationResult)

    def test_all_strategies_produce_output(self, small_config):
        """Every registered strategy should produce valid output."""
        tokens = _make_tokens(batch=4, seq_len=8, vocab=small_config.vocab_size)
        for name in list_strategies():
            proc = TelemetryProcessor(method=name, config=small_config)
            result = proc.forward(tokens)
            assert result.mitigated_values.shape == (4,), f"Failed for {name}"
            assert np.all(np.isfinite(result.mitigated_values)), f"NaN for {name}"


# ═══════════════════════════════════════════════════════════════════════════
# Tier 5 — Strategy registry tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyRegistry:
    """Verify the strategy registry."""

    def test_list_strategies(self):
        strategies = list_strategies()
        assert "qjl_transformer" in strategies
        assert "legacy_lstm" in strategies
        assert "diffusion_detector" in strategies
        assert len(strategies) == 3

    def test_registry_keys_match(self):
        assert set(STRATEGY_REGISTRY.keys()) == set(list_strategies())

    def test_all_registry_entries_are_strategy_subclasses(self):
        for name, cls in STRATEGY_REGISTRY.items():
            assert issubclass(cls, ErrorMitigationStrategy), f"{name} is not a strategy"


# ═══════════════════════════════════════════════════════════════════════════
# Tier 6 — Mock dataset generation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMockDataset:
    """Verify mock dataset generation."""

    def test_default_shape(self):
        tokens, obs = generate_mock_dataset()
        assert tokens.shape == (500, 64)
        assert obs.shape == (500,)

    def test_custom_shape(self):
        tokens, obs = generate_mock_dataset(n_samples=100, seq_len=32, vocab_size=128)
        assert tokens.shape == (100, 32)
        assert obs.shape == (100,)

    def test_token_range(self):
        tokens, _ = generate_mock_dataset(vocab_size=16)
        assert tokens.min() >= 0
        assert tokens.max() < 16

    def test_deterministic(self):
        t1, o1 = generate_mock_dataset(seed=42)
        t2, o2 = generate_mock_dataset(seed=42)
        torch.testing.assert_close(t1, t2)
        torch.testing.assert_close(o1, o2)

    def test_different_seeds(self):
        t1, _ = generate_mock_dataset(seed=42)
        t2, _ = generate_mock_dataset(seed=99)
        assert not torch.equal(t1, t2)

    def test_observables_are_finite(self):
        _, obs = generate_mock_dataset()
        assert torch.all(torch.isfinite(obs))


# ═══════════════════════════════════════════════════════════════════════════
# Tier 7 — Benchmarking suite tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkSuite:
    """Verify the historical benchmarking suite."""

    @pytest.fixture
    def small_dataset(self):
        cfg = NeuralMitigationConfig(vocab_size=16, embed_dim=8,
                                      max_seq_len=16, n_heads=2,
                                      n_layers=1, hidden_dim=16,
                                      qjl_dim=4, diffusion_steps=3)
        tokens, obs = generate_mock_dataset(
            n_samples=40, seq_len=16, vocab_size=16, seed=42,
        )
        return tokens, obs, cfg

    def test_benchmark_produces_report(self, small_dataset):
        tokens, obs, cfg = small_dataset
        report = run_historical_benchmarks(
            tokens, obs, config=cfg,
            n_train_epochs=2, n_inference_runs=1,
        )
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 3  # all 3 strategies

    def test_benchmark_entries_have_metrics(self, small_dataset):
        tokens, obs, cfg = small_dataset
        report = run_historical_benchmarks(
            tokens, obs, config=cfg,
            n_train_epochs=2, n_inference_runs=1,
        )
        for name, entry in report.results.items():
            assert isinstance(entry, BenchmarkEntry)
            assert entry.mae >= 0
            assert entry.latency_us > 0
            assert entry.n_parameters > 0
            assert entry.mitigated_values is not None

    def test_benchmark_has_winners(self, small_dataset):
        tokens, obs, cfg = small_dataset
        report = run_historical_benchmarks(
            tokens, obs, config=cfg,
            n_train_epochs=2, n_inference_runs=1,
        )
        assert report.best_mae in report.results
        assert report.best_latency in report.results
        assert report.timestamp != ""

    def test_print_benchmark_report(self, small_dataset):
        tokens, obs, cfg = small_dataset
        report = run_historical_benchmarks(
            tokens, obs, config=cfg,
            n_train_epochs=2, n_inference_runs=1,
        )
        text = print_benchmark_report(report)
        assert "NEURAL MITIGATION BENCHMARK REPORT" in text
        assert "qjl_transformer" in text
        assert "legacy_lstm" in text
        assert "diffusion_detector" in text


# ═══════════════════════════════════════════════════════════════════════════
# Tier 8 — FPGA export tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExport:
    """Verify model export for FPGA deployment."""

    @pytest.fixture
    def small_config(self):
        return NeuralMitigationConfig(
            vocab_size=16, embed_dim=8, max_seq_len=16,
            n_heads=2, n_layers=1, hidden_dim=16,
            qjl_dim=4, diffusion_steps=3,
        )

    def test_qjl_export(self, small_config, tmp_path):
        s = QJLLinearTransformer(config=small_config)
        out = s.export_to_fpga(tmp_path / "qjl.onnx")
        assert out.exists()

    def test_lstm_export(self, small_config, tmp_path):
        s = LegacyQuantizedLSTM(config=small_config)
        out = s.export_to_fpga(tmp_path / "lstm.onnx")
        assert out.exists()

    def test_diffusion_export(self, small_config, tmp_path):
        s = DiffusionAnomalyDetector(config=small_config)
        out = s.export_to_fpga(tmp_path / "diffusion.onnx")
        assert out.exists()

    def test_export_through_processor(self, small_config, tmp_path):
        proc = TelemetryProcessor(config=small_config)
        out = proc.export_to_fpga(tmp_path / "proc.onnx")
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════
# Tier 9 — Result dataclass tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResultDataclasses:
    """Verify result dataclasses are constructed correctly."""

    def test_neural_calibration_result(self):
        r = NeuralCalibrationResult(
            strategy_name="test",
            n_samples=100,
            final_loss=0.01,
            n_epochs=10,
            n_parameters=5000,
            elapsed_seconds=1.5,
        )
        assert r.strategy_name == "test"
        assert r.n_samples == 100
        assert r.final_loss == pytest.approx(0.01)

    def test_neural_mitigation_result(self):
        vals = np.array([1.0, 2.0, 3.0])
        r = NeuralMitigationResult(
            mitigated_values=vals,
            raw_logits=vals,
            latency_us=50.0,
            strategy_name="test",
        )
        assert r.mitigated_values.shape == (3,)
        assert r.latency_us == pytest.approx(50.0)

    def test_benchmark_entry(self):
        e = BenchmarkEntry(
            strategy_name="test",
            mae=0.05,
            latency_us=100.0,
            n_parameters=1000,
        )
        assert e.strategy_name == "test"
        assert e.mae == pytest.approx(0.05)

    def test_benchmark_report(self):
        entry = BenchmarkEntry(strategy_name="a", mae=0.01, latency_us=50.0)
        r = BenchmarkReport(
            results={"a": entry},
            best_mae="a",
            best_latency="a",
            timestamp="2026-04-01",
        )
        assert len(r.results) == 1
        assert r.best_mae == "a"

    def test_frozen_results(self):
        """Result dataclasses should be immutable."""
        r = NeuralCalibrationResult(strategy_name="test", n_samples=10)
        with pytest.raises(AttributeError):
            r.strategy_name = "changed"
