#!/usr/bin/env python
"""QJL Transformer Speed Optimisation Benchmark.

Compares the following inference configurations:

1. **Baseline (eager)**     — default config, plain PyTorch forward pass.
2. **torch.compile**        — baseline config + ``torch.compile(mode="default")``.
3. **ONNX Runtime**         — baseline config exported to ORT.
4. **Pruned (1L/2H)**       — ``fast_qjl_config()`` (1 layer, 2 heads).
5. **Pruned + downsample**  — fast config with Conv1D stride=4 temporal downsampling.
6. **Pruned + compile**     — fast config + ``torch.compile``.
7. **Pruned + ORT**         — fast config + ONNX Runtime inference.

Each variant is trained on the same mock dataset, then inference latency
is measured over multiple warmup + timed runs.

Usage::

    python scripts/benchmark_qjl_speed.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "qgate" / "src"))

import torch
from qgate.neural_mitigation import (
    NeuralMitigationConfig,
    QJLLinearTransformer,
    fast_qjl_config,
    generate_mock_dataset,
)

# ── Configuration ─────────────────────────────────────────────────────────

N_SAMPLES = 2000
SEQ_LEN = 64
VOCAB_SIZE = 64
TRAIN_EPOCHS = 30
BATCH_SIZE = 64
N_WARMUP = 20
N_TIMED = 100


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_default_config() -> NeuralMitigationConfig:
    """Default Tier-11 config (matches benchmark_full_stack)."""
    return NeuralMitigationConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        max_seq_len=SEQ_LEN,
        n_heads=4,
        n_layers=2,
        hidden_dim=64,
        qjl_dim=16,
    )


def _train_strategy(
    strategy: QJLLinearTransformer,
    train_tokens: torch.Tensor,
    train_targets: torch.Tensor,
) -> None:
    """Train a strategy to convergence on the mock dataset."""
    strategy.calibrate(
        train_tokens, train_targets,
        n_epochs=TRAIN_EPOCHS, lr=1e-3, batch_size=BATCH_SIZE,
    )
    strategy.model.eval()


def _measure_latency(
    fn: Callable[[torch.Tensor], Any],
    test_tokens: torch.Tensor,
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
) -> Tuple[float, float]:
    """Measure inference latency in microseconds.

    Returns:
        (mean_us, std_us) over *n_timed* runs (excluding warmup).
    """
    # Warmup
    for _ in range(n_warmup):
        fn(test_tokens)

    times: List[float] = []
    for _ in range(n_timed):
        t0 = time.perf_counter_ns()
        fn(test_tokens)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000.0)  # ns → µs

    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def _measure_mae(
    fn: Callable[[torch.Tensor], Any],
    test_tokens: torch.Tensor,
    test_targets: np.ndarray,
) -> float:
    """Compute mean absolute error."""
    with torch.no_grad():
        preds = fn(test_tokens)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    return float(np.mean(np.abs(preds - test_targets)))


# ── Variant builders ──────────────────────────────────────────────────────

def build_variants(
    train_tokens: torch.Tensor,
    train_targets: torch.Tensor,
) -> Dict[str, Tuple[Callable, QJLLinearTransformer]]:
    """Build and train all benchmark variants.

    Returns:
        dict mapping variant name → (inference_fn, strategy)
    """
    variants: Dict[str, Tuple[Callable, QJLLinearTransformer]] = {}

    # ── 1. Baseline (eager) ───────────────────────────────────────────
    print("  Building: baseline (eager, 2L/4H) ...")
    cfg_base = _make_default_config()
    s1 = QJLLinearTransformer(cfg_base)
    _train_strategy(s1, train_tokens, train_targets)
    variants["baseline_eager"] = (
        lambda tokens, _s=s1: _s.forward(tokens),
        s1,
    )

    # ── 2. torch.compile ──────────────────────────────────────────────
    print("  Building: baseline + torch.compile ...")
    cfg_compile = _make_default_config()
    s2 = QJLLinearTransformer(cfg_compile)
    _train_strategy(s2, train_tokens, train_targets)
    try:
        s2.compile_model(mode="default")
        # Trigger a warmup compilation
        with torch.no_grad():
            s2.forward(train_tokens[:1])
        variants["baseline_compile"] = (
            lambda tokens, _s=s2: _s.forward(tokens),
            s2,
        )
    except Exception as e:
        print(f"    ⚠ torch.compile failed: {e}")

    # ── 3. ONNX Runtime ──────────────────────────────────────────────
    print("  Building: baseline + ORT ...")
    cfg_ort = _make_default_config()
    s3 = QJLLinearTransformer(cfg_ort)
    _train_strategy(s3, train_tokens, train_targets)
    try:
        s3.prepare_ort_session()
        variants["baseline_ort"] = (
            lambda tokens, _s=s3: _s.forward_ort(tokens),
            s3,
        )
    except Exception as e:
        print(f"    ⚠ ORT failed: {e}")

    # ── 4. Pruned (1L/2H) — no downsampling ──────────────────────────
    print("  Building: pruned (1L/2H, no downsample) ...")
    cfg_pruned = fast_qjl_config(use_temporal_downsample=False)
    s4 = QJLLinearTransformer(cfg_pruned)
    _train_strategy(s4, train_tokens, train_targets)
    variants["pruned_1L2H"] = (
        lambda tokens, _s=s4: _s.forward(tokens),
        s4,
    )

    # ── 5. Pruned + temporal downsampling ─────────────────────────────
    print("  Building: pruned + Conv1D downsample (stride=4) ...")
    cfg_pruned_ds = fast_qjl_config()  # default has downsample enabled
    s5 = QJLLinearTransformer(cfg_pruned_ds)
    _train_strategy(s5, train_tokens, train_targets)
    variants["pruned_downsample"] = (
        lambda tokens, _s=s5: _s.forward(tokens),
        s5,
    )

    # ── 6. Pruned + compile ───────────────────────────────────────────
    print("  Building: pruned + torch.compile ...")
    cfg_pc = fast_qjl_config(use_temporal_downsample=False)
    s6 = QJLLinearTransformer(cfg_pc)
    _train_strategy(s6, train_tokens, train_targets)
    try:
        s6.compile_model(mode="default")
        with torch.no_grad():
            s6.forward(train_tokens[:1])
        variants["pruned_compile"] = (
            lambda tokens, _s=s6: _s.forward(tokens),
            s6,
        )
    except Exception as e:
        print(f"    ⚠ torch.compile (pruned) failed: {e}")

    # ── 7. Pruned + ORT ──────────────────────────────────────────────
    print("  Building: pruned + ORT ...")
    cfg_po = fast_qjl_config(use_temporal_downsample=False)
    s7 = QJLLinearTransformer(cfg_po)
    _train_strategy(s7, train_tokens, train_targets)
    try:
        s7.prepare_ort_session()
        variants["pruned_ort"] = (
            lambda tokens, _s=s7: _s.forward_ort(tokens),
            s7,
        )
    except Exception as e:
        print(f"    ⚠ ORT (pruned) failed: {e}")

    # ── 8. Pruned + downsample + ORT ─────────────────────────────────
    print("  Building: pruned + downsample + ORT ...")
    cfg_pdo = fast_qjl_config()  # downsample on
    s8 = QJLLinearTransformer(cfg_pdo)
    _train_strategy(s8, train_tokens, train_targets)
    try:
        s8.prepare_ort_session()
        variants["pruned_ds_ort"] = (
            lambda tokens, _s=s8: _s.forward_ort(tokens),
            s8,
        )
    except Exception as e:
        print(f"    ⚠ ORT (pruned+ds) failed: {e}")

    return variants


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 78)
    print("  QJL TRANSFORMER — SPEED OPTIMISATION BENCHMARK")
    print("=" * 78)
    print(f"  torch {torch.__version__}  |  Python {sys.version.split()[0]}")
    print(f"  Dataset: {N_SAMPLES} samples, seq_len={SEQ_LEN}, vocab={VOCAB_SIZE}")
    print(f"  Training: {TRAIN_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f"  Timing: {N_WARMUP} warmup + {N_TIMED} timed runs")
    print()

    # Generate data
    print("Generating mock dataset ...")
    tokens, targets = generate_mock_dataset(
        n_samples=N_SAMPLES, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=42,
    )
    n_train = int(N_SAMPLES * 0.8)
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(N_SAMPLES, generator=gen)
    train_tokens = tokens[perm[:n_train]]
    train_targets = targets[perm[:n_train]]
    test_tokens = tokens[perm[n_train:]]
    test_targets = targets[perm[n_train:]].numpy()

    # Build variants
    print("Building & training variants ...")
    variants = build_variants(train_tokens, train_targets)

    # Benchmark
    print()
    print("Running benchmarks ...")
    print()

    results: List[Dict[str, Any]] = []
    for name, (fn, strategy) in variants.items():
        with torch.no_grad():
            mae = _measure_mae(fn, test_tokens, test_targets)
            mean_us, std_us = _measure_latency(fn, test_tokens)
        n_params = sum(p.numel() for p in strategy.model.parameters())
        results.append({
            "name": name,
            "mae": mae,
            "mean_us": mean_us,
            "std_us": std_us,
            "n_params": n_params,
        })

    # Print table
    print()
    print("=" * 90)
    print(f"  {'Variant':<28} {'MAE':>10} {'Latency(µs)':>14} {'±σ':>10} {'Params':>10} {'Speedup':>9}")
    print("  " + "-" * 84)

    baseline_us = results[0]["mean_us"] if results else 1.0
    for r in results:
        speedup = baseline_us / r["mean_us"] if r["mean_us"] > 0 else 0.0
        marker = ""
        if r["name"] == min(results, key=lambda x: x["mean_us"])["name"]:
            marker = " ★fast"
        if r["name"] == min(results, key=lambda x: x["mae"])["name"]:
            marker += " ★acc"
        print(
            f"  {r['name']:<28} {r['mae']:>10.4f} {r['mean_us']:>14.1f} "
            f"{r['std_us']:>10.1f} {r['n_params']:>10,d} {speedup:>8.2f}×{marker}"
        )

    print("=" * 90)
    print()

    # Summary
    fastest = min(results, key=lambda x: x["mean_us"])
    most_accurate = min(results, key=lambda x: x["mae"])
    print(f"  Fastest:      {fastest['name']}  ({fastest['mean_us']:.1f} µs, "
          f"{baseline_us / fastest['mean_us']:.1f}× vs baseline)")
    print(f"  Most accurate: {most_accurate['name']}  (MAE={most_accurate['mae']:.4f})")
    print()


if __name__ == "__main__":
    main()
