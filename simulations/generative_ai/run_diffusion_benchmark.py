#!/usr/bin/env python3
"""
Generative AI Diffusion Acceleration Benchmark
===============================================

Demonstrates the Probabilistic Processing Unit (PPU) mitigation
pipeline from ``qgate.diffusion`` on a realistic generative-AI
problem: **latent diffusion image synthesis** (FLUX.2 Klein / SDXL class).

The benchmark compares three configurations for each prompt:

1. **Ground Truth** — 1 latent at 50 denoising steps (expensive,
   reference-quality image with crisp macro details).
2. **Raw Budget**   — 1 latent at 10 steps (cheap, noisy, artefacts:
   gear warping, texture smearing, loss of fine detail).
3. **Qgate PPU Mitigated** — 8 latents at 10 steps, filtered by
   Stage 1 Galton rejection + Stage 2 RF latent reconstruction
   → single high-fidelity fused latent.

Key metrics reported:
  - Latent FID (Fréchet Inception Distance proxy — lower is better)
  - CLIP Score (prompt–image alignment — higher is better)
  - PSNR (dB — higher is better)
  - GPU Inference Time (simulated)
  - Visual Artefacts description

Patent reference: US Provisional App. No. 64/XXX,XXX (April 2026), §22.
CIP — PPU generalisation of trajectory filtering to generative AI.

Licensed under QGATE Source Available Evaluation License v1.2.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

# ── Ensure qgate is importable from the monorepo ─────────────────────────
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir),
)
_QGATE_SRC = os.path.join(_REPO_ROOT, "packages", "qgate", "src")
if _QGATE_SRC not in sys.path:
    sys.path.insert(0, _QGATE_SRC)

from qgate.diffusion import run_diffusion_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("diffusion_benchmark")

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Scenarios
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "name": "Macro Watch • RF • batch=8",
        "prompt": (
            "A macro photography shot of a mechanical watch movement, "
            "intricate gears, ruby bearings, dramatic studio lighting, "
            "8k resolution, photorealistic."
        ),
        "gt_steps": 50,
        "budget_steps": 10,
        "n_batch": 8,
        "n_calibration": 16,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 42,
    },
    {
        "name": "Macro Watch • GBR • batch=8",
        "prompt": (
            "A macro photography shot of a mechanical watch movement, "
            "intricate gears, ruby bearings, dramatic studio lighting, "
            "8k resolution, photorealistic."
        ),
        "gt_steps": 50,
        "budget_steps": 10,
        "n_batch": 8,
        "n_calibration": 16,
        "reject_fraction": 0.25,
        "model_name": "gradient_boosting",
        "seed": 42,
    },
    {
        "name": "Macro Watch • Ridge • batch=8",
        "prompt": (
            "A macro photography shot of a mechanical watch movement, "
            "intricate gears, ruby bearings, dramatic studio lighting, "
            "8k resolution, photorealistic."
        ),
        "gt_steps": 50,
        "budget_steps": 10,
        "n_batch": 8,
        "n_calibration": 16,
        "reject_fraction": 0.25,
        "model_name": "ridge",
        "seed": 42,
    },
    {
        "name": "Macro Watch • RF • batch=16",
        "prompt": (
            "A macro photography shot of a mechanical watch movement, "
            "intricate gears, ruby bearings, dramatic studio lighting, "
            "8k resolution, photorealistic."
        ),
        "gt_steps": 50,
        "budget_steps": 10,
        "n_batch": 16,
        "n_calibration": 16,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 42,
    },
    {
        "name": "Ocean Sunset • RF • batch=8",
        "prompt": (
            "A vast ocean sunset with golden light reflecting off gentle "
            "waves, wispy cirrus clouds painted in coral and lavender, "
            "long exposure photography, award-winning nature photo."
        ),
        "gt_steps": 50,
        "budget_steps": 10,
        "n_batch": 8,
        "n_calibration": 16,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 123,
    },
    {
        "name": "City Neon • RF • batch=8",
        "prompt": (
            "A cyberpunk city street at night, neon signs in Japanese "
            "and English, rain-slicked asphalt reflecting light, "
            "blade runner aesthetic, cinematic 4k photography."
        ),
        "gt_steps": 50,
        "budget_steps": 10,
        "n_batch": 8,
        "n_calibration": 16,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 777,
    },
]

# ── Use small latent shape for fast benchmarking ─────────────────────────
_LATENT_C = 4
_LATENT_H = 32
_LATENT_W = 32


def _run_scenario(scenario: dict) -> dict:
    """Run a single benchmark scenario and return results dict."""
    name = scenario["name"]
    logger.info("=" * 70)
    logger.info("  %s", name)
    logger.info("=" * 70)

    result = run_diffusion_benchmark(
        prompt=scenario["prompt"],
        gt_steps=scenario["gt_steps"],
        budget_steps=scenario["budget_steps"],
        n_batch=scenario["n_batch"],
        n_calibration=scenario["n_calibration"],
        latent_channels=_LATENT_C,
        latent_height=_LATENT_H,
        latent_width=_LATENT_W,
        reject_fraction=scenario["reject_fraction"],
        model_name=scenario["model_name"],
        seed=scenario["seed"],
    )
    result["scenario_name"] = name
    return result


def _print_summary_table(results: list) -> None:
    """Print the business-legible summary table to stdout."""
    print()
    print("=" * 110)
    print("  PPU Diffusion Acceleration — Summary")
    print("  Patent ref: US Prov. 64/XXX,XXX §22 (April 2026)")
    print("  Architecture: Split Frontend (LatentTelemetry) + Unified Backend (Galton + RF)")
    print("=" * 110)

    # ── Main comparison table ─────────────────────────────────────────
    header = (
        f"{'Scenario':<32} {'FID↓':>8} {'CLIP↑':>8} "
        f"{'PSNR↑':>8} {'GPU Time':>10} {'FID Imp':>9} {'Visual Artefacts'}"
    )
    print()
    print("  ── Ground Truth (50-step, reference quality) ──")
    print(f"  {'':>32} {'FID':>8} {'CLIP':>8} {'PSNR':>8} {'GPU(s)':>10}")
    print(f"  {'All prompts':<32} {'0.000':>8} {'1.000':>8} {'100.0':>8} {'40.0':>10}")
    print()

    print("  ── Raw Budget (10-step, single trajectory) ──")
    print(f"  {'Scenario':<32} {'FID↓':>8} {'CLIP↑':>8} {'PSNR↑':>8} {'GPU(s)':>10} {'Artefacts'}")
    print(f"  {'-' * 100}")
    for r in results:
        raw = r["raw_budget"]
        name = r["scenario_name"][:30]
        print(
            f"  {name:<32} {raw['fid']:>8.3f} {raw['clip_score']:>8.3f} "
            f"{raw['psnr']:>8.1f} {raw['gpu_time']:>10.1f} "
            f"{raw['artifacts'][:40]}"
        )

    print()
    print("  ── Qgate PPU Mitigated (8–16× 10-step, filtered + fused) ──")
    print(f"  {'Scenario':<32} {'FID↓':>8} {'CLIP↑':>8} {'PSNR↑':>8} {'GPU(s)':>10} {'FID Imp':>9} {'Artefacts'}")
    print(f"  {'-' * 110}")
    for r in results:
        mit = r["qgate_mitigated"]
        imp = r["improvement"]
        name = r["scenario_name"][:30]
        print(
            f"  {name:<32} {mit['fid']:>8.3f} {mit['clip_score']:>8.3f} "
            f"{mit['psnr']:>8.1f} {mit['gpu_time']:>10.1f} "
            f"{imp['fid_improvement']:>8.1f}× "
            f"{mit['artifacts'][:40]}"
        )

    print(f"\n  {'-' * 110}")

    # ── Detailed per-scenario output ──────────────────────────────────
    for r in results:
        name = r["scenario_name"]
        gt = r["ground_truth"]
        raw = r["raw_budget"]
        mit = r["qgate_mitigated"]
        imp = r["improvement"]
        cal = r["calibration"]
        t = r["timing"]

        print(f"\n{'─' * 80}")
        print(f"  {name}")
        print(f"  Prompt: {r['prompt'][:65]}...")
        print(f"{'─' * 80}")
        print()

        print(f"  Ground Truth ({gt['steps']} steps):")
        print(f"    FID Score      : {gt['fid']:.4f}")
        print(f"    CLIP Score     : {gt['clip_score']:.4f}")
        print(f"    GPU Time       : {gt['gpu_time']:.1f}s")
        print(f"    Quality        : {gt['artifacts']}")
        print()

        print(f"  Raw Budget ({raw['steps']} steps, single trajectory):")
        print(f"    FID Score      : {raw['fid']:.4f}")
        print(f"    CLIP Score     : {raw['clip_score']:.4f}")
        print(f"    PSNR           : {raw['psnr']:.2f} dB")
        print(f"    GPU Time       : {raw['gpu_time']:.1f}s")
        print(f"    Artefacts      : {raw['artifacts']}")
        print()

        print(f"  Qgate PPU Mitigated ({mit['n_batch']}× {mit['steps']} steps):")
        print(f"    FID Score      : {mit['fid']:.4f}")
        print(f"    CLIP Score     : {mit['clip_score']:.4f}")
        print(f"    PSNR           : {mit['psnr']:.2f} dB")
        print(f"    GPU Time       : {mit['gpu_time']:.1f}s")
        print(f"    Stage 1 kept   : {mit['stage1_survivors']}/{mit['stage1_survivors'] + mit['stage1_rejected']}")
        print(f"    Artefacts      : {mit['artifacts']}")
        print()

        print(f"  Improvement:")
        print(f"    FID reduction  : {imp['fid_improvement']:.2f}×")
        print(f"    CLIP gain      : {imp['clip_improvement']:.2f}×")
        print(f"    PSNR gain      : +{imp['psnr_improvement_db']:.2f} dB")
        print(f"    Speedup vs GT  : {imp['speedup_vs_gt']:.2f}×")
        print()

        print(f"  Calibration:")
        print(f"    Model          : {cal['model_name']}")
        print(f"    Train MAE      : {cal['train_mae']:.6f}")
        print(f"    Cal. time      : {cal['elapsed_seconds']:.2f}s")
        print()

        print(f"  Timing:")
        print(f"    GT generation  : {t['gt_wall_seconds']:.3f}s")
        print(f"    Raw generation : {t['raw_wall_seconds']:.3f}s")
        print(f"    Mitigate       : {t['mitigate_wall_seconds']:.3f}s")


def main() -> None:
    t0 = time.monotonic()

    results = []
    for scenario in SCENARIOS:
        result = _run_scenario(scenario)
        results.append(result)

    _print_summary_table(results)

    # ── Save JSON results ─────────────────────────────────────────────
    out_dir = os.path.join(_REPO_ROOT, "simulations", "generative_ai")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"diffusion_benchmark_{ts}.json")

    # Make results JSON-serialisable (strip numpy types + ndarrays)
    def _make_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        return obj

    serialisable = _make_serialisable(results)
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    elapsed = time.monotonic() - t0
    print(f"\n✅ Total benchmark time: {elapsed:.1f}s")
    print(f"📁 Results saved: {json_path}")


if __name__ == "__main__":
    main()
