#!/usr/bin/env python3
"""
Financial Monte Carlo Acceleration Benchmark
=============================================

Demonstrates the Probabilistic Processing Unit (PPU) mitigation
pipeline from ``qgate.stochastic`` on a realistic financial pricing
problem: **Asian Call Option** under **Fractional Brownian Motion**.

The benchmark compares three configurations:

1. **Ground Truth** — 1 000 000 paths (expensive "oracle" reference).
2. **Raw Budget**   — 1 000 paths (cheap, high-variance naive MC).
3. **Qgate PPU Mitigated** — 1 000 paths + two-stage Galton + ML
   correction using only 2 000 calibration paths.

Key metrics reported:
  - Mean absolute error (MAE) vs ground truth
  - Improvement factor = |raw_error| / |mitigated_error|
  - Equivalent paths = how many raw paths would you need
  - Compute reduction = equivalent_paths / budget_paths

Patent reference: US Provisional App. No. 64/XXX,XXX (April 2026), §22.
CIP — PPU generalisation of trajectory filtering.

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
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_QGATE_SRC = os.path.join(_REPO_ROOT, "packages", "qgate", "src")
if _QGATE_SRC not in sys.path:
    sys.path.insert(0, _QGATE_SRC)

from qgate.stochastic import (
    PPUMitigationPipeline,
    StochasticConfig,
    StochasticTelemetryExtractor,
    asian_call_payoff,
    run_monte_carlo_benchmark,
    simulate_fbm_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("mc_benchmark")

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Scenarios
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "name": "Asian Call • H=0.7 (trending fBM)",
        "hurst": 0.7,
        "n_ground_truth": 1_000_000,
        "n_budget": 1_000,
        "n_calibration": 2_000,
        "n_steps": 252,
        "S0": 100.0,
        "strike": 100.0,
        "sigma": 0.2,
        "mu": 0.05,
        "T": 1.0,
        "r": 0.05,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 42,
    },
    {
        "name": "Asian Call • H=0.5 (standard GBM baseline)",
        "hurst": 0.5,
        "n_ground_truth": 1_000_000,
        "n_budget": 1_000,
        "n_calibration": 2_000,
        "n_steps": 252,
        "S0": 100.0,
        "strike": 100.0,
        "sigma": 0.2,
        "mu": 0.05,
        "T": 1.0,
        "r": 0.05,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 42,
    },
    {
        "name": "Asian Call • High Vol σ=0.4, H=0.7",
        "hurst": 0.7,
        "n_ground_truth": 1_000_000,
        "n_budget": 1_000,
        "n_calibration": 2_000,
        "n_steps": 252,
        "S0": 100.0,
        "strike": 100.0,
        "sigma": 0.4,
        "mu": 0.05,
        "T": 1.0,
        "r": 0.05,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 42,
    },
    {
        "name": "Asian Call • Gradient Boosting, H=0.7",
        "hurst": 0.7,
        "n_ground_truth": 1_000_000,
        "n_budget": 1_000,
        "n_calibration": 2_000,
        "n_steps": 252,
        "S0": 100.0,
        "strike": 100.0,
        "sigma": 0.2,
        "mu": 0.05,
        "T": 1.0,
        "r": 0.05,
        "reject_fraction": 0.25,
        "model_name": "gradient_boosting",
        "seed": 42,
    },
    {
        "name": "Asian Call • Budget=500, H=0.7",
        "hurst": 0.7,
        "n_ground_truth": 1_000_000,
        "n_budget": 500,
        "n_calibration": 2_000,
        "n_steps": 252,
        "S0": 100.0,
        "strike": 100.0,
        "sigma": 0.2,
        "mu": 0.05,
        "T": 1.0,
        "r": 0.05,
        "reject_fraction": 0.25,
        "model_name": "random_forest",
        "seed": 42,
    },
]


def _fmt_money(v: float) -> str:
    return f"${v:,.6f}"


def _fmt_int(v: int) -> str:
    return f"{v:,}"


def _fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def _run_scenario(scenario: dict) -> dict:
    """Run a single benchmark scenario and return the results dict."""
    name = scenario["name"]
    logger.info("=" * 70)
    logger.info("  %s", name)
    logger.info("=" * 70)

    result = run_monte_carlo_benchmark(
        n_ground_truth=scenario["n_ground_truth"],
        n_budget=scenario["n_budget"],
        n_calibration=scenario["n_calibration"],
        strike=scenario["strike"],
        S0=scenario["S0"],
        mu=scenario["mu"],
        sigma=scenario["sigma"],
        T=scenario["T"],
        hurst=scenario["hurst"],
        n_steps=scenario["n_steps"],
        r=scenario["r"],
        reject_fraction=scenario["reject_fraction"],
        model_name=scenario["model_name"],
        seed=scenario["seed"],
    )

    result["scenario_name"] = name
    return result


def _print_summary_table(results: list[dict]) -> None:
    """Print a business-legible summary table to stdout."""
    print()
    print("=" * 90)
    print("  PPU Monte Carlo Acceleration — Summary")
    print("  Patent ref: US Prov. 64/XXX,XXX §22 (April 2026)")
    print("=" * 90)

    header = (
        f"{'Scenario':<42} {'GT Price':>10} {'Raw MAE':>10} "
        f"{'Mit MAE':>10} {'Improv':>8} {'Equiv Paths':>13}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        name = r["scenario_name"][:40]
        gt = r["ground_truth_price"]
        raw_mae = r["raw_mae"]
        mit_mae = r["mitigated_mae"]
        imp = r["improvement_factor"]
        equiv = r["equivalent_paths"]

        print(
            f"  {name:<40} {gt:>10.4f} {raw_mae:>10.6f} "
            f"{mit_mae:>10.6f} {imp:>7.1f}× {equiv:>12,}"
        )

    print("-" * 90)
    print()

    # Detailed per-scenario output
    for r in results:
        name = r["scenario_name"]
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")
        print(f"  Ground Truth ({_fmt_int(r['params']['n_ground_truth'])} paths):")
        print(f"    Price          : {_fmt_money(r['ground_truth_price'])}")
        print(f"    Std Dev        : {r['ground_truth_std']:.6f}")
        print()
        print(f"  Raw Budget ({_fmt_int(r['params']['n_budget'])} paths):")
        print(f"    Price          : {_fmt_money(r['raw_budget_price'])}")
        print(f"    MAE vs GT      : {r['raw_mae']:.6f}")
        print()
        print(f"  Qgate PPU Mitigated ({_fmt_int(r['params']['n_budget'])} paths):")
        print(f"    Price          : {_fmt_money(r['mitigated_price'])}")
        print(f"    MAE vs GT      : {r['mitigated_mae']:.6f}")
        print(f"    Improvement    : {r['improvement_factor']:.1f}×")
        print()
        print(f"  Stage 1 Galton filter:")
        print(f"    Survivors      : {r['stage1_survivors']}")
        print(f"    Rejected       : {r['stage1_rejected']}")
        print()
        print(f"  Equivalent raw paths for same accuracy: {_fmt_int(r['equivalent_paths'])}")
        print(f"  Compute reduction factor: {r['compute_reduction']:.0f}×")
        print()
        t = r["timing"]
        print(f"  Timing:")
        print(f"    GT simulation  : {t['ground_truth_seconds']:.2f}s")
        print(f"    Budget + mitig : {t['budget_seconds'] + t['mitigate_seconds']:.3f}s")


def main() -> None:
    t0 = time.monotonic()

    results = []
    for scenario in SCENARIOS:
        result = _run_scenario(scenario)
        results.append(result)

    _print_summary_table(results)

    # ── Save JSON results ─────────────────────────────────────────────
    out_dir = os.path.join(
        _REPO_ROOT, "simulations", "financial_monte_carlo",
    )
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"mc_benchmark_{ts}.json")

    # Make results JSON-serialisable
    serialisable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sr[k] = float(v)
            elif isinstance(v, dict):
                sr[k] = {
                    dk: int(dv) if isinstance(dv, np.integer)
                    else float(dv) if isinstance(dv, np.floating)
                    else dv
                    for dk, dv in v.items()
                }
            else:
                sr[k] = v
        serialisable.append(sr)

    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    elapsed = time.monotonic() - t0
    print(f"\n✅ Total benchmark time: {elapsed:.1f}s")
    print(f"📁 Results saved: {json_path}")


if __name__ == "__main__":
    main()
