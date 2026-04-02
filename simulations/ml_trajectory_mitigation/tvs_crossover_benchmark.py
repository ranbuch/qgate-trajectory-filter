#!/usr/bin/env python3
"""TVS Crossover Benchmark — publication-ready noise sweep.

Sweeps σ_IQ from 0.01 → 1.0 (20 steps) and evaluates four TVS
pipelines at each noise level:

    1. **No TVS** — raw ML mitigator baseline (no trajectory filtering).
    2. **Level-2** — binary hard-decision fusion.
    3. **Level-1 (RBF)** — single-centroid soft-decision.
    4. **Level-1-cluster** — K-Means (k=8) multi-centroid soft-decision.

Produces a publication-ready matplotlib figure showing:
    - X-axis: σ_IQ (IQ noise, increasing → degraded readout)
    - Y-axis: MAE (log scale)
    - Four curves revealing the crossover point where Level-1 (RBF)
      degrades and Level-1-cluster takes over.

A second panel overlays the SNR metric on a twin Y-axis, visually
showing the correspondence between IQ separability and the Level-1
failure mode.

Physics rationale
-----------------
At low noise (small σ_IQ), the |0⟩ and |1⟩ IQ clouds are
well-separated.  A single-centroid RBF (Level-1) discriminates
cleanly and outperforms Level-2 because it uses soft distance
information rather than binary thresholding.

At high noise (large σ_IQ), the clouds overlap and a single centroid
produces near-uniform distances — the RBF score distribution
collapses, breaking the Galton filter's ability to separate good
from bad shots.  K-Means clustering (Level-1-cluster) recovers
local structure via multiple Voronoi reference points.

The *crossover point* is the σ_IQ at which Level-1-cluster begins
to outperform Level-1 (RBF), corresponding to SNR dropping below
~3.0.

Patent reference
----------------
US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
CIP addendum — ML-augmented TSVF trajectory mitigation.

.. warning::
   CONFIDENTIAL — DO NOT PUSH / DO NOT PUBLISH.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Add project root to path ─────────────────────────────────────────────
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root / "packages" / "qgate" / "src"))

from qgate.mitigation import MitigatorConfig, TelemetryMitigator
from qgate.tvs import (
    compute_iq_snr,
    process_telemetry_batch,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
N_SHOTS = 5_000
N_QUBITS = 8
EXACT_ENERGY = -3.1415  # Arbitrary reference

# Noise sweep range
SIGMA_IQ_MIN = 0.01
SIGMA_IQ_MAX = 1.0
N_SIGMA_STEPS = 20

# Calibrated centroids
ZERO_CENTROID = 0.95 + 0.02j
ONE_CENTROID = -0.50 + 0.10j   # Typical |1⟩ centroid (far from |0⟩)

# TVS parameters
RBF_VARIANCE = 0.15
K_CLUSTERS = 8
DROP_PERCENTILE = 25.0

# Stage-2 mitigator config
MITIGATOR_CFG = MitigatorConfig(
    keep_fraction=0.70,
    model_name="random_forest",
    random_state=SEED,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data synthesis (same model as T10)
# ═══════════════════════════════════════════════════════════════════════════


def synthesise_data(
    rng: np.random.Generator,
    n_shots: int,
    sigma_iq: float,
    p_err: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise bimodal I/Q data with controlled noise.

    Good shots cluster near ZERO_CENTROID (|0⟩), error shots scatter
    far from it with 3× noise amplification (simulating a TLS-induced
    decoherence event).

    Args:
        rng:       Generator for reproducibility.
        n_shots:   Number of shots.
        sigma_iq:  IQ cloud standard deviation (noise level).
        p_err:     Bit-flip error probability.

    Returns:
        Tuple of (iq_samples, lf_scores, hf_binary, raw_energies, acceptance).
    """
    # Binary error mask
    hf_binary = rng.binomial(1, p_err, n_shots).astype(np.int64)

    # I/Q samples: good shots near centroid, error shots far
    iq = np.empty(n_shots, dtype=np.complex128)
    good_mask = hf_binary == 0
    n_good = int(good_mask.sum())
    n_err = n_shots - n_good

    iq[good_mask] = (
        rng.normal(ZERO_CENTROID.real, sigma_iq, n_good)
        + 1j * rng.normal(ZERO_CENTROID.imag, sigma_iq, n_good)
    )
    iq[~good_mask] = (
        rng.normal(ONE_CENTROID.real, 3.0 * sigma_iq, n_err)
        + 1j * rng.normal(ONE_CENTROID.imag, 3.0 * sigma_iq, n_err)
    )

    # LF drift scores: good shots ≈ 1 − p_err/2, moderate noise
    lf_scores = np.clip(
        rng.normal(1.0 - p_err * 0.5, 0.15, n_shots), 0.0, 1.0
    )

    # Per-shot acceptance (for telemetry features)
    acceptance = 1.0 - hf_binary.astype(np.float64)

    # Synthetic raw energies: error shots have much larger noise
    is_error = hf_binary == 1
    raw_energies = np.where(
        is_error,
        EXACT_ENERGY + rng.normal(0, 3.0 + 10 * p_err, n_shots),
        EXACT_ENERGY + rng.normal(0, 0.5 + 2 * p_err, n_shots),
    )

    return iq, lf_scores, hf_binary, raw_energies, acceptance


# ═══════════════════════════════════════════════════════════════════════════
# MAE computation via Stage-2 mitigator
# ═══════════════════════════════════════════════════════════════════════════


def _build_cal_records(
    rng: np.random.Generator,
    energies: np.ndarray,
    acceptance: np.ndarray,
    p_err: float,
    exact: float,
    mask: np.ndarray,
) -> list:
    """Build calibration records from surviving shots."""
    indices = np.where(mask)[0]
    records = []
    for i in indices:
        records.append({
            "energy": float(energies[i]),
            "acceptance": float(acceptance[i]),
            "variance": float(p_err ** 2 * 100 + rng.exponential(0.01)),
            "ideal": exact,
        })
    return records


def compute_mae_arm(
    rng: np.random.Generator,
    energies: np.ndarray,
    acceptance: np.ndarray,
    p_err: float,
    exact: float,
    mask: np.ndarray,
) -> float:
    """Compute MAE after Stage-2 ML correction on surviving shots.

    Uses TelemetryMitigator.calibrate() + .estimate_batch() on the
    surviving subset, then measures MAE against the exact energy.
    """
    records = _build_cal_records(rng, energies, acceptance, p_err, exact, mask)

    if len(records) < 10:
        # Too few shots — return raw MAE
        return float(np.mean(np.abs(energies[mask] - exact)))

    mitigator = TelemetryMitigator(config=MITIGATOR_CFG)
    mitigator.calibrate(records)
    results = mitigator.estimate_batch(records)
    errors = np.array([abs(r.mitigated_value - exact) for r in results])
    return float(np.mean(errors))


# ═══════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════


def run_sweep() -> List[Dict[str, Any]]:
    """Execute the noise sweep across all σ_IQ values."""
    rng = np.random.default_rng(SEED)

    sigmas = np.linspace(SIGMA_IQ_MIN, SIGMA_IQ_MAX, N_SIGMA_STEPS)
    results: List[Dict[str, Any]] = []

    print("=" * 70)
    print("TVS CROSSOVER BENCHMARK — Noise Sweep")
    print(f"  N_shots: {N_SHOTS}  |  N_sigma_steps: {N_SIGMA_STEPS}")
    print(f"  σ_IQ range: [{SIGMA_IQ_MIN}, {SIGMA_IQ_MAX}]")
    print(f"  K-Means clusters: {K_CLUSTERS}  |  RBF variance: {RBF_VARIANCE}")
    print("=" * 70)

    for i, sigma in enumerate(sigmas):
        p_err = 0.05 + 0.20 * (sigma / SIGMA_IQ_MAX)  # Scale error with noise
        iq, lf, hf_binary, raw_energies, acceptance = synthesise_data(rng, N_SHOTS, sigma, p_err)

        # SNR at this noise level
        snr = compute_iq_snr(iq, ZERO_CENTROID, ONE_CENTROID)

        # ── Arm 1: No TVS (baseline) ────────────────────────────────
        mask_none = np.ones(N_SHOTS, dtype=bool)
        mae_none = compute_mae_arm(rng, raw_energies, acceptance, p_err, EXACT_ENERGY, mask_none)

        # ── Arm 2: Level-2 (binary) ─────────────────────────────────
        res_l2 = process_telemetry_batch(
            hf_binary, lf, force_mode="level_2",
            alpha=0.5, drop_percentile=DROP_PERCENTILE,
        )
        mae_l2 = compute_mae_arm(rng, raw_energies, acceptance, p_err, EXACT_ENERGY, res_l2["surviving_mask"])

        # ── Arm 3: Level-1 (single-centroid RBF) ────────────────────
        res_l1 = process_telemetry_batch(
            iq, lf, force_mode="level_1",
            zero_centroid=ZERO_CENTROID, variance=RBF_VARIANCE,
            drop_percentile=DROP_PERCENTILE,
        )
        mae_l1 = compute_mae_arm(rng, raw_energies, acceptance, p_err, EXACT_ENERGY, res_l1["surviving_mask"])

        # ── Arm 4: Level-1-cluster (K-Means + RBF) ──────────────────
        res_clust = process_telemetry_batch(
            iq, lf, force_mode="level_1_cluster",
            k_clusters=K_CLUSTERS, variance=RBF_VARIANCE,
            drop_percentile=DROP_PERCENTILE,
        )
        mae_clust = compute_mae_arm(rng, raw_energies, acceptance, p_err, EXACT_ENERGY, res_clust["surviving_mask"])

        # TTS metric: (n_total / n_surviving) × MAE
        tts_none = 1.0 * mae_none
        tts_l2 = (N_SHOTS / max(res_l2["n_surviving"], 1)) * mae_l2
        tts_l1 = (N_SHOTS / max(res_l1["n_surviving"], 1)) * mae_l1
        tts_clust = (N_SHOTS / max(res_clust["n_surviving"], 1)) * mae_clust

        row = {
            "sigma_iq": float(sigma),
            "p_err": float(p_err),
            "snr": float(snr) if np.isfinite(snr) else 999.0,
            "mae_none": mae_none,
            "mae_level2": mae_l2,
            "mae_level1": mae_l1,
            "mae_cluster": mae_clust,
            "tts_none": tts_none,
            "tts_level2": tts_l2,
            "tts_level1": tts_l1,
            "tts_cluster": tts_clust,
            "surv_level2": res_l2["n_surviving"],
            "surv_level1": res_l1["n_surviving"],
            "surv_cluster": res_clust["n_surviving"],
        }
        results.append(row)

        print(
            f"  [{i+1:2d}/{N_SIGMA_STEPS}] σ_IQ={sigma:.3f}  SNR={row['snr']:6.2f}  "
            f"MAE: none={mae_none:.4f}  L2={mae_l2:.4f}  "
            f"L1={mae_l1:.4f}  cluster={mae_clust:.4f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Publication-ready plot
# ═══════════════════════════════════════════════════════════════════════════


def plot_crossover(results: List[Dict[str, Any]], out_path: Path) -> None:
    """Generate a two-panel crossover plot.

    Top panel:  MAE vs σ_IQ (4 curves, log-Y)
    Bottom panel: SNR vs σ_IQ (identifies the crossover region)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = np.array([r["sigma_iq"] for r in results])
    mae_none = np.array([r["mae_none"] for r in results])
    mae_l2 = np.array([r["mae_level2"] for r in results])
    mae_l1 = np.array([r["mae_level1"] for r in results])
    mae_clust = np.array([r["mae_cluster"] for r in results])
    snrs = np.array([r["snr"] for r in results])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # ── Top panel: MAE curves ─────────────────────────────────────────
    ax1.semilogy(sigmas, mae_none, "k--", linewidth=1.5, alpha=0.6, label="No TVS")
    ax1.semilogy(sigmas, mae_l2, "b-s", linewidth=2, markersize=5, label="Level-2 (binary)")
    ax1.semilogy(sigmas, mae_l1, "r-o", linewidth=2, markersize=5, label="Level-1 (RBF)")
    ax1.semilogy(sigmas, mae_clust, "g-^", linewidth=2, markersize=5, label="Level-1-cluster (K-Means)")

    # Mark crossover region
    # Find where cluster first beats level_1
    crossover_idx = None
    for idx in range(len(sigmas)):
        if mae_clust[idx] < mae_l1[idx]:
            crossover_idx = idx
            break

    if crossover_idx is not None:
        ax1.axvspan(
            sigmas[max(0, crossover_idx - 1)],
            sigmas[min(len(sigmas) - 1, crossover_idx + 1)],
            alpha=0.15, color="orange", label="Crossover region",
        )

    ax1.set_ylabel("MAE (log scale)", fontsize=12)
    ax1.set_title(
        "TVS Pipeline Noise Sweep — Crossover Benchmark\n"
        f"$N_{{shots}}$={N_SHOTS:,}, $k$={K_CLUSTERS}, "
        f"$\\sigma^2_{{RBF}}$={RBF_VARIANCE}",
        fontsize=13,
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # ── Bottom panel: SNR ─────────────────────────────────────────────
    ax2.plot(sigmas, snrs, "m-D", linewidth=2, markersize=5)
    ax2.axhline(y=3.0, color="orange", linestyle="--", alpha=0.7, label="SNR=3.0 threshold")
    ax2.set_xlabel("σ$_{IQ}$ (readout noise)", fontsize=12)
    ax2.set_ylabel("SNR", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Figure saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a condensed summary table."""
    print(f"\n{'═' * 70}")
    print("CROSSOVER BENCHMARK SUMMARY")
    print(f"{'═' * 70}")

    header = (
        f"{'σ_IQ':>6s}  {'SNR':>6s}  {'MAE(none)':>9s}  {'MAE(L2)':>9s}  "
        f"{'MAE(L1)':>9s}  {'MAE(clust)':>10s}  {'Winner':>8s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        arms = {
            "L2": r["mae_level2"],
            "L1": r["mae_level1"],
            "Cluster": r["mae_cluster"],
        }
        winner = min(arms, key=arms.get)
        print(
            f"  {r['sigma_iq']:5.3f}  {r['snr']:6.2f}  {r['mae_none']:9.4f}  "
            f"{r['mae_level2']:9.4f}  {r['mae_level1']:9.4f}  "
            f"{r['mae_cluster']:10.4f}  {winner:>8s}"
        )

    # Overall stats
    print(f"\n{'─' * 70}")

    # Count wins
    wins = {"L2": 0, "L1": 0, "Cluster": 0}
    for r in results:
        arms = {
            "L2": r["mae_level2"],
            "L1": r["mae_level1"],
            "Cluster": r["mae_cluster"],
        }
        winner = min(arms, key=arms.get)
        wins[winner] += 1

    print(f"  Win count: Level-2={wins['L2']}, Level-1={wins['L1']}, "
          f"Cluster={wins['Cluster']}  (out of {len(results)} noise levels)")

    # Average improvement
    avg_imp_l2 = np.mean([r["mae_none"] / max(r["mae_level2"], 1e-12) for r in results])
    avg_imp_l1 = np.mean([r["mae_none"] / max(r["mae_level1"], 1e-12) for r in results])
    avg_imp_cl = np.mean([r["mae_none"] / max(r["mae_cluster"], 1e-12) for r in results])
    print(f"  Avg MAE improvement vs baseline:  L2={avg_imp_l2:.2f}×  "
          f"L1={avg_imp_l1:.2f}×  Cluster={avg_imp_cl:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the full crossover benchmark."""
    t0 = time.time()

    results = run_sweep()

    # ── Save results JSON ─────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _project_root / "runs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"tvs_crossover_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "n_shots": N_SHOTS,
                "n_sigma_steps": N_SIGMA_STEPS,
                "sigma_range": [SIGMA_IQ_MIN, SIGMA_IQ_MAX],
                "k_clusters": K_CLUSTERS,
                "rbf_variance": RBF_VARIANCE,
                "seed": SEED,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  ✓ Results JSON: {json_path}")

    # ── Generate plot ─────────────────────────────────────────────────
    fig_path = out_dir / f"FIG_tvs_crossover_{ts}.png"
    plot_crossover(results, fig_path)

    # ── Summary ───────────────────────────────────────────────────────
    print_summary(results)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
