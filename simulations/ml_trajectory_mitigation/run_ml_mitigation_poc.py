#!/usr/bin/env python3
"""
run_ml_mitigation_poc.py — Two-Stage Quantum Error Mitigation PoC
                           (Trajectory Filtering + ML Regression)

Proof-of-concept for a Continuation-In-Part (CIP) patent filing:
combines the temporal-window Galton filter from the parent application
with a supervised-learning correction stage (telemetry-driven CDR).

Architecture
============
  Stage 1  ─  Galton Filter (outlier rejection)
      Sort execution windows by their fused telemetry score.
      Discard the bottom 30 % (catastrophic thermal bursts).

  Stage 2  ─  Telemetry-Driven Regression (CDR)
      Train a model on the *surviving* training windows:
        X = [Q_noisy, LF_score, HF_score]   →   y = Q_exact
      Apply the model to the surviving target windows to predict
      the noise-free expectation value.

Key result
----------
  The ML-mitigated estimator has lower absolute error than both
  the raw ensemble and the simple filtered ensemble.

Data generation
---------------
  All data is phenomenological — no Qiskit circuits.
  Temporal noise drift is modelled by a log-normal process that
  emulates TLS fluctuations and thermal bursts on real hardware.

Usage
-----
    python simulations/ml_trajectory_mitigation/run_ml_mitigation_poc.py

Dependencies
------------
    numpy, scikit-learn, matplotlib  (all pip-installable)

Patent reference
----------------
    Parent: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
    CIP addendum: ML-augmented trajectory filtering (two-stage pipeline)

CONFIDENTIAL — DO NOT PUSH TO PUBLIC REMOTE.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # headless backend for CI / SSH
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ═══════════════════════════════════════════════════════════════════════════
# 0. Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED                = 42
N_TRAIN_WINDOWS     = 1_000        # training-circuit windows
N_TARGET_WINDOWS    = 200          # target-circuit windows
GALTON_KEEP_FRAC    = 0.70         # keep top 70 % after Stage 1
Q_TARGET_EXACT      = 0.72         # true ⟨O⟩ for the target circuit

# Noise model — log-normal temporal drift
P_MEDIAN            = 0.03         # median depolarising rate
P_SIGMA             = 0.80         # log-normal σ  (higher → heavier tail)

# ML hyper-parameters
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 8
RIDGE_ALPHA         = 1.0

# Output directory (next to this script)
OUT_DIR = Path(__file__).resolve().parent / (
    f"results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Synthetic Temporal-Window Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_windows(
    n_windows: int,
    q_exact_values: np.ndarray,
    rng: np.random.Generator,
    *,
    p_median: float = P_MEDIAN,
    p_sigma: float = P_SIGMA,
) -> dict[str, np.ndarray]:
    """
    Simulate *n_windows* execution windows.

    For each window:
      • Draw a temporally drifting noise parameter  p ~ LogNormal.
      • Compute Q_noisy  =  Q_exact · (1 − 2p) + Gaussian shot noise.
      • Compute LF_score =  1 − α·p + small noise   (degrades → 0.5).
      • Compute HF_score =  1 − β·p + small noise   (degrades → 0.5).

    Returns a dict of arrays, one entry per feature column.
    """
    # --- temporal noise parameter (log-normal drift) ---
    p = rng.lognormal(mean=np.log(p_median), sigma=p_sigma, size=n_windows)
    p = np.clip(p, 1e-6, 0.50)            # cap at 0.5 (fully depolarised)

    # --- noisy observable ---
    attenuation = (1.0 - 2.0 * p)         # depolarising channel scaling
    shot_noise  = rng.normal(0, 0.02, size=n_windows)
    q_noisy     = q_exact_values * attenuation + shot_noise

    # --- telemetry scores (probe-bit entropy gauge) ---
    #   Ideal (p→0):  score → 1.0
    #   Fully noisy (p→0.5):  score → 0.5
    alpha, beta = 8.0, 12.0
    lf_score = np.clip(1.0 - alpha * p + rng.normal(0, 0.03, n_windows), 0, 1)
    hf_score = np.clip(1.0 - beta  * p + rng.normal(0, 0.03, n_windows), 0, 1)

    return {
        "p":        p,
        "Q_exact":  q_exact_values,
        "Q_noisy":  q_noisy,
        "LF_score": lf_score,
        "HF_score": hf_score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Stage 1 — Galton Filter  (outlier rejection)
# ═══════════════════════════════════════════════════════════════════════════

def fused_telemetry_score(lf: np.ndarray, hf: np.ndarray) -> np.ndarray:
    """Weighted combination of the two telemetry channels."""
    return 0.6 * lf + 0.4 * hf


def galton_filter(
    windows: dict[str, np.ndarray],
    keep_frac: float = GALTON_KEEP_FRAC,
) -> dict[str, np.ndarray]:
    """
    Sort windows by fused telemetry score and keep the top *keep_frac*.

    This removes the catastrophic thermal-burst windows where the
    signal is completely destroyed (high p, low scores).
    """
    score = fused_telemetry_score(windows["LF_score"], windows["HF_score"])
    n_keep = int(len(score) * keep_frac)
    top_idx = np.argsort(score)[-n_keep:]          # highest scores survive

    return {k: v[top_idx] for k, v in windows.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Stage 2 — Telemetry-Driven Regression  (CDR)
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_matrix(windows: dict[str, np.ndarray]) -> np.ndarray:
    """X = [Q_noisy, LF_score, HF_score]"""
    return np.column_stack([
        windows["Q_noisy"],
        windows["LF_score"],
        windows["HF_score"],
    ])


def train_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, object]:
    """Train both a Random-Forest and a Ridge regressor."""
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=SEED,
        n_jobs=-1,
    )
    ridge = Ridge(alpha=RIDGE_ALPHA)

    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    return {"RandomForest": rf, "Ridge": ridge}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    models: dict[str, object],
    raw_target: dict[str, np.ndarray],
    filtered_target: dict[str, np.ndarray],
    q_exact: float,
) -> dict:
    """
    Compare three estimators:
      1. Raw     — mean of *all* noisy target windows.
      2. Filtered — mean of surviving (post-Galton) noisy windows.
      3. ML      — mean of the model's predictions on surviving windows.
    """
    raw_mean       = float(np.mean(raw_target["Q_noisy"]))
    filtered_mean  = float(np.mean(filtered_target["Q_noisy"]))

    X_test = build_feature_matrix(filtered_target)

    results: dict = {
        "Q_exact":          q_exact,
        "raw_mean":         raw_mean,
        "raw_abs_error":    abs(raw_mean - q_exact),
        "filtered_mean":    filtered_mean,
        "filtered_abs_err": abs(filtered_mean - q_exact),
        "models":           {},
    }

    for name, model in models.items():
        preds = model.predict(X_test)
        ml_mean = float(np.mean(preds))
        mse     = float(mean_squared_error(filtered_target["Q_exact"], preds))
        results["models"][name] = {
            "ml_mean":      ml_mean,
            "ml_abs_error": abs(ml_mean - q_exact),
            "mse":          mse,
            "predictions":  preds,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_chart(results: dict, out_dir: Path) -> Path:
    """Bar chart: absolute error for Raw / Filtered / ML estimators."""
    best_model_name = min(
        results["models"],
        key=lambda n: results["models"][n]["ml_abs_error"],
    )
    best = results["models"][best_model_name]

    labels = ["Raw", "Filtered (Galton)", f"ML ({best_model_name})"]
    errors = [
        results["raw_abs_error"],
        results["filtered_abs_err"],
        best["ml_abs_error"],
    ]
    colours = ["#d62728", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, errors, color=colours, edgecolor="black", width=0.55)
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Absolute Error  |⟨O⟩_est − ⟨O⟩_exact|")
    ax.set_title("Two-Stage Mitigation: Raw → Galton → ML")
    ax.set_ylim(0, max(errors) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = out_dir / "bar_absolute_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scatter(results: dict, out_dir: Path) -> Path:
    """Scatter: ML predicted Q vs true Q for the surviving target windows."""
    best_model_name = min(
        results["models"],
        key=lambda n: results["models"][n]["mse"],
    )
    preds = results["models"][best_model_name]["predictions"]
    true  = results["_filtered_target_Q_exact"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true, preds, alpha=0.45, s=18, c="#1f77b4", edgecolors="none")

    lo = min(true.min(), preds.min()) - 0.05
    hi = max(true.max(), preds.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal y = x")

    ax.set_xlabel("True  Q_exact")
    ax.set_ylabel(f"Predicted  Q  ({best_model_name})")
    ax.set_title("Stage-2 ML Regression: Predicted vs. True")
    ax.legend(loc="upper left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()

    path = out_dir / "scatter_predicted_vs_true.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_noise_vs_score(train_windows: dict, out_dir: Path) -> Path:
    """Extra diagnostic: telemetry score vs. underlying noise p."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, key, label, clr in [
        (axes[0], "LF_score", "LF score", "#1f77b4"),
        (axes[1], "HF_score", "HF score", "#ff7f0e"),
    ]:
        ax.scatter(train_windows["p"], train_windows[key],
                   alpha=0.3, s=8, c=clr, edgecolors="none")
        ax.set_xlabel("Noise parameter  p")
        ax.set_ylabel("Telemetry score")
        ax.set_title(label)
        ax.axhline(0.5, ls=":", c="grey", lw=0.8, label="baseline 0.5")
        ax.legend(fontsize=8)

    fig.suptitle("Telemetry Gauge vs. Temporal Noise Drift", fontsize=12)
    fig.tight_layout()
    path = out_dir / "noise_vs_telemetry.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  Two-Stage Quantum Error Mitigation PoC")
    print("  (Galton Trajectory Filter  +  ML Regression)")
    print("=" * 68)

    # ── Step 1: generate synthetic temporal windows ───────────────────────
    print("\n▸ Step 1 — Generating synthetic temporal dataset …")

    # Training circuits: each has a random true expectation value
    q_exact_train = rng.uniform(0.3, 0.9, size=N_TRAIN_WINDOWS)
    train_windows = generate_windows(N_TRAIN_WINDOWS, q_exact_train, rng)

    # Target circuit: fixed Q_exact for all windows
    q_exact_target_arr = np.full(N_TARGET_WINDOWS, Q_TARGET_EXACT)
    target_windows = generate_windows(N_TARGET_WINDOWS, q_exact_target_arr, rng)

    print(f"    Training windows : {N_TRAIN_WINDOWS}")
    print(f"    Target windows   : {N_TARGET_WINDOWS}")
    print(f"    Median noise p   : {P_MEDIAN}")
    print(f"    Log-normal σ     : {P_SIGMA}")

    # ── Step 2: Stage 1 — Galton filter ──────────────────────────────────
    print(f"\n▸ Step 2 — Galton filter (keep top {GALTON_KEEP_FRAC*100:.0f}%) …")

    train_filt   = galton_filter(train_windows)
    target_filt  = galton_filter(target_windows)

    print(f"    Training: {N_TRAIN_WINDOWS} → {len(train_filt['p'])} windows")
    print(f"    Target  : {N_TARGET_WINDOWS} → {len(target_filt['p'])} windows")
    print(f"    Mean noise p (before filter): {target_windows['p'].mean():.4f}")
    print(f"    Mean noise p (after  filter): {target_filt['p'].mean():.4f}")

    # ── Step 3: Stage 2 — train ML regressors ────────────────────────────
    print("\n▸ Step 3 — Training telemetry-driven regressors …")

    X_train = build_feature_matrix(train_filt)
    y_train = train_filt["Q_exact"]
    models  = train_regressors(X_train, y_train)

    for name, mdl in models.items():
        train_preds = mdl.predict(X_train)
        train_mse = mean_squared_error(y_train, train_preds)
        print(f"    {name:20s}  train MSE = {train_mse:.6f}")

    # ── Step 4: prediction & evaluation ──────────────────────────────────
    print("\n▸ Step 4 — Evaluating on target circuit …")

    results = evaluate(models, target_windows, target_filt, Q_TARGET_EXACT)
    results["_filtered_target_Q_exact"] = target_filt["Q_exact"]

    print(f"    Q_exact (true)         = {Q_TARGET_EXACT:.4f}")
    print(f"    Raw mean               = {results['raw_mean']:.4f}  "
          f"(err {results['raw_abs_error']:.4f})")
    print(f"    Filtered mean          = {results['filtered_mean']:.4f}  "
          f"(err {results['filtered_abs_err']:.4f})")
    for name, m in results["models"].items():
        print(f"    ML [{name:14s}]  = {m['ml_mean']:.4f}  "
              f"(err {m['ml_abs_error']:.4f}, MSE {m['mse']:.6f})")

    # ── Step 5: plots ────────────────────────────────────────────────────
    print("\n▸ Step 5 — Generating plots …")
    p1 = plot_bar_chart(results, OUT_DIR)
    p2 = plot_scatter(results, OUT_DIR)
    p3 = plot_noise_vs_score(train_windows, OUT_DIR)
    print(f"    {p1}")
    print(f"    {p2}")
    print(f"    {p3}")

    # ── Save numerical results ───────────────────────────────────────────
    serialisable = {
        k: v for k, v in results.items()
        if not k.startswith("_")
    }
    for mname in serialisable["models"]:
        serialisable["models"][mname] = {
            k: v for k, v in serialisable["models"][mname].items()
            if k != "predictions"
        }
    serialisable["config"] = {
        "N_TRAIN_WINDOWS": N_TRAIN_WINDOWS,
        "N_TARGET_WINDOWS": N_TARGET_WINDOWS,
        "GALTON_KEEP_FRAC": GALTON_KEEP_FRAC,
        "Q_TARGET_EXACT": Q_TARGET_EXACT,
        "P_MEDIAN": P_MEDIAN,
        "P_SIGMA": P_SIGMA,
        "RF_N_ESTIMATORS": RF_N_ESTIMATORS,
        "RF_MAX_DEPTH": RF_MAX_DEPTH,
        "RIDGE_ALPHA": RIDGE_ALPHA,
        "SEED": SEED,
    }
    serialisable["timestamp"] = datetime.now(timezone.utc).isoformat()

    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n    Results JSON → {json_path}")

    elapsed = time.perf_counter() - t0
    print(f"\n✓ Done in {elapsed:.2f}s.  Output → {OUT_DIR}\n")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
