#!/usr/bin/env python3
"""
run_real_data_ml.py — Two-Stage ML Mitigation on REAL Experiment Data
=====================================================================

Feeds actual experimental data from IBM Quantum hardware runs and
Aer-calibrated noise-model simulations into the two-stage
Galton-filter + ML-regression pipeline.

Data sources (all from this repository)
----------------------------------------
A.  **Noise-sweep experiment** (``results/noise_sweep_8q_15t_*.json``)
    • 8-qubit TFIM at the quantum critical point (J=1, h=1)
    • 7 depolarising noise levels × 3 estimators × 15 independent trials
    • Per-trial energy values + acceptance probabilities
    • Exact ground-state energy known: E₀ = −24.898 …

B.  **Cross-algorithm experiment** (``results/cross_algo_8q_15t_*.json``)
    • VQE / QAOA / Grover, same noise model, 15 trials each

C.  **IBM Hardware telemetry** (``simulations/ibm_hardware/results.csv``)
    • 120 rows from IBM Marrakesh: mean_score, acceptance_probability,
      TTS across multiple (N, D, W) configurations

D.  **IBM Fez Galton experiment** (``galton_results.csv``)
    • 48 rows: mean_combined_score, acceptance, N/W/D sweep

E.  **VQE-TSVF IBM Fez** (``vqe_tsvf_results.csv`` + telemetry JSONL)
    • 5 layer configs × (standard + TSVF) = 10 telemetry entries
    • Exact ground-state energy: −4.7588

F.  **QAOA-TSVF IBM Torino** (``qaoa_tsvf_results.csv`` + telemetry)
    • 5 layer configs × (standard + TSVF) = 10 telemetry entries

Pipeline
--------
1.  Load & unify all data into a flat DataFrame.
2.  Stage 1 — Galton filter: discard bottom 30 % by acceptance score.
3.  Stage 2 — Train RandomForest + Ridge on surviving training rows.
4.  Predict the true TFIM energy for the target 8-qubit circuit.
5.  Plot bar chart + scatter + feature-importance.

Usage
-----
    python simulations/ml_trajectory_mitigation/run_real_data_ml.py

CONFIDENTIAL — DO NOT PUSH TO PUBLIC REMOTE.
Patent ref: US 63/983,831 & 63/989,632 | IL 326915 | CIP: ML-augmented TSVF
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR        = ROOT / "results"
IBM_HW_DIR         = ROOT / "simulations" / "ibm_hardware"
GALTON_DIR         = IBM_HW_DIR / "galton_experiment"
VQE_TSVF_DIR       = ROOT / "simulations" / "vqe_tsvf" / "results_20260302_080312"
QAOA_TSVF_DIR      = ROOT / "simulations" / "qaoa_tsvf" / "results_20260302_074221"

OUT_DIR = Path(__file__).resolve().parent / (
    f"real_data_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED               = 42
GALTON_KEEP_FRAC   = 0.70      # Stage 1: keep top 70 %
RF_N_ESTIMATORS    = 300
RF_MAX_DEPTH       = 10
GB_N_ESTIMATORS    = 200
GB_MAX_DEPTH       = 5
GB_LEARNING_RATE   = 0.05
RIDGE_ALPHA        = 1.0

# Known exact energies
EXACT_ENERGY_8Q    = -24.89848442198773     # 8-qubit TFIM (noise sweep)
EXACT_ENERGY_4Q    = -4.7587704831436355    # 4-qubit TFIM (VQE-TSVF Fez)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_noise_sweep() -> pd.DataFrame:
    """
    Load the 8-qubit TFIM noise-sweep experiment.

    Returns one row per (noise_level, estimator, trial) with columns:
      source, noise_label, noise_level, estimator, trial_idx,
      energy, acceptance, exact_energy
    """
    path = RESULTS_DIR / "noise_sweep_8q_15t_20260304_221252.json"
    with open(path) as f:
        data = json.load(f)

    exact = data["exact_energy"]
    noise_levels = data["noise_levels"]   # [0.0, 1e-4, …, 5e-2]

    rows: list[dict] = []
    for nl_key, nl_data in data["results"].items():
        # Parse numeric noise level from the key
        if nl_key == "ideal":
            nl_val = 0.0
        else:
            nl_val = float(nl_key.split("=")[1])

        for est_name, est_data in nl_data.items():
            acc  = est_data["mean_acceptance"]
            vals = est_data.get("values", [])
            for i, energy in enumerate(vals):
                rows.append({
                    "source":       "noise_sweep",
                    "noise_label":  nl_key,
                    "noise_level":  nl_val,
                    "estimator":    est_name,
                    "trial_idx":    i,
                    "energy":       energy,
                    "acceptance":   acc,
                    "mean_value":   est_data["mean_value"],
                    "variance":     est_data["variance"],
                    "mse_reported": est_data["mse"],
                    "exact_energy": exact,
                })
    return pd.DataFrame(rows)


def load_cross_algo() -> pd.DataFrame:
    """
    Load the cross-algorithm validation experiment (VQE, QAOA, Grover).

    Note: exact energies are not stored per-algorithm in the JSON, but
    we know the VQE uses the same 8-qubit TFIM as the noise sweep.
    """
    path = RESULTS_DIR / "cross_algo_8q_15t_20260306_174443.json"
    with open(path) as f:
        data = json.load(f)

    rows: list[dict] = []
    for algo, algo_data in data["results"].items():
        # VQE exact energy is known; for QAOA/Grover we leave as NaN
        # (they measure different observables)
        exact = EXACT_ENERGY_8Q if algo == "vqe" else np.nan

        for est_name, est_data in algo_data.items():
            acc  = est_data["mean_acceptance"]
            vals = est_data.get("values", [])
            for i, energy in enumerate(vals):
                rows.append({
                    "source":       f"cross_algo_{algo}",
                    "noise_label":  "depol_1q=1e-03",
                    "noise_level":  1e-3,
                    "estimator":    est_name,
                    "trial_idx":    i,
                    "energy":       energy,
                    "acceptance":   acc,
                    "mean_value":   est_data["mean_value"],
                    "variance":     est_data["variance"],
                    "mse_reported": est_data["mse"],
                    "exact_energy": exact,
                })
    return pd.DataFrame(rows)


def load_marrakesh_csv() -> pd.DataFrame:
    """Load IBM Marrakesh conditioning sweep (120 rows)."""
    path = IBM_HW_DIR / "results.csv"
    df = pd.read_csv(path)
    df["source"] = "ibm_marrakesh"
    return df


def load_galton_fez_csv() -> pd.DataFrame:
    """Load IBM Fez Galton threshold sweep (48 rows)."""
    path = GALTON_DIR / "galton_results.csv"
    df = pd.read_csv(path)
    df["source"] = "ibm_fez_galton"
    return df


def load_vqe_tsvf() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load VQE-TSVF IBM Fez results:
      - CSV with per-layer energy + qgate telemetry
      - JSONL with detailed qgate telemetry per configuration
    """
    csv_path = VQE_TSVF_DIR / "vqe_tsvf_results.csv"
    df_csv = pd.read_csv(csv_path)
    df_csv["source"] = "vqe_tsvf_ibm_fez"
    df_csv["exact_energy"] = EXACT_ENERGY_4Q

    jsonl_path = VQE_TSVF_DIR / "vqe_tsvf_telemetry.jsonl"
    tele_rows = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            md = entry.get("metadata", {})
            galton = md.get("galton", {})
            tele_rows.append({
                "source":            "vqe_tsvf_ibm_fez",
                "algorithm":         md.get("algorithm", "unknown"),
                "layers":            md.get("layers", 0),
                "total_shots":       entry["total_shots"],
                "accepted_shots":    entry["accepted_shots"],
                "acceptance":        entry["acceptance_probability"],
                "mean_combined_score": entry["mean_combined_score"],
                "threshold":         entry["threshold_used"],
                "galton_eff_thresh": galton.get("galton_effective_threshold", None),
                "galton_accept_rolling": galton.get("galton_acceptance_rate_rolling", None),
            })
    df_tele = pd.DataFrame(tele_rows)
    return df_csv, df_tele


def load_qaoa_tsvf() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load QAOA-TSVF IBM Torino results."""
    csv_path = QAOA_TSVF_DIR / "qaoa_tsvf_results.csv"
    df_csv = pd.read_csv(csv_path)
    df_csv["source"] = "qaoa_tsvf_ibm_torino"

    jsonl_path = QAOA_TSVF_DIR / "qaoa_tsvf_telemetry.jsonl"
    tele_rows = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            md = entry.get("metadata", {})
            galton = md.get("galton", {})
            tele_rows.append({
                "source":            "qaoa_tsvf_ibm_torino",
                "algorithm":         md.get("algorithm", "unknown"),
                "layers":            md.get("layers", 0),
                "total_shots":       entry["total_shots"],
                "accepted_shots":    entry["accepted_shots"],
                "acceptance":        entry["acceptance_probability"],
                "mean_combined_score": entry["mean_combined_score"],
                "threshold":         entry["threshold_used"],
                "galton_eff_thresh": galton.get("galton_effective_threshold", None),
                "galton_accept_rolling": galton.get("galton_acceptance_rate_rolling", None),
            })
    df_tele = pd.DataFrame(tele_rows)
    return df_csv, df_tele


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def build_ml_dataset(
    df_sweep: pd.DataFrame,
    df_cross: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the unified ML feature table from the per-trial data.

    The ML target is the **correction term**:
        correction = exact_energy − energy
    i.e. how far the noisy measurement is from truth.  The model
    learns the transfer function from telemetry features to the
    magnitude of decoherence-induced bias, then at inference we
    compute:  E_predicted = energy + model.predict(features).

    This is non-trivial because the correction varies dramatically
    across noise levels (from ~0 at ideal to ~25 at depol=0.05)
    and across estimators (raw vs ancilla vs ancilla+Galton).

    Features:
      - energy              : measured (noisy) energy for this trial
      - acceptance           : qgate acceptance probability (telemetry)
      - estimator_code       : {raw=0, ancilla=1, ancilla_galton=2}
      - variance             : cross-trial variance of this estimator
      - residual_from_mean   : energy − estimator's mean
      - acceptance_x_energy  : interaction term
      - abs_energy           : |energy|

    Target:
      - correction  = exact_energy − energy
    """
    # Combine noise sweep + cross-algo VQE (which share the same TFIM target)
    df = pd.concat([df_sweep, df_cross], ignore_index=True)

    # Only keep rows with a known exact energy (VQE target)
    df = df.dropna(subset=["exact_energy"]).copy()

    # Encode estimator as ordinal
    est_map = {"raw": 0, "ancilla": 1, "ancilla_galton": 2}
    df["estimator_code"] = df["estimator"].map(est_map)

    # Engineered features
    df["residual_from_mean"] = df["energy"] - df["mean_value"]
    df["acceptance_x_energy"] = df["acceptance"] * df["energy"]
    df["abs_energy"] = df["energy"].abs()

    # ML target: the correction the model must learn
    df["correction"] = df["exact_energy"] - df["energy"]

    return df


FEATURE_COLS = [
    "energy",
    "acceptance",
    "estimator_code",
    "variance",
    "residual_from_mean",
    "acceptance_x_energy",
    "abs_energy",
]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Stage 1 — Galton filter
# ═══════════════════════════════════════════════════════════════════════════

def galton_filter(df: pd.DataFrame, keep_frac: float) -> pd.DataFrame:
    """
    Sort rows by acceptance (telemetry quality score) and keep the
    top *keep_frac* fraction — mimicking the Galton adaptive threshold.

    Within each noise level we rank by acceptance × |energy| to prefer
    high-acceptance AND non-trivial energy readings.
    """
    # Composite quality score: higher acceptance is better, and
    # we slightly prefer trials whose energy is farther from zero
    # (near-zero energy usually means fully depolarised / no signal).
    df = df.copy()
    df["quality_score"] = df["acceptance"] * (1.0 + df["abs_energy"] / 10.0)

    n_keep = int(len(df) * keep_frac)
    df_sorted = df.sort_values("quality_score", ascending=False)
    return df_sorted.head(n_keep).copy()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Stage 2 — Train regressors
# ═══════════════════════════════════════════════════════════════════════════

def train_models(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    """Train three regressors and return them."""
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=GB_N_ESTIMATORS,
            max_depth=GB_MAX_DEPTH,
            learning_rate=GB_LEARNING_RATE,
            random_state=SEED,
        ),
        "Ridge": Ridge(alpha=RIDGE_ALPHA),
    }
    for name, mdl in models.items():
        mdl.fit(X, y)
    return models


# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_hold_out(
    models: dict,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scaler: StandardScaler,
) -> dict:
    """
    Evaluate:
      (a) Raw baseline: mean of all noisy energies in the test set.
      (b) Galton-only: mean of post-filter noisy energies.
      (c) ML-mitigated: energy + predicted correction on filtered test set.
    """
    exact = df_test["exact_energy"].iloc[0]

    # (a) Raw — before any filtering
    raw_mean = df_test["energy"].mean()
    raw_err  = abs(raw_mean - exact)

    # (b) Galton-only
    df_test_filt = galton_filter(df_test, GALTON_KEEP_FRAC)
    filt_mean = df_test_filt["energy"].mean()
    filt_err  = abs(filt_mean - exact)

    # (c) ML-mitigated on filtered test set
    #     E_corrected = energy + model.predict(correction)
    X_test = scaler.transform(df_test_filt[FEATURE_COLS].values)
    energies_filt = df_test_filt["energy"].values
    true_corrections = df_test_filt["correction"].values

    model_results = {}
    for name, mdl in models.items():
        pred_corrections = mdl.predict(X_test)
        corrected_energies = energies_filt + pred_corrections
        ml_mean = float(np.mean(corrected_energies))
        ml_err  = abs(ml_mean - exact)

        # MSE/MAE of the correction prediction
        mse = float(mean_squared_error(true_corrections, pred_corrections))
        mae = float(mean_absolute_error(true_corrections, pred_corrections))

        model_results[name] = {
            "ml_mean":              ml_mean,
            "ml_abs_err":           ml_err,
            "correction_mse":       mse,
            "correction_mae":       mae,
            "mean_pred_correction": float(np.mean(pred_corrections)),
            "mean_true_correction": float(np.mean(true_corrections)),
            "corrected_energies":   corrected_energies,
        }

    return {
        "exact":            exact,
        "n_test_total":     len(df_test),
        "n_test_filtered":  len(df_test_filt),
        "raw_mean":         raw_mean,
        "raw_abs_err":      raw_err,
        "filt_mean":        filt_mean,
        "filt_abs_err":     filt_err,
        "models":           model_results,
        "_df_test_filt":    df_test_filt,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(results: dict, out_dir: Path) -> Path:
    """Bar chart comparing absolute errors of Raw / Filtered / ML."""
    best_name = min(results["models"],
                    key=lambda n: results["models"][n]["ml_abs_err"])
    best = results["models"][best_name]

    labels = ["Raw Ensemble", "Galton-Filtered", f"ML ({best_name})"]
    errors = [results["raw_abs_err"], results["filt_abs_err"], best["ml_abs_err"]]
    colours = ["#d62728", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, errors, color=colours, edgecolor="black", width=0.55)
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(errors) * 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylabel("Absolute Error  |E_est − E_exact|", fontsize=12)
    ax.set_title(
        "Two-Stage Mitigation on Real Experiment Data\n"
        f"(8Q TFIM, E₀ = {results['exact']:.4f})",
        fontsize=13,
    )
    ax.set_ylim(0, max(errors) * 1.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = out_dir / "bar_real_data_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scatter_pred_vs_true(results: dict, out_dir: Path) -> Path:
    """Scatter: ML corrected E vs. true E for test set."""
    best_name = min(results["models"],
                    key=lambda n: results["models"][n]["correction_mse"])
    corrected = results["models"][best_name]["corrected_energies"]
    true      = results["_df_test_filt"]["exact_energy"].values

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true, corrected, alpha=0.6, s=30, c="#1f77b4", edgecolors="none")

    lo = min(true.min(), corrected.min()) - 1.0
    hi = max(true.max(), corrected.max()) + 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal y = x")

    ax.set_xlabel("True E_exact", fontsize=11)
    ax.set_ylabel(f"Predicted E ({best_name})", fontsize=11)
    ax.set_title("Stage-2 ML Regression: Predicted vs. True", fontsize=12)
    ax.legend(loc="upper left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()

    path = out_dir / "scatter_pred_vs_true.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_feature_importance(models: dict, out_dir: Path) -> Path:
    """Feature importance from the RandomForest model."""
    rf = models.get("RandomForest")
    if rf is None:
        return None

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_COLS[i] for i in indices]
    sorted_imps  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(sorted_names)), sorted_imps[::-1],
            color="#4c72b0", edgecolor="black")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("RandomForest Feature Importance (Real Data)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_noise_level_analysis(df_all: pd.DataFrame, models: dict,
                              scaler: StandardScaler, out_dir: Path) -> Path:
    """
    Per-noise-level comparison: Raw vs Galton vs ML.
    Shows how each stage improves the estimate at each noise intensity.
    """
    exact = df_all["exact_energy"].iloc[0]
    noise_levels = sorted(df_all["noise_level"].unique())

    raw_errs, filt_errs, ml_errs = [], [], []
    for nl in noise_levels:
        subset = df_all[df_all["noise_level"] == nl]
        raw_errs.append(abs(subset["energy"].mean() - exact))

        filt = galton_filter(subset, GALTON_KEEP_FRAC)
        filt_errs.append(abs(filt["energy"].mean() - exact))

        X = scaler.transform(filt[FEATURE_COLS].values)
        best_name = "RandomForest"
        pred_corrections = models[best_name].predict(X)
        corrected = filt["energy"].values + pred_corrections
        ml_errs.append(abs(np.mean(corrected) - exact))

    x = np.arange(len(noise_levels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, raw_errs, width, label="Raw", color="#d62728",
           edgecolor="black")
    ax.bar(x,         filt_errs, width, label="Galton-Filtered", color="#ff7f0e",
           edgecolor="black")
    ax.bar(x + width, ml_errs, width, label="ML-Mitigated (RF)", color="#2ca02c",
           edgecolor="black")

    ax.set_xticks(x)
    labels = [f"{nl:.0e}" if nl > 0 else "ideal" for nl in noise_levels]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Depolarising Noise Level (1Q)", fontsize=11)
    ax.set_ylabel("|E_est − E_exact|", fontsize=11)
    ax.set_title("Per-Noise-Level: Raw → Galton → ML", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = out_dir / "per_noise_level_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Two-Stage ML Mitigation — REAL Experiment Data")
    print("  (Galton Trajectory Filter + Telemetry-Driven Regression)")
    print("=" * 72)
    print(f"\n  ⚠  CONFIDENTIAL — DO NOT PUSH\n")

    # ── Load all data ─────────────────────────────────────────────────────
    print("▸ Loading experimental data …")

    df_sweep = load_noise_sweep()
    print(f"    Noise sweep    : {len(df_sweep):>5} rows  "
          f"({df_sweep['noise_label'].nunique()} noise levels × "
          f"{df_sweep['estimator'].nunique()} estimators × "
          f"{df_sweep['trial_idx'].nunique()} trials)")

    df_cross = load_cross_algo()
    n_vqe = len(df_cross[df_cross["source"] == "cross_algo_vqe"])
    print(f"    Cross-algorithm: {len(df_cross):>5} rows  "
          f"(VQE={n_vqe}, QAOA+Grover={len(df_cross)-n_vqe} — "
          f"only VQE has known exact)")

    df_marrakesh = load_marrakesh_csv()
    print(f"    IBM Marrakesh  : {len(df_marrakesh):>5} rows")

    df_galton_fez = load_galton_fez_csv()
    print(f"    IBM Fez Galton : {len(df_galton_fez):>5} rows")

    vqe_csv, vqe_tele = load_vqe_tsvf()
    print(f"    VQE-TSVF Fez   : {len(vqe_csv):>5} CSV + "
          f"{len(vqe_tele)} telemetry")

    qaoa_csv, qaoa_tele = load_qaoa_tsvf()
    print(f"    QAOA-TSVF Torino: {len(qaoa_csv):>4} CSV + "
          f"{len(qaoa_tele)} telemetry")

    # ── Build unified ML dataset ──────────────────────────────────────────
    print("\n▸ Building ML feature table …")
    df_all = build_ml_dataset(df_sweep, df_cross)
    print(f"    Total rows (with known exact): {len(df_all)}")
    print(f"    Features: {FEATURE_COLS}")

    # ── Train / test split ────────────────────────────────────────────────
    # Strategy: Hold out two noise levels the model has NEVER seen
    # (1e-3 from noise_sweep + the cross_algo VQE run at the same noise).
    # This is the hardest, fairest test: the model must generalise to an
    # entirely unseen noise regime AND an independent experiment.
    print("\n▸ Splitting: hold out depol_1q=1e-3 and cross_algo_vqe …")

    HOLD_OUT_NOISE = {1e-3}     # 1e-3 is realistic IBM-class noise

    mask_test = (
        (df_all["source"] == "cross_algo_vqe")
        | ((df_all["source"] == "noise_sweep")
           & (df_all["noise_level"].isin(HOLD_OUT_NOISE)))
    )
    mask_train = ~mask_test

    df_train_raw = df_all[mask_train].copy()
    df_test_raw  = df_all[mask_test].copy()
    print(f"    Train (other noise levels) : {len(df_train_raw)} rows")
    print(f"    Test  (held-out 1e-3 + xalgo): {len(df_test_raw)} rows")
    print(f"    Train noise levels: "
          f"{sorted(df_train_raw['noise_level'].unique())}")
    print(f"    Test  noise levels: "
          f"{sorted(df_test_raw['noise_level'].unique())}")

    # ── Stage 1: Galton filter on training data ───────────────────────────
    print(f"\n▸ Stage 1 — Galton filter (keep top {GALTON_KEEP_FRAC*100:.0f}%) …")
    df_train = galton_filter(df_train_raw, GALTON_KEEP_FRAC)
    print(f"    Training: {len(df_train_raw)} → {len(df_train)} rows")
    print(f"    Dropped {len(df_train_raw) - len(df_train)} low-quality "
          f"trial windows")

    # Show quality improvement
    pre_mean_acc  = df_train_raw["acceptance"].mean()
    post_mean_acc = df_train["acceptance"].mean()
    print(f"    Mean acceptance: {pre_mean_acc:.4f} → {post_mean_acc:.4f}")

    # ── Stage 2: Train ML regressors ──────────────────────────────────────
    print("\n▸ Stage 2 — Training correction regressors on surviving data …")
    print(f"    Target: correction = exact_energy − energy")
    print(f"    Correction range: [{df_train['correction'].min():.2f}, "
          f"{df_train['correction'].max():.2f}]")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
    y_train = df_train["correction"].values

    models = train_models(X_train, y_train)

    # Cross-validation on training set
    for name, mdl in models.items():
        cv_scores = cross_val_score(
            mdl, X_train, y_train, cv=5,
            scoring="neg_mean_absolute_error",
        )
        print(f"    {name:22s}  5-fold CV MAE = {-cv_scores.mean():.4f} "
              f"(±{cv_scores.std():.4f})")

    # ── Evaluate on hold-out test set ─────────────────────────────────────
    print("\n▸ Evaluating on hold-out test set (cross-algo VQE) …")

    results = evaluate_hold_out(models, df_train, df_test_raw, scaler)

    print(f"    E_exact (true)         = {results['exact']:.4f}")
    print(f"    Raw mean               = {results['raw_mean']:.4f}  "
          f"(err {results['raw_abs_err']:.4f})")
    print(f"    Galton-filtered mean   = {results['filt_mean']:.4f}  "
          f"(err {results['filt_abs_err']:.4f})")
    for name, m in results["models"].items():
        print(f"    ML [{name:18s}] = {m['ml_mean']:.4f}  "
              f"(err {m['ml_abs_err']:.4f}, "
              f"correction_MAE {m['correction_mae']:.4f})")

    # Best model summary
    best_name = min(results["models"],
                    key=lambda n: results["models"][n]["ml_abs_err"])
    best_err = results["models"][best_name]["ml_abs_err"]
    raw_err  = results["raw_abs_err"]
    improvement = (1.0 - best_err / raw_err) * 100 if raw_err > 0 else 0
    print(f"\n    ★ Best model: {best_name}")
    print(f"    ★ Error reduction: {raw_err:.4f} → {best_err:.4f}  "
          f"({improvement:.1f}% improvement)")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n▸ Generating plots …")
    p1 = plot_bar_comparison(results, OUT_DIR)
    print(f"    {p1}")
    p2 = plot_scatter_pred_vs_true(results, OUT_DIR)
    print(f"    {p2}")
    p3 = plot_feature_importance(models, OUT_DIR)
    if p3:
        print(f"    {p3}")
    p4 = plot_noise_level_analysis(df_all[mask_train], models, scaler, OUT_DIR)
    print(f"    {p4}")

    # ── Hardware telemetry summary table ──────────────────────────────────
    print("\n▸ Hardware telemetry summary (for reference) …")
    print(f"    IBM Marrakesh: {len(df_marrakesh)} configs, "
          f"mean acceptance = "
          f"{df_marrakesh['acceptance_probability'].astype(float).mean():.4f}")
    print(f"    IBM Fez Galton: {len(df_galton_fez)} configs, "
          f"mean score = "
          f"{df_galton_fez['mean_combined_score'].astype(float).mean():.4f}")

    # Show VQE hardware vs ML prediction
    print("\n▸ Cross-reference: VQE-TSVF IBM Fez hardware energies …")
    print(f"    Exact 4Q TFIM  = {EXACT_ENERGY_4Q:.4f}")
    for _, row in vqe_csv.iterrows():
        print(f"    L={int(row['layers'])}: E_std={row['energy_std']:.3f}, "
              f"E_tsvf={row['energy_tsvf']:.3f}, "
              f"score_std={row['qgate_mean_score_std']:.3f}, "
              f"score_tsvf={row['qgate_mean_score_tsvf']:.3f}")

    # ── Save results JSON ─────────────────────────────────────────────────
    out_json = {
        "experiment":   "two_stage_ml_mitigation_real_data",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "exact_energy": results["exact"],
        "raw_mean":     results["raw_mean"],
        "raw_abs_err":  results["raw_abs_err"],
        "filt_mean":    results["filt_mean"],
        "filt_abs_err": results["filt_abs_err"],
        "models": {
            name: {k: v for k, v in mdata.items()
                   if k != "corrected_energies"}
            for name, mdata in results["models"].items()
        },
        "best_model":       best_name,
        "improvement_pct":  improvement,
        "config": {
            "GALTON_KEEP_FRAC": GALTON_KEEP_FRAC,
            "RF_N_ESTIMATORS":  RF_N_ESTIMATORS,
            "RF_MAX_DEPTH":     RF_MAX_DEPTH,
            "GB_N_ESTIMATORS":  GB_N_ESTIMATORS,
            "GB_MAX_DEPTH":     GB_MAX_DEPTH,
            "RIDGE_ALPHA":      RIDGE_ALPHA,
            "SEED":             SEED,
            "n_train":          len(df_train),
            "n_test":           len(df_test_raw),
            "n_test_filtered":  results["n_test_filtered"],
        },
        "data_sources": {
            "noise_sweep":     str(RESULTS_DIR / "noise_sweep_8q_15t_20260304_221252.json"),
            "cross_algo":      str(RESULTS_DIR / "cross_algo_8q_15t_20260306_174443.json"),
            "ibm_marrakesh":   str(IBM_HW_DIR / "results.csv"),
            "ibm_fez_galton":  str(GALTON_DIR / "galton_results.csv"),
            "vqe_tsvf_fez":    str(VQE_TSVF_DIR),
            "qaoa_tsvf_torino": str(QAOA_TSVF_DIR),
        },
        "patent_ref": "US 63/983,831 & 63/989,632 | IL 326915 | CIP addendum",
    }
    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n    Results JSON → {json_path}")

    elapsed = time.perf_counter() - t0
    print(f"\n✓ Done in {elapsed:.2f}s.  Output → {OUT_DIR}")
    print(f"\n  ⚠  REMINDER: DO NOT PUSH — CIP filing required first.\n")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
