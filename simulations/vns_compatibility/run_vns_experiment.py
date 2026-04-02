#!/usr/bin/env python3
"""
run_vns_experiment.py — Virtual Noise Scaling (VNS) + qgate Trajectory Filtering.

Proves that trajectory filtering via QgateSampler is compatible with
noise amplification and calculates the cost efficiency η(λ).

Two complementary analyses:

  **Analysis A — QgateSampler (shot-level Galton filter):**
    Uses the drop-in QgateSampler to filter individual shots via the
    energy-probe ancilla + Galton adaptive threshold.

  **Analysis B — Window-level filtering (as described in the email):**
    Runs N_WINDOWS independent batches of WINDOW_SIZE shots each.
    Each window gets an independent noise-model realisation with
    stochastic drift (simulating TLS fluctuations / thermal drift on
    real hardware).  The probe-bit bias per window serves as a noise
    thermometer.  We select the top-k% of windows (least noisy) and
    compute the estimator from those windows only — this is the actual
    mechanism described in the correspondence with Raam.

Experiment:
  1. Build a 4-qubit TFIM VQE ansatz circuit (1–2 layers).
  2. Compute exact ground-truth expectation value Q★ via noiseless sim.
  3. For each noise amplification factor λ ∈ {1, 3, 5, 7, 9} (odd integers only,
     per Uzdin KIK rule — non-odd factors use digital folding which introduces
     coherent errors that scale non-linearly on physical hardware):
     a) Scale baseline depolarising noise by λ.
     b) Run Analysis A (shot-level QgateSampler).
     c) Run Analysis B (window-level temporal filtering).
     d) Compute acceptance rate, MSE, and cost efficiency η(λ).
  4. Plot two figures (one per analysis).

The key result:  η(λ) < 1 ⟹ filtering + VNS is advantageous.

Usage:
    python simulations/vns_compatibility/run_vns_experiment.py

Requires:
    pip install -e packages/qgate[qiskit]
    pip install qiskit-aer qiskit-ibm-runtime matplotlib

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Qiskit imports ────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import SamplerV2

# ── qgate import ──────────────────────────────────────────────────────────
from qgate import QgateSampler, SamplerConfig

# Ensure project root is importable for qgate.adapters helpers
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qgate.adapters.vqe_adapter import (
    tfim_exact_ground_energy,
    compute_energy_from_bitstring,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — easy to adjust
# ═══════════════════════════════════════════════════════════════════════════

N_QUBITS = 4                      # system qubits (4-qubit TFIM chain)
N_LAYERS = 2                      # ansatz depth (hardware-efficient layers)
J_COUPLING = 1.0                  # TFIM ZZ coupling strength
H_FIELD = 1.0                     # TFIM transverse field strength
SHOTS = 20_000                    # shots per configuration (Analysis A)
SEED = 42                         # reproducibility

# Noise amplification factors (Virtual Noise Scaling)
# Per Uzdin's KIK / Layered mitigation theory, reliable noise amplification
# on physical hardware MUST use odd integer scale factors.  Non-odd factors
# (e.g. 1.5, 2.0) require "digital folding" which introduces coherent errors
# that scale non-linearly — making the amplified noise channel qualitatively
# different from the original, not merely stronger.
# See: apply_uzdin_unitary_folding() in qgate.transpiler.
LAMBDA_VALUES = [1, 3, 5, 7, 9]

# ── Baseline noise parameters (IBM-like depolarising model) ───────────────
# These are realistic single/two-qubit error rates for current hardware.
# Adjust p1 and p2 to match your target QPU.
P1_BASELINE = 1.5e-3              # 1-qubit gate depolarising probability
P2_BASELINE = 1.2e-2              # 2-qubit gate (CX) depolarising probability
P_MEAS_BASELINE = 2.0e-2          # measurement error probability

# ── Window-level filtering parameters (Analysis B) ───────────────────────
# Simulates temporal noise inhomogeneity by running many independent batches
# ("windows") and selecting the best ones based on probe-bit bias.
N_WINDOWS = 50                    # number of independent execution windows
WINDOW_SIZE = 4_000               # shots per window (~4k as in email)
TOP_FRACTION = 0.30               # keep top 30% of windows (least noisy)
NOISE_DRIFT_SIGMA = 0.3           # stochastic drift σ for temporal fluctuation

# Output
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ═══════════════════════════════════════════════════════════════════════════
# Noise Model Builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_noise_model_raw(
    lam: float,
    p1: float = P1_BASELINE,
    p2: float = P2_BASELINE,
    p_meas: float = P_MEAS_BASELINE,
) -> NoiseModel:
    """Internal: build a noise model with arbitrary (possibly fractional) λ.

    Used for drifted noise simulation where the effective λ includes a
    stochastic TLS/thermal component.  External callers should use
    :func:`build_noise_model` which enforces the Uzdin odd-factor rule.
    """
    model = NoiseModel()

    # Scale and clamp (depolarising_error requires p < 1)
    p1_scaled = min(lam * p1, 0.999)
    p2_scaled = min(lam * p2, 0.999)
    pm_scaled = min(lam * p_meas, 0.999)

    # 1-qubit gate errors
    err_1q = depolarizing_error(p1_scaled, 1)
    model.add_all_qubit_quantum_error(
        err_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )

    # 2-qubit gate errors (CX)
    err_2q = depolarizing_error(p2_scaled, 2)
    model.add_all_qubit_quantum_error(err_2q, ["cx"])

    # Measurement errors
    err_meas = depolarizing_error(pm_scaled, 1)
    model.add_all_qubit_quantum_error(err_meas, ["measure"])

    return model


def build_noise_model(
    lam: int,
    p1: float = P1_BASELINE,
    p2: float = P2_BASELINE,
    p_meas: float = P_MEAS_BASELINE,
) -> NoiseModel:
    """Build a depolarising NoiseModel with error rates scaled by λ.

    Parameters
    ----------
    lam : int
        Noise amplification factor (1 = baseline).  Must be a positive
        odd integer per Uzdin's KIK rule — non-odd factors require
        digital folding which introduces coherent errors.
    p1 : float
        Baseline 1-qubit depolarising probability.
    p2 : float
        Baseline 2-qubit (CX) depolarising probability.
    p_meas : float
        Baseline measurement depolarising probability.

    Returns
    -------
    NoiseModel
        Qiskit noise model with all probabilities multiplied by λ.
        Probabilities are clamped to [0, 1) for physical validity.

    Raises
    ------
    ValueError
        If λ is not a positive odd integer.
    """
    # Enforce Uzdin odd-factor rule
    if not isinstance(lam, int) or lam < 1 or lam % 2 == 0:
        raise ValueError(
            f"Noise amplification factor λ={lam!r} violates the Uzdin odd-factor "
            f"rule.  Only positive odd integers (1, 3, 5, 7, …) are permitted.  "
            f"Non-odd factors require digital folding which introduces coherent "
            f"errors that scale non-linearly on physical hardware."
        )

    return _build_noise_model_raw(lam, p1, p2, p_meas)


def build_drifted_noise_model(
    lam: int,
    rng: np.random.Generator,
    drift_sigma: float = NOISE_DRIFT_SIGMA,
) -> NoiseModel:
    """Build a noise model with stochastic drift, simulating temporal fluctuation.

    On real QPUs, noise rates are NOT constant — they drift due to TLS
    fluctuations, thermal instability, and crosstalk from other jobs.
    This function draws a per-window multiplier from a log-normal
    distribution centred at 1.0, so some windows experience lower and
    some higher effective noise.

    Note: The drift multiplier produces a fractional effective λ — this is
    physically correct because it models *environment noise*, not deliberate
    ZNE amplification.  The base λ must still be a valid odd integer per
    Uzdin's rule; only the stochastic drift on top of it is continuous.

    Parameters
    ----------
    lam : int
        Global VNS noise amplification factor (odd integer per Uzdin rule).
    rng : Generator
        NumPy random generator for reproducibility.
    drift_sigma : float
        Standard deviation of the log-normal drift.  Higher σ = more
        temporal inhomogeneity (more benefit from window filtering).

    Returns
    -------
    NoiseModel
        Noise model with stochastically drifted error rates.
    """
    # Validate base λ is a valid odd integer (the deliberate amplification)
    if not isinstance(lam, int) or lam < 1 or lam % 2 == 0:
        raise ValueError(
            f"Base amplification factor λ={lam!r} violates the Uzdin odd-factor rule."
        )

    # Draw drift multiplier: log-normal centred at 1.0
    drift = float(rng.lognormal(mean=0.0, sigma=drift_sigma))
    effective_lam = lam * drift

    # Build noise model directly with the drifted (fractional) effective λ.
    # This bypasses the odd-integer check because the drift is environment
    # noise simulation, not deliberate ZNE amplification.
    return _build_noise_model_raw(effective_lam)


# ═══════════════════════════════════════════════════════════════════════════
# TFIM Ansatz Circuit
# ═══════════════════════════════════════════════════════════════════════════

def build_tfim_ansatz(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    seed: int = SEED,
) -> QuantumCircuit:
    """Build a hardware-efficient VQE ansatz for the TFIM.

    Structure per layer:
        Ry(θ) + Rz(φ) on each qubit  →  CNOT entangling ladder

    Parameters are randomly initialised with identity-biased scaling
    (smaller perturbations at deeper layers to avoid barren plateaus).

    The circuit ends with measurement of all system qubits.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name=f"tfim_ansatz_L{n_layers}")

    for layer in range(n_layers):
        # Per-layer scale: early layers break symmetry, deeper layers are gentler
        scale = (math.pi / 4) / math.sqrt(1 + layer)

        for q in range(n_qubits):
            theta_ry = float(rng.uniform(-scale, scale))
            theta_rz = float(rng.uniform(-scale, scale))
            qc.ry(theta_ry, q)
            qc.rz(theta_rz, q)

        # CNOT entangling ladder (nearest-neighbour)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        qc.barrier()

    qc.measure_all()
    return qc


# ═══════════════════════════════════════════════════════════════════════════
# Observable: ZZ Energy Estimator
# ═══════════════════════════════════════════════════════════════════════════

def estimate_zz_energy(counts: dict[str, int], n_qubits: int) -> tuple[float, float]:
    """Estimate ⟨H_ZZ⟩ = −J Σ Z_i Z_{i+1}  from measurement counts.

    Returns
    -------
    (mean_energy, variance)
        The expectation value and its sample variance.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0

    energies = []
    weights = []
    for bitstring, cnt in counts.items():
        e = compute_energy_from_bitstring(bitstring, n_qubits, J_COUPLING, h_field=0.0)
        energies.append(e)
        weights.append(cnt)

    energies = np.array(energies, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    probs = weights / total

    mean_e = float(np.sum(probs * energies))
    var_e = float(np.sum(probs * (energies - mean_e) ** 2))

    return mean_e, var_e


def counts_from_pub_result(pub_result, creg_name: str = "meas") -> dict[str, int]:
    """Extract counts dict from a SamplerV2 PubResult.

    Handles both filtered (qgate) and raw PubResult objects.
    Tries common classical register names.
    """
    data = pub_result.data

    # Try the given name first, then common alternatives
    for name in [creg_name, "meas", "c"]:
        ba = getattr(data, name, None)
        if ba is not None:
            return ba.get_counts()

    # Fallback: try the first BitArray attribute found
    from qiskit.primitives.containers import BitArray
    for attr in dir(data):
        if attr.startswith("_"):
            continue
        obj = getattr(data, attr, None)
        if isinstance(obj, BitArray):
            return obj.get_counts()

    raise RuntimeError(
        f"Could not find any BitArray in PubResult.data. "
        f"Available attrs: {[a for a in dir(data) if not a.startswith('_')]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Analysis A: QgateSampler (shot-level Galton filter)
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis_a(circuit: QuantumCircuit, q_star: float) -> list[dict]:
    """Shot-level filtering via QgateSampler for each λ."""

    results = []

    print("─── Analysis A: QgateSampler (shot-level Galton filter) ─────────")
    print(f"  {'λ':>5}  {'p_acc':>8}  {'Q_raw':>10}  {'Q_filt':>10}  "
          f"{'MSE_raw':>10}  {'MSE_filt':>10}  {'η':>8}")
    print("  " + "─" * 75)

    for lam in LAMBDA_VALUES:
        t0 = time.time()

        # Build scaled noise model
        noise_model = build_noise_model(lam)
        noisy_backend = AerSimulator(noise_model=noise_model)

        # ── RAW run (standard SamplerV2, no filtering) ────────────────
        raw_sampler = SamplerV2(mode=noisy_backend)
        raw_job = raw_sampler.run([(circuit,)], shots=SHOTS)
        raw_result = raw_job.result()
        raw_counts = counts_from_pub_result(raw_result[0])
        q_raw, var_raw = estimate_zz_energy(raw_counts, N_QUBITS)

        # ── FILTERED run (QgateSampler with Galton threshold) ─────────
        qgate_cfg = SamplerConfig(
            probe_angle=math.pi / 6,       # weak coupling (30°)
            target_acceptance=0.35,         # keep top 35%
            window_size=4096,
            min_window_size=100,
            baseline_threshold=0.65,
            min_threshold=0.3,
            max_threshold=0.95,
            use_quantile=True,
            optimization_level=1,
            oversample_factor=1.0,
        )
        qgate_sampler = QgateSampler(backend=noisy_backend, config=qgate_cfg)
        qgate_sampler.reset_threshold()

        filt_job = qgate_sampler.run([(circuit,)], shots=SHOTS)
        filt_result = filt_job.result()

        # Extract metadata
        filt_meta = filt_result[0].metadata.get("qgate_filter", {})
        total_shots = filt_meta.get("total_shots", SHOTS)
        accepted_shots = filt_meta.get("accepted_shots", SHOTS)
        p_acc = filt_meta.get("acceptance_rate", 1.0)

        # Extract filtered counts
        filt_counts = counts_from_pub_result(filt_result[0])
        q_filt, var_filt = estimate_zz_energy(filt_counts, N_QUBITS)

        # MSE = bias² + variance
        mse_raw = (q_raw - q_star) ** 2 + var_raw
        mse_filt = (q_filt - q_star) ** 2 + var_filt

        # Cost efficiency
        eta = (1.0 / p_acc) * (mse_filt / mse_raw) if (p_acc > 0 and mse_raw > 0) else float("inf")

        dt = time.time() - t0
        row = {
            "analysis": "A_shot_level",
            "lambda": lam,
            "p_acc": p_acc,
            "total_shots": total_shots,
            "accepted_shots": accepted_shots,
            "q_star": q_star,
            "q_raw": q_raw,
            "q_filt": q_filt,
            "var_raw": var_raw,
            "var_filt": var_filt,
            "mse_raw": mse_raw,
            "mse_filt": mse_filt,
            "eta": eta,
            "threshold": filt_meta.get("threshold", 0.0),
            "elapsed_s": dt,
        }
        results.append(row)

        print(f"  {lam:5.1f}  {p_acc:8.4f}  {q_raw:10.4f}  {q_filt:10.4f}  "
              f"{mse_raw:10.6f}  {mse_filt:10.6f}  {eta:8.4f}  "
              f"({dt:.1f}s, {accepted_shots}/{total_shots})")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis B: Window-level temporal filtering (email scenario)
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis_b(circuit: QuantumCircuit, q_star: float) -> list[dict]:
    """Window-level filtering with simulated temporal noise drift.

    For each λ, runs N_WINDOWS independent batches, each with a
    stochastically drifted noise model.  Scores each window by its
    probe-bit bias (via QgateSampler metadata) and selects the top-k%.
    """

    results = []
    rng = np.random.default_rng(SEED + 1000)

    print()
    print("─── Analysis B: Window-level temporal filtering ─────────────────")
    print(f"  {N_WINDOWS} windows × {WINDOW_SIZE:,} shots  |  "
          f"top {TOP_FRACTION*100:.0f}% selection  |  "
          f"drift σ = {NOISE_DRIFT_SIGMA}")
    print(f"  {'λ':>5}  {'p_acc':>8}  {'Q_raw':>10}  {'Q_filt':>10}  "
          f"{'MSE_raw':>10}  {'MSE_filt':>10}  {'η':>8}")
    print("  " + "─" * 75)

    for lam in LAMBDA_VALUES:
        t0 = time.time()

        window_energies = []      # raw per-window energy estimates
        window_probe_scores = []  # probe-based quality score per window
        window_counts_list = []   # raw counts per window

        for w in range(N_WINDOWS):
            # Each window gets a stochastically drifted noise model.
            # This simulates real-QPU temporal drift (TLS, thermal, crosstalk).
            drifted_model = build_drifted_noise_model(lam, rng)
            noisy_backend = AerSimulator(noise_model=drifted_model)

            # ── Run via QgateSampler to extract probe statistics ───────
            # The probe ancilla is designed so that P(ancilla=1) is higher
            # when nearest-neighbour spins are aligned (= low-energy / less
            # decoherence).  A window with higher mean probe score had less
            # effective noise.  We use the *filtered* energy from qgate as
            # the window-level score below.
            qgate_cfg = SamplerConfig(
                probe_angle=math.pi / 6,
                target_acceptance=0.50,     # lenient — we do window selection externally
                window_size=WINDOW_SIZE,
                min_window_size=50,
                baseline_threshold=0.50,
                min_threshold=0.2,
                max_threshold=0.95,
                use_quantile=True,
                optimization_level=1,
                oversample_factor=1.0,
            )
            qgate_sampler = QgateSampler(backend=noisy_backend, config=qgate_cfg)
            qgate_sampler.reset_threshold()

            job = qgate_sampler.run([(circuit,)], shots=WINDOW_SIZE)
            result = job.result()

            # The probe score for this window = the energy estimated from
            # the FILTERED shots.  More negative energy ≈ better structural
            # preservation ≈ lower effective decoherence in this window.
            filt_counts = counts_from_pub_result(result[0])
            e_filt_window, _ = estimate_zz_energy(filt_counts, N_QUBITS)
            # Score: more negative energy = better (we negate for argsort)
            window_probe_scores.append(e_filt_window)

            # Also run raw (no probe) for baseline comparison
            raw_sampler = SamplerV2(mode=noisy_backend)
            raw_job = raw_sampler.run([(circuit,)], shots=WINDOW_SIZE)
            raw_result = raw_job.result()
            raw_counts = counts_from_pub_result(raw_result[0])

            e_window, _ = estimate_zz_energy(raw_counts, N_QUBITS)
            window_energies.append(e_window)
            window_counts_list.append(raw_counts)

        # ── Aggregate: all windows (raw) ──────────────────────────────
        all_counts: dict[str, int] = {}
        for wc in window_counts_list:
            for bs, cnt in wc.items():
                all_counts[bs] = all_counts.get(bs, 0) + cnt
        q_raw, var_raw = estimate_zz_energy(all_counts, N_QUBITS)

        # ── Window selection: keep top-k% by probe score ──────────────
        # Lower (more negative) filtered energy ≈ better structural
        # preservation ≈ less decoherence in that window.  Select the
        # windows with the most negative filtered energy.
        scores = np.array(window_probe_scores)
        n_keep = max(1, int(N_WINDOWS * TOP_FRACTION))
        top_indices = np.argsort(scores)[:n_keep]  # most negative energies

        selected_counts: dict[str, int] = {}
        for idx in top_indices:
            for bs, cnt in window_counts_list[idx].items():
                selected_counts[bs] = selected_counts.get(bs, 0) + cnt
        q_filt, var_filt = estimate_zz_energy(selected_counts, N_QUBITS)

        # Effective acceptance rate = fraction of total shots retained
        total_shots_all = N_WINDOWS * WINDOW_SIZE
        selected_shots = sum(selected_counts.values())
        p_acc = selected_shots / total_shots_all

        # MSE = bias² + variance
        mse_raw = (q_raw - q_star) ** 2 + var_raw
        mse_filt = (q_filt - q_star) ** 2 + var_filt

        # Cost efficiency
        eta = (1.0 / p_acc) * (mse_filt / mse_raw) if (p_acc > 0 and mse_raw > 0) else float("inf")

        dt = time.time() - t0

        # Score distribution stats
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))
        score_selected_mean = float(np.mean(scores[top_indices]))

        row = {
            "analysis": "B_window_level",
            "lambda": lam,
            "p_acc": p_acc,
            "n_windows": N_WINDOWS,
            "n_selected": n_keep,
            "window_size": WINDOW_SIZE,
            "total_shots": total_shots_all,
            "selected_shots": selected_shots,
            "q_star": q_star,
            "q_raw": q_raw,
            "q_filt": q_filt,
            "var_raw": var_raw,
            "var_filt": var_filt,
            "mse_raw": mse_raw,
            "mse_filt": mse_filt,
            "eta": eta,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_selected_mean": score_selected_mean,
            "elapsed_s": dt,
        }
        results.append(row)

        print(f"  {lam:5.1f}  {p_acc:8.4f}  {q_raw:10.4f}  {q_filt:10.4f}  "
              f"{mse_raw:10.6f}  {mse_filt:10.6f}  {eta:8.4f}  "
              f"({dt:.1f}s, score μ={score_mean:.3f}±{score_std:.3f})")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _plot_analysis(results: list[dict], title_suffix: str, filename: str) -> Path:
    """Generate 3-panel figure: p_acc, MSE, and η vs λ."""

    lambdas = [r["lambda"] for r in results]
    p_accs = [r["p_acc"] for r in results]
    mse_raws = [r["mse_raw"] for r in results]
    mse_filts = [r["mse_filt"] for r in results]
    etas = [r["eta"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"VNS + Trajectory Filtering — {title_suffix}\n"
        f"{N_QUBITS}-qubit TFIM  |  {N_LAYERS} ansatz layers  |  qgate",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Acceptance rate ──────────────────────────────────────
    ax = axes[0]
    ax.plot(lambdas, p_accs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Noise amplification factor λ", fontsize=11)
    ax.set_ylabel("Acceptance rate  p_acc", fontsize=11)
    ax.set_title("Acceptance Rate vs λ", fontsize=12)
    ax.set_ylim(0, max(p_accs) * 1.2 + 0.05)
    ax.grid(True, alpha=0.3)
    for x, y in zip(lambdas, p_accs):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # ── Panel 2: MSE comparison ───────────────────────────────────────
    ax = axes[1]
    ax.semilogy(lambdas, mse_raws, "s-", color="#F44336", linewidth=2,
                markersize=8, label="MSE raw")
    ax.semilogy(lambdas, mse_filts, "D-", color="#4CAF50", linewidth=2,
                markersize=8, label="MSE filtered (qgate)")
    ax.set_xlabel("Noise amplification factor λ", fontsize=11)
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.set_title("MSE: Raw vs Filtered", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Cost efficiency η ────────────────────────────────────
    ax = axes[2]
    ax.plot(lambdas, etas, "^-", color="#9C27B0", linewidth=2, markersize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5,
               label="η = 1  (break-even)")
    ax.set_xlabel("Noise amplification factor λ", fontsize=11)
    ax.set_ylabel("Cost efficiency  η(λ)", fontsize=11)
    ax.set_title("Cost Efficiency η vs λ", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Shade advantage region
    y_min = min(min(etas) * 0.8, 0)
    y_max = max(max(etas) * 1.2, 1.5)
    ax.set_ylim(y_min, y_max)
    ax.fill_between(lambdas, y_min, 1.0, alpha=0.08, color="green")
    ax.text(
        lambdas[-1], (y_min + 1.0) / 2,
        "η < 1 → filtering\nadvantageous",
        ha="right", va="center", fontsize=9, fontstyle="italic", color="green",
    )

    for x, y in zip(lambdas, etas):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / filename
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")
    return fig_path


def plot_results(results_a: list[dict], results_b: list[dict]) -> list[Path]:
    """Generate figures for both analyses."""
    paths = []
    paths.append(_plot_analysis(
        results_a,
        f"Analysis A: QgateSampler Shot-Level ({SHOTS:,} shots)",
        "vns_analysis_a_shot_level.png",
    ))
    paths.append(_plot_analysis(
        results_b,
        f"Analysis B: Window-Level ({N_WINDOWS}×{WINDOW_SIZE:,}, top {TOP_FRACTION*100:.0f}%)",
        "vns_analysis_b_window_level.png",
    ))
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("VNS + TRAJECTORY FILTERING COMPATIBILITY EXPERIMENT")
    print("Patent ref: US App. Nos. 63/983,831 & 63/989,632 | IL 326915")
    print("=" * 72)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    circuit = build_tfim_ansatz()
    print(f"Circuit: {circuit.name}")
    print(f"  Qubits: {N_QUBITS}  |  Layers: {N_LAYERS}")
    print(f"  Baseline noise: p1={P1_BASELINE}, p2={P2_BASELINE}, pm={P_MEAS_BASELINE}")
    print()

    # ── Step 1: Ideal ground truth Q★ ─────────────────────────────────
    print("Step 1: Computing ideal (noiseless) ground truth Q★ ...")
    ideal_backend = AerSimulator()
    ideal_sampler = SamplerV2(mode=ideal_backend)
    ideal_job = ideal_sampler.run([(circuit,)], shots=SHOTS)
    ideal_result = ideal_job.result()
    ideal_counts = counts_from_pub_result(ideal_result[0])
    q_star, _ = estimate_zz_energy(ideal_counts, N_QUBITS)
    exact_gs = tfim_exact_ground_energy(N_QUBITS, J_COUPLING, H_FIELD)
    print(f"  Q★ (ideal sim ZZ)      = {q_star:.6f}")
    print(f"  Exact ground-state E   = {exact_gs:.6f}")
    print()

    t0 = time.time()

    # ── Step 2: Analysis A — shot-level QgateSampler ──────────────────
    results_a = run_analysis_a(circuit, q_star)

    # ── Step 3: Analysis B — window-level temporal filtering ──────────
    results_b = run_analysis_b(circuit, q_star)

    # ── Save all results ──────────────────────────────────────────────
    all_results = {"A_shot_level": results_a, "B_window_level": results_b}
    json_path = OUTPUT_DIR / "vns_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_qubits": N_QUBITS,
                "n_layers": N_LAYERS,
                "j_coupling": J_COUPLING,
                "h_field": H_FIELD,
                "shots_analysis_a": SHOTS,
                "n_windows": N_WINDOWS,
                "window_size": WINDOW_SIZE,
                "top_fraction": TOP_FRACTION,
                "noise_drift_sigma": NOISE_DRIFT_SIGMA,
                "seed": SEED,
                "lambda_values": LAMBDA_VALUES,
                "p1_baseline": P1_BASELINE,
                "p2_baseline": P2_BASELINE,
                "p_meas_baseline": P_MEAS_BASELINE,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    print()
    plot_results(results_a, results_b)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Ideal Q★ (ZZ energy)   = {q_star:.6f}")
    print(f"  Exact ground-state E   = {exact_gs:.6f}")
    print()

    for label, res_list in [("A (shot-level)", results_a),
                            ("B (window-level)", results_b)]:
        print(f"  Analysis {label}:")
        print(f"    {'λ':>5}  {'η':>8}  {'MSE_raw':>10}  {'MSE_filt':>10}  {'p_acc':>8}  {'Verdict'}")
        print("    " + "─" * 65)
        for r in res_list:
            verdict = "✅ ADVANTAGEOUS" if r["eta"] < 1.0 else "❌ overhead > gain"
            print(f"    {r['lambda']:5.1f}  {r['eta']:8.4f}  "
                  f"{r['mse_raw']:10.6f}  {r['mse_filt']:10.6f}  "
                  f"{r['p_acc']:8.4f}  {verdict}")
        print()

    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
