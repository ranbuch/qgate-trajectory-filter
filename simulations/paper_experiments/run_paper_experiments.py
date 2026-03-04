#!/usr/bin/env python3
"""
run_paper_experiments.py — Three supplementary experiments for the paper
=======================================================================

Three experiments that strengthen the claim that qgate trajectory
filtering reduces estimation error without introducing systematic bias.

**Experiment 1 — Bias vs Ground Truth (noise sweep)**
  Sweep depolarising noise from 0 (ideal) to 5e-2 (very noisy) and
  compare E_true, E_raw, E_filtered across 7 noise levels.

**Experiment 2 — Qubit Scaling**
  Test at 8, 12, 16 qubits to see whether improvement persists as the
  system grows.

**Experiment 3 — Cross-Algorithm Validation**
  Test on VQE, QAOA (MaxCut), and Grover to demonstrate algorithm-
  agnostic effectiveness.

Each experiment produces:
  • Per-trial energy (or probability for Grover) for Raw / Ancilla /
    Ancilla+Galton estimators.
  • Aggregated bias, variance, MSE, 95% CI, Wilcoxon paired test.
  • JSON results saved to results/.

Usage:
    python run_paper_experiments.py --experiment all          # run everything (~5 h)
    python run_paper_experiments.py --experiment noise        # Exp 1 only (~2 h)
    python run_paper_experiments.py --experiment scaling      # Exp 2 only (~2 h)
    python run_paper_experiments.py --experiment cross-algo   # Exp 3 only (~50 min)
    python run_paper_experiments.py --experiment all --dry-run # quick validation (~3 min)

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Ensure qgate is importable ──
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "packages" / "qgate" / "src"))

from qgate import (
    ConditioningVariant,
    GateConfig,
    TrajectoryFilter,
    VQETSVFAdapter,
)
from qgate.adapters.grover_adapter import GroverTSVFAdapter
from qgate.adapters.qaoa_adapter import (
    QAOATSVFAdapter,
    best_maxcut,
    maxcut_value,
    random_regular_graph,
)
from qgate.adapters.vqe_adapter import (
    compute_energy_from_bitstring,
    estimate_energy_from_counts,
    tfim_exact_ground_energy,
)
from qgate.config import DynamicThresholdConfig, FusionConfig

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

J_COUPLING = 1.0
H_FIELD = 3.04
DEFAULT_SHOTS = 100_000
DEFAULT_TRIALS = 15
SEED_BASE = 42
TARGET_ACCEPTANCE = 0.15

# Noise levels for Experiment 1
NOISE_LEVELS = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

# Qubit counts for Experiment 2
QUBIT_COUNTS = [8, 12, 16]

# Fixed depth for all experiments (sweet spot from prior study)
DEFAULT_LAYERS = 3


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrialResult:
    """Result of a single trial for one estimator."""
    value: float          # energy or probability depending on algorithm
    n_shots_used: int
    acceptance_rate: float = 1.0


@dataclass
class EstimatorStats:
    """Aggregated statistics over multiple trials for one estimator."""
    name: str
    n_trials: int
    exact_value: float
    values: list[float] = field(default_factory=list)
    acceptances: list[float] = field(default_factory=list)

    # Computed
    mean_value: float = 0.0
    bias: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    mse: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    mean_acceptance: float = 0.0
    bias_pvalue: float = 0.0

    def compute(self) -> None:
        arr = np.array(self.values)
        n = len(arr)
        self.mean_value = float(np.mean(arr))
        self.bias = self.mean_value - self.exact_value
        self.variance = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        self.std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        self.mse = self.bias**2 + self.variance
        self.ci_lower, self.ci_upper = _bootstrap_ci(arr)
        if n > 1 and self.std > 0:
            from scipy import stats as sp_stats
            t_stat = (self.mean_value - self.exact_value) / (self.std / math.sqrt(n))
            self.bias_pvalue = float(sp_stats.t.sf(abs(t_stat), df=n - 1) * 2)
        else:
            self.bias_pvalue = 1.0
        self.mean_acceptance = float(np.mean(self.acceptances)) if self.acceptances else 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_value": self.mean_value,
            "bias": self.bias,
            "variance": self.variance,
            "std": self.std,
            "mse": self.mse,
            "ci_95_lower": self.ci_lower,
            "ci_95_upper": self.ci_upper,
            "bias_pvalue": self.bias_pvalue,
            "mean_acceptance": self.mean_acceptance,
            "values": self.values,
        }


def _bootstrap_ci(
    data: np.ndarray, alpha: float = 0.05, n_boot: int = 10_000, seed: int = 99,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 2:
        m = float(data[0]) if n == 1 else 0.0
        return m, m
    boot_means = np.array([
        float(np.mean(rng.choice(data, size=n, replace=True)))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, 100 * alpha / 2)), \
           float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


# ═══════════════════════════════════════════════════════════════════════════
# Noise model builder (parametric)
# ═══════════════════════════════════════════════════════════════════════════


def build_backend(depol_1q: float = 1e-3, depol_2q: float = 1e-2):
    """AerSimulator with parametric IBM Heron-class noise.

    Args:
        depol_1q: Single-qubit depolarising error rate.
        depol_2q: Two-qubit depolarising error rate.

    If both are 0, returns a noiseless AerSimulator.
    """
    from qiskit_aer import AerSimulator

    if depol_1q == 0.0 and depol_2q == 0.0:
        return AerSimulator()

    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )

    model = NoiseModel()
    t1, t2 = 300e3, 150e3
    g1q, g2q, g_meas = 60, 660, 1200

    err_1q = thermal_relaxation_error(t1, t2, g1q)
    model.add_all_qubit_quantum_error(
        err_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )
    if depol_1q > 0:
        model.add_all_qubit_quantum_error(
            depolarizing_error(depol_1q, 1),
            ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
        )

    err_2q = thermal_relaxation_error(t1, t2, g2q).expand(
        thermal_relaxation_error(t1, t2, g2q),
    )
    model.add_all_qubit_quantum_error(err_2q, ["cx"])
    if depol_2q > 0:
        model.add_all_qubit_quantum_error(
            depolarizing_error(depol_2q, 2), ["cx"],
        )

    err_meas = thermal_relaxation_error(t1, t2, g_meas)
    model.add_all_qubit_quantum_error(err_meas, ["measure"])

    return AerSimulator(noise_model=model)


# ═══════════════════════════════════════════════════════════════════════════
# Generic estimator runners (VQE / QAOA)
# ═══════════════════════════════════════════════════════════════════════════


def _run_raw(counts: dict[str, int], n_qubits: int, metric_fn) -> TrialResult:
    """Estimator A: Raw — all shots, no filtering."""
    total = sum(counts.values())
    val = metric_fn(counts, n_qubits)
    return TrialResult(value=val, n_shots_used=total, acceptance_rate=1.0)


def _postselect_ancilla(counts: dict[str, int]) -> tuple[dict[str, int], int, int]:
    """Post-select on ancilla=1, returning (accepted_counts, accepted, total)."""
    accepted: dict[str, int] = {}
    total = n_acc = 0
    for bs, cnt in counts.items():
        bs = bs.strip()
        total += cnt
        if " " in bs:
            parts = bs.split()
            anc, search = parts[0], parts[-1]
        else:
            anc, search = bs[0], bs[1:]
        if anc == "1":
            accepted[search] = accepted.get(search, 0) + cnt
            n_acc += cnt
    return accepted, n_acc, total


def _run_ancilla(counts: dict[str, int], n_qubits: int, metric_fn) -> TrialResult:
    """Estimator B: Ancilla post-selection."""
    accepted, n_acc, total = _postselect_ancilla(counts)
    if n_acc == 0:
        return TrialResult(value=0.0, n_shots_used=0, acceptance_rate=0.0)
    val = metric_fn(accepted, n_qubits)
    return TrialResult(value=val, n_shots_used=n_acc, acceptance_rate=n_acc / total)


def _run_galton(
    counts: dict[str, int],
    n_qubits: int,
    n_layers: int,
    adapter,
    metric_fn_single,
) -> TrialResult:
    """Estimator C: Ancilla + Galton trajectory filter.

    metric_fn_single(bitstring, n_qubits) → float  for individual bitstrings.
    """
    total = sum(counts.values())
    outcomes = adapter.parse_results(
        {"counts": counts}, n_subsystems=n_qubits, n_cycles=n_layers,
    )
    if not outcomes:
        return TrialResult(value=0.0, n_shots_used=0, acceptance_rate=0.0)

    config = GateConfig(
        n_subsystems=n_qubits,
        n_cycles=n_layers,
        shots=len(outcomes),
        variant=ConditioningVariant.SCORE_FUSION,
        fusion=FusionConfig(alpha=0.5, threshold=0.5),
        dynamic_threshold=DynamicThresholdConfig(
            mode="galton",
            window_size=max(len(outcomes), 500),
            min_window_size=min(100, len(outcomes)),
            target_acceptance=TARGET_ACCEPTANCE,
            robust_stats=True,
            use_quantile=True,
        ),
    )
    tf = TrajectoryFilter(config, adapter)
    result = tf.filter(outcomes)

    if result.accepted_shots == 0:
        accepted, n_acc, _ = _postselect_ancilla(counts)
        if n_acc == 0:
            return TrialResult(value=0.0, n_shots_used=0, acceptance_rate=0.0)
        val_sum = sum(
            metric_fn_single(bs, n_qubits) * c for bs, c in accepted.items()
        )
        return TrialResult(value=val_sum / n_acc, n_shots_used=n_acc, acceptance_rate=n_acc / total)

    threshold = result.threshold_used or 0.5
    scores = result.scores or []

    val_sum = 0.0
    galton_count = 0
    idx = 0
    for bs, cnt in counts.items():
        bs = bs.strip()
        if " " in bs:
            search = bs.split()[-1]
        else:
            search = bs[1:]
        for _ in range(cnt):
            if idx < len(scores) and scores[idx] >= threshold:
                val_sum += metric_fn_single(search, n_qubits)
                galton_count += 1
            idx += 1

    if galton_count == 0:
        accepted, n_acc, _ = _postselect_ancilla(counts)
        if n_acc == 0:
            return TrialResult(value=0.0, n_shots_used=0, acceptance_rate=0.0)
        val_s = sum(metric_fn_single(bs, n_qubits) * c for bs, c in accepted.items())
        return TrialResult(value=val_s / n_acc, n_shots_used=n_acc, acceptance_rate=n_acc / total)

    return TrialResult(
        value=val_sum / galton_count,
        n_shots_used=galton_count,
        acceptance_rate=galton_count / total,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Noise Sweep
# ═══════════════════════════════════════════════════════════════════════════


def run_noise_sweep(
    n_qubits: int = 8,
    shots: int = DEFAULT_SHOTS,
    n_trials: int = DEFAULT_TRIALS,
    n_layers: int = DEFAULT_LAYERS,
    noise_levels: list[float] | None = None,
    output_dir: str = "results",
) -> dict:
    """Experiment 1: Bias vs Ground Truth across noise levels."""
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    exact = tfim_exact_ground_energy(n_qubits, J_COUPLING, H_FIELD)
    print("=" * 80)
    print("  EXPERIMENT 1 — Bias vs Ground Truth (Noise Sweep)")
    print("=" * 80)
    print(f"  TFIM 1D, J={J_COUPLING}, h={H_FIELD}, {n_qubits} qubits, {n_layers} layers")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Trials: {n_trials}, Shots: {shots:,}")
    print(f"  Exact E₀ = {exact:.6f}")
    print("=" * 80)

    all_results: dict[str, dict] = {}

    for noise_idx, depol_1q in enumerate(noise_levels):
        depol_2q = depol_1q * 10  # 2Q error ~10× 1Q error (realistic)
        noise_label = f"depol_1q={depol_1q:.0e}" if depol_1q > 0 else "ideal"
        print(f"\n{'─' * 70}")
        print(f"  Noise level {noise_idx + 1}/{len(noise_levels)}: {noise_label}")
        print(f"  (depol_1q={depol_1q}, depol_2q={depol_2q})")
        print(f"{'─' * 70}")

        backend = build_backend(depol_1q, depol_2q)

        stats = {
            k: EstimatorStats(name=n, n_trials=n_trials, exact_value=exact)
            for k, n in [("raw", "A: Raw"), ("ancilla", "B: Ancilla"),
                         ("ancilla_galton", "C: Ancilla+Galton")]
        }

        for trial in range(n_trials):
            seed = SEED_BASE + trial * 1000 + noise_idx * 100
            t0 = time.time()

            # TSVF circuit
            adapter = VQETSVFAdapter(
                backend=backend, algorithm_mode="tsvf",
                n_qubits=n_qubits, j_coupling=J_COUPLING, h_field=H_FIELD,
                seed=seed, weak_angle_base=math.pi / 4,
                weak_angle_ramp=math.pi / 8, optimization_level=0,
            )
            circuit = adapter.build_circuit(n_qubits, n_layers, seed_offset=seed)
            raw_res = adapter.run(circuit, shots=shots)
            counts_tsvf = adapter._extract_counts(raw_res)

            # Standard circuit for raw estimator
            adapter_std = VQETSVFAdapter(
                backend=backend, algorithm_mode="standard",
                n_qubits=n_qubits, j_coupling=J_COUPLING, h_field=H_FIELD,
                seed=seed, optimization_level=0,
            )
            circ_std = adapter_std.build_circuit(n_qubits, n_layers, seed_offset=seed)
            raw_std = adapter_std.run(circ_std, shots=shots)
            counts_std = adapter_std._extract_counts(raw_std)

            def metric_counts(c, nq):
                return estimate_energy_from_counts(c, nq, J_COUPLING)

            def metric_single(bs, nq):
                return compute_energy_from_bitstring(bs, nq, J_COUPLING)

            res_raw = _run_raw(counts_std, n_qubits, metric_counts)
            res_anc = _run_ancilla(counts_tsvf, n_qubits, metric_counts)
            res_gal = _run_galton(counts_tsvf, n_qubits, n_layers, adapter, metric_single)

            for key, res in [("raw", res_raw), ("ancilla", res_anc), ("ancilla_galton", res_gal)]:
                stats[key].values.append(res.value)
                stats[key].acceptances.append(res.acceptance_rate)

            dt = time.time() - t0
            sys.stdout.write(
                f"\r    Trial {trial + 1:3d}/{n_trials}  "
                f"Raw={res_raw.value:+.4f}  Anc={res_anc.value:+.4f} ({res_anc.acceptance_rate:.1%})  "
                f"Gal={res_gal.value:+.4f} ({res_gal.acceptance_rate:.1%})  [{dt:.1f}s]"
            )
            sys.stdout.flush()

        print()
        for s in stats.values():
            s.compute()

        all_results[noise_label] = {k: s.to_dict() for k, s in stats.items()}
        _print_stats_table(stats, f"Noise: {noise_label}")

    # Save
    data = {
        "experiment": "noise_sweep",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "n_qubits": n_qubits, "shots": shots, "n_trials": n_trials,
        "n_layers": n_layers,
        "exact_energy": exact,
        "noise_levels": noise_levels,
        "results": all_results,
    }
    path = _save(data, f"noise_sweep_{n_qubits}q_{n_trials}t", output_dir)
    _print_noise_summary(all_results, exact, noise_levels)
    print(f"\n  Results saved to: {path}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Qubit Scaling
# ═══════════════════════════════════════════════════════════════════════════


def run_scaling(
    qubit_counts: list[int] | None = None,
    shots: int = DEFAULT_SHOTS,
    n_trials: int = DEFAULT_TRIALS,
    n_layers: int = DEFAULT_LAYERS,
    output_dir: str = "results",
) -> dict:
    """Experiment 2: Scaling with qubit count."""
    if qubit_counts is None:
        qubit_counts = QUBIT_COUNTS

    print("=" * 80)
    print("  EXPERIMENT 2 — Qubit Scaling")
    print("=" * 80)
    print(f"  TFIM 1D, J={J_COUPLING}, h={H_FIELD}, {n_layers} layers")
    print(f"  Qubit counts: {qubit_counts}")
    print(f"  Trials: {n_trials}, Shots: {shots:,}")
    print(f"  Noise: IBM Heron-class (depol_1q=1e-3, depol_2q=1e-2)")
    print("=" * 80)

    backend = build_backend(1e-3, 1e-2)
    all_results: dict[str, dict] = {}

    for nq in qubit_counts:
        exact = tfim_exact_ground_energy(nq, J_COUPLING, H_FIELD)
        print(f"\n{'─' * 70}")
        print(f"  {nq} QUBITS — exact E₀ = {exact:.6f} ({exact / nq:.6f} per site)")
        print(f"{'─' * 70}")

        stats = {
            k: EstimatorStats(name=n, n_trials=n_trials, exact_value=exact)
            for k, n in [("raw", "A: Raw"), ("ancilla", "B: Ancilla"),
                         ("ancilla_galton", "C: Ancilla+Galton")]
        }

        for trial in range(n_trials):
            seed = SEED_BASE + trial * 1000 + nq * 10
            t0 = time.time()

            adapter = VQETSVFAdapter(
                backend=backend, algorithm_mode="tsvf",
                n_qubits=nq, j_coupling=J_COUPLING, h_field=H_FIELD,
                seed=seed, weak_angle_base=math.pi / 4,
                weak_angle_ramp=math.pi / 8, optimization_level=0,
            )
            circuit = adapter.build_circuit(nq, n_layers, seed_offset=seed)
            raw_res = adapter.run(circuit, shots=shots)
            counts_tsvf = adapter._extract_counts(raw_res)

            adapter_std = VQETSVFAdapter(
                backend=backend, algorithm_mode="standard",
                n_qubits=nq, j_coupling=J_COUPLING, h_field=H_FIELD,
                seed=seed, optimization_level=0,
            )
            circ_std = adapter_std.build_circuit(nq, n_layers, seed_offset=seed)
            raw_std = adapter_std.run(circ_std, shots=shots)
            counts_std = adapter_std._extract_counts(raw_std)

            def metric_counts(c, _nq):
                return estimate_energy_from_counts(c, _nq, J_COUPLING)

            def metric_single(bs, _nq):
                return compute_energy_from_bitstring(bs, _nq, J_COUPLING)

            res_raw = _run_raw(counts_std, nq, metric_counts)
            res_anc = _run_ancilla(counts_tsvf, nq, metric_counts)
            res_gal = _run_galton(counts_tsvf, nq, n_layers, adapter, metric_single)

            for key, res in [("raw", res_raw), ("ancilla", res_anc), ("ancilla_galton", res_gal)]:
                stats[key].values.append(res.value)
                stats[key].acceptances.append(res.acceptance_rate)

            dt = time.time() - t0
            sys.stdout.write(
                f"\r    Trial {trial + 1:3d}/{n_trials}  "
                f"Raw={res_raw.value:+.4f}  Anc={res_anc.value:+.4f} ({res_anc.acceptance_rate:.1%})  "
                f"Gal={res_gal.value:+.4f} ({res_gal.acceptance_rate:.1%})  [{dt:.1f}s]"
            )
            sys.stdout.flush()

        print()
        for s in stats.values():
            s.compute()

        all_results[f"{nq}q"] = {k: s.to_dict() for k, s in stats.items()}
        _print_stats_table(stats, f"{nq} qubits")

    data = {
        "experiment": "qubit_scaling",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "qubit_counts": qubit_counts, "shots": shots, "n_trials": n_trials,
        "n_layers": n_layers,
        "noise": "depol_1q=1e-3, depol_2q=1e-2",
        "results": all_results,
    }
    path = _save(data, f"qubit_scaling_{n_trials}t", output_dir)
    _print_scaling_summary(all_results, qubit_counts)
    print(f"\n  Results saved to: {path}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Cross-Algorithm Validation
# ═══════════════════════════════════════════════════════════════════════════


def _run_vqe_trial(
    n_qubits: int, n_layers: int, shots: int, backend, seed: int,
) -> dict[str, TrialResult]:
    """Single VQE/TFIM trial returning all 3 estimators."""
    adapter = VQETSVFAdapter(
        backend=backend, algorithm_mode="tsvf",
        n_qubits=n_qubits, j_coupling=J_COUPLING, h_field=H_FIELD,
        seed=seed, weak_angle_base=math.pi / 4,
        weak_angle_ramp=math.pi / 8, optimization_level=0,
    )
    circuit = adapter.build_circuit(n_qubits, n_layers, seed_offset=seed)
    raw_res = adapter.run(circuit, shots=shots)
    counts_tsvf = adapter._extract_counts(raw_res)

    adapter_std = VQETSVFAdapter(
        backend=backend, algorithm_mode="standard",
        n_qubits=n_qubits, j_coupling=J_COUPLING, h_field=H_FIELD,
        seed=seed, optimization_level=0,
    )
    circ_std = adapter_std.build_circuit(n_qubits, n_layers, seed_offset=seed)
    raw_std = adapter_std.run(circ_std, shots=shots)
    counts_std = adapter_std._extract_counts(raw_std)

    def mc(c, nq):
        return estimate_energy_from_counts(c, nq, J_COUPLING)

    def ms(bs, nq):
        return compute_energy_from_bitstring(bs, nq, J_COUPLING)

    return {
        "raw": _run_raw(counts_std, n_qubits, mc),
        "ancilla": _run_ancilla(counts_tsvf, n_qubits, mc),
        "ancilla_galton": _run_galton(counts_tsvf, n_qubits, n_layers, adapter, ms),
    }


def _run_qaoa_trial(
    n_nodes: int, n_layers: int, shots: int, backend, seed: int, edges: list,
) -> tuple[dict[str, TrialResult], float]:
    """Single QAOA/MaxCut trial. Returns (results, exact_best_cut)."""
    _, best_cut = best_maxcut(n_nodes, edges)
    exact_ratio = 1.0  # best_cut / best_cut = 1.0

    adapter = QAOATSVFAdapter(
        backend=backend, algorithm_mode="tsvf",
        edges=edges, n_nodes=n_nodes, seed=seed,
        weak_angle_base=math.pi / 4, weak_angle_ramp=math.pi / 8,
        optimization_level=0,
    )
    circuit = adapter.build_circuit(n_nodes, n_layers, seed_offset=seed)
    raw_res = adapter.run(circuit, shots=shots)
    counts_tsvf = adapter._extract_counts(raw_res)

    adapter_std = QAOATSVFAdapter(
        backend=backend, algorithm_mode="standard",
        edges=edges, n_nodes=n_nodes, seed=seed, optimization_level=0,
    )
    circ_std = adapter_std.build_circuit(n_nodes, n_layers, seed_offset=seed)
    raw_std = adapter_std.run(circ_std, shots=shots)
    counts_std = adapter_std._extract_counts(raw_std)

    def mc_qaoa(c, nq):
        """Mean approximation ratio from counts."""
        total = sum(c.values())
        if total == 0:
            return 0.0
        total_cut = sum(maxcut_value(bs, edges) * cnt for bs, cnt in c.items())
        return (total_cut / total) / best_cut if best_cut > 0 else 0.0

    def ms_qaoa(bs, nq):
        """Single-bitstring approximation ratio."""
        return maxcut_value(bs, edges) / best_cut if best_cut > 0 else 0.0

    return {
        "raw": _run_raw(counts_std, n_nodes, mc_qaoa),
        "ancilla": _run_ancilla(counts_tsvf, n_nodes, mc_qaoa),
        "ancilla_galton": _run_galton(counts_tsvf, n_nodes, n_layers, adapter, ms_qaoa),
    }, exact_ratio


def _run_grover_trial(
    n_qubits: int, n_iters: int, shots: int, backend, seed: int, target: str,
) -> dict[str, TrialResult]:
    """Single Grover trial. Metric = P(target)."""
    adapter = GroverTSVFAdapter(
        backend=backend, algorithm_mode="tsvf",
        target_state=target, seed=seed,
        weak_angle_base=math.pi / 6, weak_angle_ramp=math.pi / 12,
        optimization_level=0,
    )
    circuit = adapter.build_circuit(n_qubits, n_iters, seed_offset=seed)
    raw_res = adapter.run(circuit, shots=shots)
    counts_tsvf = adapter._extract_counts(raw_res)

    adapter_std = GroverTSVFAdapter(
        backend=backend, algorithm_mode="standard",
        target_state=target, seed=seed, optimization_level=0,
    )
    circ_std = adapter_std.build_circuit(n_qubits, n_iters, seed_offset=seed)
    raw_std = adapter_std.run(circ_std, shots=shots)
    counts_std = adapter_std._extract_counts(raw_std)

    def mc_grover(c, nq):
        """P(target) from counts."""
        total = sum(c.values())
        if total == 0:
            return 0.0
        hits = sum(cnt for bs, cnt in c.items() if bs[-nq:] == target or bs.split()[-1] == target)
        return hits / total

    def ms_grover(bs, nq):
        """1 if target, 0 otherwise."""
        return 1.0 if bs[-nq:] == target else 0.0

    return {
        "raw": _run_raw(counts_std, n_qubits, mc_grover),
        "ancilla": _run_ancilla(counts_tsvf, n_qubits, mc_grover),
        "ancilla_galton": _run_galton(counts_tsvf, n_qubits, n_iters, adapter, ms_grover),
    }


def run_cross_algorithm(
    n_qubits: int = 8,
    shots: int = DEFAULT_SHOTS,
    n_trials: int = DEFAULT_TRIALS,
    n_layers: int = DEFAULT_LAYERS,
    output_dir: str = "results",
) -> dict:
    """Experiment 3: Cross-algorithm validation (VQE, QAOA, Grover)."""
    print("=" * 80)
    print("  EXPERIMENT 3 — Cross-Algorithm Validation")
    print("=" * 80)
    print(f"  Algorithms: VQE/TFIM, QAOA/MaxCut, Grover")
    print(f"  Qubits: {n_qubits}, Layers/Iters: {n_layers}")
    print(f"  Trials: {n_trials}, Shots: {shots:,}")
    print(f"  Noise: IBM Heron-class (depol_1q=1e-3, depol_2q=1e-2)")
    print("=" * 80)

    backend = build_backend(1e-3, 1e-2)
    all_results: dict[str, dict] = {}

    # ── VQE/TFIM ──
    exact_vqe = tfim_exact_ground_energy(n_qubits, J_COUPLING, H_FIELD)
    print(f"\n{'─' * 70}")
    print(f"  ALGORITHM: VQE / TFIM  (exact E₀ = {exact_vqe:.6f})")
    print(f"{'─' * 70}")

    stats_vqe = {
        k: EstimatorStats(name=n, n_trials=n_trials, exact_value=exact_vqe)
        for k, n in [("raw", "A: Raw"), ("ancilla", "B: Ancilla"),
                     ("ancilla_galton", "C: Ancilla+Galton")]
    }
    for trial in range(n_trials):
        seed = SEED_BASE + trial * 1000
        t0 = time.time()
        res = _run_vqe_trial(n_qubits, n_layers, shots, backend, seed)
        for key in stats_vqe:
            stats_vqe[key].values.append(res[key].value)
            stats_vqe[key].acceptances.append(res[key].acceptance_rate)
        dt = time.time() - t0
        sys.stdout.write(
            f"\r    Trial {trial + 1:3d}/{n_trials}  "
            f"Raw={res['raw'].value:+.4f}  Gal={res['ancilla_galton'].value:+.4f} [{dt:.1f}s]"
        )
        sys.stdout.flush()
    print()
    for s in stats_vqe.values():
        s.compute()
    all_results["vqe"] = {k: s.to_dict() for k, s in stats_vqe.items()}
    _print_stats_table(stats_vqe, "VQE / TFIM")

    # ── QAOA/MaxCut ──
    # For QAOA we use approximation ratio as metric (exact = 1.0)
    # Grover's target_state must match n_qubits; generate a graph with n_nodes=n_qubits
    # but Grover is fixed at 3 search qubits in the adapter (target="101").
    # We'll use n_qubits for QAOA and a smaller search for Grover.
    edges = random_regular_graph(n_qubits, degree=3, seed=SEED_BASE)
    _, best_cut = best_maxcut(n_qubits, edges)
    exact_qaoa = 1.0  # approximation ratio = 1.0 is perfect
    print(f"\n{'─' * 70}")
    print(f"  ALGORITHM: QAOA / MaxCut  ({n_qubits} nodes, {len(edges)} edges, best cut={best_cut})")
    print(f"  Metric: approximation ratio (exact = 1.0)")
    print(f"{'─' * 70}")

    stats_qaoa = {
        k: EstimatorStats(name=n, n_trials=n_trials, exact_value=exact_qaoa)
        for k, n in [("raw", "A: Raw"), ("ancilla", "B: Ancilla"),
                     ("ancilla_galton", "C: Ancilla+Galton")]
    }
    for trial in range(n_trials):
        seed = SEED_BASE + trial * 1000 + 500
        t0 = time.time()
        res, _ = _run_qaoa_trial(n_qubits, n_layers, shots, backend, seed, edges)
        for key in stats_qaoa:
            stats_qaoa[key].values.append(res[key].value)
            stats_qaoa[key].acceptances.append(res[key].acceptance_rate)
        dt = time.time() - t0
        sys.stdout.write(
            f"\r    Trial {trial + 1:3d}/{n_trials}  "
            f"Raw={res['raw'].value:.4f}  Gal={res['ancilla_galton'].value:.4f} [{dt:.1f}s]"
        )
        sys.stdout.flush()
    print()
    for s in stats_qaoa.values():
        s.compute()
    all_results["qaoa"] = {k: s.to_dict() for k, s in stats_qaoa.items()}
    _print_stats_table(stats_qaoa, "QAOA / MaxCut (approx ratio)")

    # ── Grover ──
    # Grover adapter is hardcoded for 3 search qubits (target="101")
    grover_qubits = 3
    grover_target = "101"
    grover_iters = n_layers  # use same number of iterations as layers
    exact_grover = 1.0  # P(target) = 1.0 ideally
    print(f"\n{'─' * 70}")
    print(f"  ALGORITHM: Grover  ({grover_qubits} search qubits, target=|{grover_target}⟩, "
          f"{grover_iters} iterations)")
    print(f"  Metric: P(target)  (exact = 1.0 for perfect Grover)")
    print(f"{'─' * 70}")

    stats_grover = {
        k: EstimatorStats(name=n, n_trials=n_trials, exact_value=exact_grover)
        for k, n in [("raw", "A: Raw"), ("ancilla", "B: Ancilla"),
                     ("ancilla_galton", "C: Ancilla+Galton")]
    }
    for trial in range(n_trials):
        seed = SEED_BASE + trial * 1000 + 700
        t0 = time.time()
        res = _run_grover_trial(grover_qubits, grover_iters, shots, backend, seed, grover_target)
        for key in stats_grover:
            stats_grover[key].values.append(res[key].value)
            stats_grover[key].acceptances.append(res[key].acceptance_rate)
        dt = time.time() - t0
        sys.stdout.write(
            f"\r    Trial {trial + 1:3d}/{n_trials}  "
            f"Raw={res['raw'].value:.4f}  Gal={res['ancilla_galton'].value:.4f} [{dt:.1f}s]"
        )
        sys.stdout.flush()
    print()
    for s in stats_grover.values():
        s.compute()
    all_results["grover"] = {k: s.to_dict() for k, s in stats_grover.items()}
    _print_stats_table(stats_grover, "Grover (P(target))")

    # Save
    data = {
        "experiment": "cross_algorithm",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "n_qubits_vqe_qaoa": n_qubits, "n_qubits_grover": grover_qubits,
        "shots": shots, "n_trials": n_trials, "n_layers": n_layers,
        "noise": "depol_1q=1e-3, depol_2q=1e-2",
        "qaoa_graph": {"n_nodes": n_qubits, "edges": edges, "best_cut": best_cut},
        "grover_target": grover_target,
        "results": all_results,
    }
    path = _save(data, f"cross_algo_{n_qubits}q_{n_trials}t", output_dir)
    _print_cross_algo_summary(all_results)
    print(f"\n  Results saved to: {path}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Printing helpers
# ═══════════════════════════════════════════════════════════════════════════


def _print_stats_table(stats: dict[str, EstimatorStats], title: str) -> None:
    """Print a summary table for one configuration."""
    print(f"\n    {'Estimator':<22} {'Mean':>10} {'Bias':>10} "
          f"{'Std':>10} {'MSE':>10} {'95% CI':>22} {'p':>10} {'Acc%':>8}")
    print(f"    {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} "
          f"{'─' * 22} {'─' * 10} {'─' * 8}")
    for key in ["raw", "ancilla", "ancilla_galton"]:
        s = stats[key]
        ci = f"[{s.ci_lower:+.4f}, {s.ci_upper:+.4f}]"
        sig = "***" if s.bias_pvalue < 0.001 else "** " if s.bias_pvalue < 0.01 else "*  " if s.bias_pvalue < 0.05 else "   "
        print(f"    {s.name:<22} {s.mean_value:>+10.4f} {s.bias:>+10.4f} "
              f"{s.std:>10.4f} {s.mse:>10.4f} {ci:>22} "
              f"{s.bias_pvalue:>9.4f}{sig} {s.mean_acceptance:>7.1%}")


def _print_noise_summary(all_results: dict, exact: float, noise_levels: list) -> None:
    """Print Experiment 1 grand summary."""
    from scipy import stats as sp_stats

    print("\n" + "=" * 90)
    print("  EXPERIMENT 1 SUMMARY — Bias vs Ground Truth")
    print("=" * 90)
    print(f"  Exact E₀ = {exact:.6f}\n")
    print(f"  {'Noise':>14} │ {'Raw E':>10} │ {'Anc E':>10} │ {'Gal E':>10} │ "
          f"{'ΔE(Gal-Raw)':>12} │ {'MSE↓%':>8} │ {'Var↓×':>8} │ {'Wilcoxon p':>10}")
    print(f"  {'─' * 14}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}─┼─"
          f"{'─' * 12}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 10}")

    for noise_label, res in all_results.items():
        raw_mean = res["raw"]["mean_value"]
        anc_mean = res["ancilla"]["mean_value"]
        gal_mean = res["ancilla_galton"]["mean_value"]
        raw_mse = res["raw"]["mse"]
        gal_mse = res["ancilla_galton"]["mse"]
        raw_var = res["raw"]["variance"]
        gal_var = res["ancilla_galton"]["variance"]

        delta_e = gal_mean - raw_mean
        mse_pct = (1 - gal_mse / raw_mse) * 100 if raw_mse > 0 else 0
        var_ratio = raw_var / gal_var if gal_var > 0 else float("inf")

        raw_vals = np.array(res["raw"]["values"])
        gal_vals = np.array(res["ancilla_galton"]["values"])
        diff = gal_vals - raw_vals
        try:
            _, p = sp_stats.wilcoxon(diff, alternative="two-sided")
        except Exception:
            p = 1.0

        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "n.s."
        print(f"  {noise_label:>14} │ {raw_mean:>+10.4f} │ {anc_mean:>+10.4f} │ "
              f"{gal_mean:>+10.4f} │ {delta_e:>+12.4f} │ {mse_pct:>+7.1f}% │ "
              f"{var_ratio:>7.0f}× │ {p:>9.4f} {sig}")

    print()


def _print_scaling_summary(all_results: dict, qubit_counts: list) -> None:
    """Print Experiment 2 grand summary."""
    from scipy import stats as sp_stats

    print("\n" + "=" * 90)
    print("  EXPERIMENT 2 SUMMARY — Qubit Scaling")
    print("=" * 90)
    print(f"\n  {'Qubits':>8} │ {'Raw MSE':>10} │ {'Gal MSE':>10} │ "
          f"{'MSE↓%':>8} │ {'Var↓×':>8} │ {'ΔE(Gal-Raw)':>12} │ {'Wilcoxon p':>10}")
    print(f"  {'─' * 8}─┼─{'─' * 10}─┼─{'─' * 10}─┼─"
          f"{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 12}─┼─{'─' * 10}")

    for nq in qubit_counts:
        key = f"{nq}q"
        if key not in all_results:
            continue
        res = all_results[key]
        raw_mse = res["raw"]["mse"]
        gal_mse = res["ancilla_galton"]["mse"]
        raw_var = res["raw"]["variance"]
        gal_var = res["ancilla_galton"]["variance"]
        raw_mean = res["raw"]["mean_value"]
        gal_mean = res["ancilla_galton"]["mean_value"]

        mse_pct = (1 - gal_mse / raw_mse) * 100 if raw_mse > 0 else 0
        var_ratio = raw_var / gal_var if gal_var > 0 else float("inf")
        delta_e = gal_mean - raw_mean

        raw_vals = np.array(res["raw"]["values"])
        gal_vals = np.array(res["ancilla_galton"]["values"])
        diff = gal_vals - raw_vals
        try:
            _, p = sp_stats.wilcoxon(diff, alternative="two-sided")
        except Exception:
            p = 1.0
        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "n.s."

        print(f"  {nq:>8} │ {raw_mse:>10.2f} │ {gal_mse:>10.2f} │ "
              f"{mse_pct:>+7.1f}% │ {var_ratio:>7.0f}× │ {delta_e:>+12.4f} │ "
              f"{p:>9.4f} {sig}")
    print()


def _print_cross_algo_summary(all_results: dict) -> None:
    """Print Experiment 3 grand summary."""
    from scipy import stats as sp_stats

    print("\n" + "=" * 90)
    print("  EXPERIMENT 3 SUMMARY — Cross-Algorithm Validation")
    print("=" * 90)
    print(f"\n  {'Algorithm':>14} │ {'Metric':>14} │ {'Raw':>10} │ {'Galton':>10} │ "
          f"{'MSE↓%':>8} │ {'Var↓×':>8} │ {'Wilcoxon p':>10}")
    print(f"  {'─' * 14}─┼─{'─' * 14}─┼─{'─' * 10}─┼─{'─' * 10}─┼─"
          f"{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 10}")

    algo_labels = {
        "vqe": ("VQE/TFIM", "Energy"),
        "qaoa": ("QAOA/MaxCut", "Approx ratio"),
        "grover": ("Grover", "P(target)"),
    }

    for algo_key, (label, metric_name) in algo_labels.items():
        if algo_key not in all_results:
            continue
        res = all_results[algo_key]
        raw_mean = res["raw"]["mean_value"]
        gal_mean = res["ancilla_galton"]["mean_value"]
        raw_mse = res["raw"]["mse"]
        gal_mse = res["ancilla_galton"]["mse"]
        raw_var = res["raw"]["variance"]
        gal_var = res["ancilla_galton"]["variance"]

        mse_pct = (1 - gal_mse / raw_mse) * 100 if raw_mse > 0 else 0
        var_ratio = raw_var / gal_var if gal_var > 0 else float("inf")

        raw_vals = np.array(res["raw"]["values"])
        gal_vals = np.array(res["ancilla_galton"]["values"])
        diff = gal_vals - raw_vals
        try:
            _, p = sp_stats.wilcoxon(diff, alternative="two-sided")
        except Exception:
            p = 1.0
        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "n.s."

        print(f"  {label:>14} │ {metric_name:>14} │ {raw_mean:>+10.4f} │ "
              f"{gal_mean:>+10.4f} │ {mse_pct:>+7.1f}% │ {var_ratio:>7.0f}× │ "
              f"{p:>9.4f} {sig}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════


def _save(data: dict, prefix: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(output_dir, f"{prefix}_{ts}.json")
    with open(fp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return fp


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Paper experiments: noise sweep, qubit scaling, cross-algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "noise", "scaling", "cross-algo"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick smoke test: 3 trials, 1000 shots, reduced params",
    )

    args = parser.parse_args()

    shots = args.shots
    trials = args.trials
    layers = args.layers

    if args.dry_run:
        shots = 1_000
        trials = 2

    exp = args.experiment
    t_start = time.time()

    if exp in ("all", "noise"):
        noise_levels = NOISE_LEVELS if not args.dry_run else [0.0, 1e-3, 1e-2]
        run_noise_sweep(
            n_qubits=8, shots=shots, n_trials=trials, n_layers=layers,
            noise_levels=noise_levels, output_dir=args.output,
        )

    if exp in ("all", "scaling"):
        qubit_counts = QUBIT_COUNTS if not args.dry_run else [8, 12]
        run_scaling(
            qubit_counts=qubit_counts, shots=shots, n_trials=trials,
            n_layers=layers, output_dir=args.output,
        )

    if exp in ("all", "cross-algo"):
        run_cross_algorithm(
            n_qubits=8, shots=shots, n_trials=trials, n_layers=layers,
            output_dir=args.output,
        )

    total = time.time() - t_start
    print(f"\n{'═' * 80}")
    print(f"  ALL DONE — Total wall-clock: {total / 60:.1f} min ({total / 3600:.1f} h)")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
