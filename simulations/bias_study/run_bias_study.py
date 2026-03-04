#!/usr/bin/env python3
"""
run_bias_study.py — Systematic Bias Evaluation of Trajectory Filtering
======================================================================

**Research question:** Does qgate trajectory filtering reduce estimation
error without introducing systematic bias?

**Design:**
  1. TFIM Hamiltonian at the critical point (h/J ≈ 3.04) — classically
     solvable for ground truth.
  2. Circuits at increasing depth (1, 2, 3, 4, 5 ansatz layers).
  3. Realistic IBM Heron-class noise (depolarising + thermal relaxation
     + measurement error).
  4. 100 k simulated shots per configuration.
  5. Repeated N_TRIALS times for bootstrap statistics.

**Three estimators compared:**
  A. Raw — no filtering, all shots.
  B. Ancilla — post-select on ancilla=|1⟩ (hardware energy probe).
  C. Ancilla + Galton — ancilla post-selection, then qgate Galton
     adaptive thresholding on score-fused parity outcomes.

**Metrics per estimator:**
  • Bias         = E[Ê] − E_exact
  • Variance     = Var(Ê)
  • MSE          = Bias² + Variance
  • 95% CI       via bootstrap

**Null hypothesis:** H₀: bias_filter = 0 (filtering does not introduce
systematic bias).  Tested via two-sided t-test on the trial-level
estimates.

Usage:
    python run_bias_study.py                      # default: 8 qubits, 30 trials
    python run_bias_study.py --n-qubits 6         # faster
    python run_bias_study.py --n-qubits 10        # more interesting
    python run_bias_study.py --trials 100         # tighter CIs
    python run_bias_study.py --shots 200000       # more shots per trial
    python run_bias_study.py --layers 1 2 3       # specific depths

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
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
H_FIELD = 3.04  # Critical point
DEFAULT_N_QUBITS = 8
DEFAULT_SHOTS = 100_000
DEFAULT_TRIALS = 30
DEFAULT_LAYERS = [1, 2, 3, 4, 5]
SEED_BASE = 42
TARGET_ACCEPTANCE = 0.15  # Galton target


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EstimatorResult:
    """Result of a single trial for one estimator."""
    energy: float
    n_shots_used: int
    acceptance_rate: float = 1.0


@dataclass
class EstimatorStats:
    """Aggregated statistics over multiple trials for one estimator."""
    name: str
    n_layers: int
    n_trials: int
    exact_energy: float
    energies: list[float] = field(default_factory=list)

    # Computed
    mean_energy: float = 0.0
    bias: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    mse: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    mean_acceptance: float = 0.0
    bias_pvalue: float = 0.0  # p-value for H₀: bias = 0

    def compute(self, acceptances: list[float]) -> None:
        arr = np.array(self.energies)
        n = len(arr)
        self.mean_energy = float(np.mean(arr))
        self.bias = self.mean_energy - self.exact_energy
        self.variance = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        self.std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        self.mse = self.bias**2 + self.variance

        # 95% CI via bootstrap (BCa)
        self.ci_lower, self.ci_upper = _bootstrap_ci(arr, alpha=0.05)

        # t-test for H₀: mean = exact (i.e., bias = 0)
        if n > 1 and self.std > 0:
            from scipy import stats as sp_stats
            t_stat = (self.mean_energy - self.exact_energy) / (self.std / math.sqrt(n))
            self.bias_pvalue = float(sp_stats.t.sf(abs(t_stat), df=n - 1) * 2)
        else:
            self.bias_pvalue = 1.0

        self.mean_acceptance = float(np.mean(acceptances)) if acceptances else 1.0


def _bootstrap_ci(
    data: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 99,
) -> tuple[float, float]:
    """Percentile bootstrap 95% confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 2:
        m = float(data[0]) if n == 1 else 0.0
        return m, m
    boot_means = np.array([
        float(np.mean(rng.choice(data, size=n, replace=True)))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# Noise model
# ═══════════════════════════════════════════════════════════════════════════


def build_noisy_backend():
    """AerSimulator with IBM Heron-class realistic noise.

    Models:
    - Thermal relaxation (T1/T2) on all gate types
    - Depolarising gate errors (1Q: 0.1%, 2Q: 1%)
    - Measurement thermal relaxation
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )

    model = NoiseModel()

    # IBM Heron-class parameters
    t1 = 300e3       # 300 µs
    t2 = 150e3       # 150 µs
    gate_1q = 60     # 60 ns
    gate_2q = 660    # 660 ns (CX)
    gate_meas = 1200 # 1.2 µs

    # 1Q thermal relaxation + depolarising
    err_1q = thermal_relaxation_error(t1, t2, gate_1q)
    model.add_all_qubit_quantum_error(
        err_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )
    dep_1q = depolarizing_error(1e-3, 1)
    model.add_all_qubit_quantum_error(
        dep_1q, ["rx", "ry", "rz", "h", "x", "z", "s", "sdg", "u"],
    )

    # 2Q thermal relaxation + depolarising
    err_2q = thermal_relaxation_error(t1, t2, gate_2q).expand(
        thermal_relaxation_error(t1, t2, gate_2q),
    )
    model.add_all_qubit_quantum_error(err_2q, ["cx"])
    dep_2q = depolarizing_error(1e-2, 2)
    model.add_all_qubit_quantum_error(dep_2q, ["cx"])

    # Measurement relaxation
    err_meas = thermal_relaxation_error(t1, t2, gate_meas)
    model.add_all_qubit_quantum_error(err_meas, ["measure"])

    return AerSimulator(noise_model=model)


def build_ideal_backend():
    """Noiseless AerSimulator for sanity checks."""
    from qiskit_aer import AerSimulator
    return AerSimulator()


# ═══════════════════════════════════════════════════════════════════════════
# Estimators
# ═══════════════════════════════════════════════════════════════════════════


def run_raw_estimator(
    counts: dict[str, int],
    n_qubits: int,
) -> EstimatorResult:
    """Estimator A: Raw — use all shots, no filtering."""
    total = sum(counts.values())
    energy = estimate_energy_from_counts(counts, n_qubits, J_COUPLING)
    return EstimatorResult(energy=energy, n_shots_used=total, acceptance_rate=1.0)


def run_ancilla_estimator(
    counts: dict[str, int],
    n_qubits: int,
) -> EstimatorResult:
    """Estimator B: Ancilla post-selection — keep only ancilla=1 shots."""
    accepted_counts: dict[str, int] = {}
    total = 0
    accepted = 0

    for bitstring, count in counts.items():
        bs = bitstring.strip()
        total += count

        # Parse ancilla bit (first register in TSVF circuit)
        if " " in bs:
            parts = bs.split()
            anc_bit = parts[0]
            search_bits = parts[-1]
        else:
            anc_bit = bs[0]
            search_bits = bs[1:]

        if anc_bit == "1":
            accepted_counts[search_bits] = accepted_counts.get(search_bits, 0) + count
            accepted += count

    if accepted == 0:
        return EstimatorResult(energy=0.0, n_shots_used=0, acceptance_rate=0.0)

    energy = estimate_energy_from_counts(accepted_counts, n_qubits, J_COUPLING)
    acc_rate = accepted / total if total > 0 else 0.0
    return EstimatorResult(energy=energy, n_shots_used=accepted, acceptance_rate=acc_rate)


def run_ancilla_galton_estimator(
    counts: dict[str, int],
    n_qubits: int,
    n_layers: int,
    adapter: VQETSVFAdapter,
) -> EstimatorResult:
    """Estimator C: Ancilla post-selection + Galton trajectory filtering.

    1. Post-select on ancilla=1 (same as Estimator B).
    2. Feed the accepted shots through qgate's score-fusion + Galton
       adaptive thresholding pipeline.
    3. Compute energy from the double-filtered subset.
    """
    # Step 1: ancilla post-selection (get accepted bitstrings)
    ancilla_accepted_counts: dict[str, int] = {}
    total = sum(counts.values())

    for bitstring, count in counts.items():
        bs = bitstring.strip()
        if " " in bs:
            parts = bs.split()
            anc_bit = parts[0]
            search_bits = parts[-1]
        else:
            anc_bit = bs[0]
            search_bits = bs[1:]

        if anc_bit == "1":
            ancilla_accepted_counts[search_bits] = (
                ancilla_accepted_counts.get(search_bits, 0) + count
            )

    n_ancilla_accepted = sum(ancilla_accepted_counts.values())
    if n_ancilla_accepted == 0:
        return EstimatorResult(energy=0.0, n_shots_used=0, acceptance_rate=0.0)

    # Step 2: Parse accepted shots into ParityOutcome via the adapter
    outcomes = adapter.parse_results(
        {"counts": counts},  # Full counts — adapter knows to postselect
        n_subsystems=n_qubits,
        n_cycles=n_layers,
    )

    # Step 3: Run through qgate Galton filter
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
        return EstimatorResult(energy=0.0, n_shots_used=0, acceptance_rate=0.0)

    # Step 4: Identify which outcomes were accepted (above Galton threshold)
    # and compute energy from their bitstrings
    threshold = result.threshold_used or 0.5
    scores = result.scores or []

    # Rebuild the bitstring→energy mapping for accepted outcomes
    galton_energy_sum = 0.0
    galton_count = 0

    outcome_idx = 0
    for bitstring, count in counts.items():
        bs = bitstring.strip()
        if " " in bs:
            parts = bs.split()
            anc_bit = parts[0]
            search_bits = parts[-1]
        else:
            anc_bit = bs[0]
            search_bits = bs[1:]

        for _ in range(count):
            if outcome_idx < len(scores):
                if scores[outcome_idx] >= threshold:
                    e = compute_energy_from_bitstring(search_bits, n_qubits, J_COUPLING)
                    galton_energy_sum += e
                    galton_count += 1
            outcome_idx += 1

    if galton_count == 0:
        # Fall back to ancilla-only estimate if Galton filters everything
        energy = estimate_energy_from_counts(ancilla_accepted_counts, n_qubits, J_COUPLING)
        return EstimatorResult(
            energy=energy,
            n_shots_used=n_ancilla_accepted,
            acceptance_rate=n_ancilla_accepted / total,
        )

    energy = galton_energy_sum / galton_count
    overall_acc = galton_count / total if total > 0 else 0.0
    return EstimatorResult(energy=energy, n_shots_used=galton_count, acceptance_rate=overall_acc)


# ═══════════════════════════════════════════════════════════════════════════
# Single trial
# ═══════════════════════════════════════════════════════════════════════════


def run_single_trial(
    n_qubits: int,
    n_layers: int,
    shots: int,
    backend,
    trial_seed: int,
) -> dict[str, EstimatorResult]:
    """Run one trial: build circuit, execute, compute all three estimators."""

    # Build TSVF adapter & circuit
    adapter = VQETSVFAdapter(
        backend=backend,
        algorithm_mode="tsvf",
        n_qubits=n_qubits,
        j_coupling=J_COUPLING,
        h_field=H_FIELD,
        seed=trial_seed,
        weak_angle_base=math.pi / 4,
        weak_angle_ramp=math.pi / 8,
        optimization_level=0,  # speed — Aer doesn't need heavy transpilation
    )

    circuit = adapter.build_circuit(
        n_subsystems=n_qubits,
        n_cycles=n_layers,
        seed_offset=trial_seed,
    )

    # Execute on noisy backend
    raw_results = adapter.run(circuit, shots=shots)
    counts = adapter._extract_counts(raw_results)

    # Also need the standard circuit's raw estimator
    adapter_std = VQETSVFAdapter(
        backend=backend,
        algorithm_mode="standard",
        n_qubits=n_qubits,
        j_coupling=J_COUPLING,
        h_field=H_FIELD,
        seed=trial_seed,
        optimization_level=0,
    )
    circuit_std = adapter_std.build_circuit(
        n_subsystems=n_qubits,
        n_cycles=n_layers,
        seed_offset=trial_seed,
    )
    raw_std = adapter_std.run(circuit_std, shots=shots)
    counts_std = adapter_std._extract_counts(raw_std)

    # Compute all three estimators
    return {
        "raw": run_raw_estimator(counts_std, n_qubits),
        "ancilla": run_ancilla_estimator(counts, n_qubits),
        "ancilla_galton": run_ancilla_galton_estimator(counts, n_qubits, n_layers, adapter),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════


def run_bias_study(
    n_qubits: int = DEFAULT_N_QUBITS,
    shots: int = DEFAULT_SHOTS,
    n_trials: int = DEFAULT_TRIALS,
    layers_list: list[int] | None = None,
    output_dir: str = "results",
) -> dict:
    """Run the complete bias study across all depths and trials."""

    if layers_list is None:
        layers_list = DEFAULT_LAYERS

    print("=" * 72)
    print("  BIAS STUDY — Trajectory Filtering for Quantum Circuits")
    print("=" * 72)
    print(f"  Hamiltonian : TFIM 1D, J={J_COUPLING}, h={H_FIELD} (critical point)")
    print(f"  Qubits      : {n_qubits}")
    print(f"  Shots/trial : {shots:,}")
    print(f"  Trials      : {n_trials}")
    print(f"  Layers      : {layers_list}")
    print(f"  Noise       : IBM Heron-class (T1=300µs, depol, readout)")
    print("=" * 72)

    # Ground truth
    exact_energy = tfim_exact_ground_energy(n_qubits, J_COUPLING, H_FIELD)
    print(f"\n  Exact ground-state energy: {exact_energy:.6f}")
    print(f"  Energy per site:           {exact_energy / n_qubits:.6f}\n")

    # Build noisy backend (reused across all trials)
    print("  Building noisy AerSimulator backend...")
    backend = build_noisy_backend()

    all_results: dict[int, dict[str, EstimatorStats]] = {}

    for n_layers in layers_list:
        print(f"\n{'─' * 72}")
        print(f"  DEPTH = {n_layers} layer{'s' if n_layers > 1 else ''}")
        print(f"{'─' * 72}")

        stats = {
            "raw": EstimatorStats(
                name="A: Raw", n_layers=n_layers, n_trials=n_trials,
                exact_energy=exact_energy,
            ),
            "ancilla": EstimatorStats(
                name="B: Ancilla", n_layers=n_layers, n_trials=n_trials,
                exact_energy=exact_energy,
            ),
            "ancilla_galton": EstimatorStats(
                name="C: Ancilla+Galton", n_layers=n_layers, n_trials=n_trials,
                exact_energy=exact_energy,
            ),
        }
        acceptances: dict[str, list[float]] = {k: [] for k in stats}

        for trial in range(n_trials):
            trial_seed = SEED_BASE + trial * 1000 + n_layers * 100
            t0 = time.time()

            results = run_single_trial(n_qubits, n_layers, shots, backend, trial_seed)

            for key in stats:
                stats[key].energies.append(results[key].energy)
                acceptances[key].append(results[key].acceptance_rate)

            dt = time.time() - t0
            # Progress
            raw_e = results["raw"].energy
            anc_e = results["ancilla"].energy
            gal_e = results["ancilla_galton"].energy
            anc_acc = results["ancilla"].acceptance_rate
            gal_acc = results["ancilla_galton"].acceptance_rate
            sys.stdout.write(
                f"\r    Trial {trial + 1:3d}/{n_trials}  "
                f"Raw={raw_e:+.4f}  Anc={anc_e:+.4f} ({anc_acc:.1%})  "
                f"Gal={gal_e:+.4f} ({gal_acc:.1%})  "
                f"[{dt:.1f}s]"
            )
            sys.stdout.flush()

        print()  # newline after progress

        # Compute aggregate statistics
        for key in stats:
            stats[key].compute(acceptances[key])

        all_results[n_layers] = stats

        # Print per-depth summary
        _print_depth_summary(n_layers, stats, exact_energy)

    # Final summary table
    _print_final_summary(all_results, exact_energy, n_qubits)

    # Save results
    output_path = _save_results(all_results, n_qubits, shots, n_trials, output_dir)
    print(f"\n  Results saved to: {output_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════════


def _print_depth_summary(
    n_layers: int,
    stats: dict[str, EstimatorStats],
    exact: float,
) -> None:
    """Print summary table for one depth."""
    print(f"\n    {'Estimator':<22} {'Mean E':>10} {'Bias':>10} "
          f"{'Var':>10} {'MSE':>10} {'95% CI':>22} {'p(bias=0)':>10} {'Acc%':>8}")
    print(f"    {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} "
          f"{'─' * 22} {'─' * 10} {'─' * 8}")

    for key in ["raw", "ancilla", "ancilla_galton"]:
        s = stats[key]
        ci_str = f"[{s.ci_lower:+.4f}, {s.ci_upper:+.4f}]"
        sig = "***" if s.bias_pvalue < 0.001 else "** " if s.bias_pvalue < 0.01 else "*  " if s.bias_pvalue < 0.05 else "   "
        print(
            f"    {s.name:<22} {s.mean_energy:>+10.4f} {s.bias:>+10.4f} "
            f"{s.variance:>10.4f} {s.mse:>10.4f} {ci_str:>22} "
            f"{s.bias_pvalue:>9.4f}{sig} {s.mean_acceptance:>7.1%}"
        )


def _print_final_summary(
    all_results: dict[int, dict[str, EstimatorStats]],
    exact: float,
    n_qubits: int,
) -> None:
    """Print the grand summary table across all depths."""
    print("\n" + "=" * 100)
    print("  FINAL SUMMARY — Bias Study Results")
    print("=" * 100)
    print(f"  Exact ground-state energy: {exact:.6f} ({exact / n_qubits:.6f} per site)")
    print()

    # Header
    print(f"  {'Layers':>6} │ {'Estimator':<22} │ {'Bias':>10} │ "
          f"{'Std':>10} │ {'MSE':>10} │ {'95% CI':>22} │ {'p-value':>8} │ {'Acc%':>7}")
    print(f"  {'─' * 6}─┼─{'─' * 22}─┼─{'─' * 10}─┼─"
          f"{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 22}─┼─{'─' * 8}─┼─{'─' * 7}")

    for n_layers in sorted(all_results.keys()):
        for i, key in enumerate(["raw", "ancilla", "ancilla_galton"]):
            s = all_results[n_layers][key]
            ci_str = f"[{s.ci_lower:+.4f}, {s.ci_upper:+.4f}]"
            sig = " *" if s.bias_pvalue < 0.05 else ""
            layer_str = str(n_layers) if i == 0 else ""
            print(
                f"  {layer_str:>6} │ {s.name:<22} │ {s.bias:>+10.4f} │ "
                f"{s.std:>10.4f} │ {s.mse:>10.4f} │ {ci_str:>22} │ "
                f"{s.bias_pvalue:>8.4f}{sig} │ {s.mean_acceptance:>6.1%}"
            )
        print(f"  {'─' * 6}─┼─{'─' * 22}─┼─{'─' * 10}─┼─"
              f"{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 22}─┼─{'─' * 8}─┼─{'─' * 7}")

    # Conclusion
    print()
    print("  INTERPRETATION:")
    print("  ───────────────")
    print("  NOTE: All estimators show large 'bias' relative to E_exact because the")
    print("  random ansatz (no VQE optimisation) produces states far from the ground")
    print("  state. The key question is whether filtering introduces ADDITIONAL bias")
    print("  beyond the raw estimator, or whether it simply reduces noise.")
    print()

    # Differential bias test: paired t-test (filtered - raw) across trials
    from scipy import stats as sp_stats

    for n_layers in sorted(all_results.keys()):
        raw_energies = np.array(all_results[n_layers]["raw"].energies)
        print(f"  Layers={n_layers}:")

        for key, label in [("ancilla", "Ancilla"), ("ancilla_galton", "Anc+Galton")]:
            filtered_energies = np.array(all_results[n_layers][key].energies)
            # Paired difference: positive = filtering pushed energy more negative (closer to ground)
            diff = filtered_energies - raw_energies
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))

            raw_mse = all_results[n_layers]["raw"].mse
            filt_mse = all_results[n_layers][key].mse
            mse_reduction = (1 - filt_mse / raw_mse) * 100 if raw_mse > 0 else 0

            # Wilcoxon signed-rank test (non-parametric, no normality assumption)
            if len(diff) >= 5 and np.any(diff != 0):
                try:
                    _, p_wilcoxon = sp_stats.wilcoxon(diff, alternative='two-sided')
                except Exception:
                    p_wilcoxon = 1.0
            else:
                p_wilcoxon = 1.0

            sig = "***" if p_wilcoxon < 0.001 else "** " if p_wilcoxon < 0.01 else "*  " if p_wilcoxon < 0.05 else "n.s."

            print(f"    {label:<15} ΔE = {mean_diff:+.4f} ± {std_diff:.4f}  "
                  f"MSE reduction: {mse_reduction:+.1f}%  "
                  f"Wilcoxon p={p_wilcoxon:.4f} [{sig}]")

            # Interpret direction
            if mean_diff < 0 and p_wilcoxon < 0.05:
                print(f"      → Filtering LOWERS energy (closer to ground state) — systematic & significant")
            elif mean_diff > 0 and p_wilcoxon < 0.05:
                print(f"      → Filtering RAISES energy (away from ground state) — potential bias concern")
            elif p_wilcoxon >= 0.05:
                print(f"      → Filtering effect not statistically distinguishable from zero")

        # Check variance reduction
        raw_var = all_results[n_layers]["raw"].variance
        for key, label in [("ancilla", "Ancilla"), ("ancilla_galton", "Anc+Galton")]:
            filt_var = all_results[n_layers][key].variance
            var_ratio = filt_var / raw_var if raw_var > 0 else 1.0
            print(f"    {label:<15} Variance ratio: {var_ratio:.4f} "
                  f"({'REDUCED' if var_ratio < 1 else 'increased'})")

        print()


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════


def _save_results(
    all_results: dict[int, dict[str, EstimatorStats]],
    n_qubits: int,
    shots: int,
    n_trials: int,
    output_dir: str,
) -> str:
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"bias_study_{n_qubits}q_{n_trials}t_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    data = {
        "experiment": "bias_study",
        "timestamp": timestamp,
        "n_qubits": n_qubits,
        "shots_per_trial": shots,
        "n_trials": n_trials,
        "j_coupling": J_COUPLING,
        "h_field": H_FIELD,
        "exact_energy": tfim_exact_ground_energy(n_qubits, J_COUPLING, H_FIELD),
        "noise_model": "IBM Heron-class (T1=300µs, T2=150µs, dep_1q=1e-3, dep_2q=1e-2)",
        "results": {},
    }

    for n_layers, stats_dict in all_results.items():
        layer_data = {}
        for key, s in stats_dict.items():
            layer_data[key] = {
                "name": s.name,
                "mean_energy": s.mean_energy,
                "bias": s.bias,
                "variance": s.variance,
                "std": s.std,
                "mse": s.mse,
                "ci_95_lower": s.ci_lower,
                "ci_95_upper": s.ci_upper,
                "bias_pvalue": s.bias_pvalue,
                "mean_acceptance": s.mean_acceptance,
                "energies": s.energies,
            }
        data["results"][str(n_layers)] = layer_data

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Bias study: evaluate systematic bias in trajectory filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bias_study.py                         # 8q, 30 trials
  python run_bias_study.py --n-qubits 6 --trials 50
  python run_bias_study.py --layers 1 2 3 4 5
  python run_bias_study.py --shots 200000 --trials 100
        """,
    )
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS,
                        help=f"Number of qubits (default: {DEFAULT_N_QUBITS})")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                        help=f"Shots per trial (default: {DEFAULT_SHOTS:,})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Number of independent trials (default: {DEFAULT_TRIALS})")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help=f"Ansatz layer depths (default: {DEFAULT_LAYERS})")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")

    args = parser.parse_args()

    run_bias_study(
        n_qubits=args.n_qubits,
        shots=args.shots,
        n_trials=args.trials,
        layers_list=args.layers,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
