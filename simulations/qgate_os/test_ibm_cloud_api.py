#!/usr/bin/env python3
"""
test_ibm_cloud_api.py — Cloud Plumbing smoke test for QgateSampler OS.

Goal: Prove that QgateSampler's modified circuits and repacked results
survive serialization over the *actual* IBM Quantum Cloud API —
round-trip through Qiskit Runtime on real QPU hardware.

This is NOT a physics test.  It verifies:
  ✓ Circuit with injected probe ancilla serialises to QPN/QASM correctly
  ✓ Job submits and completes on real hardware
  ✓ Returned PrimitiveResult parses without KeyError / shape mismatch
  ✓ Galton filter runs cleanly on hardware BitArrays
  ✓ Filtered result exposes .data.meas with correct shape

Protocol:
  1. Connect to IBM Quantum via saved credentials (or --token).
  2. Pick the least-busy real backend (≥2 qubits) — or use --backend.
  3. Build a trivial 2-qubit Bell state circuit.
  4. Submit via QgateSampler with 100 shots (minimal QPU time).
  5. Wait for result.
  6. Assert clean parse: PrimitiveResult → PubResult → DataBin → BitArray.
  7. Print success / failure with job ID for IBM dashboard verification.

**NOTICE — PRE-PATENT PROPRIETARY CODE**
Do NOT distribute, publish, or push to any public repository.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915

Usage:
    python simulations/qgate_os/test_ibm_cloud_api.py
    python simulations/qgate_os/test_ibm_cloud_api.py --backend ibm_brisbane
    python simulations/qgate_os/test_ibm_cloud_api.py --token <YOUR_TOKEN>
    python simulations/qgate_os/test_ibm_cloud_api.py --shots 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# ═══════════════════════════════════════════════════════════════════════════
# IBM Quantum Service + Backend
# ═══════════════════════════════════════════════════════════════════════════

def connect_ibm_service(token: str | None = None) -> Any:
    """Initialise QiskitRuntimeService with token resolution.

    Token precedence:
      1. Explicit --token argument
      2. IBMQ_TOKEN environment variable
      3. .secrets.json in repo root
      4. Previously saved credentials (ibm_quantum_platform channel)
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Resolve token
    if not token:
        token = os.environ.get("IBMQ_TOKEN")
    if not token:
        secrets = ROOT / ".secrets.json"
        if secrets.is_file():
            try:
                with open(secrets) as f:
                    token = json.load(f).get("ibmq_token")
            except Exception:
                pass

    if token:
        try:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=token,
                overwrite=True,
            )
        except Exception:
            pass  # may already be saved
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token,
        )
    else:
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
        except Exception as e:
            print(
                "  ✘ No IBM Quantum token found.\n"
                "    Provide --token <TOKEN>, set IBMQ_TOKEN, "
                "or save credentials via QiskitRuntimeService.save_account().\n"
                f"    Error: {e}"
            )
            sys.exit(1)

    return service


def select_backend(
    service: Any,
    backend_name: str | None = None,
    min_qubits: int = 2,
) -> Any:
    """Select a backend: explicit name or least-busy real hardware."""
    if backend_name:
        print(f"  Requesting specific backend: {backend_name}")
        backend = service.backend(backend_name)
    else:
        print(f"  Searching for least-busy backend (≥{min_qubits} qubits)...")
        backend = service.least_busy(
            min_num_qubits=min_qubits,
            simulator=False,
            operational=True,
        )
    print(f"  ✓ Selected: {backend.name} ({backend.num_qubits} qubits)")
    return backend


# ═══════════════════════════════════════════════════════════════════════════
# Bell state circuit
# ═══════════════════════════════════════════════════════════════════════════

def build_bell_circuit() -> Any:
    """Trivial 2-qubit Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, name="Bell_2q")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# ═══════════════════════════════════════════════════════════════════════════
# Cloud smoke test
# ═══════════════════════════════════════════════════════════════════════════

def run_cloud_smoke_test(
    backend: Any,
    shots: int = 100,
) -> dict[str, Any]:
    """Submit a Bell circuit via QgateSampler to real IBM hardware.

    Returns a report dict with all validation results.
    """
    from qgate.sampler import QgateSampler, SamplerConfig

    report: dict[str, Any] = {
        "backend": backend.name,
        "shots_requested": shots,
        "passed": False,
        "checks": {},
    }

    # --- Build circuit ---
    qc = build_bell_circuit()
    print(f"\n  Circuit: {qc.name}  ({qc.num_qubits} qubits, depth {qc.depth()})")

    # --- Initialise QgateSampler ---
    config = SamplerConfig(
        target_acceptance=0.25,
        probe_angle=math.pi / 6,
        window_size=256,
        min_window_size=20,
        baseline_threshold=0.65,
        optimization_level=1,
        oversample_factor=1.5,
    )
    sampler = QgateSampler(backend=backend, config=config)
    print(f"  QgateSampler initialised (target_acceptance={config.target_acceptance})")

    # --- Submit job ---
    print(f"\n  Submitting job ({shots} shots) to {backend.name}...")
    t0 = time.time()
    job = sampler.run([(qc,)], shots=shots)

    # The QgateSamplerResult wraps the raw job — get the raw job ID
    # from the inner result chain for dashboard tracking
    raw_result = job._raw  # the underlying PrimitiveResult or RuntimeJobV2
    job_id = getattr(raw_result, "job_id", None)
    if job_id is None:
        # Try to get it from the raw result's metadata
        job_id = "N/A (result already resolved)"

    print(f"  Job ID: {job_id}")
    print(f"  ⏳ Waiting for result from {backend.name}...")

    report["job_id"] = job_id

    # --- Wait for result ---
    try:
        result = job.result()
        elapsed = time.time() - t0
        report["elapsed_s"] = round(elapsed, 1)
        print(f"  ✓ Result received in {elapsed:.1f}s")
    except Exception as e:
        report["error"] = str(e)
        report["checks"]["result_received"] = False
        print(f"  ✘ Job failed: {e}")
        traceback.print_exc()
        return report

    report["checks"]["result_received"] = True

    # ── Check 1: PrimitiveResult structure ───────────────────────────
    try:
        assert hasattr(result, "__getitem__"), "result not indexable"
        pub_result = result[0]
        report["checks"]["primitive_result_indexable"] = True
        print("  ✓ Check 1: PrimitiveResult is indexable")
    except Exception as e:
        report["checks"]["primitive_result_indexable"] = False
        print(f"  ✘ Check 1 FAILED: {e}")
        return report

    # ── Check 2: DataBin has .meas register ──────────────────────────
    try:
        data_bin = pub_result.data
        assert hasattr(data_bin, "meas"), (
            f"DataBin missing 'meas' register. "
            f"Available: {[f for f in dir(data_bin) if not f.startswith('_')]}"
        )
        bitarray = data_bin.meas
        report["checks"]["databin_has_meas"] = True
        print(f"  ✓ Check 2: DataBin has .meas register (type={type(bitarray).__name__})")
    except Exception as e:
        report["checks"]["databin_has_meas"] = False
        print(f"  ✘ Check 2 FAILED: {e}")
        return report

    # ── Check 3: BitArray is well-formed ─────────────────────────────
    try:
        ba_shape = bitarray.shape
        ba_num_bits = bitarray.num_bits
        ba_num_shots = bitarray.num_shots
        report["bitarray"] = {
            "shape": str(ba_shape),
            "num_bits": ba_num_bits,
            "num_shots": ba_num_shots,
        }
        assert ba_num_bits == 2, f"Expected 2 bits (2-qubit Bell), got {ba_num_bits}"
        assert ba_num_shots > 0, f"Zero shots returned"
        report["checks"]["bitarray_wellformed"] = True
        print(
            f"  ✓ Check 3: BitArray well-formed "
            f"(shape={ba_shape}, bits={ba_num_bits}, shots={ba_num_shots})"
        )
    except Exception as e:
        report["checks"]["bitarray_wellformed"] = False
        print(f"  ✘ Check 3 FAILED: {e}")
        return report

    # ── Check 4: get_counts() works (no KeyError) ────────────────────
    try:
        counts = bitarray.get_counts()
        assert isinstance(counts, dict), f"get_counts() returned {type(counts)}"
        total = sum(counts.values())
        assert total > 0, "Counts sum to 0"
        report["counts"] = counts
        report["checks"]["get_counts_works"] = True
        print(f"  ✓ Check 4: get_counts() → {counts}  (total={total})")
    except Exception as e:
        report["checks"]["get_counts_works"] = False
        print(f"  ✘ Check 4 FAILED: {e}")
        return report

    # ── Check 5: Galton filter metadata present ──────────────────────
    try:
        metadata = pub_result.metadata
        assert "qgate_filter" in metadata, (
            f"Missing 'qgate_filter' in metadata. Keys: {list(metadata.keys())}"
        )
        qf = metadata["qgate_filter"]
        report["qgate_filter"] = qf
        report["checks"]["galton_metadata_present"] = True
        print(
            f"  ✓ Check 5: Galton filter metadata present — "
            f"accepted={qf.get('accepted_shots')}/{qf.get('total_shots')} "
            f"(rate={qf.get('acceptance_rate', 0):.1%}, "
            f"threshold={qf.get('threshold', 0):.4f})"
        )
    except Exception as e:
        report["checks"]["galton_metadata_present"] = False
        print(f"  ✘ Check 5 FAILED: {e}")
        return report

    # ── Check 6: probe register is stripped (no qgate_probe in data) ─
    try:
        data_fields = [f for f in dir(data_bin) if not f.startswith("_")]
        has_probe = "qgate_probe" in data_fields
        report["checks"]["probe_stripped"] = not has_probe
        if has_probe:
            print(f"  ⚠ Check 6: qgate_probe register still present in DataBin")
        else:
            print(f"  ✓ Check 6: Probe register transparently stripped from DataBin")
    except Exception as e:
        report["checks"]["probe_stripped"] = False
        print(f"  ✘ Check 6 FAILED: {e}")

    # ── Check 7: Bell state physics sanity (optional) ────────────────
    try:
        # For a Bell state we expect mostly |00⟩ and |11⟩ (even with noise)
        correlated = counts.get("00", 0) + counts.get("11", 0)
        total = sum(counts.values())
        corr_frac = correlated / total if total > 0 else 0
        report["bell_correlation"] = round(corr_frac, 3)
        report["checks"]["bell_sanity"] = corr_frac > 0.3  # very loose bar
        if corr_frac > 0.3:
            print(
                f"  ✓ Check 7: Bell correlation sanity — "
                f"|00⟩+|11⟩ = {corr_frac:.1%} (>30% threshold)"
            )
        else:
            print(
                f"  ⚠ Check 7: Low Bell correlation — "
                f"|00⟩+|11⟩ = {corr_frac:.1%} (noise may dominate at 100 shots)"
            )
    except Exception:
        report["checks"]["bell_sanity"] = None

    # ── All checks passed ────────────────────────────────────────────
    all_critical = all(
        report["checks"].get(k, False)
        for k in [
            "result_received",
            "primitive_result_indexable",
            "databin_has_meas",
            "bitarray_wellformed",
            "get_counts_works",
            "galton_metadata_present",
        ]
    )
    report["passed"] = all_critical
    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QgateSampler OS — IBM Cloud API smoke test",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="IBM Quantum API token (or set IBMQ_TOKEN env var)",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="Specific backend name (e.g. ibm_brisbane, ibm_osaka, ibm_torino)",
    )
    parser.add_argument(
        "--shots", type=int, default=100,
        help="Number of shots (default: 100 — minimal QPU cost)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  QgateSampler OS — IBM Cloud API Smoke Test")
    print("=" * 72)

    # --- Connect ---
    print("\n  Step 1: Connecting to IBM Quantum...")
    service = connect_ibm_service(token=args.token)
    print("  ✓ Connected to IBM Quantum Platform")

    # --- Select backend ---
    print("\n  Step 2: Selecting backend...")
    backend = select_backend(service, backend_name=args.backend)

    # --- Run smoke test ---
    print("\n  Step 3: Running cloud smoke test...")
    report = run_cloud_smoke_test(backend=backend, shots=args.shots)

    # --- Final verdict ---
    print("\n" + "=" * 72)
    if report["passed"]:
        print("  ✅ CLOUD SMOKE TEST PASSED")
        print()
        print(f"     Backend:         {report['backend']}")
        print(f"     Job ID:          {report.get('job_id', 'N/A')}")
        print(f"     Elapsed:         {report.get('elapsed_s', '?')}s")
        print(f"     Shots returned:  {report.get('bitarray', {}).get('num_shots', '?')}")
        if "qgate_filter" in report:
            qf = report["qgate_filter"]
            print(f"     Galton filter:   {qf.get('accepted_shots')}/{qf.get('total_shots')} accepted")
            print(f"     Threshold:       {qf.get('threshold', 0):.4f}")
        if "bell_correlation" in report:
            print(f"     Bell |00⟩+|11⟩:  {report['bell_correlation']:.1%}")
        print()
        print("     QgateSampler circuits and results survive IBM Cloud serialization.")
        print("=" * 72)
        sys.exit(0)
    else:
        print("  ❌ CLOUD SMOKE TEST FAILED")
        print()
        for check, passed in report.get("checks", {}).items():
            symbol = "✓" if passed else "✘"
            print(f"     {symbol}  {check}")
        if "error" in report:
            print(f"\n     Error: {report['error']}")
        print("=" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()
