---
description: >-
  qgate Python API reference. Auto-generated documentation for GateConfig, TrajectoryFilter,
  scoring, thresholding, run logging, ML-augmented mitigation (TelemetryMitigator,
  PulseMitigator, QgateTranspiler), Uzdin folding utilities, and all framework and
  algorithm adapters.
keywords: qgate API, Python API reference, GateConfig, TrajectoryFilter, scoring API, threshold API, adapter API, mkdocstrings, TelemetryMitigator, PulseMitigator, QgateTranspiler, Uzdin folding
---

# API Reference

## Core Modules

::: qgate.config
::: qgate.filter
::: qgate.scoring
::: qgate.threshold
::: qgate.run_logging

## ML-Augmented Mitigation Modules

!!! note "CIP Addendum"
    These modules implement the ML-augmented TSVF trajectory mitigation
    pipeline from the CIP filing.  See [ML-Augmented Mitigation](concepts/ml-mitigation.md)
    for architecture overview.

### Transpiler — ML-Aware Circuit Compilation

::: qgate.transpiler
    options:
      members:
        - QgateTranspilerConfig
        - QgateTranspiler
        - CompilationResult
        - MitigationMode
        - validate_noise_scale_factor
        - apply_uzdin_unitary_folding

### TelemetryMitigator — Level-2 Bit-String ML

::: qgate.mitigation
    options:
      members:
        - MitigatorConfig
        - TelemetryMitigator
        - CalibrationResult
        - MitigationResult
        - FEATURE_NAMES

### PulseMitigator — Level-1 IQ Analog ML

::: qgate.pulse_mitigator
    options:
      members:
        - PulseMitigatorConfig
        - PulseMitigator
        - PulseCalibrationResult
        - DriftPrediction
        - ActiveCancellationResult
        - SimulatedPulseSchedule
        - extract_iq_features
        - extract_iq_features_batch
        - IQ_FEATURE_NAMES

### TVS — Trajectory Viability Score (HF/LF Fusion)

::: qgate.tvs
    options:
      members:
        - process_telemetry_batch
        - adaptive_galton_schedule
        - normalise_hf_level1
        - normalise_hf_level2
        - compute_iq_snr
        - compute_alpha_static
        - compute_alpha_dynamic
        - fuse_scores
        - galton_filter
        - VALID_MODES
        - VALID_FORCE_MODES

## Framework Adapters

::: qgate.adapters.base
::: qgate.adapters.registry
::: qgate.adapters.qiskit_adapter

## Algorithm TSVF Adapters

::: qgate.adapters.grover_adapter
::: qgate.adapters.qaoa_adapter
::: qgate.adapters.vqe_adapter
::: qgate.adapters.qpe_adapter

## Backward-Compatible Modules

::: qgate.conditioning
::: qgate.monitors
