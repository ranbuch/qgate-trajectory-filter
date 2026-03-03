"""
run_logging.py — Structured run logging (JSON / CSV / Parquet).

Every :class:`~qgate.filter.TrajectoryFilter` run can be logged to
disk for reproducibility and analysis.

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger("qgate.run_logging")


def _get_pandas():
    """Lazy import of pandas — avoids ~200 ms cold-start penalty."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for CSV / Parquet logging.  "
            "Install with:  pip install qgate[csv]  or  pip install qgate[parquet]"
        ) from None
    return pd

# ---------------------------------------------------------------------------
# Run-ID computation (deterministic SHA-256 digest)
# ---------------------------------------------------------------------------

def compute_run_id(
    config_json: str,
    adapter_name: str = "",
    circuit_hash: str = "",
) -> str:
    """Return a deterministic 12-char hex run ID (SHA-256 prefix).

    The ID is computed from a canonical JSON blob combining
    *config_json*, *adapter_name*, and an optional *circuit_hash*.
    Two runs with identical inputs always produce the same ID, enabling
    deduplication and reproducibility checks.

    Args:
        config_json:  Serialised :class:`~qgate.config.GateConfig` JSON.
        adapter_name: Name/class of the adapter used.
        circuit_hash: Optional hash of the circuit object for extra
                      specificity.

    Returns:
        12-character lowercase hex string.
    """
    blob: dict[str, Any] = {
        "config": json.loads(config_json),
        "adapter": adapter_name,
    }
    if circuit_hash:
        blob["circuit_hash"] = circuit_hash
    # Canonical JSON: sorted keys, no extra whitespace, coerce numpy types
    canonical = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _json_default(obj: Any) -> Any:
    """JSON fallback encoder for numpy scalars and other exotic types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Structured output of a single trajectory-filter run.

    Attributes:
        run_id:                 Deterministic 12-char hex digest for
                                deduplication and reproducibility.
        variant:                Conditioning strategy used.
        total_shots:            Number of shots executed.
        accepted_shots:         Number of accepted shots.
        acceptance_probability: accepted / total.
        tts:                    Time-to-solution (1 / acceptance_probability).
        mean_combined_score:    Mean combined fusion score across shots.
        threshold_used:         Threshold at the time of filtering.
        dynamic_threshold_final: Final dynamic threshold (if enabled).
        scores:                 Per-shot combined scores.
        config_json:            Serialised GateConfig as JSON string.
        metadata:               Free-form metadata.
        timestamp:              ISO-8601 timestamp.
    """

    run_id: str = ""
    variant: str = ""
    total_shots: int = 0
    accepted_shots: int = 0
    acceptance_probability: float = 0.0
    tts: float = float("inf")
    mean_combined_score: float | None = None
    threshold_used: float = 0.65
    dynamic_threshold_final: float | None = None
    scores: list[float] = field(default_factory=list)
    config_json: str = "{}"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop large per-shot scores from the summary dict
        d.pop("scores", None)
        return d


# ---------------------------------------------------------------------------
# Run logger
# ---------------------------------------------------------------------------

class RunLogger:
    """Append-only logger that writes :class:`FilterResult` records.

    Supports JSON-Lines, CSV, and (optionally) Parquet output.

    Args:
        path:   Output file path (suffix determines format:
                ``.jsonl``, ``.csv``, or ``.parquet``).
        fmt:    Explicit format override (``"jsonl"`` | ``"csv"`` |
                ``"parquet"``).  If *None* the format is inferred
                from *path*.
    """

    def __init__(
        self,
        path: str | Path,
        fmt: Literal["jsonl", "csv", "parquet"] | None = None,
    ) -> None:
        self.path = Path(path)
        self._fmt: Literal["jsonl", "csv", "parquet"]
        if fmt is not None:
            self._fmt = fmt
        else:
            suffix = self.path.suffix.lower()
            mapping: dict[str, Literal["jsonl", "csv", "parquet"]] = {
                ".jsonl": "jsonl", ".csv": "csv", ".parquet": "parquet",
            }
            if suffix not in mapping:
                logger.warning(
                    "Unknown file extension %r for %s — defaulting to JSONL format. "
                    "Supported extensions: .jsonl, .csv, .parquet",
                    suffix, self.path,
                )
            self._fmt = mapping.get(suffix, "jsonl")
        self._records: list[dict[str, Any]] = []

    @property
    def format(self) -> str:
        return self._fmt

    def log(self, result: FilterResult) -> None:
        """Append a result to the in-memory buffer and flush to disk."""
        self._records.append(result.as_dict())
        self._flush_one(result)

    def flush_all(self) -> None:
        """Re-write the entire file from the in-memory buffer.

        Useful if you want to guarantee the file is in sync.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._fmt == "jsonl":
            with open(self.path, "w") as f:
                for rec in self._records:
                    f.write(json.dumps(rec, default=str) + "\n")
        elif self._fmt == "csv":
            pd = _get_pandas()
            df = pd.DataFrame(self._records)
            df.to_csv(self.path, index=False)
        elif self._fmt == "parquet":
            self._write_parquet()
        else:
            raise ValueError(f"Unknown format: {self._fmt}")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush remaining buffered records (especially Parquet) and release resources."""
        if self._fmt == "parquet" and self._records:
            self._write_parquet()
        logger.debug("RunLogger closed – %d records written to %s", len(self._records), self.path)

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _flush_one(self, result: FilterResult) -> None:
        """Incremental append (append-friendly for jsonl / csv)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._fmt == "jsonl":
            with open(self.path, "a") as f:
                f.write(json.dumps(result.as_dict(), default=str) + "\n")
        elif self._fmt == "csv":
            pd = _get_pandas()
            df = pd.DataFrame([result.as_dict()])
            header = not self.path.exists() or self.path.stat().st_size == 0
            df.to_csv(self.path, mode="a", index=False, header=header)
        elif self._fmt == "parquet":
            # Parquet: buffer only — flushed on close() or explicit flush_all()
            logger.debug("Parquet record buffered (total: %d)", len(self._records))
        else:
            raise ValueError(f"Unknown format: {self._fmt}")

    def _write_parquet(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet logging.  "
                "Install with:  pip install qgate[parquet]"
            ) from None
        pd = _get_pandas()
        df = pd.DataFrame(self._records)
        df.to_parquet(self.path, index=False)
