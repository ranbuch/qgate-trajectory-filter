"""
registry.py — Entry-point based adapter discovery.

Discovers adapters registered under the ``qgate.adapters`` entry-point
group and provides :func:`list_adapters` / :func:`load_adapter` for
programmatic and CLI access.

Example::

    from qgate.adapters.registry import list_adapters, load_adapter

    print(list_adapters())          # {"mock": "qgate.adapters.base:MockAdapter", ...}
    AdapterCls = load_adapter("mock")
    adapter = AdapterCls(error_rate=0.05, seed=42)

Patent reference: US App. Nos. 63/983,831 & 63/989,632 | IL App. No. 326915
"""

from __future__ import annotations

import sys
from typing import Any

from importlib.metadata import entry_points

_EP_GROUP = "qgate.adapters"


def _get_group():
    """Return entry points for the qgate.adapters group, compatible with Python 3.9–3.13+."""
    if sys.version_info >= (3, 10):
        # Python 3.10+: entry_points() accepts group= keyword directly
        return entry_points(group=_EP_GROUP)
    else:
        # Python 3.9: entry_points() returns a plain dict keyed by group
        return entry_points().get(_EP_GROUP, [])


def list_adapters() -> dict[str, str]:
    """Return ``{name: "module:Class"}`` for all registered adapters.

    Reads the ``qgate.adapters`` entry-point group.
    """
    return {ep.name: ep.value for ep in _get_group()}


def load_adapter(name: str, **kwargs: Any) -> type:
    """Load and return the adapter **class** registered under *name*.

    Args:
        name: Adapter name as registered in entry points (e.g. ``"mock"``).
        **kwargs: Currently unused; reserved for future configuration.

    Returns:
        The adapter class (not an instance).

    Raises:
        KeyError: If *name* is not a registered adapter.
        ImportError: If the adapter's optional dependency is missing.
    """
    for ep in _get_group():
        if ep.name == name:
            cls: type = ep.load()
            return cls

    available = sorted(list_adapters().keys())
    raise KeyError(
        f"Unknown adapter {name!r}. "
        f"Available adapters: {', '.join(available) or '(none)'}. "
        f"Install extras to register more (e.g. pip install qgate[qiskit])."
    )
