# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import sys
import importlib
import pkgutil
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Any] = {}
_IS_INITIALIZED: bool = False
_DISCOVERY_LOCK = Lock()


# ---------------------------------------------------------------------------
# Internal Discovery Logic
# ---------------------------------------------------------------------------

def _discover_models() -> None:
    """Discover and register public callables defined in package modules.

    This function:
    - Iterates over modules in this package directory.
    - Avoids reloading, thanks to an initialization guard.
    - Imports modules using relative imports.
    - Registers public callables that originate from the module itself.

    Discovery is idempotent and thread-safe.
    """
    global _IS_INITIALIZED

    if _IS_INITIALIZED:
        return

    with _DISCOVERY_LOCK:
        if _IS_INITIALIZED:
            return

        package_path = [str(Path(__file__).parent)]

        # Deterministic ordering: sort module names before importing
        discovered = sorted(pkgutil.iter_modules(package_path), key=lambda t: t[1])

        for finder, mod_name, is_pkg in discovered:
            if is_pkg:
                continue

            if mod_name in ("loader", "__init__"):
                continue

            try:
                module = importlib.import_module(f".{mod_name}", package=__name__)
            except ImportError as exc:
                print(f"PADDOL Warning: Failed to import '{mod_name}': {exc}", file=sys.stderr)
                continue

            for name, obj in vars(module).items():
                if (
                    callable(obj)
                    and not name.startswith("_")
                    and getattr(obj, "__module__", None) == module.__name__
                ):
                    _MODEL_REGISTRY[name] = obj

        _IS_INITIALIZED = True


# ---------------------------------------------------------------------------
# Lazy Loading (PEP 562)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """Lazy attribute access hook.

    Trigger discovery on first attribute lookup,
    then resolve the symbol from the registry.
    """
    if not _IS_INITIALIZED:
        _discover_models()

    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """Extended dir() to include lazily-discovered names."""
    if not _IS_INITIALIZED:
        _discover_models()

    base = list(globals().keys())
    dynamic = list(_MODEL_REGISTRY.keys())
    return sorted(set(base + dynamic))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_registry() -> Dict[str, Any]:
    """Return the dictionary of discovered model callables.

    Returns:
        Dict[str, Any]: Mapping of public model names to model objects.
    """
    if not _IS_INITIALIZED:
        _discover_models()
    return _MODEL_REGISTRY
