# -*- coding: utf-8 -*-
"""
``navette.interpolate`` — fast univariate interpolation (PCHIP, Sprague,
Floater-Hormann, …) with batch support.

Thin wrapper over the private native extension ``navette._interpolate``
built from ``crates/navette-interpolate``::

    maturin develop  # from the repo root
"""

from __future__ import annotations

try:
  from navette._interpolate import UniInterpolator
except ImportError as exc:  # pragma: no cover - native not built yet
  raise ImportError(
    "Could not import the compiled `navette._interpolate` extension. "
    "Build the Rust crate so it is importable, then retry:\n"
    "    maturin develop  # from the repo root"
  ) from exc

__all__ = ["UniInterpolator"]
