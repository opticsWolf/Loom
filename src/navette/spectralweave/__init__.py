# -*- coding: utf-8 -*-
"""
``navette.spectralweave`` — spectral fragment weaving and merit evaluation.

Thin wrappers over the private native extension
``navette._spectralweave`` built from ``rust/navette/src/spectralweave``::

    maturin develop  # from the repo root
"""

from __future__ import annotations

from .optical import OpticalFragment, SimulationWeaver
from .target import SpectralTarget, AngularTarget, ColorTarget, TargetCollection

try:
  from navette._spectralweave import (
    OpticalWeaver,
    SpectralDataFrame,
    OpticalCollection,
    TargetWeaver,
    calculate_merit,
  )
except ImportError as exc:  # pragma: no cover - native not built yet
  raise ImportError(
    "Could not import the compiled `navette._spectralweave` extension. "
    "Build the Rust crate so it is importable, then retry:\n"
    "    maturin develop  # from the repo root"
  ) from exc

__all__ = [
  "OpticalFragment",
  "SimulationWeaver",
  "SpectralTarget",
  "AngularTarget",
  "ColorTarget",
  "TargetCollection",
  "OpticalWeaver",
  "SpectralDataFrame",
  "OpticalCollection",
  "TargetWeaver",
  "calculate_merit",
]
