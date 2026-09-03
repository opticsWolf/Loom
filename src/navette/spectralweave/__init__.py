# -*- coding: utf-8 -*-
"""
``navette.spectralweave`` — spectral fragment weaving and merit evaluation.

Thin wrappers over the private native extension
``navette._spectralweave`` built from ``crates/navette-spectralweave``::

    maturin develop -m crates/navette-spectralweave/Cargo.toml
"""

from __future__ import annotations

from .optical import OpticalFragment, SimulationWeaver
from .target import SpectralTarget, AngularTarget, TargetCollection

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
    "    maturin develop -m crates/navette-spectralweave/Cargo.toml"
  ) from exc

__all__ = [
  "OpticalFragment",
  "SimulationWeaver",
  "SpectralTarget",
  "AngularTarget",
  "TargetCollection",
  "OpticalWeaver",
  "SpectralDataFrame",
  "OpticalCollection",
  "TargetWeaver",
  "calculate_merit",
]
