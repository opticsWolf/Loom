# -*- coding: utf-8 -*-
"""
Navette: Weaving the mathematics of light in thin film systems.

Unified Python package. Rust-accelerated subpackages are thin wrappers
over private native submodules of the single aggregated extension
``navette._navette`` (built with ``maturin develop`` from the repo root)::

==================  ========================
public wrapper      native submodule
==================  ========================
``navette.color``         ``navette._color``
``navette.interpolate``   ``navette._interpolate``
``navette.smatrix``       ``navette._smatrix``
``navette.spectralweave`` ``navette._spectralweave``
``navette.materials``     ``navette._materials``
==================  ========================

Pure-Python subpackages (no build needed):

- ``navette.structure`` — layer stacks, architect, solver arrays
- ``navette.config`` — YAML/JSON material libraries and stack configs
- ``navette.data`` — bundled CIE reference spectra

Build the Rust extension with::

    maturin develop

Wrappers raise a helpful ``ImportError`` (with the exact maturin command)
when their native module is missing, so ``import navette`` always works.
"""

from __future__ import annotations

from .__about__ import (
  __title__,
  __version__,
  __description__,
  __author__,
  __license__,
  __copyright__,
  metadata_summary,
)

__all__ = [
  "__title__",
  "__version__",
  "__description__",
  "__author__",
  "__license__",
  "__copyright__",
  "metadata_summary",
]
