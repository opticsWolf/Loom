# -*- coding: utf-8 -*-
"""
Navette: Weaving the mathematics of light in thin film systems.

Unified Python package. Rust-accelerated subpackages are thin wrappers
over private native extensions built with maturin from ``crates/*``:

==================  ============================  ===========================
public wrapper      native extension              crate
==================  ============================  ===========================
``navette.color``         ``navette._color``            ``crates/navette-color``
``navette.interpolate``   ``navette._interpolate``      ``crates/navette-interpolate``
``navette.smatrix``       ``navette._smatrix``          ``crates/navette-smatrix``
``navette.spectralweave`` ``navette._spectralweave``    ``crates/navette-spectralweave``
``navette.materials``     ``navette.materials._native`` ``crates/navette-materials-py``
==================  ============================  ===========================

Pure-Python subpackages (no build needed):

- ``navette.structure`` — layer stacks, architect, solver arrays
- ``navette.config`` — YAML/JSON material libraries and stack configs
- ``navette.data`` — bundled CIE reference spectra

Build the Rust extensions with e.g.::

    pip install -e .
    maturin develop -m crates/navette-smatrix-py/Cargo.toml
    maturin develop -m crates/navette-color-py/Cargo.toml
    ...

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
