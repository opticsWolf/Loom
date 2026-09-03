# -*- coding: utf-8 -*-
"""Shim: re-export the ``_smatrix`` submodule of the aggregated extension.

The single ``navette._navette`` native module (built by ``maturin develop``
from the repo root) hosts all engines as submodules; this file keeps the
private ``navette._smatrix`` import path stable and lazily loads the
extension only when this submodule is imported.
"""

from __future__ import annotations

from navette._navette import _smatrix as _ext

__all__ = [n for n in dir(_ext) if not n.startswith("_")]

globals().update({n: getattr(_ext, n) for n in __all__})

del _ext
