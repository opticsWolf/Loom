# -*- coding: utf-8 -*-
"""Layer and Group (bound Rust classes, re-exported).

The model lives first-class in the native ``navette._structure``
submodule; this module keeps the ``navette.structure.models`` import path
stable. behaviors (validation, expansion, states) are implemented once,
in Rust; only provider plumbing and the solve bridge stay Python-side.
"""

from __future__ import annotations

from navette._structure import Layer, Group

__all__ = ["Layer", "Group"]
