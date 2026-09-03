# -*- coding: utf-8 -*-
"""
``navette.materials`` — optical dispersion models.

Pure-Python models (numba-accelerated where available) with optional
Rust kernels from ``navette.materials._native``
(``crates/navette-materials-py``)::

    maturin develop -m crates/navette-materials-py/Cargo.toml

The Python class API is stable; only kernel call sites use ``_native``
when it is importable.
"""

from __future__ import annotations

from .material import Material
from .basic import Konstant, TableMaterial
from .cauchy_sellmeier import Cauchy, CauchyUrbach, Sellmeier, SellmeierUrbach
from .lorentz import LorentzOscillator
from .drudelorentz import Drude, DrudeLorentz
from .codylorentz import CodyLorentz
from .forouhibloomer import (
  ForouhiBloomerInterbandSingle,
  ForouhiBloomerInterbandMulti,
  ForouhiBloomerMetalSingle,
  ForouhiBloomerMetal2021,
)
from .ema_material import EffectiveMaterial, RoughnessMaterial
from .UBF_Cody_Lorentz import UBF_CodyLorentz
from .tauclorentz import TaucLorentz

try:
  from . import _native  # noqa: F401
  _HAS_NATIVE = True
except ImportError:
  _HAS_NATIVE = False

__all__ = [
  "Material",
  "Konstant",
  "TableMaterial",
  "Cauchy",
  "CauchyUrbach",
  "Sellmeier",
  "SellmeierUrbach",
  "LorentzOscillator",
  "Drude",
  "DrudeLorentz",
  "CodyLorentz",
  "ForouhiBloomerInterbandSingle",
  "ForouhiBloomerInterbandMulti",
  "ForouhiBloomerMetalSingle",
  "ForouhiBloomerMetal2021",
  "EffectiveMaterial",
  "RoughnessMaterial",
  "UBF_CodyLorentz",
  "TaucLorentz",
]
