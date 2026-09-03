# -*- coding: utf-8 -*-
"""
navette.materials — optical dispersion models.

Thin Python wrappers (parameter management, validation, caching, composition)
over the Rust kernels in ``navette.materials._native``. The public class API is
unchanged from the pre-Rust version; only the kernel call sites differ.
"""

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
