# -*- coding: utf-8 -*-
"""
``navette.structure`` — thin-film layer stacks and architect.

Pure Python (numpy). No Rust build needed.
"""

from __future__ import annotations

from .types import (
  FLOAT_TYPE,
  COMPLEX_TYPE,
  INT_TYPE,
  ErrorType,
  RoughnessType,
  ErrorMask,
  LayerMask,
  InterpolationSettings,
  SolverArrays,
)
from .materials import MaterialProvider, DictMaterialProvider, MaterialObjectProvider
from .models import Layer, Group
from .structure import Navette_Structure
from .architect import Navette_Architect, StructureBlock
from .expander import _LayerExpander

__all__ = [
  "FLOAT_TYPE",
  "COMPLEX_TYPE",
  "INT_TYPE",
  "ErrorType",
  "RoughnessType",
  "ErrorMask",
  "LayerMask",
  "InterpolationSettings",
  "SolverArrays",
  "MaterialProvider",
  "DictMaterialProvider",
  "MaterialObjectProvider",
  "Layer",
  "Group",
  "Navette_Structure",
  "Navette_Architect",
  "StructureBlock",
]
