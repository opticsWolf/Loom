# -*- coding: utf-8 -*-
"""
``navette.structure`` — thin-film layer stacks and architect.

Rust-first model (``navette._structure``) with Python provider plumbing:
:class:`Layer`/:class:`Group` are bound classes re-exported from
:mod:`navette.structure.models`; :class:`Navette_Structure` and
:class:`Navette_Architect` are thin wrappers carrying the material
provider and owning bake pour-back. Provider vocabulary (``types`` /
``materials``) and the solve bridge stay Python-side.
"""

from __future__ import annotations

from .types import (
  FLOAT_TYPE,
  COMPLEX_TYPE,
  INT_TYPE,
  ErrorType,
  BlockKind,
  LayerType,
  OptMask,
  SCHEMA_VERSION,
  check_schema_version,
  RoughnessType,
  ErrorMask,
  LayerMask,
  InterpolationSettings,
  SolverArrays,
)
from .materials import MaterialProvider, DictMaterialProvider, MaterialObjectProvider
from .models import Layer, Group
from .structure import Navette_Structure, gate_validation
from .architect import Navette_Architect, StructureBlock

import warnings
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np

__all__ = [
  "FLOAT_TYPE",
  "COMPLEX_TYPE",
  "INT_TYPE",
  "ErrorType",
  "BlockKind",
  "LayerType",
  "OptMask",
  "SCHEMA_VERSION",
  "check_schema_version",
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
  "solve_structure",
]


def _assert_provider_grids(source: Any, wl: np.ndarray) -> None:
  """Value-compare every provider grid against the solver grid.

  Length-only agreement is rejected: equal length with different
  wavelengths is silently unphysical, so grids must match byte-for-byte
  (same idiom as the provider cache signatures). Providers without a
  known grid (``grid`` missing or ``None``) keep the length-only assert
  above, plus a warning nudging the caller to attach the grid. A grid
  change always re-resolves (new solve call / cache clear), so this
  single check covers the whole run.
  """
  structs = source.unique_structures if isinstance(source, Navette_Architect) else [source]
  warned = False
  for struct in structs:
    provider = struct.materials
    grid = getattr(provider, "grid", None)
    if grid is None:
      if not warned:
        warnings.warn(
          "solve_structure: provider grid unknown (gridless dict or custom "
          "provider); only array lengths are checked — attach the grid "
          "(wavelength=/target_wavelength=) for value-level assurance.",
          stacklevel=3,
        )
        warned = True
      continue
    grid = np.ascontiguousarray(np.asarray(grid, dtype=np.float64)).ravel()
    if grid.tobytes() != wl.tobytes():
      raise ValueError(
        f"solve_structure: provider grid does not match the solver grid "
        f"({grid.shape[0]} points vs {wl.shape[0]}); resample the material "
        f"data onto the solver wavelengths first."
      )


def solve_structure(
  source: Union[Navette_Structure, Navette_Architect],
  wavelengths: Sequence[float],
  angles: Union[float, Sequence[float]],
  *,
  request=None,
  errors: bool = False,
  rng: Optional[np.random.Generator] = None,
  **solver_opts: Any,
) -> Dict[str, np.ndarray]:
  """Expand a structure/architect and solve it (the documented engine path).

  Expands ``source`` to :class:`SolverArrays`, enforces the bridge
  contract, and runs :class:`navette.smatrix.ScatterMatrix`: all lengths
  [nm] (roughness sigma included), ``indices`` (n_layers, n_wavs) on the
  given grid, first/last rows ambient/substrate (0 thickness by
  convention — warned, not forced). ``Navette_Structure`` inputs are
  validated first (any issue raises); architect validation lands with
  STRUCT-6, so architect inputs get the array-level gate only.
  ``request=None`` returns reflectance/transmittance; pass a
  ``Request`` mask for :meth:`compute` output instead.
  """
  from navette.smatrix.smatrix import ScatterMatrix  # lazy: needs native

  if isinstance(source, Navette_Structure):
    gate_validation(source.validate(), "solve_structure")
  sa = source.get_error_solver_inputs(rng=rng) if errors \
    else source.get_solver_inputs()
  wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64)).ravel()
  if sa.indices.shape != (sa.thicknesses.shape[0], wl.size):
    raise ValueError(
      f"solve_structure: provider grid mismatch: solver arrays "
      f"{sa.indices.shape} vs {sa.thicknesses.shape[0]} layers x {wl.size} wavelengths."
    )
  _assert_provider_grids(source, wl)
  if sa.thicknesses.shape[0] >= 2:
    if sa.thicknesses[0] != 0.0 or sa.thicknesses[-1] != 0.0:
      warnings.warn(
        "solve_structure: first/last thickness is not 0 (ambient/substrate "
        "convention); the engine treats row 0/last as half-spaces.",
        stacklevel=2,
      )
  sm = ScatterMatrix(
    sa.indices,
    sa.thicknesses,
    wavelengths=wl,
    angles=angles,
    incoherent_flags=np.asarray(sa.incoherent_flags, dtype=np.int32),
    roughness_types=np.asarray(sa.rough_types, dtype=np.int32),
    roughness_values=np.asarray(sa.rough_vals, dtype=np.float64),
    **solver_opts,
  )
  if request is None:
    return sm.reflectance_transmittance()
  return sm.compute(request)
