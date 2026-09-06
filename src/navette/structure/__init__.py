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
  # Thin over the native solve: expansion + provider snapshot stay
  # Python-side (providers port in R2); grid check, half-space warning
  # and the solve itself run in Rust via `_structure.solve_arrays_fn`.
  from navette._structure import solve_arrays_fn as _solve_arrays  # lazy: needs native
  from navette._smatrix import solver_rt_request as _rt_request  # lazy: needs native

  if isinstance(source, Navette_Structure):
    gate_validation(source.validate(), "solve_structure")
  sa = source.get_error_solver_inputs(rng=rng) if errors \
    else source.get_solver_inputs()
  wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64)).ravel()
  _assert_provider_grids(source, wl)
  angles_arr = np.ascontiguousarray(np.atleast_1d(np.asarray(angles, dtype=np.float64))).ravel()
  known = {"coherence_mode", "angles_in_radians"}
  unknown = set(solver_opts) - known
  if unknown:
    raise TypeError(f"solve_structure: unknown solver options {sorted(unknown)}.")
  mask = _rt_request("u") if request is None else int(request)
  out, warns = _solve_arrays(
    sa, wl, angles_arr, mask,
    radians=bool(solver_opts.get("angles_in_radians", False)),
    coherence_mode=int(solver_opts.get("coherence_mode", 0)),
  )
  for w in warns:
    warnings.warn(f"solve_structure: {w}", stacklevel=2)
  if angles_arr.size != 1:
    return dict(out)
  return {k: v[0] for k, v in out.items()}
