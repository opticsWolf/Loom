# -*- coding: utf-8 -*-
"""Positioned composition of structures over the native model.

:class:`Navette_Architect` wraps the bound ``Architect`` and tracks the
calling-side shells (so ``map_global_index_to_layer`` returns the exact
objects that were added, and provider propagation reaches them). Core
sharing/aliasing semantics live in Rust (`Rc`-shared handles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from navette._structure import Architect as _RsArchitect

__all__ = [
  "BlockKind",
  "StructureBlock",
  "Navette_Architect",
]

from .types import BlockKind


@dataclass
class StructureBlock:
  """Positioned reference view: shell + traversal flags + role.

  Views are rebuilt per access (:meth:`Navette_Architect.blocks`); the
  ``structure`` attribute is the tracked shell (identity-stable).
  """
  structure: Any
  inverted: bool = False
  repeat_count: int = 1
  label: str = ""
  kind: Union[BlockKind, int] = BlockKind.STACK


class Navette_Architect:
  """Positioned composition of structures with global-index addressing.

  Thin wrapper over the bound ``Architect``. Still dropped vs the legacy
  Python class: ``index``, ``get_block_index_by_label`` (no in-tree
  users — one-liners over :meth:`blocks` if a GUI needs them),
  ``insert/replace_structure``, ``move_block``, ``remove_structure``,
  ``clear`` (chain surgery needs core support),
  ``get_optimization_parameters`` (bound-only, unwrapped —
  ``optimization_entries`` covers reads), block-identity
  ``__contains__`` (murky across view rebuilds; use ``block in
  arch.blocks``).
  """

  def __init__(self, materials: Any = None) -> None:
    self._inner = _RsArchitect()
    self._shells: List[Any] = []
    self._view_cache: Optional[tuple] = None  # (blocks fingerprint, views)
    if materials is not None:
      self.materials = materials

  @classmethod
  def _from_native(cls, native_arch: Any, materials: Any = None,
                   shells: Optional[List[Any]] = None) -> "Navette_Architect":
    """Adopt a natively assembled architect (program restore path)."""
    obj = cls.__new__(cls)
    obj._inner = native_arch
    obj._shells = list(shells or [])
    obj._view_cache = None
    if materials is not None:
      # Same adoption guard as Navette_Structure._from_native.
      adopted = getattr(materials, "_native", None)
      if adopted is None or native_arch.materials is not adopted:
        obj._inner.materials = materials
    return obj

  # -- provider ----------------------------------------------------------
  @property
  def materials(self) -> Any:
    return self._inner.materials

  @materials.setter
  def materials(self, value: Any) -> None:
    from .materials import DictMaterialProvider
    if isinstance(value, dict):
      value = DictMaterialProvider(value)
    self._inner.materials = value  # warns per distinct shell provider
    for shell in self._shells:
      shell.materials = value

  # -- composition ---------------------------------------------------------
  def add_structure(self, structure: Any, inverted: bool = False,
                    repeat: int = 1, label: str = "",
                    kind: Union[BlockKind, int] = BlockKind.STACK) -> None:
    self._inner.add_structure(structure._inner, inverted=inverted,
                              repeat=repeat, label=label, kind=int(kind))
    if structure not in self._shells:
      self._shells.append(structure)
    if self._inner.materials is not None:
      structure.materials = self._inner.materials

  def clone_structure(self, index: int) -> None:
    self._inner.clone_structure(index)

  def __len__(self) -> int:
    return len(self._inner)

  @property
  def block_count(self) -> int:
    return len(self._inner)

  @property
  def blocks(self) -> List[StructureBlock]:
    fingerprint = self._inner.blocks_info()
    if self._view_cache is None or self._view_cache[0] != fingerprint:
      self._view_cache = (fingerprint, [
        StructureBlock(structure=self._shell_for_core(cid), inverted=inv,
                       repeat_count=rep, label=lab, kind=BlockKind(k))
        for cid, inv, rep, lab, k in fingerprint])
    return self._view_cache[1]

  def __getitem__(self, index: int) -> StructureBlock:
    return self.blocks[index]

  def __iter__(self):
    return iter(self.blocks)

  @property
  def is_empty(self) -> bool:
    return len(self._inner) == 0

  def _shell_for_core(self, core_id: int) -> Any:
    for shell in self._shells:
      if shell._inner.core_id() == core_id:
        return shell
    # New core (post-clone): adopt a fresh shell around it.
    from .structure import Navette_Structure
    for bound in self._inner.unique_structures():
      if bound.core_id() == core_id:
        shell = Navette_Structure.__new__(Navette_Structure)
        shell._inner = bound
        if self._inner.materials is not None:
          shell.materials = self._inner.materials
        self._shells.append(shell)
        return shell
    raise AssertionError("Navette_Architect: block core vanished.")

  @property
  def unique_structures(self) -> List[Any]:
    seen, out = set(), []
    for shell in self._shells:
      cid = shell._inner.core_id()
      if cid not in seen:
        seen.add(cid)
        out.append(shell)
    return out

  def get_global_layer_count(self) -> int:
    return self._inner.global_layer_count()

  # -- model API (delegated) ----------------------------------------------
  def validate(self) -> List[str]:
    return self._inner.validate()

  def get_solver_inputs(self) -> Any:
    return self._inner.solver_inputs()

  def get_error_solver_inputs(self, rng=None) -> Any:
    return self._inner.error_inputs(rng)

  def total_sub_layers(self) -> int:
    return self._inner.total_sub_layers()

  @staticmethod
  def _oob(exc: Exception, what: str):
    if "out of bounds" in str(exc):
      raise IndexError(f"{what}: {exc}") from exc
    raise exc

  def map_global_index_to_layer(self, global_idx: int):
    try:
      bi, local = self._inner.map_global_index_to_layer(global_idx)
    except ValueError as exc:
      self._oob(exc, "Global index")
    return self._shell_for_core(self._inner.block_core_id(bi)), local

  def map_solver_index_to_layer(self, solver_idx: int):
    try:
      bi, local = self._inner.map_solver_index_to_layer(solver_idx)
    except ValueError as exc:
      self._oob(exc, "Solver index")
    return self._shell_for_core(self._inner.block_core_id(bi)), local

  def get_layer_at_global(self, global_idx: int) -> Any:
    shell, local = self.map_global_index_to_layer(global_idx)
    return shell[local]

  def insert_layer_at_global(self, global_idx: int, new_layer: Any) -> None:
    self._inner.insert_layer_at_global(global_idx, new_layer)

  def split_layer_at_global(self, global_idx: int, split_ratio: float = 0.5) -> None:
    self._inner.split_layer_at_global(global_idx, split_ratio)

  def duplicate_layer_at_global(self, global_idx: int) -> None:
    self._inner.duplicate_layer_at_global(global_idx)

  def remove_layer_at_global(self, global_idx: int) -> None:
    self._inner.remove_layer_at_global(global_idx)

  def prune_thin_layers(self, min_thickness: float = 0.001) -> int:
    return self._inner.prune_thin_layers(min_thickness)

  def bake_films(self) -> int:
    return self._inner.bake_films()

  def bake_materials(self, wavelengths) -> Dict[str, str]:
    from .structure import _pour_back
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    mapping, target = self._inner.bake_materials(wl)
    _pour_back(mapping, target, self._inner.materials, wl)
    return mapping

  def set_optimization_mask(self, group_name: str, mask: List[int]) -> None:
    self._inner.set_optimization_mask(group_name, mask)

  # -- GUI conveniences (restored) --------------------------------------
  @property
  def active_material_dict(self) -> Any:
    """Alias of :attr:`materials` (legacy name)."""
    return self.materials

  @active_material_dict.setter
  def active_material_dict(self, value: Any) -> None:
    self.materials = value

  def replace_material(self, old_name: str, new_name: str) -> int:
    """Replace all occurrences of old material name with new. Returns count."""
    return sum(shell.replace_material(old_name, new_name)
               for shell in self.unique_structures)

  def get_total_physical_thickness(self) -> float:
    """Sum of all film thicknesses [nm] across the chain (× repeat)."""
    return sum(view.structure.total_physical_thickness() * view.repeat_count
               for view in self.blocks)

  def copy(self) -> "Navette_Architect":
    """Deep copy: all structures cloned, chain flags preserved.

    The carried provider is shared by reference (dicts are re-wrapped
    into an equal provider — same content, new object).
    """
    new = Navette_Architect()
    if self.materials is not None:
      new.materials = self.materials
    for view in self.blocks:
      new.add_structure(view.structure.clone(), inverted=view.inverted,
                        repeat=view.repeat_count, label=view.label,
                        kind=view.kind)
    return new

  def get_state(self) -> Dict[str, Any]:
    return self._inner.get_state()

  @classmethod
  def from_state(cls, state: Dict[str, Any], materials: Any = None) -> "Navette_Architect":
    obj = cls.__new__(cls)
    obj._inner = _RsArchitect.from_state(state)
    obj._shells = []
    obj._view_cache = None
    if materials is not None:
      obj.materials = materials
    # Adopt one shell per unique core (propagation-consistent).
    from .structure import Navette_Structure
    for bound in obj._inner.unique_structures():
      shell = Navette_Structure.__new__(Navette_Structure)
      shell._inner = bound
      if materials is not None:
        shell.materials = materials
      obj._shells.append(shell)
    # Re-link blocks order: replay shells per block core order.
    ordered: List[Any] = []
    for cid, _inv, _rep, _lab, _k in obj._inner.blocks_info():
      for shell in obj._shells:
        if shell._inner.core_id() == cid and shell not in ordered:
          ordered.append(shell)
          break
    obj._shells = ordered
    return obj
