# -*- coding: utf-8 -*-
"""Thin-film design stacks over the native model (provider plumbing).

:class:`Navette_Structure` wraps the bound ``Structure`` and adds the two
things that stay Python-side: the carried material provider (any
provider-like object — snapshotted at solve time) and ``bake_materials``
pour-back (new Table specs are registered into the carried Python
provider). Everything else — validation, expansion, states, film baking —
delegates to the core.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from navette._structure import Structure as _RsStructure


def gate_validation(issues: List[str], what: str) -> None:
  """Solve gate shared by structures and architects: re-emit advisory
  warnings via `warnings.warn` (flow continues), raise `ValueError` on
  errors. See `Navette_Structure.validate` for the severity contract."""
  import warnings
  from .types import is_warning
  for issue in issues:
    if is_warning(issue):
      warnings.warn(f"{what}: {issue}", stacklevel=3)
  errs = [i for i in issues if not is_warning(i)]
  if errs:
    raise ValueError(f"{what} invalid:\n" + "\n".join(errs))


def _pour_back(mapping: Dict[str, str], target: Any, carried: Any, wavelengths: np.ndarray) -> None:
  """Register baked Table specs from a bound target shelf into a Python provider."""
  from navette.materials import MaterialSpec
  if isinstance(carried, dict):
    shelf, gridless = carried, True
  else:
    shelf = getattr(carried, "_dict", None)
    gridless = False
  if shelf is None:
    raise ValueError(
      "bake_materials pour-back needs a dict-backed provider "
      "(dict/DictMaterialProvider/MaterialObjectProvider); "
      f"got {type(carried).__name__}."
    )
  for _old, new in mapping.items():
    payload = target.export_entry(new)
    if isinstance(payload, dict):
      shelf[new] = MaterialSpec(model=payload["model"], params=payload["params"])
    else:
      shelf[new] = payload
    if hasattr(carried, "invalidate"):
      try:
        carried.invalidate(new)
      except Exception:
        pass
  # Bare dicts stay gridless (bridge warns); Dict providers gain the grid.
  if not gridless and getattr(carried, "_wavelength", "sentinel") is None:
    carried._wavelength = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))


class Navette_Structure:
  """One design stack: ordered layers + material-keyed groups + provider.

  Thin wrapper over the bound ``Structure`` (first-class Rust model).
  ``materials`` accepts any provider-like object (provider, dict, None);
  it is snapshotted at solve time. Still dropped vs the legacy Python
  class: ``generate_simple_layer_list`` (legacy row adapter —
  ``get_solver_inputs()`` is the clean source), structure-level
  ``prune_thin_layers`` and ``get_optimization_parameters``,
  ``set_optimization_mask`` (bound-only, unwrapped — architect-level
  covers the live paths).
  """

  def __init__(self, layer_list=None, group_dict=None, materials=None) -> None:
    self._inner = _RsStructure(layer_list or [], group_dict or {})
    if materials is not None:
      self.materials = materials

  def __add__(self, other: "Navette_Structure") -> "Navette_Structure":
    if self._inner.materials is not None and other._inner.materials is not None \
            and self._inner.materials is not other._inner.materials:
      raise ValueError("Cannot merge structures with different material providers (same name could resolve differently).")
    new = self.clone()
    for layer in other._inner.layer_list:
      new._inner.append_layer(layer)
    for name, group in other._inner.group_dict.items():
      mine = new._inner.group_dict
      if name in mine:
        if mine[name].get_state() != group.get_state():
          raise ValueError(f"Group '{name}' defined differently. Cannot merge.")
      else:
        new._inner.insert_group(name, group)
    return new

  # -- provider ----------------------------------------------------------
  @property
  def materials(self) -> Any:
    return self._inner.materials

  @materials.setter
  def materials(self, value: Any) -> None:
    from .materials import DictMaterialProvider
    if isinstance(value, dict):
      value = DictMaterialProvider(value)
    self._inner.materials = value

  @property
  def group_dict(self) -> Dict[str, Any]:
    return self._inner.group_dict

  @property
  def layer_list(self) -> List[Any]:
    return self._inner.layer_list

  # -- sequence protocol (live clones) ------------------------------------
  def __len__(self) -> int:
    return len(self._inner)

  def __getitem__(self, index: int) -> Any:
    return self._inner[index]

  def __iter__(self):  # noqa: ANN204 (delegated iterator)
    return iter(self._inner)

  def __bool__(self) -> bool:
    return len(self) > 0

  # -- model API (delegated) ----------------------------------------------
  def validate(self) -> List[str]:
    return self._inner.validate()

  def get_solver_inputs(self) -> Any:
    return self._inner.solver_inputs()

  def get_error_solver_inputs(self, rng=None) -> Any:
    return self._inner.error_inputs(rng)

  def total_sub_layers(self) -> int:
    return self._inner.total_sub_layers()

  def bake_films(self) -> int:
    return self._inner.bake_films()

  def bake_materials(self, wavelengths) -> Dict[str, str]:
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    mapping, target = self._inner.bake_materials(wl)
    _pour_back(mapping, target, self._inner.materials, wl)
    return mapping

  def replace_material(self, old_name: str, new_name: str) -> int:
    return self._inner.replace_material(old_name, new_name)

  def get_group_for_material(self, material_name: str) -> Any:
    return self._inner.get_group_for_material(material_name)

  def get_state(self) -> Dict[str, Any]:
    return self._inner.get_state()

  @classmethod
  def from_state(cls, state: Dict[str, Any], materials: Any = None) -> "Navette_Structure":
    obj = cls.__new__(cls)
    obj._inner = _RsStructure.from_state(state, materials)
    return obj

  def clone(self) -> "Navette_Structure":
    obj = self.__class__.__new__(self.__class__)
    obj._inner = self._inner.clone()
    return obj

  # -- GUI conveniences (restored) --------------------------------------
  @property
  def active_material_dict(self) -> Any:
    """Alias of :attr:`materials` (legacy name)."""
    return self.materials

  @active_material_dict.setter
  def active_material_dict(self, value: Any) -> None:
    self.materials = value

  def find_layers_by_material(self, material_name: str) -> List[int]:
    return [i for i, layer in enumerate(self._inner.layer_list)
            if layer.material == material_name]

  def count_material(self, material_name: str) -> int:
    return len(self.find_layers_by_material(material_name))

  def apply_to_all_layers(self, func) -> None:
    """Call ``func(layer)`` on every layer, writing mutations back."""
    for i in range(len(self._inner)):
      layer = self._inner[i]
      func(layer)
      self._inner.replace_layer(i, layer)

  def insert_layer(self, index: int, layer: Any) -> None:
    self._inner.insert_layer(index, layer)

  def remove_layer(self, index: int) -> Any:
    return self._inner.remove_layer(index)

  def replace_layer(self, index: int, new_layer: Any) -> None:
    self._inner.replace_layer(index, new_layer)

  def total_physical_thickness(self) -> float:
    return sum(layer.thickness for layer in self._inner.layer_list)

  def __contains__(self, material_name: str) -> bool:
    return any(layer.material == material_name
               for layer in self._inner.layer_list)
