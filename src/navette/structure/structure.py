# -*- coding: utf-8 -*-
"""Ordered thin-film stack: layers, groups and the material provider.

A :class:`Navette_Structure` validates the stack, flattens it into
:class:`SolverArrays` for the native engine, and serializes via
``get_state``/``from_state``. Behaves like a read sequence of layers
(``len()``, indexing, iteration, ``+`` concatenation).

Units: all lengths are nanometres (thicknesses, interface widths,
roughness sigma). State files store unitless numbers under this
convention — files authored when roughness was recorded in Angstrom
read 10x too small; there is no auto-detection, convert old files.
"""
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np

from .types import OptMask, RoughnessType, SolverArrays
from .materials import DictMaterialProvider, MaterialProvider
from .models import Group, Layer
from .expander import _DEFAULT_GROUP, _LayerExpander

class Navette_Structure:
    """Ordered stack of :class:`Layer` with groups and a material provider."""
    def __init__(
        self,
        layer_list: Optional[List[Layer]] = None,
        group_dict: Optional[Dict[str, Group]] = None,
        materials: Optional[Union[MaterialProvider, Dict[str, Any]]] = None,
    ):
        self.layer_list: List[Layer] = layer_list or []
        self.group_dict: Dict[str, Group] = group_dict or {}

        if materials is None:
            self._materials: Optional[MaterialProvider] = None
        elif isinstance(materials, dict):
            self._materials = DictMaterialProvider(materials)
        else:
            self._materials = materials

        self.simple_layer_list: List[List[Any]] = []

    @property
    def materials(self) -> Optional[MaterialProvider]:
        """Material provider (dicts are auto-wrapped on assignment)."""
        return self._materials
    @materials.setter
    def materials(self, value: Any) -> None:
        self._materials = DictMaterialProvider(value) if isinstance(value, dict) else value

    @property
    def active_material_dict(self) -> Optional[MaterialProvider]:
        """Alias of :attr:`materials` (legacy name)."""
        return self.materials
    @active_material_dict.setter
    def active_material_dict(self, value: Any) -> None: self.materials = value

    def validate(self) -> List[str]:
        """Check thicknesses, roughness and material coverage; returns error strings."""
        errors: List[str] = []
        if not self.layer_list:
            errors.append("Structure contains no layers.")
            return errors

        for i, layer in enumerate(self.layer_list):
            if layer.thickness < 0:
                errors.append(f"Layer {i} ({layer.material}): Negative thickness {layer.thickness} nm.")
            if layer.roughness < 0:
                errors.append(f"Layer {i} ({layer.material}): Negative roughness {layer.roughness} nm.")
            try:
                RoughnessType(int(layer.rough_type))
            except (ValueError, TypeError):
                errors.append(f"Layer {i} ({layer.material}): Unknown rough_type {layer.rough_type!r}.")
            if layer.interface and layer.interface_thickness >= layer.thickness:
                errors.append(f"Layer {i} ({layer.material}): Interface thickness ({layer.interface_thickness}) >= layer thickness ({layer.thickness}).")
            if self._materials and not self._materials.contains(layer.material):
                errors.append(f"Layer {i}: Material '{layer.material}' not found in material provider.")

        seen_groups = set()
        governed = set()
        for layer in self.layer_list:
            group = self.group_dict.get(layer.material)
            governed.add(layer.material)
            if group is not None and id(group) not in seen_groups:
                seen_groups.add(id(group))
                errors.extend(group.validate())
        for name in self.group_dict:
            if name not in governed:
                errors.append(f"Group '{name}' governs no layer material (lookup is by material name; silent _DEFAULT_GROUP applies).")

        if self._materials and not any("not found in material provider" in e for e in errors):
            try:
                sa = self._expand_arrays()
            except Exception as exc:
                errors.append(f"Nominal expansion failed: {exc}")
                return errors
            if not np.all(np.isfinite(sa.thicknesses)) or not np.all(np.isfinite(sa.indices)):
                errors.append("Nominal expansion produced NaN/inf (check group factors).")
            if np.any(sa.indices.real < 0.0):
                errors.append("Nominal expansion produced n < 0 (check provider data / group n_factor).")
            if np.any(sa.indices.imag < 0.0):
                errors.append("Nominal expansion produced k < 0 (check provider data / group k_factor).")
            interior = sa.thicknesses[1:-1] if sa.thicknesses.shape[0] > 2 else np.empty(0)
            if interior.size and np.any(interior <= 0.0):
                errors.append("Nominal expansion produced interior zero-thickness rows "
                              "(group factors floored a film away; ambient/substrate may be 0).")
        return errors

    def _expand_arrays(self, *, apply_errors: bool = False, rng: Optional[np.random.Generator] = None) -> SolverArrays:
        """Unvalidated expansion core (validation lives in the callers)."""
        if not self.layer_list: raise ValueError("Structure is empty.")
        if self._materials is None: raise ValueError("No material provider set.")
        return _LayerExpander.expand(((layer, False) for layer in self.layer_list), self._materials, self.group_dict, apply_errors=apply_errors, rng=rng)

    def get_solver_inputs(self) -> SolverArrays:
        """Flatten the stack to engine arrays (nominal values, no errors)."""
        issues = self.validate()
        if issues:
            raise ValueError("Navette_Structure invalid:\n" + "\n".join(issues))
        return self._expand_arrays()

    def get_error_solver_inputs(self, rng: Optional[np.random.Generator] = None) -> SolverArrays:
        """Flatten the stack with group fabrication errors drawn (see Group)."""
        issues = self.validate()
        if issues:
            raise ValueError("Navette_Structure invalid:\n" + "\n".join(issues))
        return self._expand_arrays(apply_errors=True, rng=rng)

    def generate_simple_layer_list(self) -> List[List[Any]]:
        """Legacy [thickness, index, coherent, roughness, rough_type] rows."""
        sa = self.get_solver_inputs()
        self.simple_layer_list = [
            [sa.thicknesses[i], sa.indices[i], not sa.incoherent_flags[i], sa.rough_vals[i], sa.rough_types[i]]
            for i in range(sa.thicknesses.shape[0])
        ]
        return self.simple_layer_list

    def get_state(self) -> Dict[str, Any]:
        """Serialize layers, groups and materials to a plain dict."""
        return {
            "layers": [layer.get_state() for layer in self.layer_list],
            "groups": {name: group.get_state() for name, group in self.group_dict.items()},
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any], materials: Optional[Union[MaterialProvider, Dict[str, Any]]] = None) -> "Navette_Structure":
        """Rebuild a structure from :meth:`get_state` output."""
        layers = [Layer.from_state(ls) for ls in state.get("layers", [])]
        groups = {name: Group.from_state(gs) for name, gs in state.get("groups", {}).items()}
        return cls(layer_list=layers, group_dict=groups, materials=materials)

    def clone(self) -> "Navette_Structure":
        """Deep copy (layers, groups and provider state)."""
        return Navette_Structure(
            layer_list=[layer.clone() for layer in self.layer_list],
            group_dict={name: group.clone() for name, group in self.group_dict.items()},
            materials=self._materials,
        )

    def __len__(self) -> int: return len(self.layer_list)
    def __getitem__(self, index: int) -> Layer: return self.layer_list[index]
    def __iter__(self) -> Iterator[Layer]: return iter(self.layer_list)
    def __bool__(self) -> bool: return len(self.layer_list) > 0

    def total_physical_thickness(self) -> float: return sum(layer.thickness for layer in self.layer_list)
    def get_optimization_parameters(self) -> List[Layer]:
        """Layers eligible for optimization (flag + group THICKNESS slot)."""
        out: List[Layer] = []
        for layer in self.layer_list:
            if not layer.optimize:
                continue
            group = self.group_dict.get(layer.material)
            if group is not None and len(group.optimization_mask) == len(OptMask) \
                    and not group.optimization_mask[OptMask.THICKNESS]:
                continue
            out.append(layer)
        return out

    def set_optimization_mask(self, group_name: str, mask: List[int]) -> None:
        """Write path for a group's optimization mask (binary, 7 slots)."""
        group = self.group_dict.get(group_name)
        if group is None:
            raise KeyError(f"set_optimization_mask: unknown group '{group_name}'.")
        if len(mask) != len(OptMask) or any(v not in (0, 1) for v in mask):
            raise ValueError(f"set_optimization_mask: mask must be {len(OptMask)} binary entries (see OptMask).")
        group.optimization_mask = list(mask)
    
    def replace_material(self, old_name: str, new_name: str) -> int:
        count = 0
        for layer in self.layer_list:
            if layer.material == old_name:
                layer.material = new_name
                count += 1
        return count

    def insert_layer(self, index: int, layer: Layer) -> None: self.layer_list.insert(index, layer)
    def remove_layer(self, index: int) -> Layer: return self.layer_list.pop(index)
    def replace_layer(self, index: int, new_layer: Layer) -> None: self.layer_list[index] = new_layer

    def prune_thin_layers(self, min_thickness: float = 0.001) -> int:
        before = len(self.layer_list)
        self.layer_list = [l for l in self.layer_list if l.thickness >= min_thickness]
        return before - len(self.layer_list)

    def total_sub_layers(self) -> int:
        """Solver row count: exact via nominal expansion when materials
        are set, else the structural approximation (interface ~= +1)."""
        if not self.layer_list:
            return 0
        if self._materials is not None:
            try:
                return len(self._expand_arrays().thicknesses)
            except Exception:
                pass
        total = 0
        for i, layer in enumerate(self.layer_list):
            total += layer.sub_layer_count if (layer.inhomogen and layer.sub_layer_count > 1) else 1
            if layer.interface and i > 0: total += 1
        return total

    def find_layers_by_material(self, material_name: str) -> List[int]:
        return [i for i, layer in enumerate(self.layer_list) if layer.material == material_name]

    def count_material(self, material_name: str) -> int:
        return sum(1 for layer in self.layer_list if layer.material == material_name)

    def apply_to_all_layers(self, func: callable) -> None:
        for layer in self.layer_list: func(layer)

    def __add__(self, other: "Navette_Structure") -> "Navette_Structure":
        new = self.clone()
        new.layer_list.extend(other.clone().layer_list)
        for name, group in other.group_dict.items():
            if name in new.group_dict:
                if new.group_dict[name].get_state() != group.get_state():
                    raise ValueError(f"Group '{name}' defined differently. Cannot merge.")
            else:
                new.group_dict[name] = group
        return new

    def get_group_for_material(self, material_name: str) -> Group:
        return self.group_dict.get(material_name, _DEFAULT_GROUP)

    def __contains__(self, material_name: str) -> bool:
        return any(layer.material == material_name for layer in self.layer_list)

    def __repr__(self) -> str:
        return f"Navette_Structure(layers={len(self.layer_list)}, groups={len(self.group_dict)})"