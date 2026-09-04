# -*- coding: utf-8 -*-
"""Ordered thin-film stack: layers, groups and the material provider.

A :class:`Navette_Structure` validates the stack, flattens it into
:class:`SolverArrays` for the native engine, and serializes via
``get_state``/``from_state``. Behaves like a read sequence of layers
(``len()``, indexing, iteration, ``+`` concatenation).
"""
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np

from .types import SolverArrays
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
                errors.append(f"Layer {i} ({layer.material}): Negative roughness {layer.roughness} A.")
            if layer.interface and layer.interface_thickness >= layer.thickness:
                errors.append(f"Layer {i} ({layer.material}): Interface thickness ({layer.interface_thickness}) >= layer thickness ({layer.thickness}).")
            if self._materials and not self._materials.contains(layer.material):
                errors.append(f"Layer {i}: Material '{layer.material}' not found in material provider.")
        return errors

    def get_solver_inputs(self) -> SolverArrays:
        """Flatten the stack to engine arrays (nominal values, no errors)."""
        if not self.layer_list: raise ValueError("Structure is empty.")
        if self._materials is None: raise ValueError("No material provider set.")
        return _LayerExpander.expand(((layer, False) for layer in self.layer_list), self._materials, self.group_dict, apply_errors=False)

    def get_error_solver_inputs(self, rng: Optional[np.random.Generator] = None) -> SolverArrays:
        """Flatten the stack with group fabrication errors drawn (see Group)."""
        if not self.layer_list: raise ValueError("Structure is empty.")
        if self._materials is None: raise ValueError("No material provider set.")
        return _LayerExpander.expand(((layer, False) for layer in self.layer_list), self._materials, self.group_dict, apply_errors=True, rng=rng)

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
    def get_optimization_parameters(self) -> List[Layer]: return [layer for layer in self.layer_list if layer.optimize]
    
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