# -*- coding: utf-8 -*-
"""
Navette: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: navette_architect.py — High-level governing module for arranging
Navette_Structure objects into a contiguous simulation stack.

Rework notes (v2):
  1.  All expansion logic delegated to _LayerExpander from navette_structure.
      No more copy-pasted flatten code — the Architect builds a layer
      *iterator* of (Layer, is_inverted) tuples that handles inversion,
      repetition, and cross-structure boundaries, then hands it to the
      shared expander.
  2.  StructureBlock replaces StructureNode (avoids collision with the
      Qt node-graph concept of "Node").
  3.  Serialisable via get_state / from_state for node-graph persistence.
  4.  Accepts MaterialProvider protocol (auto-wraps legacy dicts).
  5.  CAUTION: Structures are stored by reference. Mutating a Navette_Structure
      that is used in multiple blocks will affect all those blocks. Use
      `clone()` to create independent copies.

Phase 4 improvements:
  - Removed __len__ (replaced by block_count property) to avoid confusion.
  - Module hygiene: __all__ defined, unused imports cleaned.

Phase 5 improvements:
  - get_error_solver_inputs accepts an optional np.random.Generator (rng) and
    forwards it to _LayerExpander for reproducible Monte-Carlo runs.
  - Dropped the now-unused `field` import.  (numpy is retained: it is
    referenced by the rng type annotation.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

from .structure import Navette_Structure
from .models import Layer, Group
from .materials import MaterialProvider, DictMaterialProvider
from .types import RoughnessType, SolverArrays
from .expander import _LayerExpander

__all__ = [
    "StructureBlock",
    "Navette_Architect",
]


# ═══════════════════════════════════════════════════════════════════════════════
# StructureBlock — lightweight chain entry
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class StructureBlock:
    """
    A positioned reference to a Navette_Structure in the Architect's chain.

    Attributes
    ----------
    structure : Navette_Structure
        The layer-stack definition (shared reference — edits propagate).
    inverted : bool
        If True, layers are traversed last → first.
    repeat_count : int
        Number of consecutive repetitions of this block.
    label : str
        Optional human-readable tag (for node-graph display).
    """
    structure: Navette_Structure
    inverted: bool = False
    repeat_count: int = 1
    label: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Navette_Architect
# ═══════════════════════════════════════════════════════════════════════════════
class Navette_Architect:
    """
    Governing class that strings together multiple Navette_Structure objects.

    Capabilities:
      1. Reuse of Navette_Structure instances (changes map to all references).
      2. Inversion of structures (optical path reversal).
      3. Centralized solver input generation across structure boundaries.
      4. Global-to-Local mapping for Needle optimization.
      5. Serialisable for node-graph persistence (get_state / from_state).

    CAUTION: Structures are stored by reference. If you modify a structure
    (e.g., change thicknesses, materials, groups), ALL blocks that reference
    it will see the changes. Use `clone_structure()` or `Navette_Structure.clone()`
    to create independent copies when needed.

    Group merging: if the same group name appears in two different structures
    with differing parameters, get_solver_inputs() raises ValueError rather
    than silently letting one win.  Use unique group names or keep them
    consistent across structures.
    """

    def __init__(
        self,
        materials: Optional[Union[MaterialProvider, Dict[str, Any]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        materials : MaterialProvider | dict, optional
            Shared material data source.  A plain dict is auto-wrapped in
            DictMaterialProvider for backward compatibility.
        """
        self._blocks: List[StructureBlock] = []

        if materials is None:
            self._materials: Optional[MaterialProvider] = None
        elif isinstance(materials, dict):
            self._materials = DictMaterialProvider(materials)
        else:
            self._materials = materials

    def __len__(self) -> int:
        """Return the total number of logical layers (sum over blocks)."""
        return self.get_global_layer_count()

    def __getitem__(self, index: int) -> StructureBlock:
        return self._blocks[index]
    
    def __iter__(self) -> Iterator[StructureBlock]:
        return iter(self._blocks)

    def __bool__(self) -> bool:
        return not self.is_empty

    def __contains__(self, block: StructureBlock) -> bool:
        return block in self._blocks

    def __repr__(self) -> str:
        total = self.get_global_layer_count()
        return (
            f"Navette_Architect(blocks={len(self._blocks)}, "
            f"unique_structs={len(self.unique_structures)}, "
            f"total_layers={total})"
        )

    def copy(self) -> Navette_Architect:
        """Create a deep copy (all structures cloned)."""
        new_arch = Navette_Architect(materials=self._materials)
        for block in self._blocks:
            new_arch.add_structure(
                block.structure.clone(),
                inverted=block.inverted,
                repeat=block.repeat_count,
                label=block.label,
            )
        return new_arch
    
    def validate(self) -> List[str]:
        """Return list of warnings/errors (empty if all good)."""
        issues = []
        # Check groups
        try:
            self._merged_group_dict()
        except ValueError as e:
            issues.append(str(e))
        # Check materials exist for all layers
        if self._materials:
            for layer, _ in self._iter_layers():
                if not self._materials.has_material(layer.material):
                    issues.append(f"Material '{layer.material}' not found")
        return issues

    @property
    def is_empty(self) -> bool:
        """True when the chain holds no structure blocks."""
        return len(self._blocks) == 0

    # -- material provider -------------------------------------------------
    @property
    def materials(self) -> Optional[MaterialProvider]:
        """Shared provider, propagated to every block on assignment."""
        return self._materials

    @materials.setter
    def materials(self, value: Any) -> None:
        if isinstance(value, dict):
            self._materials = DictMaterialProvider(value)
        else:
            self._materials = value
        # Propagate to all structures
        for block in self._blocks:
            block.structure.materials = self._materials

    # Backward compat
    @property
    def active_material_dict(self) -> Optional[MaterialProvider]:
        """Alias of :attr:`materials` (legacy name)."""
        return self._materials

    @active_material_dict.setter
    def active_material_dict(self, value: Any) -> None:
        self.materials = value

    def replace_material(self, old_name: str, new_name: str) -> int:
        """Replace all occurrences of old material name with new. Returns count."""
        # NOTE: walks unique structures directly — `_iter_layers` yields
        # clones for inverted blocks, which must never be mutated.
        count = 0
        for struct in self.unique_structures:
            for layer in struct.layer_list:
                if layer.material == old_name:
                    layer.material = new_name
                    count += 1
        return count

    # -- block management --------------------------------------------------
    @property
    def blocks(self) -> List[StructureBlock]:
        """The structure blocks in chain order."""
        return self._blocks

    @property
    def block_count(self) -> int:
        """Number of StructureBlocks in the architect chain."""
        return len(self._blocks)

    def index(self, block: StructureBlock) -> int:
        """Position of `block` in the chain (ValueError when absent)."""
        return self._blocks.index(block)

    def get_block_index_by_label(self, label: str) -> List[int]:
        """Chain positions of all blocks carrying `label`."""
        return [i for i, blk in enumerate(self._blocks) if blk.label == label]

    def add_structure(
        self,
        structure: Navette_Structure,
        inverted: bool = False,
        repeat: int = 1,
        label: str = "",
    ) -> None:
        """
        Append a Navette_Structure to the simulation stack.

        WARNING: The structure is stored by reference. If you modify this
        structure later, ALL blocks that reference it will see the changes.
        To avoid this, call `structure.clone()` before passing it to
        multiple blocks.

        The structure's material provider is overwritten with the Architect's
        shared provider so all structures resolve from the same data source.
        """
        if repeat < 1:
            raise ValueError("repeat_count must be >= 1")
        if self._materials is not None:
            structure.materials = self._materials

        self._blocks.append(
            StructureBlock(structure, inverted, repeat, label)
        )

    def insert_structure(
        self,
        index: int,
        structure: Navette_Structure,
        inverted: bool = False,
        repeat: int = 1,
        label: str = "",
    ) -> None:
        """Insert a structure block (optionally inverted/repeated/labelled)."""
        if repeat < 1:
            raise ValueError("repeat_count must be >= 1")
        if self._materials is not None:
            structure.materials = self._materials
        self._blocks.insert(
            index,
            StructureBlock(structure, inverted, repeat, label)
        )

    def replace_structure(self, block_index: int, new_structure: Navette_Structure) -> None:
        """Replace the structure at block_index while preserving inversion/repeat/label."""
        block = self._blocks[block_index]
        if self._materials is not None:
            new_structure.materials = self._materials
        self._blocks[block_index] = StructureBlock(
            structure=new_structure,
            inverted=block.inverted,
            repeat_count=block.repeat_count,
            label=block.label,
        )

    def move_block(self, from_index: int, to_index: int) -> None:
        """Move a block from one position to another."""
        block = self._blocks.pop(from_index)
        self._blocks.insert(to_index, block)

    def remove_structure(self, index: int) -> StructureBlock:
        """Remove and return the block at *index*."""
        return self._blocks.pop(index)

    def clear(self) -> None:
        """Remove all blocks."""
        self._blocks.clear()

    @property
    def unique_structures(self) -> List[Navette_Structure]:
        """Deduplicated list of referenced structures (by identity)."""
        seen: Set[int] = set()
        result: List[Navette_Structure] = []
        for block in self._blocks:
            sid = id(block.structure)
            if sid not in seen:
                seen.add(sid)
                result.append(block.structure)
        return result

    # -- clone structure (break aliasing) ---------------------------------
    def clone_structure(self, index: int) -> None:
        """
        Clone the structure at block `index` and replace the block's reference.
        This breaks aliasing with other blocks that used the same original structure.
        """
        if not 0 <= index < len(self._blocks):
            raise IndexError("Block index out of range")
        block = self._blocks[index]
        cloned_struct = block.structure.clone()
        # Ensure the cloned structure uses the same material provider
        cloned_struct.materials = self._materials
        self._blocks[index] = StructureBlock(
            structure=cloned_struct,
            inverted=block.inverted,
            repeat_count=block.repeat_count,
            label=block.label,
        )

    # -- layer counting ----------------------------------------------------
    def get_global_layer_count(self) -> int:
        """Total number of *logical* layers (before sub-layer expansion)."""
        return sum(
            len(b.structure.layer_list) * b.repeat_count
            for b in self._blocks
        )

    # -- layer iteration (the key abstraction) -----------------------------
    def _iter_layers(self) -> Iterator[Tuple[Layer, bool]]:
        """
        Yield (layer, is_inverted) for every logical layer across all
        blocks, respecting inversion and repetition.

        The first layer yielded is the ambient; the last is the substrate.
        This is exactly the contract _LayerExpander.expand consumes.

        Inversion mirrors plane properties: bulk/design state (material,
        thickness, coherence, grading, optimize/needle/layer_type) stays
        with the layer, but the interface slice and roughness describe
        the boundary with the forward predecessor — i.e. the incoming-
        light side. Inverted blocks therefore yield clones with the
        plane flags shifted one step toward the incident side (clone[i-1]
        carries layer[i]'s interface/roughness; the first-yielded clone
        carries layer[0]'s flags, which the expander drops at the
        incident edge exactly like the forward first layer). The carve
        follows the material: each donor clone is pre-shrunk by its
        slice width while the carrier is pre-grown by the same amount,
        so the expander's trailing-side carve reproduces the exact
        mirror bulk split. Repetition edges are exact too: the incident
        edge of the first repetition and the exit edge of the last get
        private copies without the boundary carve (interior repetition
        boundaries share the base clones). Sub-layer counts are
        preserved verbatim (direct `_thickness` writes — the property
        setter would re-refine from the adjusted thickness and change
        the count). Forward behavior is untouched (originals, no clones).
        Shared structures are never mutated.

        NOTE: `_iter_layers` yields clones for inverted blocks — callers
        that mutate layers must not use it (see `replace_material`).
        """
        for block in self._blocks:
            layers = block.structure.layer_list
            n = len(layers)
            if n == 0:
                continue

            if block.inverted:
                # Base clones with full boundary treatment (valid for
                # interior repetition boundaries on both sides).
                clones = [layer.clone() for layer in layers]
                for i in range(1, n):
                    donor = layers[i]
                    t = donor.interface_thickness if donor.interface else 0.0
                    clones[i - 1].interface = donor.interface
                    clones[i - 1].interface_thickness = donor.interface_thickness
                    clones[i - 1].roughness = donor.roughness
                    clones[i - 1].rough_type = donor.rough_type
                    clones[i]._thickness = max(0.0, clones[i]._thickness - t)
                    clones[i - 1]._thickness = clones[i - 1]._thickness + t
                first = layers[0]
                t0 = first.interface_thickness if first.interface else 0.0
                clones[n - 1].interface = first.interface
                clones[n - 1].interface_thickness = first.interface_thickness
                clones[n - 1].roughness = first.roughness
                clones[n - 1].rough_type = first.rough_type
                clones[n - 1]._thickness = clones[n - 1]._thickness + t0
                clones[0]._thickness = clones[0]._thickness - t0
            for rep in range(block.repeat_count):
                if block.inverted:
                    use = clones
                    if t0 > 0.0 and (rep == 0 or rep == block.repeat_count - 1):
                        # Edge repetitions get private copies: the incident
                        # edge hosts no slice, the exit edge is carved by none.
                        use = [c.clone() for c in clones]
                        if rep == 0:
                            use[n - 1]._thickness = use[n - 1]._thickness - t0
                        if rep == block.repeat_count - 1:
                            use[0]._thickness = use[0]._thickness + t0
                    for i in range(n - 1, -1, -1):
                        yield use[i], True
                else:
                    for i in range(n):
                        yield layers[i], False

    # -- solver array generation -------------------------------------------
    def _merged_group_dict(self) -> Dict[str, Group]:
        """
        Merge group dicts from all unique structures.

        Raises
        ------
        ValueError
            If the same group name is defined with different parameters
            in two structures.
        """
        merged: Dict[str, Group] = {}
        for struct in self.unique_structures:
            for name, group in struct.group_dict.items():
                if name in merged:
                    # Compare group states instead of object identity
                    if merged[name].get_state() != group.get_state():
                        raise ValueError(
                            f"Group name '{name}' defined differently in two structures. "
                            f"Cannot merge automatically. Use unique group names or "
                            f"ensure consistency."
                        )
                    # If identical state, keep the existing one (no need to replace)
                else:
                    merged[name] = group
        return merged

    def get_solver_inputs(self) -> SolverArrays:
        """
        Generate flattened Structure-of-Arrays for the solver across ALL
        structures.  Handles cross-structure interfaces, inversion, and
        inhomogeneous expansion.

        Returns
        -------
        SolverArrays
        """
        if not self._blocks:
            raise ValueError("Navette_Architect is empty.")
        if self._materials is None:
            raise ValueError("No material provider set.")

        return _LayerExpander.expand(
            self._iter_layers(),   # yields (layer, bool) tuples
            self._materials,
            self._merged_group_dict(),
            apply_errors=False,
        )

    def get_error_solver_inputs(
        self, rng: Optional[np.random.Generator] = None
    ) -> SolverArrays:
        """
        Generate solver arrays WITH stochastic manufacturing errors.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Generator for reproducible Monte-Carlo runs.  If None, the legacy
            global np.random state is used.
        """
        if not self._blocks:
            raise ValueError("Navette_Architect is empty.")
        if self._materials is None:
            raise ValueError("No material provider set.")

        return _LayerExpander.expand(
            self._iter_layers(),
            self._materials,
            self._merged_group_dict(),
            apply_errors=True,
            rng=rng,
        )

    # -- global ↔ local index mapping (for Needle optimization) ------------
    def map_global_index_to_layer(
        self, global_idx: int
    ) -> Tuple[Navette_Structure, int]:
        """
        Map a global simulation layer index to the specific Navette_Structure
        and its internal layer index.

        Returns
        -------
        (Navette_Structure, local_layer_index)

        Raises
        ------
        IndexError
            If global_idx is out of bounds.
        """
        current = 0

        for block in self._blocks:
            struct = block.structure
            n = len(struct.layer_list)

            for _ in range(block.repeat_count):
                if current <= global_idx < current + n:
                    local_offset = global_idx - current
                    if block.inverted:
                        return struct, (n - 1) - local_offset
                    return struct, local_offset
                current += n

        raise IndexError(
            f"Global index {global_idx} out of bounds "
            f"(total logical layers: {current})"
        )


    # -- layer manipulation at global indices ------------------------------
    def get_layer_at_global(self, global_idx: int) -> Layer:
        """Layer at a chain-wide index (spans block boundaries)."""
        struct, local = self.map_global_index_to_layer(global_idx)
        return struct.layer_list[local]

    def insert_layer_at_global(
        self, global_idx: int, new_layer: Layer
    ) -> None:
        """
        Insert a new layer at the specified global index.

        NOTE: This mutates the underlying Navette_Structure. If that structure
        is referenced elsewhere, the change will appear everywhere.
        """
        struct, local = self.map_global_index_to_layer(global_idx)
        struct.layer_list.insert(local, new_layer)

    def split_layer_at_global(
        self, global_idx: int, split_ratio: float = 0.5
    ) -> None:
        """Split the layer at global_idx into two layers of the same material."""
        struct, local = self.map_global_index_to_layer(global_idx)
        original = struct.layer_list[local]

        l1 = original.clone()
        l2 = original.clone()
        l1.thickness = original.thickness * split_ratio
        l2.thickness = original.thickness * (1.0 - split_ratio)

        struct.layer_list[local] = l1
        struct.layer_list.insert(local + 1, l2)

    def duplicate_layer_at_global(self, global_idx: int) -> None:
        """Duplicate the layer at the global index."""
        struct, local = self.map_global_index_to_layer(global_idx)
        struct.layer_list.insert(local, struct.layer_list[local].clone())

    def remove_layer_at_global(self, global_idx: int) -> None:
        """Remove the layer at the global index."""
        struct, local = self.map_global_index_to_layer(global_idx)
        del struct.layer_list[local]

    def prune_thin_layers(self, min_thickness: float = 0.001) -> int:
        """Remove layers thinner than *min_thickness* from ALL structures."""
        removed = 0
        for struct in self.unique_structures:
            before = len(struct.layer_list)
            struct.layer_list = [
                l for l in struct.layer_list if l.thickness >= min_thickness
            ]
            removed += before - len(struct.layer_list)
        return removed

    def get_optimization_parameters(self) -> List[Layer]:
        """
        Return a UNIQUE list of layers eligible for optimisation.
        Even if a structure is referenced 5 times, its layers appear once.
        """
        params: List[Layer] = []
        for struct in self.unique_structures:
            for layer in struct.layer_list:
                if layer.optimize:
                    params.append(layer)
        return params

    def total_sub_layers(self) -> int:
        """Number of physical slices after inhomogeneous subdivision (no error application)."""
        # Quick approximation: iterate layers and sum sub_layer_count
        total = 0
        prev_exists = False
        for layer, _ in self._iter_layers():
            if layer._inhomogen and layer.sub_layer_count > 1:
                total += layer.sub_layer_count
            else:
                total += 1
            # Interfaces add one extra slice per interface (except first layer)
            # But careful: interface adds only if previous layer exists
            # This is a simplified version; a full expander call would be more accurate
            # but may be expensive.
            # For exact count, use:
            # arrays = self.get_solver_inputs()
            # return len(arrays.thicknesses)
            # But that's heavy. Consider caching or a dedicated counter.
            # Count interface layer if present (adds one extra slice)
            if layer.interface and prev_exists:
                total += 1
            prev_exists = True
        return total

    def get_total_physical_thickness(self) -> float:
        """Sum of all film thicknesses [nm] across the chain."""
        return sum(layer.thickness for layer, _ in self._iter_layers())

    # -- serialisation (node-graph persistence) ----------------------------
    def get_state(self) -> Dict[str, Any]:
        """Serialise the entire architect to a dict."""
        # Map structure id(obj) → index for reference tracking
        struct_map: Dict[int, int] = {}
        struct_states: List[Dict[str, Any]] = []
        for struct in self.unique_structures:
            struct_map[id(struct)] = len(struct_states)
            struct_states.append(struct.get_state())

        block_states = [
            {
                "structure_ref": struct_map[id(b.structure)],
                "inverted": b.inverted,
                "repeat_count": b.repeat_count,
                "label": b.label,
            }
            for b in self._blocks
        ]

        return {
            "structures": struct_states,
            "blocks": block_states,
        }

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        materials: Optional[Union[MaterialProvider, Dict[str, Any]]] = None,
    ) -> "Navette_Architect":
        """Reconstruct from a serialised dict."""
        arch = cls(materials=materials)

        # Rebuild structures
        structs: List[Navette_Structure] = [
            Navette_Structure.from_state(ss, materials=materials)
            for ss in state.get("structures", [])
        ]

        # Rebuild blocks with structure references
        for bs in state.get("blocks", []):
            ref = bs.get("structure_ref", 0)
            if 0 <= ref < len(structs):
                arch.add_structure(
                    structs[ref],
                    inverted=bs.get("inverted", False),
                    repeat=bs.get("repeat_count", 1),
                    label=bs.get("label", ""),
                )
        return arch
