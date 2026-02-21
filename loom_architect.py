# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: loom_architect.py — High-level governing module for arranging
Loom_Structure objects into a contiguous simulation stack.

Rework notes (v2):
  1.  All expansion logic delegated to _LayerExpander from loom_structure.
      No more copy-pasted flatten code — the Architect builds a layer
      *iterator* that handles inversion, repetition, and cross-structure
      boundaries, then hands it to the shared expander.
  2.  StructureBlock replaces StructureNode (avoids collision with the
      Qt node-graph concept of "Node").
  3.  Serialisable via get_state / from_state for node-graph persistence.
  4.  Accepts MaterialProvider protocol (auto-wraps legacy dicts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

from loom_structure import (
    Loom_Structure,
    Layer,
    Group,
    MaterialProvider,
    DictMaterialProvider,
    MaterialObjectProvider,
    SolverArrays,
    _LayerExpander,
    FLOAT_TYPE,
    COMPLEX_TYPE,
    INT_TYPE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# StructureBlock — lightweight chain entry
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class StructureBlock:
    """
    A positioned reference to a Loom_Structure in the Architect's chain.

    Attributes
    ----------
    structure : Loom_Structure
        The layer-stack definition (shared reference — edits propagate).
    inverted : bool
        If True, layers are traversed last → first.
    repeat_count : int
        Number of consecutive repetitions of this block.
    label : str
        Optional human-readable tag (for node-graph display).
    """
    structure: Loom_Structure
    inverted: bool = False
    repeat_count: int = 1
    label: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# TF_Architect
# ═══════════════════════════════════════════════════════════════════════════════
class Loom_Architect:
    """
    Governing class that strings together multiple Loom_Structure objects.

    Capabilities:
      1. Reuse of Loom_Structure instances (changes map to all references).
      2. Inversion of structures (optical path reversal).
      3. Centralized solver input generation across structure boundaries.
      4. Global-to-Local mapping for Needle optimization.
      5. Serialisable for node-graph persistence (get_state / from_state).
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

    # -- material provider -------------------------------------------------
    @property
    def materials(self) -> Optional[MaterialProvider]:
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
        return self._materials

    @active_material_dict.setter
    def active_material_dict(self, value: Any) -> None:
        self.materials = value

    # -- block management --------------------------------------------------
    @property
    def blocks(self) -> List[StructureBlock]:
        return self._blocks

    def add_structure(
        self,
        structure: Loom_Structure,
        inverted: bool = False,
        repeat: int = 1,
        label: str = "",
    ) -> None:
        """
        Append a Loom_Structure to the simulation stack.

        The structure's material provider is overwritten with the Architect's
        shared provider so all structures resolve from the same data source.
        """
        if self._materials is not None:
            structure.materials = self._materials

        self._blocks.append(
            StructureBlock(structure, inverted, repeat, label)
        )

    def remove_structure(self, index: int) -> StructureBlock:
        """Remove and return the block at *index*."""
        return self._blocks.pop(index)

    def clear(self) -> None:
        """Remove all blocks."""
        self._blocks.clear()

    @property
    def unique_structures(self) -> List[Loom_Structure]:
        """Deduplicated list of referenced structures (by identity)."""
        seen: set = set()
        result: List[Loom_Structure] = []
        for block in self._blocks:
            sid = id(block.structure)
            if sid not in seen:
                seen.add(sid)
                result.append(block.structure)
        return result

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
        Yield (layer, is_inverted_block) for every logical layer across all
        blocks, respecting inversion and repetition.

        The first layer yielded is the ambient; the last is the substrate.
        """
        for block in self._blocks:
            layers = block.structure.layer_list
            n = len(layers)
            if n == 0:
                continue

            for _ in range(block.repeat_count):
                if block.inverted:
                    for i in range(n - 1, -1, -1):
                        yield layers[i], True
                else:
                    for i in range(n):
                        yield layers[i], False

    def _iter_layers_flat(
        self, *, skip_first: bool = False
    ) -> Iterator[Layer]:
        """
        Simple flat iterator over layers (no inversion metadata).

        Parameters
        ----------
        skip_first : bool
            If True, skip the very first layer (ambient already handled).
        """
        it = self._iter_layers()
        if skip_first:
            next(it, None)  # consume the ambient
        for layer, _ in it:
            yield layer

    # -- solver array generation -------------------------------------------
    def _merged_group_dict(self) -> Dict[str, Group]:
        """Merge group dicts from all unique structures."""
        merged: Dict[str, Group] = {}
        for struct in self.unique_structures:
            merged.update(struct.group_dict)
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
            raise ValueError("TF_Architect is empty.")
        if self._materials is None:
            raise ValueError("No material provider set.")

        # Determine if any block is inverted (for inhomogen gradient flip)
        # We need per-layer inversion info, so we use a small adapter
        # that wraps _iter_layers and passes invert_inhomogen per-layer.
        #
        # Since _LayerExpander.expand() takes a flat iterator and a single
        # invert_inhomogen flag, we need a different approach for mixed
        # inversion.  Solution: build the full flat layer list with an
        # expander that respects per-block inversion by calling expand()
        # per-block and concatenating.  However, that breaks cross-structure
        # interface mixing (prev_eff_nk must be continuous).
        #
        # Best approach: use expand() once with full iterator, and handle
        # the inhomogen flip via a wrapper layer.  OR: we extend
        # _LayerExpander to accept (layer, invert_flag) tuples.
        #
        # For cleanliness, use the extended approach:
        return _ArchitectExpander.expand(
            self._iter_layers(),
            self._materials,
            self._merged_group_dict(),
            apply_errors=False,
        )

    def get_error_solver_inputs(self) -> SolverArrays:
        """Generate solver arrays WITH stochastic manufacturing errors."""
        if not self._blocks:
            raise ValueError("TF_Architect is empty.")
        if self._materials is None:
            raise ValueError("No material provider set.")

        return _ArchitectExpander.expand(
            self._iter_layers(),
            self._materials,
            self._merged_group_dict(),
            apply_errors=True,
        )

    # -- global ↔ local index mapping (for Needle optimization) ------------
    def map_global_index_to_layer(
        self, global_idx: int
    ) -> Tuple[Loom_Structure, int]:
        """
        Map a global simulation layer index to the specific Loom_Structure
        and its internal layer index.

        Returns
        -------
        (Loom_Structure, local_layer_index)

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
    def insert_layer_at_global(
        self, global_idx: int, new_layer: Layer
    ) -> None:
        """Insert a new layer at the specified global index."""
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
    ) -> "Loom_Architect":
        """Reconstruct from a serialised dict."""
        arch = cls(materials=materials)

        # Rebuild structures
        structs: List[Loom_Structure] = [
            Loom_Structure.from_state(ss, materials=materials)
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

    def __repr__(self) -> str:
        total = self.get_global_layer_count()
        return (
            f"TF_Architect(blocks={len(self._blocks)}, "
            f"unique_structs={len(self.unique_structures)}, "
            f"total_layers={total})"
        )

    def __len__(self) -> int:
        return len(self._blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# _ArchitectExpander — variant of _LayerExpander that accepts (layer, inv) tuples
# ═══════════════════════════════════════════════════════════════════════════════
class _ArchitectExpander:
    """
    Expansion engine for TF_Architect that handles per-layer inversion flags.

    This is almost identical to _LayerExpander.expand() but accepts
    an iterator of (Layer, is_inverted) tuples so inhomogeneous gradient
    direction can be flipped per-block.
    """

    @staticmethod
    def expand(
        layers_with_inv: Iterator[Tuple[Layer, bool]],
        materials: MaterialProvider,
        group_dict: Dict[str, Group],
        *,
        apply_errors: bool = False,
    ) -> SolverArrays:
        from optical_models.ema_models import looyenga_eps

        col_thick: List[float] = []
        col_nk: List[Any] = []
        col_coh: List[bool] = []
        col_r_val: List[float] = []
        col_r_type: List[int] = []

        _DEFAULT_GROUP = Group("_default_")
        get_group = group_dict.get
        prev_eff_nk: Optional[np.ndarray] = None

        for layer, is_inv in layers_with_inv:
            mat_name = layer.material
            group = get_group(mat_name, _DEFAULT_GROUP)

            # --- Base nk + systematic factors ---
            base_nk = materials.get_nk(mat_name)

            if group.n_factor != 1.0 or group.k_factor != 1.0:
                layer_nk = base_nk * group.nk_factor
            else:
                layer_nk = base_nk

            layer_thickness = (
                layer.thickness * group.thick_factor + group.thick_summand
            )

            current_roughness = layer.roughness

            # --- Stochastic errors ---
            if apply_errors:
                if group.error_mask[0]:
                    layer_thickness = group.thickness_error(layer_thickness)

                if group.error_mask[1] or group.error_mask[2]:
                    n_part = layer_nk.real
                    k_part = layer_nk.imag
                    if group.error_mask[1]:
                        n_part = Group._apply_error(
                            n_part, group.n_error_type, group.n_error_params
                        )
                        n_part = np.maximum(0.0, n_part)
                    if group.error_mask[2]:
                        k_part = Group._apply_error(
                            k_part, group.k_error_type, group.k_error_params
                        )
                    layer_nk = n_part + 1j * k_part

                if group.error_mask[3]:
                    current_roughness = group.sr_roughness_error(
                        current_roughness, layer_thickness
                    )

            if layer_thickness < 0.0:
                layer_thickness = 0.0

            # --- A. Interface ---
            if layer.interface and prev_eff_nk is not None:
                t_interface = layer.interface_thickness

                if apply_errors and group.error_mask[5]:
                    t_interface = group.interface_error(
                        t_interface, layer.thickness
                    )
                if t_interface > layer_thickness:
                    t_interface = layer_thickness
                layer_thickness -= t_interface

                interface_nk = looyenga_eps(layer_nk, prev_eff_nk, 0.5)

                col_thick.append(t_interface)
                col_nk.append(interface_nk)
                col_coh.append(True)
                col_r_val.append(0.0)
                col_r_type.append(0)

            # --- B. Inhomogeneity ---
            if layer._inhomogen and layer.sub_layer_count > 1:
                sub_div = layer.sub_layer_count
                current_delta = (
                    (layer._inh_delta + group.inh_delta_summand) * 0.5
                )

                if apply_errors and group.error_mask[4]:
                    current_delta = group.inh_delta_error(current_delta)

                factors = np.linspace(
                    1.0 - current_delta, 1.0 + current_delta, sub_div
                )
                if is_inv:
                    factors = factors[::-1]

                step_t = layer_thickness / sub_div

                for ix, f in enumerate(factors):
                    col_thick.append(step_t)
                    col_nk.append(layer_nk * f)
                    col_coh.append(layer.coherent)
                    if ix == 0:
                        col_r_val.append(current_roughness)
                        col_r_type.append(layer.rough_type)
                    else:
                        col_r_val.append(0.0)
                        col_r_type.append(0)

            # --- C. Standard ---
            else:
                col_thick.append(layer_thickness)
                col_nk.append(layer_nk)
                col_coh.append(layer.coherent)
                col_r_val.append(current_roughness)
                col_r_type.append(layer.rough_type)

            prev_eff_nk = layer_nk

        return SolverArrays(
            indices=np.vstack(col_nk).astype(COMPLEX_TYPE),
            thicknesses=np.array(col_thick, dtype=FLOAT_TYPE),
            incoherent_flags=np.array(
                [not c for c in col_coh], dtype=np.bool_
            ),
            rough_types=np.array(col_r_type, dtype=INT_TYPE),
            rough_vals=np.array(col_r_val, dtype=FLOAT_TYPE),
        )
