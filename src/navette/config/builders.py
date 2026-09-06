# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from typing import List, Dict, Any, Mapping, Optional
from navette.structure import (
    MaterialProvider,
    MaterialObjectProvider,
    Layer,
    Group,
    Navette_Structure,
    Navette_Architect,
)
from navette.materials import MaterialSpec
from .models import (
    MaterialDefinition, LayerConfig, GroupConfig, BlockConfig,
    NamedStructureConfig,
)

def material_from_config(
    cfg: MaterialDefinition,
    wavelength: np.ndarray,
) -> MaterialSpec:
    """Create a MaterialSpec from a MaterialDefinition (evaluated natively)."""
    model = cfg.model
    if model == "TableMaterial":
        model = "Table"
    if model not in (
        "Konstant", "Table", "Cauchy", "CauchyUrbach", "Sellmeier",
        "SellmeierUrbach",
    ):
        raise ValueError(f"Unsupported material model: {cfg.model}")
    params = cfg.params.model_dump()
    if model == "Table":
        if cfg.n_data is None:
            raise ValueError("TableMaterial requires n_data")
        params["n_data"] = (
            np.array(cfg.n_data.wavelengths), np.array(cfg.n_data.values)
        )
        if cfg.k_data:
            params["k_data"] = (
                np.array(cfg.k_data.wavelengths), np.array(cfg.k_data.values)
            )
    wavelength = np.asarray(wavelength, dtype=np.float64)  # validated here
    return MaterialSpec(model=model, params=params)


def material_provider_from_library(
    library: List[MaterialDefinition],
    wavelength: np.ndarray,
    use_code_map: bool = True,
) -> MaterialObjectProvider:
    """
    Build a MaterialObjectProvider from a list of MaterialDefinition.
    If use_code_map is True, the keys in the provider are the material codes.
    Otherwise, the keys are the material names.
    """
    mat_dict = {}
    for mat_def in library:
        key = mat_def.code if (use_code_map and mat_def.code) else mat_def.name
        mat_dict[key] = material_from_config(mat_def, wavelength)
    return MaterialObjectProvider(mat_dict, wavelength)

def layer_from_config(cfg: LayerConfig, material_provider: MaterialProvider) -> Layer:
    """Create a Layer from a LayerConfig and validate that the material code exists."""
    if not material_provider.contains(cfg.material_code):
        raise KeyError(f"Material code '{cfg.material_code}' not found in provider")
    return Layer(
        thickness=cfg.thickness_nm,
        material_name=cfg.material_code,
        coherent=cfg.coherent,
        roughness=cfg.roughness_nm,
        rough_type=cfg.rough_type,
        inhomogen=cfg.inhomogen,
        inh_delta=cfg.inh_delta,
        interface=cfg.interface,
        interface_thickness=cfg.interface_thickness_nm,
        optimize=cfg.optimize,
        needle=cfg.needle,
        layer_type=cfg.layer_type,
    )

def group_from_config(cfg: GroupConfig) -> Group:
    """Create a Group from a GroupConfig."""
    group = Group(
        group_name=cfg.name,
        thick_factor=cfg.thick_factor,
        thick_summand=cfg.thick_summand,
        n_factor=cfg.n_factor,
        k_factor=cfg.k_factor,
        inh_delta_summand=cfg.inh_delta_summand,
        roughness_summand=cfg.roughness_summand,
        interface_summand=cfg.interface_summand,
    )
    # Set error masks and parameters (bound setters; params replaced whole).
    group.error_mask = cfg.error_mask.copy()
    group.optimization_mask = cfg.optimization_mask.copy()
    group.set_error_type("thickness", cfg.thickness_error_type)
    group.set_error_type("n", cfg.n_error_type)
    group.set_error_type("k", cfg.k_error_type)
    group.set_error_type("inh_delta", cfg.inh_delta_error_type)
    group.set_error_type("roughness", cfg.roughness_error_type)
    group.set_error_type("interface", cfg.interface_error_type)

    def copy_params(src, channel):
        group.set_error_params(channel, src.model_dump())

    copy_params(cfg.thickness_error_params, "thickness")
    copy_params(cfg.inh_delta_error_params, "inh_delta")
    copy_params(cfg.roughness_error_params, "roughness")
    copy_params(cfg.interface_error_params, "interface")
    copy_params(cfg.n_error_params, "n")
    copy_params(cfg.k_error_params, "k")

    return group


def structure_from_config(
    layer_cfgs: List[LayerConfig],
    group_cfgs: List[GroupConfig],
    materials: Any,
) -> Navette_Structure:
    """Build a Navette_Structure from typed configs (the live config path).

    Groups are keyed by config name, which by convention equals the
    governed material name (the expander looks groups up by material;
    `validate()` flags keys that govern nothing).
    """
    layers = [layer_from_config(c, materials) for c in layer_cfgs]
    groups = {c.name: group_from_config(c) for c in group_cfgs}
    # Navette_Structure auto-wraps plain dicts into DictMaterialProvider.
    return Navette_Structure(layer_list=layers, group_dict=groups, materials=materials)


def pipeline_from_config(
    structure: NamedStructureConfig,
    library: List[MaterialDefinition],
    wavelengths,
    *,
    contrast: Optional[Mapping[str, str]] = None,
    film_flags: Optional[Dict[str, Any]] = None,
    per_film_flags: Optional[Mapping[str, Dict[str, Any]]] = None,
    ambient_name: str = "air",
    substrate_name: str = "sub",
):
    """Native ``(DesignStack, contrast_map)`` from typed configs.

    ``structure.layers`` carry ``layer_type``: 0 = ambient row, 2 =
    substrate row, 1 = film. At most one ambient / one substrate row;
    absent rows fall back to the driver constants (n=1.0 air, n=1.52
    substrate). Films are keyed by material code — codes must be unique
    (they key the nk table; repeats raise ``ValueError``) and present
    in ``library`` (else ``KeyError``). Per-layer flags flow from each
    ``LayerConfig`` (driver key names); ``per_film_flags`` overrides by
    code (int index also accepted) on top. ``contrast`` maps host code
    → seed code. Groups ride the bound-``Group`` fast path, error
    config intact. Returns exactly what ``stack_from_layers`` returns.
    """
    from navette.synthesis.pipeline import stack_from_layers
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    specs = {}
    for mat_def in library:
        key = mat_def.code if mat_def.code else mat_def.name
        specs[key] = material_from_config(mat_def, wl)

    ambient_rows = [c for c in structure.layers if c.layer_type == 0]
    substrate_rows = [c for c in structure.layers if c.layer_type == 2]
    film_rows = [c for c in structure.layers if c.layer_type == 1]
    if len(ambient_rows) > 1:
        raise ValueError("pipeline_from_config: at most one ambient (layer_type=0) row.")
    if len(substrate_rows) > 1:
        raise ValueError("pipeline_from_config: at most one substrate (layer_type=2) row.")

    def _resolve(code):
        if code not in specs:
            raise KeyError(f"Material code '{code}' not found in library")
        return specs[code]

    ambient = ((_resolve(ambient_rows[0].material_code), ambient_name)
               if ambient_rows else (1.0, ambient_name))
    substrate = ((_resolve(substrate_rows[0].material_code), substrate_name)
                 if substrate_rows else (1.52, substrate_name))

    layers, names, auto_flags = [], [], {}
    for c in film_rows:
        if c.material_code in names:
            raise ValueError(
                f"pipeline_from_config: duplicate film material code '{c.material_code}' "
                "(film names key the nk table; split repeats into distinct codes).")
        names.append(c.material_code)
        layers.append((_resolve(c.material_code), c.thickness_nm))
        auto_flags[c.material_code] = {
            "coherent": c.coherent, "roughness": c.roughness_nm,
            "rough_type": c.rough_type, "inhomogen": c.inhomogen,
            "inh_delta": c.inh_delta, "interface": c.interface,
            "interface_thickness": c.interface_thickness_nm,
            "optimize": c.optimize, "needle": c.needle,
        }
    merged_flags = {}
    for i, nm in enumerate(names):
        merged_flags[nm] = dict(auto_flags[nm])
    for key, override in (per_film_flags or {}).items():
        nm = names[key] if isinstance(key, int) and 0 <= key < len(names) else str(key)
        if nm not in merged_flags:
            raise KeyError(f"per_film_flags: unknown film '{key}'.")
        merged_flags[nm].update(dict(override))
    cmap_in = {str(h): _resolve(s) for h, s in (contrast or {}).items()}
    groups = {c.name: group_from_config(c) for c in structure.groups}
    return stack_from_layers(layers, wl, cmap_in, ambient=ambient,
                             substrate=substrate, names=names,
                             film_flags=film_flags, groups=groups,
                             per_film_flags=merged_flags)


def architect_from_config(
    structures: Mapping[str, Navette_Structure],
    blocks: List[BlockConfig],
    materials: Any = None,
) -> Navette_Architect:
    """Build a Navette_Architect: blocks reference structures by label.

    Missing labels raise KeyError naming the block index and label.
    """
    from navette.structure import Navette_Architect
    arch = Navette_Architect(materials=materials)
    for i, b in enumerate(blocks):
        if b.structure not in structures:
            raise KeyError(
                f"architect block {i}: unknown structure label '{b.structure}'."
            )
        arch.add_structure(structures[b.structure], inverted=b.inverted,
                           repeat=b.repeat_count, label=b.label, kind=b.kind)
    return arch