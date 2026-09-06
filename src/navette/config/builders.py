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
from .models import MaterialDefinition, LayerConfig, GroupConfig, BlockConfig

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