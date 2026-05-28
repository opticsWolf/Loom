# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from typing import List, Dict, Any, Optional
from loom_structure import (
    MaterialProvider,
    MaterialObjectProvider,
    Layer,
    Group,
    Loom_Structure,
    Loom_Architect,
)
from optical_models.basic import Konstant, TableMaterial
from optical_models.cauchy_sellmeier import (
    Cauchy,
    CauchyUrbach,
    Sellmeier,
    SellmeierUrbach,
)
from .models import MaterialDefinition, LayerConfig, GroupConfig

def material_from_config(
    cfg: MaterialDefinition,
    wavelength: np.ndarray,
) -> "Material":
    """Create a Material object from a MaterialDefinition."""
    if cfg.model == "Konstant":
        return Konstant(
            n=cfg.params.n,
            k=cfg.params.k,
            wavelength=wavelength,
        )
    elif cfg.model == "Cauchy":
        return Cauchy(
            A=cfg.params.A,
            B=cfg.params.B,
            C=cfg.params.C,
            wavelength=wavelength,
        )
    elif cfg.model == "CauchyUrbach":
        return CauchyUrbach(
            A=cfg.params.A,
            B=cfg.params.B,
            C=cfg.params.C,
            alpha0=cfg.params.alpha0,
            Eu=cfg.params.Eu,
            lambda_g=cfg.params.lambda_g,
            wavelength=wavelength,
        )
    elif cfg.model == "Sellmeier":
        p = cfg.params.model_dump()
        return Sellmeier(
            B1=p["B1"],
            C1=p["C1"],
            B2=p["B2"],
            C2=p["C2"],
            B3=p["B3"],
            C3=p["C3"],
            wavelength=wavelength,
        )
    elif cfg.model == "SellmeierUrbach":
        p = cfg.params.model_dump()
        return SellmeierUrbach(
            B1=p["B1"],
            C1=p["C1"],
            B2=p["B2"],
            C2=p["C2"],
            B3=p["B3"],
            C3=p["C3"],
            alpha0=p["alpha0"],
            Eu=p["Eu"],
            lambda_g=p["lambda_g"],
            wavelength=wavelength,
        )
    elif cfg.model == "TableMaterial":
        if cfg.n_data is None:
            raise ValueError("TableMaterial requires n_data")
        n_wl = np.array(cfg.n_data.wavelengths)
        n_vals = np.array(cfg.n_data.values)
        n_data = (n_wl, n_vals)
        k_data = None
        if cfg.k_data:
            k_wl = np.array(cfg.k_data.wavelengths)
            k_vals = np.array(cfg.k_data.values)
            k_data = (k_wl, k_vals)
        return TableMaterial(
            n_data=n_data,
            k_data=k_data,
            n_factor=cfg.params.n_factor,
            k_factor=cfg.params.k_factor,
            interpolation_type_n=cfg.params.interpolation_type_n,
            interpolation_type_k=cfg.params.interpolation_type_k,
            wavelength=wavelength,
        )
    else:
        raise ValueError(f"Unsupported material model: {cfg.model}")

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
        roughness=cfg.roughness_angstrom,
        rough_type=cfg.rough_type,
        inhomogen=cfg.inhomogen,
        inh_delta=cfg.inh_delta,
        interface=cfg.interface,
        interface_thickness=cfg.interface_thickness_nm,
        optimize=cfg.optimize,
        needle=cfg.needle,
        layer_typ=cfg.layer_type,
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
    # Set error masks and parameters
    group.error_mask = cfg.error_mask.copy()
    group.optimization_mask = cfg.optimization_mask.copy()
    group.thickness_error_type = cfg.thickness_error_type
    group.n_error_type = cfg.n_error_type
    group.k_error_type = cfg.k_error_type
    group.inh_delta_error_type = cfg.inh_delta_error_type
    group.roughness_error_type = cfg.roughness_error_type
    group.interface_error_type = cfg.interface_error_type

    def copy_params(src, dst):
        for key in src.model_dump().keys():
            dst[key] = getattr(src, key)

    copy_params(cfg.thickness_error_params, group.thickness_error_params)
    copy_params(cfg.inh_delta_error_params, group.inh_delta_error_params)
    copy_params(cfg.roughness_error_params, group.roughness_error_params)
    copy_params(cfg.interface_error_params, group.interface_error_params)
    copy_params(cfg.n_error_params, group.n_error_params)
    copy_params(cfg.k_error_params, group.k_error_params)

    return group