# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

from pathlib import Path
from typing import Literal, Union, List, Dict, Any
import numpy as np

from loom_structure import (
    MaterialProvider,
    MaterialObjectProvider,
    Loom_Structure,
    Loom_Architect,
    Layer,
    Group,
)
from .models import MaterialDefinition, LayerConfig, GroupConfig, ArchitectState, StructureState
from .builders import material_from_config, material_provider_from_library, layer_from_config, group_from_config
from .io import load_yaml, save_yaml, load_json, save_json

def load_material_library(
    file_path: Union[str, Path],
    wavelength: np.ndarray,
    use_code_map: bool = True,
    fmt: Literal["yaml", "json"] = "yaml",
) -> MaterialObjectProvider:
    """
    Load a material library from a YAML or JSON file and return a MaterialObjectProvider.
    The file should contain a list of MaterialDefinition under the key "materials".
    """
    loader = load_yaml if fmt == "yaml" else load_json
    data = loader(file_path)
    materials_data = data.get("materials", [])
    definitions = [MaterialDefinition.model_validate(item) for item in materials_data]
    return material_provider_from_library(definitions, wavelength, use_code_map)

def save_material_library(
    provider: MaterialProvider,
    file_path: Union[str, Path],
    fmt: Literal["yaml", "json"] = "yaml",
) -> None:
    """
    Export a MaterialProvider to a YAML/JSON material library.
    This works only if the provider is a MaterialObjectProvider (or has a _dict attribute).
    """
    if not isinstance(provider, MaterialObjectProvider):
        raise TypeError("save_material_library currently only supports MaterialObjectProvider")
    # Access the internal material dict
    mat_dict = getattr(provider, "_dict", None)
    if mat_dict is None:
        raise AttributeError("Provider does not expose material dictionary")

    library = []
    for name, mat in mat_dict.items():
        # Try to retrieve the material definition
        # For simplicity, we only export basic parameters; for TableMaterial we also need n_data.
        # This is a minimal export; a full round-trip may require more metadata.
        # We'll implement a basic version here.
        params = mat.get_params()
        model_type = mat.__class__.__name__
        if model_type == "Konstant":
            params_model = {"n": params["n"], "k": params.get("k", 0.0)}
        elif model_type == "TableMaterial":
            # For TableMaterial, we need to export n_data and k_data.
            # This is more involved – we'll skip for now or raise.
            raise NotImplementedError("Export of TableMaterial to config not yet implemented")
        elif model_type in ("Cauchy", "CauchyUrbach", "Sellmeier", "SellmeierUrbach"):
            params_model = params
        else:
            raise ValueError(f"Unsupported material type for export: {model_type}")

        definition = {
            "name": name,
            "code": name,  # assume code equals name, could be improved
            "model": model_type,
            "params": params_model,
        }
        library.append(definition)

    out_data = {"materials": library}
    saver = save_yaml if fmt == "yaml" else save_json
    saver(out_data, file_path)

def save_structure(
    structure: Loom_Structure,
    file_path: Union[str, Path],
    fmt: Literal["yaml", "json"] = "yaml",
) -> None:
    """Save a Loom_Structure state to a file."""
    state = structure.get_state()
    saver = save_yaml if fmt == "yaml" else save_json
    saver(state, file_path)

def load_structure(
    file_path: Union[str, Path],
    material_provider: MaterialProvider,
    fmt: Literal["yaml", "json"] = "yaml",
) -> Loom_Structure:
    """Load a Loom_Structure from a state file."""
    loader = load_yaml if fmt == "yaml" else load_json
    state = loader(file_path)
    return Loom_Structure.from_state(state, materials=material_provider)

def save_architect(
    architect: Loom_Architect,
    file_path: Union[str, Path],
    fmt: Literal["yaml", "json"] = "yaml",
) -> None:
    """Save a Loom_Architect state to a file."""
    state = architect.get_state()
    saver = save_yaml if fmt == "yaml" else save_json
    saver(state, file_path)

def load_architect(
    file_path: Union[str, Path],
    material_provider: MaterialProvider,
    fmt: Literal["yaml", "json"] = "yaml",
) -> Loom_Architect:
    """Load a Loom_Architect from a state file."""
    loader = load_yaml if fmt == "yaml" else load_json
    state = loader(file_path)
    return Loom_Architect.from_state(state, materials=material_provider)