# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

from pathlib import Path
from typing import Literal, Union, List, Dict, Any
import numpy as np

from navette.structure import (
    MaterialProvider,
    MaterialObjectProvider,
    Navette_Structure,
    Navette_Architect,
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
        # Specs round-trip directly; anything else is legacy.
        model = getattr(mat, "model", None)
        params = getattr(mat, "params", None)
        if model is None or params is None:
            raise TypeError(
                f"save_material_library needs MaterialSpec values, got {type(mat).__name__}"
            )
        params_model = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in params.items()
            if k not in ("n_data", "k_data")
        }
        definition: Dict[str, Any] = {
            "name": name,
            "code": name,  # assume code equals name, could be improved
            "model": model,
            "params": params_model,
        }
        if "n_data" in params and params["n_data"] is not None:
            gw, nv = params["n_data"]
            definition["n_data"] = {
                "wavelengths": np.asarray(gw).tolist(),
                "values": np.asarray(nv).tolist(),
            }
        if params.get("k_data") is not None:
            gw, kv = params["k_data"]
            definition["k_data"] = {
                "wavelengths": np.asarray(gw).tolist(),
                "values": np.asarray(kv).tolist(),
            }
        library.append(definition)

    out_data = {"materials": library}
    saver = save_yaml if fmt == "yaml" else save_json
    saver(out_data, file_path)

def save_structure(
    structure: Navette_Structure,
    file_path: Union[str, Path],
    fmt: Literal["yaml", "json"] = "yaml",
) -> None:
    """Save a Navette_Structure state to a file."""
    state = structure.get_state()
    saver = save_yaml if fmt == "yaml" else save_json
    saver(state, file_path)

def load_structure(
    file_path: Union[str, Path],
    material_provider: MaterialProvider,
    fmt: Literal["yaml", "json"] = "yaml",
) -> Navette_Structure:
    """Load a Navette_Structure from a state file."""
    loader = load_yaml if fmt == "yaml" else load_json
    state = loader(file_path)
    return Navette_Structure.from_state(state, materials=material_provider)

def save_architect(
    architect: Navette_Architect,
    file_path: Union[str, Path],
    fmt: Literal["yaml", "json"] = "yaml",
) -> None:
    """Save a Navette_Architect state to a file."""
    state = architect.get_state()
    saver = save_yaml if fmt == "yaml" else save_json
    saver(state, file_path)

def load_architect(
    file_path: Union[str, Path],
    material_provider: MaterialProvider,
    fmt: Literal["yaml", "json"] = "yaml",
) -> Navette_Architect:
    """Load a Navette_Architect from a state file."""
    loader = load_yaml if fmt == "yaml" else load_json
    state = loader(file_path)
    return Navette_Architect.from_state(state, materials=material_provider)