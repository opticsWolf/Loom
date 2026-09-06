# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Configuration module for Loom: material libraries, layer stacks, and architect serialisation.

Supports YAML and JSON formats using Pydantic models.
"""

from .io import load_yaml, save_yaml, load_json, save_json
from .models import (
    MaterialDefinition,
    LayerConfig,
    GroupConfig,
    ArchitectState,
    StructureState,
)
from .builders import material_provider_from_library, layer_from_config, group_from_config, structure_from_config, architect_from_config, pipeline_from_config
from .models import BlockConfig, NamedStructureConfig
from .program import (
    PROGRAM_SCHEMA_VERSION,
    LoadedProgram,
    load_architect,
    load_document,
    load_groups,
    load_materials,
    load_named_structures,
    load_program,
    load_structure,
)
from .loader import (
    load_material_library,
    save_material_library,
    save_architect,
    load_architect,
    save_structure,
    load_structure,
)

__all__ = [
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "MaterialDefinition",
    "LayerConfig",
    "GroupConfig",
    "ArchitectState",
    "StructureState",
    "material_provider_from_library",
    "layer_from_config",
    "group_from_config",
    "structure_from_config",
    "load_material_library",
    "save_material_library",
    "save_architect",
    "load_architect",
    "save_structure",
    "load_structure",
]