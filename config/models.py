# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

from typing import Literal, Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np

# -------------------------------------------------------------------------
# Material parameter models
# -------------------------------------------------------------------------
class KonstantParams(BaseModel):
    n: float = Field(gt=0)
    k: float = Field(default=0.0, ge=0)

class CauchyParams(BaseModel):
    A: float
    B: float
    C: float

class CauchyUrbachParams(BaseModel):
    A: float
    B: float
    C: float
    alpha0: float = Field(gt=0)
    Eu: float = Field(gt=0)
    lambda_g: float = Field(gt=0)

class SellmeierParams(BaseModel):
    B1: float = Field(gt=0)
    C1: float = Field(gt=0)
    B2: float = Field(gt=0)
    C2: float = Field(gt=0)
    B3: float = Field(default=0.0, ge=0)
    C3: float = Field(default=0.0, ge=0)

class SellmeierUrbachParams(SellmeierParams):
    alpha0: float = Field(gt=0)
    Eu: float = Field(gt=0)
    lambda_g: float = Field(gt=0)

# Tabulated data (JSON-friendly lists)
class TabulatedData(BaseModel):
    wavelengths: List[float]
    values: List[float]

    @field_validator("wavelengths", "values")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("must not be empty")
        return v

class TableMaterialParams(BaseModel):
    n_factor: float = Field(default=1.0, ge=0)
    k_factor: float = Field(default=1.0, ge=0)
    interpolation_type_n: Literal["linear", "cubicspline", "pchip", "akima"] = "linear"
    interpolation_type_k: Literal["linear", "cubicspline", "pchip", "akima"] = "linear"

# Discriminated union for all material definitions
class MaterialDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    code: Optional[str] = None
    model: Literal[
        "Konstant",
        "TableMaterial",
        "Cauchy",
        "CauchyUrbach",
        "Sellmeier",
        "SellmeierUrbach",
    ]
    params: Union[
        KonstantParams,
        CauchyParams,
        CauchyUrbachParams,
        SellmeierParams,
        SellmeierUrbachParams,
        TableMaterialParams,
    ]
    n_data: Optional[TabulatedData] = None
    k_data: Optional[TabulatedData] = None

    @field_validator("params", mode="before")
    @classmethod
    def coerce_params(cls, v, info):
        model_name = info.data.get("model")
        if model_name == "Konstant":
            return KonstantParams.model_validate(v)
        elif model_name == "Cauchy":
            return CauchyParams.model_validate(v)
        elif model_name == "CauchyUrbach":
            return CauchyUrbachParams.model_validate(v)
        elif model_name == "Sellmeier":
            return SellmeierParams.model_validate(v)
        elif model_name == "SellmeierUrbach":
            return SellmeierUrbachParams.model_validate(v)
        elif model_name == "TableMaterial":
            return TableMaterialParams.model_validate(v)
        raise ValueError(f"Unknown model: {model_name}")

    @field_validator("n_data", "k_data")
    @classmethod
    def check_table_data(cls, v, info):
        model_name = info.data.get("model")
        if model_name == "TableMaterial" and v is None and "n_data" in info.field_name:
            raise ValueError("TableMaterial requires n_data")
        return v

# -------------------------------------------------------------------------
# Layer model
# -------------------------------------------------------------------------
class LayerConfig(BaseModel):
    material_code: str
    thickness_nm: float = Field(gt=0)
    coherent: bool = True
    roughness_angstrom: float = Field(default=0.0, ge=0)
    rough_type: int = Field(default=0, ge=0, le=3)
    inhomogen: bool = False
    inh_delta: float = Field(default=0.1, ge=0, le=1)
    interface: bool = False
    interface_thickness_nm: float = Field(default=0.0, ge=0)
    optimize: bool = True
    needle: bool = True
    layer_type: int = Field(default=1, ge=0)

# -------------------------------------------------------------------------
# Group model (error parameters)
# -------------------------------------------------------------------------
class ErrorParams(BaseModel):
    abs_mean_delta_g: float = 0.0
    abs_std_dev: float = 0.01
    rel_mean_delta_g: float = 0.0
    rel_std_dev: float = 1.0
    abs_mean_delta_h: float = 0.0
    abs_variance: float = 0.01
    rel_mean_delta_h: float = 0.0
    rel_variance: float = 1.0

class GroupConfig(BaseModel):
    name: str
    thick_factor: float = 1.0
    thick_summand: float = 0.0
    n_factor: float = 1.0
    k_factor: float = 0.0
    inh_delta_summand: float = 0.0
    roughness_summand: float = 0.0
    interface_summand: float = 0.0
    error_mask: List[int] = Field(default_factory=lambda: [0]*6)
    optimization_mask: List[int] = Field(default_factory=lambda: [0]*7)
    thickness_error_type: int = 0
    n_error_type: int = 0
    k_error_type: int = 0
    inh_delta_error_type: int = 0
    roughness_error_type: int = 0
    interface_error_type: int = 0
    thickness_error_params: ErrorParams = Field(default_factory=ErrorParams)
    inh_delta_error_params: ErrorParams = Field(default_factory=ErrorParams)
    roughness_error_params: ErrorParams = Field(default_factory=ErrorParams)
    interface_error_params: ErrorParams = Field(default_factory=ErrorParams)
    n_error_params: ErrorParams = Field(default_factory=ErrorParams)
    k_error_params: ErrorParams = Field(default_factory=ErrorParams)

# -------------------------------------------------------------------------
# Structure and Architect states (for serialisation)
# -------------------------------------------------------------------------
class StructureState(BaseModel):
    layers: List[Dict[str, Any]]   # from Layer.get_state()
    groups: Dict[str, Dict[str, Any]]  # from Group.get_state()

class ArchitectState(BaseModel):
    structures: List[StructureState]
    blocks: List[Dict[str, Any]]   # structure_ref, inverted, repeat_count, label