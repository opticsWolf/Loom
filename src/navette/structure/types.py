# -*- coding: utf-8 -*-
"""Shared dtypes, enums and solver-array containers for layer stacks.

All solver-facing arrays use fixed dtypes (``FLOAT_TYPE``/``COMPLEX_TYPE``/
``INT_TYPE``) so the Python stack model matches the native engine layouts
bit-for-bit. The ``*Mask`` enums index the per-layer flag/error vectors.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple
import numpy as np

# Standardised numeric types (match the native engine dtypes)
FLOAT_TYPE = np.float64
COMPLEX_TYPE = np.complex128
INT_TYPE = np.int32

class ErrorType(IntEnum):
    """Statistical law used when drawing fabrication errors."""
    GAUSSIAN = 0
    UNIFORM = 1
    COMBINED = 2

class RoughnessType(IntEnum):
    """Structure-level roughness switch (detail lives in smatrix.RoughnessType)."""
    NONE = 0
    SCALAR = 1

class ErrorMask(IntEnum):
    """Slots of the per-layer error vector (thickness, n/k, roughness, ...)."""
    THICKNESS = 0
    N_REAL = 1
    N_IMAG = 2
    ROUGHNESS = 3
    INH_DELTA = 4
    INTERFACE = 5

class LayerMask(IntEnum):
    """Slots of the per-layer status mask produced by :meth:`Layer.mask`."""
    ACTIVE = 0
    COHERENT = 1
    INHOMOGEN = 2
    ROUGHNESS = 3

@dataclass(frozen=True)
class InterpolationSettings:
    method: str = "linear"
    floater_hormann_d: int = 3
    robust: bool = False

class SolverArrays(NamedTuple):
    indices: np.ndarray          # complex128, shape (n_total, n_wavs)
    thicknesses: np.ndarray      # float64,    shape (n_total,)
    incoherent_flags: np.ndarray # bool,       shape (n_total,)
    rough_types: np.ndarray      # int32,      shape (n_total,)
    rough_vals: np.ndarray       # float64,    shape (n_total,)