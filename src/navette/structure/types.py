# -*- coding: utf-8 -*-
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple
import numpy as np

# Standardised numeric types (Numba-friendly)
FLOAT_TYPE = np.float64
COMPLEX_TYPE = np.complex128
INT_TYPE = np.int32

class ErrorType(IntEnum):
    GAUSSIAN = 0
    UNIFORM = 1
    COMBINED = 2

class RoughnessType(IntEnum):
    NONE = 0
    SCALAR = 1

class ErrorMask(IntEnum):
    THICKNESS = 0
    N_REAL = 1
    N_IMAG = 2
    ROUGHNESS = 3
    INH_DELTA = 4
    INTERFACE = 5

class LayerMask(IntEnum):
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