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
    """Per-interface roughness form factor (solver contract, [nm] sigma).

    Canonical definition shared by the structure model and the smatrix
    engine (`navette.smatrix.RoughnessType` re-exports this): NONE is an
    ideal interface; LINEAR/STEP/EXPONENTIAL/GAUSSIAN are analytic
    graded-index profiles; NEVOT_CROCE is the Nevot-Croce X-ray factor.
    Stored on :class:`Layer.rough_type` and passed to the engine as int.
    """
    NONE = 0
    LINEAR = 1
    STEP = 2
    EXPONENTIAL = 3
    GAUSSIAN = 4
    NEVOT_CROCE = 5

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

class OptMask(IntEnum):
    """Slots of the per-group optimization mask (`Group.optimization_mask`).

    Convention: 1 = the property may be optimized, 0 = fixed. The mask
    refines `Layer.optimize` per property for optimizers that support it;
    `get_optimization_parameters` already honors the THICKNESS slot.
    MATERIAL governs material-substitution moves (e.g. needle insertion).
    """
    THICKNESS = 0
    N = 1
    K = 2
    ROUGHNESS = 3
    INH_DELTA = 4
    INTERFACE = 5
    MATERIAL = 6

class LayerType(IntEnum):
    """Design role of a layer (vocabulary for the former open `layer_typ` int).

    Markers delimit stacks: a `STACK` block opens with `AMBIENT` and closes
    with `SUBSTRATE`; `FILM` rows are the thin-film sequence. `FILM = 1`
    keeps every pre-enum state file valid (1 was the only value in use).
    """
    AMBIENT = 0
    FILM = 1
    SUBSTRATE = 2

class BlockKind(IntEnum):
    """Composition role of an architect block (declared, never inferred).

    `STACK` spans half-space to half-space; `FILMS` is a thin-film-only
    sequence legal only between stacks (or films), taking its boundary
    media from its neighbors at expansion time.
    """
    STACK = 0
    FILMS = 1

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