# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Unified Color Engine (Gold Master)
==================================
Final production release with exact rational constants for 
machine-precision round-trip stability.

This module provides a high-performance, JIT-compiled color science engine
designed for optical engineering and colorimetry. It prioritizes:
1. Exactness: Using rational definitions for CIE constants.
2. Performance: Utilizing Numba for SIMD-friendly machine code generation.
3. Memory Efficiency: Minimizing intermediate array allocations via in-place views.

Extended Features in v2:
- Added CIE 1964 U*V*W* and CIE 1960 UCS.
- Added direct XYZ <-> Oklab.
- Added Delta E 76 and DIN99 metrics.

Changes in v2.1 (Review-Driven Improvements):
- Fix #1:  Internal ``_raw`` fast-path methods eliminate redundant
  ``handle_shapes`` overhead in chained convenience pipelines.
- Fix #2:  ``_prepare_inputs`` materialises ``broadcast_to`` views into
  contiguous arrays before passing to Numba ``prange`` kernels.
- Fix #3:  ``srgb_to_xyz`` / ``xyz_to_srgb`` accept ``clip=True|False``
  to allow HDR / scene-referred workflows.
- Fix #4:  ``xyz_to_xyY`` black-pixel convention is now clearly documented
  as a design decision (Lindbloom convention, not CIE mandated).
- Fix #5:  ``set_strict_ieee(True)`` swaps all transfer-function kernels
  to ``fastmath=False`` variants for IEEE 754 debugging.
- Enhancement #1:  ``delta_E_2000`` now accepts ``k_L``, ``k_C``, ``k_H``
  parametric weights and a ``textiles`` shortcut.
- Enhancement #2:  Added ``delta_E_94`` (CIE 1994) and ``delta_E_CMC``
  (CMC l:c 1984) metrics.
- Enhancement #3:  Spectral pipeline docstring documents Y-normalisation
  convention explicitly.

References:
    - CIE 15:2004 "Colorimetry"
    - IEC 61966-2-1:1999 (sRGB Standard)
    - Sharma, G., Wu, W., & Dalal, E. N. (2005). "The CIEDE2000 color-difference formula".
    - Clarke, McDonald, Rigg (1984). "CMC l:c colour difference formula".
    - CIE Publication 116-1995 (CIE 1994 colour difference).
"""

import functools
import time
import numpy as np
from numba import njit, float64, prange
from typing import Tuple, Final, TypeAlias, Callable, Union, Sequence, Any

__all__ = [
    # --- Type Aliases ---
    "ArrayFloat",

    # --- Constants ---
    "REF_WHITE_D65",
    "REF_WHITE_D50",
    "LAB_EPSILON",
    "LAB_KAPPA",
    "C25_7",
    "DEG2RAD",
    "RAD2DEG",

    # --- Configuration ---
    "set_strict_ieee",

    # --- Matrices ---
    # sRGB
    "M_XYZ_TO_SRGB_T",
    "M_SRGB_TO_XYZ_T",
    # Oklab (Legacy/sRGB)
    "M1_OKLAB_SRGB_T",
    "M2_OKLAB_SRGB_T",
    "M1_OKLAB_SRGB_INV_T",
    "M2_OKLAB_SRGB_INV_T",
    # Oklab (Standard/XYZ)
    "M1_XYZ_TO_LMS_OKLAB_T",
    "M1_LMS_TO_XYZ_OKLAB_T",
    "M2_LMS_TO_LAB_OKLAB_T",
    "M2_LAB_TO_LMS_OKLAB_T",
    # Bradford
    "M_BRADFORD_T",
    "M_BRADFORD_INV_T",

    # --- Decorators ---
    "handle_shapes",

    # --- Function ---
    "_fast_gamma_srgb",
    "_xyz_to_uv_prime",

    # --- Classes ---
    "ColorSpaceEngine",
    "ChromaticAdaptation",
    "GamutMapping",
    "ColorMetrics",
    "SpectralPipeline",
]

# --- Type Aliases ---
# Loosened to allow float32/float64 and views.
# NOTE: Internal kernels compile to float64. float32 inputs will incur a copy 
# during the cast to float64 inside the Numba kernels.
ArrayFloat: TypeAlias = np.typing.NDArray[np.floating]

# --- Constants & Pre-Transposed Matrices ---

# Standard Illuminants (Y=1.0)
# D65: Average daylight (approx 6500K)
REF_WHITE_D65: Final[ArrayFloat] = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
# D50: Horizon daylight (approx 5000K), standard for printing (ICC)
REF_WHITE_D50: Final[ArrayFloat] = np.array([0.96422, 1.00000, 0.82521], dtype=np.float64)

# sRGB Matrices
# Defined by IEC 61966-2-1.
# We pre-transpose these because NumPy's dot product is optimized for C-contiguous 
# arrays. A @ B.T is often faster or required depending on the layout.
_M_XYZ_TO_SRGB_BASE = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float64)
M_XYZ_TO_SRGB_T: Final[ArrayFloat] = _M_XYZ_TO_SRGB_BASE.T.copy()

_M_SRGB_TO_XYZ_BASE = np.array([
    [ 0.4124564,  0.3575761,  0.1804375],
    [ 0.2126729,  0.7151522,  0.0721750],
    [ 0.0193339,  0.1191920,  0.9503041]
], dtype=np.float64)
M_SRGB_TO_XYZ_T: Final[ArrayFloat] = _M_SRGB_TO_XYZ_BASE.T.copy()

# Oklab Matrices (sRGB oriented - Legacy Internal)
# These matrices bake in the sRGB-to-Linear conversion specific to the original 
# Oklab blog post implementation. They are kept for sRGB pipeline compatibility.
_M1_OKLAB_SRGB = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
], dtype=np.float64)
M1_OKLAB_SRGB_T: Final[ArrayFloat] = _M1_OKLAB_SRGB.T.copy()

_M2_OKLAB_SRGB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
], dtype=np.float64)
M2_OKLAB_SRGB_T: Final[ArrayFloat] = _M2_OKLAB_SRGB.T.copy()

M1_OKLAB_SRGB_INV_T: Final[ArrayFloat] = np.linalg.inv(_M1_OKLAB_SRGB).T.copy()
M2_OKLAB_SRGB_INV_T: Final[ArrayFloat] = np.linalg.inv(_M2_OKLAB_SRGB).T.copy()

# Oklab Matrices (XYZ oriented - Standard Definition)
# Standard matrices for converting XYZ -> LMS -> Oklab.
# M1: XYZ to Cone Response (LMS)
_M1_XYZ_TO_LMS_OKLAB = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070]
], dtype=np.float64)
M1_XYZ_TO_LMS_OKLAB_T: Final[ArrayFloat] = _M1_XYZ_TO_LMS_OKLAB.T.copy()
M1_LMS_TO_XYZ_OKLAB_T: Final[ArrayFloat] = np.linalg.inv(_M1_XYZ_TO_LMS_OKLAB).T.copy()

# Note: M2 is generally the same (LMS_to_Lab), but we define explicitly to match suite
# M2: LMS (cubed) to Oklab
_M2_LMS_TO_LAB_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
], dtype=np.float64)
M2_LMS_TO_LAB_OKLAB_T: Final[ArrayFloat] = _M2_LMS_TO_LAB_OKLAB.T.copy()
M2_LAB_TO_LMS_OKLAB_T: Final[ArrayFloat] = np.linalg.inv(_M2_LMS_TO_LAB_OKLAB).T.copy()


# Bradford Adaptation
# Used for Chromatic Adaptation Transforms (CAT).
# Transforms XYZ to "sharpened" cone responses for gain application.
_M_BRADFORD = np.array([
    [ 0.8951000,  0.2664000, -0.1614000],
    [-0.7502000,  1.7135000,  0.0367000],
    [ 0.0389000, -0.0685000,  1.0296000]
], dtype=np.float64)
# M_BRADFORD_T transforms row-vector XYZ to row-vector LMS
M_BRADFORD_T: Final[ArrayFloat] = _M_BRADFORD.T.copy()
# M_BRADFORD_INV_T transforms row-vector LMS to row-vector XYZ
M_BRADFORD_INV_T: Final[ArrayFloat] = np.linalg.inv(_M_BRADFORD).T.copy()

# --- Exact Rational Math Constants ---
# Defined by CIE 1976 for the Lab transformation.
# delta = 6/29 is the threshold where the function switches from cubic to linear.
_LAB_DELTA: Final[float] = 6.0 / 29.0
LAB_EPSILON: Final[float] = _LAB_DELTA * _LAB_DELTA * _LAB_DELTA  # ~0.008856
LAB_KAPPA: Final[float]   = (116.0 * 29.0 * 29.0) / (3.0 * 6.0 * 6.0) # ~903.296

C25_7: Final[float]       = 25.0**7
DEG2RAD: Final[float]     = np.pi / 180.0
RAD2DEG: Final[float]     = 180.0 / np.pi


# --- Runtime Configuration ---
# When True, Numba kernels use fastmath=False variants that preserve strict
# IEEE 754 semantics (inf / NaN propagation, no FP reassociation).  Useful
# for debugging edge-case numerical issues.
#
# Toggle at runtime via:
#     import CIE_colorengine as ce
#     ce.set_strict_ieee(True)   # enable strict mode
#     ce.set_strict_ieee(False)  # back to fast mode (default)
_STRICT_IEEE: bool = False

def set_strict_ieee(enabled: bool = True) -> None:
    """
    Toggle between fast (default) and strict IEEE 754 Numba kernels.
    
    When ``enabled=True``, all Lab/sRGB transfer functions use
    ``fastmath=False`` kernels that guarantee correct inf/NaN propagation
    at the cost of ~10-30 % throughput.
    
    Args:
        enabled: If True, use strict IEEE mode.
    """
    global _STRICT_IEEE
    _STRICT_IEEE = bool(enabled)


# =============================================================================
# 1. ROBUST DECORATORS
# =============================================================================

def handle_shapes(func: Callable[..., ArrayFloat]) -> Callable[..., ArrayFloat]:
    """
    Decorator to normalize inputs to (N, 3) and safeguard shape.
    
    This ensures that 1D inputs (single pixels) are treated as 2D batches 
    internally, simplifying the kernels.
    
    Args:
        func: The function to decorate.

    Returns:
        The wrapped function with shape handling.
        - If input is (3,), returns (3,)
        - If input is (N, 3), returns (N, 3)
    """
    @functools.wraps(func)
    def wrapper(arr: ArrayFloat, *args: Any, **kwargs: Any) -> ArrayFloat:
        # np.atleast_2d is cheap (returns view if possible)
        arr_in = np.ascontiguousarray(np.atleast_2d(arr))
        
        if arr_in.shape[-1] != 3:
            raise ValueError(f"Expected last dimension size 3, got {arr_in.shape[-1]}")
            
        res = func(arr_in, *args, **kwargs)
        
        if arr.ndim == 1:
            return res[0]
        return res
    return wrapper


# =============================================================================
# 2. LOW-LEVEL MATH KERNELS (Numba Optimized)
# =============================================================================
# NOTE: fastmath=True allows reassociation and relaxed IEEE compliance.
# This is necessary for maximum performance but may result in minute 
# differences from strict IEEE reference implementations.

@njit(cache=True, fastmath=True)
def _fast_gamma_srgb(linear: ArrayFloat) -> ArrayFloat:
    """
    Applies sRGB OETF (Gamma Correction).
    
    Standard: IEC 61966-2-1
    
    Performance Note:
        Uses an explicit loop instead of `np.where` to avoid allocating a 
        boolean mask array, which improves cache locality.
    """
    out = np.empty_like(linear)
    # Micro-opt: Use .ravel() for direct memory access which is slightly 
    # faster than .flat iterator in Numba for contiguous arrays.
    linear_flat = linear.ravel()
    out_flat = out.ravel()
    
    for i in range(linear.size):
        v = linear_flat[i]
        # IEC 61966-2-1 defines the slope as exactly 12.92
        if v <= 0.0031308:
            out_flat[i] = 12.92 * v
        else:
            out_flat[i] = 1.055 * (v ** (1.0/2.4)) - 0.055
    return out

@njit(cache=True, fastmath=True)
def _fast_inverse_gamma_srgb(srgb: ArrayFloat) -> ArrayFloat:
    """
    Applies sRGB EOTF (Inverse Gamma).
    
    Standard: IEC 61966-2-1
    """
    # Optimized: Explicit loop avoids allocating bool mask array from np.where
    out = np.empty_like(srgb)
    srgb_flat = srgb.ravel()
    out_flat = out.ravel()
    
    for i in range(srgb.size):
        v = srgb_flat[i]
        if v <= 0.04045:
            out_flat[i] = v / 12.92
        else:
            out_flat[i] = ((v + 0.055) / 1.055) ** 2.4
    return out

@njit(cache=True, fastmath=True)
def _xyz_to_lab_f(t: ArrayFloat) -> ArrayFloat:
    """
    Non-linear transfer function f(t) for CIELAB.
    
    This is the "cube root" part of the Lab transform, with a linear slope
    near zero to prevent infinite slope.
    """
    # Optimized loop form to avoid boolean masks
    out = np.empty_like(t)
    # Using ravel() for consistent hot-path performance
    t_flat = t.ravel()
    out_flat = out.ravel()
    
    for i in range(t.size):
        v = t_flat[i]
        if v > LAB_EPSILON:
            out_flat[i] = v ** (1.0/3.0)
        else:
            out_flat[i] = (LAB_KAPPA * v + 16.0) / 116.0
    return out

@njit(cache=True, fastmath=True)
def _lab_to_xyz_f_inv(t: ArrayFloat) -> ArrayFloat:
    """
    Inverse non-linear transfer function for CIELAB.
    
    Uses multiplication form (116*t - 16)/k instead of (t - 16/116)/(k/116)
    to minimize floating point division errors near the delta threshold.
    """
    out = np.empty_like(t)
    t_flat = t.ravel()
    out_flat = out.ravel()
    
    for i in range(t.size):
        v = t_flat[i]
        if v > _LAB_DELTA:
            out_flat[i] = v ** 3.0
        else:
            out_flat[i] = (116.0 * v - 16.0) / LAB_KAPPA
    return out


# --- Strict IEEE 754 kernel variants (fastmath=False) ---
# These are used when _STRICT_IEEE is True.  They guarantee correct
# inf/NaN propagation and no floating-point reassociation.

@njit(cache=True, fastmath=False)
def _fast_gamma_srgb_strict(linear: ArrayFloat) -> ArrayFloat:
    """sRGB OETF — strict IEEE 754 variant."""
    out = np.empty_like(linear)
    linear_flat = linear.ravel()
    out_flat = out.ravel()
    for i in range(linear.size):
        v = linear_flat[i]
        if v <= 0.0031308:
            out_flat[i] = 12.92 * v
        else:
            out_flat[i] = 1.055 * (v ** (1.0/2.4)) - 0.055
    return out

@njit(cache=True, fastmath=False)
def _fast_inverse_gamma_srgb_strict(srgb: ArrayFloat) -> ArrayFloat:
    """sRGB EOTF — strict IEEE 754 variant."""
    out = np.empty_like(srgb)
    srgb_flat = srgb.ravel()
    out_flat = out.ravel()
    for i in range(srgb.size):
        v = srgb_flat[i]
        if v <= 0.04045:
            out_flat[i] = v / 12.92
        else:
            out_flat[i] = ((v + 0.055) / 1.055) ** 2.4
    return out

@njit(cache=True, fastmath=False)
def _xyz_to_lab_f_strict(t: ArrayFloat) -> ArrayFloat:
    """Lab f(t) — strict IEEE 754 variant."""
    out = np.empty_like(t)
    t_flat = t.ravel()
    out_flat = out.ravel()
    for i in range(t.size):
        v = t_flat[i]
        if v > LAB_EPSILON:
            out_flat[i] = v ** (1.0/3.0)
        else:
            out_flat[i] = (LAB_KAPPA * v + 16.0) / 116.0
    return out

@njit(cache=True, fastmath=False)
def _lab_to_xyz_f_inv_strict(t: ArrayFloat) -> ArrayFloat:
    """Lab f_inv(t) — strict IEEE 754 variant."""
    out = np.empty_like(t)
    t_flat = t.ravel()
    out_flat = out.ravel()
    for i in range(t.size):
        v = t_flat[i]
        if v > _LAB_DELTA:
            out_flat[i] = v ** 3.0
        else:
            out_flat[i] = (116.0 * v - 16.0) / LAB_KAPPA
    return out


# --- Kernel dispatchers ---
# These thin wrappers check the global _STRICT_IEEE flag and delegate
# to the appropriate compiled variant.  The overhead of one Python-level
# branch is negligible relative to a vectorised Numba call.

def _gamma_srgb(linear: ArrayFloat) -> ArrayFloat:
    """Dispatch sRGB OETF to fast or strict kernel."""
    if _STRICT_IEEE:
        return _fast_gamma_srgb_strict(linear)
    return _fast_gamma_srgb(linear)

def _inverse_gamma_srgb(srgb: ArrayFloat) -> ArrayFloat:
    """Dispatch sRGB EOTF to fast or strict kernel."""
    if _STRICT_IEEE:
        return _fast_inverse_gamma_srgb_strict(srgb)
    return _fast_inverse_gamma_srgb(srgb)

def _lab_f(t: ArrayFloat) -> ArrayFloat:
    """Dispatch Lab f(t) to fast or strict kernel."""
    if _STRICT_IEEE:
        return _xyz_to_lab_f_strict(t)
    return _xyz_to_lab_f(t)

def _lab_f_inv(t: ArrayFloat) -> ArrayFloat:
    """Dispatch Lab f_inv(t) to fast or strict kernel."""
    if _STRICT_IEEE:
        return _lab_to_xyz_f_inv_strict(t)
    return _lab_to_xyz_f_inv(t)

@njit(cache=True, fastmath=True)
def _xyz_to_uv_prime(xyz_arr: ArrayFloat) -> ArrayFloat:
    """
    Calculates CIE 1976 u', v' chromaticity coordinates from XYZ.
    
    Formulas:
        u' = 4X / (X + 15Y + 3Z)
        v' = 9Y / (X + 15Y + 3Z)
    """
    out = np.zeros((xyz_arr.shape[0], 2), dtype=np.float64)
    # Flatten access for speed
    X = xyz_arr[:, 0]
    Y = xyz_arr[:, 1]
    Z = xyz_arr[:, 2]
    
    denom = X + 15.0 * Y + 3.0 * Z
    
    # Using explicit loop for Numba efficiency
    for i in range(denom.shape[0]):
        d = denom[i]
        # Avoid division by zero (black returns 0,0)
        if d > 1e-12:
            inv_d = 1.0 / d
            out[i, 0] = 4.0 * X[i] * inv_d
            out[i, 1] = 9.0 * Y[i] * inv_d
    return out

@njit(cache=True, fastmath=True)
def _lab_to_lch_kernel(lab: ArrayFloat) -> ArrayFloat:
    """
    Low-level kernel for Lab -> LCh conversion.
    Input shape (N, 3), Output shape (N, 3).
    """
    n = lab.shape[0]
    lch = np.empty_like(lab)
    
    for i in range(n):
        L, a, b = lab[i, 0], lab[i, 1], lab[i, 2]
        C = np.hypot(a, b)
        h_rad = np.arctan2(b, a)
        # OPTIMIZATION: Use constant multiplication instead of function call
        h_deg = h_rad * RAD2DEG
        if h_deg < 0: h_deg += 360.0
        lch[i, 0], lch[i, 1], lch[i, 2] = L, C, h_deg
    return lch

@njit(cache=True, fastmath=True)
def _lch_to_lab_kernel(lch: ArrayFloat) -> ArrayFloat:
    """
    Low-level kernel for LCh -> Lab conversion.
    Input shape (N, 3), Output shape (N, 3).
    """
    n = lch.shape[0]
    lab = np.empty_like(lch)
    
    for i in range(n):
        L, C, h_deg = lch[i, 0], lch[i, 1], lch[i, 2]
        # OPTIMIZATION: Use constant multiplication instead of function call
        h_rad = h_deg * DEG2RAD
        lab[i, 0] = L
        lab[i, 1] = C * np.cos(h_rad)
        lab[i, 2] = C * np.sin(h_rad)
    return lab

@njit(cache=True, fastmath=True)
def _xyz_to_uvw_kernel(xyz: ArrayFloat, un: float, vn: float) -> ArrayFloat:
    """
    Kernel for CIE 1964 U*V*W*.
    
    Implements the transformation using the 1960 UCS u,v coordinates and the 
    white point (un, vn).
    """
    n = xyz.shape[0]
    out = np.empty_like(xyz)
    
    for i in range(n):
        X = xyz[i, 0] * 100.0
        Y = xyz[i, 1] * 100.0
        Z = xyz[i, 2] * 100.0
        
        # Calculate u, v (CIE 1960)
        denom = X + 15.0 * Y + 3.0 * Z
        if denom < 1e-12:
            u, v = 0.0, 0.0
        else:
            u = (4.0 * X) / denom
            v = (6.0 * Y) / denom
            
        W_star = 25.0 * (Y ** (1.0/3.0)) - 17.0
        U_star = 13.0 * W_star * (u - un)
        V_star = 13.0 * W_star * (v - vn)
        
        out[i, 0] = U_star
        out[i, 1] = V_star
        out[i, 2] = W_star
    return out

@njit(cache=True, fastmath=True)
def _uvw_to_xyz_kernel(uvw: ArrayFloat, un: float, vn: float) -> ArrayFloat:
    """
    Kernel for CIE 1964 U*V*W* -> XYZ.
    """
    n = uvw.shape[0]
    out = np.empty_like(uvw)
    
    for i in range(n):
        U_s, V_s, W_s = uvw[i, 0], uvw[i, 1], uvw[i, 2]
        
        # Y from W*
        Y100 = ((W_s + 17.0) / 25.0) ** 3.0
        
        # u, v from U*, V*, W*
        if abs(W_s) < 1e-12:
            u, v = un, vn
        else:
            u = U_s / (13.0 * W_s) + un
            v = V_s / (13.0 * W_s) + vn
            
        # u, v -> x, y
        denom_uv = 2.0 * u - 8.0 * v + 4.0
        if abs(denom_uv) < 1e-12:
            x, y = 0.0, 0.0
        else:
            x = (3.0 * u) / denom_uv
            y = (2.0 * v) / denom_uv
            
        # x, y, Y -> X, Z
        if abs(y) < 1e-12:
            X_out, Z_out = 0.0, 0.0
        else:
            X_out = x * Y100 / y
            Z_out = (1.0 - x - y) * Y100 / y
            
        out[i, 0] = X_out / 100.0
        out[i, 1] = Y100 / 100.0
        out[i, 2] = Z_out / 100.0
    return out

@njit(cache=True, fastmath=True)
def _uv_prime_to_xy_kernel(uv_prime: ArrayFloat) -> ArrayFloat:
    """
    Kernel for CIE 1976 (u', v') -> CIE 1931 (x, y).
    Input: (N, 2), Output: (N, 2)
    """
    n = uv_prime.shape[0]
    xy = np.empty_like(uv_prime)
    
    # Flatten access
    u_p_flat = uv_prime[:, 0]
    v_p_flat = uv_prime[:, 1]
    
    for i in range(n):
        up = u_p_flat[i]
        vp = v_p_flat[i]
        
        # Denominator: 6u' - 16v' + 12
        denom = 6.0 * up - 16.0 * vp + 12.0
        
        if abs(denom) < 1e-12:
            xy[i, 0] = 0.0
            xy[i, 1] = 0.0
        else:
            inv_d = 1.0 / denom
            xy[i, 0] = 9.0 * up * inv_d
            xy[i, 1] = 4.0 * vp * inv_d
            
    return xy

@njit(cache=True, fastmath=True)
def _uv_1960_to_xy_kernel(uv: ArrayFloat) -> ArrayFloat:
    """
    Kernel for CIE 1960 (u, v) -> CIE 1931 (x, y).
    Input: (N, 2), Output: (N, 2)
    """
    n = uv.shape[0]
    xy = np.empty_like(uv)
    
    u_flat = uv[:, 0]
    v_flat = uv[:, 1]
    
    for i in range(n):
        u = u_flat[i]
        v = v_flat[i]
        
        # Denominator: 2u - 8v + 4
        denom = 2.0 * u - 8.0 * v + 4.0
        
        if abs(denom) < 1e-12:
            xy[i, 0] = 0.0
            xy[i, 1] = 0.0
        else:
            inv_d = 1.0 / denom
            xy[i, 0] = 3.0 * u * inv_d
            xy[i, 1] = 2.0 * v * inv_d
            
    return xy

@njit(cache=True, fastmath=True)
def _din99_kernel(lab1: ArrayFloat, lab2: ArrayFloat, kE: float, kCH: float) -> ArrayFloat:
    """
    Kernel for DIN99 Delta E.
    
    The DIN99 color space transforms CIELAB into a space where Euclidean distance
    better approximates perceptual differences, particularly by rotating the a,b 
    axes by 16 degrees.
    """
    n = lab1.shape[0]
    res = np.empty(n, dtype=np.float64)
    
    cos_16 = np.cos(np.radians(16.0))
    sin_16 = np.sin(np.radians(16.0))
    
    for i in range(n):
        # Sample 1
        L1, a1, b1 = lab1[i, 0], lab1[i, 1], lab1[i, 2]
        L99_1 = 105.51 * np.log(1.0 + 0.0158 * L1) * kE
        e1 = a1 * cos_16 + b1 * sin_16
        f1 = 0.7 * (-a1 * sin_16 + b1 * cos_16)
        G1 = np.sqrt(e1*e1 + f1*f1)
        if G1 < 1e-12:
            C99_1, h99_1 = 0.0, 0.0
        else:
            C99_1 = np.log(1.0 + 0.045 * G1) * kCH
            h99_1 = np.arctan2(f1, e1)
        a99_1 = C99_1 * np.cos(h99_1)
        b99_1 = C99_1 * np.sin(h99_1)
        
        # Sample 2
        L2, a2, b2 = lab2[i, 0], lab2[i, 1], lab2[i, 2]
        L99_2 = 105.51 * np.log(1.0 + 0.0158 * L2) * kE
        e2 = a2 * cos_16 + b2 * sin_16
        f2 = 0.7 * (-a2 * sin_16 + b2 * cos_16)
        G2 = np.sqrt(e2*e2 + f2*f2)
        if G2 < 1e-12:
            C99_2, h99_2 = 0.0, 0.0
        else:
            C99_2 = np.log(1.0 + 0.045 * G2) * kCH
            h99_2 = np.arctan2(f2, e2)
        a99_2 = C99_2 * np.cos(h99_2)
        b99_2 = C99_2 * np.sin(h99_2)
        
        dL = L99_1 - L99_2
        da = a99_1 - a99_2
        db = b99_1 - b99_2
        res[i] = np.sqrt(dL*dL + da*da + db*db)
        
    return res


# =============================================================================
# 3. COLOR SPACE ENGINE
# =============================================================================

class ColorSpaceEngine:
    """Static utility class for unified color space transformations.
    
    Architecture Note (v2.1):
        Core transforms provide both a public ``@handle_shapes`` decorated API
        and an internal ``_raw`` fast-path that assumes pre-validated (N, 3)
        float64 input.  Convenience pipelines (e.g. ``srgb_to_lab``) call the
        ``_raw`` variants to avoid redundant shape checks at each stage.
    """

    # =====================================================================
    #  Internal _raw fast-path methods  (assume validated (N, 3) float64)
    # =====================================================================

    @staticmethod
    def _srgb_to_xyz_raw(rgb_array: ArrayFloat, clip: bool = True) -> ArrayFloat:
        """Raw sRGB → XYZ.  *rgb_array* must be (N, 3) float64."""
        if clip:
            rgb_array = np.clip(rgb_array, 0.0, 1.0)
        linear = _inverse_gamma_srgb(rgb_array)
        return np.dot(linear, M_SRGB_TO_XYZ_T)

    @staticmethod
    def _xyz_to_srgb_raw(xyz_array: ArrayFloat, clip: bool = True) -> ArrayFloat:
        """Raw XYZ → sRGB.  *xyz_array* must be (N, 3) float64."""
        linear = np.dot(xyz_array, M_XYZ_TO_SRGB_T)
        if clip:
            linear = np.clip(linear, 0.0, 1.0)
        return _gamma_srgb(linear)

    @staticmethod
    def _xyz_to_xyY_raw(xyz_array: ArrayFloat) -> ArrayFloat:
        """Raw XYZ → xyY.  *xyz_array* must be (N, 3) float64."""
        sum_xyz = np.sum(xyz_array, axis=-1)
        mask = sum_xyz > 1e-12
        xyY = np.zeros_like(xyz_array)

        if np.any(mask):
            inv_sum = 1.0 / sum_xyz[mask]
            xyY[mask, 0] = xyz_array[mask, 0] * inv_sum
            xyY[mask, 1] = xyz_array[mask, 1] * inv_sum
            xyY[mask, 2] = xyz_array[mask, 1]

        # NOTE (Design Decision):  For zero-luminance (black) pixels this
        # implementation substitutes the D65 white-point chromaticity
        # (x=0.3127, y=0.3290) to guarantee NaN-free output.  This follows
        # the convention from Bruce Lindbloom and is *not* mandated by CIE.
        # The colour-science library (>=0.4.4) instead returns (0, 0, 0) for
        # black pixels (see colour-science/colour#1153).  Users who require
        # CIE-strict behaviour can post-process with a mask on Y==0.
        xyY[~mask, 0] = 0.3127
        xyY[~mask, 1] = 0.3290
        xyY[~mask, 2] = 0.0
        return xyY

    @staticmethod
    def _xyY_to_xyz_raw(xyY_array: ArrayFloat) -> ArrayFloat:
        """Raw xyY → XYZ.  *xyY_array* must be (N, 3) float64."""
        x, y, Y = xyY_array[..., 0], xyY_array[..., 1], xyY_array[..., 2]
        xyz = np.zeros_like(xyY_array)
        mask = y > 1e-12
        if np.any(mask):
            factor = Y[mask] / y[mask]
            xyz[mask, 0] = x[mask] * factor
            xyz[mask, 1] = Y[mask]
            xyz[mask, 2] = (1.0 - x[mask] - y[mask]) * factor
        return xyz

    @staticmethod
    def _xyz_to_lab_raw(xyz_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """Raw XYZ → Lab.  *xyz_array* must be (N, 3) float64."""
        xyz_norm = xyz_array / illuminant
        f_xyz = _lab_f(xyz_norm)

        out = np.empty_like(xyz_array)
        out[..., 0] = 116.0 * f_xyz[..., 1] - 16.0
        out[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
        out[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
        return out

    @staticmethod
    def _lab_to_xyz_raw(lab_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """Raw Lab → XYZ.  *lab_array* must be (N, 3) float64."""
        L, a, b = lab_array[..., 0], lab_array[..., 1], lab_array[..., 2]

        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        xyz = np.empty_like(lab_array)
        xyz[..., 0] = _lab_f_inv(fx)
        xyz[..., 1] = _lab_f_inv(fy)
        xyz[..., 2] = _lab_f_inv(fz)

        xyz *= illuminant
        return xyz

    @staticmethod
    def _lab_to_lch_raw(lab_array: ArrayFloat) -> ArrayFloat:
        """Raw Lab → LCh.  *lab_array* must be (N, 3) float64."""
        return _lab_to_lch_kernel(lab_array)

    @staticmethod
    def _lch_to_lab_raw(lch_array: ArrayFloat) -> ArrayFloat:
        """Raw LCh → Lab.  *lch_array* must be (N, 3) float64."""
        return _lch_to_lab_kernel(lch_array)

    @staticmethod
    def _xyz_to_luv_raw(xyz_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """Raw XYZ → Luv.  *xyz_array* must be (N, 3) float64."""
        uv_prime = _xyz_to_uv_prime(xyz_array)

        ill_2d = np.ascontiguousarray(np.atleast_2d(illuminant))
        uv_prime_n = _xyz_to_uv_prime(ill_2d)
        u_n, v_n = uv_prime_n[0, 0], uv_prime_n[0, 1]

        Y_norm = xyz_array[:, 1] / illuminant[1]
        f_y = _lab_f(Y_norm)
        L = 116.0 * f_y - 16.0

        out = np.empty_like(xyz_array)
        out[:, 0] = L
        out[:, 1] = 13.0 * L * (uv_prime[:, 0] - u_n)
        out[:, 2] = 13.0 * L * (uv_prime[:, 1] - v_n)
        return out

    @staticmethod
    def _luv_to_xyz_raw(luv_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """Raw Luv → XYZ.  *luv_array* must be (N, 3) float64."""
        L, u, v = luv_array[:, 0], luv_array[:, 1], luv_array[:, 2]

        ill_2d = np.ascontiguousarray(np.atleast_2d(illuminant))
        uv_prime_n = _xyz_to_uv_prime(ill_2d)
        u_n, v_n = uv_prime_n[0, 0], uv_prime_n[0, 1]

        mask = L > 1e-12
        u_prime = np.empty_like(L)
        v_prime = np.empty_like(L)

        u_prime[~mask] = u_n
        v_prime[~mask] = v_n

        if np.any(mask):
            inv_13L = 1.0 / (13.0 * L[mask])
            u_prime[mask] = (u[mask] * inv_13L) + u_n
            v_prime[mask] = (v[mask] * inv_13L) + v_n

        fy = (L + 16.0) / 116.0
        Y = _lab_f_inv(fy) * illuminant[1]

        X = np.zeros_like(Y)
        Z = np.zeros_like(Y)

        mask_v = (v_prime > 1e-12) & mask
        if np.any(mask_v):
            Y_valid = Y[mask_v]
            up, vp = u_prime[mask_v], v_prime[mask_v]
            inv_4vp = 1.0 / (4.0 * vp)
            X[mask_v] = Y_valid * 9.0 * up * inv_4vp
            Z[mask_v] = Y_valid * (12.0 - 3.0 * up - 20.0 * vp) * inv_4vp

        out = np.empty_like(luv_array)
        out[:, 0] = X
        out[:, 1] = Y
        out[:, 2] = Z
        return out

    @staticmethod
    def _xyz_to_oklab_raw(xyz_array: ArrayFloat) -> ArrayFloat:
        """Raw XYZ → Oklab.  *xyz_array* must be (N, 3) float64."""
        lms = np.dot(xyz_array, M1_XYZ_TO_LMS_OKLAB_T)
        lms_prime = np.sign(lms) * np.abs(lms) ** (1.0 / 3.0)
        return np.dot(lms_prime, M2_LMS_TO_LAB_OKLAB_T)

    @staticmethod
    def _oklab_to_xyz_raw(oklab_array: ArrayFloat) -> ArrayFloat:
        """Raw Oklab → XYZ.  *oklab_array* must be (N, 3) float64."""
        lms_prime = np.dot(oklab_array, M2_LAB_TO_LMS_OKLAB_T)
        lms = np.sign(lms_prime) * np.abs(lms_prime) ** 3.0
        return np.dot(lms, M1_LMS_TO_XYZ_OKLAB_T)

    # =====================================================================
    #  Public API  (shape-safe wrappers)
    # =====================================================================

    @staticmethod
    @handle_shapes
    def srgb_to_xyz(rgb_array: ArrayFloat, clip: bool = True) -> ArrayFloat:
        """
        Converts sRGB [0..1] to XYZ [0..1] (D65).
        
        Args:
            rgb_array: Input sRGB data, shape (N, 3) or (3,).
            clip: If True (default), clamps input to [0, 1] before applying 
                  the EOTF.  Set False to preserve scene-referred / HDR values.

        Returns:
            XYZ coordinates (D65 relative).
        """
        return ColorSpaceEngine._srgb_to_xyz_raw(rgb_array, clip=clip)

    @staticmethod
    @handle_shapes
    def xyz_to_srgb(xyz_array: ArrayFloat, clip: bool = True) -> ArrayFloat:
        """
        Converts XYZ [0..1] (D65) to sRGB [0..1].
        
        Args:
            xyz_array: Input XYZ data, shape (N, 3) or (3,).
            clip: If True (default), clamps linear RGB to [0, 1] before gamma.
                  Set False to preserve out-of-gamut linear values (note: the 
                  sRGB OETF is undefined for negative inputs when clip=False).

        Returns:
            sRGB coordinates, gamma corrected.
        """
        return ColorSpaceEngine._xyz_to_srgb_raw(xyz_array, clip=clip)

    @staticmethod
    @handle_shapes
    def xyz_to_xyY(xyz_array: ArrayFloat) -> ArrayFloat:
        """
        Converts XYZ to xyY (Chromaticity + Luminance).
        
        Standard formula:
            x = X / (X+Y+Z)
            y = Y / (X+Y+Z)
            Y = Y

        Design Decision — Black-Pixel Handling:
            For black pixels (X+Y+Z ≈ 0) this implementation substitutes the
            D65 white-point chromaticity (x=0.3127, y=0.3290, Y=0.0) to
            guarantee NaN-free output.  This follows the convention described
            by Bruce Lindbloom and is commonly used in display-referred
            pipelines.  It is *not* mandated by CIE 15:2004.  The
            ``colour-science`` library (≥ 0.4.4) instead returns (0, 0, 0)
            for black pixels (see colour-science/colour#1153).

        Args:
            xyz_array: Input XYZ data, shape (N, 3) or (3,).

        Returns:
            xyY coordinates.
        """
        return ColorSpaceEngine._xyz_to_xyY_raw(xyz_array)

    @staticmethod
    @handle_shapes
    def xyY_to_xyz(xyY_array: ArrayFloat) -> ArrayFloat:
        """
        Converts xyY to XYZ.

        Args:
            xyY_array: Input xyY data, shape (N, 3) or (3,).

        Returns:
            XYZ coordinates.
        """
        return ColorSpaceEngine._xyY_to_xyz_raw(xyY_array)

    @staticmethod
    @handle_shapes
    def xyz_to_lab(xyz_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """
        Converts XYZ to CIELAB (L*a*b*).
        
        Args:
            xyz_array: Input XYZ data, shape (N, 3) or (3,).
            illuminant: Reference white point (default D65).

        Returns:
            Lab coordinates.
        """
        return ColorSpaceEngine._xyz_to_lab_raw(xyz_array, illuminant)

    @staticmethod
    @handle_shapes
    def lab_to_xyz(lab_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """
        Converts CIELAB to XYZ.

        Args:
            lab_array: Input Lab data, shape (N, 3) or (3,).
            illuminant: Reference white point (default D65).

        Returns:
            XYZ coordinates.
        """
        return ColorSpaceEngine._lab_to_xyz_raw(lab_array, illuminant)

    @staticmethod
    @handle_shapes
    def lab_to_lch(lab_array: ArrayFloat) -> ArrayFloat:
        """
        Converts CIELAB to CIELCh (Cylindrical representation).
        
        Args:
            lab_array: Input Lab data, shape (N, 3) or (3,).
            
        Returns:
            LCh coordinates (Lightness, Chroma, Hue in degrees).
        """
        return ColorSpaceEngine._lab_to_lch_raw(lab_array)

    @staticmethod
    @handle_shapes
    def lch_to_lab(lch_array: ArrayFloat) -> ArrayFloat:
        """
        Converts CIELCh to CIELAB.
        
        Args:
            lch_array: Input LCh data, shape (N, 3) or (3,).
            
        Returns:
            Lab coordinates (Lightness, a, b).
        """
        return ColorSpaceEngine._lch_to_lab_raw(lch_array)

    @staticmethod
    @handle_shapes
    def xyz_to_luv(xyz_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """
        Converts XYZ to CIELUV.
        
        Useful for emitted light and white point estimation (u'v').
        
        Args:
            xyz_array: Input XYZ data.
            illuminant: Reference white point (default D65).
            
        Returns:
            Luv coordinates.
        """
        return ColorSpaceEngine._xyz_to_luv_raw(xyz_array, illuminant)

    @staticmethod
    @handle_shapes
    def luv_to_xyz(luv_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """
        Converts CIELUV to XYZ.

        Args:
            luv_array: Input Luv data.
            illuminant: Reference white point.
            
        Returns:
            XYZ coordinates.
        """
        return ColorSpaceEngine._luv_to_xyz_raw(luv_array, illuminant)

    # --- Oklab (sRGB Pipeline) ---
    @staticmethod
    @handle_shapes
    def srgb_to_oklab(rgb_array: ArrayFloat) -> ArrayFloat:
        """
        Converts sRGB to Oklab using optimized sRGB matrices.
        
        This pipeline matches the original Oklab blog post implementation which
        assumes sRGB inputs.
        """
        rgb_clipped = np.clip(rgb_array, 0.0, 1.0)
        linear_rgb = _inverse_gamma_srgb(rgb_clipped)
        
        lms = np.dot(linear_rgb, M1_OKLAB_SRGB_T)
        lms_cube = np.sign(lms) * np.abs(lms) ** (1.0/3.0)
        return np.dot(lms_cube, M2_OKLAB_SRGB_T)

    @staticmethod
    @handle_shapes
    def oklab_to_srgb(lab_array: ArrayFloat) -> ArrayFloat:
        """
        Converts Oklab to sRGB.
        
        Note: Output is clipped to [0, 1] range, making this implementation 
        display-referred. Out-of-gamut colors will be clamped.
        """
        lms_prime = np.dot(lab_array, M2_OKLAB_SRGB_INV_T)
        lms_linear = np.sign(lms_prime) * np.abs(lms_prime) ** 3.0
        
        linear_rgb = np.dot(lms_linear, M1_OKLAB_SRGB_INV_T)
        linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
        return _gamma_srgb(linear_rgb)

    # --- Oklab (Direct XYZ Pipeline - RESTORED) ---
    @staticmethod
    @handle_shapes
    def xyz_to_oklab(xyz_array: ArrayFloat) -> ArrayFloat:
        """
        Converts XYZ to Oklab directly.
        Uses standard Oklab M1/M2 matrices (compatible with colorsuite.py).
        
        Args:
            xyz_array: XYZ input (0..1 typical).
            
        Returns:
            Oklab coordinates (L, a, b).
        """
        return ColorSpaceEngine._xyz_to_oklab_raw(xyz_array)

    @staticmethod
    @handle_shapes
    def oklab_to_xyz(oklab_array: ArrayFloat) -> ArrayFloat:
        """
        Converts Oklab to XYZ directly.
        
        Args:
            oklab_array: Oklab input.
        
        Returns:
            XYZ coordinates.
        """
        return ColorSpaceEngine._oklab_to_xyz_raw(oklab_array)

    # --- Legacy & Extended Spaces (RESTORED) ---
    
    @staticmethod
    @handle_shapes
    def xyz_to_uvw(xyz_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """
        Converts XYZ to CIE 1964 U*V*W*.
        Restored from colorsuite.py.
        
        Args:
            xyz_array: XYZ input.
            illuminant: Reference white.
        """
        ill_2d = np.ascontiguousarray(np.atleast_2d(illuminant))
        uv_prime_n = _xyz_to_uv_prime(ill_2d)
        
        # 1964 system uses 1960 u, v.
        # u = u', v = (2/3)v'
        un = uv_prime_n[0, 0]
        vn = (2.0/3.0) * uv_prime_n[0, 1]
        
        return _xyz_to_uvw_kernel(xyz_array, un, vn)

    @staticmethod
    @handle_shapes
    def uvw_to_xyz(uvw_array: ArrayFloat, illuminant: ArrayFloat = REF_WHITE_D65) -> ArrayFloat:
        """
        Converts CIE 1964 U*V*W* to XYZ.
        Restored from colorsuite.py.
        """
        ill_2d = np.ascontiguousarray(np.atleast_2d(illuminant))
        uv_prime_n = _xyz_to_uv_prime(ill_2d)
        
        un = uv_prime_n[0, 0]
        vn = (2.0/3.0) * uv_prime_n[0, 1]
        
        return _uvw_to_xyz_kernel(uvw_array, un, vn)

    @staticmethod
    @handle_shapes
    def xyz_to_ucs(xyz_array: ArrayFloat) -> ArrayFloat:
        """
        Converts XYZ to CIE 1960 UCS (UVW).
        U = 2/3 X, V = Y, W = -0.5X + 1.5Y + 0.5Z
        """
        out = np.empty_like(xyz_array)
        X, Y, Z = xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2]
        out[:, 0] = (2.0/3.0) * X
        out[:, 1] = Y
        out[:, 2] = 0.5 * (-X + 3.0*Y + Z)
        return out

    @staticmethod
    @handle_shapes
    def ucs_to_xyz(ucs_array: ArrayFloat) -> ArrayFloat:
        """
        Converts CIE 1960 UCS (UVW) to XYZ.
        X = 1.5 U, Y = V, Z = 1.5 U - 3 V + 2 W
        """
        out = np.empty_like(ucs_array)
        U, V, W = ucs_array[:, 0], ucs_array[:, 1], ucs_array[:, 2]
        out[:, 0] = 1.5 * U
        out[:, 1] = V
        out[:, 2] = 1.5 * U - 3.0 * V + 2.0 * W
        return out

    @staticmethod
    @handle_shapes
    def xyz_to_ucs_uv(xyz_array: ArrayFloat) -> ArrayFloat:
        """
        Converts XYZ to CIE 1960 Chromaticity Coordinates (u, v).
        """
        # We can leverage the existing u'v' kernel and convert
        # u = u', v = (2/3) v'
        uv_prime = _xyz_to_uv_prime(xyz_array)
        uv_prime[:, 1] *= (2.0/3.0)
        return uv_prime

    @staticmethod
    def uv1976_to_xy(uv_prime_array: ArrayFloat) -> ArrayFloat:
        """
        Converts CIE 1976 (u', v') to CIE 1931 (x, y).
        
        Args:
            uv_prime_array: Input data, shape (N, 2) or (2,).
        
        Returns:
            xy coordinates, shape (N, 2) or (2,).
        """
        # Manual shape handling for 2D inputs
        arr_in = np.ascontiguousarray(np.atleast_2d(uv_prime_array))
        if arr_in.shape[-1] != 2:
            raise ValueError(f"Expected last dimension size 2, got {arr_in.shape[-1]}")
            
        res = _uv_prime_to_xy_kernel(arr_in)
        
        if uv_prime_array.ndim == 1:
            return res[0]
        return res

    @staticmethod
    def uv1960_to_xy(uv_array: ArrayFloat) -> ArrayFloat:
        """
        Converts CIE 1960 (u, v) to CIE 1931 (x, y).
        
        Args:
            uv_array: Input data, shape (N, 2) or (2,).
            
        Returns:
            xy coordinates, shape (N, 2) or (2,).
        """
        arr_in = np.ascontiguousarray(np.atleast_2d(uv_array))
        if arr_in.shape[-1] != 2:
            raise ValueError(f"Expected last dimension size 2, got {arr_in.shape[-1]}")
            
        res = _uv_1960_to_xy_kernel(arr_in)
        
        if uv_array.ndim == 1:
            return res[0]
        return res

    # --- Convenience: sRGB <-> Lab/LCh/Luv/xyY ---
    # These pipelines use _raw methods internally to avoid redundant
    # shape-checking at each intermediate stage (Fix #1).

    @staticmethod
    @handle_shapes
    def srgb_to_lab(rgb_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion sRGB -> CIELAB."""
        xyz = ColorSpaceEngine._srgb_to_xyz_raw(rgb_array)
        return ColorSpaceEngine._xyz_to_lab_raw(xyz)

    @staticmethod
    @handle_shapes
    def lab_to_srgb(lab_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion CIELAB -> sRGB."""
        xyz = ColorSpaceEngine._lab_to_xyz_raw(lab_array)
        return ColorSpaceEngine._xyz_to_srgb_raw(xyz)

    @staticmethod
    @handle_shapes
    def srgb_to_lch(rgb_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion sRGB -> CIELCh."""
        xyz = ColorSpaceEngine._srgb_to_xyz_raw(rgb_array)
        lab = ColorSpaceEngine._xyz_to_lab_raw(xyz)
        return ColorSpaceEngine._lab_to_lch_raw(lab)

    @staticmethod
    @handle_shapes
    def lch_to_srgb(lch_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion CIELCh -> sRGB."""
        lab = ColorSpaceEngine._lch_to_lab_raw(lch_array)
        xyz = ColorSpaceEngine._lab_to_xyz_raw(lab)
        return ColorSpaceEngine._xyz_to_srgb_raw(xyz)

    @staticmethod
    @handle_shapes
    def srgb_to_luv(rgb_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion sRGB -> CIELUV."""
        xyz = ColorSpaceEngine._srgb_to_xyz_raw(rgb_array)
        return ColorSpaceEngine._xyz_to_luv_raw(xyz)

    @staticmethod
    @handle_shapes
    def luv_to_srgb(luv_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion CIELUV -> sRGB."""
        xyz = ColorSpaceEngine._luv_to_xyz_raw(luv_array)
        return ColorSpaceEngine._xyz_to_srgb_raw(xyz)

    @staticmethod
    @handle_shapes
    def srgb_to_xyY(rgb_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion sRGB -> CIE xyY (Chromaticity)."""
        xyz = ColorSpaceEngine._srgb_to_xyz_raw(rgb_array)
        return ColorSpaceEngine._xyz_to_xyY_raw(xyz)

    @staticmethod
    @handle_shapes
    def xyY_to_srgb(xyY_array: ArrayFloat) -> ArrayFloat:
        """Direct conversion CIE xyY -> sRGB."""
        xyz = ColorSpaceEngine._xyY_to_xyz_raw(xyY_array)
        return ColorSpaceEngine._xyz_to_srgb_raw(xyz)


# =============================================================================
# 4. CHROMATIC ADAPTATION & GAMUT MAPPING
# =============================================================================

def _to_hashable(obj: Union[ArrayFloat, Sequence[float]]) -> Tuple[float, ...]:
    """Helper to ensure inputs are hashable tuples for caching."""
    if isinstance(obj, np.ndarray):
        return tuple(obj.ravel())
    return tuple(obj)

@functools.lru_cache(maxsize=16)
def _get_cached_bradford_matrix(src_white_tuple: Tuple[float, ...], dst_white_tuple: Tuple[float, ...]) -> ArrayFloat:
    """
    Cached worker for calculating Bradford matrix. 
    
    Derivation:
    M_composite = M_inv * Gain * M
    Since we operate on row vectors: M_comp = M.T @ Gain @ M_inv.T
    """
    src = np.array(src_white_tuple, dtype=np.float64)
    dst = np.array(dst_white_tuple, dtype=np.float64)
    
    # 1. Convert Source XYZ -> LMS (Cone Response)
    src_lms = np.dot(src, M_BRADFORD_T)
    dst_lms = np.dot(dst, M_BRADFORD_T)

    # 2. Compute Gain Factors (Von Kries)
    # Prevent divide-by-zero for extremely dark white points
    src_lms = np.where(np.abs(src_lms) < 1e-12, 1e-12, src_lms)
    gains = dst_lms / src_lms
    M_gain = np.diag(gains)
    
    # 3. Construct Composite Matrix for Row Vectors
    return M_BRADFORD_T @ M_gain @ M_BRADFORD_INV_T

class ChromaticAdaptation:
    """Handles White Point Adaptation (Bradford Method)."""

    @staticmethod
    def calc_transform_matrix(src_white: ArrayFloat, dst_white: ArrayFloat) -> ArrayFloat:
        """
        Computes the Bradford adaptation matrix between two white points.

        Args:
            src_white: Source white point (XYZ).
            dst_white: Destination white point (XYZ).
        
        Returns:
            3x3 Adaptation Matrix (for row-vector multiplication).
        """
        t_src = _to_hashable(src_white)
        t_dst = _to_hashable(dst_white)
        return _get_cached_bradford_matrix(t_src, t_dst)

    @staticmethod
    @handle_shapes
    def adapt(xyz: ArrayFloat, src_white: ArrayFloat, dst_white: ArrayFloat, clip_negative: bool = True) -> ArrayFloat:
        """
        Adapts XYZ color(s) from source to dest white point using Cached Bradford.
        
        Args:
            xyz: Input XYZ colors.
            src_white: Source white point.
            dst_white: Destination white point.
            clip_negative: If True (default), clamps negative XYZ values to 0.0.

        Returns:
            Adapted XYZ colors.
        """
        if np.allclose(src_white, dst_white):
            # Ensure clip semantics are respected even on identity transform
            if clip_negative:
                return np.where(xyz < -1e-6, 0.0, xyz)
            return xyz
        M = ChromaticAdaptation.calc_transform_matrix(src_white, dst_white)
        res = np.dot(xyz, M)
        if clip_negative:
            return np.where(res < -1e-6, 0.0, res)
        return res

class GamutMapping:
    @staticmethod
    @handle_shapes
    def clip_absolute(rgb: ArrayFloat) -> ArrayFloat:
        """Hard clip to [0, 1]."""
        return np.clip(rgb, 0.0, 1.0)


# =============================================================================
# 5. OPTIMIZED METRICS
# =============================================================================

@njit(float64(float64, float64, float64, float64, float64, float64, float64, float64, float64), cache=True, fastmath=True)
def _delta_e_2000_single_opt(L1: float, a1: float, b1: float, L2: float, a2: float, b2: float, k_L: float, k_C: float, k_H: float) -> float:
    """Highly optimized single-pixel CIEDE2000 with parametric factors."""
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) * 0.5
    C_bar_7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar_7 / (C_bar_7 + C25_7)))
    scale = 1.0 + G
    a1_p = scale * a1
    a2_p = scale * a2
    C1_p = np.hypot(a1_p, b1)
    C2_p = np.hypot(a2_p, b2)
    h1_p = np.degrees(np.arctan2(b1, a1_p)) % 360.0
    h2_p = np.degrees(np.arctan2(b2, a2_p)) % 360.0
    dL_p = L2 - L1
    dC_p = C2_p - C1_p
    dh_p = 0.0
    if C1_p * C2_p > 1e-12:
        diff = h2_p - h1_p
        if abs(diff) <= 180: dh_p = diff
        elif diff > 180: dh_p = diff - 360.0
        else: dh_p = diff + 360.0
    dH_p = 2.0 * np.sqrt(C1_p * C2_p) * np.sin((dh_p * DEG2RAD) * 0.5)
    L_bar_p = (L1 + L2) * 0.5
    C_bar_p = (C1_p + C2_p) * 0.5
    h_bar_p = h1_p + h2_p
    if C1_p * C2_p > 1e-12:
        if abs(h1_p - h2_p) <= 180: h_bar_p *= 0.5
        elif h_bar_p < 360: h_bar_p = (h_bar_p + 360.0) * 0.5
        else: h_bar_p = (h_bar_p - 360.0) * 0.5
    T = 1.0 - 0.17 * np.cos((h_bar_p - 30.0) * DEG2RAD) + \
        0.24 * np.cos((2.0 * h_bar_p) * DEG2RAD) + \
        0.32 * np.cos((3.0 * h_bar_p + 6.0) * DEG2RAD) - \
        0.20 * np.cos((4.0 * h_bar_p - 63.0) * DEG2RAD)
    d_theta = 30.0 * np.exp(-((h_bar_p - 275.0) / 25.0)**2)
    C_bar_p_7 = C_bar_p**7
    RC = 2.0 * np.sqrt(C_bar_p_7 / (C_bar_p_7 + C25_7))
    RT = -np.sin((2.0 * d_theta) * DEG2RAD) * RC
    L_term = (L_bar_p - 50.0)**2
    SL = 1.0 + (0.015 * L_term) / np.sqrt(20.0 + L_term)
    SC = 1.0 + 0.045 * C_bar_p
    SH = 1.0 + 0.015 * C_bar_p * T
    return np.sqrt((dL_p / (k_L * SL))**2 + (dC_p / (k_C * SC))**2 + (dH_p / (k_H * SH))**2 + RT * (dC_p / (k_C * SC)) * (dH_p / (k_H * SH)))

@njit(cache=True, fastmath=True, parallel=True)
def _batch_delta_e_2000(lab1: ArrayFloat, lab2: ArrayFloat, k_L: float, k_C: float, k_H: float) -> ArrayFloat:
    """Vectorized and Parallelized loop for CIEDE2000."""
    n = len(lab1)
    res = np.empty(n, dtype=np.float64)
    for i in prange(n):
        res[i] = _delta_e_2000_single_opt(lab1[i, 0], lab1[i, 1], lab1[i, 2], lab2[i, 0], lab2[i, 1], lab2[i, 2], k_L, k_C, k_H)
    return res

@njit(cache=True, fastmath=True, parallel=True)
def _batch_delta_e_76(lab1: ArrayFloat, lab2: ArrayFloat) -> ArrayFloat:
    """Vectorized and Parallelized loop for Delta E 76."""
    n = len(lab1)
    res = np.empty(n, dtype=np.float64)
    for i in prange(n):
        dL = lab1[i, 0] - lab2[i, 0]
        da = lab1[i, 1] - lab2[i, 1]
        db = lab1[i, 2] - lab2[i, 2]
        res[i] = np.sqrt(dL*dL + da*da + db*db)
    return res

@njit(cache=True, fastmath=True, parallel=True)
def _batch_delta_e_94(lab1: ArrayFloat, lab2: ArrayFloat, k_L: float, K1: float, K2: float) -> ArrayFloat:
    """
    Vectorized CIE 1994 Delta E (CIE Publication 116-1995).
    
    The formula uses the *reference* sample (lab1) to compute weighting
    functions, making it asymmetric: delta_E_94(a, b) != delta_E_94(b, a).
    """
    n = len(lab1)
    res = np.empty(n, dtype=np.float64)
    for i in prange(n):
        L1, a1, b1 = lab1[i, 0], lab1[i, 1], lab1[i, 2]
        L2, a2, b2 = lab2[i, 0], lab2[i, 1], lab2[i, 2]
        
        dL = L1 - L2
        C1 = np.sqrt(a1*a1 + b1*b1)
        C2 = np.sqrt(a2*a2 + b2*b2)
        dC = C1 - C2
        
        da = a1 - a2
        db = b1 - b2
        # dH² = da² + db² - dC²  (can be negative due to FP noise → clamp)
        dH_sq = da*da + db*db - dC*dC
        if dH_sq < 0.0:
            dH_sq = 0.0
        
        SL = 1.0
        SC = 1.0 + K1 * C1
        SH = 1.0 + K2 * C1
        
        term_L = dL / (k_L * SL)
        term_C = dC / SC
        term_H_sq = dH_sq / (SH * SH)
        
        res[i] = np.sqrt(term_L*term_L + term_C*term_C + term_H_sq)
    return res

@njit(cache=True, fastmath=True, parallel=True)
def _batch_delta_e_cmc(lab1: ArrayFloat, lab2: ArrayFloat, pl: float, pc: float) -> ArrayFloat:
    """
    Vectorized CMC l:c (1984) Delta E.
    
    Reference: Clarke, McDonald, Rigg (1984).
    """
    n = len(lab1)
    res = np.empty(n, dtype=np.float64)
    for i in prange(n):
        L1, a1, b1 = lab1[i, 0], lab1[i, 1], lab1[i, 2]
        L2, a2, b2 = lab2[i, 0], lab2[i, 1], lab2[i, 2]
        
        dL = L1 - L2
        C1 = np.sqrt(a1*a1 + b1*b1)
        C2 = np.sqrt(a2*a2 + b2*b2)
        dC = C1 - C2
        
        da = a1 - a2
        db = b1 - b2
        dH_sq = da*da + db*db - dC*dC
        if dH_sq < 0.0:
            dH_sq = 0.0
        
        # Hue angle of reference (degrees)
        h1 = np.degrees(np.arctan2(b1, a1)) % 360.0
        
        # SL
        if L1 < 16.0:
            SL = 0.511
        else:
            SL = (0.040975 * L1) / (1.0 + 0.01765 * L1)
        
        # SC
        SC = (0.0638 * C1) / (1.0 + 0.0131 * C1) + 0.638
        
        # T factor
        if 164.0 <= h1 <= 345.0:
            T = 0.56 + abs(0.2 * np.cos((h1 + 168.0) * DEG2RAD))
        else:
            T = 0.36 + abs(0.4 * np.cos((h1 + 35.0) * DEG2RAD))
        
        # F factor
        C1_4 = C1**4
        F = np.sqrt(C1_4 / (C1_4 + 1900.0))
        
        SH = SC * (F * T + 1.0 - F)
        
        term_L = dL / (pl * SL)
        term_C = dC / (pc * SC)
        term_H_sq = dH_sq / (SH * SH)
        
        res[i] = np.sqrt(term_L*term_L + term_C*term_C + term_H_sq)
    return res

class ColorMetrics:
    @staticmethod
    def _prepare_inputs(lab1: ArrayFloat, lab2: ArrayFloat) -> Tuple[ArrayFloat, ArrayFloat]:
        """
        Broadcasting helper.
        
        Uses NumPy's broadcast_to for shape matching, then ensures the result
        is contiguous.  ``broadcast_to`` creates read-only strided views which
        Numba ``prange`` kernels may silently copy internally.  Making the copy
        explicit here avoids hidden performance cliffs and guarantees
        SIMD-friendly memory layout for the hot loops (Fix #2).
        """
        l1 = np.ascontiguousarray(np.atleast_2d(lab1))
        l2 = np.ascontiguousarray(np.atleast_2d(lab2))
        
        if l1.shape[-1] != 3 or l2.shape[-1] != 3:
            raise ValueError(f"Inputs must have shape (N, 3), got {l1.shape} and {l2.shape}")

        if l1.shape[0] != l2.shape[0]:
            # broadcast_to creates a view; ascontiguousarray materialises it
            # into a dense C-contiguous array suitable for Numba prange.
            if l1.shape[0] == 1: l1 = np.ascontiguousarray(np.broadcast_to(l1, l2.shape))
            elif l2.shape[0] == 1: l2 = np.ascontiguousarray(np.broadcast_to(l2, l1.shape))
            else: raise ValueError(f"Shapes {l1.shape} and {l2.shape} are not broadcastable.")
        return l1, l2

    @staticmethod
    def delta_E_2000(lab1: ArrayFloat, lab2: ArrayFloat, 
                     k_L: float = 1.0, k_C: float = 1.0, k_H: float = 1.0,
                     textiles: bool = False) -> ArrayFloat:
        """
        Calculates CIEDE2000 Color Difference.
        
        Args:
            lab1: Reference colors, shape (N, 3) or (3,).
            lab2: Sample colors, shape (N, 3) or (3,).
            k_L: Parametric lightness weight (default 1.0).
            k_C: Parametric chroma weight (default 1.0).
            k_H: Parametric hue weight (default 1.0).
            textiles: If True, overrides k_L=2.0, k_C=1.0, k_H=1.0 as per
                      CIE recommendation for textile applications.
            
        Returns:
            DeltaE 2000 values. Supports broadcasting (e.g., 1 vs N).
        """
        if textiles:
            k_L, k_C, k_H = 2.0, 1.0, 1.0
        l1, l2 = ColorMetrics._prepare_inputs(lab1, lab2)
        res = _batch_delta_e_2000(l1, l2, k_L, k_C, k_H)
        if lab1.ndim == 1 and lab2.ndim == 1: return res[0]
        return res

    @staticmethod
    def delta_E_76(lab1: ArrayFloat, lab2: ArrayFloat) -> ArrayFloat:
        """
        Calculates CIE Delta E 1976 (Euclidean distance in Lab).
        
        Args:
            lab1: Reference colors.
            lab2: Sample colors.
        """
        l1, l2 = ColorMetrics._prepare_inputs(lab1, lab2)
        res = _batch_delta_e_76(l1, l2)
        if lab1.ndim == 1 and lab2.ndim == 1: return res[0]
        return res

    @staticmethod
    def delta_E_DIN99(lab1: ArrayFloat, lab2: ArrayFloat, textiles: bool = False) -> ArrayFloat:
        """
        Calculates DIN99 Color Difference (DIN 6176).
        
        Args:
            lab1: Reference colors.
            lab2: Sample colors.
            textiles: If True, uses kE=2.0, kCH=0.5 optimized for textile industry.
        """
        l1, l2 = ColorMetrics._prepare_inputs(lab1, lab2)
        # Constants from DIN 6176
        kE = 2.0 if textiles else 1.0
        kCH = 0.5 if textiles else 1.0
        res = _din99_kernel(l1, l2, kE, kCH)
        if lab1.ndim == 1 and lab2.ndim == 1: return res[0]
        return res

    @staticmethod
    def delta_E_94(lab1: ArrayFloat, lab2: ArrayFloat, 
                   textiles: bool = False,
                   k_L: float = 1.0, K1: float = 0.045, K2: float = 0.015) -> ArrayFloat:
        """
        Calculates CIE 1994 Color Difference (CIE Publication 116-1995).
        
        Note: This metric is **asymmetric** — lab1 is the *reference* and lab2
        is the *sample*.  Swapping them may give a different result.
        
        Args:
            lab1: Reference colors, shape (N, 3) or (3,).
            lab2: Sample colors, shape (N, 3) or (3,).
            textiles: If True, overrides k_L=2.0, K1=0.048, K2=0.014
                      (textile industry parameters).
            k_L: Lightness parametric factor (default 1.0 for graphic arts).
            K1: Chroma parametric factor (default 0.045 for graphic arts).
            K2: Hue parametric factor (default 0.015 for graphic arts).
            
        Returns:
            DeltaE 94 values.
        """
        if textiles:
            k_L, K1, K2 = 2.0, 0.048, 0.014
        l1, l2 = ColorMetrics._prepare_inputs(lab1, lab2)
        res = _batch_delta_e_94(l1, l2, k_L, K1, K2)
        if lab1.ndim == 1 and lab2.ndim == 1: return res[0]
        return res

    @staticmethod
    def delta_E_CMC(lab1: ArrayFloat, lab2: ArrayFloat, 
                    pl: float = 2.0, pc: float = 1.0) -> ArrayFloat:
        """
        Calculates CMC l:c (1984) Color Difference.
        
        The CMC metric was developed by the Colour Measurement Committee of
        the Society of Dyers and Colourists and is widely used in textiles.
        
        Note: Like CIE 1994, this metric is **asymmetric** — lab1 is the
        *reference* (standard) and lab2 is the *sample* (batch).
        
        Args:
            lab1: Reference colors, shape (N, 3) or (3,).
            lab2: Sample colors, shape (N, 3) or (3,).
            pl: Lightness parametric factor (default 2.0 for acceptability,
                use 1.0 for imperceptibility).
            pc: Chroma parametric factor (default 1.0).
            
        Returns:
            DeltaE CMC values.
        """
        l1, l2 = ColorMetrics._prepare_inputs(lab1, lab2)
        res = _batch_delta_e_cmc(l1, l2, pl, pc)
        if lab1.ndim == 1 and lab2.ndim == 1: return res[0]
        return res


# =============================================================================
# 6. UNIFIED SPECTRAL PIPELINE
# =============================================================================

class SpectralPipeline:
    @staticmethod
    def spectral_to_srgb(spd: ArrayFloat, cmfs: ArrayFloat, illuminant: ArrayFloat, interval: float = 1.0, apply_adaptation: bool = True) -> ArrayFloat:
        """
        Calculates sRGB color from Spectral Power Distribution.
        
        Normalization Convention:
            The integration uses the standard CIE Y-normalisation factor:
            
                k = 1 / Σ(E(λ) · ȳ(λ) · Δλ)
            
            This produces *relative* XYZ where the perfect reflecting
            diffuser under the given illuminant maps to Y = 1.0 (i.e.
            the white point is [Xn, 1.0, Zn]).  This is the convention
            expected by the downstream Lab/sRGB transforms in this engine.
            
            If you need *absolute* XYZ (e.g. for luminance in cd/m²),
            multiply the result of the spectral integration by the desired
            absolute scaling factor before passing to colour-space transforms.
        
        Args:
            spd: Spectral Power Distribution. 
                 Shape (N_waves,) or (N_samples, N_waves).
            cmfs: Color Matching Functions. Shape (N_waves, 3).
            illuminant: Illuminant SPD. Shape (N_waves,).
            interval: Wavelength step size (nm).
            apply_adaptation: If True, adapts from illuminant white to D65.
            
        Returns:
            sRGB coordinates (N, 3) or (3,).
        """
        # Strict Shape Validation for CMFs
        if cmfs.ndim != 2 or cmfs.shape[1] != 3:
            raise ValueError(f"CMFs must be shape (N_waves, 3), got {cmfs.shape}.")
        if spd.shape[-1] != cmfs.shape[0]:
            raise ValueError(f"SPD dimension mismatch. Expected last dim {cmfs.shape[0]}, got {spd.shape[-1]}")

        denom = np.sum(illuminant * cmfs[:, 1] * interval)
        # Robust denominator check
        k_factor = 1.0 / denom if abs(denom) > 1e-12 else 1.0
        
        # 1. Calculate weights for the sample (SPD * CMFs * Illuminant * k)
        weights = cmfs * illuminant[:, np.newaxis] * k_factor * interval
        # Dot product handles batching: (N, W) dot (W, 3) -> (N, 3)
        xyz = np.dot(spd, weights)
        
        if apply_adaptation:
            # 2. Calculate White Point of the Illuminant
            # Use raw illuminant SPD * CMFs, then normalize Y to 1.0
            raw_white_xyz = np.sum(cmfs * illuminant[:, np.newaxis] * interval, axis=0)
            source_white = raw_white_xyz / raw_white_xyz[1] if raw_white_xyz[1] > 1e-12 else raw_white_xyz
            xyz = ChromaticAdaptation.adapt(xyz, source_white, REF_WHITE_D65)
        
        return ColorSpaceEngine.xyz_to_srgb(xyz)

# =============================================================================
# 7. PHOTOMETRY ENGINE (Numba Optimized - Branchless)
# =============================================================================

@njit(float64(float64[:], float64[:], float64[:], float64, float64, float64), cache=True, fastmath=True)
def _fast_flux_kernel(spd: ArrayFloat, vp: ArrayFloat, vs: ArrayFloat, w_p: float, w_s: float, interval: float) -> float:
    """
    Branchless fused kernel. 
    Calculates: Sum(SPD * (Vp*Wp + Vs*Ws)) * interval
    """
    n = spd.shape[0]
    total = 0.0
    # SIMD-friendly loop: No branches, just fused multiply-adds
    for i in range(n):
        total += spd[i] * ((vp[i] * w_p) + (vs[i] * w_s))
    return total * interval

class PhotometryEngine:
    """High-performance calculator for Photopic, Scotopic, and Mesopic Luminous Flux."""

    def __init__(self, v_photopic: ArrayFloat, v_scotopic: ArrayFloat, km_p: float = 683.002, km_s: float = 1700.05):
        # Enforce 1D contiguous arrays for max Numba speed
        self.vp = np.ascontiguousarray(v_photopic, dtype=np.float64).ravel()
        self.vs = np.ascontiguousarray(v_scotopic, dtype=np.float64).ravel()
        self.km_p, self.km_s = km_p, km_s
        if self.vp.shape[0] != self.vs.shape[0]: raise ValueError("Curve length mismatch")

    def calculate_flux(self, spd: ArrayFloat, vision_type: str = "photopic", m: float = 1.0, interval: float = 1.0) -> float:
        """
        Calculates Luminous Flux (lumens).
        
        Args:
            spd: Spectral Power Distribution (1D array).
            vision_type: 'photopic', 'scotopic', or 'mesopic'.
            m: Mesopic adaptation factor (0.0=scotopic, 1.0=photopic).
            interval: Wavelength interval in nm.
        """
        s = np.ascontiguousarray(spd, dtype=np.float64)
        if s.ndim != 1: raise ValueError("SPD must be 1D")
        
        # Determine Weights in Python (Cheap, runs once)
        w_p, w_s = 0.0, 0.0
        if vision_type == "photopic": w_p = self.km_p
        elif vision_type == "scotopic": w_s = self.km_s
        elif vision_type == "mesopic": w_p, w_s = m * self.km_p, (1.0 - m) * self.km_s
        else: raise ValueError(f"Unknown vision type: {vision_type}")
        
        # Call Branchless Kernel (Fast, SIMD optimized)
        return _fast_flux_kernel(s, self.vp, self.vs, w_p, w_s, interval)

    def calculate_sp_ratio(self, spd: ArrayFloat, interval: float = 1.0) -> float:
        """Calculates the S/P Ratio (Scotopic / Photopic)."""
        s = np.ascontiguousarray(spd, dtype=np.float64)
        photopic = _fast_flux_kernel(s, self.vp, self.vs, self.km_p, 0.0, interval)
        if photopic < 1e-12: return 0.0
        scotopic = _fast_flux_kernel(s, self.vp, self.vs, 0.0, self.km_s, interval)
        return scotopic / photopic

# =============================================================================
# Validation Block
# =============================================================================
if __name__ == "__main__":
    print("--- PADDOL Color Engine GOLD Master Validation (Final Release + Extended) ---")
    
    # 1. Round-Trip Invariant Test (XYZ -> Lab -> XYZ)
    print("1. Testing Round-Trip Stability (XYZ->Lab)...")
    xyz_in = np.random.rand(1000, 3)
    lab = ColorSpaceEngine.xyz_to_lab(xyz_in)
    xyz_out = ColorSpaceEngine.lab_to_xyz(lab)
    max_err = np.max(np.abs(xyz_in - xyz_out))
    # Standard float64 machine epsilon is approx 2.22e-16
    print(f"   Max Error (XYZ->Lab->XYZ): {max_err:.2e} " 
          f"{'[PASS]' if max_err < 1e-14 else '[FAIL]'}")

    # 2. Shape Safety Test
    print("2. Testing Shape Safety...")
    try:
        bad_shape = np.zeros((10, 5))
        ColorSpaceEngine.srgb_to_xyz(bad_shape)
    except ValueError as e:
        print(f"   Caught expected error: {e}")

    # 3. Spectral Workflow Test
    print("3. Testing Spectral -> sRGB with Adaptation...")
    wavelengths = 81 
    spd = np.ones((1, wavelengths)) * 0.5 
    cmfs = np.random.rand(wavelengths, 3) 
    illum = np.linspace(0.5, 1.5, wavelengths) 
    
    srgb = SpectralPipeline.spectral_to_srgb(spd, cmfs, illum, interval=5.0)
    print(f"   Resulting sRGB: {srgb[0]}")

    # 4. Cache Verification
    print("4. Testing Matrix Cache...")
    w1 = np.array([0.95, 1.0, 1.08])
    w2 = np.array([0.96, 1.0, 0.82])
    # First call - cache miss
    ChromaticAdaptation.calc_transform_matrix(w1, w2)
    # Second call - cache hit
    ChromaticAdaptation.calc_transform_matrix(w1, w2)
    info = _get_cached_bradford_matrix.cache_info()
    print(f"   Cache Info: {info}")
    if info.hits >= 1:
        print("   [PASS] Cache is working.")
    else:
        print("   [FAIL] Cache not hitting.")

    # 5. Robustness Verification (Inverse Lab)
    print("5. Testing Inverse Lab Precision...")
    # Test near discontinuity threshold (delta = 6/29 ~ 0.2069)
    # f(t) threshold is epsilon ~ 0.008856
    t_vals = np.array([0.008856, 0.008857, 0.2, 0.21], dtype=np.float64)
    # Check if roundtrip holds near critical points
    lab_crit = _xyz_to_lab_f(t_vals)
    t_back = _lab_to_xyz_f_inv(lab_crit)
    inv_err = np.max(np.abs(t_vals - t_back))
    print(f"   Inverse Function Error near delta: {inv_err:.2e}")
    
    # 6. Edge Case: Zeros and Negative Inputs (HDR)
    print("6. Testing Edge Cases (HDR / Zeros)...")
    zeros = np.zeros(3)
    xyz_zeros = ColorSpaceEngine.srgb_to_xyz(zeros)
    print(f"   Zero RGB -> XYZ: {xyz_zeros} (Expected ~0)")
    
    # Negative input (allowed if clip_negative=False)
    neg_xyz = np.array([-0.1, 0.5, 0.5])
    adapt_res = ChromaticAdaptation.adapt(neg_xyz, REF_WHITE_D65, REF_WHITE_D50, clip_negative=False)
    if np.any(adapt_res < 0):
        print("   [PASS] Negative values preserved when clip_negative=False")
    else:
        print("   [FAIL] Negative values clipped unexpectedly")
        
    # 7. CIELUV Round-Trip
    print("7. Testing CIELUV Round-Trip...")
    luv = ColorSpaceEngine.xyz_to_luv(xyz_in)
    xyz_luv_out = ColorSpaceEngine.luv_to_xyz(luv)
    max_err_luv = np.max(np.abs(xyz_in - xyz_luv_out))
    print(f"   Max Error (XYZ->Luv->XYZ): {max_err_luv:.2e} "
          f"{'[PASS]' if max_err_luv < 1e-13 else '[FAIL]'}")

    # 8. LCh Round-Trip
    print("8. Testing LCh Round-Trip...")
    lch = ColorSpaceEngine.lab_to_lch(lab)
    lab_back = ColorSpaceEngine.lch_to_lab(lch)
    max_err_lch = np.max(np.abs(lab - lab_back))
    print(f"   Max Error (Lab->LCh->Lab): {max_err_lch:.2e} "
          f"{'[PASS]' if max_err_lch < 1e-12 else '[FAIL]'}")

    # --- EXTENDED TESTS ---

    # 9. Oklab Direct XYZ Round-Trip
    print("9. Testing Oklab (Direct XYZ) Round-Trip...")
    oklab = ColorSpaceEngine.xyz_to_oklab(xyz_in)
    xyz_ok_out = ColorSpaceEngine.oklab_to_xyz(oklab)
    max_err_ok = np.max(np.abs(xyz_in - xyz_ok_out))
    print(f"   Max Error (XYZ->Oklab->XYZ): {max_err_ok:.2e} "
          f"{'[PASS]' if max_err_ok < 1e-14 else '[FAIL]'}")

    # 10. CIE 1964 U*V*W* Round-Trip
    print("10. Testing CIE 1964 U*V*W* Round-Trip...")
    uvw = ColorSpaceEngine.xyz_to_uvw(xyz_in)
    xyz_uvw_out = ColorSpaceEngine.uvw_to_xyz(uvw)
    max_err_uvw = np.max(np.abs(xyz_in - xyz_uvw_out))
    print(f"   Max Error (XYZ->UVW->XYZ): {max_err_uvw:.2e} "
          f"{'[PASS]' if max_err_uvw < 1e-13 else '[FAIL]'}")

    # 11. CIE 1960 UCS Round-Trip
    print("11. Testing CIE 1960 UCS Round-Trip...")
    ucs = ColorSpaceEngine.xyz_to_ucs(xyz_in)
    xyz_ucs_out = ColorSpaceEngine.ucs_to_xyz(ucs)
    max_err_ucs = np.max(np.abs(xyz_in - xyz_ucs_out))
    # UCS is a linear transform of XYZ, so error should be extremely low
    print(f"   Max Error (XYZ->UCS->XYZ): {max_err_ucs:.2e} "
          f"{'[PASS]' if max_err_ucs < 1e-14 else '[FAIL]'}")

    # 12. CIE 1960 Chromaticity uv
    print("12. Testing CIE 1960 uv Chromaticity...")
    # We check if v = 2/3 v'
    uv_1960 = ColorSpaceEngine.xyz_to_ucs_uv(xyz_in)
    
    # Manual verification formula: u=4X/denom, v=6Y/denom
    X, Y, Z = xyz_in[:, 0], xyz_in[:, 1], xyz_in[:, 2]
    denom = X + 15*Y + 3*Z
    u_calc = 4*X / denom
    v_calc = 6*Y / denom
    
    # Combine and compare
    uv_manual = np.stack((u_calc, v_calc), axis=1)
    # Handle potentially zero denominators (though random gen makes this unlikely)
    uv_manual[np.abs(denom) < 1e-12] = 0.0
    
    err_uv = np.max(np.abs(uv_1960 - uv_manual))
    print(f"   Max Error (Formula Check): {err_uv:.2e} "
          f"{'[PASS]' if err_uv < 1e-14 else '[FAIL]'}")

    # 13. Extended Metrics Test (DE76, DIN99)
    print("13. Testing Extended Metrics (DE76, DIN99)...")
    lab_ref = np.array([[50.0, 0.0, 0.0]])
    lab_sam = np.array([[55.0, 0.0, 0.0]]) # dL=5.0
    
    de76_val = ColorMetrics.delta_E_76(lab_ref, lab_sam)[0]
    print(f"   DE76 (dL=5): {de76_val:.4f} (Expected: 5.0000)")
    
    de99_val = ColorMetrics.delta_E_DIN99(lab_ref, lab_sam, textiles=True)[0]
    print(f"   DE99 (Textile): {de99_val:.4f}")

    # 14. Stress Test Benchmark (Batch 10M)
    print("14. Benchmarking Delta E 2000 (High Stress: 10M)...")
    N_bench = 10_000_000
    print(f"   Allocating {N_bench} pixels (~240MB)...")
    ref = np.random.rand(1, 3) 
    sam = np.random.rand(N_bench, 3)
    
    print("   Running computation...")
    t0 = time.perf_counter()
    _ = ColorMetrics.delta_E_2000(ref, sam)
    t1 = time.perf_counter()
    print(f"   Processed {N_bench:,} pixels in {(t1-t0)*1000:.2f} ms")

    # --- v2.1 NEW FEATURE TESTS ---

    # 15. Clip parameter (Fix #3)
    print("15. Testing clip=False (HDR passthrough)...")
    hdr_rgb = np.array([[1.2, -0.1, 0.5]])
    xyz_clipped = ColorSpaceEngine.srgb_to_xyz(hdr_rgb, clip=True)
    xyz_unclipped = ColorSpaceEngine.srgb_to_xyz(hdr_rgb, clip=False)
    if np.allclose(xyz_clipped, xyz_unclipped):
        print("   [FAIL] clip=False should produce different output for out-of-range input")
    else:
        print(f"   [PASS] clip=True: {xyz_clipped[0]}")
        print(f"          clip=False: {xyz_unclipped[0]}")

    # 16. CIEDE2000 parametric weights (Enhancement #1)
    print("16. Testing CIEDE2000 parametric weights...")
    de00_std = ColorMetrics.delta_E_2000(lab_ref, lab_sam)
    de00_tex = ColorMetrics.delta_E_2000(lab_ref, lab_sam, textiles=True)
    de00_kl2 = ColorMetrics.delta_E_2000(lab_ref, lab_sam, k_L=2.0)
    print(f"   DE2000 (standard):  {de00_std[0]:.4f}")
    print(f"   DE2000 (textiles):  {de00_tex[0]:.4f}")
    print(f"   DE2000 (k_L=2.0):   {de00_kl2[0]:.4f}")
    if abs(de00_tex[0] - de00_kl2[0]) < 1e-10:
        print("   [PASS] textiles=True matches k_L=2.0 for pure dL case")
    else:
        print("   [FAIL] textiles and k_L=2.0 should match for pure dL")

    # 17. CIE 1994 Delta E (Enhancement #2)
    print("17. Testing CIE 1994 Delta E...")
    de94_val = ColorMetrics.delta_E_94(lab_ref, lab_sam)
    de94_tex = ColorMetrics.delta_E_94(lab_ref, lab_sam, textiles=True)
    print(f"   DE94 (graphic arts): {de94_val[0]:.4f} (Expected: 5.0000 for pure dL)")
    print(f"   DE94 (textiles):     {de94_tex[0]:.4f} (Expected: 2.5000 for dL=5, k_L=2)")

    # 18. CMC Delta E (Enhancement #2)
    print("18. Testing CMC l:c Delta E...")
    lab_a = np.array([[50.0, 25.0, 10.0]])
    lab_b = np.array([[55.0, 30.0, 15.0]])
    de_cmc_accept = ColorMetrics.delta_E_CMC(lab_a, lab_b, pl=2.0, pc=1.0)
    de_cmc_imperceptible = ColorMetrics.delta_E_CMC(lab_a, lab_b, pl=1.0, pc=1.0)
    print(f"   CMC 2:1 (acceptability):    {de_cmc_accept[0]:.4f}")
    print(f"   CMC 1:1 (imperceptibility): {de_cmc_imperceptible[0]:.4f}")
    if de_cmc_imperceptible[0] > de_cmc_accept[0]:
        print("   [PASS] pl=1.0 gives larger DE than pl=2.0 (correct)")
    else:
        print("   [FAIL] Unexpected CMC ratio")

    # 19. Strict IEEE mode (Fix #5)
    print("19. Testing Strict IEEE mode...")
    xyz_fast = ColorSpaceEngine.xyz_to_lab(xyz_in[:10])
    set_strict_ieee(True)
    xyz_strict = ColorSpaceEngine.xyz_to_lab(xyz_in[:10])
    set_strict_ieee(False)
    ieee_diff = np.max(np.abs(xyz_fast - xyz_strict))
    print(f"   Max diff (fast vs strict): {ieee_diff:.2e}")
    if ieee_diff < 1e-10:
        print("   [PASS] Fast and strict agree to high precision")
    else:
        print(f"   [INFO] Difference detected (expected with fastmath reassociation)")

    # 20. _raw fast-path validation (Fix #1)
    print("20. Testing _raw fast-path consistency...")
    test_rgb = np.random.rand(100, 3)
    # Public API (goes through handle_shapes)
    lab_pub = ColorSpaceEngine.srgb_to_lab(test_rgb)
    # Manual _raw path (what the convenience method now does internally)
    test_2d = np.ascontiguousarray(np.atleast_2d(test_rgb))
    xyz_raw = ColorSpaceEngine._srgb_to_xyz_raw(test_2d)
    lab_raw = ColorSpaceEngine._xyz_to_lab_raw(xyz_raw)
    raw_err = np.max(np.abs(lab_pub - lab_raw))
    print(f"   Max Error (public vs _raw): {raw_err:.2e} "
          f"{'[PASS]' if raw_err < 1e-15 else '[FAIL]'}")

    print("--- Validation Complete ---")