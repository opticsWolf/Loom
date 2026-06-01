# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Loom Eigenmode Solver
=====================
Finds the guided eigenmodes of a multilayer optical structure by locating
poles of the S-matrix reflection coefficient in the complex effective-index
(or k_x) plane.

Physical background
-------------------
A guided mode satisfies the self-consistency (resonance) condition:

    D(β) = 1 − r_back_A · r_front_B = 0

where β = k_x = N_eff · (2π/λ) is the in-plane wavevector. This is exactly
the Redheffer denominator that already appears in loom_matrix.py. A mode
lives at a pole of the total reflection coefficient, i.e. a zero of the
characteristic function

    f(N_eff) = 1 / r_front_total(N_eff)   →  |f| is minimised

Strategy
--------
1. Build a 2-D scan of |r_front(N_eff)| over a complex N_eff grid to map
   the pole landscape (guided modes have Re(N_eff) > max(n_ambient),
   Im(N_eff) ≥ 0 for passive media).
2. Use `scipy.optimize.minimize` with a Nelder-Mead polisher on each
   candidate found in the coarse scan to refine to machine precision.
3. Return an EigenmodeResult dataclass with per-mode effective index,
   propagation loss, field profiles, and group index (via finite difference).

Sign conventions match loom_matrix.py throughout (admittance Fresnel,
Im(cos θ) ≥ 0 branch cut, Poynting-flux-correct intensities).

Usage example
-------------
    import numpy as np
    from loom_eigenmode import LoomEigenmodeSolver

    # Simple symmetric slab: air / Si (n=3.5, d=200 nm) / air
    n_wavs = 1
    lam = np.array([1550.0])                      # nm
    n_air = np.full(n_wavs, 1.0   + 0j)
    n_si  = np.full(n_wavs, 3.476 + 0j)

    layer_indices = np.vstack([n_air, n_si, n_air])  # shape (3, n_wavs)
    thicknesses   = np.array([0.0, 200.0, 0.0])
    inc_flags     = np.zeros(3, dtype=np.int32)
    r_types       = np.zeros(3, dtype=np.int32)
    r_vals        = np.zeros(3)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, lam
    )
    modes = solver.find_modes(lam_idx=0, pol='both')
    for m in modes:
        print(m)
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings


# ─── Optional Rust backend ───────────────────────────────────────────────────
# If the compiled extension is importable we route the three hot phases
# (coarse scan, candidate detection, Nelder–Mead polish) and the field profile
# through it. Each call does a whole *phase* in one FFI crossing — there is no
# per-evaluation boundary crossing — so the parallel (rayon) scan and the
# allocation-free polish are the actual win. If the module is missing we fall
# back to the pure-NumPy/SciPy path below, byte-for-byte the original behaviour.

try:
    from navette_smatrix import (
        scan_landscape as _rs_scan_landscape,
        find_local_minima as _rs_find_local_minima,
        nelder_mead as _rs_nelder_mead,
        field_profile as _rs_field_profile,
    )
    _RUST_OK = True
    _RUST_PRESENT = True
except ImportError:
    _RUST_OK = False
    _RUST_PRESENT = False


def rust_backend_available() -> bool:
    """True if mode-finding is using the compiled navette_smatrix backend."""
    return _RUST_OK


def use_rust_backend(flag: bool) -> bool:
    """Force the backend on/off (e.g. for an honest Rust-vs-Python A/B).

    Returns the value actually set: requesting the Rust path while the
    extension is unimportable stays False.
    """
    global _RUST_OK
    if flag and not _RUST_PRESENT:
        _RUST_OK = False
    else:
        _RUST_OK = bool(flag)
    return _RUST_OK


def _as_rust_inputs(n_stk, thicknesses, r_types, r_vals):
    """Coerce solver arrays into the dtypes/layout the extension requires.

    The column slice ``layer_indices[:, k]`` is non-contiguous and the solver
    stores roughness types as int64; the PyO3 signatures need a C-contiguous
    complex128 stack and an int32 type array.
    """
    return (
        np.ascontiguousarray(n_stk, dtype=np.complex128),
        np.ascontiguousarray(thicknesses, dtype=np.float64),
        np.ascontiguousarray(r_types, dtype=np.int32),
        np.ascontiguousarray(r_vals, dtype=np.float64),
    )


def _scan_landscape_backend(n_stk, thicknesses, r_types, r_vals, lam, p_int,
                            real_range, imag_range, scan_points):
    """Return (Nr, Ni, landscape) of raw |1/r|^2 values (NOT log-scaled)."""
    if _RUST_OK:
        n_c, th_c, rt_c, rv_c = _as_rust_inputs(n_stk, thicknesses, r_types, r_vals)
        Nr, Ni, land = _rs_scan_landscape(
            n_c, th_c, rt_c, rv_c, lam, int(p_int),
            float(real_range[0]), float(real_range[1]),
            float(imag_range[0]), float(imag_range[1]),
            int(scan_points), int(scan_points),
        )
        return np.asarray(Nr, float), np.asarray(Ni, float), np.asarray(land, float)

    # Pure-Python fallback (identical grid to np.linspace).
    Nr = np.linspace(real_range[0], real_range[1], scan_points)
    Ni = np.linspace(imag_range[0], imag_range[1], scan_points)
    args = (n_stk, thicknesses, r_types, r_vals, lam, p_int)
    land = np.zeros((len(Ni), len(Nr)))
    for i, ni in enumerate(Ni):
        for j, nr in enumerate(Nr):
            land[i, j] = _char_func_xy([nr, ni], *args)
    return Nr, Ni, land


def _find_local_minima_backend(landscape, Nr, Ni):
    """Return list of (Re, Im) candidate seeds. Threshold = median (factor 1.0)."""
    if _RUST_OK:
        land_c = np.ascontiguousarray(landscape, dtype=np.float64)
        return list(_rs_find_local_minima(
            land_c, np.asarray(Nr, float), np.asarray(Ni, float), 1.0))

    # Pure-Python fallback: strictly-smaller-than-neighbours, below the median,
    # skipping the first/last real columns (lossless modes on the Im=0 edge are
    # still picked up because every imag row is scanned).
    median_val = float(np.median(landscape))
    candidates = []
    n_imag, n_real = landscape.shape

    def _is_local_min(i, j):
        v = landscape[i, j]
        if v >= median_val:
            return False
        i0 = max(i - 1, 0); i1 = min(i + 1, n_imag - 1)
        j0 = max(j - 1, 0); j1 = min(j + 1, n_real - 1)
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                if ii == i and jj == j:
                    continue
                if landscape[ii, jj] <= v:
                    return False
        return True

    for i in range(n_imag):
        for j in range(1, n_real - 1):
            if _is_local_min(i, j):
                candidates.append((Nr[j], Ni[i]))
    return candidates


def _polish_backend(n_stk, thicknesses, r_types, r_vals, lam, p_int,
                    x0, step, tol, max_iter):
    """Refine one candidate. Returns (Re, Im, char_value)."""
    if _RUST_OK:
        n_c, th_c, rt_c, rv_c = _as_rust_inputs(n_stk, thicknesses, r_types, r_vals)
        xr, xi, cv = _rs_nelder_mead(
            n_c, th_c, rt_c, rv_c, lam, int(p_int),
            (float(x0[0]), float(x0[1])),
            float(step), float(tol), int(max_iter),
        )
        return float(xr), float(xi), float(cv)

    args = (n_stk, thicknesses, r_types, r_vals, lam, p_int)
    res = minimize(
        _char_func_xy, x0=[x0[0], x0[1]], args=args, method='Nelder-Mead',
        options={
            'xatol': tol, 'fatol': tol * 1e-3,
            'maxiter': max_iter, 'adaptive': True,
            'initial_simplex': np.array([
                [x0[0],        x0[1]],
                [x0[0] + step, x0[1]],
                [x0[0],        x0[1] + step * 0.1],
            ]),
        },
    )
    return float(res.x[0]), float(res.x[1]), float(res.fun)


def _field_profile_backend(n_stk, thicknesses, r_types, r_vals, lam, N_eff,
                           p_int, n_points_per_layer):
    """Return (z, |E|, layer_bounds) where layer_bounds = [(z0, z1, n), ...]."""
    if _RUST_OK:
        n_c, th_c, rt_c, rv_c = _as_rust_inputs(n_stk, thicknesses, r_types, r_vals)
        z, e_mag, lstart, lend, ln = _rs_field_profile(
            n_c, th_c, rt_c, rv_c, lam, complex(N_eff), int(p_int),
            int(n_points_per_layer),
        )
        z = np.asarray(z, float)
        e_mag = np.asarray(e_mag, float)
        layer_bounds = [(float(a), float(b), complex(c))
                        for a, b, c in zip(lstart, lend, ln)]
        return z, e_mag, layer_bounds

    return _field_profile(n_stk, thicknesses, r_types, r_vals, lam, N_eff,
                          p_int, n_points_per_layer)


# ─── Reuse low-level helpers from loom_matrix ────────────────────────────────
# We re-implement the scalar (single-wavelength, single-angle) S-matrix
# accumulation in pure NumPy so this module works standalone without Numba
# compilation delays.  A Numba-accelerated version could be drop-in substituted.

_LOG_MIN = 1e-100
_DBL_EPS = 2.22e-16


def _redheffer(ra_rf, ra_tb, ra_tf, ra_rb,
               rb_rf, rb_tb, rb_tf, rb_rb):
    """Complex Redheffer star product (field amplitudes)."""
    denom = 1.0 - ra_rb * rb_rf
    if abs(denom) < _LOG_MIN:
        phase = denom / (abs(denom) + 1e-300)
        denom = _LOG_MIN * phase + 1e-300
    inv = 1.0 / denom
    s_rf = ra_rf + ra_tb * rb_rf * ra_tf * inv
    s_tb = ra_tb * rb_tb * inv
    s_tf = rb_tf * ra_tf * inv
    s_rb = rb_rb + rb_tf * ra_rb * rb_tb * inv
    return s_rf, s_tb, s_tf, s_rb


def _stack_reflection(
    n_stack: np.ndarray,       # complex refractive indices (n_layers,)
    d_stack: np.ndarray,       # thicknesses (n_layers,), ambient & substrate = 0
    rough_types: np.ndarray,   # int (n_layers,)
    rough_vals: np.ndarray,    # float (n_layers,)
    lam: float,                # wavelength (same units as d_stack)
    N_eff: complex,            # in-plane wavevector invariant  N·sin(θ)
    pol: int,                  # 0=s, 1=p
) -> complex:
    """
    Compute the front-face field reflection coefficient r of the entire stack
    for a given complex effective index N_eff = N0·sin(θ0).

    Returns r_front (complex).  The mode condition is |1/r_front| → 0.
    """
    n_layers = len(n_stack)

    # ── initialise identity S-matrix ──────────────────────────────────────
    sg_rf, sg_tb, sg_tf, sg_rb = 0.0+0j, 1.0+0j, 1.0+0j, 0.0+0j
    two_pi_lam = 2.0 * np.pi / lam

    def _admittance(N, cos_th):
        if pol == 0:                        # s
            return N * cos_th
        else:                               # p
            if abs(cos_th) < 1e-12:
                cos_th = 1e-12 + 0j
            return N / cos_th

    def _safe_cos(N):
        val = 1.0 - (N_eff / N) ** 2
        c = np.sqrt(val.astype(complex))
        if c.imag < 0.0:
            c = -c
        return c

    N_curr  = n_stack[0]
    cos_curr = _safe_cos(N_curr)
    Y_curr   = _admittance(N_curr, cos_curr)

    for idx in range(n_layers - 1):
        N_next  = n_stack[idx + 1]
        cos_next = _safe_cos(N_next)
        Y_next   = _admittance(N_next, cos_next)

        # ── Fresnel coefficients ──────────────────────────────────────────
        den = Y_curr + Y_next
        if abs(den) < _LOG_MIN:
            den = _LOG_MIN * (1.0 + 1.0j)
        inv_den = 1.0 / den

        r12 = (Y_curr - Y_next) * inv_den
        r21 = -r12
        t12 = 2.0 * Y_curr * inv_den
        t21 = 2.0 * Y_next * inv_den

        # ── Roughness ────────────────────────────────────────────────────
        sigma  = rough_vals[idx + 1]
        rtype  = rough_types[idx + 1]
        if rtype != 0 and sigma > 0:
            kz1 = two_pi_lam * N_curr * cos_curr
            kz2 = two_pi_lam * N_next * cos_next
            if rtype == 5:                  # Névot-Croce
                nc = np.exp(-2.0 * kz1 * kz2 * sigma * sigma)
                r12 *= nc; r21 *= nc; t12 *= nc; t21 *= nc
            else:
                al = _w_function(2.0 * kz1 * sigma, rtype)
                be = _w_function(2.0 * kz2 * sigma, rtype)
                ga = _w_function((kz1 - kz2) * sigma, rtype)
                r12 *= al; r21 *= be; t12 *= ga; t21 *= ga

        sg_rf, sg_tb, sg_tf, sg_rb = _redheffer(
            sg_rf, sg_tb, sg_tf, sg_rb,
            r12, t21, t12, r21
        )

        # ── Propagation phase ─────────────────────────────────────────────
        if idx + 1 < n_layers - 1:
            d = d_stack[idx + 1]
            if d > 1e-12:
                beta = two_pi_lam * d * N_next * cos_next
                if beta.imag < 0.0:
                    beta = complex(beta.real, -beta.imag)
                phi = np.exp(1j * beta)
                sg_rb *= phi * phi
                sg_tb *= phi
                sg_tf *= phi

        N_curr   = N_next
        cos_curr = cos_next
        Y_curr   = Y_next

    return sg_rf


_SQRT3 = 1.73205080757

def _w_function(q: complex, rough_type: int) -> complex:
    """Roughness form factor W(q) — mirrors loom_matrix.w_function."""
    if rough_type == 0:
        return 1.0 + 0j
    if rough_type == 1:
        val = q * _SQRT3
        if abs(val) < 1e-9:
            return 1.0 + 0j
        return np.sin(val) / val
    elif rough_type == 2:
        return np.cos(q)
    elif rough_type == 3:
        return 1.0 / (1.0 + q * q * 0.5)
    elif rough_type == 4:
        return np.exp(-q * q * 0.5)
    return 1.0 + 0j


# ─── Mode characteristic function ─────────────────────────────────────────────

def _char_func(N_eff: complex,
               n_stack, d_stack, rough_types, rough_vals,
               lam: float, pol: int) -> float:
    """
    |1 / r(N_eff)|^2 — minimise to find modes (zeros of 1/r = poles of r).

    Using 1/r rather than r avoids the trivial minimum at r=0 (anti-resonance).
    In the limit |r| → 1 the structure approaches perfect resonance.
    """
    r = _stack_reflection(n_stack, d_stack, rough_types, rough_vals,
                          lam, N_eff, pol)
    if abs(r) < 1e-15:
        return 1e30
    return (1.0 / abs(r)) ** 2


def _char_func_xy(xy, *args):
    """Real-valued wrapper: xy = [Re(N_eff), Im(N_eff)]."""
    N_eff = complex(xy[0], xy[1])
    return _char_func(N_eff, *args)


# ─── Field profile computation ────────────────────────────────────────────────

def _field_profile(
    n_stack: np.ndarray,
    d_stack: np.ndarray,
    rough_types: np.ndarray,
    rough_vals: np.ndarray,
    lam: float,
    N_eff: complex,
    pol: int,
    n_points_per_layer: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the electric-field amplitude profile |E(z)| through the stack.

    Strategy (rigorous left/right S-matrix decomposition):
      For each physical layer i, build:
        S_L = S-matrix of everything LEFT  of this layer (ambient -> left face)
        S_R = S-matrix of everything RIGHT of this layer (right face -> substrate)

      The forward/backward amplitudes just inside the layer are:
        E_plus  = t_fwd(S_L) / (1 - r_back(S_L)*r_front(S_R))
        E_minus = r_front(S_R) * E_plus

      Inside the layer the field is:
        E(z) = E_plus*exp(+ikz*z) + E_minus*exp(-ikz*z)
    """
    n_layers   = len(n_stack)
    two_pi_lam = 2.0 * np.pi / lam

    def _safe_cos(N):
        val = 1.0 - (N_eff / N) ** 2
        c = np.sqrt(complex(val))
        if c.imag < 0.0:
            c = -c
        return c

    def _admittance(N, cos_th):
        if pol == 0:
            return N * cos_th
        if abs(cos_th) < 1e-12:
            cos_th = 1e-12 + 0j
        return N / cos_th

    cos_arr = [_safe_cos(n_stack[i]) for i in range(n_layers)]
    Y_arr   = [_admittance(n_stack[i], cos_arr[i]) for i in range(n_layers)]

    def _interface_S(i):
        Y_a, Y_b = Y_arr[i], Y_arr[i + 1]
        den = Y_a + Y_b
        if abs(den) < _LOG_MIN:
            den = _LOG_MIN * (1.0 + 1.0j)
        inv = 1.0 / den
        r12 = (Y_a - Y_b) * inv;  r21 = -r12
        t12 = 2.0 * Y_a * inv;    t21 = 2.0 * Y_b * inv
        sigma = rough_vals[i + 1]; rtype = rough_types[i + 1]
        if rtype != 0 and sigma > 0:
            kz1 = two_pi_lam * n_stack[i]     * cos_arr[i]
            kz2 = two_pi_lam * n_stack[i + 1] * cos_arr[i + 1]
            if rtype == 5:
                nc = np.exp(-2.0 * kz1 * kz2 * sigma * sigma)
                r12 *= nc; r21 *= nc; t12 *= nc; t21 *= nc
            else:
                al = _w_function(2.0 * kz1 * sigma, rtype)
                be = _w_function(2.0 * kz2 * sigma, rtype)
                ga = _w_function((kz1 - kz2) * sigma, rtype)
                r12 *= al; r21 *= be; t12 *= ga; t21 *= ga
        return r12, t21, t12, r21

    def _prop_S(i):
        d = d_stack[i]
        if d < 1e-12:
            return 0.0+0j, 1.0+0j, 1.0+0j, 0.0+0j
        beta = two_pi_lam * d * n_stack[i] * cos_arr[i]
        if beta.imag < 0.0:
            beta = complex(beta.real, -beta.imag)
        phi = np.exp(1j * beta)
        return 0.0+0j, phi, phi, 0.0+0j

    # S_L[i]: everything left of layer i (i.e. up to and including interface i-1->i)
    S_L = [(0.0+0j, 1.0+0j, 1.0+0j, 0.0+0j)]   # identity before ambient
    for i in range(n_layers - 1):
        sl = S_L[-1]
        if d_stack[i] > 1e-12 and i > 0:
            sl = _redheffer(*sl, *_prop_S(i))
        sl = _redheffer(*sl, *_interface_S(i))
        S_L.append(sl)

    # S_R[i]: everything right of layer i (interface i->i+1 onwards)
    S_R = [None] * n_layers
    S_R[n_layers - 1] = (0.0+0j, 1.0+0j, 1.0+0j, 0.0+0j)
    for i in range(n_layers - 2, 0, -1):
        sr = S_R[i + 1]
        if d_stack[i + 1] > 1e-12:
            sr = _redheffer(*_prop_S(i + 1), *sr)
        sr = _redheffer(*_interface_S(i), *sr)
        S_R[i] = sr

    z_parts = []; E_parts = []; layer_bounds = []
    z_cursor = 0.0

    for i in range(1, n_layers - 1):
        d = d_stack[i]
        if d < 1e-12:
            continue
        sl = S_L[i]; sr = S_R[i]
        denom = 1.0 - sl[3] * sr[0]
        if abs(denom) < _LOG_MIN:
            denom = _LOG_MIN * (1.0 + 1.0j)
        E_plus  = sl[2] / denom
        E_minus = sr[0] * E_plus
        beta_d = two_pi_lam * d * n_stack[i] * cos_arr[i]
        if beta_d.imag < 0.0:
            beta_d = complex(beta_d.real, -beta_d.imag)
        zz = np.linspace(0.0, d, n_points_per_layer)
        xi = zz / d
        E_z = E_plus * np.exp(1j * beta_d * xi) + E_minus * np.exp(-1j * beta_d * xi)
        z_parts.append(z_cursor + zz)
        E_parts.append(np.abs(E_z))
        layer_bounds.append((z_cursor, z_cursor + d, n_stack[i]))
        z_cursor += d

    if not z_parts:
        return np.array([0.0]), np.array([1.0]), []
    z = np.concatenate(z_parts)
    Emag = np.concatenate(E_parts)
    if Emag.max() > 0:
        Emag /= Emag.max()
    return z, Emag, layer_bounds


# ─── Result container ──────────────────────────────────────────────────────────

@dataclass
class Eigenmode:
    """
    A single guided eigenmode of the multilayer stack.

    Attributes
    ----------
    pol         : Polarisation, 's' or 'p'.
    N_eff       : Complex effective index  N_eff = n_eff + i·k_eff.
                  Re(N_eff) is the phase index; Im(N_eff) the attenuation index.
    lam         : Free-space wavelength [same units as layer thicknesses].
    loss_dB_per_unit : Propagation loss  α = 4π·Im(N_eff)/λ  in dB per unit length.
    n_group     : Real part of the group index  n_g = n_eff − λ·(dn_eff/dλ),
                  estimated by finite difference. None if only one wavelength.
    z           : Depth coordinates for the field profile [same units as λ].
    E_profile   : |E(z)| field amplitude, normalised to max = 1.
    layer_bounds: List of (z_start, z_end, n_layer) for the physical layers.
    char_value  : Value of |1/r|² at the converged N_eff (goodness metric;
                  0 = perfect pole, large = not a real mode).
    """
    pol           : str
    N_eff         : complex
    lam           : float
    loss_dB_per_unit: float
    n_group       : Optional[float]
    z             : np.ndarray
    E_profile     : np.ndarray
    layer_bounds  : list
    char_value    : float

    def __repr__(self):
        return (
            f"Eigenmode(pol={self.pol}, λ={self.lam:.1f}, "
            f"N_eff={self.N_eff.real:.6f}{self.N_eff.imag:+.2e}j, "
            f"loss={self.loss_dB_per_unit:.3e} dB/unit, "
            f"n_g={self.n_group})"
        )


# ─── Main solver class ─────────────────────────────────────────────────────────

class LoomEigenmodeSolver:
    """
    Eigenmode solver for Loom multilayer optical structures.

    Finds guided modes by scanning the complex effective-index plane and
    polishing candidate minima of |1/r(N_eff)|² to locate exact poles.

    Parameters
    ----------
    layer_indices   : complex ndarray (n_layers, n_wavs) — same convention as
                      LoomScatterMatrix.  First row = ambient, last = substrate.
    thicknesses     : float ndarray (n_layers).
    incoherent_flags: int ndarray (n_layers).  Modes are found per coherent block;
                      incoherent layers are treated as additional absorbing slabs.
    roughness_types : int array (n_layers).
    roughness_values: float array (n_layers).
    wavls           : float ndarray — wavelengths to evaluate modes at.
    """

    def __init__(
        self,
        layer_indices: np.ndarray,
        thicknesses: np.ndarray,
        incoherent_flags: np.ndarray,
        roughness_types,
        roughness_values,
        wavls: np.ndarray,
    ):
        self.layer_indices = np.asarray(layer_indices, dtype=complex)
        self.thicknesses   = np.asarray(thicknesses,   dtype=float)
        self.inc_flags     = np.asarray(incoherent_flags, dtype=int)
        self.r_types       = np.asarray(roughness_types,  dtype=int)
        self.r_vals        = np.asarray(roughness_values, dtype=float)
        self.wavls         = np.asarray(wavls, dtype=float)

        if self.layer_indices.shape[0] != len(self.thicknesses):
            raise ValueError(
                "layer_indices.shape[0] must equal len(thicknesses). "
                f"Got {self.layer_indices.shape[0]} vs {len(self.thicknesses)}."
            )

    # ── public API ─────────────────────────────────────────────────────────

    def find_modes(
        self,
        lam_idx: int = 0,
        pol: str = 'both',
        N_eff_real_range: Optional[Tuple[float, float]] = None,
        N_eff_imag_range: Tuple[float, float] = (0.0, 0.02),
        scan_points: int = 120,
        tol: float = 1e-9,
        char_threshold: float = 0.1,
        n_points_per_layer: int = 100,
        compute_group_index: bool = True,
        delta_lam_nm: float = 1.0,
    ) -> List[Eigenmode]:
        """
        Find all guided eigenmodes at wavelength ``wavls[lam_idx]``.

        Parameters
        ----------
        lam_idx         : Index into ``self.wavls``.
        pol             : 's', 'p', or 'both'.
        N_eff_real_range: Search range for Re(N_eff). Defaults to
                          (max(Re(n_cladding)), max(Re(n_all_layers))).
        N_eff_imag_range: Search range for Im(N_eff).  (0, 0.02) covers most
                          low-loss guided modes.
        scan_points     : Number of scan points per axis in the coarse grid.
        tol             : Convergence tolerance for Nelder-Mead polishing.
        char_threshold  : Accept a mode if |1/r|² < this value after polishing.
                          Lower ⇒ stricter; raise for highly lossy structures.
        n_points_per_layer : Resolution of the field-profile z-grid per layer.
        compute_group_index: If True, estimate n_g via finite-difference dN/dλ.
        delta_lam_nm    : Wavelength step for group-index finite difference.

        Returns
        -------
        List of :class:`Eigenmode` objects, sorted by Re(N_eff) descending.
        """
        pols = []
        if pol in ('s', 'both'):
            pols.append(0)
        if pol in ('p', 'both'):
            pols.append(1)
        if not pols:
            raise ValueError("pol must be 's', 'p', or 'both'")

        lam   = float(self.wavls[lam_idx])
        n_stk = self.layer_indices[:, lam_idx]   # (n_layers,) at this λ

        # Determine search range for Re(N_eff)
        n_real_all = np.real(n_stk)
        n_cladding = max(n_real_all[0], n_real_all[-1])
        n_core_max = np.max(n_real_all)
        if N_eff_real_range is None:
            lo = n_cladding + 1e-4
            hi = n_core_max  - 1e-4
            if lo >= hi:
                warnings.warn(
                    f"No guiding possible at λ={lam}: n_cladding={n_cladding:.4f} "
                    f">= n_core_max={n_core_max:.4f}.  Returning empty list.",
                    stacklevel=2
                )
                return []
            N_eff_real_range = (lo, hi)

        all_modes: List[Eigenmode] = []

        for p in pols:
            pol_str = 's' if p == 0 else 'p'
            modes_p = self._find_modes_single_pol(
                n_stk, lam, lam_idx, p, pol_str,
                N_eff_real_range, N_eff_imag_range,
                scan_points, tol, char_threshold,
                n_points_per_layer,
                compute_group_index, delta_lam_nm,
            )
            all_modes.extend(modes_p)

        all_modes.sort(key=lambda m: -m.N_eff.real)
        return all_modes

    def scan_landscape(
        self,
        lam_idx: int = 0,
        pol: str = 's',
        N_eff_real_range: Optional[Tuple[float, float]] = None,
        N_eff_imag_range: Tuple[float, float] = (0.0, 0.02),
        scan_points: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the coarse |1/r(N_eff)|² landscape for visualisation.

        Returns
        -------
        N_real_grid : 1-D array of Re(N_eff) values
        N_imag_grid : 1-D array of Im(N_eff) values
        landscape   : 2-D array  [len(N_imag), len(N_real)], log₁₀(|1/r|²)
        """
        lam   = float(self.wavls[lam_idx])
        n_stk = self.layer_indices[:, lam_idx]
        p_int = 0 if pol == 's' else 1

        n_real_all = np.real(n_stk)
        n_cladding = max(n_real_all[0], n_real_all[-1])
        n_core_max = np.max(n_real_all)
        if N_eff_real_range is None:
            N_eff_real_range = (n_cladding + 1e-4, n_core_max - 1e-4)

        Nr, Ni, landscape = _scan_landscape_backend(
            n_stk, self.thicknesses, self.r_types, self.r_vals, lam, p_int,
            N_eff_real_range, N_eff_imag_range, scan_points,
        )
        # Public landscape is returned log-scaled for visualisation.
        landscape = np.log10(landscape + 1e-30)

        return Nr, Ni, landscape

    # ── private helpers ─────────────────────────────────────────────────────

    def _find_modes_single_pol(
        self, n_stk, lam, lam_idx, p_int, pol_str,
        N_eff_real_range, N_eff_imag_range,
        scan_points, tol, char_threshold,
        n_points_per_layer, compute_group_index, delta_lam_nm,
    ) -> List[Eigenmode]:

        # ── Coarse 2-D scan (parallel Rust backend when available) ────────
        Nr, Ni, landscape = _scan_landscape_backend(
            n_stk, self.thicknesses, self.r_types, self.r_vals, lam, p_int,
            N_eff_real_range, N_eff_imag_range, scan_points,
        )

        # ── Candidate detection (strict local minima below the median) ────
        candidates = _find_local_minima_backend(landscape, Nr, Ni)

        if not candidates:
            return []

        # ── Nelder-Mead polishing ─────────────────────────────────────────
        step = (Nr[1] - Nr[0]) * 0.5

        refined: List[Eigenmode] = []
        seen_N: List[complex] = []

        for x0_r, x0_i in candidates:
            xr, xi, cv = _polish_backend(
                n_stk, self.thicknesses, self.r_types, self.r_vals, lam, p_int,
                (x0_r, x0_i), step, tol, 5000,
            )

            if cv > char_threshold:
                continue                  # not a genuine mode

            N_eff = complex(xr, xi)

            # Guard: N_eff must lie in the physically sensible region
            if (N_eff.real < N_eff_real_range[0] - 1e-3 or
                N_eff.real > N_eff_real_range[1] + 1e-3 or
                N_eff.imag < -1e-6):       # slightly negative Im allowed for numerics
                continue

            # De-duplicate: merge modes closer than step/4 in N_eff
            duplicate = False
            for prev in seen_N:
                if abs(N_eff - prev) < step / 4:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen_N.append(N_eff)

            # ── Propagation loss ──────────────────────────────────────────
            alpha = 4.0 * np.pi * N_eff.imag / lam  # 1/length
            loss_dB = alpha * 10.0 / np.log(10.0)   # dB/length

            # ── Group index ──────────────────────────────────────────────
            n_group = None
            if compute_group_index and len(self.wavls) > 1:
                # Find nearest wavelength index for finite-difference
                lam_hi = lam + delta_lam_nm
                lam_lo = lam - delta_lam_nm
                idx_hi = int(np.argmin(np.abs(self.wavls - lam_hi)))
                idx_lo = int(np.argmin(np.abs(self.wavls - lam_lo)))

                if idx_hi != idx_lo:
                    n_stk_hi = self.layer_indices[:, idx_hi]
                    n_stk_lo = self.layer_indices[:, idx_lo]
                    lam_hi_v = float(self.wavls[idx_hi])
                    lam_lo_v = float(self.wavls[idx_lo])

                    # Re-polish at neighbouring wavelengths using same x0
                    def _refine(n_stk_w, lam_w, x0):
                        xr2, xi2, cv2 = _polish_backend(
                            n_stk_w, self.thicknesses, self.r_types, self.r_vals,
                            lam_w, p_int, x0, step, tol, 3000,
                        )
                        return complex(xr2, xi2) if cv2 < char_threshold else None

                    N_hi = _refine(n_stk_hi, lam_hi_v, (xr, xi))
                    N_lo = _refine(n_stk_lo, lam_lo_v, (xr, xi))

                    if N_hi is not None and N_lo is not None:
                        dlam   = lam_hi_v - lam_lo_v
                        dN_dlam = (N_hi.real - N_lo.real) / dlam
                        n_group = float(N_eff.real - lam * dN_dlam)

            # ── Field profile ─────────────────────────────────────────────
            z, E_mag, lbounds = _field_profile_backend(
                n_stk, self.thicknesses, self.r_types, self.r_vals,
                lam, N_eff, p_int, n_points_per_layer
            )
            if E_mag.max() > 0:
                E_mag = E_mag / E_mag.max()

            refined.append(Eigenmode(
                pol=pol_str,
                N_eff=N_eff,
                lam=lam,
                loss_dB_per_unit=loss_dB,
                n_group=n_group,
                z=z,
                E_profile=E_mag,
                layer_bounds=lbounds,
                char_value=cv,
            ))

        refined.sort(key=lambda m: -m.N_eff.real)
        return refined


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test / demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Loom Eigenmode Solver — Self-Test")
    print("=" * 70)

    # ── Test 1: Symmetric slab waveguide (Si / SiO₂ / Si) ─────────────────
    # Analytic TE₀ effective index for a slab: well-known result, easy to check.
    print("\n[TEST 1] Symmetric Si slab waveguide (SiO₂ cladding), λ = 1550 nm")
    print("-" * 60)

    lam_c   = 1550.0    # nm
    n_si    = 3.476
    n_sio2  = 1.444
    d_core  = 220.0     # nm  — standard SOI thickness

    n_wavs  = 51
    wavls   = np.linspace(1450.0, 1650.0, n_wavs)

    n_si_arr   = np.full(n_wavs, n_si   + 0j)
    n_sio2_arr = np.full(n_wavs, n_sio2 + 0j)

    # Stack: SiO₂ (semi-inf) / Si / SiO₂ (semi-inf)
    layer_indices = np.vstack([n_sio2_arr, n_si_arr, n_sio2_arr])
    thicknesses   = np.array([0.0, d_core, 0.0])
    inc_flags     = np.zeros(3, dtype=np.int32)
    r_types       = np.zeros(3, dtype=np.int32)
    r_vals        = np.zeros(3)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )

    lam_idx = int(np.argmin(np.abs(wavls - lam_c)))
    modes   = solver.find_modes(
        lam_idx=lam_idx,
        pol='both',
        scan_points=150,
        char_threshold=0.05,
        compute_group_index=True,
    )

    if modes:
        print(f"  Found {len(modes)} mode(s) at λ = {wavls[lam_idx]:.1f} nm:")
        for m in modes:
            print(f"    {m}")
    else:
        print("  No modes found — check N_eff search range or increase scan_points.")

    # ── Test 2: Landscape plot ──────────────────────────────────────────────
    print("\n[TEST 2] Generating |1/r|² landscape plot …")
    Nr, Ni, land = solver.scan_landscape(
        lam_idx=lam_idx, pol='s', scan_points=200
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.pcolormesh(Nr, Ni, land, cmap='viridis_r', shading='auto', vmax=2)
    fig.colorbar(im, ax=ax, label='log₁₀(|1/r|²)')
    ax.set_xlabel('Re(N_eff)')
    ax.set_ylabel('Im(N_eff)')
    ax.set_title(f'Mode landscape — s-pol, λ={wavls[lam_idx]:.0f} nm')
    for m in modes:
        if m.pol == 's':
            ax.plot(m.N_eff.real, m.N_eff.imag, 'r*', ms=12,
                    label=f'N_eff={m.N_eff.real:.4f}')
    ax.legend(fontsize=8)

    # ── Test 3: Field profile ───────────────────────────────────────────────
    ax2 = axes[1]
    for m in modes:
        if len(m.z) > 1:
            ax2.plot(m.z, m.E_profile,
                     label=f'{m.pol}-pol  N_eff={m.N_eff.real:.4f}')
    for (z0, z1, n_lay) in (modes[0].layer_bounds if modes else []):
        ax2.axvspan(z0, z1, alpha=0.08, color='steelblue')
        ax2.text((z0 + z1) / 2, 0.95, f'n={n_lay.real:.3f}',
                 ha='center', va='top', fontsize=7, color='steelblue',
                 transform=ax2.get_xaxis_transform())
    ax2.set_xlabel('Depth z (nm)')
    ax2.set_ylabel('|E| (normalised)')
    ax2.set_title('Modal field profiles')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.15)

    plt.tight_layout()
    out_png = '/mnt/user-data/outputs/loom_eigenmode_test.png'
    plt.savefig(out_png, dpi=150)
    print(f"  Saved landscape + field-profile plot to {out_png}")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70)
