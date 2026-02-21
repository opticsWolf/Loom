# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: Physically Rigorous Partial Coherence Engine

Sign Conventions (documented per recommendation #10):
─────────────────────────────────────────────────────
  Fresnel Coefficients:
    This module uses the ADMITTANCE convention for Fresnel coefficients:
        Y_s = N · cos(θ)          (s-polarization)
        Y_p = N / cos(θ)          (p-polarization)
        r   = (Y_curr − Y_next) / (Y_curr + Y_next)
        t   = 2·Y_curr / (Y_curr + Y_next)
    
    For p-polarization this gives r_p = −r_p(Born&Wolf). The sign flip
    is compensated in the ellipsometric extraction so that the output
    follows the STANDARD convention below.

  Ellipsometric Parameters (Azzam & Bashara / Born & Wolf convention):
        ρ = r_p / r_s = tan(Ψ) · exp(iΔ)
        Δ = δ_p − δ_s   (Born & Wolf sign)
    
    For a bare dielectric (n₁ < n₂) at external reflection:
        Δ ≈ π   below Brewster's angle
        Δ ≈ 0   above Brewster's angle
    This matches commercial SE instruments (J.A. Woollam, Horiba, etc.).

  Stokes Parameters:
        S₀ = |E_p|² + |E_s|²       (total intensity)
        S₁ = |E_p|² − |E_s|²       (linear dichroism)
        S₂ = 2 Re(E_p · E_s*)      (linear ±45° preference)
        S₃ = 2 Im(E_p · E_s*)      (circular preference)

  Roughness Models (types 0–5):
        0: None (ideal interface)           W(q) = 1
        1: Linear (triangle form factor)    W(q) = sin(√3·q) / (√3·q)
        2: Step (cosine form factor)        W(q) = cos(q)
        3: Exponential                      W(q) = 1 / (1 + q²/2)
        4: Gaussian (Debye-Waller)          W(q) = exp(−q²/2)
        5: Névot-Croce                      W(q) = exp(−2·kz1·kz2·σ²)
           (applied as a single factor to r and t, per Névot & Croce 1980)

References:
    [1] Azzam, R.M.A. & Bashara, N.M., Ellipsometry and Polarized Light (1987)
    [2] Katsidis, C.C. & Siapkas, D.I., Appl. Opt. 41(19), 3978-3987 (2002)
    [3] Névot, L. & Croce, P., Rev. Phys. Appl. 15(3), 761-779 (1980)
    [4] Redheffer, R., J. Math. Phys. 41(1), 1-41 (1962)
"""

import numpy as np
from numba import njit, prange, complex128, float64, int32
from typing import Tuple, Dict

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

POL_S: int32 = 0
POL_P: int32 = 1
DBL_EPSILON: float64 = 2.22e-16
LOG_MIN: float64 = 1e-100
SQRT3: float64 = 1.73205080757


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, inline='always')
def w_function(q, rough_type):
    """
    Calculates the roughness factor W(q), also known as the Debye-Waller-like
    factor or interfacial form factor, for optical thin films with various
    interface profiles.

    This function models how surface roughness affects reflectivity by providing
    a decay factor that depends on the momentum transfer q and the assumed
    interface profile. The implemented models are commonly used in X-ray and
    neutron reflectometry to account for non-ideal interfaces.

    Args:
        q (complex): Momentum transfer perpendicular to the interface, typically
            denoted as q_z, scaled by roughness σ (dimensionless when q = kz·σ).
        rough_type (int): Integer specifying the type of roughness model:
            0: No roughness (sharp interface)        → W = 1
            1: Linear profile (triangle form factor) → W = sin(√3·q) / (√3·q)
            2: Step function (cosine form factor)    → W = cos(q)
            3: Exponential decay (Lorentzian)        → W = 1 / (1 + q²/2)
            4: Gaussian profile (Debye-Waller)       → W = exp(−q²/2)

    Returns:
        complex: The calculated W(q) factor (real-valued for real q).

    Note:
        Roughness type 5 (Névot-Croce) is handled separately in the coherent
        block solver because it uses a fundamentally different parameterization
        involving both kz₁ and kz₂ simultaneously, rather than a simple
        form-factor approach.
    """
    if rough_type == 0:
        return 1.0 + 0j

    # Type 1: Linear (triangle form factor)
    # Derived from Fourier transform of rectangular derivative profile.
    # √3 normalizes the width parameter to standard deviation σ.
    if rough_type == 1:
        val = q * SQRT3
        if np.abs(val) < 1e-9:
            return 1.0 + 0j
        return np.sin(val) / val

    # Type 2: Step (cosine form factor)
    elif rough_type == 2:
        return np.cos(q)

    # Type 3: Exponential (Lorentzian derivative profile)
    # Used for interfaces with long-range mixing.
    elif rough_type == 3:
        return 1.0 / (1.0 + (q * q) * 0.5)

    # Type 4: Gaussian (Debye-Waller factor)
    # Classic form for normally-distributed surface height fluctuations.
    elif rough_type == 4:
        return np.exp(-(q * q) * 0.5)

    return 1.0 + 0j


@njit(cache=True, inline='always')
def redheffer_product_complex_field(
    r_A_front, t_A_back, t_A_fwd, r_A_back,
    r_B_front, t_B_back, t_B_fwd, r_B_back
):
    """
    Complex Redheffer Star Product for FIELD amplitudes.

    Combines scattering matrix of block A (upstream) with block B (downstream)
    to produce the combined scattering matrix of the composite A⊕B system.

    The S-matrix convention is:
        [b_out⁺]   [S₁₁  S₁₂] [a_in⁺ ]      S₁₁ = r_front
        [a_out⁻] = [S₂₁  S₂₂] [b_in⁻ ]      S₂₂ = r_back
                                                S₁₂ = t_back  (right→left)
                                                S₂₁ = t_fwd   (left→right)

    Args:
        r_A_front, t_A_back, t_A_fwd, r_A_back: S-matrix of block A (complex)
        r_B_front, t_B_back, t_B_fwd, r_B_back: S-matrix of block B (complex)

    Returns:
        (r_front, t_back, t_fwd, r_back): S-matrix of composite A⊕B (complex)

    Reference: Redheffer, R., J. Math. Phys. 41(1), 1-41 (1962)
    """
    # Denominator: 1 − r_back_A · r_front_B  (cavity resonance term)
    denom = 1.0 - r_A_back * r_B_front

    # Phase-preserving regularization: offset along the existing phase direction
    # to avoid introducing artificial 45° phase bias from a fixed (1+1j) offset.
    if np.abs(denom) < LOG_MIN:
        phase = denom / (np.abs(denom) + 1e-300)  # unit phasor (or 0)
        denom = LOG_MIN * phase + 1e-300
    inv_denom = 1.0 / denom

    # Redheffer star product formulas
    s_r_front = r_A_front + t_A_back * r_B_front * t_A_fwd * inv_denom
    s_t_back  = t_A_back * t_B_back * inv_denom
    s_t_fwd   = t_B_fwd * t_A_fwd * inv_denom
    s_r_back  = r_B_back + t_B_fwd * r_A_back * t_B_back * inv_denom

    return s_r_front, s_t_back, s_t_fwd, s_r_back


@njit(cache=True, inline='always')
def redheffer_product_real(ra_Rf, ra_Tb, ra_Tf, ra_Rb,
                           rb_Rf, rb_Tb, rb_Tf, rb_Rb):
    """
    Real-valued Intensity Redheffer Star Product.

    Combines intensity scattering matrices for incoherent light propagation
    where phase information has been lost. Each matrix is parameterized as:
        (R_front, T_backward, T_forward, R_back)

    This is mathematically identical to the complex Redheffer product but
    operates on |amplitude|² quantities, ensuring energy conservation for
    incoherent multiple reflections between sub-stacks.

    Reference: Katsidis & Siapkas, Appl. Opt. 41(19), 3978-3987 (2002)
    """
    denom = 1.0 - ra_Rb * rb_Rf
    if np.abs(denom) < DBL_EPSILON:
        inv_denom = 0.0
    else:
        inv_denom = 1.0 / denom

    Rf  = ra_Rf + ra_Tb * rb_Rf * ra_Tf * inv_denom
    Tb  = ra_Tb * rb_Tb * inv_denom
    Tf  = rb_Tf * ra_Tf * inv_denom
    Rb  = rb_Rb + rb_Tf * ra_Rb * rb_Tb * inv_denom
    return Rf, Tb, Tf, Rb


@njit(cache=True, inline='always')
def solve_coherent_block_fields(
    start_idx, end_idx, n_stack, d_stack,
    rough_vals, rough_types, lam,
    NSinFi, pol
):
    """
    Solves a coherent sub-stack using the S-matrix (Redheffer) method for FIELD
    amplitudes, then computes the corresponding real-valued Intensity S-matrix.

    This function computes the combined optical response of a contiguous sequence
    of thin-film layers that maintain phase coherence. It handles:
        - Fresnel coefficients at each interface (admittance formulation)
        - Interfacial roughness via form-factor models (types 1-4) or
          Névot-Croce (type 5)
        - Layer propagation phases with branch-cut-safe cosine computation
        - Conversion from field amplitudes to Poynting-flux-correct intensities

    Args:
        start_idx (int): First layer index in the coherent block (inclusive).
        end_idx (int): Last layer index in the block (exclusive). The block
            spans interfaces from start_idx→start_idx+1 through end_idx-1→end_idx.
        n_stack (complex128[:]): Complex refractive indices for all layers at
            the current wavelength.
        d_stack (float64[:]): Layer thicknesses in same units as wavelength.
        rough_vals (float64[:]): Roughness σ values for each interface.
        rough_types (int32[:]): Roughness model type (0-5) for each interface.
        lam (float64): Wavelength of incident radiation.
        NSinFi (complex128): Snell's law invariant N₀·sin(θ₀).
        pol (int32): Polarization state (POL_S=0 or POL_P=1).

    Returns:
        Tuple of 8 values:
            (r_front, t_back, t_fwd, r_back): Complex field S-matrix elements.
            (R_front, T_back, T_fwd, R_back):  Real intensity S-matrix elements.
        
        The field elements preserve phase information needed for ellipsometry.
        The intensity elements are Poynting-flux-correct, accounting for the
        admittance ratio Re(Y_exit)/Re(Y_entry) in the transmission terms.
    """
    # Initialize identity S-matrix (Fields): r=0, t=1
    sg_rf, sg_tb, sg_tf, sg_rb = 0.0+0j, 1.0+0j, 1.0+0j, 0.0+0j
    two_pi_lam = 2.0 * np.pi / lam

    # --- First layer of block: compute cos(θ) and admittance ---
    N_curr = n_stack[start_idx]

    # Branch-cut-safe sqrt: enforce Im(cos θ) ≥ 0 so evanescent waves decay.
    # Without this, np.sqrt can land on the branch where Im(cos) < 0,
    # producing exponentially growing (non-physical) evanescent fields.
    val_curr = 1.0 - (NSinFi / N_curr)**2
    cos_curr = np.sqrt(val_curr)
    if cos_curr.imag < 0.0:
        cos_curr = -cos_curr

    # Admittance: Y_s = N·cos(θ),  Y_p = N/cos(θ)
    if pol == POL_S:
        Y_curr = N_curr * cos_curr
    else:
        # Guard against cos → 0 (grazing incidence) to prevent Y → ∞
        if np.abs(cos_curr) < 1e-12:
            cos_curr = 1e-12 + 0j
        Y_curr = N_curr / cos_curr

    # Store entry admittance for final transmission intensity scaling
    Y_first = Y_curr

    for idx in range(start_idx, end_idx):
        N_next = n_stack[idx + 1]
        sigma = rough_vals[idx + 1]
        rtype = rough_types[idx + 1]

        # Branch-cut-safe cos(θ) in next layer
        val_next = 1.0 - (NSinFi / N_next)**2
        cos_next = np.sqrt(val_next)
        if cos_next.imag < 0.0:
            cos_next = -cos_next

        if pol == POL_S:
            Y_next = N_next * cos_next
        else:
            if np.abs(cos_next) < 1e-12:
                cos_next = 1e-12 + 0j
            Y_next = N_next / cos_next

        # --- Fresnel Coefficients (Field Amplitudes) ---
        #   r₁₂ = (Y₁ − Y₂) / (Y₁ + Y₂)
        #   t₁₂ = 2·Y₁ / (Y₁ + Y₂)
        den = Y_curr + Y_next
        if np.abs(den) < LOG_MIN:
            den = LOG_MIN * (1.0 + 1.0j)
        inv_den = 1.0 / den

        r12 = (Y_curr - Y_next) * inv_den
        r21 = -r12
        t12 = 2.0 * Y_curr * inv_den
        t21 = 2.0 * Y_next * inv_den

        # --- Interfacial Roughness ---
        if rtype == 5:
            # Névot-Croce: r *= exp(−2·kz₁·kz₂·σ²)
            # The most physically rigorous model for Gaussian interfaces.
            # Uses the exact correlation between wavevectors on both sides.
            # Reference: Névot & Croce, Rev. Phys. Appl. 15(3), 761-779 (1980)
            kz1 = two_pi_lam * N_curr * cos_curr
            kz2 = two_pi_lam * N_next * cos_next
            nc_factor = np.exp(-2.0 * kz1 * kz2 * sigma * sigma)
            r12 *= nc_factor
            r21 *= nc_factor
            t12 *= nc_factor
            t21 *= nc_factor
        elif rtype != 0:
            # Form-factor roughness models (types 1-4):
            # Separate attenuation factors for reflection and transmission.
            #   al = W(2·kz₁·σ)  → modifies r₁₂ (reflection from side 1)
            #   be = W(2·kz₂·σ)  → modifies r₂₁ (reflection from side 2)
            #   ga = W((kz₁−kz₂)·σ) → modifies t₁₂, t₂₁ (transmission)
            kz1 = two_pi_lam * N_curr * cos_curr
            kz2 = two_pi_lam * N_next * cos_next

            al = w_function(2.0 * kz1 * sigma, rtype)
            be = w_function(2.0 * kz2 * sigma, rtype)
            ga = w_function((kz1 - kz2) * sigma, rtype)

            r12 *= al
            r21 *= be
            t12 *= ga
            t21 *= ga

        # --- Accumulate into block S-matrix via Redheffer star product ---
        # Interface B: r_B_front=r12, t_B_back=t21 (2→1), t_B_fwd=t12 (1→2), r_B_back=r21
        sg_rf, sg_tb, sg_tf, sg_rb = redheffer_product_complex_field(
            sg_rf, sg_tb, sg_tf, sg_rb,
            r12, t21, t12, r21
        )

        # --- Propagation phase in next layer (if not the final layer) ---
        if idx + 1 < end_idx:
            d = d_stack[idx + 1]
            if d > 1e-12:
                beta = two_pi_lam * d * N_next * cos_next

                # Passivity enforcement: Im(β) ≥ 0 ensures |exp(iβ)| ≤ 1.
                # With correct branch-cut handling above (Im(cos) ≥ 0) and
                # Im(N) ≥ 0 for passive media, this should never trigger.
                if beta.imag < 0.0:
                    beta = complex(beta.real, -beta.imag)

                phi = np.exp(1j * beta)

                # S-matrix phase update:
                #   r_back  → r_back · φ²  (round trip in layer)
                #   t_back  → t_back · φ   (single pass)
                #   t_fwd   → t_fwd  · φ   (single pass)
                #   r_front is unchanged (no additional path in this layer)
                sg_rb *= (phi * phi)
                sg_tb *= phi
                sg_tf *= phi

        # Advance to next interface
        N_curr = N_next
        cos_curr = cos_next
        Y_curr = Y_next

    # ─── Convert field amplitudes to Poynting-flux-correct intensities ───
    # R = |r|²
    # T = |t|² · Re(Y_exit) / Re(Y_entry)
    #
    # The admittance ratio corrects for the different Poynting flux per unit
    # |E|² in the entry vs exit media. For TIR (purely imaginary Y), the
    # real part is zero and transmission of energy vanishes.

    R_front = np.abs(sg_rf)**2
    R_back  = np.abs(sg_rb)**2

    real_Y_first = Y_first.real
    real_Y_last  = Y_curr.real

    # Clamp negative Re(Y) to zero. In absorbing media at oblique incidence,
    # Re(Y_p) can become negative in rare edge cases. Zero is the safe bound.
    if real_Y_first < 1e-15:
        real_Y_first = 0.0
    if real_Y_last < 1e-15:
        real_Y_last = 0.0

    if real_Y_first > 1e-15:
        factor_fwd = real_Y_last / real_Y_first
    else:
        factor_fwd = 0.0

    if real_Y_last > 1e-15:
        factor_back = real_Y_first / real_Y_last
    else:
        factor_back = 0.0

    T_fwd  = np.abs(sg_tf)**2 * factor_fwd
    T_back = np.abs(sg_tb)**2 * factor_back

    return sg_rf, sg_tb, sg_tf, sg_rb, R_front, T_back, T_fwd, R_back


# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine: Full Ellipsometric Calculation
# ═══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True, cache=True)
def core_engine_rigorous_ellipsometry(
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, debug_flag
):
    """
    Computes polarized reflectance/transmittance spectra AND full ellipsometric
    parameters (Ψ, Δ, DOP) for a multilayer structure across a 2D grid of
    wavelengths and incident angles.

    This function handles both coherent and incoherent light propagation using
    a hybrid S-matrix / intensity-matrix method:
        - Coherent sub-stacks: Redheffer star product of field S-matrices
        - Incoherent boundaries: Redheffer star product of intensity matrices
        - Ellipsometry: Stokes parameters from coherent field phases +
                        incoherent intensity accumulation

    Partial Coherence Treatment for Ellipsometry:
        Reflection: S₀ and S₁ come from the total (incoherent) intensity
            Redheffer product. S₂ and S₃ come from the first coherent block's
            field amplitudes only, since incoherent echoes scramble the
            inter-polarization phase relationship.
        
        Transmission: S₀ and S₁ from intensity Redheffer (all echoes). S₂ and
            S₃ from rigorous Mueller-matrix cross-term propagation through all
            coherent blocks (single-pass), exploiting the fact that in isotropic
            media the p-s phase difference is preserved through incoherent layers
            (same kz for both polarizations). See module docstring for details.

    The calculation is parallelized across a flattened (Angles × Wavelengths)
    loop using Numba prange.

    Args:
        wavls (float64[:]): Wavelengths [length units].
        sin_theta_arr (float64[:]): Precomputed sin(θ) for each angle.
        n_layers (int): Total number of layers including ambient and substrate.
        n_stack_cache (complex128[:,:]): Refractive indices, shape (n_wavs, n_layers).
        thicknesses (float64[:]): Layer thicknesses [same units as wavls].
        incoherent_flags (int32[:]): 1 for layers that break phase coherence.
        rough_types (int32[:]): Roughness model type (0-5) per interface.
        rough_vals (float64[:]): Roughness σ per interface [same units as wavls].
        debug_flag (int32): If 1, fills energy conservation diagnostics.

    Returns:
        Tuple of 13 arrays (each shape [num_angles, num_wavs]):
            Psi_R, Delta_R, DOP_R: Reflection ellipsometric parameters
            Rs_out, Rp_out, R_avg: Polarized and unpolarized reflectance
            Psi_T, Delta_T, DOP_T: Transmission ellipsometric parameters
            Ts_out, Tp_out, T_avg: Polarized and unpolarized transmittance
            conservation_err:      |1 − R − T − A| diagnostic (debug only)
    """
    num_wavs = len(wavls)
    num_angles = len(sin_theta_arr)
    total_points = num_wavs * num_angles
    idx_N = n_layers - 1

    # ─── Output allocations ───
    Psi_R   = np.zeros((num_angles, num_wavs), dtype=float64)
    Delta_R = np.zeros((num_angles, num_wavs), dtype=float64)
    DOP_R   = np.zeros((num_angles, num_wavs), dtype=float64)
    Rs_out  = np.zeros((num_angles, num_wavs), dtype=float64)
    Rp_out  = np.zeros((num_angles, num_wavs), dtype=float64)
    R_avg   = np.zeros((num_angles, num_wavs), dtype=float64)

    Psi_T   = np.zeros((num_angles, num_wavs), dtype=float64)
    Delta_T = np.zeros((num_angles, num_wavs), dtype=float64)
    DOP_T   = np.zeros((num_angles, num_wavs), dtype=float64)
    Ts_out  = np.zeros((num_angles, num_wavs), dtype=float64)
    Tp_out  = np.zeros((num_angles, num_wavs), dtype=float64)
    T_avg   = np.zeros((num_angles, num_wavs), dtype=float64)

    conservation_err = np.zeros((num_angles, num_wavs), dtype=float64)

    for k in prange(total_points):
        a = k // num_wavs
        w = k % num_wavs

        lam = wavls[w]
        sin_theta = sin_theta_arr[a]

        current_n_stack = n_stack_cache[w, :]
        N0 = current_n_stack[0]
        NSinFi = N0 * complex(sin_theta, 0.0)

        # ─── Global Intensity S-matrices (real-valued) ───
        # Format: (R_front, T_backward, T_forward, R_back)
        # Initialize as identity (transparent): R=0, T=1
        Ig_s = (0.0, 1.0, 1.0, 0.0)
        Ig_p = (0.0, 1.0, 1.0, 0.0)

        # Phase storage for ellipsometry
        rs0_c = 0.0 + 0.0j   # First block: reflection field (s-pol)
        rp0_c = 0.0 + 0.0j   # First block: reflection field (p-pol)

        # Mueller cross-term accumulator for transmission.
        # Tracks the product of tp_k·conj(ts_k) across all coherent blocks,
        # multiplied by incoherent layer transmittances.
        # The 2×2 Mueller sub-block for cross-polarization terms has the form
        # [[C, D], [−D, C]] where z = C + iD = tp·conj(ts). Products of such
        # matrices correspond to complex multiplication of the z values.
        cross_T_acc = 1.0 + 0.0j

        first_block_processed = False

        current_idx = 0
        while current_idx < idx_N:
            # 1. Identify extent of coherent block
            next_incoh = current_idx + 1
            while next_incoh < idx_N and incoherent_flags[next_incoh] == 0:
                next_incoh += 1

            # 2. Solve coherent block for s-polarization
            rs_f, ts_b, ts_f, rs_b, Rs_Rf, Rs_Tb, Rs_Tf, Rs_Rb = \
                solve_coherent_block_fields(
                    current_idx, next_incoh, current_n_stack, thicknesses,
                    rough_vals, rough_types, lam, NSinFi, POL_S)

            # 3. Solve coherent block for p-polarization
            rp_f, tp_b, tp_f, rp_b, Rp_Rf, Rp_Tb, Rp_Tf, Rp_Rb = \
                solve_coherent_block_fields(
                    current_idx, next_incoh, current_n_stack, thicknesses,
                    rough_vals, rough_types, lam, NSinFi, POL_P)

            # 4. Phase capture
            if not first_block_processed:
                rs0_c = rs_f
                rp0_c = rp_f
                first_block_processed = True

            # 5. Accumulate Mueller cross-terms for transmission:
            #    z_block = tp · conj(ts) captures the relative p-s phase
            cross_T_acc *= (tp_f * np.conjugate(ts_f))

            # 6. Accumulate intensities via real Redheffer star product
            Ig_s = redheffer_product_real(
                Ig_s[0], Ig_s[1], Ig_s[2], Ig_s[3],
                Rs_Rf, Rs_Tb, Rs_Tf, Rs_Rb)
            Ig_p = redheffer_product_real(
                Ig_p[0], Ig_p[1], Ig_p[2], Ig_p[3],
                Rp_Rf, Rp_Tb, Rp_Tf, Rp_Rb)

            # 7. Incoherent layer propagation (if present)
            if next_incoh < idx_N and incoherent_flags[next_incoh] == 1:
                N_inc = current_n_stack[next_incoh]
                d_inc = thicknesses[next_incoh]

                val_inc = 1.0 - (NSinFi / N_inc)**2
                cos_inc = np.sqrt(val_inc)
                if cos_inc.imag < 0.0:
                    cos_inc = -cos_inc

                # Intensity attenuation: exp(−2·Im(kz)·d)
                beta_imag = (2.0 * np.pi * d_inc / lam) * (N_inc * cos_inc).imag
                if beta_imag < 0.0:
                    beta_imag = 0.0

                trans_factor = np.exp(-2.0 * beta_imag)

                # Incoherent layer: symmetric, no reflection, just absorption
                Ig_s = redheffer_product_real(
                    Ig_s[0], Ig_s[1], Ig_s[2], Ig_s[3],
                    0.0, trans_factor, trans_factor, 0.0)
                Ig_p = redheffer_product_real(
                    Ig_p[0], Ig_p[1], Ig_p[2], Ig_p[3],
                    0.0, trans_factor, trans_factor, 0.0)

                # Cross-term attenuation: same factor for both polarizations
                # in isotropic media (same kz), so only magnitude is reduced.
                cross_T_acc *= trans_factor

            current_idx = next_incoh

        # ═══════════════════════════════════════════════════════════════════
        # Results Extraction
        # ═══════════════════════════════════════════════════════════════════

        # Photometry from global intensity S-matrix
        val_Rs = Ig_s[0]   # R_front for s
        val_Ts = Ig_s[2]   # T_fwd for s
        val_Rp = Ig_p[0]   # R_front for p
        val_Tp = Ig_p[2]   # T_fwd for p

        Rs_out[a, w] = val_Rs
        Rp_out[a, w] = val_Rp
        Ts_out[a, w] = val_Ts
        Tp_out[a, w] = val_Tp
        R_avg[a, w] = 0.5 * (val_Rs + val_Rp)
        T_avg[a, w] = 0.5 * (val_Ts + val_Tp)

        # ─── Energy conservation diagnostic (debug mode) ───
        if debug_flag == 1:
            # For lossless stacks: R + T = 1 per polarization
            # For absorbing stacks: R + T ≤ 1, with the deficit being absorption
            # We report max(|1 − Rs − Ts|, |1 − Rp − Tp|) as the error metric.
            err_s = np.abs(1.0 - val_Rs - val_Ts)
            err_p = np.abs(1.0 - val_Rp - val_Tp)
            conservation_err[a, w] = max(err_s, err_p)

        # ─── Reflection Ellipsometry ───
        # S₀, S₁ from total incoherent intensities (include all echoes)
        S0_R = val_Rp + val_Rs
        S1_R = val_Rp - val_Rs

        # S₂, S₃ from first coherent block only (incoherent echoes lose phase).
        #
        # Convention fix: The admittance-based Fresnel r_p = −r_p(Born & Wolf).
        # To obtain Δ = δp(BW) − δs in the standard convention, we negate the
        # cross product to compensate for the sign flip in r_p:
        #
        #   cross_BW = (−rp_adm) · rs* = −(rp_adm · rs*)
        #   S₂ = 2·Re(cross_BW) = −2·Re(rp_adm · rs*)
        #   S₃ = 2·Im(cross_BW) = −2·Im(rp_adm · rs*)
        #
        # This yields Δ = atan2(S₃, S₂) = δp(BW) − δs per Azzam & Bashara.
        cross_R = rp0_c * np.conjugate(rs0_c)
        S2_R = -2.0 * cross_R.real
        S3_R = -2.0 * cross_R.imag
        # Flush IEEE 754 negative zero: -2.0 * 0.0 = -0.0, which makes
        # atan2(-0.0, negative) return -π instead of +π. Adding +0.0
        # converts -0.0 → +0.0 per IEEE 754 addition rules.
        S2_R = S2_R + 0.0
        S3_R = S3_R + 0.0

        DOP_R[a, w] = np.sqrt(S1_R**2 + S2_R**2 + S3_R**2) / (S0_R + 1e-20)

        if val_Rs < 1e-12:
            Psi_R[a, w] = np.pi / 2.0
            Delta_R[a, w] = 0.0
        else:
            Psi_R[a, w] = np.arctan(np.sqrt(val_Rp / val_Rs))
            Delta_R[a, w] = np.arctan2(S3_R, S2_R)

        # ─── Transmission Ellipsometry ───
        S0_T = val_Tp + val_Ts
        S1_T = val_Tp - val_Ts

        # Mueller cross-term propagation (rigorous for single-pass):
        #
        # cross_T_acc = Π_blocks(tp_k · conj(ts_k)) · Π_inc(τ_k)
        #
        # This product captures the accumulated p-s phase through all coherent
        # blocks and the intensity attenuation through incoherent layers.
        # In isotropic media, incoherent layers add the same phase (kz·d) to
        # both polarizations, so the p-s phase DIFFERENCE is preserved.
        #
        # Note: This is the single-pass contribution. Multiple reflections
        # from incoherent cavities contribute to S₀/S₁ but not to S₂/S₃
        # (the cross-polarization phase is scrambled by the different
        # reflection coefficients for s vs p at each echo).
        #
        # No convention correction needed for transmission (tp has no sign
        # flip between admittance and Born & Wolf conventions).
        S2_T = 2.0 * cross_T_acc.real + 0.0
        S3_T = 2.0 * cross_T_acc.imag + 0.0

        # Clamp DOP ≤ 1 (can slightly exceed due to single-pass approximation
        # for S₂/S₃ vs multi-bounce S₀)
        raw_dop_T = np.sqrt(S1_T**2 + S2_T**2 + S3_T**2) / (S0_T + 1e-20)
        DOP_T[a, w] = min(raw_dop_T, 1.0)

        if val_Ts < 1e-20:
            Psi_T[a, w] = np.pi / 2.0
            Delta_T[a, w] = 0.0
        else:
            Psi_T[a, w] = np.arctan(np.sqrt(val_Tp / val_Ts))
            Delta_T[a, w] = np.arctan2(S3_T, S2_T)

    return (Psi_R, Delta_R, DOP_R, Rs_out, Rp_out, R_avg,
            Psi_T, Delta_T, DOP_T, Ts_out, Tp_out, T_avg,
            conservation_err)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine: Photometry-Only Fast Path
# ═══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True, cache=True)
def core_engine_photometry_only(
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, calc_s, calc_p
):
    """
    Fast-path computation of polarized reflectance and transmittance WITHOUT
    ellipsometric parameters.

    Identical physics to `core_engine_rigorous_ellipsometry` but skips:
        - Cross-polarization phase tracking (S₂, S₃)
        - Stokes parameter computation
        - Ψ, Δ, DOP extraction

    Performance notes:
        - For mode='u' (both polarizations): marginal speedup (~5-10%) since
          the coherent block solver dominates and is shared code.
        - For mode='s' or mode='p' (single polarization): genuine ~2× speedup
          since only one polarization is computed.

    Args:
        wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
        incoherent_flags, rough_types, rough_vals: Same as ellipsometry engine.
        calc_s (int32): If 1, compute s-polarization.
        calc_p (int32): If 1, compute p-polarization.

    Returns:
        Rs_out, Rp_out, Ts_out, Tp_out: 2D arrays [num_angles, num_wavs].
    """
    num_wavs = len(wavls)
    num_angles = len(sin_theta_arr)
    total_points = num_wavs * num_angles
    idx_N = n_layers - 1

    Rs_out = np.zeros((num_angles, num_wavs), dtype=float64)
    Rp_out = np.zeros((num_angles, num_wavs), dtype=float64)
    Ts_out = np.zeros((num_angles, num_wavs), dtype=float64)
    Tp_out = np.zeros((num_angles, num_wavs), dtype=float64)

    for k in prange(total_points):
        a = k // num_wavs
        w = k % num_wavs

        lam = wavls[w]
        sin_theta = sin_theta_arr[a]
        current_n_stack = n_stack_cache[w, :]
        N0 = current_n_stack[0]
        NSinFi = N0 * complex(sin_theta, 0.0)

        # ─── S polarization ───
        if calc_s == 1:
            Ig_s = (0.0, 1.0, 1.0, 0.0)
            current_idx_s = 0

            while current_idx_s < idx_N:
                next_incoh = current_idx_s + 1
                while next_incoh < idx_N and incoherent_flags[next_incoh] == 0:
                    next_incoh += 1

                _, _, _, _, Rs_Rf, Rs_Tb, Rs_Tf, Rs_Rb = \
                    solve_coherent_block_fields(
                        current_idx_s, next_incoh, current_n_stack, thicknesses,
                        rough_vals, rough_types, lam, NSinFi, POL_S)

                Ig_s = redheffer_product_real(
                    Ig_s[0], Ig_s[1], Ig_s[2], Ig_s[3],
                    Rs_Rf, Rs_Tb, Rs_Tf, Rs_Rb)

                if next_incoh < idx_N and incoherent_flags[next_incoh] == 1:
                    N_inc = current_n_stack[next_incoh]
                    d_inc = thicknesses[next_incoh]
                    val_inc = 1.0 - (NSinFi / N_inc)**2
                    cos_inc = np.sqrt(val_inc)
                    if cos_inc.imag < 0.0:
                        cos_inc = -cos_inc
                    beta_imag = (2.0 * np.pi * d_inc / lam) * (N_inc * cos_inc).imag
                    if beta_imag < 0.0:
                        beta_imag = 0.0
                    tf = np.exp(-2.0 * beta_imag)
                    Ig_s = redheffer_product_real(
                        Ig_s[0], Ig_s[1], Ig_s[2], Ig_s[3],
                        0.0, tf, tf, 0.0)

                current_idx_s = next_incoh

            Rs_out[a, w] = Ig_s[0]
            Ts_out[a, w] = Ig_s[2]

        # ─── P polarization ───
        if calc_p == 1:
            Ig_p = (0.0, 1.0, 1.0, 0.0)
            current_idx_p = 0

            while current_idx_p < idx_N:
                next_incoh = current_idx_p + 1
                while next_incoh < idx_N and incoherent_flags[next_incoh] == 0:
                    next_incoh += 1

                _, _, _, _, Rp_Rf, Rp_Tb, Rp_Tf, Rp_Rb = \
                    solve_coherent_block_fields(
                        current_idx_p, next_incoh, current_n_stack, thicknesses,
                        rough_vals, rough_types, lam, NSinFi, POL_P)

                Ig_p = redheffer_product_real(
                    Ig_p[0], Ig_p[1], Ig_p[2], Ig_p[3],
                    Rp_Rf, Rp_Tb, Rp_Tf, Rp_Rb)

                if next_incoh < idx_N and incoherent_flags[next_incoh] == 1:
                    N_inc = current_n_stack[next_incoh]
                    d_inc = thicknesses[next_incoh]
                    val_inc = 1.0 - (NSinFi / N_inc)**2
                    cos_inc = np.sqrt(val_inc)
                    if cos_inc.imag < 0.0:
                        cos_inc = -cos_inc
                    beta_imag = (2.0 * np.pi * d_inc / lam) * (N_inc * cos_inc).imag
                    if beta_imag < 0.0:
                        beta_imag = 0.0
                    tf = np.exp(-2.0 * beta_imag)
                    Ig_p = redheffer_product_real(
                        Ig_p[0], Ig_p[1], Ig_p[2], Ig_p[3],
                        0.0, tf, tf, 0.0)

                current_idx_p = next_incoh

            Rp_out[a, w] = Ig_p[0]
            Tp_out[a, w] = Ig_p[2]

    return Rs_out, Rp_out, Ts_out, Tp_out


# ═══════════════════════════════════════════════════════════════════════════════
# Python Class Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class LoomScatterMatrix:
    """
    High-performance solver for multilayer optical structures supporting both
    coherent and incoherent (partial coherence) light propagation.

    Uses the Redheffer star product (S-matrix method) for numerical stability
    with thick absorbing layers, combined with intensity-matrix accumulation
    for incoherent boundaries per Katsidis & Siapkas (2002).

    Provides two computation modes:
        - compute_ellipsometry(): Full Ψ, Δ, DOP plus R/T spectra
        - compute_RT():          R/T spectra only (~2× faster for single-pol)

    Sign Convention:
        The output Δ follows the standard Born & Wolf / Azzam & Bashara
        convention: Δ = δ_p − δ_s. For a bare dielectric at external reflection,
        Δ ≈ π below Brewster's angle and Δ ≈ 0 above. This matches the output
        of commercial spectroscopic ellipsometers.

    Parameters:
        layer_indices: complex ndarray (n_layers × n_wavs)
            Complex refractive indices [N = n + ik] for each layer at all
            wavelengths. First row is ambient, last row is substrate.
        thicknesses: float ndarray (n_layers)
            Physical thickness of each layer [same units as wavls].
            Ambient and substrate should have thickness = 0.
        incoherent_flags: ndarray (n_layers)
            Non-zero for layers that break phase coherence (thick substrates).
        roughness_types: list or array of int
            Roughness model type (0-5) for each interface:
                0=None, 1=Linear, 2=Step, 3=Exponential, 4=Gaussian, 5=Névot-Croce
        roughness_values: list or array of float
            Roughness σ values [same units as wavls] for each interface.
        wavls: float ndarray
            Wavelengths at which to calculate spectra.
        theta: float or ndarray
            Angle(s) of incidence.
        theta_is_radians: bool
            If True, theta is in radians. Default: False (degrees).
        debug: bool
            If True, compute and return energy conservation diagnostics.
    """

    def __init__(self, layer_indices, thicknesses, incoherent_flags,
                 roughness_types, roughness_values, wavls, theta,
                 theta_is_radians=False, debug=False):
        self.wavls = np.ascontiguousarray(wavls, dtype=np.float64)

        theta_arr = np.atleast_1d(theta).astype(np.float64)
        if not theta_is_radians:
            theta_arr = np.radians(theta_arr)
        self.sin_theta_arr = np.ascontiguousarray(np.sin(theta_arr))
        self.num_angles = len(self.sin_theta_arr)

        self.n_layers = len(thicknesses)
        self.indices_T = np.ascontiguousarray(layer_indices.T, dtype=np.complex128)
        self.thicknesses = np.ascontiguousarray(thicknesses, dtype=np.float64)
        self.inc_flags = np.ascontiguousarray(incoherent_flags, dtype=np.int32)
        self.r_types = np.ascontiguousarray(roughness_types, dtype=np.int32)
        self.r_vals = np.ascontiguousarray(roughness_values, dtype=np.float64)

        self.debug = debug
        self._debug_flag = np.int32(1) if debug else np.int32(0)

    def _squeeze(self, arr):
        """Remove angle dimension if only one angle was provided."""
        return arr[0, :] if self.num_angles == 1 else arr

    def compute_ellipsometry(self) -> Dict[str, np.ndarray]:
        """
        Full ellipsometric computation: Ψ, Δ, DOP, and R/T spectra.

        Returns:
            dict with keys:
                'Rs', 'Rp', 'Ru': Reflectance (s-pol, p-pol, unpolarized)
                'Psi', 'Delta', 'DOP': Reflection ellipsometric parameters
                'Ts', 'Tp', 'Tu': Transmittance
                'Psi_T', 'Delta_T', 'DOP_T': Transmission ellipsometric params
                
            If debug=True, also includes:
                'conservation_err': max(|1−Rs−Ts|, |1−Rp−Tp|) per point.
                    For lossless stacks this should be ~0. For absorbing
                    stacks, this equals absorption and should be in [0, 1].

            All arrays are 2D [num_angles, num_wavs], or 1D [num_wavs] if
            only one angle was provided.
        """
        (Psi_R, Delta_R, DOP_R, Rs, Rp, R_avg,
         Psi_T, Delta_T, DOP_T, Ts, Tp, T_avg,
         conservation_err) = core_engine_rigorous_ellipsometry(
            self.wavls, self.sin_theta_arr, self.n_layers, self.indices_T,
            self.thicknesses, self.inc_flags, self.r_types, self.r_vals,
            self._debug_flag
        )

        result = {
            'Rs': self._squeeze(Rs), 'Rp': self._squeeze(Rp),
            'Ru': self._squeeze(R_avg),
            'Psi': self._squeeze(Psi_R), 'Delta': self._squeeze(Delta_R),
            'DOP': self._squeeze(DOP_R),
            'Ts': self._squeeze(Ts), 'Tp': self._squeeze(Tp),
            'Tu': self._squeeze(T_avg),
            'Psi_T': self._squeeze(Psi_T), 'Delta_T': self._squeeze(Delta_T),
            'DOP_T': self._squeeze(DOP_T),
        }

        if self.debug:
            result['conservation_err'] = self._squeeze(conservation_err)

        return result

    def compute_RT(self, mode='u') -> Dict[str, np.ndarray]:
        """
        Computation of reflectance and transmittance only (no Ψ/Δ/DOP).

        For mode='s' or 'p' (single polarization), this is ~2× faster than
        compute_ellipsometry() since only one polarization is solved.
        For mode='u', the speedup is marginal since the coherent block solver
        (shared code) dominates the runtime.

        Args:
            mode (str): 's', 'p', or 'u' (default = unpolarized = both).

        Returns:
            dict with keys depending on mode:
                's': 'Rs', 'Ts'
                'p': 'Rp', 'Tp'
                'u'/'both': 'Rs', 'Rp', 'Ts', 'Tp', 'Ru', 'Tu'

            All arrays are 2D [num_angles, num_wavs], or 1D [num_wavs] if
            only one angle was provided.
        """
        calc_s = np.int32(1 if mode.lower() in ('s', 'u', 'both') else 0)
        calc_p = np.int32(1 if mode.lower() in ('p', 'u', 'both') else 0)

        Rs, Rp, Ts, Tp = core_engine_photometry_only(
            self.wavls, self.sin_theta_arr, self.n_layers, self.indices_T,
            self.thicknesses, self.inc_flags, self.r_types, self.r_vals,
            calc_s, calc_p
        )

        res = {}
        if calc_s:
            res['Rs'] = self._squeeze(Rs)
            res['Ts'] = self._squeeze(Ts)
        if calc_p:
            res['Rp'] = self._squeeze(Rp)
            res['Tp'] = self._squeeze(Tp)
        if calc_s and calc_p:
            res['Ru'] = self._squeeze((Rs + Rp) / 2.0)
            res['Tu'] = self._squeeze((Ts + Tp) / 2.0)
        return res


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Test / Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("PADDOL S-Matrix Solver — Self-Test & Benchmark")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # Setup: Quarter-Wave Stack (10 pairs of Ta₂O₅/SiO₂ on BK7)
    # ──────────────────────────────────────────────────────────────────────
    n_wavs = 1000
    wavls = np.linspace(300, 800, n_wavs)
    lambda_0 = 550.0

    n_air   = np.full(n_wavs, 1.0   + 0.0j,    dtype=np.complex128)
    n_bk7   = np.full(n_wavs, 1.515 + 0.0001j, dtype=np.complex128)
    n_sio2  = np.full(n_wavs, 1.46  + 0.001j,  dtype=np.complex128)
    n_ta2o5 = np.full(n_wavs, 2.10  + 0.0j,    dtype=np.complex128)

    d_sio2  = lambda_0 / (4.0 * 1.46)
    d_ta2o5 = lambda_0 / (4.0 * 2.10)

    structure = []
    structure.append([0.0, n_air, False, 0.0])
    num_pairs = 10
    for _ in range(num_pairs):
        structure.append([d_ta2o5, n_ta2o5, False, 1.0])
        structure.append([d_sio2,  n_sio2,  False, 1.0])
    structure.append([0.0, n_bk7, False, 0.0])

    indices_list, thick_list, inc_list = [], [], []
    rough_type_list, rough_value_list = [], []

    for layer in structure:
        thick_list.append(layer[0])
        indices_list.append(layer[1])
        inc_list.append(1 if layer[2] else 0)
        r_val = layer[3]
        if r_val > 0:
            rough_type_list.append(1)
            rough_value_list.append(r_val)
        else:
            rough_type_list.append(0)
            rough_value_list.append(0.0)

    indices_arr = np.vstack(indices_list)
    thick_arr = np.array(thick_list)
    inc_arr = np.array(inc_list, dtype=np.int32)

    angles_deg = np.linspace(0, 46, 47)

    # ──────────────────────────────────────────────────────────────────────
    # TEST 0: Delta Convention Validation (Bare Substrate at Brewster's)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[TEST 0] Delta Convention Validation")
    print("-" * 50)

    n_sub = 1.5
    bare_indices = np.vstack([
        np.full(1, 1.0 + 0j, dtype=np.complex128),
        np.full(1, n_sub + 0j, dtype=np.complex128)
    ])
    brewster_deg = np.degrees(np.arctan(n_sub))
    test_angles = np.array([10.0, brewster_deg - 1.0, brewster_deg + 1.0, 70.0])

    bare_solver = LoomScatterMatrix(
        bare_indices, np.array([0.0, 0.0]),
        np.array([0, 0], dtype=np.int32),
        [0, 0], [0.0, 0.0],
        np.array([550.0]), test_angles,
        theta_is_radians=False, debug=True
    )
    bare_res = bare_solver.compute_ellipsometry()

    print(f"  Bare substrate: n = {n_sub}, Brewster's angle = {brewster_deg:.2f}°")
    print(f"  Convention check (Δ should be ≈π below Brewster, ≈0 above):")
    for i, ang in enumerate(test_angles):
        delta_val = bare_res['Delta'][i, 0]
        psi_val = bare_res['Psi'][i, 0]
        label = "BELOW" if ang < brewster_deg else "ABOVE"
        expected = "≈π" if ang < brewster_deg else "≈0"
        actual_deg = np.degrees(delta_val)
        # Accept both +π and −π (they are physically identical)
        if ang < brewster_deg:
            ok = abs(abs(delta_val) - np.pi) < 0.3
        else:
            ok = abs(delta_val) < 0.3
        status = "✓" if ok else "✗"
        print(f"    {ang:6.1f}° ({label}): Δ = {actual_deg:8.2f}° "
              f"(expected {expected})  Ψ = {np.degrees(psi_val):6.2f}°  {status}")

    cons_err = bare_res['conservation_err']
    print(f"  Energy conservation (lossless): max |1−R−T| = {np.max(cons_err):.2e} (should be ≈0)")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 1: Photometry (Full Stack)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[TEST 1] Photometric Performance ({n_wavs} λ × {len(angles_deg)} angles)")
    print("-" * 50)

    solver = LoomScatterMatrix(
        indices_arr, thick_arr, inc_arr,
        rough_type_list, rough_value_list,
        wavls, angles_deg, theta_is_radians=False, debug=True
    )

    print("  Compiling...")
    t0 = time.time()
    _ = solver.compute_ellipsometry()
    _ = solver.compute_RT()
    print(f"  Compilation: {time.time() - t0:.3f}s")

    ITERS = 100
    print(f"\n  Benchmark: compute_ellipsometry() × {ITERS}")
    t0 = time.time()
    for _ in range(ITERS):
        res = solver.compute_ellipsometry()
    dt_ellip = time.time() - t0
    pts = n_wavs * len(angles_deg) * ITERS
    print(f"  Total: {dt_ellip:.4f}s → {pts / dt_ellip / 1e6:.2f} M points/sec")

    print(f"\n  Benchmark: compute_RT(mode='u') × {ITERS}")
    t0 = time.time()
    for _ in range(ITERS):
        res_rt = solver.compute_RT('u')
    dt_rt = time.time() - t0
    print(f"  Total: {dt_rt:.4f}s → {pts / dt_rt / 1e6:.2f} M points/sec")
    print(f"  vs ellipsometry: {dt_ellip / dt_rt:.2f}×")

    print(f"\n  Benchmark: compute_RT(mode='s') × {ITERS}  (single polarization)")
    t0 = time.time()
    for _ in range(ITERS):
        res_s = solver.compute_RT('s')
    dt_s = time.time() - t0
    print(f"  Total: {dt_s:.4f}s → {pts / dt_s / 1e6:.2f} M points/sec")
    print(f"  vs unpolarized RT: {dt_rt / dt_s:.2f}×")

    # Energy conservation / absorption check
    # For absorbing stacks, 1 − R − T = Absorption (not an error).
    # The metric reports absorption, which should be in [0, 1].
    cons = res['conservation_err']
    print(f"\n  Energy budget (absorbing stack: SiO₂ k=0.001, BK7 k=0.0001):")
    print(f"    Max absorption (1−R−T): {np.max(cons):.6f}")
    print(f"    Mean absorption:        {np.mean(cons):.6f}")

    # Sample photometry output
    display_angle = 45.0
    idx_ang = np.abs(angles_deg - display_angle).argmin()
    actual_ang = angles_deg[idx_ang]

    Rs = res['Rs'][idx_ang, :]
    Rp = res['Rp'][idx_ang, :]
    Tu = res['Tu'][idx_ang, :]

    print(f"\n  Photometry @ {actual_ang:.1f}° (every 200th point):")
    print(f"  {'λ (nm)':<10} {'Rs':<10} {'Rp':<10} {'Tu':<10}")
    for i in range(0, n_wavs, 200):
        print(f"  {wavls[i]:<10.1f} {Rs[i]:<10.5f} {Rp[i]:<10.5f} {Tu[i]:<10.5f}")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 2: Ellipsometry (Full Stack)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[TEST 2] Ellipsometric Parameters @ {actual_ang:.1f}°")
    print("-" * 50)

    Psi_deg   = np.degrees(res['Psi'][idx_ang, :])
    Delta_deg = np.degrees(res['Delta'][idx_ang, :])
    DOP       = res['DOP'][idx_ang, :]

    print(f"  {'λ (nm)':<10} {'Ψ (°)':<10} {'Δ (°)':<12} {'DOP':<10}")
    for i in range(0, n_wavs, 200):
        print(f"  {wavls[i]:<10.1f} {Psi_deg[i]:<10.2f} {Delta_deg[i]:<12.2f} "
              f"{DOP[i]:<10.4f}")

    print("\n" + "=" * 70)
    print("All tests complete.")
    print("=" * 70)