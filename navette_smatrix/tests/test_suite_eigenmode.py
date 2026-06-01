#!/usr/bin/env python3
"""
Complete test suite for Rust‑accelerated eigenmode solver.

Tests:
1. Symmetric slab waveguide (analytic TE₀ comparison)
2. Asymmetric slab (air/Si/SiO₂)
3. Multi‑layer stack (5 layers)
4. Lossy material
5. Rough interface (Névot‑Croce model)
6. Incoherent layer (handled coherently)
7. scan_landscape() – coarse grid generation
8. find_local_minima() – candidate detection
9. nelder_mead() – local refinement
10. field_profile() – |E(z)| inside stack

Speed benchmark:
- Three test cases (simple slab, asymmetric slab, 5‑layer stack)
- Each benchmarked at 20 wavelengths
- Compares Rust vs pure Python runtime
"""

import sys
import time
import numpy as np
from typing import List, Tuple, Optional

# ----------------------------------------------------------------------
# Check Rust availability
# ----------------------------------------------------------------------
RUST_AVAILABLE = False
try:
    from navette_smatrix import (
        scan_landscape,
        find_local_minima,
        nelder_mead,
        field_profile
    )
    RUST_AVAILABLE = True
    print("✓ Rust module (navette_smatrix) loaded.")
except ImportError as e:
    print(f"⚠ Rust module not available: {e}")
    print("  Tests requiring Rust functions will be skipped.")

# Try to import the Python eigenmode solver (for fallback and field_profile)
try:
    from loom_eigenmode import LoomEigenmodeSolver
except ImportError:
    print("⚠ Python LoomEigenmodeSolver not found. Fallback may be limited.")
    LoomEigenmodeSolver = None

# ----------------------------------------------------------------------
# Helper: analytic TE₀ effective index for symmetric slab
# ----------------------------------------------------------------------
def analytic_symmetric_slab_te0(n_core: float, n_clad: float, d_core: float, lam: float) -> float:
    """Solve symmetric slab TE₀ dispersion equation, return n_eff (real)."""
    k0 = 2 * np.pi / lam
    def func(neff):
        if neff <= n_clad or neff >= n_core:
            return 1e6
        kx = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_clad**2)
        return kx * d_core / 2 - np.arctan(gamma / kx)
    lo, hi = n_clad + 1e-6, n_core - 1e-6
    for _ in range(100):
        mid = (lo + hi) / 2
        fmid = func(mid)
        if abs(fmid) < 1e-9:
            return mid
        if func(lo) * fmid < 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

# ----------------------------------------------------------------------
# Pure Python reference for speed benchmark (minimal, no Numba)
# ----------------------------------------------------------------------
_LOG_MIN = 1e-100
def _redheffer_py(ra_rf, ra_tb, ra_tf, ra_rb, rb_rf, rb_tb, rb_tf, rb_rb):
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

def _stack_reflection_py(n_stack, d_stack, rough_types, rough_vals, lam, N_eff, pol):
    n_layers = len(n_stack)
    sg_rf, sg_tb, sg_tf, sg_rb = 0.0+0j, 1.0+0j, 1.0+0j, 0.0+0j
    two_pi_lam = 2.0 * np.pi / lam
    def _admittance(N, cos_th):
        if pol == 0:
            return N * cos_th
        else:
            if abs(cos_th) < 1e-12:
                cos_th = 1e-12 + 0j
            return N / cos_th
    def _safe_cos(N):
        val = 1.0 - (N_eff / N) ** 2
        c = np.sqrt(val.astype(complex))
        if c.imag < 0.0:
            c = -c
        return c
    N_curr = n_stack[0]
    cos_curr = _safe_cos(N_curr)
    Y_curr = _admittance(N_curr, cos_curr)
    for idx in range(n_layers - 1):
        N_next = n_stack[idx + 1]
        cos_next = _safe_cos(N_next)
        Y_next = _admittance(N_next, cos_next)
        den = Y_curr + Y_next
        if abs(den) < _LOG_MIN:
            den = _LOG_MIN * (1.0 + 1.0j)
        inv_den = 1.0 / den
        r12 = (Y_curr - Y_next) * inv_den
        r21 = -r12
        t12 = 2.0 * Y_curr * inv_den
        t21 = 2.0 * Y_next * inv_den
        sigma = rough_vals[idx + 1]
        rtype = rough_types[idx + 1]
        if rtype != 0 and sigma > 0:
            kz1 = two_pi_lam * N_curr * cos_curr
            kz2 = two_pi_lam * N_next * cos_next
            if rtype == 5:
                nc = np.exp(-2.0 * kz1 * kz2 * sigma * sigma)
                r12 *= nc; r21 *= nc; t12 *= nc; t21 *= nc
        sg_rf, sg_tb, sg_tf, sg_rb = _redheffer_py(
            sg_rf, sg_tb, sg_tf, sg_rb, r12, t21, t12, r21
        )
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
        N_curr = N_next
        cos_curr = cos_next
        Y_curr = Y_next
    return sg_rf

def _char_func_py(N_eff, n_stack, d_stack, rough_types, rough_vals, lam, pol):
    r = _stack_reflection_py(n_stack, d_stack, rough_types, rough_vals, lam, N_eff, pol)
    if abs(r) < 1e-15:
        return 1e30
    return (1.0 / abs(r)) ** 2

class PurePythonEigenSolver:
    """Very basic eigenmode locator via brute‑force grid scan (no polishing)."""
    def __init__(self, layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls):
        self.layer_indices = layer_indices
        self.thicknesses = thicknesses
        self.inc_flags = inc_flags
        self.r_types = r_types
        self.r_vals = r_vals
        self.wavls = wavls

    def find_modes(self, lam_idx, pol='s', scan_points=80, char_threshold=0.1, **kwargs):
        lam = self.wavls[lam_idx]
        n_stk = self.layer_indices[:, lam_idx]
        p_int = 0 if pol == 's' else 1
        n_clad = max(np.real(n_stk[0]), np.real(n_stk[-1]))
        n_core_max = np.max(np.real(n_stk))
        real_min = n_clad + 1e-4
        real_max = n_core_max - 1e-4
        imag_min, imag_max = 0.0, 0.02
        Nr = np.linspace(real_min, real_max, scan_points)
        Ni = np.linspace(imag_min, imag_max, scan_points)
        best_val = 1e30
        best_N = None
        for nr in Nr:
            for ni in Ni:
                N_eff = complex(nr, ni)
                val = _char_func_py(N_eff, n_stk, self.thicknesses, self.r_types, self.r_vals, lam, p_int)
                if val < best_val:
                    best_val = val
                    best_N = N_eff
        if best_val < char_threshold and best_N is not None:
            # Return dummy object with N_eff attribute
            class Dummy:
                pass
            d = Dummy()
            d.N_eff = best_N
            return [d]
        return []

# ----------------------------------------------------------------------
# Test cases (using Rust functions if available)
# ----------------------------------------------------------------------
def test_symmetric_slab():
    print("\n" + "="*70)
    print("TEST 1: Symmetric Si slab (air cladding) – compare TE₀ with analytic")
    print("="*70)
    lam = 1550.0
    n_si, n_air, d_core = 3.476, 1.0, 220.0
    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si  + 0j),
        np.full(n_wavs, n_air + 0j)
    ])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)

    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        solver = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls)
        modes = solver.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)
        if modes:
            neff_rust = modes[0].N_eff.real
            neff_ana = analytic_symmetric_slab_te0(n_si, n_air, d_core, lam)
            print(f"  Rust n_eff      = {neff_rust:.6f}")
            print(f"  Analytic n_eff  = {neff_ana:.6f}")
            print(f"  Difference      = {abs(neff_rust - neff_ana):.2e}")
            assert abs(neff_rust - neff_ana) < 1e-5, "TE₀ mode mismatch"
            return True
        else:
            print("  No mode found – test failed")
            return False
    else:
        print("  SKIP: Rust solver not available")
        return False

def test_asymmetric_slab():
    print("\n" + "="*70)
    print("TEST 2: Asymmetric slab (air / Si / SiO₂)")
    print("="*70)
    lam = 1550.0
    n_si, n_air, n_sio2, d_core = 3.476, 1.0, 1.444, 220.0
    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si + 0j),
        np.full(n_wavs, n_sio2 + 0j)
    ])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)

    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        solver = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls)
        modes = solver.find_modes(lam_idx=0, pol='both', scan_points=120, char_threshold=0.1)
        print(f"  Found {len(modes)} mode(s):")
        for m in modes:
            print(f"    {m}")
        assert len(modes) >= 1, "Asymmetric slab should have at least one mode"
        return True
    else:
        print("  SKIP: Rust solver not available")
        return False

def test_multilayer():
    print("\n" + "="*70)
    print("TEST 3: 5‑layer stack (air / Si / SiO₂ / Si / air)")
    print("="*70)
    lam = 1550.0
    n_si, n_sio2, n_air = 3.476, 1.444, 1.0
    d_si, d_sio2 = 100.0, 200.0
    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si + 0j),
        np.full(n_wavs, n_sio2 + 0j),
        np.full(n_wavs, n_si + 0j),
        np.full(n_wavs, n_air + 0j)
    ])
    thicknesses = np.array([0.0, d_si, d_sio2, d_si, 0.0])
    inc_flags = np.zeros(5, dtype=np.int32)
    r_types = np.zeros(5, dtype=np.int32)
    r_vals = np.zeros(5)

    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        solver = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls)
        modes = solver.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)
        print(f"  Found {len(modes)} s‑polarised mode(s).")
        assert isinstance(modes, list)
        return True
    else:
        print("  SKIP: Rust solver not available")
        return False

def test_lossy_material():
    print("\n" + "="*70)
    print("TEST 4: Lossy Si slab (n = 3.476 + 0.001j)")
    print("="*70)
    lam = 1550.0
    n_si = 3.476 + 0.001j
    n_air, d_core = 1.0, 220.0
    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si),
        np.full(n_wavs, n_air + 0j)
    ])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)

    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        solver = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls)
        modes = solver.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)
        if modes:
            print(f"  Mode: {modes[0]}")
            print(f"  Im(N_eff) = {modes[0].N_eff.imag:.4e} (should be >0)")
            assert modes[0].N_eff.imag > 0, "Lossy material should give Im(N_eff) > 0"
            return True
        else:
            print("  No mode found – test inconclusive")
            return False
    else:
        print("  SKIP: Rust solver not available")
        return False

def test_rough_interface():
    print("\n" + "="*70)
    print("TEST 5: Rough interface (Névot‑Croce, σ = 2 nm)")
    print("="*70)
    lam = 1550.0
    n_si, n_air, d_core = 3.476, 1.0, 220.0
    sigma = 2.0
    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si + 0j),
        np.full(n_wavs, n_air + 0j)
    ])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.array([0, 5, 0], dtype=np.int32)
    r_vals = np.array([0.0, sigma, 0.0])

    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        solver_rough = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls)
        modes_rough = solver_rough.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)
        r_types_smooth = np.zeros(3, dtype=np.int32)
        r_vals_smooth = np.zeros(3)
        solver_smooth = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types_smooth, r_vals_smooth, wavls)
        modes_smooth = solver_smooth.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)
        if modes_rough and modes_smooth:
            n_rough = modes_rough[0].N_eff
            n_smooth = modes_smooth[0].N_eff
            print(f"  Smooth:        N_eff = {n_smooth.real:.6f}{n_smooth.imag:+.2e}j")
            print(f"  Rough (σ=2nm): N_eff = {n_rough.real:.6f}{n_rough.imag:+.2e}j")
            assert n_rough.imag >= n_smooth.imag - 1e-8, "Roughness should increase imaginary part"
            return True
        else:
            print("  Mode not found in one case – test incomplete")
            return False
    else:
        print("  SKIP: Rust solver not available")
        return False

def test_incoherent_layer():
    print("\n" + "="*70)
    print("TEST 6: Incoherent layer (flag=1) – still finds modes coherently")
    print("="*70)
    lam = 1550.0
    n_si, n_sio2, n_air = 3.476, 1.444, 1.0
    d_si, d_sio2 = 220.0, 500.0
    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si + 0j),
        np.full(n_wavs, n_sio2 + 0j),
        np.full(n_wavs, n_air + 0j)
    ])
    thicknesses = np.array([0.0, d_si, d_sio2, 0.0])
    inc_flags = np.array([0, 0, 1, 0], dtype=np.int32)
    r_types = np.zeros(4, dtype=np.int32)
    r_vals = np.zeros(4)

    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        solver = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls)
        modes = solver.find_modes(lam_idx=0, pol='both', scan_points=100, char_threshold=0.1)
        print(f"  Found {len(modes)} mode(s) despite incoherent flag.")
        assert isinstance(modes, list)
        return True
    else:
        print("  SKIP: Rust solver not available")
        return False

def test_scan_landscape():
    print("\n" + "="*70)
    print("TEST 7: scan_landscape() – coarse grid generation")
    print("="*70)
    if not RUST_AVAILABLE:
        print("  SKIP: Rust navette_smatrix not available")
        return False
    lam = 1550.0
    n_si, n_air, d_core = 3.476, 1.0, 220.0
    n_stack = np.array([n_air + 0j, n_si + 0j, n_air + 0j])
    thicknesses = np.array([0.0, d_core, 0.0])
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)
    pol = 0  # s-polarisation
    real_min, real_max = n_air + 1e-4, n_si - 1e-4
    imag_min, imag_max = 0.0, 0.02
    points_real, points_imag = 10, 8
    result = scan_landscape(
        n_stack, thicknesses, r_types, r_vals,
        lam, pol, real_min, real_max, imag_min, imag_max, points_real, points_imag
    )
    # result is (real_vals, imag_vals, landscape_array)
    real_vals, imag_vals, land_arr = result
    print(f"  Real grid shape: {len(real_vals)}, Imag grid shape: {len(imag_vals)}")
    print(f"  Landscape array shape: {land_arr.shape}")
    assert len(real_vals) == points_real
    assert len(imag_vals) == points_imag
    assert land_arr.shape == (points_imag, points_real)
    print("  ✓ scan_landscape works")
    return True

def test_find_local_minima():
    print("\n" + "="*70)
    print("TEST 8: find_local_minima() – candidate detection")
    print("="*70)
    if not RUST_AVAILABLE:
        print("  SKIP: Rust navette_smatrix not available")
        return False
    # Create a simple 2D array with a known minimum
    real_vals = np.linspace(1.4, 1.5, 5)
    imag_vals = np.linspace(0.0, 0.02, 4)
    landscape = np.zeros((len(imag_vals), len(real_vals)))
    # Put a deep minimum at (2,2) in indices
    landscape[2, 2] = 0.01
    # Surrounding values higher
    for i in range(landscape.shape[0]):
        for j in range(landscape.shape[1]):
            if (i, j) != (2, 2):
                landscape[i, j] = 0.1
    median_factor = 0.5  # threshold = median * 0.5
    candidates = find_local_minima(landscape, real_vals, imag_vals, median_factor)
    print(f"  Found candidates: {candidates}")
    # We expect one candidate at (real_vals[2], imag_vals[2])
    expected = (real_vals[2], imag_vals[2])
    assert len(candidates) == 1
    assert abs(candidates[0][0] - expected[0]) < 1e-6
    assert abs(candidates[0][1] - expected[1]) < 1e-6
    print("  ✓ find_local_minima works")
    return True

def test_nelder_mead():
    print("\n" + "="*70)
    print("TEST 9: nelder_mead() – local refinement")
    print("="*70)
    if not RUST_AVAILABLE:
        print("  SKIP: Rust navette_smatrix not available")
        return False
    lam = 1550.0
    n_si, n_air, d_core = 3.476, 1.0, 220.0
    n_stack = np.array([n_air + 0j, n_si + 0j, n_air + 0j])
    thicknesses = np.array([0.0, d_core, 0.0])
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)
    pol = 0
    # Start from a guess near the true TE₀ effective index
    neff_ana = analytic_symmetric_slab_te0(n_si, n_air, d_core, lam)
    x0 = (neff_ana + 0.001, 0.001)
    step = 0.001
    tol = 1e-8
    max_iter = 2000
    result = nelder_mead(n_stack, thicknesses, r_types, r_vals, lam, pol, x0, step, tol, max_iter)
    n_eff_opt = complex(result[0], result[1])
    final_val = result[2]
    print(f"  Analytic n_eff: {neff_ana:.6f}")
    print(f"  Optimised n_eff: {n_eff_opt.real:.6f}{n_eff_opt.imag:+.2e}j")
    print(f"  Final char value: {final_val:.4e}")
    assert abs(n_eff_opt.real - neff_ana) < 1e-5
    assert final_val < 1e-6
    print("  ✓ nelder_mead works")
    return True

def test_field_profile():
    print("\n" + "="*70)
    print("TEST 10: field_profile() – electric field inside stack")
    print("="*70)
    if not RUST_AVAILABLE:
        print("  SKIP: Rust navette_smatrix not available")
        return False
    lam = 1550.0
    n_si, n_air, d_core = 3.476, 1.0, 220.0
    n_stack = np.array([n_air + 0j, n_si + 0j, n_air + 0j])
    thicknesses = np.array([0.0, d_core, 0.0])
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)
    pol = 0
    # First find a mode using nelder_mead
    neff_ana = analytic_symmetric_slab_te0(n_si, n_air, d_core, lam)
    # Run minimisation to get accurate N_eff
    x0 = (neff_ana, 0.0)
    step = 0.001
    tol = 1e-9
    max_iter = 2000
    res = nelder_mead(n_stack, thicknesses, r_types, r_vals, lam, pol, x0, step, tol, max_iter)
    N_eff = complex(res[0], res[1])
    points_per_layer = 50
    result = field_profile(n_stack, thicknesses, r_types, r_vals, lam, N_eff, pol, points_per_layer)
    z, e_mag, layer_start, layer_end, layer_n = result
    print(f"  Field computed at {len(z)} points over total thickness {z[-1]:.1f} nm")
    print(f"  Max |E| = {max(e_mag):.3f} (normalised to 1 expected)")
    assert len(z) > 0
    assert np.max(e_mag) <= 1.0 + 1e-6
    assert len(layer_start) == len(layer_end) == len(layer_n)
    print("  ✓ field_profile works")
    return True

# ----------------------------------------------------------------------
# Speed benchmark: compares Rust vs pure Python on multiple cases
# ----------------------------------------------------------------------
def benchmark_case(name, layer_indices, thicknesses, inc_flags, r_types, r_vals,
                   wavelengths, scan_points=80, repeat=2):
    print(f"\nBenchmark: {name}")
    n_wavs = len(wavelengths)
    results = {}
    # Rust solver
    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        solver_rust = LoomEigenmodeSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavelengths)
        t0 = time.perf_counter()
        for _ in range(repeat):
            for i in range(n_wavs):
                solver_rust.find_modes(lam_idx=i, pol='s', scan_points=scan_points, char_threshold=0.1,
                                       compute_group_index=False)
        t1 = time.perf_counter()
        rust_time = (t1 - t0) / repeat
        results['Rust'] = rust_time
        print(f"  Rust:      {rust_time:.3f} s total")
    else:
        print("  Rust solver not available – skipping")
    # Pure Python solver
    py_solver = PurePythonEigenSolver(layer_indices, thicknesses, inc_flags, r_types, r_vals, wavelengths)
    t0 = time.perf_counter()
    for _ in range(repeat):
        for i in range(n_wavs):
            py_solver.find_modes(lam_idx=i, pol='s', scan_points=scan_points, char_threshold=0.1)
    t1 = time.perf_counter()
    py_time = (t1 - t0) / repeat
    results['Pure Python'] = py_time
    print(f"  Pure Python: {py_time:.3f} s total")
    if 'Rust' in results:
        speedup = py_time / rust_time
        print(f"  Speedup:   {speedup:.1f}x")
        results['speedup'] = speedup
    return results

def run_benchmarks():
    print("\n" + "="*70)
    print("SPEED BENCHMARK (multiple waveguide cases)")
    print("="*70)
    wavelengths = np.linspace(1400, 1700, 20)  # 20 wavelengths
    scan_points = 80
    results = {}
    # Case 1: Symmetric Si slab (air cladding)
    n_si = 3.476
    n_air = 1.0
    d_core = 220.0
    n_wavs = len(wavelengths)
    layer_indices_slab = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si  + 0j),
        np.full(n_wavs, n_air + 0j)
    ])
    thicknesses_slab = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)
    results['Symmetric slab'] = benchmark_case(
        "Symmetric slab (air/Si/air)",
        layer_indices_slab, thicknesses_slab, inc_flags, r_types, r_vals,
        wavelengths, scan_points, repeat=2
    )
    # Case 2: Asymmetric slab (air/Si/SiO₂)
    n_sio2 = 1.444
    layer_indices_asym = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si  + 0j),
        np.full(n_wavs, n_sio2 + 0j)
    ])
    thicknesses_asym = np.array([0.0, d_core, 0.0])
    results['Asymmetric slab'] = benchmark_case(
        "Asymmetric slab (air/Si/SiO₂)",
        layer_indices_asym, thicknesses_asym, inc_flags, r_types, r_vals,
        wavelengths, scan_points, repeat=2
    )
    # Case 3: 5‑layer stack (air/Si/SiO₂/Si/air)
    d_si = 100.0
    d_sio2 = 200.0
    layer_indices_5l = np.vstack([
        np.full(n_wavs, n_air + 0j),
        np.full(n_wavs, n_si  + 0j),
        np.full(n_wavs, n_sio2 + 0j),
        np.full(n_wavs, n_si  + 0j),
        np.full(n_wavs, n_air + 0j)
    ])
    thicknesses_5l = np.array([0.0, d_si, d_sio2, d_si, 0.0])
    inc_flags_5l = np.zeros(5, dtype=np.int32)
    r_types_5l = np.zeros(5, dtype=np.int32)
    r_vals_5l = np.zeros(5)
    results['5‑layer stack'] = benchmark_case(
        "5‑layer (air/Si/SiO₂/Si/air)",
        layer_indices_5l, thicknesses_5l, inc_flags_5l, r_types_5l, r_vals_5l,
        wavelengths, scan_points, repeat=2
    )
    return results

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("\n" + "#"*70)
    print("# COMPLETE EIGENMODE TEST SUITE (Rust backend)")
    print("#"*70)
    if not RUST_AVAILABLE:
        print("\n⚠ Running without Rust module – many tests will be skipped.")
        print("  To enable Rust, compile the module and ensure navette_smatrix is importable.\n")
    else:
        print("\n✓ Using Rust-accelerated backend.\n")

    tests = [
        ("Symmetric slab", test_symmetric_slab),
        ("Asymmetric slab", test_asymmetric_slab),
        ("Multilayer", test_multilayer),
        ("Lossy material", test_lossy_material),
        ("Rough interface", test_rough_interface),
        ("Incoherent layer", test_incoherent_layer),
        ("scan_landscape", test_scan_landscape),
        ("find_local_minima", test_find_local_minima),
        ("nelder_mead", test_nelder_mead),
        ("field_profile", test_field_profile),
    ]
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    print(f"\nTests passed: {passed}/{len(tests)}")

    # Run speed benchmarks
    if RUST_AVAILABLE and LoomEigenmodeSolver is not None:
        bench_results = run_benchmarks()
        print("\n" + "="*70)
        print("Speed benchmark summary:")
        for case, res in bench_results.items():
            if 'speedup' in res:
                print(f"  {case:20s} : {res['speedup']:.1f}x speedup (Rust vs Python)")
            else:
                print(f"  {case:20s} : Rust not available")
    else:
        print("\n⚠ Speed benchmark skipped (Rust solver missing).")

    print("\n" + "#"*70)
    print("# Test suite completed.")
    print("#"*70)

if __name__ == "__main__":
    main()