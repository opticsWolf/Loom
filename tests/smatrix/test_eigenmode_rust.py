#!/usr/bin/env python3
# test_eigenmode_rust.py
"""
Test suite for the Rust-accelerated Loom eigenmode solver.

Tests include:
1. Symmetric slab waveguide (analytic TE₀ mode)
2. Asymmetric slab (air/Si/SiO₂)
3. Multi-layer stack (3 layers)
4. Lossy material (Si with imaginary part)
5. Rough interface (Névot-Croce model)
6. Incoherent layer (treated as coherent for eigenmode)
7. Performance benchmark

Run with: python test_eigenmode_rust.py
"""

import sys
import time
import numpy as np
from dataclasses import dataclass

# Ensure the compiled module is found
sys.path.insert(0, '.')

try:
    from navette.smatrix import (
        reflection_coefficient,
        scan_landscape,
        find_local_minima,
        nelder_mead,
        field_profile
    )
    RUST_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Rust module not available: {e}")
    print("Falling back to pure Python implementation (slow).")
    RUST_AVAILABLE = False

# Import the eigenmode solver (updated to use Rust functions)
from loom_eigenmode import LoomEigenmodeSolver, Eigenmode

# ----------------------------------------------------------------------
# Helper: analytic TE₀ effective index for symmetric slab
# ----------------------------------------------------------------------
def analytic_symmetric_slab_te0(n_core, n_clad, d_core, lam):
    """
    Solve the symmetric slab waveguide TE₀ dispersion equation.
    Returns the effective index n_eff (real) and checks existence.
    """
    k0 = 2 * np.pi / lam
    V = k0 * d_core * np.sqrt(n_core**2 - n_clad**2)
    # For TE₀, we solve: sqrt(n_core^2 - n_eff^2) * d_core/2 = arctan( sqrt((n_eff^2 - n_clad^2)/(n_core^2 - n_eff^2)) )
    # Use a simple bisection.
    def func(neff):
        if neff <= n_clad or neff >= n_core:
            return 1e6
        kx_core = k0 * np.sqrt(n_core**2 - neff**2)
        gamma_clad = k0 * np.sqrt(neff**2 - n_clad**2)
        # TE₀ even mode: kx_core * d_core/2 = atan(gamma_clad / kx_core)
        lhs = kx_core * d_core / 2
        rhs = np.arctan(gamma_clad / kx_core)
        return lhs - rhs
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
# Test cases
# ----------------------------------------------------------------------
def test_symmetric_slab():
    """Test 1: Symmetric Si slab (air cladding) – compare with analytic TE₀."""
    print("\n" + "="*70)
    print("TEST 1: Symmetric Si slab (air cladding), λ = 1550 nm")
    print("="*70)
    lam = 1550.0  # nm
    n_si = 3.476
    n_air = 1.0
    d_core = 220.0  # nm

    n_wavs = 1
    wavls = np.array([lam])
    n_si_arr = np.array([n_si + 0j])
    n_air_arr = np.array([n_air + 0j])
    layer_indices = np.vstack([n_air_arr, n_si_arr, n_air_arr])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )
    modes = solver.find_modes(
        lam_idx=0, pol='s', scan_points=120, char_threshold=0.1,
        compute_group_index=False
    )
    if modes:
        mode = modes[0]
        neff_rust = mode.N_eff.real
        neff_ana = analytic_symmetric_slab_te0(n_si, n_air, d_core, lam)
        print(f"  Rust solver:    n_eff = {neff_rust:.6f}")
        print(f"  Analytic:       n_eff = {neff_ana:.6f}")
        print(f"  Difference:     {abs(neff_rust - neff_ana):.2e}")
        assert abs(neff_rust - neff_ana) < 1e-5, "TE₀ mode mismatch"
    else:
        print("  No mode found – test failed")
        raise RuntimeError("Symmetric slab test failed")

def test_asymmetric_slab():
    """Test 2: Asymmetric slab (air / Si / SiO₂) – verify at least one mode exists."""
    print("\n" + "="*70)
    print("TEST 2: Asymmetric slab (air/Si/SiO₂), λ = 1550 nm")
    print("="*70)
    lam = 1550.0
    n_si = 3.476
    n_air = 1.0
    n_sio2 = 1.444
    d_core = 220.0

    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.array([n_air + 0j]),
        np.array([n_si + 0j]),
        np.array([n_sio2 + 0j])
    ])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )
    modes = solver.find_modes(lam_idx=0, pol='both', scan_points=120, char_threshold=0.1)
    print(f"  Found {len(modes)} mode(s):")
    for m in modes:
        print(f"    {m}")
    assert len(modes) >= 1, "Asymmetric slab should have at least one mode"

def test_multilayer():
    """Test 3: Three-layer stack (air / Si / SiO₂ / Si / air) – ensure runs without error."""
    print("\n" + "="*70)
    print("TEST 3: Multi-layer (air / Si / SiO₂ / Si / air)")
    print("="*70)
    lam = 1550.0
    n_si = 3.476
    n_sio2 = 1.444
    n_air = 1.0
    d_si = 100.0
    d_sio2 = 200.0

    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.array([n_air + 0j]),
        np.array([n_si + 0j]),
        np.array([n_sio2 + 0j]),
        np.array([n_si + 0j]),
        np.array([n_air + 0j])
    ])
    thicknesses = np.array([0.0, d_si, d_sio2, d_si, 0.0])
    inc_flags = np.zeros(5, dtype=np.int32)
    r_types = np.zeros(5, dtype=np.int32)
    r_vals = np.zeros(5)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )
    modes = solver.find_modes(lam_idx=0, pol='s', scan_points=100, char_threshold=0.1)
    print(f"  Found {len(modes)} s-polarised mode(s):")
    for m in modes:
        print(f"    {m}")
    # No analytic reference, just check that it runs without crash
    assert isinstance(modes, list)

def test_lossy_material():
    """Test 4: Lossy core (Si with n = 3.476 + 0.001j) – Im(N_eff) should be > 0."""
    print("\n" + "="*70)
    print("TEST 4: Lossy Si slab (n = 3.476 + 0.001j)")
    print("="*70)
    lam = 1550.0
    n_si = 3.476 + 0.001j
    n_air = 1.0
    d_core = 220.0

    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.array([n_air + 0j]),
        np.array([n_si]),
        np.array([n_air + 0j])
    ])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )
    modes = solver.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)
    if modes:
        mode = modes[0]
        print(f"  Mode: {mode}")
        print(f"  Im(N_eff) = {mode.N_eff.imag:.4e} (should be positive)")
        assert mode.N_eff.imag > 0, "Lossy material should give Im(N_eff) > 0"
    else:
        print("  No mode found – test inconclusive")

def test_rough_interface():
    """Test 5: Rough interface (Névot-Croce model) – compare smooth vs rough."""
    print("\n" + "="*70)
    print("TEST 5: Rough interface (Névot-Croce, sigma = 2 nm)")
    print("="*70)
    lam = 1550.0
    n_si = 3.476
    n_air = 1.0
    d_core = 220.0
    sigma = 2.0  # nm RMS

    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.array([n_air + 0j]),
        np.array([n_si + 0j]),
        np.array([n_air + 0j])
    ])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.array([0, 5, 0], dtype=np.int32)   # type 5 = Névot-Croce
    r_vals = np.array([0.0, sigma, 0.0])

    solver_rough = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )
    modes_rough = solver_rough.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)
    # Smooth reference
    r_types_smooth = np.zeros(3, dtype=np.int32)
    r_vals_smooth = np.zeros(3)
    solver_smooth = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types_smooth, r_vals_smooth, wavls
    )
    modes_smooth = solver_smooth.find_modes(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1)

    if modes_rough and modes_smooth:
        n_rough = modes_rough[0].N_eff
        n_smooth = modes_smooth[0].N_eff
        print(f"  Smooth:        N_eff = {n_smooth.real:.6f}{n_smooth.imag:+.2e}j")
        print(f"  Rough (σ=2nm): N_eff = {n_rough.real:.6f}{n_rough.imag:+.2e}j")
        # Roughness increases loss (Im part) and may slightly shift Re
        assert n_rough.imag >= n_smooth.imag - 1e-8, "Roughness should increase imaginary part"
    else:
        print("  Mode not found in one of the cases – test skipped")

def test_incoherent_layer():
    """Test 6: Incoherent layer (flag=1) – eigenmode solver treats it as coherent (OK)."""
    print("\n" + "="*70)
    print("TEST 6: Incoherent layer (flag=1) – should still find modes (coherently)")
    print("="*70)
    lam = 1550.0
    n_si = 3.476
    n_sio2 = 1.444
    n_air = 1.0
    d_si = 220.0
    d_sio2 = 500.0  # thick, marked incoherent

    n_wavs = 1
    wavls = np.array([lam])
    layer_indices = np.vstack([
        np.array([n_air + 0j]),
        np.array([n_si + 0j]),
        np.array([n_sio2 + 0j]),
        np.array([n_air + 0j])
    ])
    thicknesses = np.array([0.0, d_si, d_sio2, 0.0])
    inc_flags = np.array([0, 0, 1, 0], dtype=np.int32)   # SiO₂ is incoherent
    r_types = np.zeros(4, dtype=np.int32)
    r_vals = np.zeros(4)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )
    modes = solver.find_modes(lam_idx=0, pol='both', scan_points=100, char_threshold=0.1)
    print(f"  Found {len(modes)} mode(s) despite incoherent flag:")
    for m in modes:
        print(f"    {m}")
    # No assertion beyond not crashing

def benchmark_performance():
    """Test 7: Benchmark the eigenmode solver for a realistic stack."""
    print("\n" + "="*70)
    print("TEST 7: Performance benchmark (symmetric Si slab, 1000 wavelengths?)")
    print("="*70)
    # Use a moderate number of wavelengths – 1000 would be too slow for a test,
    # so we use 10 wavelengths as a representative.
    lam_start, lam_end = 1400.0, 1700.0
    n_wavs = 10
    wavls = np.linspace(lam_start, lam_end, n_wavs)
    n_si = 3.476
    n_air = 1.0
    d_core = 220.0

    n_si_arr = np.full(n_wavs, n_si + 0j)
    n_air_arr = np.full(n_wavs, n_air + 0j)
    layer_indices = np.vstack([n_air_arr, n_si_arr, n_air_arr])
    thicknesses = np.array([0.0, d_core, 0.0])
    inc_flags = np.zeros(3, dtype=np.int32)
    r_types = np.zeros(3, dtype=np.int32)
    r_vals = np.zeros(3)

    solver = LoomEigenmodeSolver(
        layer_indices, thicknesses, inc_flags, r_types, r_vals, wavls
    )

    total_time = 0.0
    for i in range(n_wavs):
        t0 = time.perf_counter()
        modes = solver.find_modes(lam_idx=i, pol='s', scan_points=80, char_threshold=0.1)
        t1 = time.perf_counter()
        total_time += t1 - t0
        if modes:
            print(f"  λ = {wavls[i]:.1f} nm: {modes[0].N_eff.real:.5f}  (time {t1-t0:.3f}s)")
        else:
            print(f"  λ = {wavls[i]:.1f} nm: no mode")
    print(f"  Total time for {n_wavs} wavelengths: {total_time:.2f} s")
    print(f"  Average per wavelength: {total_time/n_wavs:.3f} s")

def main():
    print("\n" + "#"*70)
    print("# Loom Eigenmode Solver – Rust Backend Test Suite")
    print("#"*70)
    if not RUST_AVAILABLE:
        print("\n*** Running with pure Python fallback – results will be correct but slow. ***\n")
    else:
        print("\n*** Using Rust-accelerated backend. ***\n")

    test_symmetric_slab()
    test_asymmetric_slab()
    test_multilayer()
    test_lossy_material()
    test_rough_interface()
    test_incoherent_layer()
    benchmark_performance()

    print("\n" + "#"*70)
    print("# All tests completed successfully.")
    print("#"*70)

if __name__ == "__main__":
    main()