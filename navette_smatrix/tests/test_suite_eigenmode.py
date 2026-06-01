#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test + benchmark suite for the Loom eigenmode solver.

Design goals (vs. the previous harness):
  * Physics correctness is checked through the *same* public API
    (``LoomEigenmodeSolver.find_modes``) regardless of which backend is active,
    so a test never silently measures the wrong code path.
  * Backend parity: for every physical case the Rust backend and the pure
    Python/SciPy fallback must agree on the modes they find. This is what
    actually pins the Rust implementation to the reference.
  * The Rust primitives (scan_landscape / find_local_minima / nelder_mead /
    field_profile) are exercised directly and cross-checked against Python.
  * Regression tests for the bugs fixed in the rewrite:
        - find_local_minima must threshold on the MEDIAN, not the mean
        - the first/last real columns must be skipped
        - hoisting inv_n must not change results
  * The benchmark is an honest A/B: identical solver, identical algorithm,
    identical settings — only the backend is toggled.

Run standalone:
    python test_eigenmode.py                 # tests + benchmark
    python test_eigenmode.py --no-bench      # tests only
    python test_eigenmode.py --bench-only    # benchmark only
    python test_eigenmode.py --scan 160 --nwav 25   # heavier benchmark

Exit status is non-zero if any test fails (CI-friendly).
"""

import argparse
import sys
import time

import numpy as np

import loom_eigenmode as le
from loom_eigenmode import LoomEigenmodeSolver

RUST = le._RUST_PRESENT


# ───────────────────────── tiny test framework ──────────────────────────────

class SkipTest(Exception):
    pass


def skip(msg):
    raise SkipTest(msg)


_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


def approx(a, b, atol=1e-8, rtol=0.0):
    return abs(a - b) <= atol + rtol * abs(b)


# ───────────────────────────── helpers ──────────────────────────────────────

def analytic_slab_te0(n_core, n_clad, d_core, lam):
    """TE0 effective index of a symmetric slab via the dispersion relation."""
    k0 = 2 * np.pi / lam

    def f(neff):
        if neff <= n_clad or neff >= n_core:
            return 1e6
        kx = k0 * np.sqrt(n_core ** 2 - neff ** 2)
        gamma = k0 * np.sqrt(neff ** 2 - n_clad ** 2)
        return kx * d_core / 2 - np.arctan(gamma / kx)

    lo, hi = n_clad + 1e-6, n_core - 1e-6
    for _ in range(200):
        mid = (lo + hi) / 2
        if abs(f(mid)) < 1e-12:
            return mid
        if f(lo) * f(mid) < 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def build_solver(layer_n, thick, rtypes=None, rvals=None, wavls=None):
    """Construct a solver from per-layer (wavelength-independent) indices."""
    wavls = np.array([1550.0]) if wavls is None else np.asarray(wavls, float)
    n_wavs = len(wavls)
    layer_indices = np.vstack([np.full(n_wavs, complex(n)) for n in layer_n])
    thick = np.asarray(thick, float)
    nl = len(layer_n)
    rtypes = np.zeros(nl, np.int32) if rtypes is None else np.asarray(rtypes, np.int32)
    rvals = np.zeros(nl) if rvals is None else np.asarray(rvals, float)
    inc = np.zeros(nl, np.int32)
    return LoomEigenmodeSolver(layer_indices, thick, inc, rtypes, rvals, wavls)


# standard stacks (callables so the benchmark can rebuild at many wavelengths)
N_SI, N_AIR, N_SIO2 = 3.476, 1.0, 1.444
D_CORE = 220.0


def sym_slab(wavls=None):
    return build_solver([N_AIR, N_SI, N_AIR], [0.0, D_CORE, 0.0], wavls=wavls)


def asym_slab(wavls=None):
    return build_solver([N_AIR, N_SI, N_SIO2], [0.0, D_CORE, 0.0], wavls=wavls)


def five_layer(wavls=None):
    return build_solver([N_AIR, N_SI, N_SIO2, N_SI, N_AIR],
                        [0.0, 100.0, 200.0, 100.0, 0.0], wavls=wavls)


def lossy_slab(wavls=None):
    return build_solver([N_AIR, N_SI + 0.001j, N_AIR], [0.0, D_CORE, 0.0], wavls=wavls)


def rough_slab(wavls=None):
    # Névot–Croce (type 5), σ = 2 nm on both core interfaces (indices 1 and 2).
    return build_solver([N_AIR, N_SI, N_AIR], [0.0, D_CORE, 0.0],
                        rtypes=[0, 5, 5], rvals=[0.0, 2.0, 2.0], wavls=wavls)


def find_both_backends(factory, **kw):
    """Return (python_modes, rust_modes) for the same stack/settings."""
    le.use_rust_backend(False)
    py = factory().find_modes(**kw)
    le.use_rust_backend(True)
    rs = factory().find_modes(**kw)
    return py, rs


def assert_modes_match(py, rs, atol=1e-4):
    assert len(py) == len(rs), f"mode count differs: py={len(py)} rs={len(rs)}"
    for a, b in zip(py, rs):
        d = abs(a.N_eff - b.N_eff)
        assert d < atol, f"N_eff differs by {d:.2e}: py={a.N_eff} rs={b.N_eff}"


# ─────────────────────── backend-agnostic physics ───────────────────────────

@test
def test_backend_toggle_is_consistent():
    """use_rust_backend reports the truth; can't force-on a missing extension."""
    assert le.use_rust_backend(False) is False
    forced = le.use_rust_backend(True)
    assert forced == RUST, "force-on result must match extension presence"
    assert le.rust_backend_available() == forced


@test
def test_symmetric_slab_matches_analytic():
    """TE0 of the symmetric Si slab must match the analytic dispersion root."""
    ana = analytic_slab_te0(N_SI, N_AIR, D_CORE, 1550.0)
    modes = sym_slab().find_modes(lam_idx=0, pol='s', scan_points=120,
                                  char_threshold=0.1, compute_group_index=False)
    assert modes, "no TE mode found"
    assert approx(modes[0].N_eff.real, ana, atol=1e-5), \
        f"n_eff={modes[0].N_eff.real:.6f} vs analytic {ana:.6f}"


@test
def test_asymmetric_slab_finds_modes():
    modes = asym_slab().find_modes(lam_idx=0, pol='both', scan_points=120,
                                   char_threshold=0.1, compute_group_index=False)
    assert modes, "expected at least one guided mode"
    for m in modes:
        assert N_SIO2 < m.N_eff.real < N_SI, \
            f"n_eff {m.N_eff.real:.4f} outside cladding/core window"


@test
def test_multilayer_finds_modes():
    modes = five_layer().find_modes(lam_idx=0, pol='s', scan_points=140,
                                    char_threshold=0.1, compute_group_index=False)
    assert modes, "expected guided modes in the 5-layer stack"


@test
def test_lossy_material_has_positive_imag():
    modes = lossy_slab().find_modes(lam_idx=0, pol='s', scan_points=120,
                                    char_threshold=0.2, compute_group_index=False)
    assert modes, "no mode found in lossy slab"
    assert modes[0].N_eff.imag > 1e-6, \
        f"lossy mode should have Im(N_eff)>0, got {modes[0].N_eff.imag:.2e}"
    assert modes[0].loss_dB_per_unit > 0


@test
def test_roughness_perturbs_index_slightly():
    smooth = sym_slab().find_modes(lam_idx=0, pol='s', scan_points=120,
                                   char_threshold=0.1, compute_group_index=False)
    rough = rough_slab().find_modes(lam_idx=0, pol='s', scan_points=120,
                                    char_threshold=0.1, compute_group_index=False)
    assert smooth and rough, "both smooth and rough must find a mode"
    shift = smooth[0].N_eff.real - rough[0].N_eff.real
    assert 0.0 < shift < 1e-2, f"unexpected roughness shift: {shift:.2e}"


@test
def test_field_profile_normalised_and_monotonic():
    m = sym_slab().find_modes(lam_idx=0, pol='s', scan_points=120,
                              char_threshold=0.1, compute_group_index=False)[0]
    assert len(m.z) > 1
    assert np.all(np.diff(m.z) >= 0), "z grid must be non-decreasing"
    assert approx(float(np.max(m.E_profile)), 1.0, atol=1e-6), "max|E| must be 1"
    assert np.all(m.E_profile >= 0)
    assert m.layer_bounds, "layer_bounds must be populated"


@test
def test_char_value_below_threshold_for_real_mode():
    thr = 0.05
    modes = sym_slab().find_modes(lam_idx=0, pol='s', scan_points=120,
                                  char_threshold=thr, compute_group_index=False)
    assert modes
    assert modes[0].char_value < thr, \
        f"converged char value {modes[0].char_value:.2e} should be < {thr}"


@test
def test_incoherent_flags_are_ignored_by_mode_solver():
    """The eigenmode solver works per coherent block; inc flags must not matter."""
    s0 = sym_slab()
    s1 = build_solver([N_AIR, N_SI, N_AIR], [0.0, D_CORE, 0.0])
    s1.inc_flags = np.array([0, 1, 0])  # flag the core incoherent
    kw = dict(lam_idx=0, pol='s', scan_points=120, char_threshold=0.1,
              compute_group_index=False)
    m0 = s0.find_modes(**kw)
    m1 = s1.find_modes(**kw)
    assert len(m0) == len(m1)
    assert approx(m0[0].N_eff.real, m1[0].N_eff.real, atol=1e-9)


@test
def test_no_guiding_returns_empty():
    """Core index below cladding ⇒ no guided modes, graceful empty list."""
    import warnings
    s = build_solver([1.5, 1.4, 1.5], [0.0, 200.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modes = s.find_modes(lam_idx=0, pol='s', compute_group_index=False)
    assert modes == []


@test
def test_group_index_computed_with_multiple_wavelengths():
    wavls = np.linspace(1500.0, 1600.0, 11)
    s = sym_slab(wavls=wavls)
    lam_idx = int(np.argmin(np.abs(wavls - 1550.0)))
    modes = s.find_modes(lam_idx=lam_idx, pol='s', scan_points=120,
                         char_threshold=0.1, compute_group_index=True,
                         delta_lam_nm=5.0)
    assert modes
    assert modes[0].n_group is not None, "group index should be available"
    assert np.isfinite(modes[0].n_group)


# ───────────────────── Rust ↔ Python backend parity ─────────────────────────

def _parity(factory, **kw):
    if not RUST:
        skip("Rust extension not compiled")
    py, rs = find_both_backends(factory, **kw)
    assert py, "Python fallback found no modes (test stack is wrong)"
    assert_modes_match(py, rs)


@test
def test_parity_symmetric():
    _parity(sym_slab, lam_idx=0, pol='both', scan_points=120,
            char_threshold=0.1, compute_group_index=False)


@test
def test_parity_asymmetric():
    _parity(asym_slab, lam_idx=0, pol='both', scan_points=120,
            char_threshold=0.1, compute_group_index=False)


@test
def test_parity_multilayer():
    _parity(five_layer, lam_idx=0, pol='s', scan_points=140,
            char_threshold=0.1, compute_group_index=False)


@test
def test_parity_lossy():
    _parity(lossy_slab, lam_idx=0, pol='s', scan_points=120,
            char_threshold=0.2, compute_group_index=False)


@test
def test_parity_rough():
    _parity(rough_slab, lam_idx=0, pol='s', scan_points=120,
            char_threshold=0.1, compute_group_index=False)


# ───────────────────────── Rust primitives ──────────────────────────────────

def _rust_prims():
    from navette_smatrix import (scan_landscape, find_local_minima,
                                 nelder_mead, field_profile)
    return scan_landscape, find_local_minima, nelder_mead, field_profile


@test
def test_prim_scan_matches_python_charfunc():
    """Rust scan_landscape values must equal Python _char_func evaluations."""
    if not RUST:
        skip("Rust extension not compiled")
    scan_landscape, *_ = _rust_prims()
    n = np.ascontiguousarray([N_AIR + 0j, N_SI + 0j, N_AIR + 0j])
    th = np.array([0.0, D_CORE, 0.0])
    rt = np.zeros(3, np.int32)
    rv = np.zeros(3)
    rmin, rmax, imin, imax, pr, pi = 1.0001, 3.4759, 0.0, 0.02, 12, 10
    Nr, Ni, land = scan_landscape(n, th, rt, rv, 1550.0, 0,
                                  rmin, rmax, imin, imax, pr, pi)
    land = np.asarray(land)
    assert land.shape == (pi, pr)
    # Compare a handful of cells to the Python characteristic function.
    args = (n, th, rt.astype(int), rv, 1550.0, 0)
    for i in (0, pi // 2, pi - 1):
        for j in (0, pr // 2, pr - 1):
            ref = le._char_func_xy([Nr[j], Ni[i]], *args)
            assert approx(land[i, j], ref, rtol=1e-9, atol=1e-12), \
                f"scan[{i},{j}]={land[i, j]:.6e} vs python {ref:.6e}"


@test
def test_prim_find_minima_uses_median_not_mean():
    """
    Regression: the threshold is the MEDIAN scaled by median_factor.

    Construct a landscape whose mean is wrecked by a few sentinel cells.
      * deep genuine well  (value 1e-3) -> below median -> accepted
      * shallow dip         (value 0.7) -> above median (~0.5) -> REJECTED,
        even though it sits far below the mean.
    A mean-based threshold would wrongly accept the shallow dip.
    """
    nimag, nreal = 9, 9
    Nr = np.linspace(2.0, 3.0, nreal)
    Ni = np.linspace(0.0, 0.02, nimag)
    land = np.full((nimag, nreal), 0.5)
    land[4, 4] = 1e-3                 # deep well (interior)
    land[1:4, 5:8] = 1.0             # plateau so the dip is a strict local min
    land[2, 6] = 0.7                 # shallow dip (interior)
    land[0, 0] = land[0, 1] = land[8, 8] = 1e30  # sentinels inflate the mean

    well = (round(Nr[4], 9), round(Ni[4], 9))
    dip = (round(Nr[6], 9), round(Ni[2], 9))

    le.use_rust_backend(False)
    py = {(round(r, 9), round(i, 9)) for r, i in
          le._find_local_minima_backend(land, Nr, Ni)}
    assert well in py, "deep well must be detected (median path)"
    assert dip not in py, "shallow dip must be rejected by median threshold"

    if RUST:
        _, find_local_minima, *_ = _rust_prims()
        rs = {(round(r, 9), round(i, 9)) for r, i in
              find_local_minima(np.ascontiguousarray(land, float), Nr, Ni, 1.0)}
        assert rs == py, f"Rust minima {rs} != Python {py}"


@test
def test_prim_find_minima_skips_edge_columns():
    """A minimum sitting in the first/last real column must not be reported."""
    nimag, nreal = 9, 9
    Nr = np.linspace(2.0, 3.0, nreal)
    Ni = np.linspace(0.0, 0.02, nimag)
    land = np.full((nimag, nreal), 0.5)
    land[4, 0] = 1e-4           # well in the first column (should be skipped)
    land[5, nreal - 1] = 1e-4   # well in the last column (should be skipped)

    le.use_rust_backend(False)
    cands = le._find_local_minima_backend(land, Nr, Ni)
    reals = {round(r, 9) for r, _ in cands}
    assert round(Nr[0], 9) not in reals
    assert round(Nr[-1], 9) not in reals

    if RUST:
        _, find_local_minima, *_ = _rust_prims()
        rc = find_local_minima(np.ascontiguousarray(land, float), Nr, Ni, 1.0)
        rreals = {round(r, 9) for r, _ in rc}
        assert round(Nr[0], 9) not in rreals
        assert round(Nr[-1], 9) not in rreals


@test
def test_prim_nelder_mead_matches_scipy():
    """Rust Nelder–Mead and SciPy must converge to the same minimum."""
    if not RUST:
        skip("Rust extension not compiled")
    *_, nelder_mead, _ = _rust_prims()
    n = np.ascontiguousarray([N_AIR + 0j, N_SI + 0j, N_AIR + 0j])
    th = np.array([0.0, D_CORE, 0.0])
    rt = np.zeros(3, np.int32)
    rv = np.zeros(3)
    ana = analytic_slab_te0(N_SI, N_AIR, D_CORE, 1550.0)
    x0 = (ana + 1e-3, 1e-3)
    xr, xi, cv = nelder_mead(n, th, rt, rv, 1550.0, 0, x0, 1e-3, 1e-9, 2000)
    assert approx(xr, ana, atol=1e-5), f"rust n_eff {xr:.6f} vs analytic {ana:.6f}"
    assert cv < 1e-6, f"char value not minimised: {cv:.2e}"
    # cross-check against the SciPy fallback from the identical seed
    le.use_rust_backend(False)
    pr, pi, pcv = le._polish_backend(
        [N_AIR + 0j, N_SI + 0j, N_AIR + 0j], th, np.zeros(3, int), rv,
        1550.0, 0, x0, 1e-3, 1e-9, 5000)
    assert approx(xr, pr, atol=1e-4), f"rust {xr:.6f} vs scipy {pr:.6f}"


@test
def test_prim_field_profile():
    if not RUST:
        skip("Rust extension not compiled")
    *_, nelder_mead, field_profile = _rust_prims()
    n = np.ascontiguousarray([N_AIR + 0j, N_SI + 0j, N_AIR + 0j])
    th = np.array([0.0, D_CORE, 0.0])
    rt = np.zeros(3, np.int32)
    rv = np.zeros(3)
    ana = analytic_slab_te0(N_SI, N_AIR, D_CORE, 1550.0)
    xr, xi, _ = nelder_mead(n, th, rt, rv, 1550.0, 0, (ana, 0.0), 1e-3, 1e-9, 2000)
    z, e_mag, lstart, lend, ln = field_profile(
        n, th, rt, rv, 1550.0, complex(xr, xi), 0, 50)
    z = np.asarray(z, float)
    e_mag = np.asarray(e_mag, float)
    assert len(z) > 0 and len(z) == len(e_mag)
    assert np.max(e_mag) <= 1.0 + 1e-6
    assert len(lstart) == len(lend) == len(ln)


# ─────────────────────────── test runner ────────────────────────────────────

def run_tests():
    print("=" * 70)
    print(f"EIGENMODE TEST SUITE   (Rust backend {'PRESENT' if RUST else 'ABSENT'})")
    print("=" * 70)
    npass = nfail = nskip = 0
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
        except SkipTest as e:
            nskip += 1
            print(f"  SKIP  {name}  ({e})")
        except AssertionError as e:
            nfail += 1
            print(f"  FAIL  {name}  -> {e}")
        except Exception as e:  # noqa: BLE001
            nfail += 1
            print(f"  ERROR {name}  -> {type(e).__name__}: {e}")
        else:
            npass += 1
            print(f"  PASS  {name}")
        finally:
            le.use_rust_backend(RUST)  # restore default between tests
    print("-" * 70)
    print(f"  {npass} passed, {nfail} failed, {nskip} skipped "
          f"(of {len(_TESTS)})")
    return nfail == 0


# ──────────────────────────── benchmark ─────────────────────────────────────

def _time_solver(factory, wavls, scan_points, repeat):
    s = factory(wavls=wavls)
    # warm-up (thread pool spin-up etc.), result discarded
    s.find_modes(lam_idx=0, pol='s', scan_points=scan_points,
                 char_threshold=0.1, compute_group_index=False)
    t0 = time.perf_counter()
    for _ in range(repeat):
        for i in range(len(wavls)):
            s.find_modes(lam_idx=i, pol='s', scan_points=scan_points,
                         char_threshold=0.1, compute_group_index=False)
    return (time.perf_counter() - t0) / repeat


def run_benchmark(scan_points=120, n_wav=20, repeat=2):
    print("\n" + "=" * 70)
    print("HONEST BENCHMARK  (same solver / same algorithm, backend toggled)")
    print(f"  scan_points={scan_points}, wavelengths={n_wav}, repeat={repeat}")
    print("=" * 70)
    wavls = np.linspace(1400.0, 1700.0, n_wav)
    cases = [("Symmetric slab (air/Si/air)", sym_slab),
             ("Asymmetric slab (air/Si/SiO2)", asym_slab),
             ("5-layer (air/Si/SiO2/Si/air)", five_layer)]

    for name, factory in cases:
        print(f"\n{name}")
        le.use_rust_backend(False)
        py_t = _time_solver(factory, wavls, scan_points, repeat)
        print(f"  Python : {py_t:.3f} s")
        if RUST:
            le.use_rust_backend(True)
            rs_t = _time_solver(factory, wavls, scan_points, repeat)
            print(f"  Rust   : {rs_t:.3f} s")
            print(f"  Speedup: {py_t / rs_t:.2f}x")
        else:
            print("  Rust   : (extension not compiled — skipped)")
    le.use_rust_backend(RUST)


# ────────────────────────────── main ────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-bench", action="store_true", help="run tests only")
    ap.add_argument("--bench-only", action="store_true", help="benchmark only")
    ap.add_argument("--scan", type=int, default=120, help="benchmark scan points")
    ap.add_argument("--nwav", type=int, default=20, help="benchmark wavelengths")
    ap.add_argument("--repeat", type=int, default=2, help="benchmark repeats")
    args = ap.parse_args(argv)

    ok = True
    if not args.bench_only:
        ok = run_tests()
    if not args.no_bench:
        run_benchmark(scan_points=args.scan, n_wav=args.nwav, repeat=args.repeat)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
