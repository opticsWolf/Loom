#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loom Matrix — Rust vs Python(Numba) parity & performance test suite.

For every scenario the SAME inputs are fed to:
  * the reference Numba engines in loom_matrix.py, and
  * the Rust engines in loom_matrix_rs (compiled extension),
and the outputs are compared element-by-element.

Run:
    python3 test_loom.py            # full report (parity + benchmarks)
    pytest -q test_loom.py          # parity assertions only

Notes on tolerances
-------------------
The Numba build uses fastmath=True (FP re-association, no-NaN assumptions);
the Rust build is strict IEEE-754. Differences are therefore expected at the
~1e-9 level and occasionally larger where catastrophic cancellation occurs
(e.g. near-zero reflectance). Tolerances below are chosen accordingly.
Delta/phase fields are compared modulo 2*pi.
"""

import time
import numpy as np

import loom_matrix as ref          # Numba reference
from navette import smatrix as rs         # Rust extension

RNG = np.random.default_rng(12345)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prep(layer_indices, thicknesses, inc_flags, r_types, r_vals,
          wavls, theta_deg):
    """Replicate LoomScatterMatrix.__init__ array prep so both backends get
    byte-identical inputs (the Rust engine takes the same flattened layout)."""
    wavls = np.ascontiguousarray(wavls, dtype=np.float64)
    theta = np.radians(np.atleast_1d(theta_deg).astype(np.float64))
    sin_theta = np.ascontiguousarray(np.sin(theta))
    n_layers = len(thicknesses)
    indices_T = np.ascontiguousarray(layer_indices.T, dtype=np.complex128)
    thick = np.ascontiguousarray(thicknesses, dtype=np.float64)
    inc = np.ascontiguousarray(inc_flags, dtype=np.int32)
    rt = np.ascontiguousarray(r_types, dtype=np.int32)
    rv = np.ascontiguousarray(r_vals, dtype=np.float64)
    # Rust expects the complex n-stack flattened as [re, im, re, im, ...]
    # in (wav, layer) order. indices_T is (n_wavs, n_layers) complex128.
    flat = np.ascontiguousarray(
        indices_T.view(np.float64).reshape(-1), dtype=np.float64)
    return dict(wavls=wavls, sin_theta=sin_theta, n_layers=np.int32(n_layers),
                indices_T=indices_T, n_flat=flat, thick=thick, inc=inc,
                rt=rt, rv=rv)


def run_ref_ellip(p, debug=0):
    return ref.core_engine_rigorous_ellipsometry(
        p['wavls'], p['sin_theta'], p['n_layers'], p['indices_T'],
        p['thick'], p['inc'], p['rt'], p['rv'], np.int32(debug))


def run_rs_ellip(p, debug=0):
    return rs.core_engine_rigorous_ellipsometry(
        p['wavls'], p['sin_theta'], int(p['n_layers']), p['n_flat'],
        p['thick'], p['inc'], p['rt'], p['rv'], int(debug))


def run_ref_phot(p, cs, cp):
    return ref.core_engine_photometry_only(
        p['wavls'], p['sin_theta'], p['n_layers'], p['indices_T'],
        p['thick'], p['inc'], p['rt'], p['rv'], np.int32(cs), np.int32(cp))


def run_rs_phot(p, cs, cp):
    return rs.core_engine_photometry_only(
        p['wavls'], p['sin_theta'], int(p['n_layers']), p['n_flat'],
        p['thick'], p['inc'], p['rt'], p['rv'], int(cs), int(cp))


def cmp(name, a, b, *, atol=1e-9, rtol=1e-7, angular=False, report=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if angular:
        d = np.angle(np.exp(1j * (a - b)))   # wrap to (-pi, pi]
        err = np.abs(d)
        ok = np.all(err <= (atol + rtol * np.abs(a)))
    else:
        err = np.abs(a - b)
        ok = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    maxerr = float(np.nanmax(err)) if err.size else 0.0
    if report is not None:
        report.append((name, ok, maxerr))
    return ok, maxerr


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def scenario_bragg(n_wavs=400, n_pairs=10, n_angles=24):
    """Coherent quarter-wave Bragg mirror Ta2O5/SiO2 on BK7 (all coherent)."""
    wavls = np.linspace(300, 800, n_wavs)
    lam0 = 550.0
    n_air = np.full(n_wavs, 1.0 + 0j)
    n_bk7 = np.full(n_wavs, 1.515 + 0.0001j)
    n_sio2 = np.full(n_wavs, 1.46 + 0.001j)
    n_ta = np.full(n_wavs, 2.10 + 0j)
    d_sio2 = lam0 / (4 * 1.46)
    d_ta = lam0 / (4 * 2.10)
    idx, thk, inc, rt, rv = [n_air], [0.0], [0], [0], [0.0]
    for _ in range(n_pairs):
        idx += [n_ta, n_sio2]; thk += [d_ta, d_sio2]
        inc += [0, 0]; rt += [1, 1]; rv += [1.0, 1.0]
    idx.append(n_bk7); thk.append(0.0); inc.append(0); rt.append(0); rv.append(0.0)
    angles = np.linspace(0, 60, n_angles)
    return _prep(np.vstack(idx), np.array(thk), np.array(inc),
                 np.array(rt), np.array(rv), wavls, angles)


def scenario_incoherent(n_wavs=300, n_angles=16):
    """Thin coherent film on a THICK incoherent glass slab on air (back side).
    Exercises the incoherent-flag / intensity-Redheffer path."""
    wavls = np.linspace(400, 900, n_wavs)
    n_air = np.full(n_wavs, 1.0 + 0j)
    n_film = np.full(n_wavs, 2.3 + 0.02j)
    n_glass = np.full(n_wavs, 1.52 + 0.0005j)
    idx = np.vstack([n_air, n_film, n_glass, n_air])
    thk = np.array([0.0, 120.0, 1.0e6, 0.0])     # 1 mm glass = incoherent
    inc = np.array([0, 0, 1, 0], dtype=np.int32)
    rt = np.array([0, 4, 0, 0], dtype=np.int32)   # gaussian roughness at film
    rv = np.array([0.0, 3.0, 0.0, 0.0])
    angles = np.linspace(0, 50, n_angles)
    return _prep(idx, thk, inc, rt, rv, wavls, angles)


def scenario_roughness_sweep(n_wavs=200, n_angles=12):
    """Multi-layer stack using every roughness model 0..5 across interfaces."""
    wavls = np.linspace(450, 850, n_wavs)
    layers = [np.full(n_wavs, 1.0 + 0j)]
    ns = [1.8 + 0.05j, 1.45 + 0.0j, 2.0 + 0.1j, 1.6 + 0.01j, 1.52 + 0.0j]
    for c in ns:
        layers.append(np.full(n_wavs, c))
    idx = np.vstack(layers)
    nL = idx.shape[0]
    thk = np.array([0.0] + [80.0 + 25 * i for i in range(nL - 2)] + [0.0])
    inc = np.zeros(nL, dtype=np.int32)
    rt = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)[:nL]
    rv = np.array([0.0, 2.0, 1.5, 2.5, 1.0, 3.0])[:nL]
    angles = np.linspace(5, 55, n_angles)
    return _prep(idx, thk, inc, rt, rv, wavls, angles)


def scenario_bare(n_angles=40):
    """Bare dielectric: Delta-convention check around Brewster's angle."""
    n_sub = 1.5
    idx = np.vstack([np.full(1, 1.0 + 0j), np.full(1, n_sub + 0j)])
    thk = np.array([0.0, 0.0]); inc = np.array([0, 0], dtype=np.int32)
    rt = np.array([0, 0], dtype=np.int32); rv = np.array([0.0, 0.0])
    angles = np.linspace(1, 89, n_angles)
    return _prep(idx, thk, inc, rt, rv, np.array([550.0]), angles)


SCENARIOS = {
    'bragg_coherent': scenario_bragg,
    'incoherent_slab': scenario_incoherent,
    'roughness_sweep': scenario_roughness_sweep,
    'bare_brewster': scenario_bare,
}

ELLIP_LABELS = ['Psi_R', 'Delta_R', 'DOP_R', 'Rs', 'Rp', 'Ru',
                'Psi_T', 'Delta_T', 'DOP_T', 'Ts', 'Tp', 'Tu', 'cons']
ANGULAR_IDX = {1, 7}   # Delta_R, Delta_T compared modulo 2*pi


# ---------------------------------------------------------------------------
# Parity checks (used by both pytest and the __main__ report)
# ---------------------------------------------------------------------------

def check_scenario(name, builder, report):
    p = builder()
    # Ellipsometry (debug=1 so the conservation field is exercised too)
    ref_out = run_ref_ellip(p, debug=1)
    rs_out = run_rs_ellip(p, debug=1)
    all_ok = True
    for i, lab in enumerate(ELLIP_LABELS):
        ang = i in ANGULAR_IDX
        # near-zero reflectance makes Psi/Delta ill-conditioned; relax a touch
        atol = 1e-7 if lab.startswith(('Psi', 'Delta', 'DOP')) else 1e-9
        ok, me = cmp(f"{name}/ellip/{lab}", ref_out[i], rs_out[i],
                     atol=atol, rtol=1e-6, angular=ang, report=report)
        all_ok &= ok
    # Photometry, all three modes
    for cs, cp, tag in [(1, 1, 'u'), (1, 0, 's'), (0, 1, 'p')]:
        r = run_ref_phot(p, cs, cp)
        x = run_rs_phot(p, cs, cp)
        for j, lab in enumerate(['Rs', 'Rp', 'Ts', 'Tp']):
            ok, me = cmp(f"{name}/phot[{tag}]/{lab}", r[j], x[j],
                         atol=1e-9, rtol=1e-7, report=report)
            all_ok &= ok
    return all_ok


# pytest entry points -------------------------------------------------------

def test_parity_all_scenarios():
    report = []
    ok = True
    for name, builder in SCENARIOS.items():
        ok &= check_scenario(name, builder, report)
    bad = [r for r in report if not r[1]]
    assert ok, "Mismatches:\n" + "\n".join(
        f"  {n}: max_err={e:.3e}" for n, _, e in bad)


def test_leaf_functions_match():
    # w_function across all types and random complex q
    for rt in range(0, 6):
        for _ in range(200):
            q = complex(RNG.normal(), RNG.normal())
            a = ref.w_function(q, np.int32(rt))
            b = rs.w_function(q, rt)
            assert abs(a - b) < 1e-12, (rt, q, a, b)
    # real Redheffer
    for _ in range(500):
        v = RNG.normal(size=8) * 0.5
        a = ref.redheffer_product_real(*v)
        b = rs.redheffer_product_real(*v)
        assert np.allclose(a, b, atol=1e-12)
    # complex Redheffer
    for _ in range(500):
        v = [complex(*RNG.normal(size=2) * 0.5) for _ in range(8)]
        a = ref.redheffer_product_complex_field(*v)
        b = rs.redheffer_product_complex_field(*v)
        assert np.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def _bench(fn, iters):
    fn()                       # warm-up (JIT compile / cache)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def benchmark():
    print("\n" + "=" * 74)
    print("BENCHMARKS  (mean wall-time per call; lower is better)")
    print("=" * 74)
    # A heavier workload than the parity scenarios.
    big = scenario_bragg(n_wavs=1000, n_pairs=10, n_angles=47)
    npts = big['wavls'].size * big['sin_theta'].size
    iters = 30
    print(f"  Workload: {big['wavls'].size} λ × {big['sin_theta'].size} "
          f"angles = {npts:,} points,  {iters} iters\n")

    rows = [("kernel", "Python/Numba", "Rust", "speedup", "M pts/s (Rust)")]
    cases = [
        ("ellipsometry", lambda b: run_ref_ellip(b, 1), lambda b: run_rs_ellip(b, 1)),
        ("photometry u", lambda b: run_ref_phot(b, 1, 1), lambda b: run_rs_phot(b, 1, 1)),
        ("photometry s", lambda b: run_ref_phot(b, 1, 0), lambda b: run_rs_phot(b, 1, 0)),
    ]
    for label, pf, rf in cases:
        tp = _bench(lambda: pf(big), iters)
        tr = _bench(lambda: rf(big), iters)
        rows.append((label, f"{tp*1e3:.2f} ms", f"{tr*1e3:.2f} ms",
                     f"{tp/tr:.2f}×", f"{npts/tr/1e6:.1f}"))
    w = [max(len(r[c]) for r in rows) for c in range(5)]
    for i, r in enumerate(rows):
        print("  " + "  ".join(r[c].ljust(w[c]) for c in range(5)))
        if i == 0:
            print("  " + "  ".join("-" * w[c] for c in range(5)))


def parity_report():
    print("=" * 74)
    print("PARITY  (Rust vs Numba reference)")
    print("=" * 74)
    report = []
    overall = True
    for name, builder in SCENARIOS.items():
        ok = check_scenario(name, builder, report)
        overall &= ok
        print(f"  {name:<18} {'PASS' if ok else 'FAIL'}")
    print("\n  Worst max-abs-error per field:")
    worst = {}
    for n, ok, e in report:
        key = n.split('/', 1)[1]
        worst[key] = max(worst.get(key, 0.0), e)
    for k in sorted(worst, key=lambda x: -worst[x])[:12]:
        print(f"    {k:<28} {worst[k]:.3e}")
    return overall

def benchmark_phot_vs_ellip():
    print("\n" + "=" * 74)
    print("PATH COMPARISON: Photometric vs Ellipsometric (Intensity Only)")
    print("=" * 74)
    
    # Use a heavy workload to get clean timing signals
    big = scenario_bragg(n_wavs=1000, n_pairs=10, n_angles=47)
    iters = 30
    
    # -----------------------------------------------------------------------
    # 1. VERIFY 1:1 MATCH
    # -----------------------------------------------------------------------
    # run_rs_ellip returns 13 elements.
    # indices: 3=Rs, 4=Rp, 9=Ts, 10=Tp
    out_ellip = run_rs_ellip(big, debug=0)
    
    # run_rs_phot returns 4 elements.
    # indices: 0=Rs, 1=Rp, 2=Ts, 3=Tp
    # Using cs=1, cp=1 to compute both s and p polarizations
    out_phot = run_rs_phot(big, cs=1, cp=1)
    
    try:
        np.testing.assert_allclose(out_ellip[3], out_phot[0], atol=1e-9)  # Rs
        np.testing.assert_allclose(out_ellip[4], out_phot[1], atol=1e-9)  # Rp
        np.testing.assert_allclose(out_ellip[9], out_phot[2], atol=1e-9)  # Ts
        np.testing.assert_allclose(out_ellip[10], out_phot[3], atol=1e-9) # Tp
        print("  ✓ Intensity results are mathematically identical between both paths.")
    except AssertionError as e:
        print("  ✗ Mismatch between photometric and ellipsometric intensity outputs!")
        raise e

    # -----------------------------------------------------------------------
    # 2. BENCHMARK
    # -----------------------------------------------------------------------
    # Rust
    tr_ellip = _bench(lambda: run_rs_ellip(big, 0), iters)
    tr_phot  = _bench(lambda: run_rs_phot(big, 1, 1), iters)
    
    # Numba
    tp_ellip = _bench(lambda: run_ref_ellip(big, 0), iters)
    tp_phot  = _bench(lambda: run_ref_phot(big, 1, 1), iters)
    
    npts = big['wavls'].size * big['sin_theta'].size
    print(f"\n  Workload: {big['wavls'].size} λ × {big['sin_theta'].size} angles ({npts:,} pts), {iters} iters")
    
    print("\n  [Rust Extension]")
    print(f"    Ellipsometric Path: {tr_ellip*1e3:>7.2f} ms")
    print(f"    Photometric Path:   {tr_phot*1e3:>7.2f} ms")
    print(f"    Speedup:            {tr_ellip/tr_phot:>7.2f}x faster")

    print("\n  [Python/Numba Reference]")
    print(f"    Ellipsometric Path: {tp_ellip*1e3:>7.2f} ms")
    print(f"    Photometric Path:   {tp_phot*1e3:>7.2f} ms")
    print(f"    Speedup:            {tp_ellip/tp_phot:>7.2f}x faster\n")

if __name__ == "__main__":
    print("Loom Matrix — Rust vs Python parity & benchmark\n")
    ok = parity_report()
    benchmark()
    print("\n" + "=" * 74)
    benchmark_phot_vs_ellip()
    print("RESULT:", "ALL SCENARIOS MATCH ✓" if ok else "MISMATCHES FOUND ✗")
    print("=" * 74)
