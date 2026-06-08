#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
navette_smatrix (Rust `core_engine` / func_6) vs loom_matrix (Numba reference)
=============================================================================

Accuracy and performance suite for the unified, request-driven engine exposed
through ``navette_smatrix.ScatterMatrix``, validated against the Numba reference
in ``loom_matrix``.

What it covers
--------------
1. Leaf-function parity      — w_function, real/complex Redheffer products.
2. Class-level parity        — ScatterMatrix vs LoomScatterMatrix on the public
                               API (ellipsometry + R/T), every scenario.
3. Internal consistency      — the richer func_6 observables that have no Numba
                               reference (Stokes, DOP, Diattenuation, phases,
                               complex amplitudes, retardance) are checked
                               against each other and against the intensities.
4. Engine contract           — "output holds only what is requested"
                               (expected_keys), subset invariance, and
                               coherence-mode invariance of R/T.
5. Benchmarks                — navette vs loom per workload, plus a
                               request-granularity sweep that demonstrates the
                               payoff of the level/polarization gating.

Run
---
    python3 test_navette_smatrix.py        # full report (parity + bench)
    pytest -q test_navette_smatrix.py      # assertions only (no bench)

Tolerances
----------
The Numba build uses ``fastmath=True`` (FP re-association); the Rust build is
strict IEEE-754. Differences are expected around ~1e-9, larger where the
quantity is ill-conditioned (Psi/Delta/DOP near zero reflectance). Angular
fields (Delta, phases, retardance) are compared modulo 2*pi.

Parity uses FRONT_BLOCK coherence, because the loom reference computes the
reflection cross term as the first-coherent-block fallback rp0*conj(rs0); the
COHERENCY_MATRIX / FULLY_COHERENT modes have no Numba reference and are covered
by the consistency and contract sections instead.
"""

import sys
import time

import numpy as np

# --- backend imports (soft, so the file imports even if one is missing) ------
_IMPORT_ERR = {}
try:
    import navette.smatrix as ns
    from navette.smatrix import Request, CoherenceMode, ScatterMatrix
except Exception as e:  # pragma: no cover
    ns = None
    _IMPORT_ERR["navette_smatrix"] = repr(e)

try:
    import loom_matrix as ref
except Exception as e:  # pragma: no cover
    ref = None
    _IMPORT_ERR["loom_matrix"] = repr(e)

try:
    import pytest
except Exception:  # pragma: no cover
    pytest = None

RNG = np.random.default_rng(12345)


def _need_backends():
    if ns is None or ref is None:
        msg = "missing backend(s): " + ", ".join(
            f"{k} ({v})" for k, v in _IMPORT_ERR.items()
        )
        if pytest is not None:
            pytest.skip(msg, allow_module_level=False)
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Canonical naming: parity is compared on a single key vocabulary.
# loom's reflection ellipsometry has no _R suffix and uses Ru/Tu for averages.
# ---------------------------------------------------------------------------
LOOM_TO_CANON = {
    "Rs": "Rs", "Rp": "Rp", "Ru": "R_avg", "Ts": "Ts", "Tp": "Tp", "Tu": "T_avg",
    "Psi": "Psi_R", "Delta": "Delta_R", "DOP": "DOP_R",
    "Psi_T": "Psi_T", "Delta_T": "Delta_T", "DOP_T": "DOP_T",
    "conservation_err": "conservation",
}
ANGULAR_FIELDS = {
    "Delta_R", "Delta_T",
    "phi_rs", "phi_rp", "phi_ts", "phi_tp",
    "Retardance_R", "Retardance_T",
}


def _loom_to_canon(d):
    return {LOOM_TO_CANON.get(k, k): v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
class Scenario:
    """A physical stack + (wavelength, angle) grid, build-able on either backend.

    ``coherent`` = no incoherent boundaries (so the front block is the whole
    stack, and |r|^2 == R holds; modes are equivalent).
    ``lossless`` = real indices and no roughness (so R + T == 1 exactly).
    """

    def __init__(self, name, layer_indices, thicknesses, inc_flags, r_types,
                 r_vals, wavls, angles_deg, *, coherent, lossless):
        self.name = name
        self.layer_indices = np.ascontiguousarray(layer_indices, dtype=np.complex128)
        self.thicknesses = np.asarray(thicknesses, dtype=np.float64)
        self.inc_flags = np.asarray(inc_flags, dtype=np.int32)
        self.r_types = np.asarray(r_types, dtype=np.int32)
        self.r_vals = np.asarray(r_vals, dtype=np.float64)
        self.wavls = np.asarray(wavls, dtype=np.float64)
        self.angles_deg = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
        self.coherent = coherent
        self.lossless = lossless

    @property
    def npts(self):
        return self.wavls.size * self.angles_deg.size

    def build_loom(self, debug=False):
        return ref.LoomScatterMatrix(
            self.layer_indices, self.thicknesses, self.inc_flags,
            self.r_types, self.r_vals, self.wavls, self.angles_deg,
            theta_is_radians=False, debug=debug,
        )

    def build_navette(self, mode=None):
        if mode is None:
            mode = CoherenceMode.FRONT_BLOCK
        return ScatterMatrix(
            self.layer_indices, self.thicknesses,
            wavelengths=self.wavls, angles=self.angles_deg,
            incoherent_flags=self.inc_flags,
            roughness_types=self.r_types, roughness_values=self.r_vals,
            coherence_mode=mode, angles_in_radians=False,
        )


def scenario_bragg(n_wavs=400, n_pairs=10, n_angles=24):
    """Coherent quarter-wave Bragg mirror Ta2O5/SiO2 on BK7 (all coherent)."""
    wavls = np.linspace(300, 800, n_wavs)
    lam0 = 550.0
    n_air = np.full(n_wavs, 1.0 + 0j)
    n_bk7 = np.full(n_wavs, 1.515 + 1e-4j)
    n_sio2 = np.full(n_wavs, 1.46 + 1e-3j)
    n_ta = np.full(n_wavs, 2.10 + 0j)
    d_sio2 = lam0 / (4 * 1.46)
    d_ta = lam0 / (4 * 2.10)
    idx, thk, inc, rt, rv = [n_air], [0.0], [0], [0], [0.0]
    for _ in range(n_pairs):
        idx += [n_ta, n_sio2]; thk += [d_ta, d_sio2]
        inc += [0, 0]; rt += [1, 1]; rv += [1.0, 1.0]
    idx.append(n_bk7); thk.append(0.0); inc.append(0); rt.append(0); rv.append(0.0)
    angles = np.linspace(0, 60, n_angles)
    return Scenario("bragg_coherent", np.vstack(idx), thk, inc, rt, rv,
                    wavls, angles, coherent=True, lossless=False)


def scenario_incoherent(n_wavs=300, n_angles=16):
    """Thin coherent film on a THICK incoherent glass slab (back side)."""
    wavls = np.linspace(400, 900, n_wavs)
    n_air = np.full(n_wavs, 1.0 + 0j)
    n_film = np.full(n_wavs, 2.3 + 0.02j)
    n_glass = np.full(n_wavs, 1.52 + 5e-4j)
    idx = np.vstack([n_air, n_film, n_glass, n_air])
    thk = np.array([0.0, 120.0, 1.0e6, 0.0])   # 1 mm glass => incoherent
    inc = np.array([0, 0, 1, 0], dtype=np.int32)
    rt = np.array([0, 4, 0, 0], dtype=np.int32)  # gaussian roughness at film
    rv = np.array([0.0, 3.0, 0.0, 0.0])
    angles = np.linspace(0, 50, n_angles)
    return Scenario("incoherent_slab", idx, thk, inc, rt, rv, wavls, angles,
                    coherent=False, lossless=False)


def scenario_roughness_sweep(n_wavs=200, n_angles=12):
    """Multi-layer stack using roughness models 0..5 across interfaces."""
    wavls = np.linspace(450, 850, n_wavs)
    layers = [np.full(n_wavs, 1.0 + 0j)]
    ns_list = [1.8 + 0.05j, 1.45 + 0.0j, 2.0 + 0.1j, 1.6 + 0.01j, 1.52 + 0.0j]
    for c in ns_list:
        layers.append(np.full(n_wavs, c))
    idx = np.vstack(layers)
    nL = idx.shape[0]
    thk = np.array([0.0] + [80.0 + 25 * i for i in range(nL - 2)] + [0.0])
    inc = np.zeros(nL, dtype=np.int32)
    rt = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)[:nL]
    rv = np.array([0.0, 2.0, 1.5, 2.5, 1.0, 3.0])[:nL]
    angles = np.linspace(5, 55, n_angles)
    return Scenario("roughness_sweep", idx, thk, inc, rt, rv, wavls, angles,
                    coherent=True, lossless=False)


def scenario_bare(n_angles=40):
    """Bare lossless dielectric: Delta-convention + energy-conservation check."""
    n_sub = 1.5
    idx = np.vstack([np.full(1, 1.0 + 0j), np.full(1, n_sub + 0j)])
    thk = np.array([0.0, 0.0])
    inc = np.array([0, 0], dtype=np.int32)
    rt = np.array([0, 0], dtype=np.int32)
    rv = np.array([0.0, 0.0])
    angles = np.linspace(1, 89, n_angles)
    return Scenario("bare_brewster", idx, thk, inc, rt, rv,
                    np.array([550.0]), angles, coherent=True, lossless=True)


def all_scenarios():
    return [scenario_bragg(), scenario_incoherent(),
            scenario_roughness_sweep(), scenario_bare()]


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
def cmp(name, a, b, *, atol=1e-9, rtol=1e-7, angular=False, report=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if angular:
        d = np.angle(np.exp(1j * (a - b)))      # wrap to (-pi, pi]
        err = np.abs(d)
        ok = bool(np.all(err <= (atol + rtol * np.abs(a))))
    else:
        err = np.abs(a - b)
        ok = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    maxerr = float(np.nanmax(err)) if err.size else 0.0
    if report is not None:
        report.append((name, ok, maxerr))
    return ok, maxerr


def _tol_for(field):
    if field.startswith(("Psi", "Delta", "DOP")):
        return dict(atol=1e-7, rtol=1e-6)
    return dict(atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------
# 1. Leaf-function parity
# ---------------------------------------------------------------------------
def check_leaf(report):
    ok_all = True
    # w_function across all roughness models and random complex q
    werr = 0.0
    for rt in range(0, 6):
        for _ in range(200):
            q = complex(RNG.normal(), RNG.normal())
            a = ref.w_function(q, np.int32(rt))
            b = ns.w_function(q, rt)
            werr = max(werr, abs(a - b))
    ok = werr < 1e-12
    report.append(("leaf/w_function", ok, werr)); ok_all &= ok
    # real Redheffer
    rerr = 0.0
    for _ in range(500):
        v = RNG.normal(size=8) * 0.5
        a = np.asarray(ref.redheffer_product_real(*v), float)
        b = np.asarray(ns.redheffer_product_real(*v), float)
        rerr = max(rerr, float(np.max(np.abs(a - b))))
    ok = rerr < 1e-12
    report.append(("leaf/redheffer_real", ok, rerr)); ok_all &= ok
    # complex Redheffer
    cerr = 0.0
    for _ in range(500):
        v = [complex(*(RNG.normal(size=2) * 0.5)) for _ in range(8)]
        a = np.asarray(ref.redheffer_product_complex_field(*v), complex)
        b = np.asarray(ns.redheffer_product_complex_field(*v), complex)
        cerr = max(cerr, float(np.max(np.abs(a - b))))
    ok = cerr < 1e-12
    report.append(("leaf/redheffer_complex", ok, cerr)); ok_all &= ok
    return ok_all


# ---------------------------------------------------------------------------
# 2. Class-level parity (navette vs loom), FRONT_BLOCK mode
# ---------------------------------------------------------------------------
def check_parity(scn, report):
    ok_all = True
    loom = scn.build_loom(debug=True)
    nav = scn.build_navette(CoherenceMode.FRONT_BLOCK)

    # --- ellipsometry (reflection + transmission + conservation) ---
    L = _loom_to_canon(loom.compute_ellipsometry())
    N = dict(nav.ellipsometry(transmission=True))
    N["conservation"] = nav.energy_conservation()
    for canon in ["Rs", "Rp", "R_avg", "Ts", "Tp", "T_avg",
                  "Psi_R", "Delta_R", "DOP_R", "Psi_T", "Delta_T", "DOP_T",
                  "conservation"]:
        if canon not in L or canon not in N:
            continue
        ok, _ = cmp(f"{scn.name}/ellip/{canon}", L[canon], N[canon],
                    angular=(canon in ANGULAR_FIELDS),
                    report=report, **_tol_for(canon))
        ok_all &= ok

    # --- R/T, all three polarization selections (minimal masks) ---
    rt_masks = {
        "s": Request.RS | Request.TS,
        "p": Request.RP | Request.TP,
        "u": Request.RS | Request.RP | Request.TS | Request.TP
        | Request.R_AVG | Request.T_AVG,
    }
    for tag, mask in rt_masks.items():
        Lrt = _loom_to_canon(loom.compute_RT(tag))
        Nrt = dict(nav.compute(mask))
        for canon in ["Rs", "Rp", "Ts", "Tp", "R_avg", "T_avg"]:
            if canon not in Lrt or canon not in Nrt:
                continue
            ok, _ = cmp(f"{scn.name}/RT[{tag}]/{canon}", Lrt[canon], Nrt[canon],
                        report=report, **_tol_for(canon))
            ok_all &= ok
    return ok_all


# ---------------------------------------------------------------------------
# 3. Internal consistency of the richer func_6 observables (no Numba ref)
# ---------------------------------------------------------------------------
_CONS_MASK = (
    Request.RS | Request.RP | Request.TS | Request.TP
    | Request.S0_R | Request.S1_R | Request.S2_R | Request.S3_R
    | Request.S0_T | Request.S1_T | Request.S2_T | Request.S3_T
    | Request.DOP_R | Request.DOP_T | Request.DIATT_R | Request.DIATT_T
    | Request.PSI_R | Request.DELTA_R | Request.PSI_T | Request.DELTA_T
    | Request.RETARD_R | Request.RETARD_T | Request.CROSS_R | Request.CROSS_T
    | Request.RS_C | Request.RP_C | Request.TS_C | Request.TP_C
    | Request.PHI_RS | Request.PHI_RP | Request.PHI_TS | Request.PHI_TP
)


def _wrap(x):
    return np.angle(np.exp(1j * np.asarray(x, dtype=np.float64)))


def check_consistency(scn, report):
    ok_all = True
    nav = scn.build_navette(CoherenceMode.FRONT_BLOCK)
    o = dict(nav.compute(_CONS_MASK))

    def chk(label, a, b, *, atol=1e-9, rtol=1e-7, angular=False):
        nonlocal ok_all
        ok, _ = cmp(f"{scn.name}/cons/{label}", a, b,
                    atol=atol, rtol=rtol, angular=angular, report=report)
        ok_all &= ok

    Rs, Rp, Ts, Tp = o["Rs"], o["Rp"], o["Ts"], o["Tp"]

    # Stokes definitions
    chk("S0_R=Rp+Rs", o["S0_R"], Rp + Rs)
    chk("S1_R=Rp-Rs", o["S1_R"], Rp - Rs)
    chk("S0_T=Tp+Ts", o["S0_T"], Tp + Ts)
    chk("S1_T=Tp-Ts", o["S1_T"], Tp - Ts)
    chk("S2_R=-2Re(cross_R)", o["S2_R"], -2.0 * np.real(o["cross_R"]))
    chk("S3_R=-2Im(cross_R)", o["S3_R"], -2.0 * np.imag(o["cross_R"]))

    # DOP / Diattenuation from Stokes
    s0r = o["S0_R"]; s1r = o["S1_R"]; s2r = o["S2_R"]; s3r = o["S3_R"]
    chk("DOP_R", o["DOP_R"],
        np.sqrt(s1r ** 2 + s2r ** 2 + s3r ** 2) / (s0r + 1e-20), atol=1e-9)
    chk("Diatt_R=S1/S0", o["Diattenuation_R"], s1r / (s0r + 1e-20))

    # Psi / Delta from intensities and Stokes
    floor = 1e-12
    safe = Rs >= floor
    psi_expected = np.where(safe, np.arctan(np.sqrt(np.abs(Rp / np.where(safe, Rs, 1.0)))),
                            np.pi / 2.0)
    chk("Psi_R", o["Psi_R"], psi_expected, atol=1e-7, rtol=1e-6)
    chk("Delta_R=atan2(S3,S2)", o["Delta_R"], np.arctan2(s3r, s2r),
        atol=1e-7, rtol=1e-6, angular=True)

    # Retardance = arg(cross); Delta = arg(-cross) => differ by pi
    chk("Retard_R=arg(cross_R)", o["Retardance_R"], np.angle(o["cross_R"]),
        atol=1e-7, rtol=1e-6, angular=True)
    chk("Delta_R - Retard_R == pi", _wrap(o["Delta_R"] - o["Retardance_R"] - np.pi),
        np.zeros_like(Rs), atol=1e-6, angular=True)

    # Complex amplitudes: reflection magnitude == reflectance (coherent stacks).
    if scn.coherent:
        chk("|rs_c|^2=Rs", np.abs(o["rs_c"]) ** 2, Rs, atol=1e-9)
        chk("|rp_c|^2=Rp", np.abs(o["rp_c"]) ** 2, Rp, atol=1e-9)
    # phases == arg(complex amplitude)
    chk("phi_rs=arg(rs_c)", o["phi_rs"], np.angle(o["rs_c"]),
        atol=1e-9, angular=True)
    chk("phi_ts=arg(ts_c)", o["phi_ts"], np.angle(o["ts_c"]),
        atol=1e-9, angular=True)

    # Lossless energy conservation: A == 0 within tolerance.
    if scn.lossless:
        chk("A_s=1-Rs-Ts~0", 1.0 - Rs - Ts, np.zeros_like(Rs), atol=1e-6)
        chk("A_p=1-Rp-Tp~0", 1.0 - Rp - Tp, np.zeros_like(Rp), atol=1e-6)
    return ok_all


# ---------------------------------------------------------------------------
# 4. Engine contract: only-what's-requested, subset & mode invariance
# ---------------------------------------------------------------------------
def check_contract(scn, report):
    ok_all = True
    nav = scn.build_navette(CoherenceMode.FRONT_BLOCK)

    def note(label, ok):
        nonlocal ok_all
        report.append((f"{scn.name}/contract/{label}", ok, 0.0))
        ok_all &= ok

    # (a) output holds exactly the requested keys
    for mask in [Request.RS,
                 Request.RS | Request.TS,
                 Request.PSI_R | Request.DELTA_R,
                 Request.RS_C,
                 Request.DISP_R_S]:
        out = nav.compute(mask)
        note(f"keys({int(mask):#x})", set(out) == set(ns.expected_keys(mask)))

    # (b) subset invariance: a value is identical whether requested alone or
    #     inside a larger request (level/lazy-derive must not change numbers)
    full = nav.compute(
        Request.RS | Request.RP | Request.TS | Request.TP
        | Request.PSI_R | Request.DELTA_R | Request.DOP_R
    )
    only_rs = nav.compute(Request.RS)
    note("subset/Rs", np.array_equal(full["Rs"], only_rs["Rs"]))
    only_psi = nav.compute(Request.PSI_R | Request.RS | Request.RP)
    note("subset/Psi_R", np.array_equal(full["Psi_R"], only_psi["Psi_R"]))

    # (c) intensities are invariant to coherence mode (cross channel can't move
    #     R/T): FRONT_BLOCK vs COHERENCY_MATRIX must be bit-for-bit identical
    nav_b = scn.build_navette(CoherenceMode.COHERENCY_MATRIX)
    mask_rt = Request.RS | Request.RP | Request.TS | Request.TP
    a = nav.compute(mask_rt)
    b = nav_b.compute(mask_rt)
    same = all(np.array_equal(a[k], b[k]) for k in ("Rs", "Rp", "Ts", "Tp"))
    note("mode_invariant_RT", same)
    return ok_all


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def test_leaf_functions_match():
    _need_backends()
    report = []
    assert check_leaf(report), _fmt_fail(report)


def test_parity_all_scenarios():
    _need_backends()
    report, ok = [], True
    for scn in all_scenarios():
        ok &= check_parity(scn, report)
    assert ok, _fmt_fail(report)


def test_consistency_all_scenarios():
    _need_backends()
    report, ok = [], True
    for scn in all_scenarios():
        ok &= check_consistency(scn, report)
    assert ok, _fmt_fail(report)


def test_contract_all_scenarios():
    _need_backends()
    report, ok = [], True
    for scn in all_scenarios():
        ok &= check_contract(scn, report)
    assert ok, _fmt_fail(report)


def _fmt_fail(report):
    bad = [(n, e) for (n, ok, e) in report if not ok]
    return "Failures:\n" + "\n".join(f"  {n}: max_err={e:.3e}" for n, e in bad)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def _bench(fn, iters):
    fn()  # warm-up (Numba JIT / first-call / caches)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def benchmark_backends(iters=20):
    print("\n" + "=" * 78)
    print("BENCHMARK  navette (Rust func_6) vs loom (Numba)   [mean ms/call]")
    print("=" * 78)
    scn = scenario_bragg(n_wavs=1000, n_pairs=10, n_angles=47)
    npts = scn.npts
    loom = scn.build_loom(debug=False)
    nav = scn.build_navette(CoherenceMode.FRONT_BLOCK)
    print(f"  Workload: {scn.wavls.size} lambda x {scn.angles_deg.size} angles "
          f"= {npts:,} pts, {iters} iters\n")

    rt_u = Request.RS | Request.RP | Request.TS | Request.TP | Request.R_AVG | Request.T_AVG
    cases = [
        ("ellipsometry", lambda: loom.compute_ellipsometry(),
                          lambda: nav.ellipsometry(transmission=True)),
        ("R/T (u)",       lambda: loom.compute_RT("u"),
                          lambda: nav.compute(rt_u)),
        ("R/T (s only)",  lambda: loom.compute_RT("s"),
                          lambda: nav.compute(Request.RS | Request.TS)),
    ]
    rows = [("kernel", "Numba", "Rust", "speedup", "M pts/s (Rust)")]
    for label, lf, rf in cases:
        tl = _bench(lf, iters)
        tr = _bench(rf, iters)
        rows.append((label, f"{tl*1e3:.2f}", f"{tr*1e3:.2f}",
                     f"{tl/tr:.2f}x", f"{npts/tr/1e6:.1f}"))
    _print_table(rows)


def benchmark_request_granularity(iters=20):
    print("\n" + "=" * 78)
    print("BENCHMARK  request granularity (navette only)   [mean ms/call]")
    print("=" * 78)
    scn = scenario_bragg(n_wavs=1000, n_pairs=10, n_angles=47)
    npts = scn.npts
    nav = scn.build_navette(CoherenceMode.FRONT_BLOCK)
    print(f"  Workload: {npts:,} pts, {iters} iters")
    print("  (shows the payoff of level + polarization gating)\n")

    everything = Request(0)
    for r in Request:
        # union of all single-bit observables (skip the convenience bundles)
        if bin(int(r)).count("1") == 1:
            everything |= r

    cases = [
        ("Rs only            (int, 1 pol)", Request.RS),
        ("Rs+Ts              (int, 1 pol)", Request.RS | Request.TS),
        ("R/T unpol          (int, 2 pol)",
         Request.RS | Request.RP | Request.TS | Request.TP),
        ("rs_c               (cplx, 1 pol)", Request.RS_C),
        ("ellipsometry R     (cross, 2 pol)",
         Request.PSI_R | Request.DELTA_R | Request.DOP_R),
        ("everything         (cross, 2 pol)", everything),
    ]
    rows = [("request", "ms", "M pts/s", "vs Rs")]
    base = None
    for label, mask in cases:
        t = _bench(lambda m=mask: nav.compute(m), iters)
        if base is None:
            base = t
        rows.append((label, f"{t*1e3:.2f}", f"{npts/t/1e6:.1f}", f"{t/base:.2f}x"))
    _print_table(rows)


def _print_table(rows):
    w = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for i, r in enumerate(rows):
        print("  " + "  ".join(str(r[c]).ljust(w[c]) for c in range(len(r))))
        if i == 0:
            print("  " + "  ".join("-" * w[c] for c in range(len(r))))


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
def full_report():
    _need_backends()
    print("=" * 78)
    print("ACCURACY  navette_smatrix (Rust func_6) vs loom_matrix (Numba)")
    print("=" * 78)

    report = []
    overall = True

    leaf_ok = check_leaf(report)
    print(f"  {'leaf functions':<22} {'PASS' if leaf_ok else 'FAIL'}")
    overall &= leaf_ok

    for scn in all_scenarios():
        p = check_parity(scn, report)
        c = check_consistency(scn, report)
        k = check_contract(scn, report)
        ok = p and c and k
        overall &= ok
        flags = f"parity={'ok' if p else 'X'} consistency={'ok' if c else 'X'} contract={'ok' if k else 'X'}"
        print(f"  {scn.name:<22} {'PASS' if ok else 'FAIL'}   ({flags})")

    print("\n  Worst max-abs-error per field (top 15):")
    worst = {}
    for n, ok, e in report:
        key = n.split("/", 1)[1] if "/" in n else n
        worst[key] = max(worst.get(key, 0.0), e)
    for k in sorted(worst, key=lambda x: -worst[x])[:15]:
        print(f"    {k:<30} {worst[k]:.3e}")

    fails = [(n, e) for (n, ok, e) in report if not ok]
    if fails:
        print("\n  FAILURES:")
        for n, e in fails:
            print(f"    {n:<40} max_err={e:.3e}")

    return overall


if __name__ == "__main__":
    if ns is None or ref is None:
        print("Cannot run: " + "; ".join(f"{k}: {v}" for k, v in _IMPORT_ERR.items()))
        sys.exit(2)
    ok = full_report()
    benchmark_backends()
    benchmark_request_granularity()
    print("\n" + "=" * 78)
    print("RESULT:", "ALL ACCURACY CHECKS PASS" if ok else "ACCURACY FAILURES FOUND")
    print("=" * 78)
    sys.exit(0 if ok else 1)
