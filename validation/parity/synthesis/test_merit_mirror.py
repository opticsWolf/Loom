#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror of ``synthesis/merit.rs`` unit tests through ``navette._smatrix``.

All 30 Rust tests translated 1:1 — same grids, same hand-computed
expectations, same tolerances. ``MeritTarget`` structs become
``MeritSpec.add_target`` calls; ``SimCurves`` via ``sim_curves_from_arrays``
(or the native class directly for complex rows).

``constraint_kind_from_str`` has no binding; it is mirrored as
accept(e/a/b/r/c) + reject("x") at ``add_target``.

Run explicitly:  python validation/parity/synthesis/test_merit_mirror.py
"""

import sys

import numpy as np

from navette._smatrix import MeritSpec, SimCurves
from navette.synthesis import sim_curves_from_arrays

FAILURES = []
NW = 5
WL5 = np.array([400., 500., 600., 700., 800.])


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


def sim_rs(vals):
    return sim_curves_from_arrays(np.array([0.0]), WL5, {"Rs": np.asarray(vals).reshape(1, 5)})


def add(spec, key, wl, targets, tols, kind="e", transform="linear", nf=1.0,
        band=None, phase=False, dp=None, weight=1.0, count=None, integral=False):
    spec.add_target(key, np.asarray(wl, dtype=np.float64),
                    np.asarray(targets, dtype=np.float64),
                    np.asarray(tols, dtype=np.float64), kind, transform, float(nf),
                    band=(np.asarray(band, dtype=np.float64) if band is not None else None),
                    phase=phase, differential_passes=dp, weight=weight,
                    count_norm=count, integral=integral)


def expect_raise(name, thunk, needle=""):
    try:
        thunk()
        check(name, False, "no error raised")
    except Exception as e:
        check(name, needle in str(e) if needle else True, f"({type(e).__name__})")


def test_basics():
    print("--- linear/exact/above/below/residuals ---")
    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    nf = 1.0 / 0.75
    add(spec, k, [400., 500.], [0.5 * nf, 1.0 * nf], [0.01, 0.01], nf=nf)
    check("linear_fold_exact_zero",
          spec.merit(sim_curves_from_arrays(np.array([0.0]), WL5,
                                            {"Rs": np.array([[0.5, 1.0, 0.9, 0.8, 0.7]])}), 1e6) == 0.0)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400.], [0.5], [0.1])
    check("exact_residual_hand",
          abs(spec.merit(sim_rs([0.55, 0., 0., 0., 0.]), 1e6) - 0.25) < 1e-14)

    def mk(kind):
        s = MeritSpec()
        kk = s.add_key(0.0, "Rs")
        add(s, kk, [400.], [0.5], [0.1], kind=kind)
        return s
    above, below = mk("a"), mk("b")
    hi, lo = sim_rs([0.6, 0., 0., 0., 0.]), sim_rs([0.3, 0., 0., 0., 0.])
    check("above_satisfied", above.merit(hi, 0.0) == 0.0)
    check("above_active", abs(above.merit(lo, 0.0) - 4.0) < 1e-14)
    check("below_satisfied", below.merit(lo, 0.0) == 0.0)
    check("below_active", abs(below.merit(hi, 0.0) - 1.0) < 1e-14)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400., 500.], [0.5, 0.5], [0.1, 0.1], kind="a")
    out = np.asarray(spec.residuals(sim_rs([0.6, 0.3, 0., 0., 0.])))
    check("residual_fixed_len", len(out) == 2 and out[0] == 0.0 and abs(out[1] + 2.0) < 1e-14
          and spec.n_residuals() == 2, f"{out}")


def test_log_phase_interp():
    print("--- log/phase/interp/extrapolation ---")
    targets = np.array([0.01, 1.0])
    nf = 1.0 / max(float(np.mean(np.abs(np.log10(np.maximum(targets, 1e-12))))), 1e-12)
    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400.], [np.log10(0.01) * nf], [0.05], transform="log", nf=nf)
    check("log_exact_zero", abs(spec.merit(sim_rs([0.01, 0., 0., 0., 0.]), 0.0)) < 1e-14)
    m10 = spec.merit(sim_rs([0.1, 0., 0., 0., 0.]), 0.0)
    check("log_x10", abs(m10 - (nf / 0.05) ** 2) < 1e-9 * (nf / 0.05) ** 2, f"{m10:.6f}")

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400.], [0.0], [1.0], transform="phase")
    m = spec.merit(sim_rs([np.pi + 0.1, 0., 0., 0., 0.]), 0.0)
    check("phase_wrap", abs(m - (np.pi - 0.1) ** 2) < 1e-12, f"{m:.6f}")

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [450., 550.], [0.5, 1.5], [1.0, 1.0])
    m = spec.merit(sim_curves_from_arrays(np.array([0.0]), WL5,
                                          {"Rs": np.array([[0., 1., 2., 3., 4.]])}), 0.0)
    check("misaligned_interp", abs(m) < 1e-14, f"{m:.2e}")

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [350.], [0.3], [1.0])
    add(spec, k, [900.], [0.0], [1.0])
    check("extrap_clamp_overlap",
          spec.merit(sim_rs([0.3, 0., 0., 0., 0.]), 0.0) == 0.0 and spec.n_residuals() == 2)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Tp")
    add(spec, k, [400.], [0.0], [1.0])
    add(spec, k, [500.], [0.0], [1.0])
    k2 = spec.add_key(0.0, "Rs")
    add(spec, k2, [400.], [0.0], [1.0])
    check("missing_penalty_once", spec.merit(sim_rs([0.] * 5), 123.0) == 123.0)
    expect_raise("missing_residual_err", lambda: spec.residuals(sim_rs([0.] * 5)), "Tp")

    # angle row: demand at 25° picks the 30° row.
    sim = sim_curves_from_arrays(np.array([0., 30., 60.]), np.array([500.]),
                                 {"Ru": np.array([[10.], [20.], [30.]])})
    spec = MeritSpec()
    k = spec.add_key(25.0, "Ru")
    add(spec, k, [500.], [20.0], [0.5])
    out = np.asarray(spec.residuals(sim))
    check("angle_argmin", len(out) == 1 and abs(out[0]) < 1e-14, f"{out}")

    # aligned vs interp path.
    vals = [0.31, 0.52, 0.66, 0.71, 0.90]
    sa = MeritSpec()
    ka = sa.add_key(0.0, "Rs")
    add(sa, ka, [500.], [0.42], [0.07])
    ra = float(np.asarray(sa.residuals(sim_rs(vals)))[0])
    shifted = sim_curves_from_arrays(np.array([0.0]),
                                     WL5 + 1e-11 * np.arange(5),
                                     {"Rs": np.array([vals])})
    su = MeritSpec()
    ku = su.add_key(0.0, "Rs")
    add(su, ku, [500. + 1e-11], [0.42], [0.07])
    ru = float(np.asarray(su.residuals(shifted))[0])
    check("aligned_vs_interp", abs(ra - ru) < 1e-9, f"{ra:.6f} vs {ru:.6f}")


def test_kinds():
    print("--- kinds: from_str/r/c, two-key, absorption ---")
    for kind in ("e", "a", "b", "r", "c"):
        s = MeritSpec()
        kk = s.add_key(0.0, "Rs")
        add(s, kk, [400.], [0.5], [0.1], kind=kind,
            band=[0.05] if kind in ("r", "c") else None)
        check(f"kind_{kind}_accept", True)
    expect_raise("kind_x_reject",
                 lambda: add(MeritSpec(), MeritSpec().add_key(0.0, "Rs"),
                             [400.], [0.5], [0.1], kind="x"))

    def mkr(band):
        s = MeritSpec()
        kk = s.add_key(0.0, "Rs")
        add(s, kk, [400.], [0.5], [0.1], band=band, kind="r")
        return s
    spec = mkr([0.05])
    check("range_dead", spec.merit(sim_rs([0.5, 0., 0., 0., 0.]), 0.0) == 0.0
          and spec.merit(sim_rs([0.53, 0., 0., 0., 0.]), 0.0) == 0.0
          and abs(spec.merit(sim_rs([0.6, 0., 0., 0., 0.]), 0.0) - 0.25) < 1e-14)
    bare = mkr(None)
    check("range_bare", bare.merit(sim_rs([0.55, 0., 0., 0., 0.]), 0.0) == 0.0
          and abs(bare.merit(sim_rs([0.7, 0., 0., 0., 0.]), 0.0) - 1.0) < 1e-14)

    def mkc(band):
        s = MeritSpec()
        kk = s.add_key(0.0, "Rs")
        add(s, kk, [400.], [0.5], [0.1], band=band, kind="c")
        return s
    spec = mkc([0.05])
    check("centerband",
          spec.merit(sim_rs([0.5, 0., 0., 0., 0.]), 0.0) == 0.0
          and abs(spec.merit(sim_rs([0.52, 0., 0., 0., 0.]), 0.0) - 0.16) < 1e-14
          and abs(spec.merit(sim_rs([0.55, 0., 0., 0., 0.]), 0.0) - 1.0) < 1e-14
          and abs(spec.merit(sim_rs([0.6, 0., 0., 0., 0.]), 0.0) - 1.25) < 1e-14)
    check("centerband_bare",
          abs(mkc(None).merit(sim_rs([0.6, 0., 0., 0., 0.]), 0.0) - 1.0) < 1e-14)

    spec = MeritSpec()
    k0 = spec.add_key(0.0, "Rs")
    add(spec, k0, [400.], [0.5], [0.1])
    k1 = spec.add_key(0.0, "Rp")
    add(spec, k1, [400.], [0.5], [0.1])
    sim = sim_curves_from_arrays(np.array([0.0]), WL5,
                                 {"Rs": np.array([[0.6, 0., 0., 0., 0.]]),
                                  "Rp": np.array([[0.6, 0., 0., 0., 0.]])})
    check("two_keys_no_double", abs(spec.merit(sim, 1e6) - 2.0) < 1e-14)

    spec = MeritSpec()
    k = spec.add_key(0.0, "As")
    add(spec, k, [400.], [0.1], [0.05])
    sim = sim_curves_from_arrays(np.array([0.0]), WL5,
                                 {"Rs": np.array([[0.6, 0., 0., 0., 0.]]),
                                  "Ts": np.array([[0.3, 0., 0., 0., 0.]])})
    check("absorption_companions", spec.merit(sim, 1e6) < 1e-28
          and abs(float(np.asarray(spec.residuals(sim))[0])) < 1e-14)
    sim2 = sim_curves_from_arrays(np.array([0.0]), WL5,
                                  {"Rs": np.array([[0.6, 0., 0., 0., 0.]])})
    check("absorption_missing", spec.merit(sim2, 123.0) == 123.0)
    expect_raise("absorption_err", lambda: spec.residuals(sim2), "As")
    spec_u = MeritSpec()
    ku = spec_u.add_key(0.0, "Au")
    add(spec_u, ku, [400.], [0.2], [0.1])
    simu = sim_curves_from_arrays(np.array([0.0]), WL5,
                                  {"Rs": np.array([[0.6, 0., 0., 0., 0.]]),
                                   "Ts": np.array([[0.3, 0., 0., 0., 0.]]),
                                   "Ru": np.array([[0.5, 0., 0., 0., 0.]]),
                                   "Tu": np.array([[0.3, 0., 0., 0., 0.]])})
    check("absorption_unpolarized", spec_u.merit(simu, 1e6) < 1e-28)
    simu2 = sim_curves_from_arrays(np.array([0.0]), WL5,
                                   {"Tu": np.array([[0.3, 0., 0., 0., 0.]])})
    expect_raise("absorption_u_err", lambda: spec_u.residuals(simu2), "Au")


def cplx_sim(first, total_d=0.0, code="Rs", wl=None, angs=None):
    wl = WL5 if wl is None else wl
    angs = np.array([0.0]) if angs is None else angs
    n = angs.size * wl.size
    sim = SimCurves(angs, wl, float(total_d), 1.0, 1.0)
    arr = np.zeros(n, dtype=np.complex128)
    arr[0] = first
    sim.set_complex(code, arr)
    return sim


def test_phase_pd():
    print("--- phase + differential-phase ---")
    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400.], [0.3], [0.05], transform="linear", phase=True)
    sim = cplx_sim(0.5 * np.exp(1j * 0.3))
    check("phase_samples_arg", spec.merit(sim, 1e6) < 1e-28)
    sim_none = SimCurves(np.array([0.0]), WL5, 0.0, 1.0, 1.0)
    check("phase_missing", spec.merit(sim_none, 123.0) == 123.0)
    expect_raise("phase_missing_err", lambda: spec.residuals(sim_none), "Rs")

    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    add(spec, k, [400.], [0.3], [0.05], transform="phase", phase=True)
    sim = cplx_sim(0.7 * np.exp(1j * (0.3 - 0.01 + 2 * np.pi)), code="Ts")
    check("phase_wraps", abs(spec.merit(sim, 1e6) - 0.04) < 1e-12)

    spec = MeritSpec()
    k = spec.add_key(0.0, "As")
    expect_raise("phase_on_absorption",
                 lambda: add(spec, k, [400.], [0.0], [0.1], transform="linear", phase=True))
    spec2 = MeritSpec()
    k2 = spec2.add_key(0.0, "Ru")
    expect_raise("phase_on_unpolarized",
                 lambda: add(spec2, k2, [400.], [0.0], [0.1], transform="linear", phase=True))

    # PD: D = 100 nm air, ref(400) = pi/2.
    delta = 0.3 - np.pi / 2
    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    add(spec, k, [400.], [delta], [0.05], transform="phase", phase=True, dp=1.0)
    sim = cplx_sim(0.7 * np.exp(1j * 0.3), total_d=100.0, code="Ts")
    check("pd_subtracts_ref", spec.merit(sim, 1e6) < 1e-28)
    aspec = MeritSpec()
    ka = aspec.add_key(0.0, "Ts")
    add(aspec, ka, [400.], [delta], [0.05], transform="phase", phase=True)
    check("pd_absolute_contrast",
          abs(aspec.merit(sim, 1e6) - ((0.3 - delta) / 0.05) ** 2) < 1e-9)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    add(spec, k, [400.], [0.3], [0.05], transform="phase", phase=True, dp=1.0)
    sim0 = cplx_sim(0.7 * np.exp(1j * 0.3), total_d=0.0, code="Ts")
    check("pd_zero_d_absolute", spec.merit(sim0, 1e6) < 1e-28)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    add(spec, k, [400.], [0.3 - np.pi], [0.05], transform="phase", phase=True, dp=2.0)
    check("pd_passes_scale", spec.merit(
        cplx_sim(0.7 * np.exp(1j * 0.3), total_d=100.0, code="Ts"), 1e6) < 1e-28)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    expect_raise("pd_needs_phase",
                 lambda: add(spec, k, [400.], [0.0], [0.1], transform="linear", dp=1.0))
    for bad in (-1.0, float("nan"), float("inf")):
        s = MeritSpec()
        kk = s.add_key(0.0, "Ts")
        expect_raise(f"pd_bad_passes_{bad}",
                     lambda kk=kk, s=s: add(s, kk, [400.], [0.0], [0.1],
                                            transform="phase", phase=True, dp=bad))


def test_weight_integral_back_errors():
    print("--- weight/count/integral/back/validation ---")
    def mk(weight=1.0, count=None):
        s = MeritSpec()
        kk = s.add_key(0.0, "Rs")
        add(s, kk, [400., 500.], [0.5, 0.5], [0.1, 0.1], weight=weight, count=count)
        return s
    sim = sim_rs([0.6, 0.4, 0.0, 0.0, 0.0])
    check("base_2", abs(mk().merit(sim, 1e6) - 2.0) < 1e-12)
    sw = mk(weight=2.0)
    check("weight_4", abs(sw.merit(sim, 1e6) - 4.0) < 1e-12
          and abs(abs(float(np.asarray(sw.residuals(sim))[0])) - np.sqrt(2.0)) < 1e-12)
    check("count_1", abs(mk(count=2.0).merit(sim, 1e6) - 1.0) < 1e-12)
    check("weight_count_3", abs(mk(weight=3.0, count=2.0).merit(sim, 1e6) - 3.0) < 1e-12)
    for w, c in [(-1.0, None), (float("nan"), None), (1.0, 0.0), (1.0, -2.0)]:
        s = MeritSpec()
        kk = s.add_key(0.0, "Rs")
        expect_raise(f"trust_w{w}_c{c}",
                     lambda s=s, kk=kk: add(s, kk, [400.], [0.5], [0.1], weight=w, count=c))

    def mki(kind="e", band=None):
        s = MeritSpec()
        kk = s.add_key(0.0, "Rs")
        add(s, kk, [400., 500., 600.], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1],
            kind=kind, band=band, integral=True)
        return s
    spec = mki()
    check("integral_zero", spec.merit(sim_rs([0.6, 0.5, 0.4, 0., 0.]), 1e6) < 1e-28
          and len(np.asarray(spec.residuals(sim_rs([0.6, 0.5, 0.4, 0., 0.])))) == 1
          and abs(spec.merit(sim_rs([0.7, 0.6, 0.5, 0., 0.]), 1e6) - 1.0) < 1e-12)
    sa = mki("a")
    check("integral_above",
          sa.merit(sim_rs([0.7, 0.7, 0.4, 0., 0.]), 1e6) < 1e-28
          and abs(sa.merit(sim_rs([0.4, 0.4, 0.4, 0., 0.]), 1e6) - 1.0) < 1e-12)
    sr = mki("r", band=[0.05, 0.05, 0.05])
    check("integral_range",
          sr.merit(sim_rs([0.54, 0.54, 0.54, 0., 0.]), 1e6) < 1e-28
          and abs(sr.merit(sim_rs([0.6, 0.6, 0.6, 0., 0.]), 1e6) - 0.25) < 1e-12)
    s = MeritSpec()
    kk = s.add_key(0.0, "Rs")
    expect_raise("integral_rejects_count",
                 lambda: add(s, kk, [400.], [0.5], [0.1], count=3.0, integral=True))

    spec = MeritSpec()
    kr = spec.add_key(0.0, "RBs")
    add(spec, kr, [400.], [0.4], [0.05])
    ka = spec.add_key(0.0, "ABs")
    add(spec, ka, [400.], [0.1], [0.05])
    simb = sim_curves_from_arrays(np.array([0.0]), WL5,
                                  {"RBs": np.array([[0.4, 0., 0., 0., 0.]]),
                                   "TBs": np.array([[0.5, 0., 0., 0., 0.]])})
    check("back_r_abs", spec.merit(simb, 1e6) < 1e-28)

    s = MeritSpec()
    kk = s.add_key(0.0, "Rs")
    expect_raise("band_mismatch",
                 lambda: add(s, kk, [400.], [0.0], [1.0], band=[0.1, 0.2], kind="r"))
    s = MeritSpec()
    s.add_key(0.0, "Rs")
    expect_raise("length_mismatch",
                 lambda: add(s, 0, [400.], [0.0, 0.0], [1.0]))
    expect_raise("bad_key",
                 lambda: add(s, 7, [400.], [0.0], [1.0]))


if __name__ == "__main__":
    test_basics()
    test_log_phase_interp()
    test_kinds()
    test_phase_pd()
    test_weight_integral_back_errors()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
