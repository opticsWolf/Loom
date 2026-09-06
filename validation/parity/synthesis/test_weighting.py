# -*- coding: utf-8 -*-
"""Parity + validation for per-target weighting, count normalization and
integral (mean) targets.

- weight w ......... frame merit x w (residuals x sqrt(w)).
- normalize_count .. frame merit / N (target-level count).
- integral ......... single residual mean(d)/mean(tol); kinds on the mean.
- All three compose (weight x integral yes; count x integral rejected)
  and regular + integral targets mix freely in one collection/run.
"""

import sys

import numpy as np

from navette.spectralweave.optical import OpticalFragment, SimulationWeaver
from navette.spectralweave.target import (
    AngularTarget,
    SpectralTarget,
    TargetCollection,
    calculate_merit,
)
from navette.synthesis import build_merit_spec, sim_curves_from_arrays
from navette.synthesis import build_needle_targets

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


WL = np.array([400.0, 500.0, 600.0])
R_VALS = np.array([0.52, 0.61, 0.69])


def weaver_merit(tc, sim_rows):
    tw = tc.build_weaver()
    sw = SimulationWeaver()
    for key, vals in sim_rows:
        sw.add_fragment(OpticalFragment(WL, np.asarray(vals), *key))
    return calculate_merit(sw.backend, tw)


def spec_merit(tc, sim):
    return build_merit_spec(tc).merit(sim, 1e6)


def flat_sim(rows):
    return sim_curves_from_arrays(np.array([0.0]), WL, rows)


def test_density():
    print("--- density: 200 vs 10, normalize, weight ---")
    base = dict(tol=0.05, tv=0.6, sv=0.61)
    merits = {}
    for n in (200, 10):
        wl = np.linspace(400.0, 600.0, n)
        tc = TargetCollection()
        tc.add(SpectralTarget(wl, np.full(n, base["tv"]), np.full(n, base["tol"]),
                              0.0, "s", "R"))
        tw = tc.build_weaver()
        sw = SimulationWeaver()
        sw.add_fragment(OpticalFragment(wl, np.full(n, base["sv"]), 0.0, "s", "R"))
        merits[n] = calculate_merit(sw.backend, tw)
    check("20:1 density ratio", abs(merits[200] / merits[10] - 20.0) < 1e-9,
          f"{merits[200]:.4f} vs {merits[10]:.4f}")
    # normalize_count equalizes exactly.
    normed = {}
    for n in (200, 10):
        wl = np.linspace(400.0, 600.0, n)
        tc = TargetCollection()
        tc.add(SpectralTarget(wl, np.full(n, base["tv"]), np.full(n, base["tol"]),
                              0.0, "s", "R", normalize_count=True))
        tw = tc.build_weaver()
        sw = SimulationWeaver()
        sw.add_fragment(OpticalFragment(wl, np.full(n, base["sv"]), 0.0, "s", "R"))
        normed[n] = calculate_merit(sw.backend, tw)
    check("normalize equalizes", abs(normed[200] - normed[10]) < 1e-12,
          f"{normed[200]:.6f} vs {normed[10]:.6f}")
    # tol*sqrt(N) recipe == normalize_count flag.
    wl = np.linspace(400.0, 600.0, 200)
    tc = TargetCollection()
    tc.add(SpectralTarget(wl, np.full(200, base["tv"]),
                          np.full(200, base["tol"] * np.sqrt(200)),
                          0.0, "s", "R"))
    tw = tc.build_weaver()
    sw = SimulationWeaver()
    sw.add_fragment(OpticalFragment(wl, np.full(200, base["sv"]), 0.0, "s", "R"))
    recipe = calculate_merit(sw.backend, tw)
    check("sqrt(N) recipe == flag", abs(recipe - normed[200]) < 1e-9,
          f"{recipe:.6f} vs {normed[200]:.6f}")
    # weight scales.
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.full(3, 0.6), np.full(3, 0.05), 0.0, "s", "R",
                          weight=3.0))
    m = weaver_merit(tc, [((0.0, "s", "R"), np.full(3, 0.61))])
    tc0 = TargetCollection()
    tc0.add(SpectralTarget(WL, np.full(3, 0.6), np.full(3, 0.05), 0.0, "s", "R"))
    m0 = weaver_merit(tc0, [((0.0, "s", "R"), np.full(3, 0.61))])
    check("weight x3", abs(m / m0 - 3.0) < 1e-12, f"{m:.6f} vs {m0:.6f}")


def test_angular_count():
    print("--- angular target-level count ---")
    tc = TargetCollection()
    angs = np.array([0.0, 5.0, 10.0, 15.0])
    tc.add(AngularTarget(500.0, angs, np.full(4, 0.6), np.full(4, 0.05),
                         "s", "R", normalize_count=True))
    tw = tc.build_weaver()
    sw = SimulationWeaver()
    sw.add_fragment(OpticalFragment(WL, np.full(3, 0.61), 0.0, "s", "R"))
    sw.add_fragment(OpticalFragment(WL, np.full(3, 0.61), 5.0, "s", "R"))
    sw.add_fragment(OpticalFragment(WL, np.full(3, 0.61), 10.0, "s", "R"))
    sw.add_fragment(OpticalFragment(WL, np.full(3, 0.61), 15.0, "s", "R"))
    m = calculate_merit(sw.backend, tw)
    # each angle: sim 0.61 vs tgt 0.6, nf = 1/0.6 -> r = 1/3 per point;
    # mean over 4 angles -> (1/3)^2 = 1/9.
    check("angular /4", abs(m - 1.0 / 9.0) < 1e-9, f"got={m:.6f}")


def test_integral_hand():
    print("--- integral hand calcs ---")
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.full(3, 0.5), np.full(3, 0.1), 0.0, "s", "R",
                          kind="e", integral=True))
    m = weaver_merit(tc, [((0.0, "s", "R"), R_VALS)])
    # nf = 1/0.5 = 2; d = (R-0.5)*2 = [0.04, 0.22, 0.38]; mean = 0.64/3;
    # R = dbar/0.1 = 2.1333 -> 4.5511.
    expect = ((0.04 + 0.22 + 0.38) / 3 / 0.1) ** 2
    check("integral mean merit", abs(m - expect) < 1e-9,
          f"got={m:.6f} expect={expect:.6f}")
    # integral-a: mean above -> 0 even though one point dips.
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.full(3, 0.5), np.full(3, 0.1), 0.0, "s", "R",
                          kind="a", integral=True))
    m = weaver_merit(tc, [((0.0, "s", "R"), np.array([0.7, 0.7, 0.4]))])
    check("integral-a mean-above silent", m == 0.0, f"got={m:.6f}")
    # weight x integral.
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.full(3, 0.5), np.full(3, 0.1), 0.0, "s", "R",
                          kind="e", integral=True, weight=2.0))
    m = weaver_merit(tc, [((0.0, "s", "R"), R_VALS)])
    check("weight x integral", abs(m - 2 * expect) < 1e-9, f"got={m:.6f}")


def test_mixed_run():
    print("--- mixed regular + integral + PD in one run ---")
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.array([0.5, 0.6, 0.7]), np.full(3, 0.05),
                          0.0, "s", "R", kind="e", weight=2.0))
    tc.add(SpectralTarget(WL, np.array([0.3, 0.3, 0.3]), np.full(3, 0.05),
                          5.0, "s", "T", kind="a", integral=True))
    tc.add(AngularTarget(500.0, np.array([0.0, 10.0]), np.array([0.5, 0.6]),
                         np.array([0.05, 0.05]), "s", "R", kind="e",
                         normalize_count=True))
    rows = {"Rs": np.array([R_VALS, R_VALS, np.array([0.55, 0.62, 0.70])]),
            "Ts": np.array([np.array([0.28, 0.31, 0.33])] * 3)}
    angs = np.array([0.0, 5.0, 10.0])
    sim = sim_curves_from_arrays(angs, WL, rows)
    m_spec = spec_merit(tc, sim)
    # weaver side needs matching fragments.
    tw = tc.build_weaver()
    sw = SimulationWeaver()
    sw.add_fragment(OpticalFragment(WL, R_VALS, 0.0, "s", "R"))
    sw.add_fragment(OpticalFragment(WL, np.array([0.28, 0.31, 0.33]), 5.0, "s", "T"))
    sw.add_fragment(OpticalFragment(WL, R_VALS, 0.0, "s", "R"))
    sw.add_fragment(OpticalFragment(WL, np.array([0.55, 0.62, 0.70]), 10.0, "s", "R"))
    m_weav = calculate_merit(sw.backend, tw)
    check("mixed weaver == spec", abs(m_weav - m_spec) < 1e-9 * max(1.0, m_weav),
          f"weaver={m_weav:.6f} spec={m_spec:.6f}")
    # residuals length: 3 (R pointwise) + 1 (T integral) + 2 (angular) = 6.
    spec = build_merit_spec(tc)
    res = np.asarray(spec.residuals(sim))
    check("mixed residuals length 6", len(res) == 6, f"len={len(res)}")


def test_fold_mixed():
    print("--- fold mixed regular + integral == merit ---")
    angs = np.array([0.0, 5.0, 10.0])
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.array([0.5, 0.6, 0.7]), np.full(3, 0.05),
                          0.0, "s", "R", kind="e", weight=2.0))
    tc.add(SpectralTarget(WL, np.array([0.5, 0.5, 0.5]), np.full(3, 0.05),
                          5.0, "s", "R", kind="e", integral=True))
    tc.add(SpectralTarget(WL, np.array([0.3, 0.3, 0.3]), np.full(3, 0.05),
                          10.0, "s", "T", kind="a"))
    spec = build_merit_spec(tc)
    r0 = np.array([0.52, 0.61, 0.69])
    r1 = np.array([0.55, 0.60, 0.65])
    rows = {"Rs": np.array([r0, r1, r1]),
            "Ts": np.array([np.array([0.28, 0.29, 0.30])] * 3)}
    sim = sim_curves_from_arrays(angs, WL, rows)
    folded = build_needle_targets(spec, angs, WL, sim)
    w = np.asarray(folded["r"]["weights"]).reshape(3, 3)
    t = np.asarray(folded["r"]["targets"]).reshape(3, 3)
    s1 = r1
    # integral demand (angle 5 -> row 1) folds uniform-gradient pairs:
    # every 2*w*(s-t) must equal g = 2*W*(m-T)/N.
    g = 2 * (4.0 / 0.0025) * (0.6 - 0.5) / 3
    gs = 2 * w[1] * (s1 - t[1])
    check("integral folded gradient uniform",
          bool(np.allclose(gs, g, rtol=1e-9)),
          f"got={np.round(gs, 4).tolist()} expect={g:.4f}")
    # ... and match the uniform-shift finite difference of the merit
    # (only angle-5 rows move, so pointwise parts cancel out of the FD).
    h = 1e-7
    rows_hi = dict(rows, Rs=np.array([r0, r1 + h, r1]))
    rows_lo = dict(rows, Rs=np.array([r0, r1 - h, r1]))
    sim_hi = sim_curves_from_arrays(angs, WL, rows_hi)
    sim_lo = sim_curves_from_arrays(angs, WL, rows_lo)
    fd = (spec.merit(sim_hi, 1e6) - spec.merit(sim_lo, 1e6)) / (2 * h)
    check("integral FD == folded sum", abs(fd - float(np.sum(gs))) < 1e-6,
          f"fd={fd:.4f} folded={float(np.sum(gs)):.4f}")


def test_errors():
    print("--- error paths ---")
    cases = [
        ("py integral+count", lambda: SpectralTarget(
            WL, np.full(3, 0.5), np.full(3, 0.1), 0.0, "s", "R",
            integral=True, normalize_count=True)),
        ("py neg weight", lambda: SpectralTarget(
            WL, np.full(3, 0.5), np.full(3, 0.1), 0.0, "s", "R",
            weight=-1.0)),
        ("py nan weight", lambda: SpectralTarget(
            WL, np.full(3, 0.5), np.full(3, 0.1), 0.0, "s", "R",
            weight=float("nan"))),
    ]
    for name, thunk in cases:
        try:
            thunk()
            check(name + " rejected", False)
        except ValueError:
            check(name + " rejected", True)
    # binding-level combo + bad count.
    from navette._smatrix import MeritSpec
    ms = MeritSpec()
    ki = ms.add_key(0.0, "Rs")
    try:
        ms.add_target(ki, WL.copy(), np.full(3, 0.5), np.full(3, 0.1),
                      "e", "linear", 1.0, integral=True, count_norm=3.0)
        check("binding integral+count rejected", False)
    except ValueError:
        check("binding integral+count rejected", True)
    try:
        ms.add_target(ki, WL.copy(), np.full(3, 0.5), np.full(3, 0.1),
                      "e", "linear", 1.0, weight=-2.0)
        check("binding neg weight rejected", False)
    except ValueError:
        check("binding neg weight rejected", True)


if __name__ == "__main__":
    test_density()
    test_angular_count()
    test_integral_hand()
    test_mixed_run()
    test_fold_mixed()
    test_errors()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
