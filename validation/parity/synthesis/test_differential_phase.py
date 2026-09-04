# -*- coding: utf-8 -*-
"""Parity + validation for differential-phase targets (PDts / PDtp).

Dphi(λ) = arg(t(λ)) − passes·2π·n_inc·D·cosθ/λ — the coating-induced transmitted
phase with the equivalent incidence-medium layer subtracted.

Strategy (each check prints OK/FAIL, non-zero exit on any FAIL):

1. ``hand`` ......... single-point Dphi vs hand arithmetic (s + p, oblique).
2. ``oracle`` ....... native differential demand == absolute-phase demand on
                      numpy-rotated rows (``apply_reference_rotation``), for
                      kinds e/a/b/r/c × tolerances/bands. 1e-12.
3. ``numpy-2x2`` .... independent characteristic-matrix oracle (s AND p,
                      oblique 10°): solver-free Dphi merit == hand merit.
4. ``dM/dD`` ........ FD slope of merit over total_d == analytic
                      Σ−2·kz·Δ/tol² == fold ``phi_gain_shift``.
5. ``fold`` ......... phi2/phi3 buckets (Ts→ch2, Tp→ch3... see note), folded
                      == merit for non-overlapping layouts; gain_shift exact.
6. ``zero-D`` ....... total_d = 0 → differential ≡ absolute bit-for-bit.
7. ``errors`` ....... PD+phase=False rejected; PD pol mismatch rejected;
                      differential-without-phase (binding) rejected;
                      bad passes rejected; unknown label still rejected.
8. ``fold-equiv`` ... fold(native differential) == fold(absolute on rotated).

NOTE on channels: Ts→2, Tp→2 (front T element, s/p share the channel; the
engine separates polarizations by branch, the fold by channel).
"""

import sys

import numpy as np

from navette.spectralweave.target import (
    AngularTarget,
    SpectralTarget,
    TargetCollection,
)
from navette.synthesis import (
    apply_reference_rotation,
    build_merit_spec,
    sim_curves_from_arrays,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# numpy 2x2 characteristic-matrix oracle (s and p, arbitrary angle)
# ---------------------------------------------------------------------------
def oracle_tf(n0, n1, ns, d, lam, theta0_deg, pol):
    """Complex forward-t for one film. Macleod convention (conjugate of the
    crate solver — callers compare phases up to that documented flip, or
    feed magnitudes / solver-space values)."""
    th0 = np.radians(theta0_deg)
    s0 = n0 * np.sin(th0)
    c0 = np.cos(th0)
    # Snell through the film (real indices here).
    sin1 = s0 / n1
    cos1 = np.sqrt(1.0 - sin1 ** 2)
    sins = s0 / ns
    coss = np.sqrt(1.0 - sins ** 2)
    if pol == "s":
        e0, e1, es = n0 * c0, n1 * cos1, ns * coss
    else:
        e0, e1, es = n0 / c0, n1 / cos1, ns / coss
    delta = 2 * np.pi * n1 * d * cos1 / lam
    s, c = np.sin(delta), np.cos(delta)
    m11, m12 = c, 1j * s / e1
    m21, m22 = 1j * s * e1, c
    return 2 * e0 / (e0 * (m11 + m12 * es) + (m21 + m22 * es))


WL = np.array([400.0, 500.0, 600.0])
N_L, N_S, D_NM, TH_DEG = np.sqrt(1.52), 1.52, 200.0, 10.0


def solver_space_tfs(pol):
    """Oracle amplitudes in CRATE convention (conjugated Macleod)."""
    return np.array([[oracle_tf(1.0, N_L, N_S, D_NM, lam, TH_DEG, pol)
                      .conjugate() for lam in WL]])


def ref_array(n_inc=1.0, d=D_NM, theta=TH_DEG, passes=1.0):
    return (passes * 2 * np.pi * n_inc * d * np.cos(np.radians(theta)) / WL)


def test_hand():
    print("--- hand calc (oblique, s + p) ---")
    for pol, curve in [("s", "Ts"), ("p", "Tp")]:
        t = solver_space_tfs(pol)[0]
        expect = np.angle(t) - ref_array()
        tc = TargetCollection()
        tc.add(SpectralTarget(WL, expect, np.full(3, 0.05), TH_DEG, pol,
                              "PDt" + pol, kind="e", phase=True))
        spec = build_merit_spec(tc)
        sim = sim_curves_from_arrays(
            np.array([TH_DEG]), WL, {},
            {"Ts": solver_space_tfs("s"), "Tp": solver_space_tfs("p")},
            total_d=D_NM, n_front=1.0)
        m = spec.merit(sim, 1e6)
        check(f"PDt{pol} exact-target merit ~ 0", m < 1e-24, f"merit={m:.3g}")


def pd_collection(kind="e", band=None, tol=0.05, offset=0.0):
    tc = TargetCollection()
    t = solver_space_tfs("s")[0]
    vals = np.angle(t) - ref_array() + offset
    tc.add(SpectralTarget(WL, vals, np.full(3, tol), TH_DEG, "s",
                          "PDts", kind=kind, band=band, phase=True))
    return tc


def test_oracle_kinds():
    print("--- oracle: native differential == absolute on rotated rows ---")
    rows = solver_space_tfs("s")
    cases = [
        ("e", None, 0.05, 0.01),
        ("a", None, 0.05, -0.02),   # sim below target → active
        ("b", None, 0.05, 0.02),    # sim above target → active
        ("r", 0.005, 0.05, 0.02),   # violated
        ("r", 0.05, 0.05, 0.005),   # in-band → 0
        ("c", 0.05, 0.05, 0.01),    # inside
        ("c", 0.005, 0.05, 0.03),   # outside (+1 level)
    ]
    for kind, band, tol, offset in cases:
        tc = pd_collection(kind, band, tol, offset)
        spec = build_merit_spec(tc)
        sim = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                     {"Ts": rows}, total_d=D_NM, n_front=1.0)
        m_nat = spec.merit(sim, 1e6)
        # Absolute-phase twin on rotated rows, same targets/tols.
        rot = apply_reference_rotation(rows, WL, TH_DEG, 1.0, D_NM, 1.0)
        tc2 = TargetCollection()
        t = np.angle(rows[0]) - ref_array() + offset
        tc2.add(SpectralTarget(WL, t, np.full(3, tol), TH_DEG, "s",
                               "T", kind=kind, band=band, phase=True))
        spec2 = build_merit_spec(tc2)
        sim2 = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                      {"Ts": rot})
        m_abs = spec2.merit(sim2, 1e6)
        check(f"kind={kind} band={band} off={offset}",
              abs(m_nat - m_abs) < 1e-12 * max(1.0, m_abs),
              f"nat={m_nat:.6f} abs={m_abs:.6f}")


def test_numpy_2x2():
    print("--- numpy 2x2 oracle (solver-free Dphi merit) ---")
    for pol in ["s", "p"]:
        t = solver_space_tfs(pol)[0]
        delta = np.angle(t) - ref_array()
        tol = 0.05
        off = 0.02
        expect = float(np.sum(((delta - (delta + off)) / tol) ** 2))
        tc = TargetCollection()
        tc.add(SpectralTarget(WL, delta + off, np.full(3, tol), TH_DEG, pol,
                              "PDt" + pol, kind="e", phase=True))
        spec = build_merit_spec(tc)
        sim = sim_curves_from_arrays(
            np.array([TH_DEG]), WL, {},
            {"Ts": solver_space_tfs("s"), "Tp": solver_space_tfs("p")},
            total_d=D_NM, n_front=1.0)
        m = spec.merit(sim, 1e6)
        check(f"PDt{pol} hand merit", abs(m - expect) < 1e-12,
              f"got={m:.6f} expect={expect:.6f}")


def test_dmdD():
    print("--- dM/dD: FD == analytic == fold gain_shift ---")
    rows = solver_space_tfs("s")
    off, tol = 0.01, 0.05
    tc = pd_collection("e", None, tol, off)
    spec = build_merit_spec(tc)
    base = dict(rows={"Ts": rows})
    sim = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                 base["rows"], total_d=D_NM, n_front=1.0)
    h = 1e-3
    sim_hi = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                    base["rows"], total_d=D_NM + h, n_front=1.0)
    sim_lo = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                    base["rows"], total_d=D_NM - h, n_front=1.0)
    fd = (spec.merit(sim_hi, 1e6) - spec.merit(sim_lo, 1e6)) / (2 * h)
    kz = 2 * np.pi * 1.0 * np.cos(np.radians(TH_DEG)) / WL
    delta = np.angle(rows[0]) - ref_array()
    r = (delta - (delta + off)) / tol
    analytic = float(np.sum(2 * r * (-kz / tol)))
    from navette.synthesis import build_needle_targets
    folded = build_needle_targets(spec, np.array([TH_DEG]), WL, sim)
    gs = float(folded["phi2"]["gain_shift"])
    check("FD == analytic", abs(fd - analytic) < 1e-6 * max(1.0, abs(analytic)),
          f"fd={fd:.6f} analytic={analytic:.6f}")
    check("fold gain_shift == FD", abs(gs - fd) < 1e-6 * max(1.0, abs(fd)),
          f"gain_shift={gs:.6f}")
    check("other channels zero",
          all(float(folded[f"phi{i}"]["gain_shift"]) == 0.0 for i in [0, 1, 3]))


def test_fold_values():
    print("--- fold values (non-overlapping) == merit ---")
    rows = solver_space_tfs("s")
    rowp = solver_space_tfs("p")
    off, tol = 0.01, 0.05
    tc = TargetCollection()
    angs = np.array([TH_DEG, TH_DEG + 5.0])
    for i, (pol, rr) in enumerate([("s", rows), ("p", rowp)]):
        delta = np.angle(rr[0]) - ref_array(theta=angs[i])
        tc.add(SpectralTarget(WL, delta + off, np.full(3, tol), angs[i], pol,
                              "PDt" + pol, kind="e", phase=True))
    spec = build_merit_spec(tc)
    sim = sim_curves_from_arrays(angs, WL, {},
                                 {"Ts": np.vstack([rows[0], rows[0]]),
                                  "Tp": np.vstack([rowp[0], rowp[0]])},
                                 total_d=D_NM, n_front=1.0)
    from navette.synthesis import build_needle_targets
    folded = build_needle_targets(spec, angs, WL, sim)
    m = spec.merit(sim, 1e6)
    # folded merit from (targets, weights) vs Dphi sims.
    # channel 2 shared by s/p: evaluate per-angle rows explicitly
    w2 = np.asarray(folded["phi2"]["weights"]).reshape(2, 3)
    t2 = np.asarray(folded["phi2"]["targets"]).reshape(2, 3)
    s_s = np.angle(rows[0]) - ref_array(theta=angs[0])
    s_p = np.angle(rowp[0]) - ref_array(theta=angs[1])
    fsum = (float(np.sum(w2[0] * (t2[0] - s_s) ** 2))
            + float(np.sum(w2[1] * (t2[1] - s_p) ** 2)))
    check("folded == merit", abs(fsum - m) < 1e-9 * max(1.0, m),
          f"folded={fsum:.6f} merit={m:.6f}")


def test_zero_d():
    print("--- zero-D: differential == absolute ---")
    rows = solver_space_tfs("s")
    # targets near arg() itself (no reference embedded): at D = 0 the
    # reference vanishes, so both paths must read 3 * (0.01/0.05)^2 = 0.12.
    t = np.angle(rows[0]) + 0.01
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, t, np.full(3, 0.05), TH_DEG, "s",
                          "PDts", kind="e", phase=True))
    spec = build_merit_spec(tc)
    sim_d = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                   {"Ts": rows}, total_d=0.0, n_front=1.0)
    m_d = spec.merit(sim_d, 1e6)
    tc2 = TargetCollection()
    tc2.add(SpectralTarget(WL, t, np.full(3, 0.05), TH_DEG, "s",
                           "T", kind="e", phase=True))
    spec2 = build_merit_spec(tc2)
    sim_a = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                   {"Ts": rows})
    m_a = spec2.merit(sim_a, 1e6)
    check("zero-D match", abs(m_d - m_a) < 1e-12 and abs(m_d - 0.12) < 1e-12,
          f"d={m_d:.6f} a={m_a:.6f}")


def test_errors():
    print("--- error paths ---")
    # PD without phase=True
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.zeros(3), np.full(3, 0.05), TH_DEG, "s",
                          "PDts", kind="e", phase=False))
    try:
        build_merit_spec(tc)
        check("PD + phase=False rejected", False)
    except ValueError:
        check("PD + phase=False rejected", True)
    # PD pol mismatch
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.zeros(3), np.full(3, 0.05), TH_DEG, "p",
                          "PDts", kind="e", phase=True))
    try:
        build_merit_spec(tc)
        check("PDts + pol=p rejected", False)
    except ValueError:
        check("PDts + pol=p rejected", True)
    # binding-level: differential without phase
    from navette._smatrix import MeritSpec
    ms = MeritSpec()
    ki = ms.add_key(0.0, "Ts")
    try:
        ms.add_target(ki, WL.copy(), np.zeros(3), np.full(3, 0.05),
                      "e", "phase", 1.0, phase=False, differential_passes=1.0)
        check("binding differential-without-phase rejected", False)
    except ValueError:
        check("binding differential-without-phase rejected", True)
    try:
        ms.add_target(ki, WL.copy(), np.zeros(3), np.full(3, 0.05),
                      "e", "phase", 1.0, phase=True, differential_passes=-1.0)
        check("binding negative passes rejected", False)
    except ValueError:
        check("binding negative passes rejected", True)
    # unknown label still rejected
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.zeros(3), np.full(3, 0.05), TH_DEG, "s",
                          "Q", kind="e"))
    try:
        build_merit_spec(tc)
        check("unknown label rejected", False)
    except ValueError:
        check("unknown label rejected", True)


def test_fold_equiv():
    print("--- fold-equivalence: differential vs absolute-on-rotated ---")
    from navette.synthesis import build_needle_targets
    rows = solver_space_tfs("s")
    tc = pd_collection("e", None, 0.05, 0.01)
    spec = build_merit_spec(tc)
    sim = sim_curves_from_arrays(np.array([TH_DEG]), WL, {},
                                 {"Ts": rows}, total_d=D_NM, n_front=1.0)
    f1 = build_needle_targets(spec, np.array([TH_DEG]), WL, sim)
    rot = apply_reference_rotation(rows, WL, TH_DEG, 1.0, D_NM, 1.0)
    tc2 = TargetCollection()
    t = np.angle(rows[0]) - ref_array() + 0.01
    tc2.add(SpectralTarget(WL, t, np.full(3, 0.05), TH_DEG, "s",
                           "T", kind="e", phase=True))
    spec2 = build_merit_spec(tc2)
    sim2 = sim_curves_from_arrays(np.array([TH_DEG]), WL, {}, {"Ts": rot})
    f2 = build_needle_targets(spec2, np.array([TH_DEG]), WL, sim2)
    same_t = np.allclose(f1["phi2"]["targets"], f2["phi2"]["targets"], atol=0)
    same_w = np.allclose(f1["phi2"]["weights"], f2["phi2"]["weights"], atol=0)
    check("fold targets identical", bool(same_t))
    check("fold weights identical", bool(same_w))
    check("rotated fold gain_shift == 0",
          float(f2["phi2"]["gain_shift"]) == 0.0)


def test_angular_pd():
    print("--- angular PDts demand ---")
    rows = solver_space_tfs("s")
    tc = TargetCollection()
    ang = np.array([TH_DEG, TH_DEG + 5.0])
    ref0 = 1.0 * 2 * np.pi * 1.0 * D_NM * np.cos(np.radians(ang[0])) / 500.0
    delta = np.angle(rows[0])[1] - ref0  # Dphi at (500 nm, TH_DEG)
    tc.add(AngularTarget(500.0, ang, np.full(2, delta + 0.01),
                         np.full(2, 0.05), "s", "PDts", kind="e", phase=True))
    spec = build_merit_spec(tc)
    sim = sim_curves_from_arrays(ang, WL, {},
                                 {"Ts": np.vstack([rows[0], rows[0]])},
                                 total_d=D_NM, n_front=1.0)
    m = spec.merit(sim, 1e6)
    # hand: point0 exact (Dphi(500) at TH_DEG), point1 residual from angle shift
    ref1 = 1.0 * 2 * np.pi * 1.0 * D_NM * np.cos(np.radians(ang[1])) / 500.0
    d0 = (delta - (delta + 0.01)) / 0.05
    # sim row1 mirrors row0 (same complex row): Dphi differs only via reference
    s1 = np.angle(rows[0])[1] - ref1
    d1 = (s1 - (delta + 0.01)) / 0.05
    expect = d0 ** 2 + d1 ** 2
    check("angular PD merit", abs(m - expect) < 1e-9,
          f"got={m:.6f} expect={expect:.6f}")


if __name__ == "__main__":
    test_hand()
    test_oracle_kinds()
    test_numpy_2x2()
    test_dmdD()
    test_fold_values()
    test_zero_d()
    test_errors()
    test_fold_equiv()
    test_angular_pd()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
