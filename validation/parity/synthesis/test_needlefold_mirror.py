#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror of the ``build_needle_targets`` fold tests in
``synthesis/needle_pass.rs`` through ``navette._smatrix``.

Translated 1:1 (same demands, same hand-computed expectations):
gain-shift FD, weight/count in the fold, integral uniform-gradient folds,
and all twelve ``targets_builder_*`` fold-shape tests (exact, overlap,
above-masking, range edges, centerband weights, T/A/back/phase buckets,
log rejection).

Gaps (``run_needle_pass`` internals with no Python exposure):
``scan_sites_mirror_python_loop``, ``profile_matches_p_function_oracle``,
``dual_pol_equals_sum_of_branches``, ``best_selects_most_negative_site``,
``interp_clamped_edges``, ``invalid_inputs_rejected`` — see
``rust_mirror_COVERAGE.md``.

Run explicitly:  python validation/parity/synthesis/test_needlefold_mirror.py
"""

import sys

import numpy as np

from navette._smatrix import MeritSpec, SimCurves, build_needle_targets

FAILURES = []
TAU = 2 * np.pi


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


def add(spec, key, wl, targets, tols, kind="e", transform="linear", nf=1.0,
        band=None, phase=False, dp=None, weight=1.0, count=None, integral=False):
    spec.add_target(key, np.asarray(wl, dtype=np.float64),
                    np.asarray(targets, dtype=np.float64),
                    np.asarray(tols, dtype=np.float64), kind, transform, float(nf),
                    band=(np.asarray(band, dtype=np.float64) if band is not None else None),
                    phase=phase, differential_passes=dp, weight=weight,
                    count_norm=count, integral=integral)


def spec_single(angle, curve, wl, norm_targets, tols, nf=1.0, kind="e", phase=False,
                transform="linear"):
    s = MeritSpec()
    k = s.add_key(angle, curve)
    add(s, k, wl, norm_targets, tols, kind=kind, nf=nf, phase=phase, transform=transform)
    return s


def sim_curve(angle, wl, code, val, total_d=0.0):
    wl = np.asarray(wl, dtype=np.float64)
    angs = np.atleast_1d(np.asarray(angle, dtype=np.float64))
    n = angs.size * wl.size
    sim = SimCurves(angs, wl, float(total_d), 1.0, 1.0)
    sim.set_curve(code, np.full(n, float(val)))
    return sim


def test_gain_shift():
    print("--- phi_gain_shift_matches_fd + fold weight/count ---")
    ref = lambda w: TAU * 100.0 / w
    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    add(spec, k, [400., 500.], [0.3 - ref(400.) + 0.01, 0.3 - ref(500.) + 0.01],
        [0.05, 0.05], transform="phase", phase=True, dp=1.0)

    def mk_sim(d):
        sim = SimCurves(np.array([0.0]), np.array([400., 500.]), float(d), 1.0, 1.0)
        sim.set_complex("Ts", np.array([0.7 * np.exp(1j * 0.3)] * 2))
        return sim

    nt = build_needle_targets(spec, np.array([0.0]), np.array([400., 500.]), mk_sim(100.0))
    expect = 8.0 * (TAU / 400.0 + TAU / 500.0)
    check("gain_shift_hand", abs(float(nt["phi2"]["gain_shift"]) - expect) < 1e-9,
          f"{float(nt['phi2']['gain_shift']):.6f} vs {expect:.6f}")
    check("gain_shift_others_zero",
          all(float(nt[f"phi{i}"]["gain_shift"]) == 0.0 for i in (0, 1, 3)))
    h = 1e-3
    fd = (spec.merit(mk_sim(100.0 + h), 1e6) - spec.merit(mk_sim(100.0 - h), 1e6)) / (2 * h)
    check("gain_shift_fd", abs(fd - expect) < 1e-6
          and abs(float(nt["phi2"]["gain_shift"]) - fd) < 1e-6, f"fd={fd:.6f}")
    check("gain_shift_merit", abs(spec.merit(mk_sim(100.0), 1e6) - 0.08) < 1e-12)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400.], [0.5], [0.1], weight=3.0, count=2.0)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.0]))
    check("fold_weight_count",
          abs(float(np.asarray(nt["r"]["weights"])[0]) - 150.0) < 1e-9
          and abs(float(np.asarray(nt["r"]["targets"])[0]) - 0.5) < 1e-14)

    pspec = MeritSpec()
    pk = pspec.add_key(0.0, "Ts")
    add(pspec, pk, [400.], [0.3 - ref(400.) + 0.01], [0.05],
        transform="phase", phase=True, dp=1.0, weight=2.0)
    sim = SimCurves(np.array([0.0]), np.array([400.]), 100.0, 1.0, 1.0)
    sim.set_complex("Ts", np.array([0.7 * np.exp(1j * 0.3)]))
    pnt = build_needle_targets(pspec, np.array([0.0]), np.array([400.]), sim)
    expect = -2.0 * (TAU / 400.0) * 800.0 * (-0.01)
    check("gain_shift_weighted",
          abs(float(pnt["phi2"]["gain_shift"]) - expect) < 1e-9, f"{expect:.6f}")


def test_integral_fold():
    print("--- integral folds ---")
    wl = np.array([400., 500., 600.])
    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, wl, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], integral=True)
    sim = SimCurves(np.array([0.0]), wl, 0.0, 1.0, 1.0)
    sim.set_curve("Rs", np.array([0.7, 0.6, 0.5]))
    nt = build_needle_targets(spec, np.array([0.0]), wl, sim)
    w = np.asarray(nt["r"]["weights"])
    t = np.asarray(nt["r"]["targets"])
    g = 2.0 * 100.0 * 0.1 / 3.0
    ok = bool(np.allclose(w, 100.0 / 9.0, rtol=0, atol=1e-9))
    ok &= bool(np.allclose(t, np.array([0.7, 0.6, 0.5]) - 0.3, rtol=0, atol=1e-12))
    ok &= bool(np.allclose(2 * w * (np.array([0.7, 0.6, 0.5]) - t), g, rtol=0, atol=1e-9))
    h = 1e-6
    sim_hi = SimCurves(np.array([0.0]), wl, 0.0, 1.0, 1.0)
    sim_hi.set_curve("Rs", np.array([0.7 + h, 0.6 + h, 0.5 + h]))
    sim_lo = SimCurves(np.array([0.0]), wl, 0.0, 1.0, 1.0)
    sim_lo.set_curve("Rs", np.array([0.7 - h, 0.6 - h, 0.5 - h]))
    fd = (spec.merit(sim_hi, 1e6) - spec.merit(sim_lo, 1e6)) / (2 * h)
    ok &= abs(fd - 3.0 * g) < 1e-6
    check("integral_uniform_gradient", ok, f"fd={fd:.4f} expect={3.0 * g:.4f}")

    rspec = MeritSpec()
    rk = rspec.add_key(0.0, "Rs")
    add(rspec, rk, wl, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], kind="r", integral=True)
    sim_in = SimCurves(np.array([0.0]), wl, 0.0, 1.0, 1.0)
    sim_in.set_curve("Rs", np.full(3, 0.54))
    nt = build_needle_targets(rspec, np.array([0.0]), wl, sim_in)
    check("integral_range_skip", bool(np.all(np.asarray(nt["r"]["weights"]) == 0.0)))
    sim_out = SimCurves(np.array([0.0]), wl, 0.0, 1.0, 1.0)
    sim_out.set_curve("Rs", np.full(3, 0.7))
    nt2 = build_needle_targets(rspec, np.array([0.0]), wl, sim_out)
    w2 = np.asarray(nt2["r"]["weights"])
    t2 = np.asarray(nt2["r"]["targets"])
    check("integral_range_edge",
          bool(np.allclose(w2, 100.0 / 9.0, rtol=0, atol=1e-9))
          and bool(np.allclose(t2, 0.4, rtol=0, atol=1e-12)), f"{t2}")


def test_builders():
    print("--- targets_builder_* (12) ---")
    nf = 4.0 / 3.0
    spec = spec_single(0.0, "Rs", [400., 500.], [0.5 * nf, 1.0 * nf], [0.01, 0.02], nf=nf)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400., 500., 600.]))
    tgt = np.asarray(nt["r"]["targets"])
    wgt = np.asarray(nt["r"]["weights"])
    check("linear_fold_exact",
          abs(tgt[0] - 0.5) < 1e-14 and abs(tgt[1] - 1.0) < 1e-14
          and tgt[2] == 0.0 and wgt[2] == 0.0
          and abs(wgt[0] - nf * nf / 1e-4) < 1e-9 and abs(wgt[1] - nf * nf / 4e-4) < 1e-6,
          f"{tgt} {wgt}")

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rp")
    add(spec, k, [500.], [0.4 * 2.0], [0.1], nf=2.0)
    add(spec, k, [500.], [0.6 * 3.0], [0.2], nf=3.0)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([500.]))
    wa, wb = 4.0 / 0.01, 9.0 / 0.04
    check("exact_overlap",
          abs(float(np.asarray(nt["r"]["targets"])[0]) - (wa * 0.4 + wb * 0.6) / (wa + wb)) < 1e-12
          and abs(float(np.asarray(nt["r"]["weights"])[0]) - (wa + wb)) < 1e-9)

    spec = spec_single(0.0, "Ru", [400.], [0.5], [0.1], kind="a")
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]),
                              sim_curve(0.0, [400.], "Ru", 0.7))
    check("above_masked", float(np.asarray(nt["r"]["weights"])[0]) == 0.0
          and float(np.asarray(nt["r"]["targets"])[0]) == 0.0)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]),
                              sim_curve(0.0, [400.], "Ru", 0.3))
    check("above_active", float(np.asarray(nt["r"]["weights"])[0]) > 0.0
          and abs(float(np.asarray(nt["r"]["targets"])[0]) - 0.5) < 1e-14)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400.], [0.5], [0.1], band=[0.05], kind="r")
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]),
                              sim_curve(0.0, [400.], "Rs", 0.52))
    check("range_inband", float(np.asarray(nt["r"]["weights"])[0]) == 0.0)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]),
                              sim_curve(0.0, [400.], "Rs", 0.6))
    check("range_upper", abs(float(np.asarray(nt["r"]["targets"])[0]) - 0.55) < 1e-14
          and abs(float(np.asarray(nt["r"]["weights"])[0]) - 100.0) < 1e-9)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]),
                              sim_curve(0.0, [400.], "Rs", 0.3))
    check("range_lower", abs(float(np.asarray(nt["r"]["targets"])[0]) - 0.45) < 1e-14
          and abs(float(np.asarray(nt["r"]["weights"])[0]) - 100.0) < 1e-9)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]))
    check("range_nosim", abs(float(np.asarray(nt["r"]["targets"])[0]) - 0.5) < 1e-14
          and abs(float(np.asarray(nt["r"]["weights"])[0]) - 100.0) < 1e-9)

    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    add(spec, k, [400.], [0.5], [0.1], band=[0.05], kind="c")
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]),
                              sim_curve(0.0, [400.], "Rs", 0.52))
    check("centerband_inside", abs(float(np.asarray(nt["r"]["targets"])[0]) - 0.5) < 1e-14
          and abs(float(np.asarray(nt["r"]["weights"])[0]) - 400.0) < 1e-9)
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]),
                              sim_curve(0.0, [400.], "Rs", 0.6))
    check("centerband_outside", abs(float(np.asarray(nt["r"]["targets"])[0]) - 0.55) < 1e-14
          and abs(float(np.asarray(nt["r"]["weights"])[0]) - 100.0) < 1e-9)

    spec = spec_single(0.0, "Ts", [400.], [0.3], [0.1])
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]))
    check("transmission_bucket", abs(float(np.asarray(nt["t"]["targets"])[0]) - 0.3) < 1e-14
          and abs(float(np.asarray(nt["t"]["weights"])[0]) - 100.0) < 1e-9
          and float(np.asarray(nt["r"]["weights"])[0]) == 0.0
          and float(np.asarray(nt["a"]["weights"])[0]) == 0.0)

    spec = spec_single(0.0, "As", [400.], [0.2], [0.1])
    sim = sim_curve(0.0, [400.], "Rs", 0.6)
    sim.set_curve("Ts", np.array([0.3]))
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]), sim)
    check("absorption_companions", abs(float(np.asarray(nt["a"]["targets"])[0]) - 0.2) < 1e-14
          and abs(float(np.asarray(nt["a"]["weights"])[0]) - 100.0) < 1e-9
          and float(np.asarray(nt["r"]["weights"])[0]) == 0.0)
    spec2 = spec_single(0.0, "As", [400.], [0.05], [0.1], kind="a")
    nt2 = build_needle_targets(spec2, np.array([0.0]), np.array([400.]), sim)
    check("absorption_above_masked", float(np.asarray(nt2["a"]["weights"])[0]) == 0.0)

    spec = spec_single(0.0, "RBs", [400.], [0.4], [0.1])
    nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]))
    check("back_buckets", abs(float(np.asarray(nt["rb"]["targets"])[0]) - 0.4) < 1e-14
          and abs(float(np.asarray(nt["rb"]["weights"])[0]) - 100.0) < 1e-9
          and float(np.asarray(nt["r"]["weights"])[0]) == 0.0)

    ok = True
    for curve, ch in (("Rs", 0), ("Ts", 2)):
        spec = spec_single(0.0, curve, [400.], [1.0], [0.1], phase=True, transform="phase")
        nt = build_needle_targets(spec, np.array([0.0]), np.array([400.]))
        ok &= abs(float(np.asarray(nt[f"phi{ch}"]["targets"])[0]) - 1.0) < 1e-14
        ok &= abs(float(np.asarray(nt[f"phi{ch}"]["weights"])[0]) - 100.0) < 1e-9
        for i in range(4):
            if i != ch:
                ok &= float(np.asarray(nt[f"phi{i}"]["weights"])[0]) == 0.0
    check("phase_channels", ok)
    slog = MeritSpec()
    kl = slog.add_key(0.0, "Rs")
    add(slog, kl, [400.], [1.0], [0.01], transform="log", phase=True)
    try:
        build_needle_targets(slog, np.array([0.0]), np.array([400.]))
        check("phase_log_rejected", False)
    except Exception:
        check("phase_log_rejected", True)
    slog2 = MeritSpec()
    kl2 = slog2.add_key(0.0, "Rs")
    add(slog2, kl2, [400.], [1.0], [0.01], transform="log")
    try:
        build_needle_targets(slog2, np.array([0.0]), np.array([400.]))
        check("log_rejected", False)
    except Exception:
        check("log_rejected", True)


if __name__ == "__main__":
    test_gain_shift()
    test_integral_fold()
    test_builders()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
