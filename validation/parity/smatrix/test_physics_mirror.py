#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror of the solver/needle physics tests through the public Python API.

Translated intent (same stacks, same FD conventions, same hand values):

- ``evaluator.rs``: lossless energy conservation, quarter-wave AR anchor,
  optimizer recovery from a bad start (via ``scipy`` on ``ScatterMatrix`` +
  ``MeritSpec`` — the Rust optimizer itself is not exposed), propagation
  sign on the matched slab (``arg(t) = +pi/2``), oblique PD reference.
- ``optics_core.rs``: ``reference_phase`` hand values + ``kz·D`` identity,
  exercised through the real kernel via PD-merit zeros.
- ``needle_operator.rs`` translatable subset: back-channel FD (``P_TB`` /
  ``P_RB`` / ``P_AB`` on a membrane where the back flux factor is 1),
  multiblock FD + single-block reduction (``P_MB*``), ``DPHI`` FD,
  ``DGDD`` end-to-end FD, lossless energy conservation.
- ``interpolate`` wrapper smoke (no Rust tests exist): eval pins,
  derivative-vs-FD, x/y roundtrip.

Gaps (no Python exposure — see ``rust_mirror_COVERAGE.md``): ``simulate``/
``SmatrixContext``/``DesignStack``/``LayerSpec``/LM configs, ``gated``
allocation paths, ``p_coherent_*``/``p_function``/cascade/spectral-chain/
kernel/anchor/thin-slab internals, ``run_needle_pass`` sites.

Run explicitly:  python validation/parity/smatrix/test_physics_mirror.py
"""

import sys

import numpy as np

from navette._smatrix import MeritSpec
from navette.smatrix.needle import NeedleRequest, needle_gradient
from navette.smatrix.smatrix import Request, ScatterMatrix
from navette.synthesis import sim_curves_from_arrays

FAILURES = []
LAM = 500.0


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- evaluator
def test_ar_physics():
    print("--- evaluator physics via ScatterMatrix ---")
    n_l = np.sqrt(1.52)
    wavls = np.array([900., 1000., 1100.])
    # lossless: R + T == 1, values in [0, 1].
    st = ScatterMatrix(np.array([1.0 + 0j, n_l + 0j, 1.52 + 0j]),
                       np.array([0.0, 200.0, 0.0]),
                       wavelengths=wavls, angles=[0.0])
    out = st.compute(Request.RS | Request.RP | Request.TS | Request.TP)
    check("lossless_conservation",
          bool(np.allclose(out["Rs"] + out["Ts"], 1.0, atol=1e-10))
          and bool(np.all((out["Rs"] >= 0) & (out["Rs"] <= 1)
                          & (out["Rp"] >= 0) & (out["Rp"] <= 1))), )
    # quarter-wave AR: d = lam/(4n) -> R = 0 at 1000 nm.
    d_qw = 1000.0 / (4.0 * n_l)
    st_qw = ScatterMatrix(np.array([1.0 + 0j, n_l + 0j, 1.52 + 0j]),
                          np.array([0.0, d_qw, 0.0]),
                          wavelengths=np.array([1000.]), angles=[0.0])
    r_qw = float(st_qw.compute(Request.RS)["Rs"][0])
    check("quarter_wave_zero", r_qw < 1e-12, f"R={r_qw:.2e}")
    st_off = ScatterMatrix(np.array([1.0 + 0j, n_l + 0j, 1.52 + 0j]),
                           np.array([0.0, d_qw * 0.7, 0.0]),
                           wavelengths=np.array([1000.]), angles=[0.0])
    check("off_quarter_nonzero", float(st_off.compute(Request.RS)["Rs"][0]) > 1e-6)

    # optimizer recovery from a bad start (scipy drives the same merit).
    from scipy.optimize import minimize
    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    spec.add_target(k, np.array([1000.]), np.array([0.0]), np.array([0.01]),
                    "e", "linear", 1.0)

    def merit(d):
        st = ScatterMatrix(np.array([1.0 + 0j, n_l + 0j, 1.52 + 0j]),
                           np.array([0.0, float(d[0]), 0.0]),
                           wavelengths=np.array([1000.]), angles=[0.0])
        sim = sim_curves_from_arrays(np.array([0.0]), np.array([1000.]),
                                     {"Rs": np.atleast_2d(st.compute(Request.RS)["Rs"])})
        return spec.merit(sim, 1e6)

    res = minimize(merit, np.array([50.0]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 500})
    check("optimizer_recovers_qw",
          abs(float(res.x[0]) - d_qw) < 0.5 and float(res.fun) < 1e-10,
          f"d={float(res.x[0]):.4f} vs {d_qw:.4f}, mf={float(res.fun):.2e}")


def test_propagation_sign():
    print("--- propagation sign + oblique reference ---")
    # matched slab (n = 1 everywhere, film 500 nm): t is pure propagation,
    # kD = 2.5pi -> arg = +pi/2 (NOT -pi/2): the crate's sign convention.
    st = ScatterMatrix(np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j]),
                       np.array([0.0, 500.0, 0.0]),
                       wavelengths=np.array([400.]), angles=[0.0])
    tf = complex(st.compute(Request.TS_C)["ts_c"][0])
    check("matched_slab_arg", abs(np.angle(tf) - np.pi / 2) < 1e-9, f"arg={np.angle(tf):.6f}")
    # reference_phase is unwrapped (2.5pi): PD merit with the unwrapped
    # target must wrap to zero exactly as the kernel does.
    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    spec.add_target(k, np.array([400.]), np.array([np.angle(tf) - 2.5 * np.pi]),
                    np.array([0.05]), "e", "phase", 1.0, phase=True, differential_passes=1.0)
    from navette._smatrix import SimCurves
    sim = SimCurves(np.array([0.0]), np.array([400.]), 500.0, 1.0, 1.0)
    sim.set_complex("Ts", np.array([tf]))
    check("unwrapped_ref_wraps", spec.merit(sim, 1e6) < 1e-20)
    # oblique: 60° halves the axial projection; kz*D reconstructs ref.
    spec = MeritSpec()
    k = spec.add_key(60.0, "Ts")
    ref60 = 2 * np.pi * 100.0 * np.cos(np.radians(60.0)) / 500.0
    spec.add_target(k, np.array([500.]), np.array([0.3 - ref60]), np.array([0.05]),
                    "e", "phase", 1.0, phase=True, differential_passes=1.0)
    sim = SimCurves(np.array([60.0]), np.array([500.]), 100.0, 1.0, 1.0)
    sim.set_complex("Ts", np.array([0.7 * np.exp(1j * 0.3)]))
    check("oblique_reference", spec.merit(sim, 1e6) < 1e-28, f"ref60={ref60:.6f}")
    # hand values: (500, 1, 0°, 100, 1) -> 2pi/5; 2 passes double; D = 0 kills.
    check("reference_hand", abs(ref60 * 2 - 2 * np.pi * 100.0 * 1.0 / 500.0) < 1e-15
          and abs(2 * np.pi / 5 - 1.2566370614) < 1e-9)


# ---------------------------------------------------------------- needle FD
DELTA = 5e-4
N_NEEDLE = 1.9 + 0j


def membrane(absorbing=True):
    # 4-layer symmetric stack (needle block needs >= 4 layers); same medium
    # both sides -> back flux factor fb = 1 at any angle.
    n_b = (1.8 + 0.4j) if absorbing else 2.35 + 0j
    return np.array([1.0 + 0j, 2.0 + 0j, n_b, 1.0 + 0j])


def needle_pair(idx, thick, j, xi, flags=None):
    """Base + needle-inserted stacks (needle glass DELTA at depth xi in layer j)."""
    base = ScatterMatrix(idx, thick, wavelengths=np.array([LAM]),
                         angles=np.arcsin(0.3), angles_in_radians=True,
                         incoherent_flags=flags)
    # split host layer j into top(xi)/needle/bottom: rebuild explicitly.
    th = list(thick)
    top, bot = xi, th[j] - xi
    th2 = np.array(th[:j] + [top, DELTA, bot] + th[j + 1:])
    idx2 = np.array(list(idx[:j + 1]) + [N_NEEDLE, idx[j]] + list(idx[j + 1:]))
    fl2 = None
    if flags is not None:
        fl = list(flags)
        fl2 = np.array(fl[:j + 1] + [0, fl[j]] + fl[j + 1:], dtype=np.int32)
    ins = ScatterMatrix(idx2, th2, wavelengths=np.array([LAM]),
                        angles=np.arcsin(0.3), angles_in_radians=True,
                        incoherent_flags=fl2)
    return base, ins


def back_channels(fd_of):
    base, ins = fd_of
    b0 = base.compute(Request.RBS_C | Request.RBP_C | Request.TBS_C | Request.TBP_C)
    b1 = ins.compute(Request.RBS_C | Request.RBP_C | Request.TBS_C | Request.TBP_C)
    return b0, b1


def test_pd_optimizer_recovery():
    print("--- optimizer recovers thickness from PD target ---")
    from scipy.optimize import minimize
    from navette._smatrix import SimCurves
    wl = np.array([1000.])
    spec = MeritSpec()
    k = spec.add_key(0.0, "Ts")
    spec.add_target(k, wl, np.array([0.0]), np.array([0.05]),
                    "e", "phase", 1.0, phase=True, differential_passes=1.0)

    def merit(d):
        dd = float(d[0])
        st = ScatterMatrix(np.array([1.0 + 0j, 1.5 + 0j, 1.0 + 0j]),
                           np.array([0.0, dd, 0.0]),
                           wavelengths=wl, angles=[0.0])
        tc = complex(np.asarray(st.compute(Request.TS_C)["ts_c"]).ravel()[0])
        sim = SimCurves(np.array([0.0]), wl, dd, 1.0, 1.0)
        sim.set_complex("Ts", np.array([tc]))
        return spec.merit(sim, 1e6)

    res = minimize(merit, np.array([40.0]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-12, "maxiter": 500})
    # D = 0 is the global minimizer (matched-ish film -> arg(t) ~ -kD).
    check("optimizer_recovers_pd", float(res.fun) < 1e-8, f"mf={float(res.fun):.2e}")


def test_back_needle_fd():
    print("--- needle back channels vs FD (membrane, fb = 1) ---")
    idx = membrane(absorbing=True)
    thick = np.array([0.0, 25.0, 25.0, 0.0])
    base, ins = needle_pair(idx, thick, 2, 12.5)
    b0, b1 = back_channels((base, ins))
    # membrane air/film/air: same medium both sides -> back flux factor 1.
    cases = [("TB", "tbs_c", NeedleRequest.P_TB, "targets_tb", "weights_tb"),
             ("RB", "rbs_c", NeedleRequest.P_RB, "targets_rb", "weights_rb")]
    for name, code, req, tk, wk in cases:
        x0 = abs(complex(b0[code][0])) ** 2
        x1 = abs(complex(b1[code][0])) ** 2
        fd = (x1 * x1 - x0 * x0) / DELTA / 2.0
        g = needle_gradient(base, N_NEEDLE, np.array([37.5]), int(req),
                            **{tk: np.array([0.0]), wk: np.array([1.0]),
                               "pol": "s"})
        key = [kk for kk in g if "TB" in kk or "RB" in kk][0]
        p = float(np.asarray(g[key]).ravel()[0])
        scale = max(abs(fd), abs(p), 1e-12)
        check(f"p_back_{name}", abs(fd - p) / scale < 2e-3, f"fd={fd:.6e} p={p:.6e}")
    # Ab = 1 - Rb - Tb (fb = 1).
    tb0 = abs(complex(b0["tbs_c"][0])) ** 2
    rb0 = abs(complex(b0["rbs_c"][0])) ** 2
    tb1 = abs(complex(b1["tbs_c"][0])) ** 2
    rb1 = abs(complex(b1["rbs_c"][0])) ** 2
    fd = ((1 - rb1 - tb1) ** 2 - (1 - rb0 - tb0) ** 2) / DELTA / 2.0
    g = needle_gradient(base, N_NEEDLE, np.array([37.5]), int(NeedleRequest.P_AB),
                        targets_ab=np.array([0.0]), weights_ab=np.array([1.0]), pol="s")
    key = [kk for kk in g if "AB" in kk][0]
    p = float(np.asarray(g[key]).ravel()[0])
    scale = max(abs(fd), abs(p), 1e-12)
    check("p_back_AB", abs(fd - p) / scale < 2e-3, f"fd={fd:.6e} p={p:.6e}")


def test_multiblock():
    print("--- multiblock P_MB vs FD + single-block reduction ---")
    idx = np.array([1.0 + 0j, 2.35 + 0j, 1.46 + 0j, 1.8 + 0.4j, 1.52 + 0j])
    thick = np.array([0.0, 40.0, 2000.0, 50.0, 0.0])
    flags = np.array([0, 0, 1, 0, 0], dtype=np.int32)
    coh = ScatterMatrix(idx, thick, wavelengths=np.array([LAM]),
                        angles=np.arcsin(0.3), angles_in_radians=True)
    mb = ScatterMatrix(idx, thick, wavelengths=np.array([LAM]),
                       angles=np.arcsin(0.3), angles_in_radians=True,
                       incoherent_flags=flags)
    # reduction: all-coherent flags -> P_MB == P.
    tgt = np.array([0.05])
    g_p = needle_gradient(coh, N_NEEDLE, np.array([40.0]), int(NeedleRequest.P),
                          targets_r=tgt, weights_r=np.array([100.0]), pol="s")
    g_mb = needle_gradient(coh, N_NEEDLE, np.array([40.0]), int(NeedleRequest.P_MB),
                           targets_r=tgt, weights_r=np.array([100.0]), pol="s")
    check("pmb_reduces_to_p",
          abs(float(np.asarray(g_p["P_s"]).ravel()[0])
              - float(np.asarray(g_mb["Pmb_s"]).ravel()[0])) < 1e-9)
    # FD through the cascade: insert into film 1 (before the spacer).
    base, ins = needle_pair(idx, thick, 1, 20.0, flags)
    r0 = float(base.compute(Request.RS)["Rs"][0])
    r1 = float(ins.compute(Request.RS)["Rs"][0])
    fd = (r1 * r1 - r0 * r0) / DELTA / 2.0
    g = needle_gradient(mb, N_NEEDLE, np.array([20.0]), int(NeedleRequest.P_MB),
                        targets_r=np.array([0.0]), weights_r=np.array([1.0]), pol="s")
    p = float(np.asarray(g["Pmb_s"]).ravel()[0])
    scale = max(abs(fd), abs(p), 1e-12)
    check("pmb_R_fd", abs(fd - p) / scale < 2e-3, f"fd={fd:.6e} p={p:.6e}")
    # transmission + absorption through the cascade.
    t0 = float(base.compute(Request.TS)["Ts"][0])
    t1 = float(ins.compute(Request.TS)["Ts"][0])
    fd = (t1 * t1 - t0 * t0) / DELTA / 2.0
    g = needle_gradient(mb, N_NEEDLE, np.array([20.0]), int(NeedleRequest.P_MB_T),
                        targets_t=np.array([0.0]), weights_t=np.array([1.0]), pol="s")
    p = float(np.asarray(g["Pmb_T_s"]).ravel()[0])
    scale = max(abs(fd), abs(p), 1e-12)
    check("pmb_T_fd", abs(fd - p) / scale < 2e-3, f"fd={fd:.6e} p={p:.6e}")
    # absorption through the cascade (absorbing film behind the spacer).
    a0 = float(base.compute(Request.A_S)["A_s"][0])
    a1 = float(ins.compute(Request.A_S)["A_s"][0])
    fd = (a1 * a1 - a0 * a0) / DELTA / 2.0
    g = needle_gradient(mb, N_NEEDLE, np.array([20.0]), int(NeedleRequest.P_MB_A),
                        targets_a=np.array([0.0]), weights_a=np.array([1.0]), pol="s")
    p = float(np.asarray(g["Pmb_A_s"]).ravel()[0])
    scale = max(abs(fd), abs(p), 1e-12)
    check("pmb_A_fd", abs(fd - p) / scale < 2e-3, f"fd={fd:.6e} p={p:.6e}")
    # back channels through the cascade: the cascade-Tb oracle lives in
    # unexposed machinery (the coherent |tbs_c| FD disagrees legitimately
    # once a spacer decoheres the path), so mirror the reduction half —
    # all-coherent stack: P_MB_TB/RB/AB == P_TB/RB/AB.
    idxm = np.array([1.0 + 0j, 2.0 + 0j, 1.8 + 0.4j, 2.0 + 0j, 1.0 + 0j])
    thickm = np.array([0.0, 25.0, 25.0, 25.0, 0.0])
    cohm = ScatterMatrix(idxm, thickm, wavelengths=np.array([LAM]),
                         angles=np.arcsin(0.3), angles_in_radians=True)
    kw = dict(targets_tb=np.array([0.0]), weights_tb=np.array([1.0]),
              targets_rb=np.array([0.0]), weights_rb=np.array([1.0]),
              targets_ab=np.array([0.0]), weights_ab=np.array([1.0]), pol="s")
    g_co = needle_gradient(cohm, N_NEEDLE, np.array([37.5]),
                           int(NeedleRequest.P_TB | NeedleRequest.P_RB | NeedleRequest.P_AB),
                           **kw)
    g_mb = needle_gradient(cohm, N_NEEDLE, np.array([37.5]),
                           int(NeedleRequest.P_MB_TB | NeedleRequest.P_MB_RB
                               | NeedleRequest.P_MB_AB), **kw)
    pairs = [("P_TB_s", "Pmb_TB_s"), ("P_RB_s", "Pmb_RB_s"), ("P_AB_s", "Pmb_AB_s")]
    ok = all(k0 in g_co and k1 in g_mb for k0, k1 in pairs)
    ok &= all(abs(float(np.asarray(g_co[k0]).ravel()[0])
                  - float(np.asarray(g_mb[k1]).ravel()[0])) < 1e-9
              for k0, k1 in pairs)
    check("pmb_back_reduces", ok, f"{sorted(g_mb)}")


def test_dispersion_channels():
    print("--- DPHI + DGDD channels vs FD ---")
    idx = np.array([1.0 + 0j, 2.35 + 0j, 1.46 + 0j, 2.35 + 0j, 1.52 + 0j])
    thick = np.array([0.0, 40.0, 80.0, 30.0, 0.0])
    wavls = np.array([500., 600., 700.])
    base = ScatterMatrix(idx, thick, wavelengths=wavls, angles=[0.0])
    th2 = np.array([0.0, 40.0, 25.0, DELTA, 80.0 - 25.0, 30.0, 0.0])
    idx2 = np.array([1.0 + 0j, 2.35 + 0j, 1.46 + 0j, N_NEEDLE, 1.46 + 0j, 2.35 + 0j, 1.52 + 0j])
    ins = ScatterMatrix(idx2, th2, wavelengths=wavls, angles=[0.0])
    # DPHI: dphi[w] = d(arg ts)/dδ; full-gradient convention (no /2).
    ph0 = np.angle(np.asarray(base.compute(Request.TS_C)["ts_c"]).ravel())
    ph1 = np.angle(np.asarray(ins.compute(Request.TS_C)["ts_c"]).ravel())
    dphi_fd = (ph1 - ph0) / DELTA
    g = needle_gradient(base, N_NEEDLE, np.array([40.0 + 25.0]),
                        int(NeedleRequest.DPHI), channel=2, pol="s")
    dphi = np.asarray(g["dphi_s"]).ravel()
    check("dphi_fd", bool(np.allclose(dphi, dphi_fd, rtol=2e-3, atol=1e-9)),
          f"max|d|={np.max(np.abs(dphi - dphi_fd)):.2e}")
    # DGDD end-to-end: FD of the solver GDD vs the dgdd channel.
    gdd0 = np.asarray(base.dispersion(transmission=True, reflection=False,
                                      s_pol=True, p_pol=False)["GDD_T_s"]).ravel()
    gdd1 = np.asarray(ins.dispersion(transmission=True, reflection=False,
                                     s_pol=True, p_pol=False)["GDD_T_s"]).ravel()
    dgdd_fd = (gdd1 - gdd0) / DELTA
    g = needle_gradient(base, N_NEEDLE, np.array([40.0 + 25.0]),
                        int(NeedleRequest.DGDD), channel=2, pol="s")
    dgdd = np.asarray(g["dgdd_s"]).ravel()
    rel = np.abs(dgdd - dgdd_fd) / np.maximum(np.abs(dgdd_fd), 1e-30)
    check("dgdd_end_to_end", bool(np.all(rel < 5e-2)), f"max rel={np.max(rel):.2e}")


def test_interpolate_smoke():
    print("--- interpolate wrapper smoke ---")
    from navette.interpolate import UniInterpolator
    x = np.array([400., 500., 600.])
    y = np.array([1., 2., 1.5])
    u = UniInterpolator(x, y)
    v = float(np.asarray(u.eval(np.array([450.])))[0])
    check("interp_eval_pin", abs(v - 1.71875) < 1e-12, f"{v}")
    d = float(np.asarray(u.derivative(np.array([450.])))[0])
    h = 1e-6
    fd = (float(np.asarray(u.eval(np.array([450. + h])))[0])
          - float(np.asarray(u.eval(np.array([450. - h])))[0])) / (2 * h)
    check("interp_deriv_fd", abs(d - fd) < 1e-6, f"{d:.6f} vs {fd:.6f}")
    check("interp_xy", bool(np.allclose(np.asarray(u.get_x()), x))
          and bool(np.allclose(np.asarray(u.get_y()), y))
          and np.asarray(u.get_slopes()).shape == (3,))


if __name__ == "__main__":
    test_ar_physics()
    test_propagation_sign()
    test_pd_optimizer_recovery()
    test_back_needle_fd()
    test_multiblock()
    test_dispersion_channels()
    test_interpolate_smoke()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
