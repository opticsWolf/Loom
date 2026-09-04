#!/usr/bin/env python3
"""Backside solver outputs: analytic + symmetry + energy validation.

Covers Request.PHI_RBS/RBP/TBS/TBP and RBS_C/RBP_C/TBS_C/TBP_C:
  1. Single interface vs Fresnel analytics (both pols, normal + oblique).
  2. Palindromic membrane: back == front on all 8 outputs (both pols).
  3. Energy: 0 <= A_back <= 1; A_back == A_front on symmetric stacks.
  4. No-state-pollution: old-mask outputs identical before/after a BACKSIDE
     request in the same session.

Run explicitly:  python validation/parity/smatrix/test_backside_solver.py
"""

import numpy as np

from navette.smatrix.smatrix import ScatterMatrix, Request

WAVLS = np.array([450.0, 550.0, 650.0])
OK = True


def check(name, cond, detail=""):
    global OK
    print(f"  {name}: {'OK' if cond else 'FAIL'} {detail}")
    OK = bool(cond) and OK


def fresnel(n0, n1, sin_t0, pol):
    """Amplitude (rb, tb) for incidence FROM medium 1 (back experiment)."""
    sin_t1 = n0 * sin_t0 / n1
    cos0 = np.sqrt(1 - sin_t0 ** 2 + 0j)
    cos1 = np.sqrt(1 - sin_t1 ** 2 + 0j)
    if pol == "s":
        y0, y1 = n0 * cos0, n1 * cos1
    else:
        y0, y1 = n0 / cos0, n1 / cos1
    # incident from medium 1 onto medium 0
    rb = (y1 - y0) / (y1 + y0)
    tb = 2 * y1 / (y1 + y0)
    return rb, tb


def test_analytic():
    print("--- single interface vs Fresnel ---")
    n0, n1 = 1.0 + 0j, 1.52 + 0.03j
    for deg in [0.0, 35.0]:
        th = np.radians(deg)
        s = ScatterMatrix(np.array([n0, n1]), np.array([0.0, 0.0]),
                          wavelengths=WAVLS, angles=[deg])
        out = s.compute(Request.BACKSIDE)
        st0 = np.sin(th)
        for pol, rs_k, tb_k, ph_r, ph_t in [("s", "rbs_c", "tbs_c", "phi_rbs", "phi_tbs"),
                                            ("p", "rbp_c", "tbp_c", "phi_rbp", "phi_tbp")]:
            rb, tb = fresnel(n0, n1, st0, pol)
            rb = np.full(len(WAVLS), rb, dtype=np.complex128)
            tb = np.full(len(WAVLS), tb, dtype=np.complex128)
            got_r = np.asarray(out[rs_k]).ravel()
            got_t = np.asarray(out[tb_k]).ravel()
            check(f"back amp {pol} @{deg}deg", np.allclose(got_r, rb, rtol=0, atol=1e-14)
                  and np.allclose(got_t, tb, rtol=0, atol=1e-14))
            check(f"back phase {pol} @{deg}deg",
                  np.allclose(np.asarray(out[ph_r]).ravel(), np.angle(rb), atol=1e-12)
                  and np.allclose(np.asarray(out[ph_t]).ravel(), np.angle(tb), atol=1e-12))


def membrane(absorbing=False):
    n_mid = (1.8 + 0.4j) if absorbing else (1.45 + 0j)
    idx = np.array([1.0 + 0j, 2.35 + 0j, n_mid, 2.35 + 0j, 1.0 + 0j])
    return ScatterMatrix(idx, np.array([0.0, 40.0, 80.0, 40.0, 0.0]),
                         wavelengths=WAVLS, angles=[0.0, 30.0])


def test_symmetry():
    print("--- palindromic membrane: back == front ---")
    pairs = [("phi_rbs", "phi_rs"), ("phi_rbp", "phi_rp"),
             ("phi_tbs", "phi_ts"), ("phi_tbp", "phi_tp"),
             ("rbs_c", "rs_c"), ("rbp_c", "rp_c"),
             ("tbs_c", "ts_c"), ("tbp_c", "tp_c")]
    for absorbing in [False, True]:
        st = membrane(absorbing)
        out = st.compute(Request.BACKSIDE | Request.PHI_RS | Request.PHI_RP |
                         Request.PHI_TS | Request.PHI_TP | Request.RS_C |
                         Request.RP_C | Request.TS_C | Request.TP_C)
        for back, front in pairs:
            b = np.asarray(out[back]).ravel()
            f = np.asarray(out[front]).ravel()
            if back.startswith("phi"):
                d = (b - f + np.pi) % (2 * np.pi) - np.pi
                good = bool((np.abs(d) < 1e-12).all())
            else:
                sc = np.maximum(np.abs(b), np.abs(f)).clip(min=1e-30)
                good = bool((np.abs(b - f) / sc < 1e-12).all())
            check(f"{back}=={front} (absorbing={absorbing})", good)


def test_energy():
    print("--- backside energy bounds ---")
    for absorbing in [False, True]:
        st = membrane(absorbing)
        out = st.compute(Request.ABSORPTION)
        full = st.compute(Request.RBS_C | Request.TBS_C)
        fb = 1.0  # air/air membrane: backward flux factor is 1
        ab = 1 - np.abs(np.asarray(full["rbs_c"])).ravel() ** 2 \
            - np.abs(np.asarray(full["tbs_c"])).ravel() ** 2 * fb
        good = bool(((ab >= -1e-12) & (ab <= 1 + 1e-12)).all())
        check(f"0<=A_back<=1 (absorbing={absorbing})", good,
              f"A range [{ab.min():.4f}, {ab.max():.4f}]")
        if not absorbing:
            check("lossless A_back ~ 0", bool((np.abs(ab) < 1e-12).all()))
        else:
            af = np.asarray(out["A_s"]).ravel()
            af = np.broadcast_to(af, ab.shape) if af.shape != ab.shape else af
            check("symmetric A_back == A_front",
                  bool((np.abs(ab - af) < 1e-12).all()))


def test_no_pollution():
    print("--- old-mask outputs unaffected by BACKSIDE requests ---")
    st = membrane(absorbing=True)
    mask = (Request.PHOTOMETRY | Request.ELLIPSOMETRY | Request.ABSORPTION |
            Request.RS_C | Request.RP_C | Request.TS_C | Request.TP_C)
    before = st.compute(mask)
    _ = st.compute(Request.BACKSIDE)
    after = st.compute(mask)
    same = all(np.array_equal(np.asarray(before[k]), np.asarray(after[k]))
               for k in before)
    check("bit-identical across BACKSIDE request", same)


def main():
    test_analytic()
    test_symmetry()
    test_energy()
    test_no_pollution()
    print("ALL OK" if OK else "MISMATCH")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
