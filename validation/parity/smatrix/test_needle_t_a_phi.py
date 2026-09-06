#!/usr/bin/env python3
"""Parity/validation for needle T/A/phase gradients (Rust vs solver FD).

Covers the coherent needle path added for transmission, absorption and
phase targets:
  1. Wiring: forward-solver R/T/A/phi fed back as needle targets must give
     ~zero P_T/P_A/P_PHI (same conventions both sides).
  2. End-to-end FD: insert a thin needle slab at depth z, difference the
     w*(X-0)^2 merit, compare against 2*P (half-gradient convention).

Run explicitly:  python validation/parity/smatrix/test_needle_t_a_phi.py
"""

import numpy as np

from navette.smatrix.smatrix import ScatterMatrix, Request
from navette.smatrix.needle import NeedleRequest, needle_gradient

WAVLS = np.array([500.0, 600.0])
ANGLES = [0.0]
N_NEEDLE = 1.9 + 0j
DELTA = 5e-4  # nm, matches the Rust FD tests
TOL = 2e-3


def base_stack(absorbing=False):
    n_sio2 = (1.8 + 0.4j) if absorbing else (1.45 + 0j)
    idx = np.array([1.0 + 0j, 2.35 + 0j, n_sio2, 2.35 + 0j, 1.52 + 0j])
    thick = np.array([0.0, 40.0, 80.0, 30.0, 0.0])
    return ScatterMatrix(idx, thick, wavelengths=WAVLS, angles=ANGLES)


def needle_stack(xi, absorbing=False):
    """Base stack with a DELTA slab of needle glass at depth xi in layer 2."""
    n_sio2 = (1.8 + 0.4j) if absorbing else (1.45 + 0j)
    idx = np.array([1.0 + 0j, 2.35 + 0j, n_sio2, N_NEEDLE, n_sio2, 2.35 + 0j, 1.52 + 0j])
    thick = np.array([0.0, 40.0, xi, DELTA, 80.0 - xi, 30.0, 0.0])
    return ScatterMatrix(idx, thick, wavelengths=WAVLS, angles=ANGLES)


XI = 30.0
Z = [40.0 + XI]  # absolute, from top of layer 1


def as_point_array(d, key):
    v = np.asarray(d[key], dtype=np.float64)
    return np.ascontiguousarray(v.ravel())


def check_wiring():
    print("--- wiring: solver outputs as needle targets -> P ~ 0 ---")
    ok = True
    st = base_stack()
    fwd = st.compute(Request.TS | Request.TP | Request.A_S | Request.PHI_RS | Request.PHI_TS)
    nn = np.full(len(WAVLS), N_NEEDLE, dtype=np.complex128)
    out = needle_gradient(
        st, nn, Z,
        NeedleRequest.P_T | NeedleRequest.P_A | NeedleRequest.P_PHI,
        targets_t=as_point_array(fwd, "Ts"), targets_a=as_point_array(fwd, "A_s"),
        targets_phi=as_point_array(fwd, "phi_ts"), channel=2, pol="s",
    )
    for k in ("P_T_s", "P_A_s", "P_PHI_s"):
        m = abs(out[k]).max()
        good = m < 1e-9
        ok &= good
        print(f"  {k}: max|P| = {m:.3e} {'OK' if good else 'FAIL'}")
    # r-phase channel as well
    out0 = needle_gradient(
        st, nn, Z, NeedleRequest.P_PHI,
        targets_phi=as_point_array(fwd, "phi_rs"), channel=0, pol="s",
    )
    m = abs(out0["P_PHI_s"]).max()
    good = m < 1e-9
    ok &= good
    print(f"  P_PHI_s(ch0): max|P| = {m:.3e} {'OK' if good else 'FAIL'}")
    return ok


def check_fd(quantity, req_flag, needle_req, absorbing=False, channel=2):
    """FD of the w*(X-0)^2 merit vs 2*P at depth Z."""
    st0 = base_stack(absorbing)
    st1 = needle_stack(XI, absorbing)
    q0 = np.asarray(st0.compute(req_flag)[quantity], dtype=np.float64).ravel()
    q1 = np.asarray(st1.compute(req_flag)[quantity], dtype=np.float64).ravel()
    fd = (q1 * q1 - q0 * q0) / DELTA  # == 2*P under the half convention
    nn = np.full(len(WAVLS), N_NEEDLE, dtype=np.complex128)
    kwargs = {} if channel == 2 else {"channel": channel}
    out = needle_gradient(st0, nn, Z, needle_req, pol="s", **kwargs)
    key = [k for k in out if k.startswith("P_")][0]
    p = np.asarray(out[key]).ravel()  # (n_wavs,) after ravel of (1, W, 1)
    scale = np.maximum(np.abs(fd), np.abs(p)).clip(min=1e-12)
    err = np.abs(fd - 2.0 * p) / scale
    good = bool((err < TOL).all())
    print(f"  {key}: max rel err = {err.max():.2e} {'OK' if good else 'FAIL'}")
    return good


def incoh_stack(needle_xi=None, absorbing=False):
    n_sio2 = (1.8 + 0.4j) if absorbing else (1.45 + 0j)
    if needle_xi is None:
        idx = np.array([1.0 + 0j, 2.35 + 0j, n_sio2, 2.35 + 0j, 1.52 + 0j])
        thick = np.array([0.0, 40.0, 60.0, 50.0, 0.0])
        flags = [0, 0, 1, 0, 0]
    else:
        idx = np.array([1.0 + 0j, 2.35 + 0j, n_sio2, 2.35 + 0j,
                        N_NEEDLE, 2.35 + 0j, 1.52 + 0j])
        thick = np.array([0.0, 40.0, 60.0, needle_xi, DELTA,
                          50.0 - needle_xi, 0.0])
        flags = [0, 0, 1, 0, 0, 0, 0]
    return ScatterMatrix(idx, thick, wavelengths=WAVLS, angles=ANGLES,
                         incoherent_flags=flags)


Z_MB = [120.0]  # block 1 (layer 3 spans 100..150), xi = 20


def check_pmb():
    print("--- Pmb_T/A: wiring + FD on incoherent stack ---")
    ok = True
    st = incoh_stack()
    fwd = st.compute(Request.TS | Request.A_S)
    nn = np.full(len(WAVLS), N_NEEDLE, dtype=np.complex128)
    out = needle_gradient(
        st, nn, Z_MB, NeedleRequest.P_MB_T | NeedleRequest.P_MB_A,
        targets_t=as_point_array(fwd, "Ts"),
        targets_a=as_point_array(fwd, "A_s"), pol="s",
    )
    for k in ("Pmb_T_s", "Pmb_A_s"):
        m = abs(out[k]).max()
        good = m < 1e-9
        ok &= good
        print(f"  {k}: max|P| = {m:.3e} {'OK' if good else 'FAIL'}")
    # FD of the cascade T^2 / A^2 merits (absorbing spacer-stack for A)
    for qty, req, key, absorbing in [("T", Request.TS, "Ts", False),
                                      ("A", Request.A_S, "A_s", True)]:
        s0 = incoh_stack(absorbing=absorbing)
        s1 = incoh_stack(needle_xi=20.0, absorbing=absorbing)
        q0 = np.asarray(s0.compute(req)[key], dtype=np.float64).ravel()
        q1 = np.asarray(s1.compute(req)[key], dtype=np.float64).ravel()
        fd = (q1 * q1 - q0 * q0) / DELTA
        flag = NeedleRequest.P_MB_T if qty == "T" else NeedleRequest.P_MB_A
        out = needle_gradient(s0, nn, Z_MB, flag, pol="s")
        p = np.asarray(out[f"Pmb_{qty}_s"]).ravel()
        scale = np.maximum(np.abs(fd), np.abs(p)).clip(min=1e-12)
        err = np.abs(fd - 2.0 * p) / scale
        good = bool((err < 1e-2).all())
        ok &= good
        print(f"  Pmb_{qty}_s: max rel err = {err.max():.2e} {'OK' if good else 'FAIL'}")
    return ok


def membrane(absorbing=False, incoherent=False, needle_delta=None):
    """Palindromic membrane air/TiO2/SiO2/TiO2/air (mirror symmetry makes
    back-incidence quantities equal front ones, so the forward solver
    validates back-channel plumbing). Center plane at z = 80."""
    n_mid = (1.8 + 0.4j) if absorbing else (1.45 + 0j)
    if needle_delta is None:
        idx = np.array([1.0 + 0j, 2.35 + 0j, n_mid, 2.35 + 0j, 1.0 + 0j])
        thick = np.array([0.0, 40.0, 80.0, 40.0, 0.0])
        flags = [0, 0, 1 if incoherent else 0, 0, 0]
    else:
        # Twin half-slabs at the mirror plane: an exact palindromic
        # insertion (total +delta), so back == front still holds exactly
        # for the FD oracle (a single off-center slab would do too, but a
        # centered substitution would measure the wrong derivative).
        h = needle_delta / 2.0
        idx = np.array([1.0 + 0j, 2.35 + 0j, n_mid, N_NEEDLE, N_NEEDLE,
                        n_mid, 2.35 + 0j, 1.0 + 0j])
        thick = np.array([0.0, 40.0, 40.0, h, h, 40.0, 40.0, 0.0])
        flags = [0, 0, 1 if incoherent else 0, 0, 0, 0, 0, 0]
    return ScatterMatrix(idx, thick, wavelengths=WAVLS, angles=ANGLES,
                         incoherent_flags=flags)


Z_C = [80.0]  # membrane mirror plane


def check_back_coherent():
    print("--- back channels (coherent): wiring + center FD ---")
    ok = True
    nn = np.full(len(WAVLS), N_NEEDLE, dtype=np.complex128)
    st = membrane()
    fwd = st.compute(Request.TS | Request.RS)
    out = needle_gradient(
        st, nn, Z_C,
        NeedleRequest.P_TB | NeedleRequest.P_RB,
        targets_tb=as_point_array(fwd, "Ts"),
        targets_rb=as_point_array(fwd, "Rs"), pol="s",
    )
    for k in ("P_TB_s", "P_RB_s"):
        m = abs(out[k]).max()
        good = m < 1e-9
        ok &= good
        print(f"  {k}: max|P| = {m:.3e} {'OK' if good else 'FAIL'}")
    sta = membrane(absorbing=True)
    fa = sta.compute(Request.A_S)
    outa = needle_gradient(sta, nn, Z_C, NeedleRequest.P_AB,
                            targets_ab=as_point_array(fa, "A_s"), pol="s")
    m = abs(outa["P_AB_s"]).max()
    good = m < 1e-9
    ok &= good
    print(f"  P_AB_s: max|P| = {m:.3e} {'OK' if good else 'FAIL'}")
    # Twin-slab insertion FD on the absorbing membrane (stays palindromic):
    # back == front throughout, so the forward FD oracles apply to P_XB.
    # Merit change ~= 2*(P_a*d/2 + P_b*d/2) -> FD/2 vs mean slab P.
    s0 = membrane(absorbing=True)
    s1 = membrane(absorbing=True, needle_delta=DELTA)
    req_of = {"Ts": Request.TS, "Rs": Request.RS, "A_s": Request.A_S}
    z_pair = [80.0 + DELTA / 4.0, 80.0 + 3.0 * DELTA / 4.0]
    both = needle_gradient(
        s0, nn, z_pair,
        NeedleRequest.P_TB | NeedleRequest.P_RB | NeedleRequest.P_AB |
        NeedleRequest.P_T | NeedleRequest.P | NeedleRequest.P_A,
        pol="s",
    )
    for qty, nkey, bkey in [("Ts", "P_T_s", "P_TB_s"),
                             ("Rs", "P_s", "P_RB_s"),
                             ("A_s", "P_A_s", "P_AB_s")]:
        q0 = np.asarray(s0.compute(req_of[qty])[qty], dtype=np.float64).ravel()
        q1 = np.asarray(s1.compute(req_of[qty])[qty], dtype=np.float64).ravel()
        fd = (q1 * q1 - q0 * q0) / DELTA / 2.0  # half convention
        p = np.asarray(both[bkey]).reshape(len(WAVLS), 2).mean(axis=1)
        scale = np.maximum(np.abs(fd), np.abs(p)).clip(min=1e-12)
        err = np.abs(fd - p) / scale
        good = bool((err < 1e-2).all())
        ok &= good
        print(f"  {bkey}: FD err = {err.max():.2e} {'OK' if good else 'FAIL'}")
    # Mirror symmetry on the base stack: back == front exactly at z = 80.
    sym = needle_gradient(
        s0, nn, Z_C,
        NeedleRequest.P_TB | NeedleRequest.P_RB | NeedleRequest.P_AB |
        NeedleRequest.P_T | NeedleRequest.P | NeedleRequest.P_A,
        pol="s",
    )
    for nkey, bkey in [("P_T_s", "P_TB_s"), ("P_s", "P_RB_s"),
                        ("P_A_s", "P_AB_s")]:
        pf = np.asarray(sym[nkey]).ravel()
        pb = np.asarray(sym[bkey]).ravel()
        sc = np.maximum(np.abs(pf), np.abs(pb)).clip(min=1e-12)
        good = bool((np.abs(pb - pf) / sc < 1e-12).all())
        ok &= good
        print(f"  {bkey}=={nkey}: {'OK' if good else 'FAIL'}")
    return ok


def check_back_pmb():
    print("--- back channels (multiblock): wiring ---")
    ok = True
    nn = np.full(len(WAVLS), N_NEEDLE, dtype=np.complex128)
    st = membrane(incoherent=True)
    fwd = st.compute(Request.TS | Request.RS | Request.A_S)
    out = needle_gradient(
        st, nn, [15.0],
        NeedleRequest.P_MB_TB | NeedleRequest.P_MB_RB | NeedleRequest.P_MB_AB,
        targets_tb=as_point_array(fwd, "Ts"),
        targets_rb=as_point_array(fwd, "Rs"),
        targets_ab=as_point_array(fwd, "A_s"), pol="s",
    )
    for k in ("Pmb_TB_s", "Pmb_RB_s", "Pmb_AB_s"):
        m = abs(out[k]).max()
        good = m < 1e-9
        ok &= good
        print(f"  {k}: max|P| = {m:.3e} {'OK' if good else 'FAIL'}")
    return ok


def main():
    ok = True
    ok &= check_wiring()
    ok &= check_pmb()
    ok &= check_back_coherent()
    ok &= check_back_pmb()
    print("--- FD: P_T vs T^2 merit (s-pol) ---")
    ok &= check_fd("Ts", Request.TS, NeedleRequest.P_T)
    print("--- FD: P_A vs A^2 merit on absorbing stack (s-pol) ---")
    ok &= check_fd("A_s", Request.A_S, NeedleRequest.P_A, absorbing=True)
    print("--- FD: P_PHI(t_fwd) vs wrapped-phase merit (s-pol) ---")
    # Phase merit needs an offset target (residual nonzero); emulate here by
    # shifting: FD of (wrap(phi-t))^2 with t = phi0 - 0.05 per wavelength.
    st0 = base_stack()
    st1 = needle_stack(XI)
    phi0 = np.asarray(st0.compute(Request.PHI_TS)["phi_ts"], dtype=np.float64).ravel()
    phi1 = np.asarray(st1.compute(Request.PHI_TS)["phi_ts"], dtype=np.float64).ravel()
    tgt = phi0 - 0.05
    wrap = lambda x: x - 2 * np.pi * np.round(x / (2 * np.pi))
    fd = (wrap(phi1 - tgt) ** 2 - 0.05 ** 2) / DELTA
    nn = np.full(len(WAVLS), N_NEEDLE, dtype=np.complex128)
    npts = len(ANGLES) * len(WAVLS)
    out = needle_gradient(st0, nn, Z, NeedleRequest.P_PHI, pol="s",
                          targets_phi=np.ascontiguousarray(tgt), channel=2)
    p = np.asarray(out["P_PHI_s"]).ravel()
    scale = np.maximum(np.abs(fd), np.abs(p)).clip(min=1e-12)
    err = np.abs(fd - p) / scale  # phase is the FULL gradient (no half factor)
    good = bool((err < TOL).all())
    ok &= good
    print(f"  P_PHI_s: max rel err = {err.max():.2e} {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
