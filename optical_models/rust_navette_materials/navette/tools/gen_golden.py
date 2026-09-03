#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden-reference generator for the navette.materials Rust port.

With fastmath disabled on the Python side, plain float64 NumPy evaluations of
the documented kernel formulas ARE the reference. This script writes, per case,
three .npy files into ../golden/:

    <case>__wl.npy   wavelength grid [nm]        (float64)
    <case>__re.npy   Re(nk)  (or Re(eps) for EMA cases)
    <case>__im.npy   Im(nk)  (or Im(eps) for EMA cases)

The Rust parity test (tests/parity.rs) mirrors the same parameter values, runs
the ported kernels on the wl grid, and asserts agreement. Keeping params in two
places is deliberate for now: the .npy holds only arrays, no metadata, so the
Rust side needs no JSON parser.
"""

import os
import numpy as np

HC = 1239.8419843320028  # eV·nm
OUT = os.path.join(os.path.dirname(__file__), "..", "golden")
os.makedirs(OUT, exist_ok=True)


def save(case, wl, nk):
    np.save(os.path.join(OUT, f"{case}__wl.npy"), np.asarray(wl, dtype=np.float64))
    np.save(os.path.join(OUT, f"{case}__re.npy"), np.asarray(nk.real, dtype=np.float64))
    np.save(os.path.join(OUT, f"{case}__im.npy"), np.asarray(nk.imag, dtype=np.float64))
    print(f"  {case:28s} n={len(wl):4d}  "
          f"re[0]={nk.real[0]:.6g}  im[0]={nk.imag[0]:.6g}")


# ---- reference kernels (faithful to the @njit bodies, fastmath off) ----

def cauchy_n(wl_um2, A, B, C):
    return A + B / wl_um2 + C / (wl_um2 ** 2)

def urbach_k(wl_nm, alpha0, Eu, lambda_g):
    E = HC / wl_nm
    E_g = HC / lambda_g
    wl_m = wl_nm * 1e-9
    k = np.zeros_like(wl_nm)
    mask = E < E_g
    k[mask] = alpha0 * np.exp((E[mask] - E_g) / Eu) * wl_m[mask] / (4 * np.pi)
    return k

def sellmeier_n(l2, B1, C1, B2, C2, B3, C3):
    t1 = B1 * l2 / (l2 - C1)
    t2 = B2 * l2 / (l2 - C2)
    t3 = np.where(B3 != 0.0, B3 * l2 / (l2 - C3), 0.0)
    return np.sqrt(1.0 + t1 + t2 + t3)

def lorentz_nk(wl_nm, osc, eps_inf):
    E = HC / wl_nm
    E_sq = E * E
    eps = np.full(wl_nm.shape, eps_inf + 0j, dtype=np.complex128)
    for (E0, Gamma, f0) in osc:
        eps += (f0 * E0 * E0) / ((E0 * E0 - E_sq) - 1j * (E * Gamma))
    return np.sqrt(eps)

def drude_nk(wl_nm, wp, gamma, eps_inf):
    E = HC / wl_nm + 1e-12
    eps = np.full(wl_nm.shape, eps_inf + 0j, dtype=np.complex128)
    eps -= (wp ** 2) / (E * E + 1j * (gamma * E))
    return np.sqrt(eps)

def drude_lorentz_nk(wl_nm, wp, gamma_d, eps_inf, osc):
    E = HC / wl_nm + 1e-12
    E_sq = E * E
    eps = np.full(wl_nm.shape, eps_inf + 0j, dtype=np.complex128)
    eps -= (wp ** 2) / (E_sq + 1j * (gamma_d * E))
    for (E0, Gamma, f0) in osc:
        eps += (f0 * E0 * E0) / ((E0 * E0 - E_sq) - 1j * (E * Gamma))
    return np.sqrt(eps)

# EMA mixers (return epsilon; harness stores Re/Im of eps directly)
def looyenga_eps(n_i, n_h, f):
    ei, eh = n_i * n_i, n_h * n_h
    p = 1.0 / 3.0
    cbrt = f * ei ** p + (1 - f) * eh ** p
    return cbrt ** 3.0

def maxwell_garnett_eps(n_i, n_h, f):
    ei, eh = n_i * n_i, n_h * n_h
    diff = ei - eh
    eh2 = 2.0 * eh
    return eh * ((ei + eh2 + 2.0 * f * diff) / (ei + eh2 - f * diff))

def bruggeman_eps(n_i, n_h, f, max_iter=100, tol=1e-9):
    ei, eh = n_i * n_i, n_h * n_h
    eps = (ei + eh) * 0.5
    out = np.empty_like(eps)
    for k in range(len(eps)):
        e = eps[k]
        ev_i, ev_h = ei[k], eh[k]
        for _ in range(max_iter):
            den_i = ev_i + 2.0 * e
            den_h = ev_h + 2.0 * e
            f_total = f * (ev_i - e) / den_i + (1 - f) * (ev_h - e) / den_h
            df = f * (-3.0 * ev_i) / (den_i * den_i) + (1 - f) * (-3.0 * ev_h) / (den_h * den_h)
            delta = -f_total / (df + 1e-15)
            e += delta
            if (delta.real ** 2 + delta.imag ** 2) < tol * tol:
                break
        out[k] = e
    return out


def cody_eps2(E, Eg, Et, Eu, osc):
    """ε₂(E) for the multi-oscillator continuous Cody-Lorentz model (no fastmath)."""
    E = np.asarray(E, dtype=np.float64)
    out = np.zeros_like(E)
    Et2 = Et * Et
    A_t_total = 0.0
    for (E0, A, Gam, Ep) in osc:
        E0sq = E0 * E0
        cody_Et = ((Et - Eg) ** 2) / ((Et - Eg) ** 2 + Ep * Ep) if Et > Eg else 0.0
        denom_Et = (Et2 - E0sq) ** 2 + (Et * Gam) ** 2
        A_t_total += A * cody_Et * (Et * Gam) / denom_Et
    band = (E >= Et) & (E > Eg)
    Eb = E[band]; dsq = (Eb - Eg) ** 2; Eb2 = Eb * Eb
    val = np.zeros_like(Eb)
    for (E0, A, Gam, Ep) in osc:
        E0sq = E0 * E0
        cody = dsq / (dsq + Ep * Ep)
        denom = (Eb2 - E0sq) ** 2 + (Eb * Gam) ** 2
        val += A * cody * (Eb * Gam) / denom
    out[band] = val
    urb = (E < Et) & (E > 1e-9)
    out[urb] = A_t_total * (Et / E[urb]) * np.exp((E[urb] - Et) / Eu)
    return out


_CL_GRID_N = 8192
_CL_M = 1
while _CL_M < 2 * _CL_GRID_N + 1:
    _CL_M <<= 1                          # 32768
_CL_E_FULL = np.linspace(0.01, 80.0, _CL_GRID_N)
_CL_H = np.empty(_CL_M // 2 + 1, dtype=np.complex128)
_CL_H[0] = 0.0
_CL_H[1:] = -1j


def cody_kk(eps2, eps_inf):
    N, M = _CL_GRID_N, _CL_M
    buf = np.zeros(M, dtype=np.float64)
    buf[N + 1: N + 1 + N] = eps2
    buf[1: N + 1] = -eps2[::-1]
    F = np.fft.rfft(buf)
    hilb = np.fft.irfft(F * _CL_H, n=M)
    return eps_inf - hilb[N: N + N]


def cody_lorentz_nk(wl, Eg, Et, Eu, osc, eps_inf):
    target_E = HC / wl
    eps2_full = cody_eps2(_CL_E_FULL, Eg, Et, Eu, osc)
    eps1_full = cody_kk(eps2_full, eps_inf)
    eps1_t = np.interp(target_E, _CL_E_FULL, eps1_full)
    eps2_t = cody_eps2(target_E, Eg, Et, Eu, osc)
    return np.sqrt(eps1_t + 1j * eps2_t)


def fb_term(E, Eg, A, B, C):
    disc = 4 * C - B ** 2
    Q = 1e-6 if disc <= 1e-12 else 0.5 * np.sqrt(disc)
    B0 = (A / Q) * (-(B ** 2 / 2) + Eg * B - Eg ** 2 + C)
    C0 = (A / Q) * ((Eg ** 2 + C) * (B / 2) - 2 * Eg * C)
    denom = E ** 2 - B * E + C
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    k = np.zeros_like(E)
    m = E >= Eg
    k[m] = (A * (E[m] - Eg) ** 2) / denom[m]
    n = (B0 * E + C0) / denom
    return n + 1j * k


def fb_interband(wl, n_inf, ib):
    E = HC / wl
    n = np.full(E.shape, n_inf)
    k = np.zeros_like(E)
    for (Eg, A, B, C) in ib:
        t = fb_term(E, Eg, A, B, C); n += t.real; k += t.imag
    return n + 1j * k


def fb_metal(wl, n_inf, fe, ib):
    E = HC / wl
    n = np.full(E.shape, n_inf)
    k = np.zeros_like(E)
    if fe[0] > 0.0:
        t = fb_term(E, 0.0, fe[0], fe[1], fe[2]); n += t.real; k += t.imag
    for (Eg, A, B, C) in ib:
        t = fb_term(E, Eg, A, B, C); n += t.real; k += t.imag
    return n + 1j * k


def ubf_eps2(E, osc):
    """Monolog-Lorentz ε₂; osc rows [Eg, Ec, β, A, Γ, γ]. Replicates fast paths."""
    E = np.asarray(E, dtype=np.float64)
    out = np.zeros_like(E)
    for i in range(E.shape[0]):
        Ei = E[i]
        if Ei < 1e-9:
            continue
        Ei_sq = Ei * Ei
        val = 0.0
        for (Eg, Ec, Beta, A, G, Y) in osc:
            x = Beta * (Ei - Eg)
            if x > 50.0:
                base = x
            elif x < -50.0:
                base = 0.0
            else:
                base = np.log(1.0 + np.exp(x))
            if Y == 2.0:
                band = base * base
            elif Y == 0.5:
                band = np.sqrt(base)
            elif Y == 1.0:
                band = base
            else:
                band = base ** Y
            denom = (Ei_sq - Ec * Ec) ** 2 + (G * Ei) ** 2
            lorentz = (Ei * G * Ec) / denom
            val += (A / Ei) * band * lorentz
        out[i] = val
    return out


def ubf_nk(wl, osc, eps_inf):
    target_E = HC / wl
    eps2_full = ubf_eps2(_CL_E_FULL, osc)
    eps1_full = cody_kk(eps2_full, eps_inf)
    eps1_t = np.interp(target_E, _CL_E_FULL, eps1_full)
    eps2_t = ubf_eps2(target_E, osc)
    return np.sqrt(eps1_t + 1j * eps2_t)


def tauc_eps2(E, Eg, osc):
    """Tauc-Lorentz ε₂ (Jellison-Modine); osc rows (A, E0, C); shared Eg."""
    E = np.asarray(E, dtype=np.float64)
    out = np.zeros_like(E)
    m = E > Eg
    Em = E[m]
    tauc = (Em - Eg) ** 2
    val = np.zeros_like(Em)
    for (A, E0, C) in osc:
        denom = ((Em ** 2 - E0 ** 2) ** 2 + C ** 2 * Em ** 2) * Em
        val += A * E0 * C * tauc / denom
    out[m] = val
    return out


def tauc_lorentz_nk(wl, Eg, osc, eps_inf):
    target_E = HC / wl
    eps2_full = tauc_eps2(_CL_E_FULL, Eg, osc)
    eps1_full = cody_kk(eps2_full, eps_inf)
    eps1_t = np.interp(target_E, _CL_E_FULL, eps1_full)
    eps2_t = tauc_eps2(target_E, Eg, osc)
    return np.sqrt(eps1_t + 1j * eps2_t)


def main():
    print("Generating golden references ->", os.path.abspath(OUT))

    # --- Cauchy ---
    wl = np.linspace(400.0, 700.0, 100)
    l2 = (wl / 1000.0) ** 2
    save("cauchy_basic", wl, cauchy_n(l2, 1.5, 0.01, 0.0001) + 0j)
    save("cauchy_single", np.array([550.0]), cauchy_n((np.array([550.0]) / 1000.0) ** 2, 1.5, 0.01, 0.0001) + 0j)

    # --- Cauchy + Urbach (straddles band edge at 400 nm) ---
    wl = np.linspace(300.0, 700.0, 200)
    l2 = (wl / 1000.0) ** 2
    nk = cauchy_n(l2, 2.5, 0.02, 0.0005) + 1j * urbach_k(wl, 1e4, 0.05, 400.0)
    save("cauchy_urbach", wl, nk)

    # --- Sellmeier (BK7, three-term) ---
    wl = np.linspace(400.0, 700.0, 100)
    l2 = (wl / 1000.0) ** 2
    save("sellmeier_bk7", wl, sellmeier_n(l2, 1.03961212, 0.00600069867,
                                          0.231792344, 0.0200179144,
                                          1.01046945, 103.560653) + 0j)

    # --- Sellmeier (two-term: B3=0 branch) ---
    save("sellmeier_2term", wl, sellmeier_n(l2, 1.0, 0.01, 0.3, 0.05, 0.0, 0.0) + 0j)

    # --- Sellmeier + Urbach ---
    wl = np.linspace(330.0, 800.0, 250)
    l2 = (wl / 1000.0) ** 2
    nk = sellmeier_n(l2, 1.4313, 0.01, 0.65, 0.025, 0.0, 0.0) + 1j * urbach_k(wl, 1e5, 0.06, 380.0)
    save("sellmeier_urbach", wl, nk)

    # --- Lorentz (2 oscillators) ---
    wl = np.linspace(200.0, 800.0, 200)
    osc = [(3.0, 0.2, 0.5), (4.5, 0.1, 0.7)]
    save("lorentz_2osc", wl, lorentz_nk(wl, osc, 1.0))

    # --- Drude ---
    wl = np.linspace(150.0, 600.0, 150)
    save("drude_basic", wl, drude_nk(wl, 2.5, 0.3, 3.5))

    # --- Drude-Lorentz (2 oscillators) ---
    wl = np.linspace(200.0, 2000.0, 300)
    osc = [(2.0, 0.5, 1.0), (3.5, 0.8, 0.4)]
    save("drude_lorentz", wl, drude_lorentz_nk(wl, 9.0, 0.05, 1.0, osc))

    # --- EMA cases (store eps, not nk) ---
    wl = np.linspace(300.0, 900.0, 120)
    # Two dispersive-ish constituents built from simple complex ramps.
    n_i = np.linspace(2.0, 2.4, 120) + 1j * np.linspace(0.05, 0.2, 120)
    n_h = np.linspace(1.4, 1.5, 120) + 1j * np.linspace(0.0, 0.02, 120)
    save("ema_looyenga", wl, looyenga_eps(n_i, n_h, 0.3))
    save("ema_maxwell_garnett", wl, maxwell_garnett_eps(n_i, n_h, 0.3))
    save("ema_bruggeman", wl, bruggeman_eps(n_i, n_h, 0.3))

    # --- Cody-Lorentz (FFT-KK path): single and multi-oscillator ---
    wl = np.linspace(300.0, 1200.0, 400)
    osc1 = [(3.40, 60.0, 2.4, 1.0)]
    save("cody_single", wl, cody_lorentz_nk(wl, 1.64, 1.80, 0.15, osc1, 1.0))
    osc2 = [(3.40, 60.0, 2.4, 1.0), (4.70, 40.0, 1.8, 0.5)]
    save("cody_multi", wl, cody_lorentz_nk(wl, 1.64, 1.80, 0.15, osc2, 1.0))

    # --- Forouhi-Bloomer (2019 interband, 2021 metal) ---
    wl = np.linspace(200.0, 800.0, 200)
    save("fb_single", wl, fb_interband(wl, 1.5, [(3.0, 0.1, 6.0, 12.0)]))
    save("fb_multi", wl, fb_interband(wl, 1.2, [(3.0, 0.1, 6.0, 12.0), (4.5, 0.05, 9.0, 22.0)]))
    # Unphysical discriminant (4C < B²) → Q = 1e-6 branch.
    save("fb_edge", wl, fb_interband(wl, 1.0, [(2.0, 0.1, 10.0, 20.0)]))
    wl = np.linspace(200.0, 2000.0, 300)
    save("fb_metal", wl, fb_metal(wl, 1.0, [5.0, 0.5, 0.3], [(2.0, 0.2, 4.0, 5.0)]))

    # --- UBF Cody-Lorentz (Monolog-Lorentz; osc rows [Eg, Ec, β=1/Eu, A, Γ, γ]) ---
    wl = np.linspace(300.0, 1000.0, 300)
    ubf1 = [(1.5, 3.0, 1.0 / 0.2, 10.0, 1.0, 2.0)]
    save("ubf_single", wl, ubf_nk(wl, ubf1, 1.0))
    ubf2 = [(1.5, 3.0, 1.0 / 0.2, 10.0, 1.0, 2.0), (2.2, 4.5, 1.0 / 0.15, 6.0, 1.5, 0.5)]
    save("ubf_multi", wl, ubf_nk(wl, ubf2, 1.0))

    # --- Tauc-Lorentz (Jellison-Modine; osc rows (A, E0, C); shared Eg) ---
    wl = np.linspace(250.0, 1000.0, 300)
    save("tauc_single", wl, tauc_lorentz_nk(wl, 1.2, [(100.0, 4.0, 2.0)], 1.0))
    save("tauc_multi", wl, tauc_lorentz_nk(wl, 1.2, [(100.0, 4.0, 2.0), (50.0, 6.5, 1.5)], 1.5))

    print("Done.")


if __name__ == "__main__":
    main()
