#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror of the Rust color tests through the installed ``navette.color``.

Covers ``crates/navette-color/src/parity.rs`` (15 golden tests, same
vectors/tolerance ``|a-b| <= 1e-12 + 1e-9*|b|``), the per-function unit
tests (``func_01``..``func_14``), the broadcast semantics (``metrics.rs``)
and the white-point/matrix checks (``common.rs``/``func_08.rs``).

Bonus beyond Rust: the ``DE2000`` golden vector exists in ``golden.rs``
but is asserted by no Rust test — it is asserted here.

Gaps (Rust internals with no Python exposure, listed in
``rust_mirror_COVERAGE.md``): ``signed_pow``/``lab_f``/``LAB_*``
constants, ``mat3_*`` helpers, ``din99_coords``, ``xyz_to_uv_prime``,
``vec_mul_mat3`` (covered equivalently with numpy), ``broadcast_pair``.

Run explicitly:  python validation/parity/color/test_color_mirror.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_mirror as G

from navette import color as C

FAILURES = []
EPS = 1e-12


def close(a, b):
    return abs(float(a) - float(b)) <= EPS + 1e-9 * abs(float(b))


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


def check_rows(name, got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    ok_shape = got.shape == want.shape
    worst = float(np.max(np.abs(got - want))) if ok_shape else float("nan")
    tol_ok = bool(np.all(np.abs(got - want) <= EPS + 1e-9 * np.abs(want))) if ok_shape else False
    check(name, ok_shape and tol_ok, f"max|d|={worst:.3e}")


def check_vec(name, got, want):
    check_rows(name, np.asarray(got).ravel(), np.asarray(want).ravel())


# ---------------------------------------------------------------- parity.rs
def test_parity_goldens():
    print("--- parity.rs goldens (15 + DE2000 bonus) ---")
    check_rows("func_01_xyz_to_xyy", C.XYZ_to_xyY(G.XYZ_IN), G.F01_XYZ_TO_XYY)
    check_rows("func_01_xyy_to_xyz", C.xyY_to_XYZ(C.XYZ_to_xyY(G.XYZ_IN)), G.F01_XYY_TO_XYZ)
    check_rows("func_02_lab_to_lch", C.Lab_to_LCHab(G.LAB1_IN), G.F02_LAB_TO_LCH)
    check_rows("func_02_lch_to_lab", C.LCHab_to_Lab(C.Lab_to_LCHab(G.LAB1_IN)), G.F02_LCH_TO_LAB)
    check_rows("func_03_xyz_to_luv", C.XYZ_to_Luv(G.XYZ_IN), G.F03_XYZ_TO_LUV)
    check_rows("func_03_luv_to_xyz", C.Luv_to_XYZ(C.XYZ_to_Luv(G.XYZ_IN)), G.F03_LUV_TO_XYZ)
    check_rows("func_04_xyz_to_oklab", C.XYZ_to_Oklab(G.XYZ_IN), G.F04_XYZ_TO_OKLAB)
    check_rows("func_04_oklab_to_xyz", C.Oklab_to_XYZ(C.XYZ_to_Oklab(G.XYZ_IN)), G.F04_OKLAB_TO_XYZ)
    check_rows("func_05_srgb_to_oklab", C.sRGB_to_Oklab(G.SRGB_IN), G.F05_SRGB_TO_OKLAB)
    check_rows("func_05_oklab_to_srgb", C.Oklab_to_sRGB(C.sRGB_to_Oklab(G.SRGB_IN)), G.F05_OKLAB_TO_SRGB)
    un, vn = C.white_point_uv1960(np.asarray(C.REF_WHITE_D65))
    check("func_06_white_point", close(un, float(G.F06_UN_D65)) and close(vn, float(G.F06_VN_D65)),
          f"u={un:.6f} v={vn:.6f}")
    check_rows("func_06_xyz_to_uvw", C.XYZ_to_UVW(G.XYZ_IN), G.F06_XYZ_TO_UVW)
    check_rows("func_06_uvw_to_xyz", C.UVW_to_XYZ(C.XYZ_to_UVW(G.XYZ_IN)), G.F06_UVW_TO_XYZ)
    check_rows("func_07_xyz_to_ucs", C.XYZ_to_UCS(G.XYZ_IN), G.F07_XYZ_TO_UCS)
    check_rows("func_07_ucs_to_xyz", C.UCS_to_XYZ(C.XYZ_to_UCS(G.XYZ_IN)), G.F07_UCS_TO_XYZ)
    check_rows("func_07_xyz_to_ucs_uv", C.XYZ_to_UCS_uv(G.XYZ_IN), G.F07_XYZ_TO_UCS_UV)
    check_rows("func_07_uv1976_to_xy", C.Luv_uv_to_xy(G.F07_UVP_IN), G.F07_UV1976_TO_XY)
    check_rows("func_07_uv1960_to_xy", C.UCS_uv_to_xy(G.F07_UVP_IN), G.F07_UV1960_TO_XY)
    check_rows("func_08_adapt_d65_to_d50",
               C.chromatic_adaptation_VonKries(G.XYZ_IN, np.asarray(C.REF_WHITE_D65),
                                               np.asarray(C.REF_WHITE_D50)),
               G.F08_ADAPT_D65_TO_D50)
    check_rows("func_08_bradford_matrix",
               C.calc_transform_matrix(np.asarray(C.REF_WHITE_D65),
                                       np.asarray(C.REF_WHITE_D50)),
               G.F08_BRADFORD_MATRIX)
    check_vec("func_09_de76", C.delta_E_CIE1976(G.LAB1_IN, G.LAB2_IN), G.F09_DE76)
    check_vec("func_10_de94_graphic", C.delta_E_CIE1994(G.LAB1_IN, G.LAB2_IN), G.F10_DE94_GRAPHIC)
    check_vec("func_10_de94_textiles", C.delta_E_CIE1994(G.LAB1_IN, G.LAB2_IN, textiles=True),
              G.F10_DE94_TEXTILES)
    check_vec("func_11_cmc_2_1", C.delta_E_CMC(G.LAB1_IN, G.LAB2_IN, 2.0, 1.0), G.F11_DE_CMC_2_1)
    check_vec("func_11_cmc_1_1", C.delta_E_CMC(G.LAB1_IN, G.LAB2_IN, 1.0, 1.0), G.F11_DE_CMC_1_1)
    check_vec("func_12_din99", C.delta_E_DIN99(G.LAB1_IN, G.LAB2_IN), G.F12_DE_DIN99)
    check_vec("func_12_din99_textiles", C.delta_E_DIN99(G.LAB1_IN, G.LAB2_IN, textiles=True),
              G.F12_DE_DIN99_TEX)
    check_vec("func_13_spectral_adapt",
              C.spectral_to_sRGB(np.asarray(G.F13_SPD), np.asarray(G.F13_CMFS),
                                 np.asarray(G.F13_ILLUM), float(G.F13_INTERVAL),
                                 apply_adaptation=True), G.F13_SRGB_ADAPT)
    check_vec("func_13_spectral_noadapt",
              C.spectral_to_sRGB(np.asarray(G.F13_SPD), np.asarray(G.F13_CMFS),
                                 np.asarray(G.F13_ILLUM), float(G.F13_INTERVAL),
                                 apply_adaptation=False), G.F13_SRGB_NOADAPT)
    pe = C.PhotometryEngine(np.asarray(G.F14_VP), np.asarray(G.F14_VS),
                            float(G.F14_KM_P), float(G.F14_KM_S))
    iv = float(G.F13_INTERVAL)
    spd = np.asarray(G.F13_SPD)
    check("func_14_flux_photopic", close(pe.calculate_flux(spd, "photopic", 1.0, iv),
                                         float(G.F14_FLUX_PHOTOPIC)))
    check("func_14_flux_scotopic", close(pe.calculate_flux(spd, "scotopic", 1.0, iv),
                                         float(G.F14_FLUX_SCOTOPIC)))
    check("func_14_flux_mesopic", close(pe.calculate_flux(spd, "mesopic", 0.5, iv),
                                        float(G.F14_FLUX_MESOPIC_05)))
    check("func_14_sp_ratio", close(pe.calculate_sp_ratio(spd, iv), float(G.F14_SP_RATIO)))
    got = C.delta_E_CIE1976(np.asarray([G.LAB1_IN[0]]), G.LAB2_IN)
    check("func_15_broadcast_len", len(got) == len(G.LAB2_IN), f"len={len(got)}")
    # DE2000 golden: present in golden.rs but asserted by NO Rust test.
    check_vec("de2000_golden_bonus", C.delta_E_CIE2000(G.LAB1_IN, G.LAB2_IN), G.DE2000)


# ---------------------------------------------------------------- func_0x unit tests
def test_unit_conversions():
    print("--- func_01..08 unit tests ---")
    check_rows("f01_black", C.XYZ_to_xyY(np.zeros((1, 3))), np.zeros((1, 3)))
    xyz = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.95047, 1.0, 1.08883]])
    check_rows("f01_roundtrip", C.xyY_to_XYZ(C.XYZ_to_xyY(xyz)), xyz)
    lab = np.array([[50.0, 10.0, 5.0], [0.0, 0.0, 0.0], [100.0, -20.0, 30.0]])
    check_rows("f02_roundtrip", C.LCHab_to_Lab(C.Lab_to_LCHab(lab)), lab)
    h = C.Lab_to_LCHab(np.array([[50.0, 0.0, -1.0]]))[0, 2]
    check("f02_hue_wrap_270", abs(h - 270.0) < 1e-12, f"h={h}")
    check_rows("f03_black_luv", C.XYZ_to_Luv(np.zeros((1, 3))), np.zeros((1, 3)))
    check_rows("f03_roundtrip", C.Luv_to_XYZ(C.XYZ_to_Luv(xyz)), xyz)
    xyz4 = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.95047, 1.0, 1.08883],
                     [-0.1, 0.5, 0.2]])
    check_rows("f04_roundtrip_neg", C.Oklab_to_XYZ(C.XYZ_to_Oklab(xyz4)), xyz4)
    rgb5 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5],
                     [0.2, 0.6, 0.9], [0.9, 0.1, 0.3]])
    check_rows("f05_roundtrip_gamut", C.Oklab_to_sRGB(C.sRGB_to_Oklab(rgb5)), rgb5)
    for gval in (0.4, 0.7):
        ab = C.sRGB_to_Oklab(np.full((1, 3), gval))[0, 1:]
        check(f"f05_grey_{gval}", bool(np.all(np.abs(ab) < 1e-7)), f"a,b={ab}")
    c1 = C.sRGB_to_Oklab(np.array([[1.5, -0.2, 0.8]]))
    c2 = C.sRGB_to_Oklab(np.array([[1.0, 0.0, 0.8]]))
    check_rows("f05_clip", c1, c2)
    un, vn = C.white_point_uv1960(np.asarray(C.REF_WHITE_D65))
    check("f06_white_approx", abs(un - 0.1978) < 1e-4 and abs(vn - 0.3122) < 1e-4)
    check_rows("f06_roundtrip", C.UVW_to_XYZ(C.XYZ_to_UVW(xyz)), xyz)
    check_rows("f07_ucs_roundtrip", C.UCS_to_XYZ(C.XYZ_to_UCS(xyz)), xyz)
    uv = C.XYZ_to_UCS_uv(np.asarray([C.REF_WHITE_D65]))[0]
    check("f07_uv_consistency", abs(uv[0] - un) < 1e-12 and abs(uv[1] - vn) < 1e-12,
          f"uv={uv} vs ({un:.6f},{vn:.6f})")
    xy = C.Luv_uv_to_xy(np.array([[0.1978, 0.4683]]))[0]
    check("f07_uv1976_known", abs(xy[0] - 0.3127) < 1e-4 and abs(xy[1] - 0.3290) < 1e-4,
          f"xy={xy}")
    d65 = np.asarray(C.REF_WHITE_D65)
    check_rows("f08_identity", C.chromatic_adaptation_VonKries(
        np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]), d65, d65),
        np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]))
    neg = C.chromatic_adaptation_VonKries(np.array([[-0.1, 0.5, 0.5]]), d65,
                                          np.asarray(C.REF_WHITE_D50))[0]
    check("f08_neg_clip", neg[0] == 0.0 and neg[1] > 0.0 and neg[2] > 0.0, f"{neg}")
    adapted = C.chromatic_adaptation_VonKries(d65.reshape(1, 3), d65,
                                              np.asarray(C.REF_WHITE_D50))[0]
    check("f08_d65_to_d50", bool(np.all(np.abs(adapted - C.REF_WHITE_D50) < 1e-6)),
          f"{adapted}")
    # calc_matrix round trip via numpy row-vector convention (v @ M).
    m = np.asarray(C.calc_transform_matrix(d65, np.asarray(C.REF_WHITE_D50)))
    white = d65 @ m
    check("f08_matrix_rowvec", bool(np.all(np.abs(white - C.REF_WHITE_D50) < 1e-12)),
          f"{white}")


def test_unit_metrics():
    print("--- func_09..12 + metrics unit tests ---")
    check_vec("f09_single", C.delta_E_CIE1976(np.array([[50., 0., 0.]]),
                                              np.array([[55., 0., 0.]])), [5.0])
    check_vec("f09_broadcast", C.delta_E_CIE1976(np.array([[50., 0., 0.]]),
                                                 np.array([[55., 0., 0.], [60., 0., 0.]])),
              [5.0, 10.0])
    a = np.array([[50., 10., 20.], [60., 0., 0.]])
    b = np.array([[55., 11., 22.], [65., 0., 0.]])
    hand0 = float(np.sqrt(25 + 1 + 4))
    check_vec("f09_equal_hand", C.delta_E_CIE1976(a, b), [hand0, 5.0])
    check_vec("f10_graphic_light", C.delta_E_CIE1994(np.array([[50., 0., 0.]]),
                                                     np.array([[55., 0., 0.]])), [5.0])
    check_vec("f10_textile_light", C.delta_E_CIE1994(np.array([[50., 0., 0.]]),
                                                     np.array([[55., 0., 0.]]),
                                                     textiles=True), [2.5])
    check_vec("f10_chroma", C.delta_E_CIE1994(np.array([[50., 10., 0.]]),
                                              np.array([[50., 15., 0.]])),
              [5.0 / (1.0 + 0.045 * 10.0)])
    ab = float(C.delta_E_CIE1994(np.array([[50., 10., 0.]]), np.array([[50., 15., 0.]]))[0])
    ba = float(C.delta_E_CIE1994(np.array([[50., 15., 0.]]), np.array([[50., 10., 0.]]))[0])
    check("f10_asymmetry", abs(ab - ba) > 1e-6, f"{ab:.4f} vs {ba:.4f}")
    sl = (0.040975 * 50.0) / (1.0 + 0.01765 * 50.0)
    check_vec("f11_lightness", C.delta_E_CMC(np.array([[50., 0., 0.]]),
                                             np.array([[55., 0., 0.]]), 2.0, 1.0),
              [5.0 / (2.0 * sl)])
    z1 = float(C.delta_E_CMC(np.array([[50., np.cos(np.radians(164.)), np.sin(np.radians(164.))]]),
                             np.array([[50., np.cos(np.radians(164.)), np.sin(np.radians(164.))]]),
                             2.0, 1.0)[0])
    z2 = float(C.delta_E_CMC(np.array([[50., np.cos(np.radians(345.)), np.sin(np.radians(345.))]]),
                             np.array([[50., np.cos(np.radians(345.)), np.sin(np.radians(345.))]]),
                             2.0, 1.0)[0])
    check("f11_branch_zero", abs(z1) < 1e-12 and abs(z2) < 1e-12, f"{z1:.2e},{z2:.2e}")
    lo = float(C.delta_E_CMC(np.array([[15., 10., 5.]]), np.array([[15.5, 10., 5.]]), 2.0, 1.0)[0])
    hi = float(C.delta_E_CMC(np.array([[16., 10., 5.]]), np.array([[15.5, 10., 5.]]), 2.0, 1.0)[0])
    check("f11_sl_branch", abs(lo - hi) > 1e-6, f"{lo:.4f} vs {hi:.4f}")
    l1 = 105.509 * np.log1p(0.0158 * 50.0)
    l2 = 105.509 * np.log1p(0.0158 * 55.0)
    check_vec("f12_lightness", C.delta_E_DIN99(np.array([[50., 0., 0.]]),
                                               np.array([[55., 0., 0.]]), ),
              [abs(l1 - l2)])
    check_vec("f12_achromatic_zero", C.delta_E_DIN99(np.array([[50., 0., 0.]]),
                                                     np.array([[50., 0., 0.]])), [0.0])
    g = float(C.delta_E_DIN99(np.array([[50., 0., 0.]]), np.array([[55., 0., 0.]]))[0])
    t = float(C.delta_E_DIN99(np.array([[50., 0., 0.]]), np.array([[55., 0., 0.]]),
                              textiles=True)[0])
    check("f12_textile_ratio", abs(t / g - 2.0) < 1e-12, f"{t / g}")
    d_chroma = float(C.delta_E_DIN99(np.array([[50., 1., 0.]]), np.array([[50., 0., 0.]]))[0])
    check("f12_rotation", abs(d_chroma - 1.0) > 1e-3, f"d={d_chroma:.4f} (Lab Euclid = 1)")
    # metrics.rs broadcast semantics through a ΔE entry point.
    check_vec("m_equal", C.delta_E_CIE1976(np.full((3, 3), 1.0), np.full((3, 3), 2.0)),
              [np.sqrt(3.0)] * 3)
    check_vec("m_one_vs_n", C.delta_E_CIE1976(np.full((1, 3), 1.0),
                                              np.array([[2., 1., 1.], [3., 1., 1.]])), [1.0, 2.0])
    check_vec("m_n_vs_one", C.delta_E_CIE1976(np.array([[1., 1., 1.], [2., 1., 1.]]),
                                              np.full((1, 3), 3.0)), [np.sqrt(12.0), 3.0])
    try:
        C.delta_E_CIE1976(np.zeros((2, 3)), np.zeros((3, 3)))
        check("m_incompatible_raises", False)
    except ValueError as e:
        check("m_incompatible_raises", "not broadcastable" in str(e), f"({e})")


def test_unit_spectral_photometry():
    print("--- func_13/14 unit tests ---")
    n = 81
    rgb = C.spectral_to_sRGB(np.ones(n), np.tile([0.5, 1.0, 0.5], (n, 1)),
                             np.ones(n), 5.0, apply_adaptation=True)
    check("f13_neutral_range", bool(np.all((rgb >= 0.0) & (rgb <= 1.0))), f"{rgb}")
    pe = C.PhotometryEngine(np.array([1.0, 0.5, 0.0]), np.array([0.0, 0.2, 0.8]))
    fp = pe.calculate_flux(np.array([10., 20., 30.]), "photopic", 0.0, 1.0)
    check("f14_photopic_hand", abs(fp - (10.0 * 683.002 + 20.0 * 683.002 * 0.5)) < 1e-9,
          f"{fp}")
    fs = pe.calculate_flux(np.array([10., 20., 30.]), "scotopic", 0.0, 1.0)
    check("f14_scotopic_hand", abs(fs - (20.0 * 1700.05 * 0.2 + 30.0 * 1700.05 * 0.8)) < 1e-9,
          f"{fs}")
    pe2 = C.PhotometryEngine(np.array([1.0, 0.5]), np.array([0.0, 0.2]))
    fm = pe2.calculate_flux(np.array([10., 20.]), "mesopic", 0.3, 1.0)
    wp, ws = 0.3 * 683.002, 0.7 * 1700.05
    check("f14_mesopic_hand", abs(fm - (10.0 * wp + 20.0 * (0.5 * wp + 0.2 * ws))) < 1e-9,
          f"{fm}")
    pe3 = C.PhotometryEngine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    r = pe3.calculate_sp_ratio(np.array([100., 200.]), 1.0)
    check("f14_sp_hand", abs(r - (200.0 * 1700.05) / (100.0 * 683.002)) < 1e-9, f"{r}")


if __name__ == "__main__":
    test_parity_goldens()
    test_unit_conversions()
    test_unit_metrics()
    test_unit_spectral_photometry()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
