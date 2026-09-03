//! Parity tests: assert Rust outputs against exact golden vectors produced by
//! the Python reference engine (refgen/gen_golden.py).

#[path = "golden.rs"]
mod golden;

use navette_color::common::REF_WHITE_D65;
use navette_color::prelude::*;

const EPS: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS + 1e-9 * b.abs()
}

fn assert_rows(got: &[[f64; 3]], want: &[[f64; 3]], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    let mut max = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for c in 0..3 {
            assert!(close(g[c], w[c]), "{tag}[{i}][{c}]: got {} want {}", g[c], w[c]);
            max = max.max((g[c] - w[c]).abs());
        }
    }
    println!("CORRECTNESS {tag} PASS | diff_max={max:.3e}");
}

fn assert_rows2(got: &[[f64; 2]], want: &[[f64; 2]], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    let mut max = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for c in 0..2 {
            assert!(close(g[c], w[c]), "{tag}[{i}][{c}]: got {} want {}", g[c], w[c]);
            max = max.max((g[c] - w[c]).abs());
        }
    }
    println!("CORRECTNESS {tag} PASS | diff_max={max:.3e}");
}

fn assert_vec(got: &[f64], want: &[f64], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    let mut max = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(close(*g, *w), "{tag}[{i}]: got {g} want {w}");
        max = max.max((g - w).abs());
    }
    println!("CORRECTNESS {tag} PASS | diff_max={max:.3e}");
}

fn run3(f: impl Fn(&[[f64; 3]], &mut [[f64; 3]]), input: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0; 3]; input.len()];
    f(input, &mut out);
    out
}

#[test]
fn func_01_xyy() {
    assert_rows(&run3(xyz_to_xyy, &golden::XYZ_IN), &golden::F01_XYZ_TO_XYY, "func_01_xyz_to_xyy");
    let xyy = run3(xyz_to_xyy, &golden::XYZ_IN);
    assert_rows(&run3(xyy_to_xyz, &xyy), &golden::F01_XYY_TO_XYZ, "func_01_xyy_to_xyz");
}

#[test]
fn func_02_lch() {
    assert_rows(&run3(lab_to_lch, &golden::LAB1_IN), &golden::F02_LAB_TO_LCH, "func_02_lab_to_lch");
    let lch = run3(lab_to_lch, &golden::LAB1_IN);
    assert_rows(&run3(lch_to_lab, &lch), &golden::F02_LCH_TO_LAB, "func_02_lch_to_lab");
}

#[test]
fn func_03_luv() {
    let f = |i: &[[f64; 3]], o: &mut [[f64; 3]]| xyz_to_luv(i, &REF_WHITE_D65, o);
    let g = |i: &[[f64; 3]], o: &mut [[f64; 3]]| luv_to_xyz(i, &REF_WHITE_D65, o);
    assert_rows(&run3(f, &golden::XYZ_IN), &golden::F03_XYZ_TO_LUV, "func_03_xyz_to_luv");
    let luv = run3(f, &golden::XYZ_IN);
    assert_rows(&run3(g, &luv), &golden::F03_LUV_TO_XYZ, "func_03_luv_to_xyz");
}

#[test]
fn func_04_oklab_xyz() {
    assert_rows(&run3(xyz_to_oklab, &golden::XYZ_IN), &golden::F04_XYZ_TO_OKLAB, "func_04_xyz_to_oklab");
    let lab = run3(xyz_to_oklab, &golden::XYZ_IN);
    assert_rows(&run3(oklab_to_xyz, &lab), &golden::F04_OKLAB_TO_XYZ, "func_04_oklab_to_xyz");
}

#[test]
fn func_05_oklab_srgb() {
    assert_rows(&run3(srgb_to_oklab, &golden::SRGB_IN), &golden::F05_SRGB_TO_OKLAB, "func_05_srgb_to_oklab");
    let lab = run3(srgb_to_oklab, &golden::SRGB_IN);
    assert_rows(&run3(oklab_to_srgb, &lab), &golden::F05_OKLAB_TO_SRGB, "func_05_oklab_to_srgb");
}

#[test]
fn func_06_uvw() {
    let (un, vn) = white_point_uv1960(&REF_WHITE_D65);
    assert!(close(un, golden::F06_UN_D65) && close(vn, golden::F06_VN_D65), "white point");
    let f = |i: &[[f64; 3]], o: &mut [[f64; 3]]| xyz_to_uvw(i, un, vn, o);
    let g = |i: &[[f64; 3]], o: &mut [[f64; 3]]| uvw_to_xyz(i, un, vn, o);
    assert_rows(&run3(f, &golden::XYZ_IN), &golden::F06_XYZ_TO_UVW, "func_06_xyz_to_uvw");
    let uvw = run3(f, &golden::XYZ_IN);
    assert_rows(&run3(g, &uvw), &golden::F06_UVW_TO_XYZ, "func_06_uvw_to_xyz");
}

#[test]
fn func_07_ucs() {
    assert_rows(&run3(xyz_to_ucs, &golden::XYZ_IN), &golden::F07_XYZ_TO_UCS, "func_07_xyz_to_ucs");
    let ucs = run3(xyz_to_ucs, &golden::XYZ_IN);
    assert_rows(&run3(ucs_to_xyz, &ucs), &golden::F07_UCS_TO_XYZ, "func_07_ucs_to_xyz");

    let mut uv = vec![[0.0; 2]; golden::XYZ_IN.len()];
    xyz_to_ucs_uv(&golden::XYZ_IN, &mut uv);
    assert_rows2(&uv, &golden::F07_XYZ_TO_UCS_UV, "func_07_xyz_to_ucs_uv");

    let mut xy = vec![[0.0; 2]; golden::F07_UVP_IN.len()];
    uv1976_to_xy(&golden::F07_UVP_IN, &mut xy);
    assert_rows2(&xy, &golden::F07_UV1976_TO_XY, "func_07_uv1976_to_xy");
    uv1960_to_xy(&golden::F07_UVP_IN, &mut xy);
    assert_rows2(&xy, &golden::F07_UV1960_TO_XY, "func_07_uv1960_to_xy");
}

#[test]
fn func_08_bradford() {
    let mut out = vec![[0.0; 3]; golden::XYZ_IN.len()];
    adapt(&golden::XYZ_IN, &navette_color::common::REF_WHITE_D65, &navette_color::common::REF_WHITE_D50, true, &mut out);
    assert_rows(&out, &golden::F08_ADAPT_D65_TO_D50, "func_08_adapt_d65_to_d50");
    let m = calc_transform_matrix(&navette_color::common::REF_WHITE_D65, &navette_color::common::REF_WHITE_D50);
    assert_rows(&m, &golden::F08_BRADFORD_MATRIX, "func_08_bradford_matrix");
}

#[test]
fn func_09_de76() {
    assert_vec(&delta_e_76(&golden::LAB1_IN, &golden::LAB2_IN), &golden::F09_DE76, "func_09_de76");
}

#[test]
fn func_10_de94() {
    assert_vec(&delta_e_94(&golden::LAB1_IN, &golden::LAB2_IN, De94Params::GRAPHIC), &golden::F10_DE94_GRAPHIC, "func_10_de94_graphic");
    assert_vec(&delta_e_94(&golden::LAB1_IN, &golden::LAB2_IN, De94Params::TEXTILES), &golden::F10_DE94_TEXTILES, "func_10_de94_textiles");
}

#[test]
fn func_11_cmc() {
    assert_vec(&delta_e_cmc(&golden::LAB1_IN, &golden::LAB2_IN, 2.0, 1.0), &golden::F11_DE_CMC_2_1, "func_11_cmc_2_1");
    assert_vec(&delta_e_cmc(&golden::LAB1_IN, &golden::LAB2_IN, 1.0, 1.0), &golden::F11_DE_CMC_1_1, "func_11_cmc_1_1");
}

#[test]
fn func_12_din99() {
    assert_vec(&delta_e_din99(&golden::LAB1_IN, &golden::LAB2_IN, 1.0, 1.0), &golden::F12_DE_DIN99, "func_12_din99");
    assert_vec(&delta_e_din99(&golden::LAB1_IN, &golden::LAB2_IN, 2.0, 0.5), &golden::F12_DE_DIN99_TEX, "func_12_din99_textiles");
}

#[test]
fn func_13_spectral() {
    let a = spectral_to_srgb(&golden::F13_SPD, &golden::F13_CMFS, &golden::F13_ILLUM, golden::F13_INTERVAL, true);
    assert_vec(&a, &golden::F13_SRGB_ADAPT, "func_13_spectral_adapt");
    let n = spectral_to_srgb(&golden::F13_SPD, &golden::F13_CMFS, &golden::F13_ILLUM, golden::F13_INTERVAL, false);
    assert_vec(&n, &golden::F13_SRGB_NOADAPT, "func_13_spectral_noadapt");
}

#[test]
fn func_14_photometry() {
    let pe = PhotometryEngine::with_constants(
        golden::F14_VP.to_vec(),
        golden::F14_VS.to_vec(),
        golden::F14_KM_P,
        golden::F14_KM_S,
    );
    let iv = golden::F13_INTERVAL;
    assert!(close(pe.calculate_flux(&golden::F13_SPD, Vision::Photopic, 1.0, iv), golden::F14_FLUX_PHOTOPIC));
    assert!(close(pe.calculate_flux(&golden::F13_SPD, Vision::Scotopic, 1.0, iv), golden::F14_FLUX_SCOTOPIC));
    assert!(close(pe.calculate_flux(&golden::F13_SPD, Vision::Mesopic, 0.5, iv), golden::F14_FLUX_MESOPIC_05));
    assert!(close(pe.calculate_sp_ratio(&golden::F13_SPD, iv), golden::F14_SP_RATIO));
    println!("CORRECTNESS func_14_photometry PASS");
}

#[test]
fn func_15_broadcast() {
    // 1 reference vs N samples must broadcast the reference.
    let one = [golden::LAB1_IN[0]];
    let got = delta_e_76(&one, &golden::LAB2_IN);
    assert_eq!(got.len(), golden::LAB2_IN.len());
}
