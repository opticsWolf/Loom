//! Golden-parity tests: load NumPy-generated references and assert the Rust
//! kernels reproduce them. Parameter values mirror `tools/gen_golden.py`.

use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use num_complex::Complex64;

use navette::materials::{
    cauchy, cody_lorentz, drude, ema, forouhi_bloomer, lorentz, sellmeier, tauc_lorentz, ubf,
};

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../validation/parity/materials/goldens/");

fn load(case: &str, part: &str) -> Array1<f64> {
    let path = format!("{GOLDEN}{case}__{part}.npy");
    read_npy(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"))
}

/// Assert two complex arrays agree within absolute+relative tolerance.
fn assert_close(case: &str, got: &Array1<Complex64>, re: &Array1<f64>, im: &Array1<f64>, rtol: f64, atol: f64) {
    assert_eq!(got.len(), re.len(), "{case}: length mismatch");
    let mut worst = 0.0_f64;
    for i in 0..got.len() {
        let dr = (got[i].re - re[i]).abs();
        let di = (got[i].im - im[i]).abs();
        let tr = atol + rtol * re[i].abs();
        let ti = atol + rtol * im[i].abs();
        worst = worst.max((dr / tr).max(di / ti));
        assert!(
            dr <= tr && di <= ti,
            "{case}[{i}] λ-index: got ({}, {}) vs ref ({}, {})  |Δre|={dr:.3e} (tol {tr:.3e})  |Δim|={di:.3e} (tol {ti:.3e})",
            got[i].re, got[i].im, re[i], im[i]
        );
    }
    eprintln!("  {case:28} OK  (worst tol-ratio {worst:.2})");
}

fn osc2(rows: &[[f64; 3]]) -> Array2<f64> {
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), 3), flat).unwrap()
}

#[test]
fn cauchy_basic() {
    let wl = load("cauchy_basic", "wl");
    let got = cauchy::cauchy_nk(wl.view(), 1.5, 0.01, 0.0001);
    assert_close("cauchy_basic", &got, &load("cauchy_basic", "re"), &load("cauchy_basic", "im"), 1e-10, 1e-12);
}

#[test]
fn cauchy_single() {
    let wl = load("cauchy_single", "wl");
    let got = cauchy::cauchy_nk(wl.view(), 1.5, 0.01, 0.0001);
    assert_close("cauchy_single", &got, &load("cauchy_single", "re"), &load("cauchy_single", "im"), 1e-10, 1e-12);
}

#[test]
fn cauchy_urbach() {
    let wl = load("cauchy_urbach", "wl");
    let got = cauchy::cauchy_urbach_nk(wl.view(), 2.5, 0.02, 0.0005, 1e4, 0.05, 400.0);
    // Urbach k spans many orders of magnitude; relative tol on tiny k is fine.
    assert_close("cauchy_urbach", &got, &load("cauchy_urbach", "re"), &load("cauchy_urbach", "im"), 1e-9, 1e-14);
}

#[test]
fn sellmeier_bk7() {
    let wl = load("sellmeier_bk7", "wl");
    let got = sellmeier::sellmeier_nk(wl.view(), 1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653);
    assert_close("sellmeier_bk7", &got, &load("sellmeier_bk7", "re"), &load("sellmeier_bk7", "im"), 1e-10, 1e-12);
}

#[test]
fn sellmeier_2term() {
    let wl = load("sellmeier_2term", "wl");
    let got = sellmeier::sellmeier_nk(wl.view(), 1.0, 0.01, 0.3, 0.05, 0.0, 0.0);
    assert_close("sellmeier_2term", &got, &load("sellmeier_2term", "re"), &load("sellmeier_2term", "im"), 1e-10, 1e-12);
}

#[test]
fn sellmeier_urbach() {
    let wl = load("sellmeier_urbach", "wl");
    let got = sellmeier::sellmeier_urbach_nk(wl.view(), 1.4313, 0.01, 0.65, 0.025, 0.0, 0.0, 1e5, 0.06, 380.0);
    assert_close("sellmeier_urbach", &got, &load("sellmeier_urbach", "re"), &load("sellmeier_urbach", "im"), 1e-9, 1e-14);
}

#[test]
fn lorentz_2osc() {
    let wl = load("lorentz_2osc", "wl");
    let osc = osc2(&[[3.0, 0.2, 0.5], [4.5, 0.1, 0.7]]);
    let got = lorentz::lorentz_nk(wl.view(), osc.view(), 1.0);
    assert_close("lorentz_2osc", &got, &load("lorentz_2osc", "re"), &load("lorentz_2osc", "im"), 1e-10, 1e-12);
}

#[test]
fn drude_basic() {
    let wl = load("drude_basic", "wl");
    let got = drude::drude_nk(wl.view(), 2.5, 0.3, 3.5);
    assert_close("drude_basic", &got, &load("drude_basic", "re"), &load("drude_basic", "im"), 1e-10, 1e-12);
}

#[test]
fn drude_lorentz() {
    let wl = load("drude_lorentz", "wl");
    let osc = osc2(&[[2.0, 0.5, 1.0], [3.5, 0.8, 0.4]]);
    let got = drude::drude_lorentz_nk(wl.view(), 9.0, 0.05, 1.0, osc.view());
    assert_close("drude_lorentz", &got, &load("drude_lorentz", "re"), &load("drude_lorentz", "im"), 1e-10, 1e-12);
}

// --- EMA: references store epsilon (Re/Im of eps_eff), not nk ---

#[test]
fn fb_single() {
    let wl = load("fb_single", "wl");
    let ib = osc4(&[[3.0, 0.1, 6.0, 12.0]]);
    let got = forouhi_bloomer::fb_interband_nk(wl.view(), 1.5, ib.view());
    assert_close("fb_single", &got, &load("fb_single", "re"), &load("fb_single", "im"), 1e-10, 1e-12);
}

#[test]
fn fb_multi() {
    let wl = load("fb_multi", "wl");
    let ib = osc4(&[[3.0, 0.1, 6.0, 12.0], [4.5, 0.05, 9.0, 22.0]]);
    let got = forouhi_bloomer::fb_interband_nk(wl.view(), 1.2, ib.view());
    assert_close("fb_multi", &got, &load("fb_multi", "re"), &load("fb_multi", "im"), 1e-10, 1e-12);
}

#[test]
fn fb_edge() {
    // 4C < B²: exercises the unphysical Q = 1e-6 fallback (large magnitudes).
    let wl = load("fb_edge", "wl");
    let ib = osc4(&[[2.0, 0.1, 10.0, 20.0]]);
    let got = forouhi_bloomer::fb_interband_nk(wl.view(), 1.0, ib.view());
    assert_close("fb_edge", &got, &load("fb_edge", "re"), &load("fb_edge", "im"), 1e-10, 1e-12);
}

#[test]
fn fb_metal() {
    let wl = load("fb_metal", "wl");
    let fe = Array1::from(vec![5.0, 0.5, 0.3]);
    let ib = osc4(&[[2.0, 0.2, 4.0, 5.0]]);
    let got = forouhi_bloomer::fb_metal_nk(wl.view(), 1.0, fe.view(), ib.view());
    assert_close("fb_metal", &got, &load("fb_metal", "re"), &load("fb_metal", "im"), 1e-10, 1e-12);
}

fn osc3(rows: &[[f64; 3]]) -> Array2<f64> {
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), 3), flat).unwrap()
}

#[test]
fn tauc_single() {
    let wl = load("tauc_single", "wl");
    let osc = osc3(&[[100.0, 4.0, 2.0]]);
    let got = tauc_lorentz::tauc_lorentz_nk(wl.view(), 1.2, osc.view(), 1.0).unwrap();
    assert_close("tauc_single", &got, &load("tauc_single", "re"), &load("tauc_single", "im"), 1e-9, 1e-11);
}

#[test]
fn tauc_multi() {
    let wl = load("tauc_multi", "wl");
    let osc = osc3(&[[100.0, 4.0, 2.0], [50.0, 6.5, 1.5]]);
    let got = tauc_lorentz::tauc_lorentz_nk(wl.view(), 1.2, osc.view(), 1.5).unwrap();
    assert_close("tauc_multi", &got, &load("tauc_multi", "re"), &load("tauc_multi", "im"), 1e-9, 1e-11);
}

fn osc6(rows: &[[f64; 6]]) -> Array2<f64> {
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), 6), flat).unwrap()
}

#[test]
fn ubf_single() {
    let wl = load("ubf_single", "wl");
    let osc = osc6(&[[1.5, 3.0, 1.0 / 0.2, 10.0, 1.0, 2.0]]);
    let got = ubf::ubf_nk(wl.view(), osc.view(), 1.0).unwrap();
    assert_close("ubf_single", &got, &load("ubf_single", "re"), &load("ubf_single", "im"), 1e-9, 1e-11);
}

#[test]
fn ubf_multi() {
    let wl = load("ubf_multi", "wl");
    let osc = osc6(&[
        [1.5, 3.0, 1.0 / 0.2, 10.0, 1.0, 2.0],
        [2.2, 4.5, 1.0 / 0.15, 6.0, 1.5, 0.5],
    ]);
    let got = ubf::ubf_nk(wl.view(), osc.view(), 1.0).unwrap();
    assert_close("ubf_multi", &got, &load("ubf_multi", "re"), &load("ubf_multi", "im"), 1e-9, 1e-11);
}

fn osc4(rows: &[[f64; 4]]) -> Array2<f64> {
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), 4), flat).unwrap()
}

#[test]
fn cody_single() {
    let wl = load("cody_single", "wl");
    let osc = osc4(&[[3.40, 60.0, 2.4, 1.0]]);
    let got = cody_lorentz::cody_lorentz_nk(wl.view(), 1.64, 1.80, 0.15, osc.view(), 1.0).unwrap();
    // FFT-KK path: relaxed tolerance per project decision (FFT-library differences).
    assert_close("cody_single", &got, &load("cody_single", "re"), &load("cody_single", "im"), 1e-9, 1e-11);
}

#[test]
fn cody_multi() {
    let wl = load("cody_multi", "wl");
    let osc = osc4(&[[3.40, 60.0, 2.4, 1.0], [4.70, 40.0, 1.8, 0.5]]);
    let got = cody_lorentz::cody_lorentz_nk(wl.view(), 1.64, 1.80, 0.15, osc.view(), 1.0).unwrap();
    assert_close("cody_multi", &got, &load("cody_multi", "re"), &load("cody_multi", "im"), 1e-9, 1e-11);
}

fn ema_inputs() -> (Array1<Complex64>, Array1<Complex64>) {
    let n_i: Vec<Complex64> = (0..120)
        .map(|k| {
            let t = k as f64 / 119.0;
            Complex64::new(2.0 + 0.4 * t, 0.05 + 0.15 * t)
        })
        .collect();
    let n_h: Vec<Complex64> = (0..120)
        .map(|k| {
            let t = k as f64 / 119.0;
            Complex64::new(1.4 + 0.1 * t, 0.0 + 0.02 * t)
        })
        .collect();
    (Array1::from(n_i), Array1::from(n_h))
}

#[test]
fn ema_looyenga() {
    let (n_i, n_h) = ema_inputs();
    let got = ema::looyenga(n_i.view(), n_h.view(), 0.3);
    assert_close("ema_looyenga", &got, &load("ema_looyenga", "re"), &load("ema_looyenga", "im"), 1e-9, 1e-12);
}

#[test]
fn ema_maxwell_garnett() {
    let (n_i, n_h) = ema_inputs();
    let got = ema::maxwell_garnett(n_i.view(), n_h.view(), 0.3);
    assert_close("ema_maxwell_garnett", &got, &load("ema_maxwell_garnett", "re"), &load("ema_maxwell_garnett", "im"), 1e-10, 1e-12);
}

#[test]
fn ema_bruggeman() {
    let (n_i, n_h) = ema_inputs();
    let got = ema::bruggeman(n_i.view(), n_h.view(), 0.3, 100, 1e-9);
    // Iterative solver: looser tol than the analytic mixers.
    assert_close("ema_bruggeman", &got, &load("ema_bruggeman", "re"), &load("ema_bruggeman", "im"), 1e-6, 1e-9);
}
