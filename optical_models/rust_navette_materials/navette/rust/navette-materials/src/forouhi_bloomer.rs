//! Forouhi–Bloomer dispersion (2019 "new formulation" + 2021 metal).
//!
//! Each interband/free-electron transition contributes a causal rational
//! function (faithful port of `compute_single_fb2019_term`):
//!
//!   disc = 4C − B²;   Q = ½√disc  (or 1e-6 in the unphysical disc ≤ 1e-12 case)
//!   B0 = (A/Q)·(−B²/2 + Eg·B − Eg² + C)
//!   C0 = (A/Q)·((Eg²+C)·B/2 − 2·Eg·C)
//!   D(E) = E² − B·E + C   (clamped to 1e-15 when |D| < 1e-15)
//!   k(E) = A·(E−Eg)² / D(E)   for E ≥ Eg, else 0
//!   n(E) = (B0·E + C0) / D(E)
//!
//! Models sum these over terms on top of `n_inf`. The metal driver prepends a
//! free-electron term evaluated at Eg = 0 when its amplitude is positive.
//! Oscillator/term layout is (N_terms, 4): columns (Eg, A, B, C).

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64;

use crate::common::map_nk;
use crate::units::energy_ev;

/// Single FB-2019 rational term contribution (n + ik) at one energy.
#[inline]
fn fb_term(e: f64, eg: f64, a: f64, b: f64, c: f64) -> Complex64 {
    let disc = 4.0 * c - b * b;
    let q = if disc <= 1e-12 { 1e-6 } else { 0.5 * disc.sqrt() };
    let b0 = (a / q) * (-(b * b / 2.0) + eg * b - eg * eg + c);
    let c0 = (a / q) * ((eg * eg + c) * (b / 2.0) - 2.0 * eg * c);

    let mut denom = e * e - b * e + c;
    if denom.abs() < 1e-15 {
        denom = 1e-15;
    }
    let k = if e >= eg {
        a * (e - eg) * (e - eg) / denom
    } else {
        0.0
    };
    let n = (b0 * e + c0) / denom;
    Complex64::new(n, k)
}

/// Interband-only model: n̂ = n_inf + Σ terms.  `ib` is (N, 4): (Eg, A, B, C).
/// Used by the single/multi interband classes and by the 2021 metal class
/// (which packs its free-electron term as an Eg = 0 row of `ib`).
pub fn fb_interband_nk(
    wavelength_nm: ArrayView1<f64>,
    n_inf: f64,
    ib: ArrayView2<f64>,
) -> Array1<Complex64> {
    map_nk(wavelength_nm, |w| {
        let e = energy_ev(w);
        let mut n = n_inf;
        let mut k = 0.0;
        for row in ib.rows() {
            let t = fb_term(e, row[0], row[1], row[2], row[3]);
            n += t.re;
            k += t.im;
        }
        Complex64::new(n, k)
    })
}

/// Metal model: free-electron term (Eg = 0, added when its amplitude > 0) plus
/// interband terms, on top of n_inf. `fe` is (A_fe, B_fe, C_fe); `ib` is (N, 4).
pub fn fb_metal_nk(
    wavelength_nm: ArrayView1<f64>,
    n_inf: f64,
    fe: ArrayView1<f64>,
    ib: ArrayView2<f64>,
) -> Array1<Complex64> {
    let (a_fe, b_fe, c_fe) = (fe[0], fe[1], fe[2]);
    map_nk(wavelength_nm, |w| {
        let e = energy_ev(w);
        let mut n = n_inf;
        let mut k = 0.0;
        if a_fe > 0.0 {
            let t = fb_term(e, 0.0, a_fe, b_fe, c_fe);
            n += t.re;
            k += t.im;
        }
        for row in ib.rows() {
            let t = fb_term(e, row[0], row[1], row[2], row[3]);
            n += t.re;
            k += t.im;
        }
        Complex64::new(n, k)
    })
}
