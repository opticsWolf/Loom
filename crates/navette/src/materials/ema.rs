//! Effective-medium approximation (EMA) mixing kernels.
//!
//! Each function takes the inclusion and host **refractive indices** (n + ik)
//! and a scalar volume fraction `f`, and returns the effective **permittivity**
//! ε_eff (the caller applies √ to get n̂, matching the Python `_parallel_sqrt`
//! step in `EffectiveMaterial`). Faithful ports of `ema_models.py`.
//!
//! Analytic mixers are element-wise; Bruggeman solves an implicit equation per
//! point with Newton–Raphson (parallelised over points, like the Numba
//! `prange` version).

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use rayon::prelude::*;

#[inline]
fn eps_of(n: Complex64) -> Complex64 {
    n * n
}

/// Lichtenecker logarithmic mixing: ε_eff = exp(f·ln ε_i + (1−f)·ln ε_h).
pub fn lichtenecker(n_i: ArrayView1<Complex64>, n_h: ArrayView1<Complex64>, f: f64) -> Array1<Complex64> {
    let inv = 1.0 - f;
    Array1::from_iter(n_i.iter().zip(n_h.iter()).map(|(&ni, &nh)| {
        (f * eps_of(ni).ln() + inv * eps_of(nh).ln()).exp()
    }))
}

/// Looyenga (Landau–Lifshitz–Looyenga): (ε_eff)^⅓ = f·ε_i^⅓ + (1−f)·ε_h^⅓.
pub fn looyenga(n_i: ArrayView1<Complex64>, n_h: ArrayView1<Complex64>, f: f64) -> Array1<Complex64> {
    let inv = 1.0 - f;
    let p = 1.0 / 3.0;
    Array1::from_iter(n_i.iter().zip(n_h.iter()).map(|(&ni, &nh)| {
        let cbrt = f * eps_of(ni).powf(p) + inv * eps_of(nh).powf(p);
        cbrt.powf(3.0)
    }))
}

/// General power law (Birchak): ε_eff = (f·ε_i^α + (1−f)·ε_h^α)^(1/α).
/// For |α| < 1e-6 falls back to the Lichtenecker (logarithmic) limit.
pub fn general_power_law(
    n_i: ArrayView1<Complex64>,
    n_h: ArrayView1<Complex64>,
    f: f64,
    alpha: f64,
) -> Array1<Complex64> {
    if alpha.abs() < 1.0e-6 {
        return lichtenecker(n_i, n_h, f);
    }
    let inv = 1.0 - f;
    let inv_alpha = 1.0 / alpha;
    Array1::from_iter(n_i.iter().zip(n_h.iter()).map(|(&ni, &nh)| {
        let p = f * eps_of(ni).powf(alpha) + inv * eps_of(nh).powf(alpha);
        p.powf(inv_alpha)
    }))
}

/// Maxwell-Garnett (dilute spherical inclusions in host).
pub fn maxwell_garnett(n_i: ArrayView1<Complex64>, n_h: ArrayView1<Complex64>, f: f64) -> Array1<Complex64> {
    Array1::from_iter(n_i.iter().zip(n_h.iter()).map(|(&ni, &nh)| {
        let ei = eps_of(ni);
        let eh = eps_of(nh);
        let diff = ei - eh;
        let eh2 = 2.0 * eh;
        let num = ei + eh2 + 2.0 * f * diff;
        let den = ei + eh2 - f * diff;
        eh * (num / den)
    }))
}

/// Mori–Tanaka for ellipsoidal inclusions with depolarisation factor `l`.
pub fn mori_tanaka(
    n_i: ArrayView1<Complex64>,
    n_h: ArrayView1<Complex64>,
    f: f64,
    l: f64,
) -> Array1<Complex64> {
    let inv = 1.0 - f;
    Array1::from_iter(n_i.iter().zip(n_h.iter()).map(|(&ni, &nh)| {
        let ei = eps_of(ni);
        let eh = eps_of(nh);
        let diff = ei - eh;
        let num = f * diff * eh;
        let den = eh + inv * l * diff;
        eh + num / den
    }))
}

/// Wiener bounds → (lower, upper).
pub fn wiener_bounds(
    n_i: ArrayView1<Complex64>,
    n_h: ArrayView1<Complex64>,
    f: f64,
) -> (Array1<Complex64>, Array1<Complex64>) {
    let inv = 1.0 - f;
    let lower = Array1::from_iter(n_i.iter().zip(n_h.iter()).map(|(&ni, &nh)| {
        let ei = eps_of(ni);
        let eh = eps_of(nh);
        (ei * eh) / (f * eh + inv * ei)
    }));
    let upper = Array1::from_iter(n_i.iter().zip(n_h.iter()).map(|(&ni, &nh)| {
        f * eps_of(ni) + inv * eps_of(nh)
    }));
    (lower, upper)
}

/// 50:50 roughness interface = Looyenga at f = 0.5.
pub fn roughness_interface(n_bottom: ArrayView1<Complex64>, n_top: ArrayView1<Complex64>) -> Array1<Complex64> {
    looyenga(n_bottom, n_top, 0.5)
}

/// Bruggeman effective permittivity via per-point Newton–Raphson.
///
/// Implicit: f·(ε_i − ε)/(ε_i + 2ε) + (1−f)·(ε_h − ε)/(ε_h + 2ε) = 0.
/// Initialised at the arithmetic mean; parallel over points.
pub fn bruggeman(
    n_i: ArrayView1<Complex64>,
    n_h: ArrayView1<Complex64>,
    f: f64,
    max_iter: usize,
    tol: f64,
) -> Array1<Complex64> {
    let inv_f = 1.0 - f;
    let tiny = Complex64::new(1.0e-15, 0.0);
    let tol_sq = tol * tol;

    let solve = |ni: Complex64, nh: Complex64| -> Complex64 {
        let ei = ni * ni;
        let eh = nh * nh;
        let mut eps = (ei + eh) * 0.5;
        for _ in 0..max_iter {
            let den_i = ei + 2.0 * eps;
            let den_h = eh + 2.0 * eps;
            let term_i = (ei - eps) / den_i;
            let term_h = (eh - eps) / den_h;
            let f_total = f * term_i + inv_f * term_h;

            let deriv_i = (-3.0 * ei) / (den_i * den_i);
            let deriv_h = (-3.0 * eh) / (den_h * den_h);
            let df = f * deriv_i + inv_f * deriv_h;

            let delta = -f_total / (df + tiny);
            eps += delta;
            if delta.norm_sqr() < tol_sq {
                break;
            }
        }
        eps
    };

    if n_i.len() >= crate::materials::common::PAR_THRESHOLD {
        let v: Vec<Complex64> = (0..n_i.len())
            .into_par_iter()
            .map(|k| solve(n_i[k], n_h[k]))
            .collect();
        Array1::from(v)
    } else {
        Array1::from_iter((0..n_i.len()).map(|k| solve(n_i[k], n_h[k])))
    }
}

/// Convert an array of permittivities to refractive indices (the √ε step).
pub fn eps_to_nk(eps: ArrayView1<Complex64>) -> Array1<Complex64> {
    if eps.len() >= crate::materials::common::PAR_THRESHOLD {
        let v: Vec<Complex64> = eps
            .as_slice()
            .expect("contiguous")
            .par_iter()
            .map(|z| z.sqrt())
            .collect();
        Array1::from(v)
    } else {
        eps.mapv(|z| z.sqrt())
    }
}
