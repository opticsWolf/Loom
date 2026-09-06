//! Cauchy dispersion model.
//!
//! n(λ) = A + B/λ² + C/λ⁴   with λ in µm.  k = 0 (or Urbach tail for the
//! `_urbach` variant). Faithful port of `compute_cauchy_*` from the Python.

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;

use crate::materials::common::{map_nk, urbach_k};
use crate::materials::units::wl_um2;

/// Real refractive index from the Cauchy polynomial. `wl_um2` is λ² in µm².
#[inline]
pub fn cauchy_n(wl_um2_val: f64, a: f64, b: f64, c: f64) -> f64 {
    a + b / wl_um2_val + c / (wl_um2_val * wl_um2_val)
}

/// Cauchy complex refractive index (k = 0). `wavelength_nm` in nm.
pub fn cauchy_nk(wavelength_nm: ArrayView1<f64>, a: f64, b: f64, c: f64) -> Array1<Complex64> {
    map_nk(wavelength_nm, |w| {
        Complex64::new(cauchy_n(wl_um2(w), a, b, c), 0.0)
    })
}

/// Cauchy dispersion with an Urbach absorption tail on k.
#[allow(clippy::too_many_arguments)]
pub fn cauchy_urbach_nk(
    wavelength_nm: ArrayView1<f64>,
    a: f64,
    b: f64,
    c: f64,
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Array1<Complex64> {
    map_nk(wavelength_nm, |w| {
        let n = cauchy_n(wl_um2(w), a, b, c);
        let k = urbach_k(w, alpha0, eu, lambda_g);
        Complex64::new(n, k)
    })
}
