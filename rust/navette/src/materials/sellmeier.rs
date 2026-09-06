//! Sellmeier dispersion model.
//!
//! n²(λ) = 1 + Σᵢ Bᵢ·λ² / (λ² − Cᵢ)   with λ in µm.  The third term is only
//! included when B3 ≠ 0 (matching the Python `np.where(B3 != 0.0, …, 0.0)`).
//! k = 0 (or Urbach tail for the `_urbach` variant).

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;

use crate::materials::common::{map_nk, urbach_k};
use crate::materials::units::wl_um2;

/// Real refractive index from the (up to) three-term Sellmeier equation.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn sellmeier_n(
    l2: f64, // λ² in µm²
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    b3: f64,
    c3: f64,
) -> f64 {
    let term1 = b1 * l2 / (l2 - c1);
    let term2 = b2 * l2 / (l2 - c2);
    // Faithful to the Python branch: contributes only when B3 is nonzero.
    let term3 = if b3 != 0.0 { b3 * l2 / (l2 - c3) } else { 0.0 };
    let n_sq = 1.0 + term1 + term2 + term3;
    n_sq.sqrt()
}

/// Sellmeier complex refractive index (k = 0).
#[allow(clippy::too_many_arguments)]
pub fn sellmeier_nk(
    wavelength_nm: ArrayView1<f64>,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    b3: f64,
    c3: f64,
) -> Array1<Complex64> {
    map_nk(wavelength_nm, |w| {
        Complex64::new(sellmeier_n(wl_um2(w), b1, c1, b2, c2, b3, c3), 0.0)
    })
}

/// Sellmeier dispersion with an Urbach absorption tail on k.
#[allow(clippy::too_many_arguments)]
pub fn sellmeier_urbach_nk(
    wavelength_nm: ArrayView1<f64>,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    b3: f64,
    c3: f64,
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Array1<Complex64> {
    map_nk(wavelength_nm, |w| {
        let n = sellmeier_n(wl_um2(w), b1, c1, b2, c2, b3, c3);
        let k = urbach_k(w, alpha0, eu, lambda_g);
        Complex64::new(n, k)
    })
}
