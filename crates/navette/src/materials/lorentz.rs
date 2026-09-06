//! Lorentz oscillator dispersion model.
//!
//! ε(E) = ε∞ + Σⱼ fⱼ·E0ⱼ² / ((E0ⱼ² − E²) − i·E·Γⱼ),   n̂ = √ε.
//!
//! Faithful port of `compute_lorentz_complex_nk`. Oscillators are passed as a
//! row-major (N_osc, 3) array with columns (E0, Gamma, f0) — exactly the
//! `_lorentz_params` array that `_sync()` already builds on the Python side.

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64;

use crate::materials::common::map_nk;
use crate::materials::units::energy_ev;

/// Complex permittivity for a single energy, summed over all oscillators.
#[inline]
fn lorentz_eps(e: f64, osc: ArrayView2<f64>, eps_inf: f64) -> Complex64 {
    let e_sq = e * e;
    let mut eps = Complex64::new(eps_inf, 0.0);
    for row in osc.rows() {
        let e0 = row[0];
        let gamma = row[1];
        let f0 = row[2];
        let e0_sq = e0 * e0;
        // (E0² − E²) − i·(E·Γ)
        let denom = Complex64::new(e0_sq - e_sq, -e * gamma);
        eps += Complex64::new(f0 * e0_sq, 0.0) / denom;
    }
    eps
}

/// Lorentz complex refractive index n̂ = √ε. `wavelength_nm` in nm.
pub fn lorentz_nk(
    wavelength_nm: ArrayView1<f64>,
    osc: ArrayView2<f64>,
    eps_inf: f64,
) -> Array1<Complex64> {
    map_nk(wavelength_nm, |w| {
        let e = energy_ev(w);
        lorentz_eps(e, osc, eps_inf).sqrt()
    })
}
