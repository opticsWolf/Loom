//! Drude and Drude–Lorentz dispersion models.
//!
//! Drude:          ε = ε∞ − ωp² / (E² + i·γ·E)
//! Drude–Lorentz:  ε = ε∞ − ωp² / (E² + i·γ_d·E) + Σⱼ fⱼ·E0ⱼ²/((E0ⱼ²−E²) − i·E·Γⱼ)
//!
//! Both guard against E = 0 with E_eff = E + 1e-12, exactly as the Python
//! kernels (`compute_drude_complex_nk`, `compute_drude_lorentz_complex_nk`).
//! Note the Drude–Lorentz oscillator term uses E_eff (not raw E) in its
//! denominator — kept faithful to the Python source.

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64;

use crate::materials::common::map_nk;
use crate::materials::units::energy_ev;

const E_GUARD: f64 = 1.0e-12;

/// Drude-only complex refractive index.
pub fn drude_nk(
    wavelength_nm: ArrayView1<f64>,
    omega_p: f64,
    gamma: f64,
    eps_inf: f64,
) -> Array1<Complex64> {
    let wp2 = omega_p * omega_p;
    map_nk(wavelength_nm, |w| {
        let e = energy_ev(w) + E_GUARD;
        let denom = Complex64::new(e * e, gamma * e);
        let eps = Complex64::new(eps_inf, 0.0) - Complex64::new(wp2, 0.0) / denom;
        eps.sqrt()
    })
}

/// Drude term plus Lorentz oscillators. `osc` is (N_osc, 3): (E0, Gamma, f0).
pub fn drude_lorentz_nk(
    wavelength_nm: ArrayView1<f64>,
    omega_p: f64,
    gamma_d: f64,
    eps_inf: f64,
    osc: ArrayView2<f64>,
) -> Array1<Complex64> {
    let wp2 = omega_p * omega_p;
    map_nk(wavelength_nm, |w| {
        let e = energy_ev(w) + E_GUARD;
        let e_sq = e * e;

        // Drude term.
        let drude_denom = Complex64::new(e_sq, gamma_d * e);
        let mut eps = Complex64::new(eps_inf, 0.0) - Complex64::new(wp2, 0.0) / drude_denom;

        // Lorentz terms (use E_eff in the damping term, per the Python source).
        for row in osc.rows() {
            let e0 = row[0];
            let gamma_l = row[1];
            let f0 = row[2];
            let e0_sq = e0 * e0;
            let denom = Complex64::new(e0_sq - e_sq, -e * gamma_l);
            eps += Complex64::new(f0 * e0_sq, 0.0) / denom;
        }
        eps.sqrt()
    })
}
