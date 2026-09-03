//! UBF (Monolog-Lorentz) Cody–Lorentz dielectric model.
//!
//! Same FFT Kramers–Kronig pipeline as [`crate::cody_lorentz`] — only the ε₂
//! generator differs. For each oscillator [Eg, Ec, β, A, Γ, γ] (β = 1/Eu):
//!
//!   ε₂(E) += (A/E) · [ln(1 + exp(β(E−Eg)))]^γ · E·Γ·Ec / ((E²−Ec²)² + Γ²E²)
//!
//! Faithful port of `_eps2_monolog`, including the overflow guards on the log
//! term and the γ ∈ {2, 0.5, 1} fast paths (so parity matches the Python kernel
//! exactly, not just to within pow() rounding). The oscillator array already
//! carries β in column 2 — the Python `_sync()` converts Eu → β before the call.

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::common::PAR_THRESHOLD;
use crate::kk;
use crate::units::energy_ev;

/// Monolog-Lorentz ε₂ at a single energy.
#[inline]
fn eps2_at(e: f64, osc: ArrayView2<f64>) -> f64 {
    if e < 1e-9 {
        return 0.0;
    }
    let e_sq = e * e;
    let mut val = 0.0;
    for row in osc.rows() {
        let (eg, ec, beta, a, g, y) = (row[0], row[1], row[2], row[3], row[4], row[5]);

        // band term: [ln(1 + e^x)]^γ with overflow/underflow guards
        let x = beta * (e - eg);
        let base = if x > 50.0 {
            x
        } else if x < -50.0 {
            0.0
        } else {
            (1.0 + x.exp()).ln()
        };
        let band = if y == 2.0 {
            base * base
        } else if y == 0.5 {
            base.sqrt()
        } else if y == 1.0 {
            base
        } else {
            base.powf(y)
        };

        let denom = (e_sq - ec * ec).powi(2) + (g * e).powi(2);
        let lorentz = (e * g * ec) / denom;
        val += (a / e) * band * lorentz;
    }
    val
}

/// Monolog-Lorentz ε₂ over an energy array (parallel above the threshold).
pub fn eps2_monolog(energies: &[f64], osc: ArrayView2<f64>) -> Vec<f64> {
    if energies.len() >= PAR_THRESHOLD {
        energies.par_iter().map(|&e| eps2_at(e, osc)).collect()
    } else {
        energies.iter().map(|&e| eps2_at(e, osc)).collect()
    }
}

/// UBF Cody–Lorentz complex refractive index. `osc` is (N, 6): (Eg, Ec, β, A, Γ, γ).
/// Returns `Err` if any target energy lies outside the KK grid.
pub fn ubf_nk(
    wavelength_nm: ArrayView1<f64>,
    osc: ArrayView2<f64>,
    eps_inf: f64,
) -> Result<Array1<Complex64>, String> {
    let e_full = kk::energy_grid();
    let eps2_full = eps2_monolog(&e_full, osc);
    let eps1_full = kk::kk_fft(&eps2_full, eps_inf);

    let target: Vec<f64> = wavelength_nm.iter().map(|&w| energy_ev(w)).collect();
    let (lo, hi) = (e_full[0], e_full[e_full.len() - 1]);
    let tmin = target.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = target.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if tmin < lo || tmax > hi {
        return Err(format!(
            "target energies [{tmin:.4}, {tmax:.4}] eV exceed KK grid [{lo:.4}, {hi:.4}] eV"
        ));
    }

    let eps2_t = eps2_monolog(&target, osc);
    let out: Vec<Complex64> = target
        .iter()
        .enumerate()
        .map(|(i, &e)| Complex64::new(kk::interp(e, &e_full, &eps1_full), eps2_t[i]).sqrt())
        .collect();
    Ok(Array1::from(out))
}
