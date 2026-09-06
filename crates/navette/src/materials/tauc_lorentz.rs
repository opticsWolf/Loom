//! Tauc–Lorentz dielectric model (Jellison & Modine, 1996).
//!
//! ε₂ is the analytic Tauc–Lorentz form; ε₁ is recovered by the same FFT
//! Kramers–Kronig path used by [`crate::materials::cody_lorentz`] and [`crate::materials::ubf`]
//! (so the KK convention is identical across all KK-based models here, and the
//! result is parity-tested against the NumPy FFT reference). For a shared
//! optical gap `Eg` and oscillators j = (A, E0, C):
//!
//!   ε₂(E) = Σⱼ  Aⱼ·E0ⱼ·Cⱼ·(E−Eg)² / ( ((E²−E0ⱼ²)² + Cⱼ²·E²) · E )   for E > Eg
//!   ε₂(E) = 0                                                        for E ≤ Eg
//!
//! n̂ = √(ε₁ + i·ε₂).  Oscillator layout is (N_osc, 3): columns (A, E0, C).
//!
//! Note: ε₁ here is the numerical FFT-KK transform of ε₂ on the 0.01–80 eV
//! grid, matching the library's other KK models. If the closed-form
//! Jellison–Modine ε₁ (atan/ln terms) is ever required instead, it would be a
//! separate analytic routine; the FFT-KK route is chosen for consistency.

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::materials::common::PAR_THRESHOLD;
use crate::materials::kk;
use crate::materials::units::energy_ev;

/// Tauc–Lorentz ε₂ at a single energy (shared gap `eg`).
#[inline]
fn eps2_at(e: f64, eg: f64, osc: ArrayView2<f64>) -> f64 {
    if e <= eg {
        return 0.0;
    }
    let e_sq = e * e;
    let tauc = (e - eg) * (e - eg);
    let mut val = 0.0;
    for row in osc.rows() {
        let (a, e0, c) = (row[0], row[1], row[2]);
        let e0_sq = e0 * e0;
        let denom = ((e_sq - e0_sq).powi(2) + c * c * e_sq) * e;
        val += a * e0 * c * tauc / denom;
    }
    val
}

/// Tauc–Lorentz ε₂ over an energy array (parallel above the threshold).
pub fn eps2_multi(energies: &[f64], eg: f64, osc: ArrayView2<f64>) -> Vec<f64> {
    if energies.len() >= PAR_THRESHOLD {
        energies.par_iter().map(|&e| eps2_at(e, eg, osc)).collect()
    } else {
        energies.iter().map(|&e| eps2_at(e, eg, osc)).collect()
    }
}

/// Tauc–Lorentz complex refractive index. `wavelength_nm` in nm; `osc` is
/// (N, 3): (A, E0, C). Returns `Err` if any target energy is outside the KK grid.
pub fn tauc_lorentz_nk(
    wavelength_nm: ArrayView1<f64>,
    eg: f64,
    osc: ArrayView2<f64>,
    eps_inf: f64,
) -> Result<Array1<Complex64>, String> {
    let e_full = kk::energy_grid();
    let eps2_full = eps2_multi(&e_full, eg, osc);
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

    let eps2_t = eps2_multi(&target, eg, osc);
    let out: Vec<Complex64> = target
        .iter()
        .enumerate()
        .map(|(i, &e)| Complex64::new(kk::interp(e, &e_full, &eps1_full), eps2_t[i]).sqrt())
        .collect();
    Ok(Array1::from(out))
}
