//! Cody–Lorentz / UBF Cody–Lorentz dielectric model.
//!
//! ε₂ is generated on the cached 8192-pt energy grid (band region above Et,
//! C⁰-matched Urbach tail below Et), Kramers–Kronig transformed to ε₁ via
//! [`crate::materials::kk`], linearly interpolated to the target energies, and combined
//! with ε₂ evaluated exactly at the targets: n̂ = √(ε₁ + i·ε₂).
//!
//! Oscillator layout is (N_osc, 4): columns (E0, A, Gamma, Ep).
//!
//! The ε₂ generator is faithful to the (no-fastmath) Python kernel; the KK
//! step carries a relaxed tolerance by design (FFT-library differences).

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::materials::common::PAR_THRESHOLD;
use crate::materials::kk;
use crate::materials::units::energy_ev;

/// Urbach amplitude: total ε₂ at Et, summed over oscillators (C⁰ match point).
fn a_t_total(eg: f64, et: f64, osc: ArrayView2<f64>) -> f64 {
    let et2 = et * et;
    let mut s = 0.0;
    for row in osc.rows() {
        let (e0, a, gam, ep) = (row[0], row[1], row[2], row[3]);
        let e0sq = e0 * e0;
        let cody_et = if et > eg {
            let d = et - eg;
            (d * d) / (d * d + ep * ep)
        } else {
            0.0
        };
        let denom_et = (et2 - e0sq).powi(2) + (et * gam).powi(2);
        s += a * cody_et * (et * gam) / denom_et;
    }
    s
}

/// ε₂ at a single energy.
#[inline]
fn eps2_at(e: f64, eg: f64, et: f64, inv_eu: f64, att: f64, osc: ArrayView2<f64>) -> f64 {
    if e >= et {
        if e > eg {
            let d = e - eg;
            let dsq = d * d;
            let e2 = e * e;
            let mut val = 0.0;
            for row in osc.rows() {
                let (e0, a, gam, ep) = (row[0], row[1], row[2], row[3]);
                let e0sq = e0 * e0;
                let cody = dsq / (dsq + ep * ep);
                let denom = (e2 - e0sq).powi(2) + (e * gam).powi(2);
                val += a * cody * (e * gam) / denom;
            }
            val
        } else {
            0.0
        }
    } else if e > 1e-9 {
        att * (et / e) * ((e - et) * inv_eu).exp()
    } else {
        0.0
    }
}

/// ε₂ over an energy array (parallel above the threshold; the 8192-pt grid is).
pub fn eps2_multi(energies: &[f64], eg: f64, et: f64, eu: f64, osc: ArrayView2<f64>) -> Vec<f64> {
    let inv_eu = 1.0 / eu;
    let att = a_t_total(eg, et, osc);
    if energies.len() >= PAR_THRESHOLD {
        energies
            .par_iter()
            .map(|&e| eps2_at(e, eg, et, inv_eu, att, osc))
            .collect()
    } else {
        energies
            .iter()
            .map(|&e| eps2_at(e, eg, et, inv_eu, att, osc))
            .collect()
    }
}

/// Cody–Lorentz complex refractive index. `wavelength_nm` in nm.
///
/// Returns `Err` if any target energy lies outside the KK grid
/// [`kk::GRID_MIN`, `kk::GRID_MAX`] (the binding maps this to `ValueError`).
pub fn cody_lorentz_nk(
    wavelength_nm: ArrayView1<f64>,
    eg: f64,
    et: f64,
    eu: f64,
    osc: ArrayView2<f64>,
    eps_inf: f64,
) -> Result<Array1<Complex64>, String> {
    let e_full = kk::energy_grid();
    let eps2_full = eps2_multi(&e_full, eg, et, eu, osc);
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

    let eps2_t = eps2_multi(&target, eg, et, eu, osc);
    let out: Vec<Complex64> = target
        .iter()
        .enumerate()
        .map(|(i, &e)| Complex64::new(kk::interp(e, &e_full, &eps1_full), eps2_t[i]).sqrt())
        .collect();
    Ok(Array1::from(out))
}
