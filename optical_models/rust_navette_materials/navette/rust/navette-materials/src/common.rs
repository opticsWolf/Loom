//! Internal shared helpers used across dispersion modules.

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::units::{energy_ev, wl_m, HC_EV_NM};

/// Above this length the element-wise kernels parallelise with rayon; below it
/// the rayon overhead would dominate, so we run serially. Element-wise maps
/// have no cross-element reduction, so serial and parallel are bit-identical.
pub(crate) const PAR_THRESHOLD: usize = 4096;

/// Map `f` over `wavelength_nm`, in parallel for large arrays. Deterministic:
/// each output element depends only on its own input, so results match the
/// serial path exactly (important for golden parity).
#[inline]
pub(crate) fn map_nk<F>(wavelength_nm: ArrayView1<f64>, f: F) -> Array1<Complex64>
where
    F: Fn(f64) -> Complex64 + Sync + Send,
{
    if wavelength_nm.len() >= PAR_THRESHOLD {
        let v: Vec<Complex64> = wavelength_nm
            .as_slice()
            .expect("contiguous view")
            .par_iter()
            .map(|&w| f(w))
            .collect();
        Array1::from(v)
    } else {
        wavelength_nm.mapv(f)
    }
}

/// Urbach exponential band-tail extinction coefficient k(λ).
///
/// Faithful port of `compute_urbach_k_part`:
///   E_g = h·c / λ_g
///   k(λ) = α₀ · exp((E − E_g)/E_u) · λ_m / (4π)   for  E < E_g,  else 0.
///
/// `wavelength_nm` is the user grid; `alpha0` [1/cm], `eu` [eV], `lambda_g` [nm].
#[inline]
pub(crate) fn urbach_k(wavelength_nm: f64, alpha0: f64, eu: f64, lambda_g: f64) -> f64 {
    let e = energy_ev(wavelength_nm);
    let e_g = HC_EV_NM / lambda_g;
    if e < e_g {
        let absorption = alpha0 * ((e - e_g) / eu).exp();
        absorption * wl_m(wavelength_nm) / (4.0 * std::f64::consts::PI)
    } else {
        0.0
    }
}

/// Build a contiguous, owned `Array1` from a 1-D view (used at the binding edge
/// when we need `Send` data to release the GIL).
pub fn to_owned(v: ArrayView1<f64>) -> Array1<f64> {
    v.to_owned()
}
