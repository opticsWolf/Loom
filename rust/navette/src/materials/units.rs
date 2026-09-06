//! Physical constants and unit conversions.
//!
//! Single source of truth for the wavelength/energy conversions that were
//! previously scattered across the Python kernels. Every model takes the
//! user-facing wavelength in **nanometres** at the boundary and converts here,
//! so the Python wrapper never has to prepare derived arrays.

use ndarray::{Array1, ArrayView1};

/// h·c in eV·nm. Matches `_HC_EV_NM` in the Python implementation exactly.
pub const HC_EV_NM: f64 = 1239.8419843320028;

/// Photon energy [eV] from wavelength [nm]:  E = h·c / λ.
#[inline]
pub fn energy_ev(wavelength_nm: f64) -> f64 {
    HC_EV_NM / wavelength_nm
}

/// Wavelength squared in µm²:  (λ_nm / 1000)².  (Cauchy / Sellmeier work in µm.)
#[inline]
pub fn wl_um2(wavelength_nm: f64) -> f64 {
    let um = wavelength_nm * 1.0e-3;
    um * um
}

/// Wavelength in metres:  λ_nm · 1e-9.  (Urbach tail uses metres.)
#[inline]
pub fn wl_m(wavelength_nm: f64) -> f64 {
    wavelength_nm * 1.0e-9
}

/// Vectorised energy conversion.
pub fn energy_ev_arr(wavelength_nm: ArrayView1<f64>) -> Array1<f64> {
    wavelength_nm.mapv(energy_ev)
}

/// Vectorised µm² conversion.
pub fn wl_um2_arr(wavelength_nm: ArrayView1<f64>) -> Array1<f64> {
    wavelength_nm.mapv(wl_um2)
}
