//! Tabulated and constant dispersion models.
//!
//! [`konstant_nk`] is a wavelength-independent `n + ik`.
//! [`table_nk`] linearly interpolates measured `(wavelength, n, k)` tables
//! with `np.interp` (clamped) semantics via [`crate::kk::interp`].
//! Only linear interpolation is supported; resample exotic tables offline.

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;

use crate::kk::interp;

/// Wavelength-independent `n + ik`.
pub fn konstant_nk(
    wavelength_nm: ArrayView1<f64>,
    n: f64,
    k: f64,
) -> Array1<Complex64> {
    let nk = Complex64::new(n, k);
    wavelength_nm.mapv(|_| nk)
}

/// Linear table lookup with `np.interp` (clamped) semantics.
///
/// `grid_wl` ascending in nm; `k_vals` may be `None` (→ `k = 0`).
/// `n_factor` / `k_factor` scale the interpolated values.
#[allow(clippy::too_many_arguments)]
pub fn table_nk(
    wavelength_nm: ArrayView1<f64>,
    grid_wl: ArrayView1<f64>,
    n_vals: ArrayView1<f64>,
    k_vals: Option<ArrayView1<f64>>,
    n_factor: f64,
    k_factor: f64,
) -> Array1<Complex64> {
    assert!(
        grid_wl.len() >= 2,
        "table_nk needs at least 2 grid points, got {}",
        grid_wl.len()
    );
    assert_eq!(
        grid_wl.len(),
        n_vals.len(),
        "grid/n length mismatch ({} vs {})",
        grid_wl.len(),
        n_vals.len()
    );
    if let Some(k) = &k_vals {
        assert_eq!(
            grid_wl.len(),
            k.len(),
            "grid/k length mismatch ({} vs {})",
            grid_wl.len(),
            k.len()
        );
    }
    let gw: Vec<f64> = grid_wl.to_vec();
    let nv: Vec<f64> = n_vals.to_vec();
    let kv: Option<Vec<f64>> = k_vals.map(|k| k.to_vec());
    wavelength_nm.mapv(|w| {
        let n = interp(w, &gw, &nv) * n_factor;
        let k = match &kv {
            Some(k) => interp(w, &gw, k) * k_factor,
            None => 0.0,
        };
        Complex64::new(n, k)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn konstant_is_flat() {
        let wl = array![400.0, 500.0, 600.0];
        let nk = konstant_nk(wl.view(), 1.5, 0.01);
        assert!(nk.iter().all(|&z| z == Complex64::new(1.5, 0.01)));
    }

    #[test]
    fn table_exact_at_nodes_and_clamped() {
        let grid = array![400.0, 500.0, 600.0];
        let n = array![1.0, 2.0, 3.0];
        let wl = array![400.0, 450.0, 600.0, 700.0];
        let nk = table_nk(wl.view(), grid.view(), n.view(), None, 1.0, 1.0);
        assert!((nk[0].re - 1.0).abs() < 1e-12);
        assert!((nk[1].re - 1.5).abs() < 1e-12);
        assert!((nk[2].re - 3.0).abs() < 1e-12);
        assert!((nk[3].re - 3.0).abs() < 1e-12); // clamped
        assert!(nk.iter().all(|&z| z.im == 0.0));
    }
}
