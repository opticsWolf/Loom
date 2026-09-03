// redheffer_intensity.rs
//
// Real-valued (intensity) and p-s cross-coherency Redheffer star products —
// Python wrappers. Numerics live in `optics_core`.

use num_complex::Complex64;
use pyo3::prelude::*;

pub use crate::optics_core::{redheffer_product_cross_inner, redheffer_product_real_inner};

#[pyfunction]
pub fn redheffer_product_real(
    ra_rf: f64,
    ra_tb: f64,
    ra_tf: f64,
    ra_rb: f64,
    rb_rf: f64,
    rb_tb: f64,
    rb_tf: f64,
    rb_rb: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(redheffer_product_real_inner(
        ra_rf, ra_tb, ra_tf, ra_rb, rb_rf, rb_tb, rb_tf, rb_rb,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn redheffer_product_cross(
    a_cf: Complex64,
    a_db: Complex64,
    a_df: Complex64,
    a_cb: Complex64,
    b_cf: Complex64,
    b_db: Complex64,
    b_df: Complex64,
    b_cb: Complex64,
) -> PyResult<(Complex64, Complex64, Complex64, Complex64)> {
    Ok(redheffer_product_cross_inner(
        a_cf, a_db, a_df, a_cb, b_cf, b_db, b_df, b_cb,
    ))
}
