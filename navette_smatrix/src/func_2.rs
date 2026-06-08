use num_complex::Complex64;
use num_complex::ComplexFloat; // for .norm()/.recip()
use pyo3::prelude::*;

/// Real-valued (intensity) Redheffer star product. Pure Rust, inlined.
#[inline(always)]
pub fn redheffer_product_real_inner(
    ra_rf: f64,
    ra_tb: f64,
    ra_tf: f64,
    ra_rb: f64,
    rb_rf: f64,
    rb_tb: f64,
    rb_tf: f64,
    rb_rb: f64,
) -> (f64, f64, f64, f64) {
    const DBL_EPSILON: f64 = 2.22e-16;

    let denom = 1.0 - ra_rb * rb_rf;
    let inv_denom = if denom.abs() < DBL_EPSILON {
        0.0
    } else {
        1.0 / denom
    };

    let rf = ra_rf + ra_tb * rb_rf * ra_tf * inv_denom;
    let tb = ra_tb * rb_tb * inv_denom;
    let tf = rb_tf * ra_tf * inv_denom;
    let rb = rb_rb + rb_tf * ra_rb * rb_tb * inv_denom;

    (rf, tb, tf, rb)
}

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

/// Complex (coherency) Redheffer star product over the p-s cross-amplitudes
/// C = (p-field)·conj(s-field). Structurally identical to
/// `redheffer_product_real_inner`; the denominator `1 - C_Ab·C_Bf` is the
/// n=m term of the incoherent multiple-reflection geometric series in the
/// cross channel (different bounce orders are mutually incoherent). On the
/// diagonal (C = |field|²) this collapses exactly onto the real product, which
/// is why R/T are unaffected by the coherency mode.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn redheffer_product_cross_inner(
    a_cf: Complex64,
    a_db: Complex64,
    a_df: Complex64,
    a_cb: Complex64,
    b_cf: Complex64,
    b_db: Complex64,
    b_df: Complex64,
    b_cb: Complex64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    const DBL_EPSILON: f64 = 2.22e-16;

    let denom = Complex64::new(1.0, 0.0) - a_cb * b_cf;
    let inv = if denom.norm() < DBL_EPSILON {
        Complex64::new(0.0, 0.0)
    } else {
        denom.recip()
    };

    let cf = a_cf + a_db * b_cf * a_df * inv;
    let db = a_db * b_db * inv;
    let df = b_df * a_df * inv;
    let cb = b_cb + b_df * a_cb * b_db * inv;

    (cf, db, df, cb)
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
