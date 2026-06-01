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
