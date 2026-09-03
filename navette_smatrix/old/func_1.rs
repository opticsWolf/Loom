use num_complex::Complex64;
use num_complex::ComplexFloat; // for .recip() / .abs()
use pyo3::prelude::*;

/// Complex (field-amplitude) Redheffer star product. Pure Rust, inlined.
#[inline(always)]
pub fn redheffer_product_complex_field_inner(
    r_a_front: Complex64,
    t_a_back: Complex64,
    t_a_fwd: Complex64,
    r_a_back: Complex64,
    r_b_front: Complex64,
    t_b_back: Complex64,
    t_b_fwd: Complex64,
    r_b_back: Complex64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    const LOG_MIN: f64 = 1e-100;
    const EPS: f64 = 1e-300;

    let mut denom = Complex64::new(1.0, 0.0) - r_a_back * r_b_front;
    if denom.abs() < LOG_MIN {
        // Phase-preserving regularization (matches the reference implementation).
        let phase = denom / (denom.abs() + EPS);
        denom = Complex64::new(LOG_MIN, 0.0) * phase + EPS;
    }
    let inv_denom = denom.recip();

    let s_r_front = r_a_front + t_a_back * r_b_front * t_a_fwd * inv_denom;
    let s_t_back = t_a_back * t_b_back * inv_denom;
    let s_t_fwd = t_b_fwd * t_a_fwd * inv_denom;
    let s_r_back = r_b_back + t_b_fwd * r_a_back * t_b_back * inv_denom;

    (s_r_front, s_t_back, s_t_fwd, s_r_back)
}

#[pyfunction]
pub fn redheffer_product_complex_field(
    r_a_front: Complex64,
    t_a_back: Complex64,
    t_a_fwd: Complex64,
    r_a_back: Complex64,
    r_b_front: Complex64,
    t_b_back: Complex64,
    t_b_fwd: Complex64,
    r_b_back: Complex64,
) -> PyResult<(Complex64, Complex64, Complex64, Complex64)> {
    Ok(redheffer_product_complex_field_inner(
        r_a_front, t_a_back, t_a_fwd, r_a_back, r_b_front, t_b_back, t_b_fwd, r_b_back,
    ))
}
