// redheffer_field.rs
//
// Complex (field-amplitude) Redheffer star product — Python wrapper.
// Numerics live in `optics_core::redheffer_product_complex_field_inner`.

use num_complex::Complex64;
use pyo3::prelude::*;

pub use crate::optics_core::redheffer_product_complex_field_inner;

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
