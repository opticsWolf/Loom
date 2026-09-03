// roughness.rs
//
// Interface roughness form factors W(q). The numerics live in
// `optics_core::w_function_inner`; this module only provides the Python-facing
// wrapper under the historical name `w_function`.

use num_complex::Complex64;
use pyo3::prelude::*;

pub use crate::optics_core::w_function_inner;

/// Python-facing wrapper. Identical signature/behaviour to the original.
#[pyfunction]
pub fn w_function(q: Complex64, rough_type: i32) -> PyResult<Complex64> {
    Ok(w_function_inner(q, rough_type))
}
