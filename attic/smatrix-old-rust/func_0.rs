use num_complex::Complex64;
use pyo3::prelude::*;

const SQRT3: f64 = 1.73205080757;

/// Pure-Rust roughness form factor W(q). No Python/PyO3 overhead so it can be
/// called millions of times from the hot loop and inlined by the compiler.
#[inline(always)]
pub fn w_function_inner(q: Complex64, rough_type: i32) -> Complex64 {
    match rough_type {
        0 => Complex64::new(1.0, 0.0),
        1 => {
            let val = q * SQRT3;
            if val.norm() < 1e-9 {
                Complex64::new(1.0, 0.0)
            } else {
                val.sin() / val
            }
        }
        2 => q.cos(),
        3 => {
            let denom = Complex64::new(1.0, 0.0) + (q * q) * 0.5;
            Complex64::new(1.0, 0.0) / denom
        }
        4 => (-(q * q) * 0.5).exp(),
        _ => Complex64::new(1.0, 0.0),
    }
}

/// Python-facing wrapper. Identical signature/behaviour to the original.
#[pyfunction]
pub fn w_function(q: Complex64, rough_type: i32) -> PyResult<Complex64> {
    Ok(w_function_inner(q, rough_type))
}
