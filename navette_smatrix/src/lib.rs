// Loom Matrix -- Rust Rewrite of Numba-optimized thin-film optical solver
use pyo3::prelude::*;

mod func_0; // w_function
mod func_1; // redheffer_product_complex_field
mod func_2; // redheffer_product_real / redheffer_product_cross
mod func_3; // solve_coherent_block_fields
mod func_4; // core_engine (unified, request-driven)
mod func_eigenmode; // eigenmode_solver

#[pymodule]
fn _smatrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all public functions under their original Python names
    m.add_function(wrap_pyfunction!(func_0::w_function, m)?)?;
    m.add_function(wrap_pyfunction!(func_1::redheffer_product_complex_field, m)?)?;
    m.add_function(wrap_pyfunction!(func_2::redheffer_product_real, m)?)?;
    m.add_function(wrap_pyfunction!(func_2::redheffer_product_cross, m)?)?;
    m.add_function(wrap_pyfunction!(func_3::solve_coherent_block_fields, m)?)?;
    m.add_function(wrap_pyfunction!(func_4::core_engine, m)?)?;

    m.add_function(wrap_pyfunction!(func_eigenmode::scan_landscape, m)?)?;
    m.add_function(wrap_pyfunction!(func_eigenmode::find_local_minima, m)?)?;
    m.add_function(wrap_pyfunction!(func_eigenmode::nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(func_eigenmode::field_profile, m)?)?;

    Ok(())
}
