// Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//
// Module map:
//   optics_core           — pure-Rust shared primitives (stars, roughness,
//                           fast complex kernels, spectral differentiation)
//   roughness             — Python wrapper: w_function
//   redheffer_field       — Python wrapper: redheffer_product_complex_field
//   redheffer_intensity   — Python wrappers: redheffer_product_real / _cross
//   coherent_block        — solve_coherent_block_fields (s/p/dual solvers)
//   core_engine           — request-driven unified engine (core_engine)
//   optimizer             — landscape scan, Nelder-Mead, field profiles
//   needle_operator       — analytic needle-operator sensitivities (pure Rust)
//   needle_engine         — request-driven rayon/pyo3 API over the operator
//   synthesis             — automated design synthesis (DesignStack, …)

use pyo3::prelude::*;

mod optics_core;
mod roughness; // w_function
mod redheffer_field; // redheffer_product_complex_field
mod redheffer_intensity; // redheffer_product_real / redheffer_product_cross
mod coherent_block; // solve_coherent_block_fields
mod core_engine; // core_engine (unified, request-driven)
mod optimizer; // scan_landscape / find_local_minima / nelder_mead / field_profile
pub mod needle_operator; // analytic needle sensitivities (pure Rust core)
pub mod needle_engine; // needle_engine (request-driven, rayon + pyo3)
pub mod synthesis; // automated design synthesis (pure Rust core)

#[pymodule]
fn _smatrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all public functions under their original Python names
    m.add_function(wrap_pyfunction!(roughness::w_function, m)?)?;
    m.add_function(wrap_pyfunction!(
        redheffer_field::redheffer_product_complex_field,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(redheffer_intensity::redheffer_product_real, m)?)?;
    m.add_function(wrap_pyfunction!(redheffer_intensity::redheffer_product_cross, m)?)?;
    m.add_function(wrap_pyfunction!(
        coherent_block::solve_coherent_block_fields,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(core_engine::core_engine, m)?)?;

    m.add_function(wrap_pyfunction!(optimizer::scan_landscape, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::find_local_minima, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::field_profile, m)?)?;
    m.add_function(wrap_pyfunction!(needle_engine::needle_engine, m)?)?;

    // Needle-request bitmask constants (mirror as a Python IntFlag).
    m.add("NREQ_P", needle_engine::NREQ_P)?;
    m.add("NREQ_P_MB", needle_engine::NREQ_P_MB)?;
    m.add("NREQ_DPHI", needle_engine::NREQ_DPHI)?;
    m.add("NREQ_DGD", needle_engine::NREQ_DGD)?;
    m.add("NREQ_DGDD", needle_engine::NREQ_DGDD)?;
    m.add("NREQ_DTOD", needle_engine::NREQ_DTOD)?;
    m.add("NREQ_DFOD", needle_engine::NREQ_DFOD)?;

    Ok(())
}
