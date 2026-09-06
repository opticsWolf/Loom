//! Aggregated PyO3 bindings for Navette — builds the single
//! `navette._navette` extension containing all five native submodules:
//!
//! - `_color` — colorimetry (over `navette::color`)
//! - `_interpolate` — univariate interpolation (over `navette::interpolate`)
//! - `_smatrix` — S-matrix thin-film engine (over `navette::smatrix`)
//! - `_spectralweave` — spectral weaving + targets (over `navette::spectralweave`)
//! - `_materials` — dispersion models (over `navette::materials`)
//!
//! No physics here: every kernel lives in the pure-Rust `navette`
//! umbrella. Wrappers own NumPy inputs, release the GIL while
//! rayon-parallel kernels run, and return NumPy.

mod color;
mod config;
mod interpolate;
mod materials;
mod smatrix;
mod structure;
mod synthesis_merit;
mod synthesis_pipeline;
mod spectralweave;
mod spectralweave_optical;
mod spectralweave_target;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

use crate::color::_color;
use crate::interpolate::_interpolate;
use crate::materials::_materials;
use crate::smatrix::_smatrix;
use crate::spectralweave::_spectralweave;

// Use mimalloc instead of the system allocator. The hot paths here allocate
// many short-lived buffers (and a few large ones); the default Windows heap is
// slow under that pattern.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Aggregate all five engine submodules into `navette._navette`.
#[pymodule]
fn _navette(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(_color))?;
    m.add_wrapped(wrap_pymodule!(_interpolate))?;
    m.add_wrapped(wrap_pymodule!(_smatrix))?;
    m.add_wrapped(wrap_pymodule!(structure::_structure))?;
    m.add_wrapped(wrap_pymodule!(_spectralweave))?;
    m.add_wrapped(wrap_pymodule!(_materials))?;
    Ok(())
}
