//! Thin PyO3 bindings for the Navette spectral-weaving engine.
//!
//! No physics here: caching, distribution, and merit logic live in the
//! pure-Rust `navette-spectralweave` core. Wrappers own NumPy inputs,
//! release the GIL while core kernels run, and return NumPy.

mod opticalweaver_py;
mod targetweaver_py;

use pyo3::prelude::*;

use crate::opticalweaver_py::{PyOpticalCollection, PyOpticalWeaver, PySpectralDataFrame};
use crate::targetweaver_py::{calculate_merit, PyTargetWeaver};

// Use mimalloc instead of the system allocator. The hot paths here allocate
// many short-lived buffers (and a few large ones); the default Windows heap is
// slow under that pattern.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[pymodule]
fn _spectralweave(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Optical Data Structures
    m.add_class::<PySpectralDataFrame>()?;
    m.add_class::<PyOpticalCollection>()?;
    m.add_class::<PyOpticalWeaver>()?;

    // Target/Optimization Constraints
    m.add_class::<PyTargetWeaver>()?;
    m.add_function(wrap_pyfunction!(calculate_merit, m)?)?;

    Ok(())
}
