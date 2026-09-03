pub mod opticalweaver;
pub mod targetweaver;

use pyo3::prelude::*;

use crate::opticalweaver::{
    PyOpticalCollection, PyOpticalWeaver, PySpectralDataFrame,
};
use crate::targetweaver::{calculate_merit, PyTargetWeaver};

// Use mimalloc instead of the system allocator. The hot paths here allocate
// many short-lived buffers (and a few large ones); the default Windows heap is
// slow under that pattern.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ---------------------------------------------------------------------------
// Module Registration
// ---------------------------------------------------------------------------
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