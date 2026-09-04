//! Spectral-weaving submodule of the aggregated `navette._navette` extension.
//!
//! Re-exports the optical/target wrappers; registration lives here so the
//! aggregator root only calls `wrap_pymodule!(_spectralweave)`.

use pyo3::prelude::*;

pub(crate) use super::spectralweave_optical::{
    PyOpticalCollection, PyOpticalWeaver, PySpectralDataFrame,
};
pub(crate) use super::spectralweave_target::{calculate_merit, PyTargetWeaver};

/// Register the spectral-weaving submodule (`navette._navette._spectralweave`).
#[pymodule]
pub fn _spectralweave(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Optical Data Structures
    m.add_class::<PySpectralDataFrame>()?;
    m.add_class::<PyOpticalCollection>()?;
    m.add_class::<PyOpticalWeaver>()?;

    // Target/Optimization Constraints
    m.add_class::<PyTargetWeaver>()?;
    m.add_function(wrap_pyfunction!(calculate_merit, m)?)?;

    Ok(())
}
