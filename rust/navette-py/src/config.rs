//! Thin PyO3 config types: validated constructors over `navette::config`.
//!
//! These replace the pydantic models. Each class validates its input
//! dict natively on construction (`ValueError` on failure, unknown
//! fields refused) and round-trips via `to_dict`. No schema lives here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use navette::config::BlockCfg;
use navette::smatrix::synthesis::design_config::{
  GroupRow, LayerRow, MaterialDef, StructureCfg,
};

fn from_dict<T: for<'de> serde::Deserialize<'de>>(d: &Bound<'_, PyDict>) -> PyResult<T> {
  let v = crate::structure::py_to_json(d.as_any())?;
  serde_json::from_value(v).map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))
}

fn to_dict(py: Python<'_>, v: serde_json::Value) -> PyResult<Py<PyAny>> {
  crate::structure::json_to_py(py, &v)
}

macro_rules! config_type {
  ($pyname:literal, $cls:ident, $core:ty) => {
    #[pyclass(name = $pyname)]
    #[derive(Clone)]
    pub struct $cls {
      inner: $core,
    }

    #[pymethods]
    impl $cls {
      #[new]
      fn new(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let inner: $core = from_dict(d)?;
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
      }

      fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(PyValueError::new_err)
      }

      fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let v =
          serde_json::to_value(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))?;
        to_dict(py, v)
      }
    }
  };
}

config_type!("MaterialDefinition", PyMaterialDef, MaterialDef);
config_type!("LayerConfig", PyLayerRow, LayerRow);
config_type!("GroupConfig", PyGroupRow, GroupRow);
config_type!("BlockConfig", PyBlockCfg, BlockCfg);
config_type!("NamedStructureConfig", PyStructureCfg, StructureCfg);

/// Gate a program document (JSON text) natively: version, kind,
/// unknown keys, sections presence. Returns `(kind, name, payload)`
/// with payload as JSON text for the caller to parse.
#[pyfunction]
pub(crate) fn gate_document(request_json: &str) -> PyResult<(String, Option<String>, String)> {
  let raw: serde_json::Value = serde_json::from_str(request_json)
    .map_err(|e| PyValueError::new_err(format!("program: invalid JSON: {e}")))?;
  let (kind, name, payload) = navette::config::gate_document(&raw).map_err(PyValueError::new_err)?;
  let text =
    serde_json::to_string(&payload).map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok((kind, name, text))
}

/// Validate a target-set document natively (thin hook for the
/// dataclass holders; the authoritative checks run at ingestion).
#[pyfunction]
pub(crate) fn validate_targets(request_json: &str) -> PyResult<()> {
  let set: navette::smatrix::synthesis::targets::TargetSet = serde_json::from_str(request_json)
    .map_err(|e| PyValueError::new_err(format!("invalid target set: {e}")))?;
  for t in &set.spectral {
    check_target(
      &t.values,
      &t.wavelengths,
      &t.tolerances,
      t.weight,
      t.integral,
      t.normalize_count,
      t.band.as_ref().map(|b| match b {
        navette::smatrix::synthesis::targets::Band::Scalar(x) => vec![*x],
        navette::smatrix::synthesis::targets::Band::Points(v) => v.clone(),
      }),
      t.kind.as_str(),
    )?;
  }
  for t in &set.angular {
    check_target(
      &t.values,
      &t.angles,
      &t.tolerances,
      t.weight,
      t.integral,
      t.normalize_count,
      t.band.as_ref().map(|b| match b {
        navette::smatrix::synthesis::targets::Band::Scalar(x) => vec![*x],
        navette::smatrix::synthesis::targets::Band::Points(v) => v.clone(),
      }),
      t.kind.as_str(),
    )?;
  }
  // Color demands: full single-validator check (curve vocabulary, compat
  // matrix, table resolution) — key stamped at compile, unchecked here.
  for t in &set.color {
    navette::smatrix::synthesis::targets::check_color_demand(t)
      .map(|_| ())
      .map_err(PyValueError::new_err)?;
  }
  Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_target(
  values: &[f64],
  grid: &[f64],
  tolerances: &[f64],
  weight: f64,
  integral: bool,
  normalize_count: bool,
  band: Option<Vec<f64>>,
  kind: &str,
) -> PyResult<()> {
  if values.len() != grid.len() || values.len() != tolerances.len() {
    return Err(PyValueError::new_err("target shape mismatch."));
  }
  if !weight.is_finite() || weight < 0.0 {
    return Err(PyValueError::new_err(format!(
      "weight must be finite and >= 0 (got {weight})."
    )));
  }
  if integral && normalize_count {
    return Err(PyValueError::new_err(
      "integral targets reject normalize_count (the mean already is one).",
    ));
  }
  if !["e", "a", "b", "r", "c"].contains(&kind) {
    return Err(PyValueError::new_err(
      "Invalid kind (use 'e', 'a', 'b', 'r', or 'c').",
    ));
  }
  match band {
    None => Ok(()),
    Some(b) => {
      if b.len() == 1 {
        if b[0] < 0.0 {
          return Err(PyValueError::new_err("band must be >= 0."));
        }
        Ok(())
      } else if b.len() != values.len() {
        Err(PyValueError::new_err("band shape mismatch."))
      } else if b.iter().any(|x| *x < 0.0) {
        Err(PyValueError::new_err("band must be >= 0."))
      } else {
        Ok(())
      }
    }
  }
}
