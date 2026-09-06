//! Thin PyO3 bindings for the `navette-structure` model core.
//!
//! No model logic here: wrappers own the conversions (dicts, numpy,
//! warnings) and delegate everything to the pure-Rust core. Conventions
//! match the other modules: `Result<_, String>` → `PyValueError`, GIL
//! released around expansion, numpy arrays in/out.
//!
//! Boundary adaptations (documented, all tested):
//! - warnings return as `warning:`-prefixed strings from `validate` (so
//!   `gate_validation`/`is_warning` work unchanged); `solver_inputs` and
//!   friends re-emit via `warnings.warn` and return clean outputs.
//! - RNG is `seed: Option<u64>` (`None` = thread RNG, NOT NumPy global).
//! - providers cross as `DictProvider` snapshots: the Python bridge
//!   materializes custom/weaver providers before calling (documented in
//!   Phase C); `.nk` duck-objects stay Python-side.
//! - masks cross as fixed-length int lists (wrong lengths refuse at the
//!   boundary instead of flagging in `validate`).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyComplex, PyDict, PyList};
use serde_json::Value;

use navette::structure::{
  expand,
  Architect, BlockKind, DictProvider, Entry, ExpandOptions, Group, Layer, LayerType, MaterialProvider,
  MaterialSpec, RoughnessType, SharedGroup, SharedStructure, SolverArrays, SpecProvider, Structure,
};

fn ver<T>(r: Result<T, String>) -> PyResult<T> {
  r.map_err(PyValueError::new_err)
}

/// Re-emit Rust-side warnings through Python `warnings.warn`.
pub(crate) fn emit_warnings(py: Python<'_>, what: &str, warnings: &[String]) -> PyResult<()> {
  if warnings.is_empty() {
    return Ok(());
  }
  let mod_warn = py.import("warnings")?;
  for w in warnings {
    mod_warn.call_method("warn", (format!("{what}: warning: {w}"),), Some(&[("stacklevel", 3)].into_py_dict(py)?))?;
  }
  Ok(())
}

/// Recursively convert Python values to JSON (dict/list/tuple/str/int/
/// float/bool/None/numpy arrays; nested spec dicts pass through as maps).
fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
  if value.is_none() {
    return Ok(Value::Null);
  }
  if let Ok(b) = value.extract::<bool>() {
    return Ok(Value::Bool(b));
  }
  if let Ok(i) = value.extract::<i64>() {
    return Ok(Value::from(i));
  }
  if let Ok(f) = value.extract::<f64>() {
    return Ok(serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null));
  }
  if let Ok(s) = value.extract::<String>() {
    return Ok(Value::String(s));
  }
  if let Ok(d) = value.cast::<PyDict>() {
    let mut map = serde_json::Map::new();
    for (k, v) in d.iter() {
      map.insert(k.extract::<String>()?, py_to_json(&v)?);
    }
    return Ok(Value::Object(map));
  }
  if let Ok(l) = value.cast::<PyList>() {
    return l.iter().map(|v| py_to_json(&v)).collect::<PyResult<Vec<_>>>().map(Value::Array);
  }
  if let Ok(t) = value.extract::<Vec<Bound<'_, PyAny>>>() {
    return t.iter().map(py_to_json).collect::<PyResult<Vec<_>>>().map(Value::Array);
  }
  // Numpy arrays (params rarely hold them, but accept 1-D float/complex).
  if let Ok(a) = value.extract::<PyReadonlyArray1<f64>>() {
    return Ok(Value::Array(a.as_slice()?.iter().map(|x| serde_json::json!(x)).collect()));
  }
  Err(PyValueError::new_err(format!("cannot convert {value:?} to a material param")))
}

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
  match value {
    Value::Null => Ok(py.None()),
    Value::Bool(b) => Ok(b.into_pyobject(py)?.as_any().clone().unbind()),
    Value::Number(n) => {
      if let Some(i) = n.as_i64() {
        Ok(i.into_pyobject(py)?.into_any().unbind())
      } else {
        Ok(n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any().unbind())
      }
    }
    Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
    Value::Array(a) => {
      let items: Vec<Py<PyAny>> = a.iter().map(|v| json_to_py(py, v)).collect::<PyResult<_>>()?;
      Ok(PyList::new(py, items)?.into_any().unbind())
    }
    Value::Object(m) => {
      let d = PyDict::new(py);
      for (k, v) in m {
        d.set_item(k, json_to_py(py, v)?)?;
      }
      Ok(d.into_any().unbind())
    }
  }
}

fn props_from_dict(d: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, Value>> {
  let mut out = BTreeMap::new();
  for (k, v) in d.iter() {
    out.insert(k.extract::<String>()?, py_to_json(&v)?);
  }
  Ok(out)
}

// ---- Layer ----

#[pyclass(name = "Layer", from_py_object)]
#[derive(Clone)]
pub struct PyLayer {
  inner: Layer,
}

#[pymethods]
impl PyLayer {
  #[new]
  #[pyo3(signature = (thickness=1.0, material_name="", coherent=true, roughness=0.0, rough_type=0, inhomogen=false, inh_delta=0.1, interface=false, interface_thickness=0.0, optimize=true, needle=true, layer_type=1))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    thickness: f64,
    material_name: &str,
    coherent: bool,
    roughness: f64,
    rough_type: i32,
    inhomogen: bool,
    inh_delta: f64,
    interface: bool,
    interface_thickness: f64,
    optimize: bool,
    needle: bool,
    layer_type: i32,
  ) -> PyResult<Self> {
    Ok(Self {
      inner: Layer {
        material: material_name.to_string(),
        thickness,
        coherent,
        inhomogen,
        rough_type: ver(RoughnessType::try_from_i32(rough_type))?,
        inh_delta,
        roughness,
        interface,
        interface_thickness,
        optimize,
        needle,
        layer_type: ver(LayerType::try_from_i32(layer_type))?,
      },
    })
  }

  #[getter]
  fn thickness(&self) -> f64 {
    self.inner.thickness
  }
  #[setter]
  fn set_thickness(&mut self, v: f64) {
    self.inner.thickness = v;
  }
  #[getter]
  fn material(&self) -> &str {
    &self.inner.material
  }
  #[setter]
  fn set_material(&mut self, v: String) {
    self.inner.material = v;
  }
  #[getter]
  fn coherent(&self) -> bool {
    self.inner.coherent
  }
  #[setter]
  fn set_coherent(&mut self, v: bool) {
    self.inner.coherent = v;
  }
  #[getter]
  fn inhomogen(&self) -> bool {
    self.inner.inhomogen
  }
  #[setter]
  fn set_inhomogen(&mut self, v: bool) {
    self.inner.inhomogen = v;
  }
  #[getter]
  fn rough_type(&self) -> i32 {
    self.inner.rough_type as i32
  }
  #[setter]
  fn set_rough_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.rough_type = ver(RoughnessType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn inh_delta(&self) -> f64 {
    self.inner.inh_delta
  }
  #[setter]
  fn set_inh_delta(&mut self, v: f64) {
    self.inner.inh_delta = v;
  }
  #[getter]
  fn roughness(&self) -> f64 {
    self.inner.roughness
  }
  #[setter]
  fn set_roughness(&mut self, v: f64) {
    self.inner.roughness = v;
  }
  #[getter]
  fn interface(&self) -> bool {
    self.inner.interface
  }
  #[setter]
  fn set_interface(&mut self, v: bool) {
    self.inner.interface = v;
  }
  #[getter]
  fn interface_thickness(&self) -> f64 {
    self.inner.interface_thickness
  }
  #[setter]
  fn set_interface_thickness(&mut self, v: f64) {
    self.inner.interface_thickness = v;
  }
  #[getter]
  fn optimize(&self) -> bool {
    self.inner.optimize
  }
  #[setter]
  fn set_optimize(&mut self, v: bool) {
    self.inner.optimize = v;
  }
  #[getter]
  fn needle(&self) -> bool {
    self.inner.needle
  }
  #[setter]
  fn set_needle(&mut self, v: bool) {
    self.inner.needle = v;
  }
  #[getter]
  fn layer_type(&self) -> i32 {
    self.inner.layer_type as i32
  }
  #[setter]
  fn set_layer_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.layer_type = ver(LayerType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn sub_layer_count(&self) -> u32 {
    self.inner.sub_layer_count()
  }
  #[getter]
  fn mask(&self) -> Vec<i32> {
    self.inner.mask().to_vec()
  }

  fn as_pair(&self) -> (String, f64) {
    let (m, t) = self.inner.as_pair();
    (m.to_string(), t)
  }

  fn __call__(&self) -> (String, f64) {
    self.as_pair()
  }

  fn get_properties(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    self.get_state(py)
  }

  fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }

  #[staticmethod]
  fn from_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
    let v = py_to_json(state.as_any())?;
    Ok(Self { inner: ver(serde_json::from_value::<Layer>(v).map_err(|e| e.to_string()))? })
  }

  fn set_properties(&mut self, py: Python<'_>, props: &Bound<'_, PyDict>) -> PyResult<()> {
    let map = props_from_dict(props)?;
    for w in self.inner.set_properties(&map) {
      emit_warnings(py, "Layer", &[w.message])?;
    }
    Ok(())
  }

  fn clone_layer(&self) -> Self {
    self.clone()
  }

  fn __repr__(&self) -> String {
    self.inner.to_string()
  }
}

impl PyLayer {
  pub(crate) fn from_inner(inner: Layer) -> Self {
    Self { inner }
  }
  pub(crate) fn inner_clone(&self) -> Layer {
    self.inner.clone()
  }
}

// ---- Group ----

#[pyclass(name = "Group", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyGroup {
  inner: SharedGroup,
}

#[pymethods]
impl PyGroup {
  #[new]
  #[pyo3(signature = (group_name, thick_factor=1.0, thick_summand=0.0, n_factor=1.0, k_factor=1.0, inh_delta_summand=0.0, roughness_summand=0.0, interface_summand=0.0))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    group_name: &str,
    thick_factor: f64,
    thick_summand: f64,
    n_factor: f64,
    k_factor: f64,
    inh_delta_summand: f64,
    roughness_summand: f64,
    interface_summand: f64,
  ) -> Self {
    let mut g = Group::new(group_name);
    g.thick_factor = thick_factor;
    g.thick_summand = thick_summand;
    g.n_factor = n_factor;
    g.k_factor = k_factor;
    g.inh_delta_summand = inh_delta_summand;
    g.roughness_summand = roughness_summand;
    g.interface_summand = interface_summand;
    Self { inner: Rc::new(RefCell::new(g)) }
  }

  #[getter]
  fn group_name(&self) -> String {
    self.inner.borrow().group_name.clone()
  }
  #[getter]
  fn thick_factor(&self) -> f64 {
    self.inner.borrow().thick_factor
  }
  #[setter]
  fn set_thick_factor(&mut self, v: f64) {
    self.inner.borrow_mut().thick_factor = v;
  }
  #[getter]
  fn thick_summand(&self) -> f64 {
    self.inner.borrow().thick_summand
  }
  #[setter]
  fn set_thick_summand(&mut self, v: f64) {
    self.inner.borrow_mut().thick_summand = v;
  }
  #[getter]
  fn n_factor(&self) -> f64 {
    self.inner.borrow().n_factor
  }
  #[setter]
  fn set_n_factor(&mut self, v: f64) {
    self.inner.borrow_mut().n_factor = v;
  }
  #[getter]
  fn k_factor(&self) -> f64 {
    self.inner.borrow().k_factor
  }
  #[setter]
  fn set_k_factor(&mut self, v: f64) {
    self.inner.borrow_mut().k_factor = v;
  }
  #[getter]
  fn nk_factor(&self) -> Complex64 {
    self.inner.borrow().nk_factor()
  }
  #[getter]
  fn inh_delta_summand(&self) -> f64 {
    self.inner.borrow().inh_delta_summand
  }
  #[setter]
  fn set_inh_delta_summand(&mut self, v: f64) {
    self.inner.borrow_mut().inh_delta_summand = v;
  }
  #[getter]
  fn roughness_summand(&self) -> f64 {
    self.inner.borrow().roughness_summand
  }
  #[setter]
  fn set_roughness_summand(&mut self, v: f64) {
    self.inner.borrow_mut().roughness_summand = v;
  }
  #[getter]
  fn interface_summand(&self) -> f64 {
    self.inner.borrow().interface_summand
  }
  #[setter]
  fn set_interface_summand(&mut self, v: f64) {
    self.inner.borrow_mut().interface_summand = v;
  }
  #[getter]
  fn error_mask(&self) -> Vec<i32> {
    self.inner.borrow().error_mask.to_vec()
  }
  #[setter]
  fn set_error_mask(&mut self, v: Vec<i32>) -> PyResult<()> {
    if v.len() != 6 {
      return Err(PyValueError::new_err("error_mask must have 6 entries."));
    }
    self.inner.borrow_mut().error_mask = [v[0], v[1], v[2], v[3], v[4], v[5]];
    Ok(())
  }
  #[getter]
  fn optimization_mask(&self) -> Vec<i32> {
    self.inner.borrow().optimization_mask.to_vec()
  }
  #[setter]
  fn set_optimization_mask(&mut self, v: Vec<i32>) -> PyResult<()> {
    if v.len() != 7 {
      return Err(PyValueError::new_err("optimization_mask must have 7 entries."));
    }
    self.inner.borrow_mut().optimization_mask = [v[0], v[1], v[2], v[3], v[4], v[5], v[6]];
    Ok(())
  }

  fn error_type_get(&self, channel: &str) -> PyResult<i32> {
    self.error_type(channel)
  }

  fn set_error_type(&mut self, channel: &str, value: i32) -> PyResult<()> {
    let e = ver(navette::structure::ErrorType::try_from_i32(value))?;
    match channel {
      "thickness" => self.inner.borrow_mut().thickness_error_type = e,
      "n" => self.inner.borrow_mut().n_error_type = e,
      "k" => self.inner.borrow_mut().k_error_type = e,
      "inh_delta" => self.inner.borrow_mut().inh_delta_error_type = e,
      "roughness" => self.inner.borrow_mut().roughness_error_type = e,
      "interface" => self.inner.borrow_mut().interface_error_type = e,
      _ => return Err(PyValueError::new_err(format!("unknown error channel '{channel}'"))),
    }
    Ok(())
  }

  fn set_error_params(&mut self, channel: &str, params: &Bound<'_, PyDict>) -> PyResult<()> {
    let v = py_to_json(params.as_any())?;
    let p: navette::structure::ErrorParams =
      ver(serde_json::from_value(v).map_err(|e| format!("set_error_params: malformed params ({e})")))?;
    match channel {
      "thickness" => self.inner.borrow_mut().thickness_error_params = p,
      "n" => self.inner.borrow_mut().n_error_params = p,
      "k" => self.inner.borrow_mut().k_error_params = p,
      "inh_delta" => self.inner.borrow_mut().inh_delta_error_params = p,
      "roughness" => self.inner.borrow_mut().roughness_error_params = p,
      "interface" => self.inner.borrow_mut().interface_error_params = p,
      _ => return Err(PyValueError::new_err(format!("unknown error channel '{channel}'"))),
    }
    Ok(())
  }

  fn params_dict(&self, py: Python<'_>, channel: &str) -> PyResult<Py<PyAny>> {
    self.error_params(py, channel)
  }

  fn error_type(&self, channel: &str) -> PyResult<i32> {
    Ok(match channel {
      "thickness" => self.inner.borrow().thickness_error_type as i32,
      "n" => self.inner.borrow().n_error_type as i32,
      "k" => self.inner.borrow().k_error_type as i32,
      "inh_delta" => self.inner.borrow().inh_delta_error_type as i32,
      "roughness" => self.inner.borrow().roughness_error_type as i32,
      "interface" => self.inner.borrow().interface_error_type as i32,
      _ => return Err(PyValueError::new_err(format!("unknown error channel '{channel}'"))),
    })
  }

  fn error_params(&self, py: Python<'_>, channel: &str) -> PyResult<Py<PyAny>> {
    let p = match channel {
      "thickness" => &self.inner.borrow().thickness_error_params,
      "n" => &self.inner.borrow().n_error_params,
      "k" => &self.inner.borrow().k_error_params,
      "inh_delta" => &self.inner.borrow().inh_delta_error_params,
      "roughness" => &self.inner.borrow().roughness_error_params,
      "interface" => &self.inner.borrow().interface_error_params,
      _ => return Err(PyValueError::new_err(format!("unknown error channel '{channel}'"))),
    };
    let v = serde_json::to_value(p).map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }

  /// Validate factor domains; warnings carry the `warning:` prefix.
  fn validate(&self) -> Vec<String> {
    self.inner.borrow().validate().iter().map(prefixed).collect()
  }

  fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    json_to_py(py, &self.inner.borrow().to_state())
  }

  #[staticmethod]
  fn from_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {
    let v = py_to_json(state.as_any())?;
    Ok(Self { inner: Rc::new(RefCell::new(ver(Group::from_state(&v))?)) })
  }

  fn set_properties(&mut self, py: Python<'_>, props: &Bound<'_, PyDict>) -> PyResult<()> {
    let map = props_from_dict(props)?;
    for w in self.inner.borrow_mut().set_properties(&map) {
      emit_warnings(py, "Group", &[w.message])?;
    }
    Ok(())
  }

  #[getter]
  fn thickness_error_type(&self) -> i32 {
    self.inner.borrow().thickness_error_type as i32
  }
  #[setter]
  fn set_thickness_error_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.borrow_mut().thickness_error_type = ver(navette::structure::ErrorType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn thickness_error_params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(&self.inner.borrow().thickness_error_params)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }
  #[getter]
  fn n_error_type(&self) -> i32 {
    self.inner.borrow().n_error_type as i32
  }
  #[setter]
  fn set_n_error_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.borrow_mut().n_error_type = ver(navette::structure::ErrorType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn n_error_params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(&self.inner.borrow().n_error_params)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }
  #[getter]
  fn k_error_type(&self) -> i32 {
    self.inner.borrow().k_error_type as i32
  }
  #[setter]
  fn set_k_error_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.borrow_mut().k_error_type = ver(navette::structure::ErrorType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn k_error_params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(&self.inner.borrow().k_error_params)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }
  #[getter]
  fn inh_delta_error_type(&self) -> i32 {
    self.inner.borrow().inh_delta_error_type as i32
  }
  #[setter]
  fn set_inh_delta_error_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.borrow_mut().inh_delta_error_type = ver(navette::structure::ErrorType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn inh_delta_error_params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(&self.inner.borrow().inh_delta_error_params)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }
  #[getter]
  fn roughness_error_type(&self) -> i32 {
    self.inner.borrow().roughness_error_type as i32
  }
  #[setter]
  fn set_roughness_error_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.borrow_mut().roughness_error_type = ver(navette::structure::ErrorType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn roughness_error_params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(&self.inner.borrow().roughness_error_params)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }
  #[getter]
  fn interface_error_type(&self) -> i32 {
    self.inner.borrow().interface_error_type as i32
  }
  #[setter]
  fn set_interface_error_type(&mut self, v: i32) -> PyResult<()> {
    self.inner.borrow_mut().interface_error_type = ver(navette::structure::ErrorType::try_from_i32(v))?;
    Ok(())
  }
  #[getter]
  fn interface_error_params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(&self.inner.borrow().interface_error_params)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    json_to_py(py, &v)
  }
  /// Full property dict (settings-store shape; mirrors `Layer`).
  fn get_properties(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    self.get_state(py)
  }

  /// Perturbed thickness (floored at 0) under this group's configured
  /// channel law. `seed=None` = thread RNG (non-reproducible).
  #[pyo3(signature = (value, seed=None))]
  fn thickness_error(&self, value: f64, seed: Option<u64>) -> f64 {
    self.inner.borrow().thickness_error(value, &mut rng_for(seed))
  }

  /// Perturbed grading strength (unfloored, like legacy).
  #[pyo3(signature = (value, seed=None))]
  fn inh_delta_error(&self, value: f64, seed: Option<u64>) -> f64 {
    self.inner.borrow().inh_delta_error(value, &mut rng_for(seed))
  }

  /// Perturbed surface roughness [nm] (floored at 0).
  ///
  /// NOTE vs legacy: the old `thickness` parameter was accepted but
  /// never used (relative terms always scaled to `value`, as here).
  #[pyo3(signature = (value, seed=None))]
  fn sr_roughness_error(&self, value: f64, seed: Option<u64>) -> f64 {
    self.inner.borrow().sr_roughness_error(value, &mut rng_for(seed))
  }

  /// Perturbed interface width [nm] (floored at 0; same legacy note).
  #[pyo3(signature = (value, seed=None))]
  fn interface_error(&self, value: f64, seed: Option<u64>) -> f64 {
    self.inner.borrow().interface_error(value, &mut rng_for(seed))
  }

  /// Perturbed index (n floored at 0, k untouched). Accepts complex or
  /// any `(real, imag)` pair (incl. numpy complex scalars).
  #[pyo3(signature = (nk_value, seed=None))]
  fn nk_error(
    &self,
    py: Python<'_>,
    nk_value: &Bound<'_, PyAny>,
    seed: Option<u64>,
  ) -> PyResult<Py<PyAny>> {
    let pair: (f64, f64) = match (nk_value.getattr("real"), nk_value.getattr("imag")) {
      (Ok(r), Ok(i)) => (r.extract()?, i.extract()?),
      _ => nk_value.extract()?,
    };
    let z = self
      .inner
      .borrow()
      .nk_error(Complex64::new(pair.0, pair.1), &mut rng_for(seed));
    Ok(PyComplex::from_doubles(py, z.re, z.im).into_any().unbind())
  }

  fn clone_group(&self, py: Python<'_>) -> Self {
    let _ = py;
    Self { inner: Rc::new(RefCell::new(self.inner.borrow().clone())) }
  }

  fn __repr__(&self) -> String {
    self.inner.borrow().to_string()
  }
}

impl PyGroup {
  pub(crate) fn from_inner(inner: Group) -> Self {
    Self { inner: Rc::new(RefCell::new(inner)) }
  }
  pub(crate) fn inner_clone(&self) -> Group {
    self.inner.borrow().clone()
  }
}

fn prefixed(issue: &navette::structure::ValidationIssue) -> String {
  if issue.is_error() {
    issue.message.clone()
  } else {
    format!("warning: {}", issue.message)
  }
}



// ---- DictProvider ----

#[pyclass(name = "DictProvider", skip_from_py_object)]
pub struct PyDictProvider {
  inner: DictProvider,
}

#[pymethods]
impl PyDictProvider {
  #[new]
  #[pyo3(signature = (mat_dict, wavelength=None))]
  fn new(mat_dict: &Bound<'_, PyDict>, wavelength: Option<PyReadonlyArray1<f64>>) -> PyResult<Self> {
    let mut entries = HashMap::new();
    for (k, v) in mat_dict.iter() {
      let name = k.extract::<String>()?;
      entries.insert(name.clone(), py_named_entry(&v, &name)?);
    }
    let grid = wavelength.map(|w| w.as_slice().map(|s| s.to_vec())).transpose()?;
    let mut inner = DictProvider::new();
    inner.refresh(entries, grid);
    Ok(Self { inner })
  }

  fn get_nk<'py>(
    &self,
    py: Python<'py>,
    material_name: &str,
    wavelengths: PyReadonlyArray1<f64>,
  ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let wl = wavelengths.as_slice()?;
    let nk = ver(self.inner.nk(material_name, wl))?;
    Ok(PyArray1::from_vec(py, nk))
  }

  fn contains(&self, material_name: &str) -> bool {
    self.inner.contains(material_name)
  }

  #[getter]
  fn grid<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
    self.inner.grid().map(|g| PyArray1::from_vec(py, g.to_vec()))
  }

  /// Export one entry for pour-back: arrays → numpy, specs → spec dict.
  fn export_entry(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
    let found = self.inner.entries_snapshot().into_iter().find(|(n, _)| n == name);
    match found {
      None => Err(PyValueError::new_err(format!("unknown material '{name}'"))),
      Some((_, Entry::Array(nk))) => Ok(PyArray1::from_vec(py, nk).into_any().unbind()),
      Some((_, Entry::Spec(spec))) => {
        let v = serde_json::to_value(spec).map_err(|e| PyValueError::new_err(e.to_string()))?;
        json_to_py(py, &v)
      }
    }
  }

  /// Upsert one entry (arrays, spec dicts). Arrays are length-checked
  /// against the known grid; specs are grid-agnostic until served.
  fn insert(&mut self, name: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
    match py_named_entry(value, &name)? {
      Entry::Array(nk) => ver(self.inner.insert_array(name, nk)),
      Entry::Spec(spec) => {
        self.inner.insert_spec(name, spec);
        Ok(())
      }
    }
  }

  /// Establish the grid when unset (specs need it to resolve).
  fn set_grid(&mut self, wavelength: PyReadonlyArray1<f64>) -> PyResult<()> {
    self.inner.set_grid(wavelength.as_slice()?.to_vec());
    Ok(())
  }

  fn refresh(&mut self, mat_dict: &Bound<'_, PyDict>, wavelength: Option<PyReadonlyArray1<f64>>) -> PyResult<()> {
    let mut entries = HashMap::new();
    for (k, v) in mat_dict.iter() {
      let name = k.extract::<String>()?;
      entries.insert(name.clone(), py_named_entry(&v, &name)?);
    }
    let grid = wavelength.map(|w| w.as_slice().map(|s| s.to_vec())).transpose()?;
    self.inner.refresh(entries, grid);
    Ok(())
  }
}

/// Spec/array library on one mandatory grid, memoized (mirrors
/// `MaterialObjectProvider`). Holds RefCell cache → unsendable.
#[pyclass(name = "SpecProvider", unsendable)]
pub struct PySpecProvider {
  inner: SpecProvider,
}

#[pymethods]
impl PySpecProvider {
  #[new]
  fn new(mat_dict: &Bound<'_, PyDict>, wavelength: PyReadonlyArray1<f64>) -> PyResult<Self> {
    let mut entries = HashMap::new();
    for (k, v) in mat_dict.iter() {
      let name = k.extract::<String>()?;
      entries.insert(name.clone(), py_named_entry(&v, &name)?);
    }
    let grid = wavelength.as_slice()?.to_vec();
    let inner = ver(SpecProvider::new(entries, grid))?;
    Ok(Self { inner })
  }

  fn get_nk<'py>(
    &self,
    py: Python<'py>,
    material_name: &str,
    wavelengths: PyReadonlyArray1<f64>,
  ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    use navette::structure::MaterialProvider as _MP;
    let nk = ver(self.inner.nk(material_name, wavelengths.as_slice()?))?;
    Ok(PyArray1::from_vec(py, nk))
  }

  fn contains(&self, material_name: &str) -> bool {
    use navette::structure::MaterialProvider as _MP;
    self.inner.contains(material_name)
  }

  #[getter]
  fn grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    use navette::structure::MaterialProvider as _MP;
    PyArray1::from_vec(py, self.inner.grid().unwrap_or(&[]).to_vec())
  }

  fn set_wavelength(&mut self, wavelength: PyReadonlyArray1<f64>) -> PyResult<()> {
    self.inner.set_grid(wavelength.as_slice()?.to_vec());
    Ok(())
  }

  #[pyo3(signature = (material_name=None))]
  fn invalidate(&self, material_name: Option<String>) -> PyResult<()> {
    self.inner.invalidate(material_name.as_deref());
    Ok(())
  }

  /// Upsert one entry (arrays, spec dicts, MaterialSpec objects) and
  /// drop its memoized curve. Write-through path for edits.
  fn insert(&mut self, name: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let entry = py_named_entry(value, &name)?;
    self.inner.upsert(name, entry);
    Ok(())
  }

  fn has(&self, material_name: &str) -> bool {
    use navette::structure::MaterialProvider as _MP;
    self.inner.contains(material_name)
  }
}

/// Live n/k curves woven from a native `OpticalWeaver` backend (mirrors
/// `WeaverMaterialProvider` on the native path). Holds RefCell cache →
/// unsendable. Duck-typed (non-native) backends stay on the Python adapter.
#[pyclass(name = "WeaverProvider", unsendable)]
pub struct PyWeaverProvider {
  inner: navette::structure::WeaverProvider<std::sync::Arc<navette::spectralweave::opticalweaver::OpticalWeaver>>,
}

#[pymethods]
impl PyWeaverProvider {
  #[new]
  #[pyo3(signature = (weaver, target_wavelength, key_prefix=0.0, n_label="n", k_label="k", method="linear", robust=false, fh_d=3, strict=false))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    weaver: &crate::spectralweave_optical::PyOpticalWeaver,
    target_wavelength: PyReadonlyArray1<f64>,
    key_prefix: f64,
    n_label: &str,
    k_label: &str,
    method: &str,
    robust: bool,
    fh_d: usize,
    strict: bool,
  ) -> PyResult<Self> {
    let inner = navette::structure::WeaverProvider::new(
      weaver.inner.clone(),
      target_wavelength.as_slice()?.to_vec(),
      key_prefix,
      n_label,
      k_label,
      navette::structure::InterpSettings {
        method: method.to_string(),
        robust,
        fh_d,
      },
      strict,
    );
    Ok(Self { inner })
  }

  fn get_nk<'py>(
    &self,
    py: Python<'py>,
    material_name: &str,
    wavelengths: PyReadonlyArray1<f64>,
  ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    use navette::structure::MaterialProvider as _MP;
    let nk = ver(self.inner.nk(material_name, wavelengths.as_slice()?))?;
    Ok(PyArray1::from_vec(py, nk))
  }

  fn contains(&self, material_name: &str) -> bool {
    use navette::structure::MaterialProvider as _MP;
    self.inner.contains(material_name)
  }

  #[getter]
  fn grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    use navette::structure::MaterialProvider as _MP;
    PyArray1::from_vec(py, self.inner.grid().unwrap_or(&[]).to_vec())
  }

  fn set_target(&mut self, wavelength: PyReadonlyArray1<f64>) -> PyResult<()> {
    self.inner.set_target(wavelength.as_slice()?.to_vec());
    Ok(())
  }

  fn is_exact(&self, material_name: &str) -> bool {
    self.inner.is_exact(material_name)
  }

  #[pyo3(signature = (material_name=None))]
  fn invalidate(&self, material_name: Option<String>) -> PyResult<()> {
    self.inner.invalidate(material_name.as_deref());
    Ok(())
  }

  #[getter]
  fn strict(&self) -> bool {
    self.inner.strict()
  }

  #[setter]
  fn set_strict(&mut self, value: bool) {
    self.inner.set_strict(value);
  }
}

/// Load a program document (JSON text) into native objects (thin over
/// `config::load_program_json_prefixed`). Returns a dict: `name`,
/// `materials` (SpecProvider), `groups` ({name: Group}), `structures`
/// ({label: Structure} with materials attached), `architect?`.
#[pyfunction]
#[pyo3(signature = (text, wavelengths, prefix=None))]
fn load_program(
  py: Python<'_>,
  text: &str,
  wavelengths: PyReadonlyArray1<f64>,
  prefix: Option<String>,
) -> PyResult<Py<PyDict>> {
  let wl = wavelengths.as_slice()?.to_vec();
  let prog = ver(navette::config::load_program_json_prefixed(text, &wl, prefix.as_deref()))?;
  let out = PyDict::new(py);
  out.set_item("name", prog.name)?;
  let pymats = match prog.materials {
    None => py.None().into_any(),
    Some(m) => Py::new(py, PySpecProvider { inner: m })?.into_any(),
  };
  out.set_item("materials", pymats)?;
  let groups = PyDict::new(py);
  for (k, g) in &prog.groups {
    groups.set_item(k, PyGroup::from_inner(g.clone()))?;
  }
  out.set_item("groups", groups)?;
  // Attach materials to each structure for out-of-the-box solving.
  let mats: Py<PyAny> = out
    .get_item("materials")?
    .ok_or_else(|| PyValueError::new_err("load_program: missing materials"))?
    .extract()?;
  let structures = PyDict::new(py);
  for (label, st) in prog.structures {
    let shared: SharedStructure = std::rc::Rc::new(std::cell::RefCell::new(st));
    structures.set_item(
      label,
      PyStructure { inner: shared, materials: Some(mats.clone_ref(py)) },
    )?;
  }
  out.set_item("structures", structures)?;
  match prog.architect {
    None => out.set_item("architect", py.None())?,
    Some(a) => {
      out.set_item("architect", PyArchitect { inner: a, materials: None })?;
    }
  }
  Ok(out.into())
}

/// Python value → shelf entry: complex/float arrays, or spec dicts.
fn py_entry(value: &Bound<'_, PyAny>) -> PyResult<Entry> {
  if let Ok(a) = value.extract::<PyReadonlyArray1<Complex64>>() {
    return Ok(Entry::Array(a.as_slice()?.to_vec()));
  }
  if let Ok(a) = value.extract::<PyReadonlyArray1<f64>>() {
    return Ok(Entry::Array(a.as_slice()?.iter().map(|x| Complex64::new(*x, 0.0)).collect()));
  }
  if let Ok(d) = value.cast::<PyDict>() {
    let v = py_to_json(d.as_any())?;
    let spec: MaterialSpec = ver(serde_json::from_value(v).map_err(|e| e.to_string()))?;
    if !navette::structure::MODELS.contains(&spec.model.as_str()) {
      return Err(PyValueError::new_err(format!("Unknown material model {:?}", spec.model)));
    }
    return Ok(Entry::Spec(spec));
  }
  Err(PyValueError::new_err("provider values must be nk arrays or spec dicts"))
}

// ---- SolverArrays ----

#[pyclass(name = "SolverArrays", skip_from_py_object)]
#[derive(Clone)]
pub struct PySolverArrays {
  inner: SolverArrays,
}

#[pymethods]
impl PySolverArrays {
  #[getter]
  fn thicknesses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, self.inner.thicknesses.clone())
  }
  #[getter]
  fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Complex64>> {
    let n = self.inner.n_wavelengths;
    let rows = self.inner.n_rows();
    PyArray2::from_vec2(py, &self.inner.indices.chunks(n).map(|r| r.to_vec()).collect::<Vec<_>>())
      .unwrap_or_else(|_| PyArray2::zeros(py, [rows, n], false))
  }
  #[getter]
  fn incoherent_flags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
    PyArray1::from_vec(py, self.inner.incoherent.clone())
  }
  #[getter]
  fn rough_types<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
    PyArray1::from_vec(py, self.inner.rough_types.clone())
  }
  #[getter]
  fn rough_vals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, self.inner.rough_vals.clone())
  }
  #[getter]
  fn n_rows(&self) -> usize {
    self.inner.n_rows()
  }
}

/// Solve pre-expanded arrays (thin over `solver::solve_arrays`).
/// Returns `(result_dict, warnings)`; warnings re-emit as Python warnings.
#[pyfunction]
#[pyo3(signature = (sa, wavelengths, angles, requested, radians=false, coherence_mode=0))]
fn solve_arrays_fn(
  py: Python<'_>,
  sa: &PySolverArrays,
  wavelengths: numpy::PyReadonlyArray1<f64>,
  angles: numpy::PyReadonlyArray1<f64>,
  requested: u64,
  radians: bool,
  coherence_mode: i32,
) -> pyo3::PyResult<(pyo3::Py<pyo3::types::PyDict>, Vec<String>)> {
  let (sol, warnings) = navette::smatrix::solver::solve_arrays(
    &sa.inner.indices,
    &sa.inner.thicknesses,
    &sa.inner.incoherent,
    &sa.inner.rough_types,
    &sa.inner.rough_vals,
    wavelengths.as_slice()?,
    angles.as_slice()?,
    radians,
    requested,
    coherence_mode,
  )
  .map_err(pyo3::exceptions::PyValueError::new_err)?;
  Ok((crate::smatrix::solution_to_dict(py, &sol)?, warnings))
}

// ---- Structure ----

impl MaterialProvider for PyDictProvider {
  fn nk(&self, name: &str, wavelengths: &[f64]) -> Result<Vec<Complex64>, String> {
    self.inner.nk(name, wavelengths)
  }
  fn contains(&self, name: &str) -> bool {
    self.inner.contains(name)
  }
  fn grid(&self) -> Option<&[f64]> {
    self.inner.grid()
  }
  fn names(&self) -> Vec<String> {
    self.inner.names().cloned().collect()
  }
}

/// Snapshot any provider-like object into a bound `DictProvider`.
///
/// Bound providers clone; dicts / Python providers resolve `needed`
/// materials (via `_dict` entries or `get_nk` calls); spec dicts AND
/// `MaterialSpec` objects both parse. Grid: `.grid`, else `.wavelength`,
/// else unknown.
fn snapshot_provider(
  _py: Python<'_>,
  obj: &Bound<'_, PyAny>,
  needed: &[String],
) -> PyResult<DictProvider> {
  if let Ok(p) = obj.cast::<PyDictProvider>() {
    return Ok(p.borrow().inner.clone());
  }
  // Native spec/weaver providers snapshot WITHOUT callbacks: evaluate
  // needed materials on the provider grid (unknown names skip —
  // validation reports them, mirroring the Python path).
  use navette::structure::MaterialProvider as _MPSnap;
  if let Ok(p) = obj.cast::<PySpecProvider>() {
    let b = p.borrow();
    let grid = b.inner.grid().unwrap_or(&[]).to_vec();
    let mut entries = HashMap::new();
    for name in needed {
      if !b.inner.contains(name) {
        continue;
      }
      entries.insert(name.clone(), Entry::Array(ver(b.inner.nk(name, &grid))?));
    }
    let mut inner = DictProvider::new();
    inner.refresh(entries, Some(grid));
    return finish_snapshot(inner);
  }
  if let Ok(p) = obj.cast::<PyWeaverProvider>() {
    let b = p.borrow();
    let grid = b.inner.grid().unwrap_or(&[]).to_vec();
    let mut entries = HashMap::new();
    for name in needed {
      if !b.inner.contains(name) {
        continue;
      }
      entries.insert(name.clone(), Entry::Array(ver(b.inner.nk(name, &grid))?));
    }
    let mut inner = DictProvider::new();
    inner.refresh(entries, Some(grid));
    return finish_snapshot(inner);
  }
  // Bare dicts coerce to a gridless shelf (mirrors the auto-wrap).
  if let Ok(d) = obj.cast::<PyDict>() {
    let mut entries = HashMap::new();
    for (k, v) in d.iter() {
      let name = k.extract::<String>()?;
      entries.insert(name.clone(), py_named_entry(&v, &name)?);
    }
    let mut inner = DictProvider::new();
    inner.refresh(entries, None);
    return Ok(inner);
  }
  let grid_item = obj
    .getattr("grid")
    .ok()
    .filter(|g| !g.is_none())
    .or_else(|| obj.getattr("wavelength").ok());
  let grid: Option<Vec<f64>> = match grid_item {
    None => None,
    Some(g) => {
      let arr: PyReadonlyArray1<f64> = g.extract()?;
      Some(arr.as_slice()?.to_vec())
    }
  };
  let has_dict = obj.hasattr("_dict").unwrap_or(false);
  let has_contains = obj.hasattr("contains").unwrap_or(false);
  let mut entries = HashMap::new();
  // _dict shelves snapshot ENTIRELY (collision checks need full contents,
  // mirroring Python); get_nk-only providers resolve `needed` on demand.
  if has_dict {
    let d = obj.getattr("_dict")?;
    let dict = d.cast::<PyDict>()?;
    for (k, v) in dict.iter() {
      let name = k.extract::<String>()?;
      entries.insert(name.clone(), py_named_entry(&v, &name)?);
    }
    let mut inner = DictProvider::new();
    inner.refresh(entries, grid);
    return finish_snapshot(inner);
  }
  for name in needed {
    // Unknown materials are SKIPPED, not raised: validation reports them
    // (mirrors Python, where `contains` gates before `get_nk` ever runs).
    if has_contains {
      let known: bool = obj.call_method("contains", (name,), None)?.extract().unwrap_or(true);
      if !known {
        continue;
      }
    }
    let entry = if has_dict {
      let d = obj.getattr("_dict")?;
      let v = match d.get_item(name) {
        Ok(v) => v,
        Err(_) => continue,
      };
      py_named_entry(&v, name)?
    } else if obj.hasattr("get_nk").unwrap_or(false) {
      let arr = match obj.call_method("get_nk", (name,), None) {
        Ok(a) => a,
        Err(_) => continue,
      };
      Entry::Array(complex_array(&arr, name)?)
    } else {
      return Err(PyValueError::new_err(format!(
        "cannot snapshot provider of type {typename}: need _dict or get_nk",
        typename = obj.get_type().name()?
      )));
    };
    entries.insert(name.clone(), entry);
  }
  let mut inner = DictProvider::new();
  inner.refresh(entries, grid);
  finish_snapshot(inner)
}

/// Length-check arrays against a known grid (fail-closed, like Python).
fn finish_snapshot(inner: DictProvider) -> PyResult<DictProvider> {
  if let Some(g) = inner.grid() {
    let g = g.to_vec();
    for (name, entry) in inner.entries_snapshot() {
      if let Entry::Array(nk) = entry {
        if nk.len() != g.len() {
          return Err(PyValueError::new_err(format!(
            "DictProvider: '{name}' has {} points, provider grid has {}.",
            nk.len(),
            g.len()
          )));
        }
      }
    }
  }
  Ok(inner)
}

/// Python value → shelf entry (arrays, spec dicts, MaterialSpec objects,
/// `.nk` duck-objects).
fn py_named_entry(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Entry> {
  match py_entry(value) {
    Ok(e) => Ok(e),
    Err(_) => {
      // MaterialSpec objects (model/params attributes) and .nk duck-objects.
      if value.hasattr("model").unwrap_or(false) && value.hasattr("params").unwrap_or(false) {
        let model: String = value.getattr("model")?.extract()?;
        let params = value.getattr("params")?;
        let v = py_to_json(&params)?;
        let map: BTreeMap<String, Value> =
          ver(serde_json::from_value(v).map_err(|e| e.to_string()))?;
        return Ok(Entry::Spec(MaterialSpec { model, params: map }));
      }
      if let Ok(nk) = value.getattr("nk") {
        if !nk.is_none() {
          return complex_array(&nk, name).map(Entry::Array);
        }
      }
      Err(PyValueError::new_err(format!(
        "provider value for '{name}' must be an nk array, spec, or .nk object"
      )))
    }
  }
}

fn complex_array(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<Complex64>> {
  if let Ok(a) = value.extract::<PyReadonlyArray1<Complex64>>() {
    return Ok(a.as_slice()?.to_vec());
  }
  if let Ok(a) = value.extract::<PyReadonlyArray1<f64>>() {
    return Ok(a.as_slice()?.iter().map(|x| Complex64::new(*x, 0.0)).collect());
  }
  Err(PyValueError::new_err(format!("material '{name}': expected a complex nk array")))
}

/// Resolve snapshot + simulation grid: stored grid wins; specless shelves
/// fall back to a positional grid (arrays pass through untouched — only
/// the length matters); spec-bearing gridless shelves refuse.
fn resolve_inputs(
  snapshot: &DictProvider,
) -> PyResult<Vec<f64>> {
  if let Some(g) = snapshot.grid() {
    return Ok(g.to_vec());
  }
  let mut len: Option<usize> = None;
  for entry in snapshot.entries_snapshot() {
    match entry {
      (_, Entry::Array(nk)) => len = Some(len.map_or(nk.len(), |n| n.max(nk.len()))),
      (_, Entry::Spec(_)) => {
        return Err(PyValueError::new_err(
          "provider grid unknown and spec entries present: attach the grid (wavelength=) to evaluate specs.",
        ))
      }
    }
  }
  match len {
    Some(n) => Ok((0..n).map(|i| i as f64).collect()),
    None => Err(PyValueError::new_err("provider is empty; nothing to resolve.")),
  }
}

#[pyclass(name = "Structure", unsendable, skip_from_py_object)]
pub struct PyStructure {
  inner: SharedStructure,
  materials: Option<Py<PyAny>>,
}

#[pymethods]
impl PyStructure {
  #[new]
  #[pyo3(signature = (layer_list=None, group_dict=None, materials=None))]
  fn new(
    layer_list: Option<Vec<PyLayer>>,
    group_dict: Option<HashMap<String, PyGroup>>,
    materials: Option<Py<PyAny>>,
  ) -> Self {
    Self {
      inner: Rc::new(RefCell::new(Structure::with_shared(
        layer_list.unwrap_or_default().into_iter().map(|l| l.inner).collect(),
        group_dict.unwrap_or_default().into_iter().map(|(k, g)| (k, g.inner)).collect(),
      ))),
      materials,
    }
  }

  #[getter]
  fn get_materials(&self, py: Python<'_>) -> Option<Py<PyAny>> {
    self.materials.as_ref().map(|m| m.clone_ref(py))
  }

  pub(crate) fn set_materials_silent(&mut self, value: Py<PyAny>) {
    self.materials = Some(value);
  }

  #[setter]
  fn set_materials(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
    if let Some(old) = &self.materials {
      if !old.bind(py).is(value.bind(py)) {
        emit_warnings(
          py,
          "Navette_Structure",
          &["overwriting the material provider; same names may now resolve differently.".to_string()],
        )?;
      }
    }
    self.materials = Some(value);
    Ok(())
  }

  #[getter]
  fn group_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    for (k, g) in &self.inner.borrow().groups {
      d.set_item(k, PyGroup { inner: g.clone() })?;
    }
    Ok(d.into_any().unbind())
  }

  #[getter]
  fn layer_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let items: Vec<PyLayer> =
      self.inner.borrow().layers.iter().cloned().map(PyLayer::from_inner).collect();
    Ok(PyList::new(py, items)?.into_any().unbind())
  }

  fn __len__(&self) -> usize {
    self.inner.borrow().layers.len()
  }

  fn __getitem__(&self, idx: isize) -> PyResult<PyLayer> {
    let borrowed = self.inner.borrow();
    let n = borrowed.layers.len() as isize;
    let i = if idx < 0 { n + idx } else { idx };
    if i < 0 || i >= n {
      return Err(PyValueError::new_err("layer index out of range"));
    }
    Ok(PyLayer::from_inner(borrowed.layers[i as usize].clone()))
  }

  fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let items: Vec<PyLayer> =
      self.inner.borrow().layers.iter().cloned().map(PyLayer::from_inner).collect();
    Ok(PyList::new(py, items)?.into_any().call_method0("__iter__")?.unbind())
  }



  fn validate(&self, py: Python<'_>) -> PyResult<Vec<String>> {
    // Providerless validation skips material coverage + dry run (mirrors Python).
    let snapshot = match &self.materials {
      None => None,
      Some(m) => {
        let borrowed = self.inner.borrow();
        let needed: Vec<String> = borrowed.layers.iter().map(|l| l.material.clone()).collect();
        drop(borrowed);
        Some(snapshot_provider(py, m.bind(py), &needed)?)
      }
    };
    let borrowed = self.inner.borrow();
    let p = snapshot.as_ref().map(|s| s as &dyn navette::structure::MaterialProvider);
    Ok(borrowed.validate(p).iter().map(prefixed).collect())
  }

  fn solver_inputs(&self, py: Python<'_>) -> PyResult<PySolverArrays> {
    let (snapshot, wl) = carried_snapshot_of(&self.inner, &self.materials, py)?;
    let (groups, seq) = {
      let borrowed = self.inner.borrow();
      let warnings = ver(borrowed.gate(&borrowed.validate(Some(&snapshot)), "Navette_Structure"))?;
      emit_warnings(py, "Navette_Structure", &warnings)?;
      let seq: Vec<(Layer, bool)> = borrowed.layers.iter().cloned().map(|l| (l, false)).collect();
      let groups: HashMap<String, Group> =
        borrowed.groups.iter().map(|(k, g)| (k.clone(), g.borrow().clone())).collect();
      (groups, seq)
    };
    let (sa, _) = ver(py.detach(|| expand(&seq, &snapshot, &wl, &groups, ExpandOptions::deterministic())))?;
    Ok(PySolverArrays { inner: sa })
  }

  #[pyo3(signature = (rng=None))]
  fn error_inputs(&self, py: Python<'_>, rng: Option<u64>) -> PyResult<PySolverArrays> {
    let (snapshot, wl) = carried_snapshot_of(&self.inner, &self.materials, py)?;
    let (groups, seq) = {
      let borrowed = self.inner.borrow();
      let warnings = ver(borrowed.gate(&borrowed.validate(Some(&snapshot)), "Navette_Structure"))?;
      emit_warnings(py, "Navette_Structure", &warnings)?;
      let seq: Vec<(Layer, bool)> = borrowed.layers.iter().cloned().map(|l| (l, false)).collect();
      let groups: HashMap<String, Group> =
        borrowed.groups.iter().map(|(k, g)| (k.clone(), g.borrow().clone())).collect();
      (groups, seq)
    };
    let (sa, _) = ver(py.detach(|| {
      expand(&seq, &snapshot, &wl, &groups, ExpandOptions { apply_errors: true, seed: rng })
    }))?;
    Ok(PySolverArrays { inner: sa })
  }

  fn total_sub_layers(&self, py: Python<'_>) -> PyResult<usize> {
    let (snapshot, wl) = match &self.materials {
      None => return Ok(self.inner.borrow().total_sub_layers(None, &[])),
      Some(m) => {
        let borrowed = self.inner.borrow();
        let needed: Vec<String> = borrowed.layers.iter().map(|l| l.material.clone()).collect();
        drop(borrowed);
        let snapshot = snapshot_provider(py, m.bind(py), &needed)?;
        let wl = resolve_inputs(&snapshot).unwrap_or_default();
        (snapshot, wl)
      }
    };
    Ok(self.inner.borrow().total_sub_layers(Some(&snapshot), &wl))
  }

  fn bake_films(&self) -> PyResult<usize> {
    ver(self.inner.borrow_mut().bake_films())
  }

  /// Bake n/k into fresh Table specs: returns (old->new, target shelf).
  /// The thin wrapper pours the new specs back into the carried Python
  /// provider (it knows that provider's concrete type).
  fn bake_materials(
    &self,
    py: Python<'_>,
    wavelengths: PyReadonlyArray1<f64>,
  ) -> PyResult<(HashMap<String, String>, PyDictProvider)> {
    let wl = wavelengths.as_slice()?.to_vec();
    if wl.is_empty() {
      return Err(PyValueError::new_err("bake_materials: wavelengths must be non-empty."));
    }
    let snapshot = match &self.materials {
      None => return Err(PyValueError::new_err("bake_materials: no material provider set.")),
      Some(m) => {
        let borrowed = self.inner.borrow();
        let needed: Vec<String> = borrowed.layers.iter().map(|l| l.material.clone()).collect();
        drop(borrowed);
        snapshot_provider(py, m.bind(py), &needed)?
      }
    };
    let mut target = DictProvider::new();
    let mapping =
      ver(self.inner.borrow_mut().bake_materials(&wl, &snapshot, &mut target))?;
    Ok((mapping.into_iter().collect(), PyDictProvider { inner: target }))
  }

  fn replace_material(&self, old: &str, new: &str) -> usize {
    self.inner.borrow_mut().replace_material(old, new)
  }

  fn optimization_entries(&self) -> Vec<usize> {
    self.inner.borrow().optimization_entries()
  }

  fn set_optimization_mask(&self, group_name: &str, mask: Vec<i32>) -> PyResult<()> {
    if mask.len() != 7 {
      return Err(PyValueError::new_err("optimization_mask must have 7 entries."));
    }
    ver(self.inner.borrow_mut().set_optimization_mask(
      group_name,
      [mask[0], mask[1], mask[2], mask[3], mask[4], mask[5], mask[6]],
    ))
  }

  fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    json_to_py(py, &self.inner.borrow().to_state())
  }

  #[staticmethod]
  #[pyo3(signature = (state, materials=None))]
  fn from_state(state: &Bound<'_, PyDict>, materials: Option<Py<PyAny>>) -> PyResult<Self> {
    let v = py_to_json(state.as_any())?;
    Ok(Self {
      inner: Rc::new(RefCell::new(ver(Structure::from_state(&v))?)),
      materials,
    })
  }

  fn clone(&self, py: Python<'_>) -> Self {
    Self {
      inner: Rc::new(RefCell::new(self.inner.borrow().clone())),
      materials: self.materials.as_ref().map(|m| m.clone_ref(py)),
    }
  }

  /// Core identity (wrapper shell matching across clone boundaries).
  fn core_id(&self) -> usize {
    self.core_id_impl()
  }

  /// Append a layer (wrapper merge path; RefCell interior mutability).
  fn append_layer(&self, layer: &PyLayer) -> PyResult<()> {
    self.inner.borrow_mut().layers.push(layer.inner_clone());
    Ok(())
  }

  /// Insert a layer at `index` (list semantics: negatives count back,
  /// out-of-range clamps — mirrors the legacy `list.insert`).
  fn insert_layer(&self, index: isize, layer: &PyLayer) -> PyResult<()> {
    let mut core = self.inner.borrow_mut();
    let len = core.layers.len() as isize;
    let at = (if index < 0 { len + index } else { index }).clamp(0, len) as usize;
    core.layers.insert(at, layer.inner_clone());
    Ok(())
  }

  /// Remove + return the layer at `index` (negatives ok; IndexError).
  fn remove_layer(&self, index: isize) -> PyResult<PyLayer> {
    let mut core = self.inner.borrow_mut();
    let len = core.layers.len() as isize;
    let at = if index < 0 { len + index } else { index };
    if at < 0 || at >= len {
      return Err(PyIndexError::new_err(format!(
        "remove_layer: index {index} out of bounds ({len} layers)"
      )));
    }
    Ok(PyLayer::from_inner(core.layers.remove(at as usize)))
  }

  /// Replace the layer at `index` (negatives ok; IndexError).
  fn replace_layer(&self, index: isize, layer: &PyLayer) -> PyResult<()> {
    let mut core = self.inner.borrow_mut();
    let len = core.layers.len() as isize;
    let at = if index < 0 { len + index } else { index };
    if at < 0 || at >= len {
      return Err(PyIndexError::new_err(format!(
        "replace_layer: index {index} out of bounds ({len} layers)"
      )));
    }
    core.layers[at as usize] = layer.inner_clone();
    Ok(())
  }

  /// Insert/replace a group (wrapper merge path).
  fn insert_group(&self, name: String, group: &PyGroup) -> PyResult<()> {
    self.inner.borrow_mut().groups.insert(name, group.inner.clone());
    Ok(())
  }

  /// Independent group copy (default identity when unlisted).
  fn get_group_for_material(&self, material_name: &str) -> PyGroup {
    match self.inner.borrow().groups.get(material_name) {
      Some(g) => PyGroup { inner: Rc::new(RefCell::new(g.borrow().clone())) },
      None => PyGroup::from_inner(Group::new("_default_")),
    }
  }
}

impl PyStructure {
  pub(crate) fn shared(&self) -> SharedStructure {
    self.inner.clone()
  }

  /// Core identity (wrapper shell matching across clone boundaries).
  fn core_id_impl(&self) -> usize {
    Rc::as_ptr(&self.inner) as usize
  }
}

/// Materials a structure resolves from (provider, dict, or None).
fn carried_snapshot_of(
  inner: &SharedStructure,
  materials: &Option<Py<PyAny>>,
  py: Python<'_>,
) -> PyResult<(DictProvider, Vec<f64>)> {
  let borrowed = inner.borrow();
  let needed: Vec<String> = borrowed.layers.iter().map(|l| l.material.clone()).collect();
  drop(borrowed);
  match materials {
    None => Err(PyValueError::new_err("No material provider set.")),
    Some(m) => {
      let snapshot = snapshot_provider(py, m.bind(py), &needed)?;
      let wl = resolve_inputs(&snapshot)?;
      Ok((snapshot, wl))
    }
  }
}


// ---- Architect ----

#[pyclass(name = "Architect", unsendable, skip_from_py_object)]
pub struct PyArchitect {
  inner: Architect,
  materials: Option<Py<PyAny>>,
}

#[pymethods]
impl PyArchitect {
  #[new]
  #[pyo3(signature = (materials=None))]
  fn new(materials: Option<Py<PyAny>>) -> Self {
    Self { inner: Architect::new(), materials }
  }

  #[getter]
  fn get_materials(&self, py: Python<'_>) -> Option<Py<PyAny>> {
    self.materials.as_ref().map(|m| m.clone_ref(py))
  }

  #[setter]
  fn set_materials(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
    if let Some(old) = &self.materials {
      if !old.bind(py).is(value.bind(py)) {
        emit_warnings(
          py,
          "Navette_Architect",
          &["overwriting the material provider; same names may now resolve differently.".to_string()],
        )?;
      }
    }
    self.materials = Some(value);
    Ok(())
  }

  #[pyo3(signature = (structure, inverted=false, repeat=1, label="", kind=0))]
  fn add_structure(
    &mut self,
    py: Python<'_>,
    structure: Bound<'_, PyStructure>,
    inverted: bool,
    repeat: usize,
    label: &str,
    kind: i32,
  ) -> PyResult<()> {
    ver(self.inner.add_shared(
      structure.borrow().shared(),
      inverted,
      repeat,
      label,
      ver(BlockKind::try_from_i32(kind))?,
    ))?;
    // The architect's provider overwrites the shell's (mirrors Python).
    if let Some(m) = &self.materials {
      structure.borrow_mut().set_materials_silent(m.clone_ref(py));
    }
    Ok(())
  }

  fn clone_structure(&mut self, index: usize) -> PyResult<()> {
    ver(self.inner.clone_structure(index))
  }

  /// Snapshot the architect-carried provider (required for solving).
  /// NOTE: real helper is free-standing (`architect_snapshot_of`).
  #[allow(dead_code)]
  fn carried_snapshot_unused(&self) {}


  fn validate(&self, py: Python<'_>) -> PyResult<Vec<String>> {
    let snapshot = match &self.materials {
      None => None,
      Some(m) => {
        let mut needed: Vec<String> = Vec::new();
        for shared in self.inner.unique_structures() {
          for layer in &shared.borrow().layers {
            if !needed.iter().any(|x| x == &layer.material) {
              needed.push(layer.material.clone());
            }
          }
        }
        Some(snapshot_provider(py, m.bind(py), &needed)?)
      }
    };
    let p = snapshot.as_ref().map(|s| s as &dyn navette::structure::MaterialProvider);
    Ok(self.inner.validate(p).iter().map(prefixed).collect())
  }

  fn solver_inputs(&self, py: Python<'_>) -> PyResult<PySolverArrays> {
    let (snapshot, wl) = architect_snapshot_of(&self.inner, &self.materials, py)?;
    let warnings = gate_arch_issues(&self.inner.validate(Some(&snapshot)))?;
    emit_warnings(py, "Navette_Architect", &warnings)?;
    let seq = self.inner.iter_entries();
    let merged = self.inner.merged_groups().map_err(|e| PyValueError::new_err(format!("Navette_Architect invalid:
{e}")))?;
    let (sa, _) = ver(py.detach(|| expand(&seq, &snapshot, &wl, &merged, ExpandOptions::deterministic())))?;
    Ok(PySolverArrays { inner: sa })
  }

  #[pyo3(signature = (rng=None))]
  fn error_inputs(&self, py: Python<'_>, rng: Option<u64>) -> PyResult<PySolverArrays> {
    let (snapshot, wl) = architect_snapshot_of(&self.inner, &self.materials, py)?;
    let warnings = gate_arch_issues(&self.inner.validate(Some(&snapshot)))?;
    emit_warnings(py, "Navette_Architect", &warnings)?;
    let seq = self.inner.iter_entries();
    let merged = self.inner.merged_groups().map_err(|e| PyValueError::new_err(format!("Navette_Architect invalid:
{e}")))?;
    let (sa, _) = ver(py.detach(|| {
      expand(&seq, &snapshot, &wl, &merged, ExpandOptions { apply_errors: true, seed: rng })
    }))?;
    Ok(PySolverArrays { inner: sa })
  }

  fn map_global_index_to_layer(&self, global_idx: usize) -> PyResult<(usize, usize)> {
    ver(self.inner.map_global(global_idx))
  }

  fn map_solver_index_to_layer(&self, py: Python<'_>, solver_idx: usize) -> PyResult<(usize, usize)> {
    let (snapshot, wl) = architect_snapshot_of(&self.inner, &self.materials, py)?;
    ver(self.inner.map_solver(&snapshot, &wl, solver_idx))
  }

  fn get_layer_at_global(&self, global_idx: usize) -> PyResult<PyLayer> {
    ver(self.inner.layer_at_global(global_idx)).map(PyLayer::from_inner)
  }

  fn insert_layer_at_global(&self, global_idx: usize, layer: &PyLayer) -> PyResult<()> {
    ver(self.inner.insert_at_global(global_idx, layer.inner.clone()))
  }

  #[pyo3(signature = (global_idx, split_ratio=0.5))]
  fn split_layer_at_global(&self, global_idx: usize, split_ratio: f64) -> PyResult<()> {
    ver(self.inner.split_at_global(global_idx, split_ratio))
  }

  fn duplicate_layer_at_global(&self, global_idx: usize) -> PyResult<()> {
    ver(self.inner.duplicate_at_global(global_idx))
  }

  fn remove_layer_at_global(&self, global_idx: usize) -> PyResult<()> {
    ver(self.inner.remove_at_global(global_idx))
  }

  #[pyo3(signature = (min_thickness=0.001))]
  fn prune_thin_layers(&self, min_thickness: f64) -> usize {
    self.inner.prune_thin_layers(min_thickness)
  }

  /// Optimization-eligible layers as live-index entries is `optimization_entries`;
  /// this returns the layer objects (snapshot clones, like Python refs for reads).
  fn get_optimization_parameters(&self, py: Python<'_>) -> PyResult<Vec<PyLayer>> {
    let _ = py;
    Ok(
      self
        .inner
        .optimization_entries()
        .iter()
        .filter_map(|(bi, local)| {
          self.inner.blocks.get(*bi)?.structure.borrow().layers.get(*local).cloned()
        })
        .map(PyLayer::from_inner)
        .collect(),
    )
  }

  fn optimization_entries(&self) -> Vec<(usize, usize)> {
    self.inner.optimization_entries()
  }

  fn set_optimization_mask(&self, group_name: &str, mask: Vec<i32>) -> PyResult<()> {
    if mask.len() != 7 {
      return Err(PyValueError::new_err("optimization_mask must have 7 entries."));
    }
    ver(self.inner.set_optimization_mask(
      group_name,
      [mask[0], mask[1], mask[2], mask[3], mask[4], mask[5], mask[6]],
    ))
  }

  fn bake_films(&self) -> PyResult<usize> {
    ver(self.inner.bake_films())
  }

  /// Bake n/k into fresh Table specs: returns (old->new, target shelf).
  fn bake_materials(
    &self,
    py: Python<'_>,
    wavelengths: PyReadonlyArray1<f64>,
  ) -> PyResult<(HashMap<String, String>, PyDictProvider)> {
    let wl = wavelengths.as_slice()?.to_vec();
    if wl.is_empty() {
      return Err(PyValueError::new_err("bake_materials: wavelengths must be non-empty."));
    }
    let (snapshot, _) = architect_snapshot_of(&self.inner, &self.materials, py)?;
    let mut target = DictProvider::new();
    let merged = ver(self.inner.bake_materials(&wl, &snapshot, &mut target))?;
    Ok((merged.into_iter().collect(), PyDictProvider { inner: target }))
  }

  fn total_sub_layers(&self, py: Python<'_>) -> PyResult<usize> {
    match &self.materials {
      None => Ok(self.inner.total_sub_layers(None, &[])),
      Some(m) => {
        let mut needed: Vec<String> = Vec::new();
        for shared in self.inner.unique_structures() {
          for layer in &shared.borrow().layers {
            if !needed.iter().any(|x| x == &layer.material) {
              needed.push(layer.material.clone());
            }
          }
        }
        let snapshot = snapshot_provider(py, m.bind(py), &needed)?;
        let wl = resolve_inputs(&snapshot).unwrap_or_default();
        Ok(self.inner.total_sub_layers(Some(&snapshot), &wl))
      }
    }
  }


  fn global_layer_count(&self) -> usize {
    self.inner.global_layer_count()
  }

  fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    json_to_py(py, &self.inner.to_state())
  }

  #[staticmethod]
  #[pyo3(signature = (state, materials=None))]
  fn from_state(state: &Bound<'_, PyDict>, materials: Option<Py<PyAny>>) -> PyResult<Self> {
    let v = py_to_json(state.as_any())?;
    Ok(Self { inner: ver(Architect::from_state(&v))?, materials })
  }

  fn __len__(&self) -> usize {
    self.inner.blocks.len()
  }

  /// Per-block views for the thin wrapper: (core id, inverted, repeat,
  /// label, kind). Shells are matched by core id wrapper-side.
  fn blocks_info(&self) -> Vec<(usize, bool, usize, String, i32)> {
    self
      .inner
      .blocks
      .iter()
      .map(|b| {
        (
          Rc::as_ptr(&b.structure) as usize,
          b.inverted,
          b.repeat_count,
          b.label.clone(),
          b.kind as i32,
        )
      })
      .collect()
  }

  /// Core identity of one block's structure (wrapper shell matching).
  fn block_core_id(&self, index: usize) -> PyResult<usize> {
    self
      .inner
      .blocks
      .get(index)
      .map(|b| Rc::as_ptr(&b.structure) as usize)
      .ok_or_else(|| PyValueError::new_err("Block index out of range"))
  }

  /// Unique structures as fresh shells sharing the blocks' cores.
  fn unique_structures(&self) -> Vec<PyStructure> {
    self
      .inner
      .unique_structures()
      .into_iter()
      .map(|rc| PyStructure { inner: rc, materials: None })
      .collect()
  }
}



fn gate_arch_issues(issues: &[navette::structure::ValidationIssue]) -> PyResult<Vec<String>> {
  let (errors, warnings): (Vec<_>, Vec<_>) = issues.iter().partition(|i| i.is_error());
  if !errors.is_empty() {
    return Err(PyValueError::new_err(format!(
      "Navette_Architect invalid:
{}",
      errors.iter().map(|e| e.message.as_str()).collect::<Vec<_>>().join("
")
    )));
  }
  Ok(warnings.iter().map(|w| w.message.clone()).collect())
}

/// Needed materials across unique structures + snapshot + grid.
fn architect_snapshot_of(
  inner: &Architect,
  materials: &Option<Py<PyAny>>,
  py: Python<'_>,
) -> PyResult<(DictProvider, Vec<f64>)> {
  let mut needed: Vec<String> = Vec::new();
  for shared in inner.unique_structures() {
    for layer in &shared.borrow().layers {
      if !needed.iter().any(|x| x == &layer.material) {
        needed.push(layer.material.clone());
      }
    }
  }
  match materials {
    None => Err(PyValueError::new_err("No material provider set.")),
    Some(m) => {
      let snapshot = snapshot_provider(py, m.bind(py), &needed)?;
      let wl = resolve_inputs(&snapshot)?;
      Ok((snapshot, wl))
    }
  }
}

/// One scalar error draw (draw core behind every `*_error` helper).
///
/// `params` is the 8-key error-params dict; `seed = None` draws from the
/// thread RNG. Returns the perturbed value (unfloored — floors live in
/// the typed helpers, mirroring the core).
#[pyfunction]
#[pyo3(name = "apply_error", signature = (value, error_type, params, seed=None))]
fn apply_error_fn(
  value: f64,
  error_type: i32,
  params: &Bound<'_, PyDict>,
  seed: Option<u64>,
) -> PyResult<f64> {
  use navette::structure::{ErrorParams, ErrorType};
  use rand::SeedableRng;
  let et = ver(ErrorType::try_from_i32(error_type))?;
  let v = py_to_json(params.as_any())?;
  let p: ErrorParams =
    ver(serde_json::from_value(v).map_err(|e| format!("apply_error: malformed params ({e})")))?;
  let mut seeded;
  let mut thread;
  let rng: &mut dyn rand::RngCore = match seed {
    Some(s) => {
      seeded = rand::rngs::StdRng::seed_from_u64(s);
      &mut seeded
    }
    None => {
      thread = rand::rng();
      &mut thread
    }
  };
  Ok(Group::apply_error(value, et, &p, rng))
}

/// Seed-selected RNG for the per-channel draw helpers (`None` = thread RNG).
enum SeedRng {
  Seeded(rand::rngs::StdRng),
  Thread(rand::rngs::ThreadRng),
}

impl rand::RngCore for SeedRng {
  fn next_u32(&mut self) -> u32 {
    match self {
      SeedRng::Seeded(r) => r.next_u32(),
      SeedRng::Thread(r) => r.next_u32(),
    }
  }
  fn next_u64(&mut self) -> u64 {
    match self {
      SeedRng::Seeded(r) => r.next_u64(),
      SeedRng::Thread(r) => r.next_u64(),
    }
  }
  fn fill_bytes(&mut self, dest: &mut [u8]) {
    match self {
      SeedRng::Seeded(r) => r.fill_bytes(dest),
      SeedRng::Thread(r) => r.fill_bytes(dest),
    }
  }
}

fn rng_for(seed: Option<u64>) -> SeedRng {
  use rand::SeedableRng;
  match seed {
    Some(s) => SeedRng::Seeded(rand::rngs::StdRng::seed_from_u64(s)),
    None => SeedRng::Thread(rand::rng()),
  }
}

/// Native thin-film stack model: Layer/Group providers, two-phase
/// expansion into SolverArrays, and the Structure/Architect composers.
/// First-class home of the model — the Python `navette.structure` package
/// re-exports these classes and adds provider plumbing around them.
#[pymodule]
pub fn _structure(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(apply_error_fn, m)?)?;
  m.add_function(wrap_pyfunction!(solve_arrays_fn, m)?)?;
  m.add_class::<PyLayer>()?;
  m.add_class::<PyGroup>()?;
  m.add_class::<PyDictProvider>()?;
  m.add_class::<PySpecProvider>()?;
  m.add_function(wrap_pyfunction!(load_program, m)?)?;
  m.add_class::<PyWeaverProvider>()?;
  m.add_class::<PySolverArrays>()?;
  m.add_class::<PyStructure>()?;
  m.add_class::<PyArchitect>()?;
  Ok(())
}
