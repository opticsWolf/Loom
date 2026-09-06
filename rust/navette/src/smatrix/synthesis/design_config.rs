//! synthesis::design_config — typed config → native `(DesignStack, ContrastMap)`.
//!
//! Rust-first assembly for synthesis designs: a [`DesignRequest`] (the
//! JSON mirror of the Python pydantic configs) evaluates materials on
//! the grid, builds authoring films + groups, and calls
//! [`DesignStack::from_design`] exactly once. Python's
//! `pipeline_from_config` is a thin wrapper (validate → dump JSON →
//! call [`build_design`]); Rust consumers run standalone with no Python.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use num_complex::Complex64;
use serde::Deserialize;
use serde_json::Value;

use super::cycle::ContrastMap;
use super::structure::{DesignStack, LayerSpec};
use crate::structure::{
  ErrorParams, ErrorType, Group, Layer, LayerType, MaterialSpec, RoughnessType,
};

// ---------------------------------------------------------------------------
// Request schema (JSON mirror of the pydantic configs)
// ---------------------------------------------------------------------------

fn d_true() -> bool {
  true
}
fn d_inh_delta() -> f64 {
  0.1
}
fn d_one() -> f64 {
  1.0
}
fn d_layer_type() -> i32 {
  1
}
fn d_air() -> String {
  "air".to_string()
}
fn d_sub() -> String {
  "sub".to_string()
}

/// `{wavelengths: [...], values: [...]}` table grid.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TabData {
  #[serde(default)]
  pub wavelengths: Vec<f64>,
  #[serde(default)]
  pub values: Vec<f64>,
}

/// One material library entry.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialDef {
  pub name: String,
  pub code: Option<String>,
  pub model: String,
  #[serde(default)]
  pub params: BTreeMap<String, Value>,
  #[serde(default)]
  pub n_data: Option<TabData>,
  #[serde(default)]
  pub k_data: Option<TabData>,
}

/// One stack row: film (`layer_type` 1), ambient (0), substrate (2).
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRow {
  pub material_code: String,
  pub thickness_nm: f64,
  #[serde(default = "d_true")]
  pub coherent: bool,
  #[serde(default)]
  pub roughness_nm: f64,
  #[serde(default)]
  pub rough_type: i32,
  #[serde(default)]
  pub inhomogen: bool,
  #[serde(default = "d_inh_delta")]
  pub inh_delta: f64,
  #[serde(default)]
  pub interface: bool,
  #[serde(default)]
  pub interface_thickness_nm: f64,
  #[serde(default = "d_true")]
  pub optimize: bool,
  #[serde(default = "d_true")]
  pub needle: bool,
  #[serde(default = "d_layer_type")]
  pub layer_type: i32,
}

/// One fabrication-error channel (mirrors the Python `ErrorParams` model).
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorParamsCfg {

  #[serde(default)]
  pub abs_mean_delta_g: f64,
  #[serde(default = "d_abs_std")]
  pub abs_std_dev: f64,
  #[serde(default)]
  pub rel_mean_delta_g: f64,
  #[serde(default = "d_one")]
  pub rel_std_dev: f64,
  #[serde(default)]
  pub abs_mean_delta_h: f64,
  #[serde(default = "d_abs_std")]
  pub abs_variance: f64,
  #[serde(default)]
  pub rel_mean_delta_h: f64,
  #[serde(default)]
  pub rel_variance: f64,
}

fn d_abs_std() -> f64 {
  0.01
}

impl Default for ErrorParamsCfg {
  /// Mirrors the Python `ErrorParams` model defaults.
  fn default() -> Self {
    Self {
      abs_mean_delta_g: 0.0,
      abs_std_dev: 0.01,
      rel_mean_delta_g: 0.0,
      rel_std_dev: 1.0,
      abs_mean_delta_h: 0.0,
      abs_variance: 0.01,
      rel_mean_delta_h: 0.0,
      rel_variance: 0.0,
    }
  }
}

impl ErrorParamsCfg {
  fn build(&self) -> ErrorParams {
    ErrorParams {
      abs_mean_delta_g: self.abs_mean_delta_g,
      abs_std_dev: self.abs_std_dev,
      rel_mean_delta_g: self.rel_mean_delta_g,
      rel_std_dev: self.rel_std_dev,
      abs_mean_delta_h: self.abs_mean_delta_h,
      abs_variance: self.abs_variance,
      rel_mean_delta_h: self.rel_mean_delta_h,
      rel_variance: self.rel_variance,
    }
  }
}

/// One group entry (mirrors the Python `GroupConfig` model).
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct GroupRow {
  pub name: String,
  #[serde(default = "d_one")]
  pub thick_factor: f64,
  #[serde(default)]
  pub thick_summand: f64,
  #[serde(default = "d_one")]
  pub n_factor: f64,
  #[serde(default = "d_one")]
  pub k_factor: f64,
  #[serde(default)]
  pub inh_delta_summand: f64,
  #[serde(default)]
  pub roughness_summand: f64,
  #[serde(default)]
  pub interface_summand: f64,
  #[serde(default = "d_mask6")]
  pub error_mask: Vec<i32>,
  #[serde(default = "d_mask7")]
  pub optimization_mask: Vec<i32>,
  #[serde(default)]
  pub thickness_error_type: i32,
  #[serde(default)]
  pub n_error_type: i32,
  #[serde(default)]
  pub k_error_type: i32,
  #[serde(default)]
  pub inh_delta_error_type: i32,
  #[serde(default)]
  pub roughness_error_type: i32,
  #[serde(default)]
  pub interface_error_type: i32,
  #[serde(default)]
  pub thickness_error_params: ErrorParamsCfg,
  #[serde(default)]
  pub inh_delta_error_params: ErrorParamsCfg,
  #[serde(default)]
  pub roughness_error_params: ErrorParamsCfg,
  #[serde(default)]
  pub interface_error_params: ErrorParamsCfg,
  #[serde(default)]
  pub n_error_params: ErrorParamsCfg,
  #[serde(default)]
  pub k_error_params: ErrorParamsCfg,
}

fn d_mask6() -> Vec<i32> {
  vec![0; 6]
}
fn d_mask7() -> Vec<i32> {
  vec![1; 7]
}

/// Named authoring structure: layers + own groups.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct StructureCfg {
  pub label: String,
  pub layers: Vec<LayerRow>,
  #[serde(default)]
  pub groups: Vec<GroupRow>,
}

/// Full design request: structure + material library + options.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesignRequest {
  pub structure: StructureCfg,
  pub library: Vec<MaterialDef>,
  #[serde(default)]
  pub contrast: BTreeMap<String, String>,
  #[serde(default)]
  pub film_flags: BTreeMap<String, Value>,
  #[serde(default)]
  pub per_film_flags: BTreeMap<String, BTreeMap<String, Value>>,
  #[serde(default = "d_air")]
  pub ambient_name: String,
  #[serde(default = "d_sub")]
  pub substrate_name: String,
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

fn as_bool(v: &Value, key: &str) -> Result<bool, String> {
  v.as_bool().ok_or_else(|| format!("flag {key:?} must be a boolean"))
}

fn as_f64(v: &Value, key: &str) -> Result<f64, String> {
  v.as_f64().ok_or_else(|| format!("flag {key:?} must be a number"))
}

fn as_i32(v: &Value, key: &str) -> Result<i32, String> {
  v.as_i64()
    .and_then(|n| i32::try_from(n).ok())
    .ok_or_else(|| format!("flag {key:?} must be an integer"))
}

/// Apply a driver-key flag map onto an authoring film. Keys mirror the
/// Python driver split (`rough_val` aliases `roughness`); unknown keys
/// are errors, never ignored.
pub(crate) fn apply_flag_map(layer: &mut Layer, map: &BTreeMap<String, Value>) -> Result<(), String> {
  for (k, v) in map {
    match k.as_str() {
      "coherent" => layer.coherent = as_bool(v, k)?,
      "roughness" | "rough_val" => layer.roughness = as_f64(v, k)?,
      "rough_type" => layer.rough_type = RoughnessType::try_from_i32(as_i32(v, k)?)
        .map_err(|e| format!("flag {k:?}: {e}"))?,
      "inhomogen" => layer.inhomogen = as_bool(v, k)?,
      "inh_delta" => layer.inh_delta = as_f64(v, k)?,
      "interface" => layer.interface = as_bool(v, k)?,
      "interface_thickness" => layer.interface_thickness = as_f64(v, k)?,
      "optimize" => layer.optimize = as_bool(v, k)?,
      "needle" => layer.needle = as_bool(v, k)?,
      _ => return Err(format!("unknown film flag {k:?}")),
    }
  }
  Ok(())
}

pub(crate) fn apply_row(layer: &mut Layer, row: &LayerRow) -> Result<(), String> {
  layer.coherent = row.coherent;
  layer.roughness = row.roughness_nm;
  layer.rough_type =
    RoughnessType::try_from_i32(row.rough_type).map_err(|e| format!("rough_type: {e}"))?;
  layer.inhomogen = row.inhomogen;
  layer.inh_delta = row.inh_delta;
  layer.interface = row.interface;
  layer.interface_thickness = row.interface_thickness_nm;
  layer.optimize = row.optimize;
  layer.needle = row.needle;
  Ok(())
}

pub(crate) fn build_group(row: &GroupRow) -> Result<Group, String> {
  let mut g = Group::new(&row.name);
  g.thick_factor = row.thick_factor;
  g.thick_summand = row.thick_summand;
  g.n_factor = row.n_factor;
  g.k_factor = row.k_factor;
  g.inh_delta_summand = row.inh_delta_summand;
  g.roughness_summand = row.roughness_summand;
  g.interface_summand = row.interface_summand;
  if row.error_mask.len() != 6 {
    return Err(format!("group {:?}: error_mask needs 6 entries", row.name));
  }
  if row.optimization_mask.len() != 7 {
    return Err(format!("group {:?}: optimization_mask needs 7 binary entries", row.name));
  }
  for (i, v) in row.error_mask.iter().enumerate() {
    g.error_mask[i] = *v;
  }
  for (i, v) in row.optimization_mask.iter().enumerate() {
    if *v != 0 && *v != 1 {
      return Err(format!("group {:?}: optimization_mask must be binary", row.name));
    }
    g.optimization_mask[i] = *v;
  }
  g.thickness_error_type =
    ErrorType::try_from_i32(row.thickness_error_type).map_err(|e| format!("group {:?}: {e}", row.name))?;
  g.n_error_type =
    ErrorType::try_from_i32(row.n_error_type).map_err(|e| format!("group {:?}: {e}", row.name))?;
  g.k_error_type =
    ErrorType::try_from_i32(row.k_error_type).map_err(|e| format!("group {:?}: {e}", row.name))?;
  g.inh_delta_error_type = ErrorType::try_from_i32(row.inh_delta_error_type)
    .map_err(|e| format!("group {:?}: {e}", row.name))?;
  g.roughness_error_type = ErrorType::try_from_i32(row.roughness_error_type)
    .map_err(|e| format!("group {:?}: {e}", row.name))?;
  g.interface_error_type = ErrorType::try_from_i32(row.interface_error_type)
    .map_err(|e| format!("group {:?}: {e}", row.name))?;
  g.thickness_error_params = row.thickness_error_params.build();
  g.inh_delta_error_params = row.inh_delta_error_params.build();
  g.roughness_error_params = row.roughness_error_params.build();
  g.interface_error_params = row.interface_error_params.build();
  g.n_error_params = row.n_error_params.build();
  g.k_error_params = row.k_error_params.build();
  Ok(g)
}

/// Evaluate one library entry on the grid.
pub(crate) fn eval_material(def: &MaterialDef, wavelengths: &[f64]) -> Result<Vec<Complex64>, String> {
  let model = if def.model == "TableMaterial" { "Table" } else { def.model.as_str() };
  let mut params = def.params.clone();
  if model == "Table" {
    let n = def.n_data.as_ref().ok_or_else(|| "TableMaterial requires n_data".to_string())?;
    params.insert(
      "n_data".to_string(),
      Value::Array(vec![
        Value::Array(n.wavelengths.iter().map(|x| Value::from(*x)).collect()),
        Value::Array(n.values.iter().map(|x| Value::from(*x)).collect()),
      ]),
    );
    if let Some(k) = def.k_data.as_ref() {
      params.insert(
        "k_data".to_string(),
        Value::Array(vec![
          Value::Array(k.wavelengths.iter().map(|x| Value::from(*x)).collect()),
          Value::Array(k.values.iter().map(|x| Value::from(*x)).collect()),
        ]),
      );
    }
  }
  spec_from_def(def)?.evaluate(wavelengths)
}

/// Build the evaluatable spec without evaluating (lazy shelves).
pub(crate) fn spec_from_def(def: &MaterialDef) -> Result<MaterialSpec, String> {
  let model = if def.model == "TableMaterial" { "Table" } else { def.model.as_str() };
  let mut params = def.params.clone();
  if model == "Table" {
    let n = def.n_data.as_ref().ok_or_else(|| "TableMaterial requires n_data".to_string())?;
    params.insert(
      "n_data".to_string(),
      Value::Array(vec![
        Value::Array(n.wavelengths.iter().map(|x| Value::from(*x)).collect()),
        Value::Array(n.values.iter().map(|x| Value::from(*x)).collect()),
      ]),
    );
    if let Some(k) = def.k_data.as_ref() {
      params.insert(
        "k_data".to_string(),
        Value::Array(vec![
          Value::Array(k.wavelengths.iter().map(|x| Value::from(*x)).collect()),
          Value::Array(k.values.iter().map(|x| Value::from(*x)).collect()),
        ]),
      );
    }
  }
  Ok(MaterialSpec::new(model, params))
}

fn fixed_spec(name: &str, nk: Vec<Complex64>) -> LayerSpec {
  LayerSpec {
    material: Arc::from(name),
    nk: Arc::from(nk),
    d_nm: 0.0,
    coherent: true,
    rough_type: 0,
    rough_val: 0.0,
    optimize: false,
    needle: false,
  }
}

/// Assemble a design: evaluate the library, build films + groups, expand
/// once via [`DesignStack::from_design`]. Returns `(stack, contrast,
/// warnings)` — warnings (homogenized graded films) are for the caller
/// to re-emit; nothing is ever refused or silent.
pub fn build_design(
  req: &DesignRequest,
  wavelengths: &[f64],
) -> Result<(DesignStack, ContrastMap, Vec<String>), String> {
  // Library → nk on the grid, keyed by code (else name).
  let mut nk_table: HashMap<Arc<str>, Vec<Complex64>> = HashMap::new();
  for def in &req.library {
    let key = def.code.as_deref().unwrap_or(&def.name);
    nk_table.insert(Arc::from(key), eval_material(def, wavelengths)?);
  }
  let resolve = |code: &str| -> Result<Vec<Complex64>, String> {
    nk_table
      .get(code)
      .cloned()
      .ok_or_else(|| format!("material code {code:?} not found in library"))
  };

  // Split rows by layer type.
  let mut ambient_rows = vec![];
  let mut substrate_rows = vec![];
  let mut film_rows = vec![];
  for row in &req.structure.layers {
    match row.layer_type {
      0 => ambient_rows.push(row),
      2 => substrate_rows.push(row),
      1 => film_rows.push(row),
      t => return Err(format!("layer_type must be 0, 1 or 2 (got {t})")),
    }
  }
  if ambient_rows.len() > 1 {
    return Err("at most one ambient (layer_type=0) row".to_string());
  }
  if substrate_rows.len() > 1 {
    return Err("at most one substrate (layer_type=2) row".to_string());
  }
  let half_space = |rows: &[&LayerRow], name: &str, n: f64| -> Result<LayerSpec, String> {
    match rows.first() {
      Some(row) => Ok(fixed_spec(name, resolve(&row.material_code)?)),
      None => Ok(fixed_spec(
        name,
        vec![Complex64::new(n, 0.0); wavelengths.len()],
      )),
    }
  };
  let amb = half_space(&ambient_rows, &req.ambient_name, 1.0)?;
  let sub = half_space(&substrate_rows, &req.substrate_name, 1.52)?;

  // Films: global flags → row → per-film override. Codes key the table.
  let mut films: Vec<Layer> = Vec::with_capacity(film_rows.len());
  let mut seen: HashSet<&str> = HashSet::new();
  for row in film_rows {
    if !seen.insert(row.material_code.as_str()) {
      return Err(format!(
        "duplicate film material code {:?} (film names key the nk table)",
        row.material_code
      ));
    }
    resolve(&row.material_code)?; // fail fast with the code in the message
    let mut layer = Layer::film(row.thickness_nm, &row.material_code);
    layer.layer_type = LayerType::Film;
    apply_flag_map(&mut layer, &req.film_flags)?;
    apply_row(&mut layer, row)?;
    if let Some(over) = req.per_film_flags.get(&row.material_code) {
      apply_flag_map(&mut layer, over)?;
    }
    films.push(layer);
  }

  // Contrast seeds: zero-thickness, coherent, optimizable carriers.
  let mut cmap = ContrastMap::new();
  for (host, seed) in &req.contrast {
    cmap.insert(
      Arc::from(host.as_str()),
      LayerSpec {
        material: Arc::from(format!("{host}_seed")),
        nk: Arc::from(resolve(seed)?),
        d_nm: 0.0,
        coherent: true,
        rough_type: 0,
        rough_val: 0.0,
        optimize: true,
        needle: true,
      },
    );
  }

  let mut groups = HashMap::new();
  for row in &req.structure.groups {
    groups.insert(row.name.clone(), build_group(row)?);
  }

  // Background is implied, not declared (mirrors the Python driver).
  let background: HashSet<String> = films
    .iter()
    .filter(|l| l.inhomogen && !l.optimize && !l.needle)
    .map(|l| l.material.clone())
    .collect();

  let (stack, warnings) =
    DesignStack::from_design(amb, sub, &films, &nk_table, &groups, wavelengths, &background)?;
  Ok((stack, cmap, warnings))
}

// ---------------------------------------------------------------------------
// Tests (standalone: no Python)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn lib() -> Vec<MaterialDef> {
    vec![
      MaterialDef {
        name: "L".to_string(),
        code: None,
        model: "Konstant".to_string(),
        params: [("n".to_string(), Value::from(1.45))].into_iter().collect(),
        n_data: None,
        k_data: None,
      },
      MaterialDef {
        name: "H".to_string(),
        code: None,
        model: "Konstant".to_string(),
        params: [("n".to_string(), Value::from(2.1))].into_iter().collect(),
        n_data: None,
        k_data: None,
      },
    ]
  }

  fn film(code: &str, d: f64) -> LayerRow {
    LayerRow {
      material_code: code.to_string(),
      thickness_nm: d,
      coherent: true,
      roughness_nm: 0.0,
      rough_type: 0,
      inhomogen: false,
      inh_delta: 0.1,
      interface: false,
      interface_thickness_nm: 0.0,
      optimize: true,
      needle: true,
      layer_type: 1,
    }
  }

  fn req(films: Vec<LayerRow>) -> DesignRequest {
    DesignRequest {
      structure: StructureCfg { label: "t".to_string(), layers: films, groups: vec![] },
      library: lib(),
      contrast: [("H".to_string(), "L".to_string())].into_iter().collect(),
      film_flags: BTreeMap::new(),
      per_film_flags: BTreeMap::new(),
      ambient_name: "air".to_string(),
      substrate_name: "sub".to_string(),
    }
  }

  #[test]
  fn builds_flat_stack_standalone() {
    let wl = vec![500.0, 600.0];
    let (stack, cmap, warnings) =
      build_design(&req(vec![film("L", 100.0), film("H", 60.0)]), &wl).unwrap();
    assert!(warnings.is_empty());
    assert_eq!(cmap.len(), 1);
    assert!(cmap.contains_key("H"));
    // 2 flat films (half-spaces are not films).
    assert_eq!(stack.films().len(), 2);
  }

  #[test]
  fn duplicate_film_code_refused() {
    let wl = vec![500.0];
    let err = build_design(&req(vec![film("L", 100.0), film("L", 50.0)]), &wl).unwrap_err();
    assert!(err.contains("duplicate film"), "got: {err}");
  }

  #[test]
  fn unknown_code_refused() {
    let wl = vec![500.0];
    let err = build_design(&req(vec![film("X", 100.0)]), &wl).unwrap_err();
    assert!(err.contains("\"X\""), "got: {err}");
  }

  #[test]
  fn unknown_flag_refused() {
    let wl = vec![500.0];
    let mut r = req(vec![film("L", 100.0)]);
    r.film_flags.insert("bogus".to_string(), Value::from(true));
    let err = build_design(&r, &wl).unwrap_err();
    assert!(err.contains("unknown film flag"), "got: {err}");
  }

  #[test]
  fn graded_pinned_expands_background() {
    let wl = vec![500.0, 600.0];
    let mut g = film("H", 100.0);
    g.inhomogen = true;
    g.optimize = false;
    g.needle = false;
    let (stack, _, warnings) = build_design(&req(vec![film("L", 50.0), g]), &wl).unwrap();
    assert!(warnings.is_empty()); // pinned: silent
    let n = stack.films().len();
    assert!(n > 2, "graded film must expand into slices, got {n}");
  }
}
