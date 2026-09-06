//! navette::config — versioned program documents and section assembly.
//!
//! Rust-first port of the Python `config.program` + section builders:
//! envelope gating, legacy-flat detection, prefix namespacing, and
//! dependency-order assembly (materials → groups → structures →
//! architect) into native objects. File reading is JSON-only (YAML
//! stays Python-side authoring); the Python layer thins to dict
//! handover + native calls. Context-merging (live objects) stays
//! Python-side dict assembly.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use serde_json::Value;

use crate::smatrix::synthesis::design_config::{
  GroupRow, LayerRow, MaterialDef, StructureCfg, apply_row, build_group, spec_from_def,
};
use crate::structure::{
  Architect, BlockKind, Entry, Group, Layer, LayerType, MaterialProvider, SharedGroup,
  SpecProvider, Structure,
};

/// Program envelope version (single canonical gate).
pub const PROGRAM_SCHEMA_VERSION: u32 = 1;

/// One architect block: structure label reference + placement.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct BlockCfg {
  pub structure: String,
  #[serde(default)]
  pub inverted: bool,
  #[serde(default = "d_repeat")]
  pub repeat_count: i64,
  #[serde(default)]
  pub label: String,
  #[serde(default = "d_kind")]
  pub kind: i32,
}

fn d_repeat() -> i64 {
  1
}

fn d_kind() -> i32 {
  1
}

/// Authoring-time validation (the pydantic bound inventory, natively).
/// Called by the PyO3 config constructors; assembly assumes validity.
impl MaterialDef {
  pub fn validate(&self) -> Result<(), String> {
    const MODELS: [&str; 6] = [
      "Konstant", "TableMaterial", "Cauchy", "CauchyUrbach", "Sellmeier", "SellmeierUrbach",
    ];
    if !MODELS.contains(&self.model.as_str()) {
      return Err(format!("Unknown model: {}", self.model));
    }
    let num = |key: &str| -> Result<f64, String> {
      self
        .params
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{}: '{}' must be a number.", self.model, key))
    };
    let pos = |key: &str| -> Result<(), String> {
      if num(key)? <= 0.0 {
        return Err(format!("{}: '{}' must be > 0.", self.model, key));
      }
      Ok(())
    };
    let nonneg = |key: &str, default: f64| -> Result<(), String> {
      let v = self.params.get(key).and_then(|v| v.as_f64()).unwrap_or(default);
      if v < 0.0 {
        return Err(format!("{}: '{}' must be >= 0.", self.model, key));
      }
      Ok(())
    };
    match self.model.as_str() {
      "Konstant" => {
        pos("n")?;
        nonneg("k", 0.0)?;
      }
      "TableMaterial" => {
        if self.n_data.is_none() {
          return Err("TableMaterial requires n_data".to_string());
        }
        nonneg("n_factor", 1.0)?;
        nonneg("k_factor", 1.0)?;
      }
      "Cauchy" => {
        num("A")?;
        num("B")?;
        num("C")?;
      }
      "CauchyUrbach" => {
        num("A")?;
        num("B")?;
        num("C")?;
        pos("alpha0")?;
        pos("Eu")?;
        pos("lambda_g")?;
      }
      "Sellmeier" => {
        pos("B1")?;
        pos("C1")?;
        pos("B2")?;
        pos("C2")?;
        nonneg("B3", 0.0)?;
        nonneg("C3", 0.0)?;
      }
      _ => {
        pos("B1")?;
        pos("C1")?;
        pos("B2")?;
        pos("C2")?;
        nonneg("B3", 0.0)?;
        nonneg("C3", 0.0)?;
        pos("alpha0")?;
        pos("Eu")?;
        pos("lambda_g")?;
      }
    }
    Ok(())
  }
}

impl LayerRow {
  pub fn validate(&self) -> Result<(), String> {
    if !(self.thickness_nm > 0.0) {
      return Err("thickness_nm must be > 0.".to_string());
    }
    if self.roughness_nm < 0.0 {
      return Err("roughness_nm must be >= 0.".to_string());
    }
    crate::structure::RoughnessType::try_from_i32(self.rough_type)
      .map_err(|e| format!("rough_type: {e}"))?;
    if !(0.0..=1.0).contains(&self.inh_delta) {
      return Err("inh_delta must be in [0, 1].".to_string());
    }
    if self.interface_thickness_nm < 0.0 {
      return Err("interface_thickness_nm must be >= 0.".to_string());
    }
    match self.layer_type {
      0 | 1 | 2 => {}
      t => return Err(format!("layer_type must be 0, 1 or 2 (got {t}).")),
    }
    Ok(())
  }
}

impl GroupRow {
  pub fn validate(&self) -> Result<(), String> {
    if self.error_mask.len() != 6 {
      return Err(format!("group {:?}: error_mask needs 6 entries.", self.name));
    }
    if self.optimization_mask.len() != 7
      || self.optimization_mask.iter().any(|x| *x != 0 && *x != 1)
    {
      return Err(format!("group {:?}: optimization_mask must be 7 binary entries.", self.name));
    }
    for (key, v) in [
      ("thickness", self.thickness_error_type),
      ("n", self.n_error_type),
      ("k", self.k_error_type),
      ("inh_delta", self.inh_delta_error_type),
      ("roughness", self.roughness_error_type),
      ("interface", self.interface_error_type),
    ] {
      crate::structure::ErrorType::try_from_i32(v)
        .map_err(|_| format!("group {:?}: {key} error type must be 0, 1 or 2.", self.name))?;
    }
    Ok(())
  }
}

impl StructureCfg {
  /// Nested validation: every layer + group row.
  pub fn validate(&self) -> Result<(), String> {
    for (i, row) in self.layers.iter().enumerate() {
      row.validate().map_err(|e| format!("layers[{i}]: {e}"))?;
    }
    for row in &self.groups {
      row.validate()?;
    }
    Ok(())
  }
}

impl BlockCfg {
  pub fn validate(&self) -> Result<(), String> {
    if self.repeat_count < 1 {
      return Err("repeat_count must be >= 1.".to_string());
    }
    crate::structure::BlockKind::try_from_i32(self.kind)
      .map_err(|e| format!("kind: {e}"))?;
    Ok(())
  }
}

/// Everything a program file restores (absent sections stay empty).
#[derive(Debug, Default)]
pub struct LoadedProgram {
  pub name: Option<String>,
  pub materials: Option<SpecProvider>,
  pub groups: HashMap<String, Group>,
  pub structures: HashMap<String, crate::structure::Structure>,
  pub architect: Option<Architect>,
}

fn px(value: &str, prefix: Option<&str>) -> String {
  match prefix {
    Some(p) => format!("{p}{value}"),
    None => value.to_string(),
  }
}

fn code_key(def: &MaterialDef) -> &str {
  def.code.as_deref().unwrap_or(&def.name)
}

// ---------------------------------------------------------------------------
// Envelope gate (mirrors `_gate` + `load_document`, JSON form)
// ---------------------------------------------------------------------------

/// Gate + classify a parsed document.
/// Returns `(kind, name, payload)`; legacy-flat maps to
/// `("materials"|"structure", None, section-content)`.
pub fn gate_document(raw: &Value) -> Result<(String, Option<String>, Value), String> {
  let top = raw.as_object().ok_or_else(|| "document top level must be a mapping.".to_string())?;
  if !top.contains_key("kind") {
    // Legacy flat form (no envelope to gate).
    if let Some(items) = top.get("materials") {
      return Ok(("materials".to_string(), None, items.clone()));
    }
    if let Some(layers) = top.get("layers") {
      let mut map = serde_json::Map::new();
      map.insert("label".to_string(), Value::from("stack"));
      map.insert("layers".to_string(), layers.clone());
      map.insert(
        "groups".to_string(),
        top.get("groups").cloned().unwrap_or(Value::Array(vec![])),
      );
      return Ok(("structure".to_string(), None, Value::Object(map)));
    }
    return Err("legacy document needs 'materials' or 'layers' at top level.".to_string());
  }
  let kinds = ["materials", "groups", "structure", "architect", "program"];
  let kind = top
    .get("kind")
    .and_then(|v| v.as_str())
    .ok_or_else(|| "program kind must be a string.".to_string())?;
  if !kinds.contains(&kind) {
    return Err(format!("program kind {kind:?} unknown (expected one of {}).", kinds.join(", ")));
  }
  match top.get("schema_version").and_then(|v| v.as_u64()) {
    Some(1) => {}
    other => {
      return Err(format!(
        "program schema_version {other:?} unsupported (code reads {PROGRAM_SCHEMA_VERSION})."
      ))
    }
  }
  let payload_keys: &[&str] = match kind {
    "materials" => &["materials"],
    "groups" => &["groups"],
    "structure" => &["label", "layers", "groups"],
    "architect" => &["structures", "blocks"],
    _ => &["sections"],
  };
  let unknown: Vec<&String> =
    top.keys().filter(|k| *k != "schema_version" && *k != "kind" && *k != "name" && !payload_keys.contains(&k.as_str())).collect();
  if !unknown.is_empty() {
    return Err(format!("unknown top-level keys: {unknown:?}."));
  }
  if kind == "program" {
    match top.get("sections") {
      Some(Value::Object(_)) => {}
      _ => return Err("program document needs a 'sections' mapping.".to_string()),
    }
  } else if top.contains_key("sections") {
    return Err(format!("standalone {kind:?} document must not carry 'sections'."));
  }
  let name = top.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
  if kind == "program" {
    return Ok((kind.to_string(), name, top["sections"].clone()));
  }
  if kind == "materials" || kind == "groups" {
    let key = kind;
    let items = top.get(key).cloned().unwrap_or(Value::Array(vec![]));
    return Ok((kind.to_string(), name, items));
  }
  let mut payload = serde_json::Map::new();
  for (k, v) in top {
    if k != "schema_version" && k != "kind" && k != "name" {
      payload.insert(k.clone(), v.clone());
    }
  }
  Ok((kind.to_string(), name, Value::Object(payload)))
}

// ---------------------------------------------------------------------------
// Section assembly (each usable standalone or nested)
// ---------------------------------------------------------------------------

fn from_json<T: for<'de> serde::Deserialize<'de>>(v: &Value, what: &str) -> Result<T, String> {
  serde_json::from_value(v.clone()).map_err(|e| format!("{what}: invalid section: {e}"))
}

/// `SpecProvider` from a `materials` section (prefix-aware).
pub fn load_materials(
  items: &Value,
  grid: &[f64],
  prefix: Option<&str>,
) -> Result<SpecProvider, String> {
  let items: Vec<Value> =
    from_json(items, "materials").map_err(|_| "materials section must be a list.".to_string())?;
  let mut entries = HashMap::new();
  for item in &items {
    let mut def: MaterialDef = from_json(item, "material")?;
    def.name = px(&def.name, prefix);
    let code = px(code_key(&def), prefix);
    def.code = Some(code.clone());
    entries.insert(code, Entry::Spec(spec_from_def(&def)?));
  }
  SpecProvider::new(entries, grid.to_vec())
}

/// `{name: Group}` from a `groups` section (prefix-aware).
pub fn load_groups(items: &Value, prefix: Option<&str>) -> Result<HashMap<String, Group>, String> {
  let items: Vec<Value> =
    from_json(items, "groups").map_err(|_| "groups section must be a list.".to_string())?;
  let mut out = HashMap::new();
  for item in &items {
    let mut row: GroupRow = from_json(item, "group")?;
    row.name = px(&row.name, prefix);
    out.insert(row.name.clone(), build_group(&row)?);
  }
  Ok(out)
}

fn layer_from_row(
  row: &LayerRow,
  provider: Option<&dyn MaterialProvider>,
  prefix: Option<&str>,
) -> Result<Layer, String> {
  let code = px(&row.material_code, prefix);
  if let Some(p) = provider {
    if !p.contains(&code) {
      // NOTE: prefixed codes only resolve against prefixed providers;
      // unprefixed fallthrough mirrors the file-wins rule.
      let unprefixed = row.material_code.as_str();
      if !p.contains(unprefixed) {
        return Err(format!("material code {code:?} not found in provider"));
      }
      let mut layer = Layer::film(row.thickness_nm, unprefixed);
      apply_layer_fields(&mut layer, row)?;
      return Ok(layer);
    }
  }
  let mut layer = Layer::film(row.thickness_nm, &code);
  apply_layer_fields(&mut layer, row)?;
  Ok(layer)
}

fn apply_layer_fields(layer: &mut Layer, row: &LayerRow) -> Result<(), String> {
  layer.layer_type = LayerType::Film;
  apply_row(layer, row)
}

/// Native `Structure` from a `structure` section.
/// Per-section `groups` merge over `library_groups` (own wins).
/// `materials` is required (mirrors the `KeyError` when absent).
pub fn load_structure(
  payload: &Value,
  materials: Option<&dyn MaterialProvider>,
  library_groups: &HashMap<String, Group>,
  prefix: Option<&str>,
) -> Result<Structure, String> {
  let cfg: StructureCfg = from_json(payload, "structure")?;
  assemble_named(&cfg, materials, library_groups, prefix)
}

/// `{label: Structure}` (duplicate labels raise).
pub fn load_named_structures(
  items: &Value,
  materials: Option<&dyn MaterialProvider>,
  library_groups: &HashMap<String, Group>,
  prefix: Option<&str>,
) -> Result<HashMap<String, Structure>, String> {
  let items: Vec<Value> =
    from_json(items, "structures").map_err(|_| "structures section must be a list.".to_string())?;
  let mut out = HashMap::new();
  for item in &items {
    let cfg: StructureCfg = from_json(item, "structure")?;
    let label = px(&cfg.label, prefix);
    if out.contains_key(&label) {
      return Err(format!("duplicate structure label '{label}'."));
    }
    let st = assemble_named(&cfg, materials, library_groups, prefix)?;
    out.insert(label, st);
  }
  Ok(out)
}

fn assemble_named(
  cfg: &StructureCfg,
  materials: Option<&dyn MaterialProvider>,
  library_groups: &HashMap<String, Group>,
  prefix: Option<&str>,
) -> Result<Structure, String> {
  let provider =
    materials.ok_or_else(|| "structure section needs materials: no provider given.".to_string())?;
  let mut layers = Vec::with_capacity(cfg.layers.len());
  for row in &cfg.layers {
    layers.push(layer_from_row(row, Some(provider), prefix)?);
  }
  let mut merged: HashMap<String, SharedGroup> = HashMap::new();
  for (k, g) in library_groups {
    merged.insert(k.clone(), crate::structure::group::shared_group(g.clone()));
  }
  for row in &cfg.groups {
    let mut named = row.clone();
    named.name = px(&named.name, prefix);
    merged.insert(
      named.name.clone(),
      crate::structure::group::shared_group(build_group(&named)?),
    );
  }
  Ok(Structure { layers, groups: merged })
}

/// Native `Architect`: blocks reference `structures` by label.
pub fn load_architect(
  payload: &Value,
  structures: &HashMap<String, Structure>,
  prefix: Option<&str>,
) -> Result<Architect, String> {
  let blocks: Vec<Value> = payload
    .get("blocks")
    .and_then(|v| v.as_array())
    .ok_or_else(|| "architect section needs 'blocks'.".to_string())?
    .clone();
  let mut arch = Architect::new();
  for (i, raw) in blocks.iter().enumerate() {
    let b: BlockCfg = from_json(raw, "block")?;
    let target = px(&b.structure, prefix);
    let st = structures.get(&target).ok_or_else(|| {
      format!("architect block {i}: unknown structure label '{target}'.")
    })?;
    if b.repeat_count < 1 {
      return Err(format!("architect block {i}: repeat_count must be >= 1."));
    }
    let kind = BlockKind::try_from_i32(b.kind)
      .map_err(|e| format!("architect block {i}: {e}"))?;
    let label = if b.label.is_empty() { b.label.clone() } else { px(&b.label, prefix) };
    arch
      .add_structure(st.clone(), b.inverted, b.repeat_count as usize, label, kind)
      .map_err(|e| format!("architect block {i}: {e}"))?;
  }
  Ok(arch)
}

// ---------------------------------------------------------------------------
// Full program (dependency order; file sections win)
// ---------------------------------------------------------------------------

/// Restore a program from parsed JSON text + evaluation grid.
/// Sections load in dependency order (materials → groups → structures →
/// architect); a standalone section document loads just that part.
pub fn load_program_json(text: &str, grid: &[f64]) -> Result<LoadedProgram, String> {
  load_program_json_prefixed(text, grid, None)
}

/// Prefix-namespaced variant (multi-load collisions).
pub fn load_program_json_prefixed(
  text: &str,
  grid: &[f64],
  prefix: Option<&str>,
) -> Result<LoadedProgram, String> {
  let raw: Value =
    serde_json::from_str(text).map_err(|e| format!("program: invalid JSON: {e}"))?;
  let (kind, name, payload) = gate_document(&raw)?;
  load_program_parts(&kind, name, &payload, grid, prefix)
}

fn load_program_parts(
  kind: &str,
  name: Option<String>,
  payload: &Value,
  grid: &[f64],
  prefix: Option<&str>,
) -> Result<LoadedProgram, String> {
  let mut prog = LoadedProgram { name, ..Default::default() };
  let sections: BTreeMap<String, Value> = if kind == "program" {
    payload
      .as_object()
      .ok_or_else(|| "program document needs a 'sections' mapping.".to_string())?
      .iter()
      .map(|(k, v)| (k.clone(), v.clone()))
      .collect()
  } else {
    [(kind.to_string(), payload.clone())].into_iter().collect()
  };

  if let Some(items) = sections.get("materials") {
    prog.materials = Some(load_materials(items, grid, prefix)?);
  }
  if let Some(items) = sections.get("groups") {
    prog.groups = load_groups(items, prefix)?;
  }
  let provider = prog.materials.as_ref().map(|p| p as &dyn MaterialProvider);

  if kind == "structure" {
    let label = payload
      .get("label")
      .and_then(|v| v.as_str())
      .unwrap_or("stack");
    prog.structures.insert(
      px(label, prefix),
      load_structure(payload, provider, &prog.groups, prefix)?,
    );
  } else if let Some(items) = sections.get("structures") {
    prog.structures = load_named_structures(items, provider, &prog.groups, prefix)?;
  }

  if kind == "architect" {
    if payload.get("structures").is_none() {
      return Err("standalone architect document needs 'structures' + 'blocks'.".to_string());
    }
    prog.structures = load_named_structures(
      &payload["structures"],
      provider,
      &prog.groups,
      prefix,
    )?;
  }
  if let Some(arch_payload) = sections.get("architect") {
    prog.architect = Some(load_architect(arch_payload, &prog.structures, prefix)?);
  }
  Ok(prog)
}

/// Restore a program from a JSON file.
pub fn load_program_file(
  path: &str,
  grid: &[f64],
  prefix: Option<&str>,
) -> Result<LoadedProgram, String> {
  let text =
    std::fs::read_to_string(path).map_err(|e| format!("program: cannot read {path}: {e}"))?;
  load_program_json_prefixed(&text, grid, prefix)
}

// ---------------------------------------------------------------------------
// Tests (standalone: no Python)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn grid() -> Vec<f64> {
    vec![500.0, 600.0]
  }

  fn program_json() -> String {
    serde_json::json!({
      "schema_version": 1,
      "kind": "program",
      "name": "demo",
      "sections": {
        "materials": [
          {"name": "L", "code": "L", "model": "Konstant",
           "params": {"n": 1.45}},
          {"name": "H", "code": "H", "model": "Konstant",
           "params": {"n": 2.1}},
        ],
        "groups": [
          {"name": "H", "thick_factor": 1.0},
        ],
        "structures": [
          {"label": "main", "layers": [
            {"material_code": "L", "thickness_nm": 100.0},
            {"material_code": "H", "thickness_nm": 60.0},
          ], "groups": []},
        ],
        "architect": {"blocks": [
          {"structure": "main", "label": "run", "repeat_count": 2},
        ]},
      }
    })
    .to_string()
  }

  #[test]
  fn full_program_restores() {
    let prog = load_program_json(&program_json(), &grid()).unwrap();
    assert_eq!(prog.name.as_deref(), Some("demo"));
    assert!(prog.materials.as_ref().unwrap().contains("L"));
    assert!(prog.groups.contains_key("H"));
    assert_eq!(prog.structures["main"].layers.len(), 2);
    let arch = prog.architect.unwrap();
    assert_eq!(arch.blocks.len(), 1);
    assert_eq!(arch.blocks[0].repeat_count, 2);
  }

  #[test]
  fn prefix_namespaces() {
    let prog = load_program_json_prefixed(&program_json(), &grid(), Some("p_")).unwrap();
    assert!(prog.materials.as_ref().unwrap().contains("p_L"));
    assert!(prog.groups.contains_key("p_H"));
    assert!(prog.structures.contains_key("p_main"));
  }

  #[test]
  fn gate_refuses() {
    let grid = grid();
    assert!(load_program_json("{\"kind\": \"program\"}", &grid).is_err()); // no version
    assert!(load_program_json(
      "{\"schema_version\": 2, \"kind\": \"program\", \"sections\": {}}", &grid).is_err());
    assert!(load_program_json(
      "{\"schema_version\": 1, \"kind\": \"nope\"}", &grid).is_err());
    assert!(load_program_json(
      "{\"schema_version\": 1, \"kind\": \"groups\", \"groups\": [], \"bogus\": 1}", &grid).is_err());
    assert!(load_program_json("{\"layers\": []}", &grid).is_err()); // legacy, no materials key
  }

  #[test]
  fn legacy_flat_materials() {
    let prog = load_program_json(
      "{\"materials\": [{\"name\": \"L\", \"model\": \"Konstant\", \"params\": {\"n\": 1.45}}]}",
      &grid(),
    )
    .unwrap();
    assert!(prog.materials.as_ref().unwrap().contains("L"));
  }

  #[test]
  fn refs_must_resolve() {
    let mut v: Value = serde_json::from_str(&program_json()).unwrap();
    v["sections"]["structures"][0]["layers"][0]["material_code"] = Value::from("X");
    assert!(load_program_json(&v.to_string(), &grid()).is_err());
    let mut v: Value = serde_json::from_str(&program_json()).unwrap();
    v["sections"]["architect"]["blocks"][0]["structure"] = Value::from("ghost");
    assert!(load_program_json(&v.to_string(), &grid()).is_err());
  }
}
