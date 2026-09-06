//! One design stack: layers + material-keyed groups.
//!
//! Mirrors `navette.structure.structure.Navette_Structure`. The provider
//! stays OUTSIDE the struct (passed to calls that need it) — states never
//! carried it, so neither do we. All messages are prefix-free (severity is
//! the channel); the Python boundary re-attaches `warning:` for compat.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Value, json};

use num_complex::Complex64;

use crate::expansion::{ExpandOptions, SolverArrays, Span, expand};
use crate::group::Group;
use crate::layer::Layer;
use crate::providers::MaterialProvider;
use crate::validation::ValidationIssue;
use crate::version::{SCHEMA_VERSION, check_schema_version};

/// One design stack: ordered layers + material-keyed group policies.
#[derive(Debug, Clone, Default)]
pub struct Structure {
  pub layers: Vec<Layer>,
  pub groups: HashMap<String, crate::SharedGroup>,
}

impl PartialEq for Structure {
  fn eq(&self, other: &Self) -> bool {
    if self.layers != other.layers || self.groups.len() != other.groups.len() {
      return false;
    }
    self.groups.iter().all(|(k, g)| other.groups.get(k).is_some_and(|o| *o.borrow() == *g.borrow()))
  }
}

impl Structure {
  /// Value snapshot of the group map (expansion input; Send-clean).
  fn snapshot_groups(&self) -> HashMap<String, Group> {
    self.groups.iter().map(|(k, g)| (k.clone(), g.borrow().clone())).collect()
  }

  pub fn new(layers: Vec<Layer>, groups: HashMap<String, Group>) -> Self {
    Self {
      layers,
      groups: groups.into_iter().map(|(k, g)| (k, crate::shared_group(g))).collect(),
    }
  }

  /// Share the caller's group handles (binding path: bakes stay visible).
  pub fn with_shared(layers: Vec<Layer>, groups: HashMap<String, crate::SharedGroup>) -> Self {
    Self { layers, groups }
  }


  /// Collect every issue (empty if all good); never raises.
  /// `provider = None` skips material-coverage and the dry run.
  pub fn validate(&self, provider: Option<&dyn MaterialProvider>) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if self.layers.is_empty() {
      issues.push(ValidationIssue::error("Structure contains no layers."));
      return issues;
    }
    for (i, layer) in self.layers.iter().enumerate() {
      if layer.thickness < 0.0 {
        issues.push(ValidationIssue::error(format!(
          "Layer {i} ({}): Negative thickness {} nm.",
          layer.material, layer.thickness
        )));
      }
      if layer.roughness < 0.0 {
        issues.push(ValidationIssue::error(format!(
          "Layer {i} ({}): Negative roughness {} nm.",
          layer.material, layer.roughness
        )));
      }
      // Overhang is LEGAL but suspicious: advisory, never blocking.
      if layer.interface && layer.interface_thickness >= layer.thickness {
        issues.push(ValidationIssue::warning(format!(
          "Layer {i} ({}): Interface thickness ({}) >= layer thickness ({}); clamped at expansion.",
          layer.material, layer.interface_thickness, layer.thickness
        )));
      }
      if layer.interface_thickness < 0.0 {
        issues.push(ValidationIssue::error(format!(
          "Layer {i} ({}): Negative interface thickness {} nm.",
          layer.material, layer.interface_thickness
        )));
      }
      if let Some(p) = provider
        && !p.contains(layer.material.as_str()) {
          issues.push(ValidationIssue::error(format!(
            "Layer {i}: Material '{}' not found in material provider.",
            layer.material
          )));
        }
    }
    let mut seen: HashSet<*const ()> = HashSet::new();
    let mut governed: HashSet<&str> = HashSet::new();
    for layer in &self.layers {
      governed.insert(layer.material.as_str());
      if let Some(handle) = self.groups.get(layer.material.as_str()) {
        let ptr = std::rc::Rc::as_ptr(handle) as *const ();
        if seen.insert(ptr) {
          issues.extend(handle.borrow().validate());
        }
      }
    }
    for name in self.groups.keys() {
      if !governed.contains(name.as_str()) {
        issues.push(ValidationIssue::warning(format!(
          "Group '{name}' governs no layer material (lookup is by material name; silent _DEFAULT_GROUP applies)."
        )));
      }
    }
    // Nominal dry run (skipped when materials are unknown).
    if let Some(p) = provider
      && !issues.iter().any(|e| e.is_error() && e.message.contains("not found in material provider")) {
        // Dry run needs a grid: use the provider's own when known.
        if let Some(grid) = p.grid() {
          let grid = grid.to_vec();
          let seq: Vec<(Layer, bool)> = self.layers.iter().cloned().map(|l| (l, false)).collect();
          match expand(&seq, p, &grid, &self.snapshot_groups(), ExpandOptions::deterministic()) {
            Err(e) => issues.push(ValidationIssue::error(format!("Nominal expansion failed: {e}"))),
            Ok((sa, spans)) => {
              if !sa.thicknesses.iter().all(|t| t.is_finite())
                || !sa.indices.iter().all(|z| z.re.is_finite() && z.im.is_finite())
              {
                issues.push(ValidationIssue::error(
                  "Nominal expansion produced NaN/inf (check group factors).".to_string(),
                ));
              }
              if sa.indices.iter().any(|z| z.re < 0.0) {
                issues.push(ValidationIssue::error(
                  "Nominal expansion produced n < 0 (check provider data / group n_factor).".to_string(),
                ));
              }
              if sa.indices.iter().any(|z| z.im < 0.0) {
                issues.push(ValidationIssue::error(
                  "Nominal expansion produced k < 0 (check provider data / group k_factor).".to_string(),
                ));
              }
              let n_rows = sa.n_rows();
              for j in 1..n_rows.saturating_sub(1).max(1) {
                if sa.thicknesses[j] > 0.0 {
                  continue;
                }
                if self.carve_explained(j, &spans, &sa.thicknesses) {
                  continue;
                }
                issues.push(ValidationIssue::error(
                  "Nominal expansion produced interior zero-thickness rows (group factors floored a film away; ambient/substrate may be 0).".to_string(),
                ));
                break;
              }
            }
          }
        }
      }
    issues
  }

  /// True when a zero row is a clamped interface carve, not a floored film.
  fn carve_explained(&self, row: usize, spans: &[Span], thicknesses: &[f64]) -> bool {
    for s in spans {
      if !(s.start <= row && row < s.end) || s.logical >= self.layers.len() {
        continue;
      }
      if !self.layers[s.logical].interface {
        return false;
      }
      if row == s.start {
        return true;
      }
      return thicknesses[s.start + 1..s.end].iter().all(|t| *t <= 0.0);
    }
    false
  }

  /// Solve gate: warnings flow (returned), errors raise.
  pub fn gate(&self, issues: &[ValidationIssue], what: &str) -> Result<Vec<String>, String> {
    let (errors, warnings) =
      (issues.iter().filter(|i| i.is_error()).collect::<Vec<_>>(), issues.iter().filter(|i| !i.is_error()).collect::<Vec<_>>());
    if !errors.is_empty() {
      return Err(format!(
        "{what} invalid:\n{}",
        errors.iter().map(|e| e.message.as_str()).collect::<Vec<_>>().join("\n")
      ));
    }
    Ok(warnings.iter().map(|w| w.message.clone()).collect())
  }

  /// Flatten to engine arrays (nominal, gated). Returns warnings + arrays.
  pub fn solver_inputs(
    &self,
    provider: &dyn MaterialProvider,
    wavelengths: &[f64],
  ) -> Result<(Vec<String>, SolverArrays, Vec<Span>), String> {
    let warnings = self.gate(&self.validate(Some(provider)), "Navette_Structure")?;
    let seq: Vec<(Layer, bool)> = self.layers.iter().cloned().map(|l| (l, false)).collect();
    let (sa, spans) = expand(&seq, provider, wavelengths, &self.snapshot_groups(), ExpandOptions::deterministic())?;
    Ok((warnings, sa, spans))
  }

  /// Flatten with fabrication-error draws (gated).
  pub fn error_inputs(
    &self,
    provider: &dyn MaterialProvider,
    wavelengths: &[f64],
    seed: Option<u64>,
  ) -> Result<(Vec<String>, SolverArrays, Vec<Span>), String> {
    let warnings = self.gate(&self.validate(Some(provider)), "Navette_Structure")?;
    let seq: Vec<(Layer, bool)> = self.layers.iter().cloned().map(|l| (l, false)).collect();
    let (sa, spans) =
      expand(&seq, provider, wavelengths, &self.snapshot_groups(), ExpandOptions { apply_errors: true, seed })?;
    Ok((warnings, sa, spans))
  }

  /// Solver row count: exact via nominal expansion when a provider is set,
  /// else the structural approximation (interface ≈ +1).
  pub fn total_sub_layers(
    &self,
    provider: Option<&dyn MaterialProvider>,
    wavelengths: &[f64],
  ) -> usize {
    if self.layers.is_empty() {
      return 0;
    }
    if let Some(p) = provider {
      let seq: Vec<(Layer, bool)> = self.layers.iter().cloned().map(|l| (l, false)).collect();
      if let Ok((sa, _)) = expand(&seq, p, wavelengths, &self.snapshot_groups(), ExpandOptions::deterministic()) {
        return sa.n_rows();
      }
    }
    let mut total = 0;
    for (i, layer) in self.layers.iter().enumerate() {
      total += if layer.inhomogen && layer.sub_layer_count() > 1 { layer.sub_layer_count() as usize } else { 1 };
      if layer.interface && i > 0 {
        total += 1;
      }
    }
    total
  }

  pub fn total_physical_thickness(&self) -> f64 {
    self.layers.iter().map(|l| l.thickness).sum()
  }

  pub fn find_layers_by_material(&self, material: &str) -> Vec<usize> {
    self.layers.iter().enumerate().filter(|(_, l)| l.material == material).map(|(i, _)| i).collect()
  }

  /// Rename a material on all layers; returns the renamed count.
  pub fn replace_material(&mut self, old: &str, new: &str) -> usize {
    let mut count = 0;
    for layer in &mut self.layers {
      if layer.material == old {
        layer.material = new.to_string();
        count += 1;
      }
    }
    count
  }

  /// Layers eligible for optimization (flag + group THICKNESS slot).
  pub fn optimization_entries(&self) -> Vec<usize> {
    self
      .layers
      .iter()
      .enumerate()
      .filter(|(_, layer)| {
        if !layer.optimize {
          return false;
        }
        match self.groups.get(layer.material.as_str()) {
          Some(g) => g.borrow().optimization_mask[crate::enums::OptMask::Thickness as usize] != 0,
          None => true,
        }
      })
      .map(|(i, _)| i)
      .collect()
  }

  /// Write path for a group's optimization mask (binary, 7 slots).
  pub fn set_optimization_mask(&mut self, group_name: &str, mask: [i32; 7]) -> Result<(), String> {
    if mask.iter().any(|v| *v != 0 && *v != 1) {
      return Err("set_optimization_mask: mask must be 7 binary entries (see OptMask).".to_string());
    }
    match self.groups.get(group_name) {
      Some(g) => {
        g.borrow_mut().optimization_mask = mask;
        Ok(())
      }
      None => Err(format!("set_optimization_mask: unknown group '{group_name}'.")),
    }
  }

  /// Fold group film adjustments into the layers (film baking).
  /// Atomic: n/k-scaled groups refuse BEFORE any write (nothing half-baked).
  pub fn bake_films(&mut self) -> Result<usize, String> {
    for layer in &self.layers {
      if let Some(handle) = self.groups.get(layer.material.as_str()) {
        let group = handle.borrow();
        if group.n_factor != 1.0 || group.k_factor != 1.0 {
          return Err(format!(
            "bake_films: group '{}' has n/k scaling ({}, {}); run bake_materials() first.",
            group.group_name, group.n_factor, group.k_factor
          ));
        }
      }
    }
    let mut used: Vec<String> = Vec::new();
    for layer in &self.layers {
      if self.groups.contains_key(layer.material.as_str())
        && !used.iter().any(|n| n == layer.material.as_str())
      {
        used.push(layer.material.clone());
      }
    }
    let mut baked = 0;
    for i in 0..self.layers.len() {
      let material = self.layers[i].material.clone();
      if let Some(handle) = self.groups.get(material.as_str()) {
        let group = handle.borrow();
        let layer = &mut self.layers[i];
        layer.thickness = (layer.thickness * group.thick_factor + group.thick_summand).max(0.0);
        layer.inh_delta += group.inh_delta_summand;
        layer.roughness = (layer.roughness + group.roughness_summand).max(0.0);
        layer.interface_thickness += group.interface_summand;
        baked += 1;
      }
    }
    for name in &used {
      if let Some(group) = self.groups.get(name.as_str()) {
        let mut group = group.borrow_mut();
        group.thick_factor = 1.0;
        group.thick_summand = 0.0;
        group.inh_delta_summand = 0.0;
        group.roughness_summand = 0.0;
        group.interface_summand = 0.0;
      }
    }
    Ok(baked)
  }

  /// Fold group n/k scaling into new Table materials (material baking).
  /// Reads base nk via `provider`, registers specs into `target`, renames
  /// governed layers, resets n/k to (1,1); groups follow their material.
  pub fn bake_materials(
    &mut self,
    wavelengths: &[f64],
    provider: &dyn MaterialProvider,
    target: &mut crate::providers::DictProvider,
  ) -> Result<BTreeMap<String, String>, String> {
    if wavelengths.is_empty() {
      return Err("bake_materials: wavelengths must be non-empty.".to_string());
    }
    let mut materials: Vec<String> = Vec::new();
    for layer in &self.layers {
      if !materials.iter().any(|m| m == layer.material.as_str()) {
        materials.push(layer.material.clone());
      }
    }
    materials.sort();
    let mut mapping = BTreeMap::new();
    for material in &materials {
      let group = match self.groups.get(material.as_str()) {
        Some(g) => g.borrow().clone(),
        None => continue,
      };
      if group.n_factor == 1.0 && group.k_factor == 1.0 {
        continue;
      }
      let base = provider.nk(material.as_str(), wavelengths)?;
      if base.len() != wavelengths.len() {
        return Err(format!(
          "bake_materials: '{material}' has {} points, grid has {}.",
          base.len(),
          wavelengths.len()
        ));
      }
      let nk: Vec<Complex64> = base
        .iter()
        .map(|z| Complex64::new(z.re * group.n_factor, z.im * group.k_factor))
        .collect();
      let mut taken: HashSet<String> = target.names().cloned().collect();
      taken.extend(self.groups.keys().cloned());
      taken.extend(provider.names());
      let new_name = next_table_name(material, &taken);
      taken.insert(new_name.clone());
      let spec = crate::specs::MaterialSpec::new(
        "Table",
        BTreeMap::from([
          ("n_data".to_string(), json!([wavelengths, nk.iter().map(|z| z.re).collect::<Vec<_>>()])),
          ("k_data".to_string(), json!([wavelengths, nk.iter().map(|z| z.im).collect::<Vec<_>>()])),
        ]),
      );
      target.insert_spec(new_name.clone(), spec);
      mapping.insert(material.clone(), new_name);
    }
    // Establish the target grid when unset (specs need it to resolve).
    if target.grid().is_none() {
      target.set_grid(wavelengths.to_vec());
    }
    for (old, new) in &mapping {
      let handle = self.groups.remove(old.as_str()).expect("governed group present");
      {
        let mut group = handle.borrow_mut();
        group.n_factor = 1.0;
        group.k_factor = 1.0;
        group.group_name = new.clone();
      }
      self.groups.insert(new.clone(), handle);
      self.replace_material(old, new);
    }
    Ok(mapping)
  }

  /// Serialize layers + groups (materials travel outside the state).
  pub fn to_state(&self) -> Value {
    json!({
      "schema_version": SCHEMA_VERSION,
      "layers": self.layers.iter().map(|l| serde_json::to_value(l).unwrap()).collect::<Vec<_>>(),
      "groups": self.groups.iter().map(|(n, g)| (n, g.borrow().to_state())).collect::<BTreeMap<_, _>>(),
    })
  }

  /// Rebuild from state: version-checked; layers/groups default to empty.
  pub fn from_state(value: &Value) -> Result<Self, String> {
    let found = value.get("schema_version").and_then(|v| v.as_u64()).map(|v| v as u32);
    check_schema_version(found, "Navette_Structure")?;
    let layers = match value.get("layers") {
      None => Vec::new(),
      Some(v) => serde_json::from_value::<Vec<Layer>>(v.clone())
        .map_err(|e| format!("Navette_Structure: malformed layers ({e})"))?,
    };
    let mut groups = HashMap::new();
    if let Some(Value::Object(map)) = value.get("groups") {
      for (name, gs) in map {
        groups.insert(name.clone(), crate::shared_group(Group::from_state(gs)?));
      }
    }
    Ok(Self { layers, groups })
  }
}

/// Baked-material name: `X` → `X_table`; `X_table[N]` → `X_table[N+1]`
/// (case-insensitive terminal `table`), skipping taken names.
pub fn next_table_name(name: &str, taken: &HashSet<String>) -> String {
  let mut candidate = name.to_string();
  loop {
    candidate = advance_table_name(&candidate);
    if !taken.contains(&candidate) {
      return candidate;
    }
  }
}

fn advance_table_name(name: &str) -> String {
  let bytes = name.as_bytes();
  let mut digit_start = bytes.len();
  while digit_start > 0 && bytes[digit_start - 1].is_ascii_digit() {
    digit_start -= 1;
  }
  let (stem, digits) = (&name[..digit_start], &name[digit_start..]);
  if stem.len() >= 5 && stem[stem.len() - 5..].eq_ignore_ascii_case("table") {
    let next = if digits.is_empty() { 2 } else { digits.parse::<u64>().unwrap_or(1) + 1 };
    format!("{}table{next}", &stem[..stem.len() - 5])
  } else {
    format!("{name}_table")
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::providers::{DictProvider, Entry};

  fn wl() -> Vec<f64> {
    vec![1000.0, 1500.0]
  }

  fn mats() -> DictProvider {
    let mut entries = HashMap::new();
    entries.insert(
      "glass".to_string(),
      Entry::Array(vec![Complex64::new(1.52, 0.0), Complex64::new(1.51, 0.0)]),
    );
    entries.insert(
      "TiO2".to_string(),
      Entry::Array(vec![Complex64::new(2.35, 0.01), Complex64::new(2.33, 0.008)]),
    );
    DictProvider::with_grid(entries, wl()).unwrap()
  }

  fn glass_only() -> DictProvider {
    let mut entries = HashMap::new();
    entries.insert(
      "glass".to_string(),
      Entry::Array(vec![Complex64::new(1.52, 0.0), Complex64::new(1.51, 0.0)]),
    );
    DictProvider::with_grid(entries, wl()).unwrap()
  }

  fn flat() -> Structure {
    Structure::new(
      vec![Layer::film(0.0, "glass"), Layer::film(50.0, "TiO2"), Layer::film(0.0, "glass")],
      HashMap::new(),
    )
  }

  #[test]
  fn validate_collects_like_python() {
    // Oracle twin: negative thickness/roughness, overhang warning, orphan
    // warning, unknown material (messages mirror Python minus `warning: `).
    let mut groups = HashMap::new();
    groups.insert("Orphan".to_string(), Group::new("Orphan"));
    let st = Structure::new(vec![Layer {
      thickness: -5.0,
      material: "TiO2".to_string(),
      roughness: -1.0,
      interface: true,
      interface_thickness: 9.0,
      ..Layer::default()
    }], groups);
    let issues = st.validate(Some(&glass_only()));
    let texts: Vec<_> = issues.iter().map(|i| (i.is_error(), i.message.as_str())).collect();
    assert!(texts.contains(&(true, "Layer 0 (TiO2): Negative thickness -5 nm.")));
    assert!(texts.contains(&(true, "Layer 0 (TiO2): Negative roughness -1 nm.")));
    assert!(texts.iter().any(|(e, m)| !e && m.contains("Interface thickness (9) >= layer thickness (-5)")));
    assert!(texts.iter().any(|(e, m)| *e && m.contains("Material 'TiO2' not found")));
    assert!(texts.iter().any(|(e, m)| !e && m.contains("Group 'Orphan' governs no layer")));
    // Providerless skips material coverage + dry run.
    let free = st.validate(None);
    assert!(!free.iter().any(|i| i.message.contains("not found")));
  }

  #[test]
  fn gate_and_solve_paths() {
    let st = flat();
    let (warnings, sa, spans) = st.solver_inputs(&mats(), &wl()).unwrap();
    assert!(warnings.is_empty());
    assert_eq!(sa.n_rows(), 3);
    assert_eq!(spans.len(), 3);
    assert_eq!(st.total_sub_layers(Some(&mats()), &wl()), 3);
    assert!((st.total_physical_thickness() - 50.0).abs() < 1e-12);
    // Empty refused; unknown material fails the gate.
    let empty = Structure::default();
    assert!(empty.solver_inputs(&mats(), &wl()).is_err());
    let mut bad = flat();
    bad.layers[1].material = "Nope".to_string();
    assert!(bad.solver_inputs(&mats(), &wl()).is_err());
  }

  #[test]
  fn bake_films_matches_python() {
    let mut groups = HashMap::new();
    let mut g = Group::new("TiO2");
    g.thick_factor = 2.0;
    groups.insert("TiO2".to_string(), g);
    let mut st = Structure::new(vec![Layer::film(50.0, "TiO2")], groups);
    assert_eq!(st.bake_films().unwrap(), 1);
    assert_eq!(st.layers[0].thickness, 100.0);
    assert_eq!(st.groups["TiO2"].borrow().thick_factor, 1.0);
    // n/k refusal is pre-flight (atomic: nothing baked on Err).
    let mut groups = HashMap::new();
    let mut g = Group::new("TiO2");
    g.thick_factor = 2.0;
    g.n_factor = 1.1;
    groups.insert("TiO2".to_string(), g);
    let mut st = Structure::new(vec![Layer::film(50.0, "TiO2"), Layer::film(60.0, "TiO2")], groups);
    assert!(st.bake_films().is_err());
    assert_eq!(st.layers[0].thickness, 50.0);
  }

  #[test]
  fn bake_materials_naming_and_registers() {
    let mut groups = HashMap::new();
    let mut g = Group::new("TiO2");
    g.n_factor = 2.0;
    groups.insert("TiO2".to_string(), g);
    let mut st = Structure::new(vec![Layer::film(50.0, "TiO2")], groups);
    let mut target = DictProvider::new();
    let mapping = st.bake_materials(&wl(), &mats(), &mut target).unwrap();
    assert_eq!(mapping.get("TiO2").unwrap(), "TiO2_table");
    assert_eq!(st.layers[0].material, "TiO2_table");
    assert_eq!((st.groups["TiO2_table"].borrow().n_factor, st.groups["TiO2_table"].borrow().k_factor), (1.0, 1.0));
    let nk = target.nk("TiO2_table", &wl()).unwrap();
    assert_eq!(nk.len(), 2);
    assert!((nk[0].re - 4.7).abs() < 1e-12 && (nk[0].im - 0.01).abs() < 1e-12);
    // Naming chain + collision skip.
    let taken: HashSet<String> = ["TiO2_table".to_string()].into_iter().collect();
    assert_eq!(next_table_name("TiO2", &taken), "TiO2_table2");
    assert_eq!(next_table_name("TiO2_table", &HashSet::new()), "TiO2_table2");
    assert_eq!(next_table_name("TiO2_table2", &HashSet::new()), "TiO2_table3");
    assert_eq!(next_table_name("Xtable", &HashSet::new()), "Xtable2");
  }

  #[test]
  fn state_round_trip() {
    let mut groups = HashMap::new();
    groups.insert("TiO2".to_string(), Group::new("TiO2"));
    let st = Structure::new(vec![Layer::film(50.0, "TiO2")], groups);
    let back = Structure::from_state(&st.to_state()).unwrap();
    assert_eq!(back, st);
    let mut keys: Vec<_> = st.to_state().as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["groups", "layers", "schema_version"]);
    assert!(Structure::from_state(&json!({"layers": []})).is_err()); // untagged
  }

  #[test]
  fn helpers_behave() {
    let mut st = flat();
    assert_eq!(st.find_layers_by_material("TiO2"), vec![1]);
    assert_eq!(st.replace_material("TiO2", "Si"), 1);
    assert_eq!(st.optimization_entries(), vec![0, 1, 2]);
    st.set_optimization_mask("Nope", [1; 7]).unwrap_err();
    let mut groups = HashMap::new();
    groups.insert("Si".to_string(), crate::shared_group(Group::new("Si")));
    st.groups = groups;
    st.set_optimization_mask("Si", [1, 1, 1, 1, 1, 1, 1]).unwrap();
    assert!(st.set_optimization_mask("Si", [1, 1, 1, 1, 1, 1, 2]).is_err());
    // THICKNESS slot gates the entry list.
    st.set_optimization_mask("Si", [0, 1, 1, 1, 1, 1, 1]).unwrap();
    assert_eq!(st.optimization_entries(), vec![0, 2]);
  }
}
