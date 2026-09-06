//! Positioned composition of structures: blocks, chains, global indices.
//!
//! Mirrors `navette.structure.architect.Navette_Architect`. Structures live
//! behind `Rc<RefCell<..>>` so sharing (edits propagate) and
//! `clone_structure` (aliasing broken) behave exactly like Python.
//! Providers stay outside (passed to calls that need them).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use serde_json::{Value, json};

use crate::structure::enums::{BlockKind, LayerType};
use crate::structure::expansion::{ExpandOptions, SolverArrays, Span, expand};
use crate::structure::group::Group;
use crate::structure::layer::Layer;
use crate::structure::providers::MaterialProvider;
use crate::structure::structure::Structure;
use crate::structure::validation::ValidationIssue;
use crate::structure::version::{SCHEMA_VERSION, check_schema_version};

/// Shared structure handle (edits through one block propagate to all
/// blocks referencing it — use `clone_structure` to break aliasing).
pub type SharedStructure = Rc<RefCell<Structure>>;

/// A positioned reference to a structure in the chain.
#[derive(Debug, Clone)]
pub struct Block {
  pub structure: SharedStructure,
  pub inverted: bool,
  pub repeat_count: usize,
  pub label: String,
  pub kind: BlockKind,
}

/// Positioned composition of structures with global-index addressing.
#[derive(Debug, Clone, Default)]
pub struct Architect {
  pub blocks: Vec<Block>,
}

impl Architect {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn is_empty(&self) -> bool {
    self.blocks.is_empty()
  }

  /// Append an owned structure (fresh shared handle).
  pub fn add_structure(
    &mut self,
    structure: Structure,
    inverted: bool,
    repeat: usize,
    label: impl Into<String>,
    kind: BlockKind,
  ) -> Result<(), String> {
    self.push_shared(Rc::new(RefCell::new(structure)), inverted, repeat, label, kind)
  }

  /// Append an already-shared structure (aliasing: edits propagate).
  pub fn add_shared(
    &mut self,
    structure: SharedStructure,
    inverted: bool,
    repeat: usize,
    label: impl Into<String>,
    kind: BlockKind,
  ) -> Result<(), String> {
    self.push_shared(structure, inverted, repeat, label, kind)
  }

  fn push_shared(
    &mut self,
    structure: SharedStructure,
    inverted: bool,
    repeat: usize,
    label: impl Into<String>,
    kind: BlockKind,
  ) -> Result<(), String> {
    if repeat < 1 {
      return Err("repeat_count must be >= 1".to_string());
    }
    self.blocks.push(Block { structure, inverted, repeat_count: repeat, label: label.into(), kind });
    Ok(())
  }

  /// Clone the structure at block `index`, breaking aliasing.
  pub fn clone_structure(&mut self, index: usize) -> Result<(), String> {
    if index >= self.blocks.len() {
      return Err("Block index out of range".to_string());
    }
    let block = &self.blocks[index];
    let cloned = block.structure.borrow().clone();
    self.blocks[index] = Block {
      structure: Rc::new(RefCell::new(cloned)),
      inverted: block.inverted,
      repeat_count: block.repeat_count,
      label: block.label.clone(),
      kind: block.kind,
    };
    Ok(())
  }

  /// Deduplicated structures by handle identity (insertion order).
  pub fn unique_structures(&self) -> Vec<SharedStructure> {
    let mut seen: HashSet<*const RefCell<Structure>> = HashSet::new();
    let mut out = Vec::new();
    for block in &self.blocks {
      let key = Rc::as_ptr(&block.structure);
      if seen.insert(key) {
        out.push(block.structure.clone());
      }
    }
    out
  }

  /// Total logical layers (repeat-aware, pre-expansion).
  pub fn global_layer_count(&self) -> usize {
    self.blocks.iter().map(|b| b.structure.borrow().layers.len() * b.repeat_count).sum()
  }

  /// Order-only traversal: originals last-first with the flag set under
  /// inversion; all mirror physics lives in the expander.
  pub fn iter_entries(&self) -> Vec<(Layer, bool)> {
    let mut out = Vec::new();
    for block in &self.blocks {
      let borrowed = block.structure.borrow();
      let n = borrowed.layers.len();
      if n == 0 {
        continue;
      }
      for _ in 0..block.repeat_count {
        if block.inverted {
          for i in (0..n).rev() {
            out.push((borrowed.layers[i].clone(), true));
          }
        } else {
          for layer in borrowed.layers.iter() {
            out.push((layer.clone(), false));
          }
        }
      }
    }
    out
  }

  /// Merge group dicts across unique structures; identical states merge,
  /// conflicting definitions refuse.
  pub fn merged_groups(&self) -> Result<HashMap<String, Group>, String> {
    let mut merged: HashMap<String, Group> = HashMap::new();
    for shared in self.unique_structures() {
      let borrowed = shared.borrow();
      for (name, handle) in &borrowed.groups {
        let group = handle.borrow();
        if let Some(existing) = merged.get(name) {
          if existing != &*group {
            return Err(format!(
              "Group name '{name}' defined differently in two structures. Cannot merge automatically. Use unique group names or ensure consistency."
            ));
          }
        } else {
          merged.insert(name.clone(), group.clone());
        }
      }
    }
    Ok(merged)
  }

  /// Block composition rules (chain must start/end with STACK; markers).
  pub fn validate_chain(&self) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if self.blocks.is_empty() {
      issues.push(ValidationIssue::error("Navette_Architect chain is empty."));
      return issues;
    }
    if self.blocks[0].kind != BlockKind::Stack {
      issues.push(ValidationIssue::error("Chain must start with a STACK block."));
    }
    if self.blocks[self.blocks.len() - 1].kind != BlockKind::Stack {
      issues.push(ValidationIssue::error("Chain must end with a STACK block."));
    }
    for (b, block) in self.blocks.iter().enumerate() {
      let borrowed = block.structure.borrow();
      let layers = &borrowed.layers;
      if layers.is_empty() {
        issues.push(ValidationIssue::error(format!("Block {b} ('{}'): empty structure.", block.label)));
        continue;
      }
      let roles: Vec<LayerType> = layers.iter().map(|l| l.layer_type).collect();
      if block.kind == BlockKind::Films {
        if roles.iter().any(|r| *r != LayerType::Film) {
          issues.push(ValidationIssue::error(format!(
            "Block {b} ('{}'): FILMS block must hold FILM rows only.",
            block.label
          )));
        }
      } else {
        let marked: Vec<&LayerType> =
          roles.iter().filter(|r| **r == LayerType::Ambient || **r == LayerType::Substrate).collect();
        if !marked.is_empty() {
          if roles[0] != LayerType::Ambient {
            issues.push(ValidationIssue::error(format!(
              "Block {b} ('{}'): STACK must open with an AMBIENT row.",
              block.label
            )));
          }
          if roles[roles.len() - 1] != LayerType::Substrate {
            issues.push(ValidationIssue::error(format!(
              "Block {b} ('{}'): STACK must close with a SUBSTRATE row.",
              block.label
            )));
          }
          if roles.len() > 2
            && roles[1..roles.len() - 1]
              .iter()
              .any(|r| *r == LayerType::Ambient || *r == LayerType::Substrate)
          {
            issues.push(ValidationIssue::error(format!(
              "Block {b} ('{}'): half-space markers inside the film sequence.",
              block.label
            )));
          }
        }
      }
    }
    issues
  }

  /// Merge conflicts + chain + each unique structure's own validation
  /// (deduplicated). Never raises.
  pub fn validate(&self, provider: Option<&dyn MaterialProvider>) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if let Err(e) = self.merged_groups() { issues.push(ValidationIssue::error(e)) }
    issues.extend(self.validate_chain());
    for shared in self.unique_structures() {
      for issue in shared.borrow().validate(provider) {
        if !issues.contains(&issue) {
          issues.push(issue);
        }
      }
    }
    issues
  }

  /// Flatten the whole chain (gated). Returns warnings + arrays + spans.
  pub fn solver_inputs(
    &self,
    provider: &dyn MaterialProvider,
    wavelengths: &[f64],
  ) -> Result<(Vec<String>, SolverArrays, Vec<Span>), String> {
    if self.blocks.is_empty() {
      return Err("Navette_Architect is empty.".to_string());
    }
    let issues = self.validate(Some(provider));
    let (errors, warnings): (Vec<_>, Vec<_>) = issues.iter().partition(|i| i.is_error());
    if !errors.is_empty() {
      return Err(format!(
        "Navette_Architect invalid:\n{}",
        errors.iter().map(|e| e.message.as_str()).collect::<Vec<_>>().join("\n")
      ));
    }
    let merged = self.merged_groups().map_err(|e| format!("Navette_Architect invalid:\n{e}"))?;
    let seq = self.iter_entries();
    let (sa, spans) = expand(&seq, provider, wavelengths, &merged, ExpandOptions::deterministic())?;
    Ok((warnings.iter().map(|w| w.message.clone()).collect(), sa, spans))
  }

  /// Flatten with fabrication-error draws (gated).
  pub fn error_inputs(
    &self,
    provider: &dyn MaterialProvider,
    wavelengths: &[f64],
    seed: Option<u64>,
  ) -> Result<(Vec<String>, SolverArrays, Vec<Span>), String> {
    if self.blocks.is_empty() {
      return Err("Navette_Architect is empty.".to_string());
    }
    let issues = self.validate(Some(provider));
    let (errors, warnings): (Vec<_>, Vec<_>) = issues.iter().partition(|i| i.is_error());
    if !errors.is_empty() {
      return Err(format!(
        "Navette_Architect invalid:\n{}",
        errors.iter().map(|e| e.message.as_str()).collect::<Vec<_>>().join("\n")
      ));
    }
    let merged = self.merged_groups().map_err(|e| format!("Navette_Architect invalid:\n{e}"))?;
    let seq = self.iter_entries();
    let (sa, spans) =
      expand(&seq, provider, wavelengths, &merged, ExpandOptions { apply_errors: true, seed })?;
    Ok((warnings.iter().map(|w| w.message.clone()).collect(), sa, spans))
  }

  /// Global logical index → (block, local). Inverted blocks address their
  /// originals last-first (mirror of `_iter_layers` order).
  pub fn map_global(&self, global_idx: usize) -> Result<(usize, usize), String> {
    let mut current = 0;
    for (bi, block) in self.blocks.iter().enumerate() {
      let n = block.structure.borrow().layers.len();
      for _ in 0..block.repeat_count {
        if current <= global_idx && global_idx < current + n {
          let local_offset = global_idx - current;
          let local =
            if block.inverted { (n - 1) - local_offset } else { local_offset };
          return Ok((bi, local));
        }
        current += n;
      }
    }
    Err(format!("Global index {global_idx} out of bounds (total logical layers: {current})"))
  }

  /// Solver row → (block, local) via one nominal expansion + span lookup.
  pub fn map_solver(
    &self,
    provider: &dyn MaterialProvider,
    wavelengths: &[f64],
    solver_idx: usize,
  ) -> Result<(usize, usize), String> {
    let merged = self.merged_groups().map_err(|e| format!("Navette_Architect invalid:\n{e}"))?;
    let seq = self.iter_entries();
    let (_, spans) = expand(&seq, provider, wavelengths, &merged, ExpandOptions::deterministic())?;
    for s in &spans {
      if s.start <= solver_idx && solver_idx < s.end {
        return self.map_global(s.logical);
      }
    }
    Err(format!("Solver index {solver_idx} out of bounds."))
  }

  /// Layer clone at a chain-wide index.
  pub fn layer_at_global(&self, global_idx: usize) -> Result<Layer, String> {
    let (bi, local) = self.map_global(global_idx)?;
    Ok(self.blocks[bi].structure.borrow().layers[local].clone())
  }

  /// Insert at a chain-wide index (mutates the underlying structure:
  /// shared references see the change, like Python).
  pub fn insert_at_global(&self, global_idx: usize, layer: Layer) -> Result<(), String> {
    let (bi, local) = self.map_global(global_idx)?;
    self.blocks[bi].structure.borrow_mut().layers.insert(local, layer);
    Ok(())
  }

  /// Split into two same-material halves (thickness by ratio; interface
  /// definition preserved verbatim on both — process parameter, not bulk).
  pub fn split_at_global(&self, global_idx: usize, split_ratio: f64) -> Result<(), String> {
    let (bi, local) = self.map_global(global_idx)?;
    let mut borrowed = self.blocks[bi].structure.borrow_mut();
    let original = borrowed.layers[local].clone();
    let mut l1 = original.clone();
    let mut l2 = original.clone();
    l1.thickness = original.thickness * split_ratio;
    l2.thickness = original.thickness * (1.0 - split_ratio);
    borrowed.layers[local] = l1;
    borrowed.layers.insert(local + 1, l2);
    Ok(())
  }

  /// Verbatim-clone the layer at a chain-wide index.
  pub fn duplicate_at_global(&self, global_idx: usize) -> Result<(), String> {
    let (bi, local) = self.map_global(global_idx)?;
    let mut borrowed = self.blocks[bi].structure.borrow_mut();
    let copy = borrowed.layers[local].clone();
    borrowed.layers.insert(local, copy);
    Ok(())
  }

  /// Remove the layer at a chain-wide index.
  pub fn remove_at_global(&self, global_idx: usize) -> Result<(), String> {
    let (bi, local) = self.map_global(global_idx)?;
    if local >= self.blocks[bi].structure.borrow().layers.len() {
      return Err(format!("Global index {global_idx} out of bounds."));
    }
    self.blocks[bi].structure.borrow_mut().layers.remove(local);
    Ok(())
  }

  /// Remove sub-threshold layers from ALL unique structures.
  pub fn prune_thin_layers(&self, min_thickness: f64) -> usize {
    let mut removed = 0;
    for shared in self.unique_structures() {
      let mut borrowed = shared.borrow_mut();
      let before = borrowed.layers.len();
      borrowed.layers.retain(|l| l.thickness >= min_thickness);
      removed += before - borrowed.layers.len();
    }
    removed
  }

  /// Unique (block, local) entries eligible for optimization.
  pub fn optimization_entries(&self) -> Vec<(usize, usize)> {
    let merged = self.merged_groups().unwrap_or_default();
    let mut out = Vec::new();
    for shared in self.unique_structures() {
      let borrowed = shared.borrow();
      let bi = self.blocks.iter().position(|b| Rc::ptr_eq(&b.structure, &shared)).unwrap_or(0);
      for (local, layer) in borrowed.layers.iter().enumerate() {
        if !layer.optimize {
          continue;
        }
        match merged.get(layer.material.as_str()) {
          Some(g) if g.optimization_mask[crate::structure::enums::OptMask::Thickness as usize] == 0 => continue,
          _ => {}
        }
        out.push((bi, local));
      }
    }
    out
  }

  /// Write path for a group's optimization mask (binary, 7 slots).
  pub fn set_optimization_mask(&self, group_name: &str, mask: [i32; 7]) -> Result<(), String> {
    if mask.iter().any(|v| *v != 0 && *v != 1) {
      return Err("set_optimization_mask: mask must be 7 binary entries (see OptMask).".to_string());
    }
    for shared in self.unique_structures() {
      if shared.borrow().groups.contains_key(group_name) {
        shared.borrow().groups[group_name].borrow_mut().optimization_mask = mask;
        return Ok(());
      }
    }
    Err(format!("set_optimization_mask: unknown group '{group_name}'."))
  }

  /// Fold every unique structure's film adjustments into its layers.
  pub fn bake_films(&self) -> Result<usize, String> {
    let mut total = 0;
    for shared in self.unique_structures() {
      total += shared.borrow_mut().bake_films()?;
    }
    Ok(total)
  }

  /// Fold every unique structure's n/k scaling into new materials.
  pub fn bake_materials(
    &self,
    wavelengths: &[f64],
    provider: &dyn MaterialProvider,
    target: &mut crate::structure::providers::DictProvider,
  ) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut merged = std::collections::BTreeMap::new();
    for shared in self.unique_structures() {
      for (old, new) in shared.borrow_mut().bake_materials(wavelengths, provider, target)? {
        merged.insert(old, new);
      }
    }
    Ok(merged)
  }

  /// Solver row count: exact via nominal expansion when possible, else the
  /// structural approximation over the traversal (interface ≈ +1 past the
  /// first entry).
  pub fn total_sub_layers(
    &self,
    provider: Option<&dyn MaterialProvider>,
    wavelengths: &[f64],
  ) -> usize {
    if let Some(p) = provider
      && let Ok((_, sa, _)) = self.solver_inputs(p, wavelengths) {
        return sa.n_rows();
      }
    let mut total = 0;
    let mut prev_exists = false;
    for (layer, _) in self.iter_entries() {
      total += if layer.inhomogen && layer.sub_layer_count() > 1 {
        layer.sub_layer_count() as usize
      } else {
        1
      };
      if layer.interface && prev_exists {
        total += 1;
      }
      prev_exists = true;
    }
    total
  }

  /// Serialize structures + blocks (reference-preserving via indices).
  pub fn to_state(&self) -> Value {
    let uniques = self.unique_structures();
    let mut index_of: HashMap<*const RefCell<Structure>, usize> = HashMap::new();
    for (i, shared) in uniques.iter().enumerate() {
      index_of.insert(Rc::as_ptr(shared), i);
    }
    json!({
      "schema_version": SCHEMA_VERSION,
      "structures": uniques.iter().map(|s| s.borrow().to_state()).collect::<Vec<_>>(),
      "blocks": self.blocks.iter().map(|b| json!({
        "structure_ref": index_of[&(Rc::as_ptr(&b.structure))],
        "inverted": b.inverted,
        "repeat_count": b.repeat_count,
        "label": b.label,
        "kind": b.kind as i32,
      })).collect::<Vec<_>>(),
    })
  }

  /// Rebuild from state: version-checked; out-of-range refs refuse.
  pub fn from_state(value: &Value) -> Result<Self, String> {
    let found = value.get("schema_version").and_then(|v| v.as_u64()).map(|v| v as u32);
    check_schema_version(found, "Navette_Architect")?;
    let structs: Vec<SharedStructure> = match value.get("structures") {
      None => Vec::new(),
      Some(v) => v
        .as_array()
        .ok_or_else(|| "Navette_Architect: 'structures' must be an array.".to_string())?
        .iter()
        .map(|ss| Structure::from_state(ss).map(|s| Rc::new(RefCell::new(s)) as SharedStructure))
        .collect::<Result<Vec<_>, _>>()?,
    };
    let mut arch = Self::new();
    let blocks = value
      .get("blocks")
      .and_then(|v| v.as_array())
      .ok_or_else(|| "Navette_Architect: 'blocks' must be an array.".to_string())?;
    for (bi, bs) in blocks.iter().enumerate() {
      let rf = bs.get("structure_ref").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
      if rf >= structs.len() {
        return Err(format!("from_state: block {bi} has out-of-range structure_ref {rf}."));
      }
      let kind = bs
        .get("kind")
        .and_then(|v| v.as_i64())
        .map(|v| crate::structure::enums::BlockKind::try_from_i32(v as i32))
        .transpose()?
        .unwrap_or(crate::structure::enums::BlockKind::Stack);
      arch.add_shared(
        structs[rf].clone(),
        bs.get("inverted").and_then(|v| v.as_bool()).unwrap_or(false),
        bs.get("repeat_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize,
        bs.get("label").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        kind,
      )?;
    }
    Ok(arch)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::structure::providers::{DictProvider, Entry};
  use num_complex::Complex64;

  fn wl() -> Vec<f64> {
    vec![1000.0]
  }

  fn mats() -> DictProvider {
    let mut entries = HashMap::new();
    entries.insert("glass".to_string(), Entry::Array(vec![Complex64::new(1.52, 0.0)]));
    entries.insert("TiO2".to_string(), Entry::Array(vec![Complex64::new(2.35, 0.01)]));
    DictProvider::with_grid(entries, wl()).unwrap()
  }

  fn flat() -> Structure {
    Structure::new(
      vec![Layer::film(0.0, "glass"), Layer::film(50.0, "TiO2"), Layer::film(0.0, "glass")],
      HashMap::new(),
    )
  }

  #[test]
  fn chain_rules_match_python() {
    let mats = mats();
    let mut ok = Architect::new();
    ok.add_structure(flat(), false, 1, "", BlockKind::Stack).unwrap();
    assert!(ok.validate(Some(&mats)).is_empty());
    let mut bad = Architect::new();
    bad.add_structure(flat(), false, 1, "", BlockKind::Films).unwrap();
    let issues = bad.validate_chain();
    assert!(issues.iter().any(|i| i.message.contains("must start with a STACK")));
    assert!(issues.iter().any(|i| i.message.contains("must end with a STACK")));
    let empty = Architect::new();
    assert!(empty.validate_chain().iter().any(|i| i.message.contains("chain is empty")));
    assert!(empty.solver_inputs(&mats, &wl()).is_err());
    // add_structure refuses repeat < 1; clone_structure bounds-checked.
    let mut a = Architect::new();
    assert!(a.add_structure(flat(), false, 0, "", BlockKind::Stack).is_err());
    assert!(a.clone_structure(0).is_err());
  }

  #[test]
  fn marked_stacks_and_films_blocks() {
    let mut films_ok = Structure::new(vec![Layer::film(10.0, "TiO2")], HashMap::new());
    let _ = &mut films_ok;
    let mut marked = Structure::new(
      vec![
        Layer { layer_type: LayerType::Ambient, ..Layer::film(0.0, "glass") },
        Layer::film(50.0, "TiO2"),
        Layer { layer_type: LayerType::Substrate, ..Layer::film(0.0, "glass") },
      ],
      HashMap::new(),
    );
    let _ = &mut marked;
    let mut a = Architect::new();
    a.add_structure(marked, false, 1, "", BlockKind::Stack).unwrap();
    assert!(a.validate_chain().is_empty());
    // Interior marker trips the rule.
    let mut bad_mark = Structure::new(
      vec![
        Layer { layer_type: LayerType::Ambient, ..Layer::film(0.0, "glass") },
        Layer { layer_type: LayerType::Ambient, ..Layer::film(50.0, "TiO2") },
        Layer { layer_type: LayerType::Substrate, ..Layer::film(0.0, "glass") },
      ],
      HashMap::new(),
    );
    let _ = &mut bad_mark;
    let mut a2 = Architect::new();
    a2.add_structure(bad_mark, false, 1, "", BlockKind::Stack).unwrap();
    assert!(a2.validate_chain().iter().any(|i| i.message.contains("inside the film sequence")));
    // FILMS with a marked row trips.
    let mut a3 = Architect::new();
    a3.add_structure(
      Structure::new(vec![Layer { layer_type: LayerType::Ambient, ..Layer::film(0.0, "glass") }], HashMap::new()),
      false, 1, "", BlockKind::Films,
    ).unwrap();
    assert!(a3.validate_chain().iter().any(|i| i.message.contains("FILM rows only")));
  }

  #[test]
  fn global_mapping_mirrors_python() {
    let mut a = Architect::new();
    a.add_shared(Rc::new(RefCell::new(flat())), false, 1, "", BlockKind::Stack).unwrap();
    a.add_shared(Rc::new(RefCell::new(flat())), true, 1, "", BlockKind::Stack).unwrap();
    // Forward block: identity; inverted block: last-first.
    assert_eq!(a.map_global(0).unwrap(), (0, 0));
    assert_eq!(a.map_global(2).unwrap(), (0, 2));
    assert_eq!(a.map_global(3).unwrap(), (1, 2));
    assert_eq!(a.map_global(5).unwrap(), (1, 0));
    assert!(a.map_global(6).is_err());
    assert_eq!(a.global_layer_count(), 6);
    // Solver rows resolve through spans to logical layers.
    let mats = mats();
    assert_eq!(a.map_solver(&mats, &wl(), 4).unwrap(), (1, 1));
    assert!(a.map_solver(&mats, &wl(), 99).is_err());
  }

  #[test]
  fn sharing_and_mutation_match_python() {
    let shared = Rc::new(RefCell::new(flat()));
    let mut a = Architect::new();
    a.add_shared(shared.clone(), false, 1, "", BlockKind::Stack).unwrap();
    a.add_shared(shared.clone(), false, 1, "", BlockKind::Stack).unwrap();
    assert_eq!(a.unique_structures().len(), 1);
    // Mutation through a global index propagates to both blocks.
    a.split_at_global(1, 0.5).unwrap();
    assert_eq!(shared.borrow().layers.len(), 4);
    assert!((shared.borrow().layers[1].thickness - 25.0).abs() < 1e-12);
    // clone_structure breaks the alias.
    a.clone_structure(1).unwrap();
    assert_eq!(a.unique_structures().len(), 2);
    a.remove_at_global(0).unwrap();
    assert_eq!(a.blocks[0].structure.borrow().layers.len(), 3);
    assert_eq!(a.blocks[1].structure.borrow().layers.len(), 4);
  }

  #[test]
  fn merge_conflict_refuses() {
    let mut s1 = flat();
    s1.groups.insert("g".to_string(), crate::structure::shared_group(Group::new("g")));
    let mut s2 = flat();
    let mut g2 = Group::new("g");
    g2.thick_factor = 2.0;
    s2.groups.insert("g".to_string(), crate::structure::shared_group(g2));
    let mut a = Architect::new();
    a.add_structure(s1, false, 1, "", BlockKind::Stack).unwrap();
    a.add_structure(s2, false, 1, "", BlockKind::Stack).unwrap();
    assert!(a.merged_groups().is_err());
    assert!(a.validate(Some(&mats())).iter().any(|i| i.message.contains("defined differently")));
  }

  #[test]
  fn state_preserves_sharing() {
    let shared = Rc::new(RefCell::new(flat()));
    let mut a = Architect::new();
    a.add_shared(shared, false, 2, "x", BlockKind::Stack).unwrap();
    let back = Architect::from_state(&a.to_state()).unwrap();
    assert_eq!(back.blocks.len(), 1);
    assert_eq!(back.blocks[0].repeat_count, 2);
    assert_eq!(back.blocks[0].label, "x");
    // Untagged + bad-ref states refuse.
    assert!(Architect::from_state(&json!({"structures": [], "blocks": []})).is_err());
    let mut bad = a.to_state();
    bad["blocks"][0]["structure_ref"] = json!(7);
    assert!(Architect::from_state(&bad).is_err());
  }
}
