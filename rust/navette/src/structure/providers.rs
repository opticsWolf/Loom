//! Material providers: name → nk with explicit grids.
//!
//! Mirrors `navette.structure.materials` providers, with the A3 promise
//! kept at the type level: the grid travels in the `nk()` signature, so a
//! provider returns values *known* to be on the requested grid instead of
//! values *hoped* to be. Gridless serving survives only as an explicit
//! `None` (same residual as Python gridless dicts, now visible).
//!
//! Conformance levels:
//! - exact-grid providers ([`DictProvider`]): stored grid must equal the
//!   request grid bit-for-bit, else `Err` (never resample silently);
//! - resampling providers (weaver-style, future): interpolate onto the
//!   request grid and document it.

use std::collections::HashMap;

use num_complex::Complex64;

/// Structural contract for material index sources.
///
/// `nk` returns complex `n + ik` on `wavelengths`; `contains` reports
/// servability without evaluating; `grid` exposes the provider's known
/// grid (`None` = unknown — the gridless residual).
pub trait MaterialProvider {
  fn nk(&self, name: &str, wavelengths: &[f64]) -> Result<Vec<Complex64>, String>;
  fn contains(&self, name: &str) -> bool;
  fn grid(&self) -> Option<&[f64]>;
  /// Shelved names (bake collision checks; defaults to none).
  fn names(&self) -> Vec<String> {
    Vec::new()
  }
}

/// Bit-exact grid equality (length + raw f64 bits, no tolerance).
/// Same idiom as the weaver fingerprints and the Python `_wl_sig` compare.
pub fn grids_equal(a: &[f64], b: &[f64]) -> bool {
  a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Bridge assert: provider grid (when known) must equal the solver grid.
/// Unknown grids pass (length discipline stays with the caller); mismatches
/// refuse with resample guidance. A grid change always re-resolves, so one
/// check per run covers every downstream solve.
pub fn assert_provider_grid(
  provider: &impl MaterialProvider,
  wavelengths: &[f64],
  what: &str,
) -> Result<(), String> {
  match provider.grid() {
    None => Ok(()),
    Some(g) if grids_equal(g, wavelengths) => Ok(()),
    Some(g) => Err(format!(
      "{what}: provider grid does not match the solver grid \
       ({} points vs {}); resample the material data onto the solver \
       wavelengths first.",
      g.len(),
      wavelengths.len()
    )),
  }
}

/// One shelf entry: an evaluated array or a recipe (mirrors the
/// `DictMaterialProvider` value kinds; `.nk` duck-objects stay Python-side).
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
  Array(Vec<Complex64>),
  Spec(crate::structure::specs::MaterialSpec),
}

/// Passive shelf for arrays and specs (mirrors `DictMaterialProvider`).
///
/// Holds the map by value; `refresh` swaps contents + grid atomically.
/// Array entries are grid-checked at serve time; spec entries evaluate on
/// the stored grid when known (mirroring Python), else on the request grid.
#[derive(Debug, Clone, Default)]
pub struct DictProvider {
  entries: HashMap<String, Entry>,
  grid: Option<Vec<f64>>,
}

impl DictProvider {
  /// Empty gridless shelf (length discipline only — the residual).
  pub fn new() -> Self {
    Self::default()
  }

  /// Shelf with a known grid: served arrays are length-checked, and `nk`
  /// requires the request grid to equal it bit-for-bit.
  pub fn with_grid(entries: HashMap<String, Entry>, grid: Vec<f64>) -> Result<Self, String> {
    for (name, entry) in &entries {
      if let Entry::Array(nk) = entry
        && nk.len() != grid.len() {
          return Err(format!(
            "DictProvider: '{name}' has {} points, grid has {}.",
            nk.len(),
            grid.len()
          ));
        }
    }
    Ok(Self { entries, grid: Some(grid) })
  }

  /// Atomically replace contents AND grid (the safe update path).
  pub fn refresh(&mut self, entries: HashMap<String, Entry>, grid: Option<Vec<f64>>) {
    self.entries = entries;
    self.grid = grid;
  }

  /// Insert one array entry (length-checked when the grid is known).
  pub fn insert_array(&mut self, name: impl Into<String>, nk: Vec<Complex64>) -> Result<(), String> {
    let name = name.into();
    if let Some(g) = &self.grid
      && nk.len() != g.len() {
        return Err(format!(
          "DictProvider: '{name}' has {} points, grid has {}.",
          nk.len(),
          g.len()
        ));
      }
    self.entries.insert(name, Entry::Array(nk));
    Ok(())
  }

  /// Insert one spec entry (grid-agnostic until served).
  pub fn insert_spec(&mut self, name: impl Into<String>, spec: crate::structure::specs::MaterialSpec) {
    self.entries.insert(name.into(), Entry::Spec(spec));
  }

  /// Establish the grid when unset (specs need it to resolve).
  pub fn set_grid(&mut self, grid: Vec<f64>) {
    self.grid = Some(grid);
  }

  /// Cloned (name, entry) pairs (binding snapshot + length checks).
  pub fn entries_snapshot(&self) -> Vec<(String, Entry)> {
    self.entries.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
  }

  /// Names currently shelved (bake naming checks providers AND groups).
  pub fn names(&self) -> impl Iterator<Item = &String> {
    self.entries.keys()
  }
}

impl MaterialProvider for DictProvider {
  fn nk(&self, name: &str, wavelengths: &[f64]) -> Result<Vec<Complex64>, String> {
    let entry =
      self.entries.get(name).ok_or_else(|| format!("DictProvider: unknown material '{name}'."))?;
    match &self.grid {
      Some(g) if !grids_equal(g, wavelengths) => Err(format!(
        "DictProvider: '{name}' lives on a {}-point grid, requested {} points; \
         resample first (exact-grid providers never interpolate silently).",
        g.len(),
        wavelengths.len()
      )),
      _ => match entry {
        Entry::Array(nk) => {
          if let Some(g) = &self.grid
            && nk.len() != g.len() {
              return Err(format!(
                "DictProvider: '{name}' has {} points, provider grid has {}.",
                nk.len(),
                g.len()
              ));
            }
          Ok(nk.clone())
        }
        // Stored grid known (and equal to the request, checked above):
        // evaluate on it like Python; else on the request grid.
        Entry::Spec(spec) => {
          let grid = self.grid.as_deref().unwrap_or(wavelengths);
          spec.evaluate(grid)
        }
      },
    }
  }

  fn contains(&self, name: &str) -> bool {
    self.entries.contains_key(name)
  }

  fn names(&self) -> Vec<String> {
    self.entries.keys().cloned().collect()
  }

  fn grid(&self) -> Option<&[f64]> {
    self.grid.as_deref()
  }
}

/// Spec/array library on one mandatory grid, with memoization (mirrors
/// `MaterialObjectProvider`). Arrays pass through (length-checked);
/// specs evaluate once per material and memoize. Grid is mandatory and
/// exact: `nk` refuses off-grid requests bit-for-bit.
#[derive(Debug, Default)]
pub struct SpecProvider {
  entries: HashMap<String, Entry>,
  grid: Vec<f64>,
  cache: std::cell::RefCell<HashMap<String, Vec<Complex64>>>,
}

impl SpecProvider {
  /// Library + shared grid. Arrays are length-checked eagerly.
  pub fn new(entries: HashMap<String, Entry>, grid: Vec<f64>) -> Result<Self, String> {
    for (name, entry) in &entries {
      if let Entry::Array(nk) = entry
        && nk.len() != grid.len() {
          return Err(format!(
            "SpecProvider: '{name}' has {} points, grid has {}.",
            nk.len(),
            grid.len()
          ));
        }
    }
    Ok(Self { entries, grid, cache: std::cell::RefCell::new(HashMap::new()) })
  }

  /// Reassign the grid. Byte-identical grids are a no-op; anything else
  /// clears the memo cache (mirrors the `tobytes` comparison).
  pub fn set_grid(&mut self, grid: Vec<f64>) {
    if !grids_equal(&grid, &self.grid) {
      self.grid = grid;
      self.cache.borrow_mut().clear();
    }
  }

  /// Upsert one entry and drop its memoized curve.
  pub fn upsert(&mut self, name: String, entry: Entry) {
    self.entries.insert(name.clone(), entry);
    self.cache.borrow_mut().remove(&name);
  }

  /// Drop memoized evaluations (one material, or all when `None`).
  /// Required after mutating a spec in place — the cache keys on the
  /// material name only and cannot detect param edits.
  pub fn invalidate(&self, material: Option<&str>) {
    match material {
      None => self.cache.borrow_mut().clear(),
      Some(name) => {
        self.cache.borrow_mut().remove(name);
      }
    }
  }
}

impl MaterialProvider for SpecProvider {
  fn nk(&self, name: &str, wavelengths: &[f64]) -> Result<Vec<Complex64>, String> {
    if !grids_equal(wavelengths, &self.grid) {
      return Err(format!(
        "SpecProvider: solve grid ({} points) != provider grid ({} points); \
         point the provider at the solve grid.",
        wavelengths.len(),
        self.grid.len()
      ));
    }
    if let Some(hit) = self.cache.borrow().get(name) {
      return Ok(hit.clone());
    }
    let entry =
      self.entries.get(name).ok_or_else(|| format!("SpecProvider: unknown material '{name}'."))?;
    let nk = match entry {
      Entry::Array(v) => {
        if v.len() != self.grid.len() {
          return Err(format!(
            "SpecProvider: '{name}' has {} points, grid has {}.",
            v.len(),
            self.grid.len()
          ));
        }
        v.clone()
      }
      Entry::Spec(spec) => spec.evaluate(&self.grid)?,
    };
    self.cache.borrow_mut().insert(name.to_string(), nk.clone());
    Ok(nk)
  }

  fn contains(&self, name: &str) -> bool {
    self.entries.contains_key(name)
  }

  fn names(&self) -> Vec<String> {
    self.entries.keys().cloned().collect()
  }

  fn grid(&self) -> Option<&[f64]> {
    Some(&self.grid)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn shelf() -> (DictProvider, Vec<f64>) {
    let grid = vec![900.0, 1000.0, 1100.0];
    let mut entries = HashMap::new();
    entries.insert("TiO2".to_string(), Entry::Array(vec![Complex64::new(2.35, 0.01); 3]));
    entries.insert("glass".to_string(), Entry::Array(vec![Complex64::new(1.52, 0.0); 3]));
    (DictProvider::with_grid(entries, grid.clone()).unwrap(), grid)
  }

  #[test]
  fn serve_contains_grid_match_python() {
    let (p, grid) = shelf();
    assert!(p.contains("TiO2"));
    assert!(!p.contains("Nope"));
    assert_eq!(p.nk("TiO2", &grid).unwrap(), vec![Complex64::new(2.35, 0.01); 3]);
    assert!(p.nk("Nope", &grid).is_err());
    assert_eq!(p.grid().unwrap(), grid.as_slice());
  }

  #[test]
  fn exact_grid_enforced_at_serve() {
    let (p, _) = shelf();
    // Same length, other values → refused (the Python silent case, closed).
    assert!(p.nk("TiO2", &[800.0, 900.0, 1000.0]).is_err());
    assert!(p.nk("TiO2", &[900.0, 1000.0]).is_err());
  }

  #[test]
  fn construction_and_insert_length_checked() {
    let mut bad = HashMap::new();
    bad.insert("x".to_string(), Entry::Array(vec![Complex64::new(1.0, 0.0); 5]));
    assert!(DictProvider::with_grid(bad, vec![1.0, 2.0]).is_err());
    let (mut p, grid) = shelf();
    assert!(p.insert_array("new", vec![Complex64::new(1.0, 0.0); 2]).is_err());
    p.insert_array("new", vec![Complex64::new(1.0, 0.0); 3]).unwrap();
    assert_eq!(p.nk("new", &grid).unwrap().len(), 3);
  }

  #[test]
  fn refresh_swaps_atomically() {
    let (mut p, _) = shelf();
    let mut entries = HashMap::new();
    entries.insert("Si".to_string(), Entry::Array(vec![Complex64::new(3.5, 0.0); 2]));
    p.refresh(entries, Some(vec![500.0, 600.0]));
    assert!(!p.contains("TiO2"));
    assert_eq!(p.nk("Si", &[500.0, 600.0]).unwrap().len(), 2);
    assert!(p.nk("Si", &[900.0, 1000.0, 1100.0]).is_err());
  }

  #[test]
  fn bridge_assert_passes_warns_refuses() {
    let (p, grid) = shelf();
    assert!(assert_provider_grid(&p, &grid, "solve").is_ok());
    assert!(assert_provider_grid(&p, &[800.0, 900.0, 1000.0], "solve").is_err());
    let gridless = DictProvider::new();
    assert!(assert_provider_grid(&gridless, &grid, "solve").is_ok());
  }

  #[test]
  fn grids_equal_is_bit_exact() {
    assert!(grids_equal(&[1.0, 2.0], &[1.0, 2.0]));
    assert!(!grids_equal(&[1.0, 2.0], &[1.0, 2.0, 3.0]));
    assert!(!grids_equal(&[1.0], &[1.0 + f64::EPSILON]));
  }
}

#[cfg(test)]
mod spec_tests {
  use super::*;
  use std::collections::BTreeMap;

  fn spec(model: &str, n: f64) -> Entry {
    let mut params = BTreeMap::new();
    params.insert("n".to_string(), serde_json::Value::from(n));
    Entry::Spec(crate::structure::specs::MaterialSpec::new(model, params))
  }

  fn lib() -> (SpecProvider, Vec<f64>) {
    let grid = vec![500.0, 600.0];
    let mut entries = HashMap::new();
    entries.insert("L".to_string(), spec("Konstant", 1.45));
    entries.insert(
      "A".to_string(),
      Entry::Array(vec![Complex64::new(2.0, 0.0), Complex64::new(2.0, 0.0)]),
    );
    (SpecProvider::new(entries, grid.clone()).unwrap(), grid)
  }

  #[test]
  fn memoizes_and_serves() {
    let (p, grid) = lib();
    let a = p.nk("L", &grid).unwrap();
    let b = p.nk("L", &grid).unwrap();
    assert_eq!(a, b);
    assert!((a[0].re - 1.45).abs() < 1e-15);
    assert!(p.nk("X", &grid).is_err());
    assert!(p.nk("L", &[500.0]).is_err());
  }

  #[test]
  fn grid_setter_clears_on_change_only() {
    let (mut p, grid) = lib();
    let _ = p.nk("L", &grid).unwrap();
    assert_eq!(p.cache.borrow().len(), 1);
    p.set_grid(grid.clone());
    assert_eq!(p.cache.borrow().len(), 1);
    p.set_grid(vec![500.0, 700.0]);
    assert_eq!(p.cache.borrow().len(), 0);
    assert!(p.nk("L", &grid).is_err());
  }

  #[test]
  fn invalidate_targets() {
    let (p, grid) = lib();
    let _ = p.nk("L", &grid).unwrap();
    let _ = p.nk("A", &grid).unwrap();
    p.invalidate(Some("L"));
    assert_eq!(p.cache.borrow().len(), 1);
    p.invalidate(None);
    assert_eq!(p.cache.borrow().len(), 0);
  }
}
