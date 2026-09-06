//! Native woven-grid provider (Phase E, triggered: Rust-only runs must
//! serve measured n/k curves without touching Python).
//!
//! Faithful port of Python `WeaverMaterialProvider`: per material the `n`
//! fragment is REQUIRED (absent → error) while a missing `k` fragment
//! silently defaults to zeros. An exact-grid fast path (shape AND values,
//! numpy `array_equal` semantics: `-0.0 == 0.0`, `NaN != NaN`) skips
//! interpolation; otherwise the SAME `navette-interpolate::UniInterpolator`
//! kernel Python calls resamples — bit-identity by construction, pinned by
//! frozen oracles. `strict` refuses off-grid fragments instead of
//! interpolating (absent k still defaults to zeros — absence is not
//! staleness). Contents are memoized per material; the cache cannot see
//! backend edits (`invalidate`, target reassignment).
//!
//! Caller-grid contract: the provider serves ONE grid (`target`). `nk`
//! refuses any other grid — the check Python's `solve_structure` bridge
//! performs, enforced here so Rust-only runs stay grid-rigorous.

use std::cell::RefCell;
use std::collections::HashMap;

use num_complex::Complex64;

use crate::structure::providers::{MaterialProvider, grids_equal};

/// Woven-fragment backend: `(prefix, label, material)` → `(grid, values)`.
///
/// `None` = unknown key OR empty fragment (both are "absent" upstream).
pub trait WovenBackend {
  fn weaved(&self, prefix: f64, label: &str, material: &str) -> Option<(Vec<f64>, Vec<f64>)>;
}

/// Normalize a key prefix to Python-dict lookup semantics: `NaN` never
/// hits (Python `nan != nan`), `-0.0` hits `0.0` (Python `-0.0 == 0.0`).
/// Returns `None` for "never present".
pub fn norm_prefix(prefix: f64) -> Option<f64> {
  if prefix.is_nan() {
    None
  } else if prefix == 0.0 {
    Some(0.0)
  } else {
    Some(prefix)
  }
}

impl WovenBackend for std::sync::Arc<crate::spectralweave::opticalweaver::OpticalWeaver> {
  fn weaved(&self, prefix: f64, label: &str, material: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    (**self).weaved(prefix, label, material)
  }
}

impl WovenBackend for crate::spectralweave::opticalweaver::OpticalWeaver {
  fn weaved(&self, prefix: f64, label: &str, material: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    let p = norm_prefix(prefix)?;
    let key = crate::spectralweave::opticalweaver::OpticalKey {
      wavelength: p,
      data_type: label.into(),
      polarisation: material.into(),
    };
    match self.get_weaved(&key) {
      Err(_) => None,
      Ok((w, _)) if w.is_empty() => None,
      Ok(v) => Some(v),
    }
  }
}

/// Interpolation recipe. Defaults mirror Python `InterpolationSettings`
/// (`linear`, `d=3`, non-robust); `extrap` is always `"linear"` (the
/// binding default `materials.py` relies on by not passing it).
#[derive(Debug, Clone)]
pub struct InterpSettings {
  pub method: String,
  pub robust: bool,
  pub fh_d: usize,
}

impl Default for InterpSettings {
  fn default() -> Self {
    Self { method: "linear".to_string(), robust: false, fh_d: 3 }
  }
}

/// numpy `array_equal` on float grids: same length AND `==` element-wise
/// (`-0.0` equals `0.0`, `NaN` never equals — NOT bit comparison).
fn grids_value_eq(a: &[f64], b: &[f64]) -> bool {
  a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Woven n/k curves resampled onto one target grid. `B` is usually
/// [`crate::spectralweave::opticalweaver::OpticalWeaver`].
/// Single-threaded memoization (`RefCell`, like the Python dict cache).
pub struct WeaverProvider<B> {
  backend: B,
  target: Vec<f64>,
  key_prefix: f64,
  n_label: String,
  k_label: String,
  interp: InterpSettings,
  strict: bool,
  cache: RefCell<HashMap<String, Vec<Complex64>>>,
}

impl<B: WovenBackend> WeaverProvider<B> {
  pub fn new(
    backend: B,
    target: Vec<f64>,
    key_prefix: f64,
    n_label: &str,
    k_label: &str,
    interp: InterpSettings,
    strict: bool,
  ) -> Self {
    Self {
      backend,
      target,
      key_prefix,
      n_label: n_label.to_string(),
      k_label: k_label.to_string(),
      interp,
      strict,
      cache: RefCell::new(HashMap::new()),
    }
  }

  pub fn set_strict(&mut self, strict: bool) {
    self.strict = strict;
  }

  pub fn strict(&self) -> bool {
    self.strict
  }

  /// Reassign the target grid. Byte-identical grids are a no-op; anything
  /// else clears the cache (mirrors the `tobytes` comparison).
  pub fn set_target(&mut self, wavelengths: Vec<f64>) {
    if !grids_equal(&wavelengths, &self.target) {
      self.target = wavelengths;
      self.cache.borrow_mut().clear();
    }
  }

  /// True when the weave sits exactly on the target grid (probes the `n`
  /// fragment; false for unknown materials). No fallback, no serving.
  pub fn is_exact(&self, material: &str) -> bool {
    match self.backend.weaved(self.key_prefix, &self.n_label, material) {
      None => false,
      Some((w, _)) => grids_value_eq(&w, &self.target),
    }
  }

  /// Drop memoized curves (one material, or all when `None`). Required
  /// after re-weaving the backend.
  pub fn invalidate(&self, material: Option<&str>) {
    match material {
      None => self.cache.borrow_mut().clear(),
      Some(name) => {
        self.cache.borrow_mut().remove(name);
      }
    }
  }

  fn fetch(&self, label: &str, material: &str) -> Result<Option<Vec<f64>>, String> {
    let key_repr = format!("({}, '{}', '{}')", self.key_prefix, label, material);
    let (src_wl, src_data) = match self.backend.weaved(self.key_prefix, label, material) {
      None => return Ok(None),
      Some(v) => v,
    };
    if grids_value_eq(&src_wl, &self.target) {
      return Ok(Some(src_data));
    }
    if self.strict {
      return Err(format!(
        "WeaverMaterialProvider(strict): weave {} is not on the target grid \
         ({} vs {} points); re-weave onto the target grid or disable strict mode.",
        key_repr,
        src_wl.len(),
        self.target.len()
      ));
    }
    let n = src_wl.len();
    let x = ndarray017::Array1::from_vec(src_wl);
    let y = ndarray017::Array2::from_shape_vec((1, n), src_data)
      .map_err(|e| format!("WeaverMaterialProvider: fragment shape: {e}"))?;
    let spline = crate::interpolate::UniInterpolator::new(
      x,
      y,
      false,
      &self.interp.method,
      self.interp.robust,
      self.interp.fh_d,
      "linear",
    )
    .map_err(|e| format!("WeaverMaterialProvider: resample {key_repr}: {e}"))?;
    Ok(Some(spline.evaluate(&self.target, 0, None).row(0).to_vec()))
  }
}

impl<B: WovenBackend> MaterialProvider for WeaverProvider<B> {
  fn nk(&self, name: &str, wavelengths: &[f64]) -> Result<Vec<Complex64>, String> {
    if !grids_equal(wavelengths, &self.target) {
      return Err(format!(
        "WeaverMaterialProvider: solve grid ({} points) != provider target grid ({} points); \
         re-weave onto the solve grid or set target_wavelength first.",
        wavelengths.len(),
        self.target.len()
      ));
    }
    if let Some(hit) = self.cache.borrow().get(name) {
      return Ok(hit.clone());
    }
    let n_arr = match self.fetch(&self.n_label.clone(), name)? {
      None => {
        return Err(format!("WeaverMaterialProvider: material '{name}' not found."));
      }
      Some(v) => v,
    };
    let k_arr = match self.fetch(&self.k_label.clone(), name)? {
      None => vec![0.0; n_arr.len()],
      Some(v) => v,
    };
    let nk: Vec<Complex64> =
      n_arr.iter().zip(k_arr.iter()).map(|(n, k)| Complex64::new(*n, *k)).collect();
    self.cache.borrow_mut().insert(name.to_string(), nk.clone());
    Ok(nk)
  }

  fn contains(&self, name: &str) -> bool {
    // n-fragment presence only (n-but-no-k serves zeros for k).
    self.backend.weaved(self.key_prefix, &self.n_label, name).is_some()
  }

  fn grid(&self) -> Option<&[f64]> {
    // Target grid is always known.
    Some(&self.target)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::cell::RefCell as Cell;

  /// Dict-backed fake backend: `(bits(prefix), label, material)` → fragment.
  struct Fake {
    map: Cell<HashMap<(u64, String, String), (Vec<f64>, Vec<f64>)>>,
  }

  impl Fake {
    fn new(
      entries: Vec<((f64, &str, &str), (Vec<f64>, Vec<f64>))>,
    ) -> Self {
      let map = entries
        .into_iter()
        .map(|((p, l, m), v)| ((p.to_bits(), l.to_string(), m.to_string()), v))
        .collect();
      Self { map: Cell::new(map) }
    }

    fn put(&self, key: (f64, &str, &str), frag: (Vec<f64>, Vec<f64>)) {
      self.map.borrow_mut().insert(
        (key.0.to_bits(), key.1.to_string(), key.2.to_string()),
        frag,
      );
    }
  }

  impl WovenBackend for Fake {
    fn weaved(&self, prefix: f64, label: &str, material: &str) -> Option<(Vec<f64>, Vec<f64>)> {
      let p = norm_prefix(prefix)?;
      self
        .map
        .borrow()
        .get(&(p.to_bits(), label.to_string(), material.to_string()))
        .cloned()
    }
  }

  fn target() -> Vec<f64> {
    vec![400.0, 500.0, 600.0, 700.0, 800.0]
  }

  fn coarse() -> Vec<f64> {
    vec![400.0, 600.0, 800.0]
  }

  fn backend() -> Fake {
    Fake::new(vec![
      ((0.0, "n", "A"), (coarse(), vec![2.0, 2.3, 2.4])),
      ((0.0, "k", "A"), (target(), vec![0.01, 0.02, 0.03, 0.04, 0.05])),
      ((0.0, "n", "B"), (target(), vec![1.5; 5])),
    ])
  }

  fn provider<B: WovenBackend>(
    backend: B,
    method: &str,
    strict: bool,
  ) -> WeaverProvider<B> {
    WeaverProvider::new(
      backend,
      target(),
      0.0,
      "n",
      "k",
      InterpSettings { method: method.to_string(), robust: false, fh_d: 3 },
      strict,
    )
  }

  fn re(v: &[Complex64]) -> Vec<f64> {
    v.iter().map(|z| z.re).collect()
  }

  fn im(v: &[Complex64]) -> Vec<f64> {
    v.iter().map(|z| z.im).collect()
  }

  /// Frozen oracles captured from Python `WeaverMaterialProvider`
  /// (dict-backed fake, same fragments) as HEX — numpy array printing
  /// truncates to 8 decimals, so `repr` lies by up to an ulp; `tobytes`
  /// does not. `f64::from_bits` asserts true bit-identity with the
  /// shared `UniInterpolator` kernel.
  fn bits(hex_le: &[&str]) -> Vec<f64> {
    hex_le
      .iter()
      .map(|h| {
        let b: Vec<u8> =
          (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap()).collect();
        f64::from_le_bytes(b.try_into().unwrap())
      })
      .collect()
  }

  #[test]
  fn frozen_oracles_linear_pchip_makima() {
    let t = target();
    let p = provider(backend(), "linear", false);
    assert_eq!(
      re(&p.nk("A", &t).unwrap()),
      bits(&["0000000000000040", "3333333333330140", "6666666666660240", "cccccccccccc0240", "3333333333330340"])
    );
    assert_eq!(im(&p.nk("A", &t).unwrap()), vec![0.01, 0.02, 0.03, 0.04, 0.05]);
    let p = provider(backend(), "pchip", false);
    assert_eq!(
      re(&p.nk("A", &t).unwrap()),
      bits(&["0000000000000040", "3433333333730140", "6666666666660240", "3333333333f30240", "3333333333330340"])
    );
    let p = provider(backend(), "makima", false);
    assert_eq!(
      re(&p.nk("A", &t).unwrap()),
      bits(&["0000000000000040", "abaaaaaaaa6a0140", "6666666666660240", "3333333333f30240", "3333333333330340"])
    );
  }

  #[test]
  fn exact_and_missing_k_zeros() {
    let t = target();
    let p = provider(backend(), "linear", false);
    // B exact on target, no k → lossless zeros.
    assert_eq!(re(&p.nk("B", &t).unwrap()), vec![1.5; 5]);
    assert_eq!(im(&p.nk("B", &t).unwrap()), vec![0.0; 5]);
    assert!(p.contains("A") && p.contains("B") && !p.contains("C"));
    assert!(p.grid().is_some());
  }

  #[test]
  fn missing_n_and_strict_and_caller_grid() {
    let t = target();
    let p = provider(backend(), "linear", false);
    let err = p.nk("C", &t).unwrap_err();
    assert!(err.contains("not found"), "{err}");
    let s = provider(backend(), "linear", true);
    let err = s.nk("A", &t).unwrap_err();
    assert!(err.contains("(strict)") && err.contains("3 vs 5 points"), "{err}");
    // Absent k still defaults under strict (absence ≠ staleness).
    assert_eq!(im(&s.nk("B", &t).unwrap()), vec![0.0; 5]);
    // Caller grid ≠ target refuses (bridge contract, enforced here).
    let err = p.nk("B", &coarse()).unwrap_err();
    assert!(err.contains("!= provider target grid"), "{err}");
  }

  #[test]
  fn cache_target_and_invalidate() {
    let b = backend();
    let t = target();
    let mut p = provider(b, "linear", false);
    assert!(p.is_exact("B") && !p.is_exact("A") && !p.is_exact("C"));
    // Prime the memo, then mutate the backend: stale until invalidated.
    assert_eq!(re(&p.nk("B", &t).unwrap()), vec![1.5; 5]);
    p.backend.put((0.0, "n", "B"), (target(), vec![9.0; 5]));
    assert_eq!(re(&p.nk("B", &t).unwrap()), vec![1.5; 5]);
    p.invalidate(Some("B"));
    assert_eq!(re(&p.nk("B", &t).unwrap()), vec![9.0; 5]);
    p.backend.put((0.0, "n", "B"), (target(), vec![7.0; 5]));
    p.invalidate(None);
    assert_eq!(re(&p.nk("B", &t).unwrap()), vec![7.0; 5]);
    // Same-grid target set is a no-op (cache survives); new grid clears.
    p.set_target(target());
    assert_eq!(re(&p.nk("B", &t).unwrap()), vec![7.0; 5]);
    let t2: Vec<f64> = vec![400.0, 800.0];
    p.set_target(t2.clone());
    assert_eq!(re(&p.nk("B", &t2).unwrap()), vec![7.0, 7.0]);
  }

  /// Pure-Rust end to end: a real `OpticalWeaver` backend feeds
  /// expansion with zero Python in the loop.
  #[test]
  fn real_weaver_backend_drives_expansion() {
    use crate::spectralweave::opticalweaver::{OpticalKey, OpticalWeaver, Unit};
    use crate::structure::{ExpandOptions, Layer, expand};
    use std::collections::HashMap;
    let w = OpticalWeaver::new(8);
    let key = |label: &str| OpticalKey {
      wavelength: 0.0,
      data_type: label.into(),
      polarisation: "TiO2".into(),
    };
    w.set_data(key("n"), &[2.35, 2.35], &[400.0, 800.0], Unit::NM, Unit::RAW).unwrap();
    w.set_data(key("k"), &[0.01, 0.01], &[400.0, 800.0], Unit::NM, Unit::RAW).unwrap();
    let wl = vec![400.0, 600.0, 800.0];
    let groups: HashMap<String, crate::structure::Group> = HashMap::new();
    let p = WeaverProvider::new(w, wl.clone(), 0.0, "n", "k", InterpSettings::default(), false);
    let nk = p.nk("TiO2", &wl).unwrap();
    assert!(nk.iter().all(|z| (z.re - 2.35).abs() < 1e-12 && (z.im - 0.01).abs() < 1e-12));
    let seq = vec![(Layer::film(50.0, "TiO2"), false)];
    let (sa, _) = expand(&seq, &p, &wl, &groups, ExpandOptions::deterministic()).unwrap();
    assert_eq!(sa.n_rows(), 1);
    assert!((sa.thicknesses[0] - 50.0).abs() < 1e-12);
  }

  #[test]
  fn key_normalization_matches_python_dict() {
    assert_eq!(norm_prefix(-0.0), Some(0.0));
    assert_eq!(norm_prefix(f64::NAN), None);
    // -0.0 prefix hits the 0.0 entry (Python dict would too).
    let p = WeaverProvider::new(
      backend(),
      target(),
      -0.0,
      "n",
      "k",
      InterpSettings::default(),
      false,
    );
    assert!(p.contains("B"));
  }
}
