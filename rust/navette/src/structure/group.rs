//! Scaling/error policy shared by a set of layers.
//!
//! Mirrors `navette.structure.models.Group` exactly (factors, summands,
//! masks, error laws + params, floors, state keys). Deliberate differences:
//! - Randomness is full-Rust (§9.2): draws take `&mut dyn RngCore`
//!   (`StdRng::seed_from_u64` = reproducible, `rand::rng()` = thread).
//!   Streams differ from NumPy by algorithm (ChaCha vs PCG64) —
//!   acceptance is statistical agreement + per-side determinism.
//! - Unknown error laws are unrepresentable (typed `ErrorType`); Python
//!   falls through to unperturbed values. Fail-closed by construction.
//! - `validate` returns typed issues; `set_properties` returns warnings.
//! - States require complete param maps (Python replaces whole dicts and
//!   fails later at draw time; refusing at load is fail-closed).

use std::collections::BTreeMap;
use std::fmt;

use num_complex::Complex64;
use rand::RngCore;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::structure::enums::ErrorType;
use crate::structure::validation::ValidationIssue;
use crate::structure::version::{SCHEMA_VERSION, check_schema_version};

fn one() -> f64 {
  1.0
}

/// Eight-parameter error vocabulary, shared by every channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorParams {
  pub abs_mean_delta_g: f64,
  pub abs_std_dev: f64,
  pub rel_mean_delta_g: f64,
  pub rel_std_dev: f64,
  pub abs_mean_delta_h: f64,
  pub abs_variance: f64,
  pub rel_mean_delta_h: f64,
  pub rel_variance: f64,
}

impl ErrorParams {
  /// Default law params (all channels except roughness).
  pub fn standard() -> Self {
    Self {
      abs_mean_delta_g: 0.0,
      abs_std_dev: 0.01,
      rel_mean_delta_g: 0.0,
      rel_std_dev: 1.0,
      abs_mean_delta_h: 0.0,
      abs_variance: 0.01,
      rel_mean_delta_h: 0.0,
      rel_variance: 1.0,
    }
  }

  /// Roughness-channel defaults: `abs_*` x0.1 so the physical magnitude is
  /// unchanged by the Å→nm switch (0.01 Å == 0.001 nm); `rel_*` untouched.
  pub fn roughness() -> Self {
    Self {
      abs_mean_delta_g: 0.0,
      abs_std_dev: 0.001,
      rel_mean_delta_g: 0.0,
      rel_std_dev: 1.0,
      abs_mean_delta_h: 0.0,
      abs_variance: 0.001,
      rel_mean_delta_h: 0.0,
      rel_variance: 1.0,
    }
  }
}

/// Wrap a group in a shared handle (structure storage).
pub fn shared_group(group: Group) -> crate::structure::SharedGroup {
  std::rc::Rc::new(std::cell::RefCell::new(group))
}

/// Scaling/error policy shared by a set of layers (material-keyed).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Group {
  pub group_name: String,
  pub thick_factor: f64,
  pub thick_summand: f64,
  pub n_factor: f64,
  pub k_factor: f64,
  pub inh_delta_summand: f64,
  pub roughness_summand: f64,
  pub interface_summand: f64,
  pub error_mask: [i32; 6],
  pub optimization_mask: [i32; 7],
  pub thickness_error_type: ErrorType,
  pub n_error_type: ErrorType,
  pub k_error_type: ErrorType,
  pub inh_delta_error_type: ErrorType,
  pub roughness_error_type: ErrorType,
  pub interface_error_type: ErrorType,
  pub thickness_error_params: ErrorParams,
  pub inh_delta_error_params: ErrorParams,
  pub roughness_error_params: ErrorParams,
  pub interface_error_params: ErrorParams,
  pub n_error_params: ErrorParams,
  pub k_error_params: ErrorParams,
}

impl Group {
  /// Named group at identity (factors (1,1), zero summands, GAUSSIAN laws).
  pub fn new(group_name: impl Into<String>) -> Self {
    Self {
      group_name: group_name.into(),
      thick_factor: 1.0,
      thick_summand: 0.0,
      n_factor: 1.0,
      k_factor: 1.0,
      inh_delta_summand: 0.0,
      roughness_summand: 0.0,
      interface_summand: 0.0,
      error_mask: [0; 6],
      optimization_mask: [1; 7],
      thickness_error_type: ErrorType::Gaussian,
      n_error_type: ErrorType::Gaussian,
      k_error_type: ErrorType::Gaussian,
      inh_delta_error_type: ErrorType::Gaussian,
      roughness_error_type: ErrorType::Gaussian,
      interface_error_type: ErrorType::Gaussian,
      thickness_error_params: ErrorParams::standard(),
      inh_delta_error_params: ErrorParams::standard(),
      roughness_error_params: ErrorParams::roughness(),
      interface_error_params: ErrorParams::standard(),
      n_error_params: ErrorParams::standard(),
      k_error_params: ErrorParams::standard(),
    }
  }

  /// `(n_factor, k_factor)` as a complex multiplier (Python `nk_factor`).
  pub fn nk_factor(&self) -> Complex64 {
    Complex64::new(self.n_factor, self.k_factor)
  }

  /// Factor domains: identity (1,1); negatives unphysical; NaN invalid;
  /// optimization mask must be 7 binary entries.
  pub fn validate(&self) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if self.thick_factor < 0.0 || self.thick_factor.is_nan() {
      issues.push(ValidationIssue::error(format!(
        "Group '{}': thick_factor {} < 0 (NaN counts as invalid).",
        self.group_name, self.thick_factor
      )));
    }
    if self.n_factor < 0.0 || self.n_factor.is_nan() {
      issues.push(ValidationIssue::error(format!(
        "Group '{}': n_factor {} < 0 (no negative-index media).",
        self.group_name, self.n_factor
      )));
    }
    if self.k_factor < 0.0 || self.k_factor.is_nan() {
      issues.push(ValidationIssue::error(format!(
        "Group '{}': k_factor {} < 0 (no gain media).",
        self.group_name, self.k_factor
      )));
    }
    if self.optimization_mask.iter().any(|v| *v != 0 && *v != 1) {
      issues.push(ValidationIssue::error(format!(
        "Group '{}': optimization_mask must be 7 binary entries (see OptMask).",
        self.group_name
      )));
    }
    issues
  }

  /// Draw one perturbation (Python `_apply_error`, order-preserved).
  /// Non-positive spreads contribute their mean deterministically (NumPy
  /// draws exactly at zero spread; no RNG consumed either way that matters
  /// — streams are per-side by §9.2).
  pub fn apply_error(
    value: f64,
    error_type: ErrorType,
    params: &ErrorParams,
    rng: &mut dyn RngCore,
  ) -> f64 {
    let g_abs = gauss_draw(params.abs_mean_delta_g, params.abs_std_dev, rng);
    match error_type {
      ErrorType::Gaussian => {
        value + g_abs + gauss_draw(params.rel_mean_delta_g, params.rel_std_dev, rng) * value
      }
      ErrorType::Uniform => {
        value + unif_draw(params.abs_variance, rng) + unif_draw(params.rel_variance, rng) * value
      }
      ErrorType::Combined => {
        value
          + g_abs
          + gauss_draw(params.rel_mean_delta_g, params.rel_std_dev, rng) * value
          + unif_draw(params.abs_variance, rng)
          + unif_draw(params.rel_variance, rng) * value
      }
    }
  }

  /// Perturbed thickness (floored at 0).
  pub fn thickness_error(&self, value: f64, rng: &mut dyn RngCore) -> f64 {
    Self::apply_error(value, self.thickness_error_type, &self.thickness_error_params, rng).max(0.0)
  }

  /// Perturbed grading strength (unfloored, like Python).
  pub fn inh_delta_error(&self, value: f64, rng: &mut dyn RngCore) -> f64 {
    Self::apply_error(value, self.inh_delta_error_type, &self.inh_delta_error_params, rng)
  }

  /// Perturbed surface roughness [nm] (floored at 0).
  pub fn sr_roughness_error(&self, value: f64, rng: &mut dyn RngCore) -> f64 {
    Self::apply_error(value, self.roughness_error_type, &self.roughness_error_params, rng).max(0.0)
  }

  /// Perturbed interface width [nm] (floored at 0).
  pub fn interface_error(&self, value: f64, rng: &mut dyn RngCore) -> f64 {
    Self::apply_error(value, self.interface_error_type, &self.interface_error_params, rng).max(0.0)
  }

  /// Perturbed index with n floored at 0 (k untouched by the floor).
  pub fn nk_error(&self, nk_value: Complex64, rng: &mut dyn RngCore) -> Complex64 {
    let n = Self::apply_error(nk_value.re, self.n_error_type, &self.n_error_params, rng).max(0.0);
    let k = Self::apply_error(nk_value.im, self.k_error_type, &self.k_error_params, rng);
    Complex64::new(n, k)
  }

  /// Serialize all slots (Python `get_state`: schema_version + slots).
  pub fn to_state(&self) -> Value {
    let mut v = serde_json::to_value(self).expect("Group serialization is infallible");
    v.as_object_mut()
      .expect("Group serializes as a map")
      .insert("schema_version".to_string(), Value::from(SCHEMA_VERSION));
    v
  }

  /// Rebuild from state (Python `from_state`): version-checked, unknown
  /// keys ignored, deep-copied by value. Missing `group_name` → "default".
  pub fn from_state(value: &Value) -> Result<Self, String> {
    let found = value.get("schema_version").and_then(|v| v.as_u64()).map(|v| v as u32);
    check_schema_version(found, "Group")?;
    #[derive(Deserialize)]
    struct Raw {
      #[serde(default = "default_group_name")]
      group_name: String,
      #[serde(default = "one")]
      thick_factor: f64,
      #[serde(default)]
      thick_summand: f64,
      #[serde(default = "one")]
      n_factor: f64,
      #[serde(default = "one")]
      k_factor: f64,
      #[serde(default)]
      inh_delta_summand: f64,
      #[serde(default)]
      roughness_summand: f64,
      #[serde(default)]
      interface_summand: f64,
      #[serde(default)]
      error_mask: [i32; 6],
      #[serde(default = "default_opt_mask")]
      optimization_mask: [i32; 7],
      #[serde(default = "gaussian")]
      thickness_error_type: ErrorType,
      #[serde(default = "gaussian")]
      n_error_type: ErrorType,
      #[serde(default = "gaussian")]
      k_error_type: ErrorType,
      #[serde(default = "gaussian")]
      inh_delta_error_type: ErrorType,
      #[serde(default = "gaussian")]
      roughness_error_type: ErrorType,
      #[serde(default = "gaussian")]
      interface_error_type: ErrorType,
      #[serde(default = "ErrorParams::standard")]
      thickness_error_params: ErrorParams,
      #[serde(default = "ErrorParams::standard")]
      inh_delta_error_params: ErrorParams,
      #[serde(default = "ErrorParams::roughness")]
      roughness_error_params: ErrorParams,
      #[serde(default = "ErrorParams::standard")]
      interface_error_params: ErrorParams,
      #[serde(default = "ErrorParams::standard")]
      n_error_params: ErrorParams,
      #[serde(default = "ErrorParams::standard")]
      k_error_params: ErrorParams,
    }
    fn default_group_name() -> String {
      "default".to_string()
    }
    fn default_opt_mask() -> [i32; 7] {
      [1; 7]
    }
    fn gaussian() -> ErrorType {
      ErrorType::Gaussian
    }
    let raw: Raw = serde_json::from_value(value.clone())
      .map_err(|e| format!("Group: malformed state ({e})."))?;
    Ok(Self {
      group_name: raw.group_name,
      thick_factor: raw.thick_factor,
      thick_summand: raw.thick_summand,
      n_factor: raw.n_factor,
      k_factor: raw.k_factor,
      inh_delta_summand: raw.inh_delta_summand,
      roughness_summand: raw.roughness_summand,
      interface_summand: raw.interface_summand,
      error_mask: raw.error_mask,
      optimization_mask: raw.optimization_mask,
      thickness_error_type: raw.thickness_error_type,
      n_error_type: raw.n_error_type,
      k_error_type: raw.k_error_type,
      inh_delta_error_type: raw.inh_delta_error_type,
      roughness_error_type: raw.roughness_error_type,
      interface_error_type: raw.interface_error_type,
      thickness_error_params: raw.thickness_error_params,
      inh_delta_error_params: raw.inh_delta_error_params,
      roughness_error_params: raw.roughness_error_params,
      interface_error_params: raw.interface_error_params,
      n_error_params: raw.n_error_params,
      k_error_params: raw.k_error_params,
    })
  }

  /// Bulk-set known properties; unknown keys become returned warnings.
  pub fn set_properties(&mut self, props: &BTreeMap<String, Value>) -> Vec<ValidationIssue> {
    let mut warnings = Vec::new();
    let bad = |msg: String| ValidationIssue::warning(format!("Group.set_properties: {msg}"));
    for (key, value) in props {
      match key.as_str() {
        "group_name" => match value.as_str() {
          Some(s) => self.group_name = s.to_string(),
          None => warnings.push(bad("ignoring non-string 'group_name'.".to_string())),
        },
        "thick_factor" | "thick_summand" | "n_factor" | "k_factor" | "inh_delta_summand"
        | "roughness_summand" | "interface_summand" => match value.as_f64() {
          Some(v) => match key.as_str() {
            "thick_factor" => self.thick_factor = v,
            "thick_summand" => self.thick_summand = v,
            "n_factor" => self.n_factor = v,
            "k_factor" => self.k_factor = v,
            "inh_delta_summand" => self.inh_delta_summand = v,
            "roughness_summand" => self.roughness_summand = v,
            _ => self.interface_summand = v,
          },
          None => warnings.push(bad(format!("ignoring non-numeric '{key}'."))),
        },
        "error_mask" => match parse_i32_array::<6>(value) {
          Some(m) => self.error_mask = m,
          None => warnings.push(bad("ignoring malformed 'error_mask'.".to_string())),
        },
        "optimization_mask" => match parse_i32_array::<7>(value) {
          Some(m) => self.optimization_mask = m,
          None => warnings.push(bad("ignoring malformed 'optimization_mask'.".to_string())),
        },
        t @ ("thickness_error_type" | "n_error_type" | "k_error_type" | "inh_delta_error_type"
        | "roughness_error_type" | "interface_error_type") => {
          match value.as_i64().map(|v| v as i32).map(ErrorType::try_from_i32) {
            Some(Ok(e)) => match t {
              "thickness_error_type" => self.thickness_error_type = e,
              "n_error_type" => self.n_error_type = e,
              "k_error_type" => self.k_error_type = e,
              "inh_delta_error_type" => self.inh_delta_error_type = e,
              "roughness_error_type" => self.roughness_error_type = e,
              _ => self.roughness_error_type = e,
            },
            _ => warnings.push(bad(format!("unknown {t}; ignoring."))),
          }
        }
        p @ ("thickness_error_params" | "inh_delta_error_params" | "roughness_error_params"
        | "interface_error_params" | "n_error_params" | "k_error_params") => {
          match serde_json::from_value::<ErrorParams>(value.clone()) {
            Ok(ep) => match p {
              "thickness_error_params" => self.thickness_error_params = ep,
              "inh_delta_error_params" => self.inh_delta_error_params = ep,
              "roughness_error_params" => self.roughness_error_params = ep,
              "interface_error_params" => self.interface_error_params = ep,
              "n_error_params" => self.n_error_params = ep,
              _ => self.k_error_params = ep,
            },
            Err(_) => warnings.push(bad(format!("ignoring malformed '{p}'."))),
          }
        }
        other => warnings.push(bad(format!("ignoring unknown attribute '{other}'."))),
      }
    }
    warnings
  }
}

pub(crate) fn gauss_draw<R: RngCore + ?Sized>(mean: f64, std: f64, rng: &mut R) -> f64 {
  if std <= 0.0 {
    mean
  } else {
    Normal::new(mean, std).map(|d| d.sample(rng)).unwrap_or(mean)
  }
}

pub(crate) fn unif_draw<R: RngCore + ?Sized>(half_width: f64, rng: &mut R) -> f64 {
  if half_width <= 0.0 {
    0.0
  } else {
    Uniform::new(-half_width, half_width).map(|d| d.sample(rng)).unwrap_or(0.0)
  }
}

fn parse_i32_array<const N: usize>(value: &Value) -> Option<[i32; N]> {
  let arr = value.as_array()?;
  if arr.len() != N {
    return None;
  }
  let mut out = [0i32; N];
  for (i, v) in arr.iter().enumerate() {
    out[i] = v.as_i64()? as i32;
  }
  Some(out)
}

impl fmt::Display for Group {
  /// Python `__repr__` format, verbatim.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Group(name='{}', thick_factor={:.3})", self.group_name, self.thick_factor)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use serde_json::json;

  fn rng() -> StdRng {
    StdRng::seed_from_u64(7)
  }

  #[test]
  fn defaults_match_python_ctor() {
    let g = Group::new("TiO2");
    assert_eq!((g.thick_factor, g.thick_summand), (1.0, 0.0));
    assert_eq!((g.n_factor, g.k_factor), (1.0, 1.0));
    assert_eq!(g.nk_factor(), Complex64::new(1.0, 1.0));
    assert_eq!(g.error_mask, [0; 6]);
    assert_eq!(g.optimization_mask, [1; 7]);
    assert_eq!(g.thickness_error_type, ErrorType::Gaussian);
    assert_eq!(g.thickness_error_params.abs_std_dev, 0.01);
    // Roughness abs defaults x0.1 (Å→nm magnitude preservation).
    assert_eq!(g.roughness_error_params.abs_std_dev, 0.001);
    assert_eq!(g.roughness_error_params.abs_variance, 0.001);
    assert_eq!(g.roughness_error_params.rel_std_dev, 1.0);
    assert_eq!(g.to_string(), "Group(name='TiO2', thick_factor=1.000)");
  }

  #[test]
  fn validate_domains_match_python() {
    assert!(Group::new("g").validate().is_empty());
    let mut g = Group::new("g");
    g.thick_factor = -1.0;
    g.n_factor = -0.5;
    g.k_factor = f64::NAN;
    g.optimization_mask[3] = 2;
    let issues = g.validate();
    assert_eq!(issues.len(), 4);
    assert!(issues.iter().all(|i| i.is_error()));
  }

  #[test]
  fn error_laws_agree_statistically() {
    // GAUSSIAN: abs N(1,2) only → mean ≈ 11, std ≈ 2 on value 10.
    let p = ErrorParams {
      abs_mean_delta_g: 1.0,
      abs_std_dev: 2.0,
      rel_mean_delta_g: 0.0,
      rel_std_dev: 0.0,
      ..ErrorParams::standard()
    };
    // Zero rel spread contributes its mean (0.0) deterministically.
    let mut r = rng();
    let n = 20_000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
      let d = Group::apply_error(10.0, ErrorType::Gaussian, &p, &mut r);
      sum += d;
      sum2 += d * d;
    }
    let mean = sum / n as f64;
    let std = ((sum2 / n as f64) - mean * mean).sqrt();
    assert!((mean - 11.0).abs() < 0.1, "mean {mean}");
    assert!((std - 2.0).abs() < 0.1, "std {std}");
    // UNIFORM: abs U(-3,3) → bounded.
    let pu = ErrorParams { abs_variance: 3.0, rel_variance: 0.0, ..ErrorParams::standard() };
    let mut r = rng();
    for _ in 0..1000 {
      let d = Group::apply_error(10.0, ErrorType::Uniform, &pu, &mut r);
      assert!((7.0..=13.0).contains(&d), "out of range {d}");
    }
    // COMBINED runs finite.
    let mut r = rng();
    for _ in 0..100 {
      let d = Group::apply_error(10.0, ErrorType::Combined, &ErrorParams::standard(), &mut r);
      assert!(d.is_finite());
    }
  }

  #[test]
  fn draws_are_deterministic_per_seed() {
    let p = ErrorParams::standard();
    let mut a = rng();
    let mut b = StdRng::seed_from_u64(7);
    for _ in 0..50 {
      let x = Group::apply_error(10.0, ErrorType::Combined, &p, &mut a);
      let y = Group::apply_error(10.0, ErrorType::Combined, &p, &mut b);
      assert_eq!(x, y);
    }
  }

  #[test]
  fn floors_match_python() {
    // Extreme negative systematic offset forces every floor.
    let mut g = Group::new("g");
    for params in [
      &mut g.thickness_error_params,
      &mut g.roughness_error_params,
      &mut g.interface_error_params,
      &mut g.n_error_params,
    ] {
      params.abs_mean_delta_g = -100.0;
      params.abs_std_dev = 0.0;
    }
    let mut r = rng();
    assert_eq!(g.thickness_error(50.0, &mut r), 0.0);
    assert_eq!(g.sr_roughness_error(1.0, &mut r), 0.0);
    assert_eq!(g.interface_error(2.0, &mut r), 0.0);
    let nk = g.nk_error(Complex64::new(2.0, 5.0), &mut r);
    assert_eq!(nk.re, 0.0);
    // k untouched by the floor (default k law: zero-mean, unit-scale —
    // finite and unclamped is the assertion).
    assert!(nk.im.is_finite());
    // inh_delta unfloored.
    g.inh_delta_error_params.abs_mean_delta_g = -100.0;
    g.inh_delta_error_params.abs_std_dev = 0.0;
    assert!(g.inh_delta_error(0.2, &mut r) < 0.0);
  }

  #[test]
  fn state_round_trip_and_fingerprint() {
    let g = Group::new("TiO2");
    let v = g.to_state();
    let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(
      keys,
      ["error_mask", "group_name", "inh_delta_error_params", "inh_delta_error_type",
       "inh_delta_summand", "interface_error_params", "interface_error_type",
       "interface_summand", "k_error_params", "k_error_type", "k_factor",
       "n_error_params", "n_error_type", "n_factor", "optimization_mask",
       "roughness_error_params", "roughness_error_type", "roughness_summand",
       "schema_version", "thick_factor", "thick_summand", "thickness_error_params",
       "thickness_error_type"]
    );
    let back = Group::from_state(&v).unwrap();
    assert_eq!(back, g);
    // Unknown keys ignored; version enforced; group_name defaults.
    let mut with_bogus = v.clone();
    with_bogus["bogus"] = json!(1);
    assert_eq!(Group::from_state(&with_bogus).unwrap(), g);
    let mut nover = v.clone();
    nover.as_object_mut().unwrap().remove("schema_version");
    assert!(Group::from_state(&nover).is_err());
    let mut minimal = json!({"schema_version": 1});
    let d = Group::from_state(&minimal).unwrap();
    assert_eq!(d.group_name, "default");
    assert_eq!((d.thick_factor, d.n_factor, d.k_factor), (1.0, 1.0, 1.0));
    // Bad enum discriminant refused (stricter than Python — fail-closed).
    let mut bad_enum = v.clone();
    bad_enum["n_error_type"] = json!(9);
    assert!(Group::from_state(&bad_enum).is_err());
  }
}
