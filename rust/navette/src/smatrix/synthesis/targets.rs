//! synthesis::targets — typed target sets → native `MeritSpec`.
//!
//! Rust-first port of the Python `build_merit_spec` converter: typed
//! [`TargetSet`] ingestion into a native `TargetWeaver`, grid-content
//! join of exported entries with their targets, curve-id mapping, and
//! `MeritSpec` assembly. Python's converter thins onto
//! [`compile_merit_spec`]; Rust consumers run standalone with no Python.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Deserialize;

use super::merit::{ConstraintKind, CurveId, MeritKey, MeritSpec, MeritTarget, SimTransform};
use crate::spectralweave::opticalweaver::{OpticalKey, SpectralData};
use crate::spectralweave::targetweaver::{TargetKind, TargetWeaver};

// ---------------------------------------------------------------------------
// Request schema (JSON mirror of the target dataclasses)
// ---------------------------------------------------------------------------

fn d_weight() -> f64 {
  1.0
}

/// One spectral constraint curve (value vs wavelength at fixed angle).
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralTarget {
  pub wavelengths: Vec<f64>,
  pub values: Vec<f64>,
  pub tolerances: Vec<f64>,
  pub angle: f64,
  pub polarization: String,
  pub spectral: String,
  #[serde(default = "d_kind")]
  pub kind: String,
  #[serde(default = "d_norm")]
  pub normalization_mode: String,
  #[serde(default)]
  pub band: Option<Band>,
  #[serde(default)]
  pub phase: bool,
  #[serde(default = "d_weight")]
  pub weight: f64,
  #[serde(default)]
  pub normalize_count: bool,
  #[serde(default)]
  pub integral: bool,
}

/// One angular constraint curve (value vs angle at fixed wavelength).
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AngularTarget {
  pub wavelength: f64,
  pub angles: Vec<f64>,
  pub values: Vec<f64>,
  pub tolerances: Vec<f64>,
  pub polarization: String,
  pub spectral: String,
  #[serde(default = "d_kind")]
  pub kind: String,
  #[serde(default = "d_norm")]
  pub normalization_mode: String,
  #[serde(default)]
  pub band: Option<Band>,
  #[serde(default)]
  pub phase: bool,
  #[serde(default = "d_weight")]
  pub weight: f64,
  #[serde(default)]
  pub normalize_count: bool,
  #[serde(default)]
  pub integral: bool,
}

/// Scalar (broadcast) or per-point band half-widths.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum Band {
  Scalar(f64),
  Points(Vec<f64>),
}

fn d_kind() -> String {
  "e".to_string()
}

fn d_norm() -> String {
  "auto".to_string()
}

/// Full target set: spectral + angular curves and weaver tuning.
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSet {
  #[serde(default)]
  pub spectral: Vec<SpectralTarget>,
  #[serde(default)]
  pub angular: Vec<AngularTarget>,
  #[serde(default = "d_cache")]
  pub cache_size: usize,
  #[serde(default = "d_floor")]
  pub tolerance_floor: f64,
}

fn d_cache() -> usize {
  128
}

fn d_floor() -> f64 {
  1e-12
}

// ---------------------------------------------------------------------------
// Validation (mirrors the dataclass guards)
// ---------------------------------------------------------------------------

fn check_weight(weight: f64, label: &str) -> Result<(), String> {
  if !weight.is_finite() || weight < 0.0 {
    return Err(format!("{label}: weight must be finite and >= 0 (got {weight})."));
  }
  Ok(())
}

fn check_integral(integral: bool, normalize_count: bool, label: &str) -> Result<(), String> {
  if integral && normalize_count {
    return Err(format!(
      "{label}: integral targets reject normalize_count (the mean already is one)."
    ));
  }
  Ok(())
}

fn resolve_band(band: &Option<Band>, n: usize, label: &str) -> Result<Vec<f64>, String> {
  match band {
    None => Ok(Vec::new()),
    Some(Band::Scalar(b)) => {
      if *b < 0.0 {
        return Err(format!("{label}: band must be >= 0."));
      }
      Ok(vec![*b; n])
    }
    Some(Band::Points(v)) => {
      if v.len() != n {
        return Err(format!("{label} band shape mismatch: {} != {n}.", v.len()));
      }
      if v.iter().any(|x| *x < 0.0) {
        return Err(format!("{label}: band must be >= 0."));
      }
      Ok(v.clone())
    }
  }
}

// ---------------------------------------------------------------------------
// Curve-id vocabulary (mirrors SPECTRAL_MAP + _DIFFERENTIAL)
// ---------------------------------------------------------------------------

fn curve_id(spectral: &str, polarization: &str) -> Result<String, String> {
  if let Some((curve, pol, _)) = differential(spectral) {
    if polarization != pol {
      return Err(format!(
        "Cannot convert spectral={spectral:?}, polarization={polarization:?}: \
         {spectral:?} is {pol}-polarized (label encodes polarization)."
      ));
    }
    return Ok(curve.to_string());
  }
  let curve = match (spectral, polarization) {
    ("R", "s") => "Rs",
    ("R", "p") => "Rp",
    ("R", "u") => "Ru",
    ("T", "s") => "Ts",
    ("T", "p") => "Tp",
    ("T", "u") => "Tu",
    ("A", "s") => "As",
    ("A", "p") => "Ap",
    ("A", "u") => "Au",
    ("RB", "s") => "RBs",
    ("RB", "p") => "RBp",
    ("RB", "u") => "RBu",
    ("TB", "s") => "TBs",
    ("TB", "p") => "TBp",
    ("TB", "u") => "TBu",
    ("AB", "s") => "ABs",
    ("AB", "p") => "ABp",
    ("AB", "u") => "ABu",
    _ => {
      return Err(format!(
        "Cannot convert spectral={spectral:?}, polarization={polarization:?}."
      ))
    }
  };
  Ok(curve.to_string())
}

/// Differential-phase labels: spectral → (host CurveId, polarization, passes).
fn differential(spectral: &str) -> Option<(&'static str, &'static str, f64)> {
  match spectral {
    "PDts" => Some(("Ts", "s", 1.0)),
    "PDtp" => Some(("Tp", "p", 1.0)),
    _ => None,
  }
}

fn allclose(a: &[f64], b: &[f64]) -> bool {
  a.len() == b.len()
    && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-08 + 1e-05 * y.abs())
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

/// Ingested entry with its grid (native mirror of one export dict).
struct Exported {
  uid: usize,
  angle: f64,
  polarization: String,
  spectral: String,
  wavelengths: Vec<f64>,
  targets: Vec<f64>,
  tolerances: Vec<f64>,
  band: Vec<f64>,
  kind: String,
  mode: String,
  norm_factor: f64,
  count_norm: Option<f64>,
  integral: bool,
}

fn ingest_spectral(
  weaver: &TargetWeaver,
  t: &SpectralTarget,
) -> Result<(), String> {
  check_weight(t.weight, "SpectralTarget")?;
  check_integral(t.integral, t.normalize_count, "SpectralTarget")?;
  let n = t.values.len();
  if t.wavelengths.len() != n || t.tolerances.len() != n {
    return Err("SpectralTarget shape mismatch".to_string());
  }
  let band = resolve_band(&t.band, n, "SpectralTarget")?;
  let kind =
    TargetKind::from_str(&t.kind).ok_or_else(|| "Invalid kind (use 'e', 'a', 'b', 'r', or 'c')".to_string())?;
  let mode = if t.spectral == "PDts" || t.spectral == "PDtp" { "phase" } else { t.normalization_mode.as_str() };
  let key = OpticalKey::from((t.angle, t.polarization.clone(), t.spectral.clone()));
  let frame = weaver.create_dedicated_frame(&t.wavelengths)?;
  frame.set_data(key.clone(), SpectralData::from_arc(Arc::from(t.values.as_slice())), Some(&t.wavelengths))?;
  weaver.inner.inner.map_frame_to_key(&key, &frame);
  let count_norm = t.normalize_count.then(|| n as f64);
  weaver.register_metadata(
    frame.uid, key, &t.values, &t.tolerances, kind, mode, &band, t.weight, count_norm, t.integral,
  );
  Ok(())
}

fn ingest_angular(
  weaver: &TargetWeaver,
  t: &AngularTarget,
) -> Result<(), String> {
  check_weight(t.weight, "AngularTarget")?;
  check_integral(t.integral, t.normalize_count, "AngularTarget")?;
  let n = t.values.len();
  if t.angles.len() != n || t.tolerances.len() != n {
    return Err("AngularTarget shape mismatch".to_string());
  }
  let band = resolve_band(&t.band, n, "AngularTarget")?;
  let kind =
    TargetKind::from_str(&t.kind).ok_or_else(|| "Invalid kind (use 'e', 'a', 'b', 'r', or 'c')".to_string())?;
  let mode = if t.spectral == "PDts" || t.spectral == "PDtp" { "phase" } else { t.normalization_mode.as_str() };
  // Resolve normalization ONCE over the full curve and share it.
  let (shared_mode, shared_nf) = TargetWeaver::resolve_norm(&t.values, mode);
  let wl_point = vec![t.wavelength];
  let frame = weaver.create_dedicated_frame(&wl_point)?;
  for i in 0..n {
    let key = OpticalKey::from((t.angles[i], t.polarization.clone(), t.spectral.clone()));
    frame.set_data(
      key.clone(),
      SpectralData::from_arc(Arc::from([t.values[i]].as_slice())),
      Some(&wl_point),
    )?;
    weaver.inner.inner.map_frame_to_key(&key, &frame);
    let count_norm = t.normalize_count.then(|| n as f64);
    let band_one = if band.is_empty() { vec![] } else { vec![band[i]] };
    weaver.register_metadata_resolved(
      frame.uid, key, &[t.values[i]], &[t.tolerances[i]], kind, shared_mode, shared_nf,
      &band_one, t.weight, count_norm, t.integral,
    );
  }
  Ok(())
}

fn export_all(weaver: &TargetWeaver) -> Vec<Exported> {
  let meta = weaver.target_metadata.read();
  let mut out = Vec::new();
  for frame in weaver.inner.inner.frames_snapshot() {
    let wl = frame.wavelength().to_vec();
    let entries = match meta.get(&frame.uid) {
      Some(m) => m,
      None => continue,
    };
    for key in frame.keys() {
      let entry = match entries.entries.get(&key) {
        Some(e) => e,
        None => continue,
      };
      let (angle, polarization, spectral) = key.as_tuple();
      out.push(Exported {
        uid: frame.uid,
        angle,
        polarization,
        spectral,
        wavelengths: wl.clone(),
        targets: entry.normalized_targets.to_vec(),
        tolerances: entry.tolerances.to_vec(),
        band: entry.band.to_vec(),
        kind: entry.kind.as_str().to_string(),
        mode: entry.resolved_mode.as_str().to_string(),
        norm_factor: entry.norm_factor,
        count_norm: entry.count_norm,
        integral: entry.integral,
      });
    }
  }
  out
}

/// Compile a [`TargetSet`] into a native `MeritSpec`.
pub fn compile_merit_spec(set: &TargetSet) -> Result<MeritSpec, String> {
  let weaver = TargetWeaver::new(set.cache_size, set.tolerance_floor);
  for t in &set.spectral {
    ingest_spectral(&weaver, t)?;
  }
  for t in &set.angular {
    ingest_angular(&weaver, t)?;
  }
  let entries = export_all(&weaver);
  let mut by_uid: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
  for (i, e) in entries.iter().enumerate() {
    by_uid.entry(e.uid).or_default().push(i);
  }
  let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();

  // Join each exported entry with its target by grid content.
  let mut spectral_pairs: Vec<(usize, usize)> = Vec::new();
  for (ti, t) in set.spectral.iter().enumerate() {
    let mut found = None;
    for (uid, idxs) in &by_uid {
      if used.contains(uid) {
        continue;
      }
      if idxs.len() != 1 {
        continue;
      }
      let e = &entries[idxs[0]];
      if e.spectral == t.spectral
        && e.polarization == t.polarization
        && e.angle == t.angle
        && allclose(&e.wavelengths, &t.wavelengths)
      {
        found = Some((idxs[0], ti));
        used.insert(*uid);
        break;
      }
    }
    spectral_pairs.push(found.ok_or_else(|| {
      format!(
        "compile_merit_spec: no exported entry matches a spectral target \
         (spectral={:?}, polarization={:?}, angle={:?}).",
        t.spectral, t.polarization, t.angle
      )
    })?);
  }
  let mut angular_groups: Vec<(Vec<usize>, usize)> = Vec::new();
  for (ti, t) in set.angular.iter().enumerate() {
    let mut found = None;
    for (uid, idxs) in &by_uid {
      if used.contains(uid) {
        continue;
      }
      if idxs.len() != t.angles.len() {
        continue;
      }
      let mut got: Vec<f64> = idxs.iter().map(|i| entries[*i].angle).collect();
      got.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let mut want = t.angles.clone();
      want.sort_by(|a, b| a.partial_cmp(b).unwrap());
      if !allclose(&got, &want) {
        continue;
      }
      let ok = idxs.iter().all(|i| {
        let e = &entries[*i];
        e.spectral == t.spectral
          && e.polarization == t.polarization
          && e.wavelengths.len() == 1
          && (e.wavelengths[0] - t.wavelength).abs() <= 1e-08 + 1e-05 * t.wavelength.abs()
      });
      if !ok {
        continue;
      }
      let mut sorted = idxs.clone();
      sorted.sort_by(|a, b| {
        entries[*a].angle.partial_cmp(&entries[*b].angle).unwrap()
      });
      found = Some((sorted, ti));
      used.insert(*uid);
      break;
    }
    angular_groups.push(found.ok_or_else(|| {
      format!(
        "compile_merit_spec: no exported entries match an angular target \
         (spectral={:?}, polarization={:?}, wavelength={:?}).",
        t.spectral, t.polarization, t.wavelength
      )
    })?);
  }

  let mut spec = MeritSpec::new();
  let mut keys: BTreeMap<(u64, String), u32> = BTreeMap::new();
  let mut get_key = |spec: &mut MeritSpec, angle: f64, curve: &str| -> u32 {
    let k = (angle.to_bits(), curve.to_string());
    if let Some(idx) = keys.get(&k) {
      return *idx;
    }
    let idx = spec.add_key(MeritKey {
      angle,
      curve: CurveId::from_str(curve).expect("curve id from fixed vocabulary"),
    });
    keys.insert(k, idx as u32);
    idx as u32
  };

  // Spectral pairs then angular groups (mirrors the Python order).
  let mut jobs: Vec<(usize, usize)> = spectral_pairs;
  for (idxs, ti) in &angular_groups {
    for i in idxs {
      jobs.push((*i, 0x8000_0000 | ti));
    }
  }
  for (ei, ti) in jobs {
    let e = &entries[ei];
    let t = if ti & 0x8000_0000 == 0 {
      (&set.spectral[ti].spectral, &set.spectral[ti].polarization, set.spectral[ti].phase, set.spectral[ti].weight)
    } else {
      let a = &set.angular[ti & 0x7fff_ffff];
      (&a.spectral, &a.polarization, a.phase, a.weight)
    };
    let curve = curve_id(t.0, t.1)?;
    let ki = get_key(&mut spec, e.angle, &curve);
    let mut nf = e.norm_factor;
    let diff = differential(t.0);
    if diff.is_some() && !t.2 {
      return Err(format!("spectral={:?} is differential-phase: pass phase=True.", t.0));
    }
    let (norm, band, mode, out_nf) = if t.2 {
      nf = nf.max(1e-300);
      (
        e.targets.iter().map(|x| x / nf).collect(),
        e.band.iter().map(|x| x / nf).collect(),
        "phase".to_string(),
        1.0,
      )
    } else {
      (e.targets.clone(), e.band.clone(), e.mode.clone(), nf)
    };
    let kind = ConstraintKind::from_str(&e.kind)
      .ok_or_else(|| format!("Invalid kind {:?}", e.kind))?;
    let transform = SimTransform::from_str(&mode)
      .ok_or_else(|| format!("Invalid transform {mode:?}"))?;
    spec
      .add_target(MeritTarget {
        key_idx: ki,
        wavelengths: Arc::from(e.wavelengths.as_slice()),
        kind,
        transform,
        norm_factor: out_nf,
        normalized_targets: Arc::from(norm),
        tolerances: Arc::from(e.tolerances.as_slice()),
        band: Arc::from(band),
        phase: t.2,
        differential_passes: diff.map(|(_, _, p)| p),
        weight: t.3,
        count_norm: e.count_norm,
        integral: e.integral,
      })
      .map_err(|e| format!("compile_merit_spec: {e}"))?;
  }
  Ok(spec)
}

// ---------------------------------------------------------------------------
// Tests (standalone: no Python)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn spec_target() -> SpectralTarget {
    SpectralTarget {
      wavelengths: vec![500.0, 600.0],
      values: vec![0.05, 0.05],
      tolerances: vec![0.01, 0.01],
      angle: 0.0,
      polarization: "s".to_string(),
      spectral: "R".to_string(),
      kind: "e".to_string(),
      normalization_mode: "auto".to_string(),
      band: None,
      phase: false,
      weight: 1.0,
      normalize_count: false,
      integral: false,
    }
  }

  #[test]
  fn compiles_flat_spec() {
    let set = TargetSet {
      spectral: vec![spec_target()],
      angular: vec![],
      cache_size: 128,
      tolerance_floor: 1e-12,
    };
    let spec = compile_merit_spec(&set).unwrap();
    assert_eq!(spec.key_count(), 1);
    assert_eq!(spec.target_count(), 1);
  }

  #[test]
  fn refuses_bad_targets() {
    let mut set = TargetSet {
      spectral: vec![spec_target()],
      angular: vec![],
      cache_size: 128,
      tolerance_floor: 1e-12,
    };
    set.spectral[0].weight = -1.0;
    assert!(compile_merit_spec(&set).is_err());
    set.spectral[0].weight = 1.0;
    set.spectral[0].integral = true;
    set.spectral[0].normalize_count = true;
    assert!(compile_merit_spec(&set).is_err());
    set.spectral[0].integral = false;
    set.spectral[0].normalize_count = false;
    set.spectral[0].spectral = "Nope".to_string();
    assert!(compile_merit_spec(&set).is_err());
  }

  #[test]
  fn curve_vocabulary_matches() {
    assert_eq!(curve_id("R", "s").unwrap(), "Rs");
    assert_eq!(curve_id("PDts", "s").unwrap(), "Ts");
    assert!(curve_id("PDts", "p").is_err());
    assert!(curve_id("Nope", "s").is_err());
  }
}
