//! Two-phase stack expansion: design layers → solver rows.
//!
//! Transliteration of `navette.structure.expander._LayerExpander`:
//! phase 1 resolves every entry's bulk (group scaling, no RNG); phase 2
//! emits rows in traversal order with error draws. The mirror rule
//! (inverted-run planes take the traversal predecessor's flags; incident
//! edges clean), donor-side carve with buffered-row rescale, owner+carrier
//! Looyenga mix at f=0.5, roughness-follows-plane, and the draw order
//! (thick, nk, rough, iface, inhg) are all preserved exactly.
//!
//! Randomness is full-Rust: `seed = Some` → reproducible `StdRng`,
//! `None` → thread RNG. Forward output must equal Python bit-for-bit
//! (pinned below); Monte-Carlo paths agree statistically (§9.2).

use std::collections::HashMap;

use num_complex::Complex64;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::{StdRng, ThreadRng};

use crate::enums::{ErrorType, RoughnessType};
use crate::group::{Group, gauss_draw, unif_draw};
use crate::layer::Layer;
use crate::providers::MaterialProvider;

/// One emitted solver row-block: rows `[start, end)` belong to logical
/// entry `logical` (interface slices resolve to their carrier).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
  pub start: usize,
  pub end: usize,
  pub logical: usize,
}

/// Engine-ready arrays: row-major `indices` (`n_rows × n_wavelengths`).
#[derive(Debug, Clone, PartialEq)]
pub struct SolverArrays {
  pub thicknesses: Vec<f64>,
  pub indices: Vec<Complex64>,
  pub n_wavelengths: usize,
  pub incoherent: Vec<bool>,
  pub rough_types: Vec<i32>,
  pub rough_vals: Vec<f64>,
}

impl SolverArrays {
  pub fn n_rows(&self) -> usize {
    self.thicknesses.len()
  }

  pub fn row(&self, i: usize) -> &[Complex64] {
    let s = i * self.n_wavelengths;
    &self.indices[s..s + self.n_wavelengths]
  }
}

/// Expansion options: error draws on/off + draw stream.
#[derive(Debug, Clone, Copy)]
pub struct ExpandOptions {
  pub apply_errors: bool,
  pub seed: Option<u64>,
}

impl ExpandOptions {
  pub fn deterministic() -> Self {
    Self { apply_errors: false, seed: None }
  }
}

enum AnyRng {
  Seeded(Box<StdRng>),
  Thread(ThreadRng),
}

impl AnyRng {
  fn rng(&mut self) -> &mut dyn RngCore {
    match self {
      AnyRng::Seeded(r) => r,
      AnyRng::Thread(r) => r,
    }
  }
}

/// Expand `(layer, inverted)` traversal entries to solver rows + spans.
///
/// `wavelengths` is the simulation grid (provider arrays resolve on it).
/// Empty sequences are refused. Deterministic output (errors off) must
/// equal the Python expander bit-for-bit.
pub fn expand(
  seq: &[(Layer, bool)],
  provider: &dyn MaterialProvider,
  wavelengths: &[f64],
  groups: &HashMap<String, Group>,
  opts: ExpandOptions,
) -> Result<(SolverArrays, Vec<Span>), String> {
  if seq.is_empty() {
    return Err("_LayerExpander.expand: No layers to expand. Empty layer sequence provided.".to_string());
  }
  let default_group = Group::new("_default_");
  let group_of = |material: &str| groups.get(material).unwrap_or(&default_group);

  // ---- Phase 1: deterministic bulk resolution (no RNG). ----
  let mut bulk_nk: Vec<Vec<Complex64>> = Vec::with_capacity(seq.len());
  let mut bulk_t: Vec<f64> = Vec::with_capacity(seq.len());
  let mut owner_of: Vec<Option<usize>> = Vec::with_capacity(seq.len());
  for (k, (layer, inv)) in seq.iter().enumerate() {
    let group = group_of(layer.material.as_str());
    let base = provider.nk(layer.material.as_str(), wavelengths)?;
    let scaled = if group.n_factor != 1.0 || group.k_factor != 1.0 {
      base.iter().map(|z| Complex64::new(z.re * group.n_factor, z.im * group.k_factor)).collect()
    } else {
      base
    };
    bulk_nk.push(scaled);
    bulk_t.push(layer.thickness * group.thick_factor + group.thick_summand);
    owner_of.push(if *inv && k > 0 && seq[k - 1].1 {
      Some(k - 1)
    } else if !inv && k > 0 {
      Some(k)
    } else {
      None
    });
  }

  // ---- Phase 2: emission in traversal order (RNG draws here). ----
  let mut rng = match opts.seed {
    Some(s) => AnyRng::Seeded(Box::new(StdRng::seed_from_u64(s))),
    None if opts.apply_errors => AnyRng::Thread(rand::rng()),
    None => AnyRng::Seeded(Box::new(StdRng::seed_from_u64(0))), // unused; errors off
  };
  let mut col_thick: Vec<f64> = Vec::new();
  let mut col_nk: Vec<Complex64> = Vec::new();
  let mut col_coh: Vec<bool> = Vec::new();
  let mut col_r_val: Vec<f64> = Vec::new();
  let mut col_r_type: Vec<i32> = Vec::new();
  let mut spans: Vec<Span> = Vec::new();
  let mut bulk_spans: Vec<(usize, usize)> = Vec::new();
  let mut err_nk: Vec<Vec<Complex64>> = Vec::with_capacity(seq.len());
  let mut err_t: Vec<f64> = Vec::with_capacity(seq.len());
  let mut prev_eff_nk: Option<Vec<Complex64>> = None;

  for (k, (layer, inv)) in seq.iter().enumerate() {
    let group = group_of(layer.material.as_str());
    let mut layer_nk = bulk_nk[k].clone();
    let mut layer_thickness = bulk_t[k];
    if opts.apply_errors {
      if group.error_mask[crate::enums::ErrorMask::Thickness as usize] != 0 {
        layer_thickness = group.thickness_error(layer_thickness, rng.rng());
      }
      let me = crate::enums::ErrorMask::NReal as usize;
      let ke = crate::enums::ErrorMask::NImag as usize;
      if group.error_mask[me] != 0 || group.error_mask[ke] != 0 {
        perturb_nk(&mut layer_nk, group, rng.rng());
      }
    }
    layer_thickness = layer_thickness.max(0.0);
    err_nk.push(layer_nk.clone());
    err_t.push(layer_thickness);
    let o = owner_of[k];

    // Bulk roughness (owner-group draws; RNG order thick, nk, rough, ...).
    let (current_roughness, rtype) = if *inv {
      match o {
        Some(oi) => {
          let (olayer, _) = &seq[oi];
          let ogroup = group_of(olayer.material.as_str());
          let mut r = (olayer.roughness + ogroup.roughness_summand).max(0.0);
          if opts.apply_errors && ogroup.error_mask[crate::enums::ErrorMask::Roughness as usize] != 0 {
            r = ogroup.sr_roughness_error(r, rng.rng());
          }
          (r, olayer.rough_type as i32)
        }
        None => (0.0, RoughnessType::None as i32),
      }
    } else {
      let mut r = (layer.roughness + group.roughness_summand).max(0.0);
      if opts.apply_errors && group.error_mask[crate::enums::ErrorMask::Roughness as usize] != 0 {
        r = group.sr_roughness_error(r, rng.rng());
      }
      (r, layer.rough_type as i32)
    };

    let start = col_thick.len();

    // Plane slice (flag owner's group governs summand + draws).
    if let Some(oi) = o {
      let (olayer, _) = &seq[oi];
      let ogroup = group_of(olayer.material.as_str());
      if olayer.interface {
        let mut t_interface = olayer.interface_thickness + ogroup.interface_summand;
        if opts.apply_errors && ogroup.error_mask[crate::enums::ErrorMask::Interface as usize] != 0 {
          t_interface = ogroup.interface_error(t_interface, rng.rng());
        }
        let carve_total = if oi == k { layer_thickness } else { err_t[oi] };
        t_interface = t_interface.min(carve_total);
        let mix = if oi == k {
          layer_thickness -= t_interface;
          let prev = prev_eff_nk.as_deref().unwrap_or(&err_nk[k]);
          looyenga_mix(&layer_nk, prev)
        } else {
          let (start_o, end_o) = bulk_spans[oi];
          if carve_total > 0.0 && end_o > start_o {
            let scale = (carve_total - t_interface) / carve_total;
            for row in &mut col_thick[start_o..end_o] {
              *row *= scale;
            }
          }
          looyenga_mix(&err_nk[oi], &err_nk[k])
        };
        push_row(
          &mut col_thick,
          &mut col_nk,
          &mut col_coh,
          &mut col_r_val,
          &mut col_r_type,
          t_interface,
          mix,
          true,
          0.0,
          RoughnessType::None as i32,
        );
      }
    }

    let bulk_start = col_thick.len();
    let sub = layer.sub_layer_count();
    if layer.inhomogen && sub > 1 {
      let mut current_delta = (layer.inh_delta + group.inh_delta_summand) * 0.5;
      if opts.apply_errors && group.error_mask[crate::enums::ErrorMask::InhDelta as usize] != 0 {
        current_delta = group.inh_delta_error(current_delta, rng.rng());
      }
      let mut factors: Vec<f64> =
        (0..sub).map(|i| 1.0 - current_delta + 2.0 * current_delta * f64::from(i) / f64::from(sub - 1)).collect();
      if *inv {
        factors.reverse();
      }
      let step_t = layer_thickness / f64::from(sub);
      for (ix, f) in factors.iter().enumerate() {
        let row_nk: Vec<Complex64> = layer_nk.iter().map(|z| z * f).collect();
        push_row(
          &mut col_thick,
          &mut col_nk,
          &mut col_coh,
          &mut col_r_val,
          &mut col_r_type,
          step_t,
          row_nk,
          layer.coherent,
          if ix == 0 { current_roughness } else { 0.0 },
          if ix == 0 { rtype } else { RoughnessType::None as i32 },
        );
      }
    } else {
      push_row(
        &mut col_thick,
        &mut col_nk,
        &mut col_coh,
        &mut col_r_val,
        &mut col_r_type,
        layer_thickness,
        layer_nk.clone(),
        layer.coherent,
        current_roughness,
        rtype,
      );
    }
    spans.push(Span { start, end: col_thick.len(), logical: k });
    bulk_spans.push((bulk_start, col_thick.len()));

    prev_eff_nk = Some(layer_nk);
  }

  let n_rows = col_thick.len();
  let sa = SolverArrays {
    thicknesses: col_thick,
    indices: col_nk,
    n_wavelengths: wavelengths.len(),
    incoherent: col_coh.iter().map(|c| !c).collect(),
    rough_types: col_r_type,
    rough_vals: col_r_val,
  };
  debug_assert_eq!(sa.indices.len(), n_rows * wavelengths.len());
  Ok((sa, spans))
}

#[allow(clippy::too_many_arguments)]
fn push_row(
  col_thick: &mut Vec<f64>,
  col_nk: &mut Vec<Complex64>,
  col_coh: &mut Vec<bool>,
  col_r_val: &mut Vec<f64>,
  col_r_type: &mut Vec<i32>,
  thickness: f64,
  nk: Vec<Complex64>,
  coherent: bool,
  rough_val: f64,
  rough_type: i32,
) {
  col_thick.push(thickness);
  col_nk.extend(nk);
  col_coh.push(coherent);
  col_r_val.push(rough_val);
  col_r_type.push(rough_type);
}

/// Owner+carrier Looyenga mix at f=0.5 (native kernel; eps→n at insertion).
fn looyenga_mix(owner_nk: &[Complex64], other_nk: &[Complex64]) -> Vec<Complex64> {
  use ndarray::Array1;
  let ni = Array1::from_vec(owner_nk.to_vec());
  let nh = Array1::from_vec(other_nk.to_vec());
  let eps = navette_materials::ema::looyenga(ni.view(), nh.view(), 0.5);
  navette_materials::ema::eps_to_nk(eps.view()).to_vec()
}

/// Array-wide nk perturbation with scalar draws (systematic fabrication
/// offset, not per-wavelength noise — mirrors `Group._apply_error` on
/// arrays: n floored at 0, k untouched).
fn perturb_nk(nk: &mut [Complex64], group: &Group, rng: &mut dyn RngCore) {
  use crate::enums::ErrorMask;
  let me = ErrorMask::NReal as usize;
  let ke = ErrorMask::NImag as usize;
  if group.error_mask[me] != 0 {
    let (dn_abs, dn_rel) = channel_draws(group.n_error_type, &group.n_error_params, rng);
    for z in nk.iter_mut() {
      z.re = (z.re + dn_abs + dn_rel * z.re).max(0.0);
    }
  }
  if group.error_mask[ke] != 0 {
    let (dk_abs, dk_rel) = channel_draws(group.k_error_type, &group.k_error_params, rng);
    for z in nk.iter_mut() {
      z.im += dk_abs + dk_rel * z.im;
    }
  }
}

/// Scalar (abs, rel) draws for one channel in legacy order.
fn channel_draws(
  error_type: ErrorType,
  params: &crate::group::ErrorParams,
  rng: &mut dyn RngCore,
) -> (f64, f64) {
  match error_type {
    ErrorType::Gaussian => (
      gauss_draw(params.abs_mean_delta_g, params.abs_std_dev, rng),
      gauss_draw(params.rel_mean_delta_g, params.rel_std_dev, rng),
    ),
    ErrorType::Uniform => (
      unif_draw(params.abs_variance, rng),
      unif_draw(params.rel_variance, rng),
    ),
    ErrorType::Combined => (
      gauss_draw(params.abs_mean_delta_g, params.abs_std_dev, rng)
        + unif_draw(params.abs_variance, rng),
      gauss_draw(params.rel_mean_delta_g, params.rel_std_dev, rng)
        + unif_draw(params.rel_variance, rng),
    ),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::layer::Layer;
  use crate::providers::{DictProvider, Entry};
  use std::collections::{HashMap, HashSet};

  const WL: [f64; 2] = [1000.0, 1500.0];

  fn mats() -> DictProvider {
    let mut entries = HashMap::new();
    entries.insert("glass".to_string(), Entry::Array(vec![Complex64::new(1.52, 0.0), Complex64::new(1.51, 0.0)]));
    entries.insert("TiO2".to_string(), Entry::Array(vec![Complex64::new(2.35, 0.01), Complex64::new(2.33, 0.008)]));
    DictProvider::with_grid(entries, WL.to_vec()).unwrap()
  }

  fn flat_seq() -> Vec<(Layer, bool)> {
    vec![
      (Layer::film(0.0, "glass"), false),
      (Layer::film(50.0, "TiO2"), false),
      (Layer::film(0.0, "glass"), false),
    ]
  }

  fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
      assert!((x - y).abs() < 1e-12, "{x} vs {y}");
    }
  }

  fn close_c(a: &[Complex64], re: &[f64], im: &[f64]) {
    assert_eq!(a.len(), re.len());
    for (z, (r, i)) in a.iter().zip(re.iter().zip(im)) {
      assert!((z.re - r).abs() < 1e-12, "{} vs {r}", z.re);
      assert!((z.im - i).abs() < 1e-12, "{} vs {i}", z.im);
    }
  }

  /// Oracle twin: flat forward (Python FLAT).
  #[test]
  fn flat_forward_matches_python() {
    let (sa, spans) = expand(&flat_seq(), &mats(), &WL, &HashMap::new(), ExpandOptions::deterministic()).unwrap();
    assert_eq!(sa.n_rows(), 3);
    close(&sa.thicknesses, &[0.0, 50.0, 0.0]);
    close_c(sa.row(1), &[2.35, 2.33], &[0.01, 0.008]);
    assert_eq!(sa.incoherent, vec![false; 3]);
    assert_eq!(sa.rough_types, vec![0; 3]);
    assert_eq!(spans, vec![
      Span { start: 0, end: 1, logical: 0 },
      Span { start: 1, end: 2, logical: 1 },
      Span { start: 2, end: 3, logical: 2 },
    ]);
  }

  /// Oracle twin: scaling + slice + grading + roughness (Python FULL).
  #[test]
  fn full_stack_matches_python() {
    let mut layers = vec![(Layer::film(0.0, "glass"), false)];
    let mut l = Layer::film(50.0, "TiO2");
    l.roughness = 2.0;
    l.rough_type = RoughnessType::Gaussian;
    l.interface = true;
    l.interface_thickness = 6.0;
    l.inhomogen = true;
    l.inh_delta = 0.2;
    layers.push((l, false));
    let mut groups = HashMap::new();
    let mut g = Group::new("TiO2");
    g.thick_factor = 1.1;
    g.thick_summand = 1.0;
    g.n_factor = 1.05;
    g.k_factor = 0.5;
    g.roughness_summand = 1.0;
    g.interface_summand = 2.0;
    groups.insert("TiO2".to_string(), g);
    let (sa, _) = expand(&layers, &mats(), &WL, &groups, ExpandOptions::deterministic()).unwrap();
    assert_eq!(sa.n_rows(), 13);
    let mut want_t = vec![0.0, 8.0];
    want_t.extend(vec![48.0 / 11.0; 11]);
    close(&sa.thicknesses, &want_t);
    // Slice mix + first/last graded factors (2.4675/2.4465 x 0.9 … x 1.1).
    close_c(sa.row(1), &[1.974736420464, 1.959531653889], &[0.002321082511, 0.00185737228]);
    close_c(sa.row(2), &[2.22075, 2.20185], &[0.0045, 0.0036]);
    close_c(sa.row(12), &[2.71425, 2.69115], &[0.0055, 0.0044]);
    assert_eq!(sa.rough_types[2], 4);
    assert!((sa.rough_vals[2] - 3.0).abs() < 1e-12);
    assert!(sa.rough_vals[3..].iter().all(|v| *v == 0.0));
  }

  /// Oracle twin: whole-chain mirror appends cleanly (Python INV).
  #[test]
  fn inverted_mirror_matches_python() {
    let mut seq = flat_seq();
    let mut mirror: Vec<(Layer, bool)> =
      flat_seq().into_iter().rev().map(|(l, _)| (l, true)).collect();
    seq.append(&mut mirror);
    let (sa, _) = expand(&seq, &mats(), &WL, &HashMap::new(), ExpandOptions::deterministic()).unwrap();
    assert_eq!(sa.n_rows(), 6);
    close(&sa.thicknesses, &[0.0, 50.0, 0.0, 0.0, 50.0, 0.0]);
    close_c(sa.row(4), &[2.35, 2.33], &[0.01, 0.008]);
  }

  #[test]
  fn empty_sequence_refused() {
    assert!(expand(&[], &mats(), &WL, &HashMap::new(), ExpandOptions::deterministic()).is_err());
  }

  #[test]
  fn error_paths_deterministic_per_seed() {
    let mut groups = HashMap::new();
    let mut g = Group::new("TiO2");
    g.error_mask = [1, 1, 1, 1, 1, 1];
    groups.insert("TiO2".to_string(), g);
    let opts = ExpandOptions { apply_errors: true, seed: Some(11) };
    let (a, _) = expand(&flat_seq(), &mats(), &WL, &groups, opts).unwrap();
    let (b, _) = expand(&flat_seq(), &mats(), &WL, &groups, opts).unwrap();
    assert_eq!(a, b);
    // Draws perturbed something (thickness floored draws or nk offsets).
    let plain = mats().nk("TiO2", &WL).unwrap();
    let mut expected = Vec::new();
    for _ in 0..3 {
      expected.extend(plain.clone());
    }
    assert!(a.thicknesses != vec![0.0, 50.0, 0.0] || a.indices != expected);
  }

  #[test]
  fn spans_cover_rows_exactly_once() {
    let (sa, spans) = expand(&flat_seq(), &mats(), &WL, &HashMap::new(), ExpandOptions::deterministic()).unwrap();
    let mut seen = HashSet::new();
    for s in &spans {
      for r in s.start..s.end {
        assert!(seen.insert(r), "row {r} covered twice");
      }
    }
    assert_eq!(seen.len(), sa.n_rows());
  }
}
