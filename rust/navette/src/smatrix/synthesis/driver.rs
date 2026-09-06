//! synthesis::driver — array-level design assembly + end-to-end runs.
//!
//! Rust-first port of the Python `stack_from_layers`/`run_needle`
//! orchestration: evaluated nk arrays + flag structs in, expanded
//! `DesignStack` or full `PipelineResult` out. Config-file assembly
//! (`DesignRequest`) lives in `design_config`; both converge on
//! `DesignStack::from_design` here through [`assemble_stack`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use num_complex::Complex64;

use super::cycle::ContrastMap;
use super::evaluator::SmatrixContext;
use super::pipeline::{NeedlePipeline, PipelineResult};
use super::structure::{DesignStack, LayerSpec};
use super::config::PipelineConfig;
use super::cycle::NeedleCycleConfig;
use super::thick_opt::LmConfig;
use super::merit::MeritSpec;
use super::pipeline::SpectralInputs;
use crate::structure::{Group, Layer, LayerType};

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// One film: evaluated nk on the grid + authoring flags.
#[derive(Clone, Debug)]
pub struct ArrayFilm {
  pub name: String,
  pub nk: Vec<Complex64>,
  pub d_nm: f64,
  pub coherent: bool,
  pub roughness: f64,
  pub rough_type: i32,
  pub inhomogen: bool,
  pub inh_delta: f64,
  pub interface: bool,
  pub interface_thickness: f64,
  pub optimize: bool,
  pub needle: bool,
}

/// One contrast seed: host name → fresh seed carrier.
#[derive(Clone, Debug)]
pub struct ArraySeed {
  pub host: String,
  pub seed_name: String,
  pub nk: Vec<Complex64>,
}

// ---------------------------------------------------------------------------
// Assembly + runs
// ---------------------------------------------------------------------------

fn fixed_half(name: &str, nk: Vec<Complex64>) -> LayerSpec {
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

/// Expand evaluated arrays into a `DesignStack` (single `from_design`).
/// Returns `(stack, warnings)` — warnings re-emit upstream.
#[allow(clippy::too_many_arguments)]
pub fn assemble_stack(
  ambient_name: &str,
  ambient_nk: Vec<Complex64>,
  substrate_name: &str,
  substrate_nk: Vec<Complex64>,
  films: &[ArrayFilm],
  groups: &HashMap<String, Group>,
  wavelengths: &[f64],
) -> Result<(DesignStack, Vec<String>), String> {
  use crate::structure::RoughnessType;
  let mut seen = HashSet::new();
  for f in films {
    if !seen.insert(f.name.as_str()) {
      return Err(format!(
        "duplicate film name {:?} (film names key the nk table)",
        f.name
      ));
    }
    if f.nk.len() != wavelengths.len() {
      return Err(format!(
        "film {:?}: nk length {} != {} wavelengths",
        f.name,
        f.nk.len(),
        wavelengths.len()
      ));
    }
  }
  let mut design = Vec::with_capacity(films.len());
  let mut nk_map: HashMap<Arc<str>, Vec<Complex64>> = HashMap::new();
  for f in films {
    let mut layer = Layer::film(f.d_nm, &f.name);
    layer.layer_type = LayerType::Film;
    layer.coherent = f.coherent;
    layer.roughness = f.roughness;
    layer.rough_type =
      RoughnessType::try_from_i32(f.rough_type).map_err(|e| format!("film {:?}: {e}", f.name))?;
    layer.inhomogen = f.inhomogen;
    layer.inh_delta = f.inh_delta;
    layer.interface = f.interface;
    layer.interface_thickness = f.interface_thickness;
    layer.optimize = f.optimize;
    layer.needle = f.needle;
    nk_map.insert(Arc::from(f.name.as_str()), f.nk.clone());
    design.push(layer);
  }
  // Background is implied, not declared (mirrors the other drivers).
  let background: HashSet<String> = design
    .iter()
    .filter(|l| l.inhomogen && !l.optimize && !l.needle)
    .map(|l| l.material.clone())
    .collect();
  DesignStack::from_design(
    fixed_half(ambient_name, ambient_nk),
    fixed_half(substrate_name, substrate_nk),
    &design,
    &nk_map,
    groups,
    wavelengths,
    &background,
  )
}

/// End-to-end design run: assemble, fold demands, execute the macro-loop.
/// `angles_deg` in degrees; `callback` fires per macro-cycle.
#[allow(clippy::too_many_arguments)]
pub fn run_design(
  ambient_name: &str,
  ambient_nk: Vec<Complex64>,
  substrate_name: &str,
  substrate_nk: Vec<Complex64>,
  films: &[ArrayFilm],
  groups: &HashMap<String, Group>,
  seeds: &[ArraySeed],
  wavelengths: &[f64],
  angles_deg: &[f64],
  spec: &MeritSpec,
  cfg: PipelineConfig,
  needle_cfg: NeedleCycleConfig,
  lm: LmConfig,
  mut callback: impl FnMut(usize, &super::pipeline::PipelinePhaseResult) -> Result<(), String>,
) -> Result<(PipelineResult, DesignStack), String> {
  let (stack, _warnings) = assemble_stack(
    ambient_name,
    ambient_nk,
    substrate_name,
    substrate_nk,
    films,
    groups,
    wavelengths,
  )?;
  let mut cmap = ContrastMap::new();
  for s in seeds {
    if s.nk.len() != wavelengths.len() {
      return Err(format!(
        "contrast '{}': nk length {} != {} wavelengths",
        s.host,
        s.nk.len(),
        wavelengths.len()
      ));
    }
    cmap.insert(
      Arc::from(s.host.as_str()),
      LayerSpec {
        material: Arc::from(s.seed_name.as_str()),
        nk: Arc::from(s.nk.clone()),
        d_nm: 0.0,
        coherent: true,
        rough_type: 0,
        rough_val: 0.0,
        optimize: true,
        needle: true,
      },
    );
  }
  let spectral = SpectralInputs::from_spec(spec, angles_deg, wavelengths)?;
  let cfg = cfg.validated()?;
  let sin_theta: Vec<f64> = angles_deg.iter().map(|a| a.to_radians().sin()).collect();
  let mut ctx = SmatrixContext {
    wavls: wavelengths.to_vec(),
    sin_theta,
    spec: spec.clone(),
    clamp_min_nm: cfg.clamp_min_nm,
    clamp_max_nm: cfg.clamp_max_nm,
    lm,
  };
  let mut pipe = NeedlePipeline::new(stack, spectral, cfg, needle_cfg, cmap)?;
  let report = pipe.run(&mut ctx, |cycle, phase, _det| callback(cycle, phase))?;
  Ok((report, pipe.stack))
}

// ---------------------------------------------------------------------------
// Tests (standalone: no Python)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn films() -> Vec<ArrayFilm> {
    vec![
      ArrayFilm {
        name: "L".to_string(),
        nk: vec![Complex64::new(1.45, 0.0); 2],
        d_nm: 100.0,
        coherent: true,
        roughness: 0.0,
        rough_type: 0,
        inhomogen: false,
        inh_delta: 0.1,
        interface: false,
        interface_thickness: 0.0,
        optimize: true,
        needle: true,
      },
      ArrayFilm {
        name: "H".to_string(),
        nk: vec![Complex64::new(2.1, 0.0); 2],
        d_nm: 60.0,
        coherent: true,
        roughness: 0.0,
        rough_type: 0,
        inhomogen: false,
        inh_delta: 0.1,
        interface: false,
        interface_thickness: 0.0,
        optimize: true,
        needle: true,
      },
    ]
  }

  #[test]
  fn assemble_flat_stack() {
    let wl = vec![500.0, 600.0];
    let (stack, warns) = assemble_stack(
      "air",
      vec![Complex64::new(1.0, 0.0); 2],
      "sub",
      vec![Complex64::new(1.52, 0.0); 2],
      &films(),
      &HashMap::new(),
      &wl,
    )
    .unwrap();
    assert!(warns.is_empty());
    assert_eq!(stack.films().len(), 2);
  }

  #[test]
  fn assemble_refuses() {
    let wl = vec![500.0, 600.0];
    let air = vec![Complex64::new(1.0, 0.0); 2];
    let sub = vec![Complex64::new(1.52, 0.0); 2];
    let mut dup = films();
    dup[1].name = "L".to_string();
    assert!(assemble_stack("air", air.clone(), "sub", sub.clone(), &dup, &HashMap::new(), &wl).is_err());
    let mut bad = films();
    bad[0].nk = vec![Complex64::new(1.45, 0.0); 3];
    assert!(assemble_stack("air", air, "sub", sub, &bad, &HashMap::new(), &wl).is_err());
  }
}
