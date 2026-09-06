//! smatrix::solver — configured thin-film solver over `core_engine`.
//!
//! Rust-first port of the Python `ScatterMatrix` driver: input validation,
//! the parallel per-point solve, per-point derives (Stokes, Psi/Delta,
//! phases), the cross-wavelength dispersion post-pass, and keyed output.
//! The PyO3 `core_engine` binding thins onto this; Rust consumers solve
//! with no Python.

use std::f64::consts::PI;

use num_complex::{Complex64, ComplexFloat};
use rayon::prelude::*;

use super::core_engine::*;
use super::needle_engine::*;
use super::optimizer::{char_func, char_func_xy};
use super::optics_core::{redheffer_product_complex_field_inner, w_function_inner};
use super::needle_operator::*;
use super::optics_core::C_NM_PER_FS;

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Configured solver: validated inputs + precomputed index caches.
/// `indices_layer_major` is `(n_layers, n_wavs)` row-major complex.
pub struct Solver {
  wavls: Vec<f64>,
  sin_theta: Vec<f64>,
  n_cache: Vec<Vec<Complex64>>,
  inv_n_cache: Vec<Vec<Complex64>>,
  flat_cache: Vec<f64>,
  thicknesses: Vec<f64>,
  incoherent_flags: Vec<i32>,
  rough_types: Vec<i32>,
  rough_vals: Vec<f64>,
  coherence_mode: i32,
  n_layers: usize,
}

/// Request masks for the `ScatterMatrix` convenience views (mirror the
/// Python view methods exactly; `pol` is 's', 'p' or 'u').
pub fn rt_request(pol: &str) -> Result<u64, String> {
  let mut req = REQ_R_AVG | REQ_T_AVG;
  match pol {
    "s" => req |= REQ_RS | REQ_TS,
    "p" => req |= REQ_RP | REQ_TP,
    "u" => req |= REQ_RS | REQ_TS | REQ_RP | REQ_TP,
    _ => return Err("pol must be 's', 'p', or 'u'".to_string()),
  }
  Ok(req)
}

pub fn ellipsometry_request(transmission: bool) -> u64 {
  let mut req = REQ_PSI_R | REQ_DELTA_R | REQ_DOP_R | REQ_RS | REQ_RP | REQ_R_AVG;
  if transmission {
    req |= REQ_PSI_T | REQ_DELTA_T | REQ_DOP_T | REQ_TS | REQ_TP | REQ_T_AVG;
  }
  req
}

pub fn absorption_request() -> u64 {
  REQ_A_S | REQ_A_P | REQ_A_AVG
}

pub fn amplitudes_request() -> u64 {
  REQ_RS_C | REQ_RP_C | REQ_TS_C | REQ_TP_C
}

pub fn stokes_request(reflection: bool, transmission: bool) -> Result<u64, String> {
  let mut req = 0;
  if reflection {
    req |= REQ_S0_R | REQ_S1_R | REQ_S2_R | REQ_S3_R;
  }
  if transmission {
    req |= REQ_S0_T | REQ_S1_T | REQ_S2_T | REQ_S3_T;
  }
  if req == 0 {
    return Err("select reflection and/or transmission".to_string());
  }
  Ok(req)
}

pub fn dispersion_request(
  reflection: bool,
  transmission: bool,
  s_pol: bool,
  p_pol: bool,
) -> Result<u64, String> {
  let mut req = 0;
  if reflection && s_pol {
    req |= REQ_DISP_R_S;
  }
  if reflection && p_pol {
    req |= REQ_DISP_R_P;
  }
  if transmission && s_pol {
    req |= REQ_DISP_T_S;
  }
  if transmission && p_pol {
    req |= REQ_DISP_T_P;
  }
  if req == 0 {
    return Err("select at least one channel for dispersion".to_string());
  }
  Ok(req)
}

/// `max(|1-Rs-Ts|, |1-Rp-Tp|)` per grid point (energy-conservation residual).
/// Slices must share length; empty input is refused.
pub fn energy_conservation(
  rs: &[f64],
  rp: &[f64],
  ts: &[f64],
  tp: &[f64],
) -> Result<Vec<f64>, String> {
  if rs.len() != rp.len() || rs.len() != ts.len() || rs.len() != tp.len() {
    return Err("energy_conservation: R/T slices must share length".to_string());
  }
  if rs.is_empty() {
    return Err("energy_conservation: empty input".to_string());
  }
  Ok(rs.iter().zip(rp).zip(ts).zip(tp)
    .map(|(((s, p), t_s), t_p)| ((1.0 - s - t_s).abs()).max((1.0 - p - t_p).abs()))
    .collect())
}

impl Solver {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    wavelengths: &[f64],
    sin_theta: &[f64],
    indices_layer_major: &[Complex64],
    n_layers: usize,
    thicknesses: &[f64],
    incoherent_flags: &[i32],
    rough_types: &[i32],
    rough_vals: &[f64],
    coherence_mode: i32,
  ) -> Result<Self, String> {
    if wavelengths.is_empty() {
      return Err("wavelengths must be non-empty".to_string());
    }
    if sin_theta.is_empty() {
      return Err("angles must be non-empty".to_string());
    }
    if n_layers < 2 {
      return Err("need at least 2 layers (ambient + substrate)".to_string());
    }
    if indices_layer_major.len() != n_layers * wavelengths.len() {
      return Err(format!(
        "indices length {} must equal n_layers ({}) × n_wavs ({})",
        indices_layer_major.len(),
        n_layers,
        wavelengths.len()
      ));
    }
    for (name, v) in [
      ("thicknesses", thicknesses.len()),
      ("incoherent_flags", incoherent_flags.len()),
      ("roughness_types", rough_types.len()),
      ("roughness_values", rough_vals.len()),
    ] {
      if v != n_layers {
        return Err(format!("{name} length {v} must equal n_layers {n_layers}"));
      }
    }
    if !(MODE_A..=MODE_C).contains(&coherence_mode) {
      return Err(
        "coherence_mode must be 0 (front_block), 1 (coherency_matrix), or 2 (fully_coherent)."
          .to_string(),
      );
    }
    let n_wavs = wavelengths.len();
    let mut flat_cache = Vec::with_capacity(n_layers * n_wavs * 2);
    for w in 0..n_wavs {
      for l in 0..n_layers {
        let nv = indices_layer_major[l * n_wavs + w];
        flat_cache.push(nv.re);
        flat_cache.push(nv.im);
      }
    }
    let mut n_cache = Vec::with_capacity(n_wavs);
    let mut inv_n_cache = Vec::with_capacity(n_wavs);
    for w in 0..n_wavs {
      let mut layer_n = Vec::with_capacity(n_layers);
      let mut layer_inv = Vec::with_capacity(n_layers);
      for l in 0..n_layers {
        let nv = indices_layer_major[l * n_wavs + w];
        layer_n.push(nv);
        layer_inv.push(nv.recip());
      }
      n_cache.push(layer_n);
      inv_n_cache.push(layer_inv);
    }
    Ok(Self {
      wavls: wavelengths.to_vec(),
      sin_theta: sin_theta.to_vec(),
      n_cache,
      inv_n_cache,
      flat_cache,
      thicknesses: thicknesses.to_vec(),
      incoherent_flags: incoherent_flags.to_vec(),
      rough_types: rough_types.to_vec(),
      rough_vals: rough_vals.to_vec(),
      coherence_mode,
      n_layers,
    })
  }

  /// Raw-input constructor mirroring the Python driver: angles in degrees
  /// unless `radians` is set; `indices` is either per-layer (len ==
  /// `n_layers`, broadcast over wavelengths) or full `(n_layers, n_wavs)`
  /// row-major; per-layer option slices default when `None`.
  #[allow(clippy::too_many_arguments)]
  pub fn from_raw(
    wavelengths: &[f64],
    angles: &[f64],
    radians: bool,
    indices: &[Complex64],
    n_layers: usize,
    thicknesses: Option<&[f64]>,
    incoherent_flags: Option<&[i32]>,
    roughness_types: Option<&[i32]>,
    roughness_values: Option<&[f64]>,
    coherence_mode: i32,
  ) -> Result<Self, String> {
    if wavelengths.is_empty() {
      return Err("`wavelengths` must be non-empty.".to_string());
    }
    if angles.is_empty() {
      return Err("`angles` must be non-empty.".to_string());
    }
    let nw = wavelengths.len();
    let full = if indices.len() == n_layers {
      let mut v = Vec::with_capacity(n_layers * nw);
      for l in 0..n_layers {
        for _ in 0..nw {
          v.push(indices[l]);
        }
      }
      v
    } else if indices.len() == n_layers * nw {
      indices.to_vec()
    } else {
      return Err("`indices` must be per-layer or (n_layers, n_wavs).".to_string());
    };
    let theta: Vec<f64> = angles
      .iter()
      .map(|a| if radians { *a } else { a.to_radians() })
      .collect();
    let sin_theta: Vec<f64> = theta.iter().map(|t| t.sin()).collect();
    let zero_f = vec![0.0; n_layers];
    let zero_i = vec![0; n_layers];
    Self::new(
      wavelengths,
      &sin_theta,
      &full,
      n_layers,
      thicknesses.unwrap_or(&zero_f),
      incoherent_flags.unwrap_or(&zero_i),
      roughness_types.unwrap_or(&zero_i),
      roughness_values.unwrap_or(&zero_f),
      coherence_mode,
    )
  }

  pub fn n_angles(&self) -> usize {
    self.sin_theta.len()
  }

  pub fn n_wavs(&self) -> usize {
    self.wavls.len()
  }

  /// Needle-operator gradients over the solver's own grid and stack.
  /// Per-point target/weight inputs are scalars (broadcast) or full
  /// angle-major vectors; `None` means target 0 / weight 1.
  #[allow(clippy::too_many_arguments)]
  pub fn needle_gradient(
    &self,
    needle_n_per_wav: &[Complex64],
    z_grid: &[f64],
    requested: u64,
    incoherent_flags: Option<&[i32]>,
    targets_r: Option<&[f64]>,
    weights_r: Option<&[f64]>,
    targets_t: Option<&[f64]>,
    weights_t: Option<&[f64]>,
    targets_a: Option<&[f64]>,
    weights_a: Option<&[f64]>,
    targets_phi: Option<&[f64]>,
    weights_phi: Option<&[f64]>,
    targets_tb: Option<&[f64]>,
    weights_tb: Option<&[f64]>,
    targets_rb: Option<&[f64]>,
    weights_rb: Option<&[f64]>,
    targets_ab: Option<&[f64]>,
    weights_ab: Option<&[f64]>,
    start_idx: usize,
    end_idx: Option<usize>,
    channel: usize,
    calc_s: bool,
    calc_p: bool,
    host_mask: Option<&[bool]>,
    gain_shift_phi: f64,
  ) -> Result<NeedleSolution, String> {
    needle_gradient(
      &self.wavls,
      &self.sin_theta,
      self.n_layers,
      &self.flat_cache,
      &self.thicknesses,
      &self.rough_types,
      &self.rough_vals,
      needle_n_per_wav,
      z_grid,
      requested,
      incoherent_flags.or(Some(&self.incoherent_flags)),
      targets_r,
      weights_r,
      targets_t,
      weights_t,
      targets_a,
      weights_a,
      targets_phi,
      weights_phi,
      targets_tb,
      weights_tb,
      targets_rb,
      weights_rb,
      targets_ab,
      weights_ab,
      start_idx,
      end_idx,
      channel,
      calc_s,
      calc_p,
      host_mask,
      gain_shift_phi,
    )
  }

  /// Solve every (angle, wavelength) point for `requested` and derive the
  /// keyed output maps (flat `[n_angles × n_wavs]` angle-major).
  pub fn solve(&self, requested: u64) -> Result<Solution, String> {
    if requested == 0 {
      return Err("empty request mask: select at least one observable".to_string());
    }
    let num_wavs = self.wavls.len();
    let num_angles = self.sin_theta.len();
    let total_points = num_wavs * num_angles;
    let idx_n = self.n_layers - 1;

    let Plan { need_s, need_p, need_cross, level } = resolve_plan(requested);
    let want_phi_rs = requested & (REQ_PHI_RS | REQ_DISP_R_S) != 0;
    let want_phi_rp = requested & (REQ_PHI_RP | REQ_DISP_R_P) != 0;
    let want_phi_ts = requested & (REQ_PHI_TS | REQ_DISP_T_S) != 0;
    let want_phi_tp = requested & (REQ_PHI_TP | REQ_DISP_T_P) != 0;

    let states: Vec<OpticalState> = (0..total_points)
      .into_par_iter()
      .map(|k| {
        let a = k / num_wavs;
        let w = k % num_wavs;
        match level {
          Level::Intensities => solve_point_intensity(
            idx_n,
            self.wavls[w],
            self.sin_theta[a],
            &self.n_cache[w],
            &self.inv_n_cache[w],
            &self.thicknesses,
            &self.incoherent_flags,
            &self.rough_types,
            &self.rough_vals,
            self.coherence_mode,
            need_s,
            need_p,
          ),
          Level::ComplexAmps | Level::Cross => solve_point(
            idx_n,
            self.wavls[w],
            self.sin_theta[a],
            &self.n_cache[w],
            &self.inv_n_cache[w],
            &self.thicknesses,
            &self.incoherent_flags,
            &self.rough_types,
            &self.rough_vals,
            self.coherence_mode,
            need_s,
            need_p,
            need_cross,
          ),
        }
      })
      .collect();

    macro_rules! f64buf {
      ($name:ident, $cond:expr) => {
        let mut $name: Option<Vec<f64>> =
          if $cond { Some(vec![0.0; total_points]) } else { None };
      };
    }
    macro_rules! cbuf {
      ($name:ident, $bit:expr) => {
        let mut $name: Option<Vec<Complex64>> = if requested & $bit != 0 {
          Some(vec![Complex64::new(0.0, 0.0); total_points])
        } else {
          None
        };
      };
    }
    macro_rules! put {
      ($buf:ident, $k:expr, $val:expr) => {
        if let Some(b) = $buf.as_mut() {
          b[$k] = $val;
        }
      };
    }

    f64buf!(b_rs, requested & REQ_RS != 0);
    f64buf!(b_rp, requested & REQ_RP != 0);
    f64buf!(b_ts, requested & REQ_TS != 0);
    f64buf!(b_tp, requested & REQ_TP != 0);
    f64buf!(b_ravg, requested & REQ_R_AVG != 0);
    f64buf!(b_tavg, requested & REQ_T_AVG != 0);
    f64buf!(b_as, requested & REQ_A_S != 0);
    f64buf!(b_ap, requested & REQ_A_P != 0);
    f64buf!(b_aavg, requested & REQ_A_AVG != 0);
    f64buf!(b_psi_r, requested & REQ_PSI_R != 0);
    f64buf!(b_psi_t, requested & REQ_PSI_T != 0);
    f64buf!(b_delta_r, requested & REQ_DELTA_R != 0);
    f64buf!(b_delta_t, requested & REQ_DELTA_T != 0);
    f64buf!(b_dop_r, requested & REQ_DOP_R != 0);
    f64buf!(b_dop_t, requested & REQ_DOP_T != 0);
    f64buf!(b_diatt_r, requested & REQ_DIATT_R != 0);
    f64buf!(b_diatt_t, requested & REQ_DIATT_T != 0);
    f64buf!(b_s0r, requested & REQ_S0_R != 0);
    f64buf!(b_s1r, requested & REQ_S1_R != 0);
    f64buf!(b_s2r, requested & REQ_S2_R != 0);
    f64buf!(b_s3r, requested & REQ_S3_R != 0);
    f64buf!(b_s0t, requested & REQ_S0_T != 0);
    f64buf!(b_s1t, requested & REQ_S1_T != 0);
    f64buf!(b_s2t, requested & REQ_S2_T != 0);
    f64buf!(b_s3t, requested & REQ_S3_T != 0);
    f64buf!(b_retard_r, requested & REQ_RETARD_R != 0);
    f64buf!(b_retard_t, requested & REQ_RETARD_T != 0);
    f64buf!(b_phi_rs, want_phi_rs);
    f64buf!(b_phi_rp, want_phi_rp);
    f64buf!(b_phi_ts, want_phi_ts);
    f64buf!(b_phi_tp, want_phi_tp);
    f64buf!(b_phi_rbs, requested & REQ_PHI_RBS != 0);
    f64buf!(b_phi_rbp, requested & REQ_PHI_RBP != 0);
    f64buf!(b_phi_tbs, requested & REQ_PHI_TBS != 0);
    f64buf!(b_phi_tbp, requested & REQ_PHI_TBP != 0);

    cbuf!(b_rs_c, REQ_RS_C);
    cbuf!(b_rp_c, REQ_RP_C);
    cbuf!(b_ts_c, REQ_TS_C);
    cbuf!(b_tp_c, REQ_TP_C);
    cbuf!(b_rbs_c, REQ_RBS_C);
    cbuf!(b_rbp_c, REQ_RBP_C);
    cbuf!(b_tbs_c, REQ_TBS_C);
    cbuf!(b_tbp_c, REQ_TBP_C);
    cbuf!(b_cross_r, REQ_CROSS_R);
    cbuf!(b_cross_t, REQ_CROSS_T);

    for (k, s) in states.iter().enumerate() {
      let rs = s.rs;
      let rp = s.rp;
      let ts = s.ts;
      let tp = s.tp;

      put!(b_rs, k, rs);
      put!(b_rp, k, rp);
      put!(b_ts, k, ts);
      put!(b_tp, k, tp);
      put!(b_ravg, k, 0.5 * (rs + rp));
      put!(b_tavg, k, 0.5 * (ts + tp));
      put!(b_as, k, 1.0 - rs - ts);
      put!(b_ap, k, 1.0 - rp - tp);
      put!(b_aavg, k, 1.0 - 0.5 * (rs + rp) - 0.5 * (ts + tp));

      let s0r = rp + rs;
      let s1r = rp - rs;
      let s2r = -2.0 * s.cross_r.re + 0.0;
      let s3r = -2.0 * s.cross_r.im + 0.0;
      put!(b_s0r, k, s0r);
      put!(b_s1r, k, s1r);
      put!(b_s2r, k, s2r);
      put!(b_s3r, k, s3r);
      let s0t = tp + ts;
      let s1t = tp - ts;
      let s2t = 2.0 * s.cross_t.re + 0.0;
      let s3t = 2.0 * s.cross_t.im + 0.0;
      put!(b_s0t, k, s0t);
      put!(b_s1t, k, s1t);
      put!(b_s2t, k, s2t);
      put!(b_s3t, k, s3t);

      put!(b_diatt_r, k, s1r / (s0r + 1e-20));
      put!(b_diatt_t, k, s1t / (s0t + 1e-20));

      put!(b_dop_r, k, (s1r * s1r + s2r * s2r + s3r * s3r).sqrt() / (s0r + 1e-20));
      put!(b_dop_t, k, ((s1t * s1t + s2t * s2t + s3t * s3t).sqrt() / (s0t + 1e-20)).min(1.0));

      put!(b_psi_r, k, if rs < RS_FLOOR { PI / 2.0 } else { (rp / rs).sqrt().atan() });
      put!(b_delta_r, k, if rs < RS_FLOOR { 0.0 } else { s3r.atan2(s2r) });
      put!(b_psi_t, k, if ts < TS_FLOOR { PI / 2.0 } else { (tp / ts).sqrt().atan() });
      put!(b_delta_t, k, if ts < TS_FLOOR { 0.0 } else { s3t.atan2(s2t) });

      put!(b_retard_r, k, s.cross_r.arg());
      put!(b_retard_t, k, s.cross_t.arg());

      put!(b_phi_rs, k, s.rs_c.arg());
      put!(b_phi_rp, k, s.rp_c.arg());
      put!(b_phi_ts, k, s.ts_c.arg());
      put!(b_phi_tp, k, s.tp_c.arg());
      put!(b_phi_rbs, k, s.rbs_c.arg());
      put!(b_phi_rbp, k, s.rbp_c.arg());
      put!(b_phi_tbs, k, s.tbs_c.arg());
      put!(b_phi_tbp, k, s.tbp_c.arg());

      if let Some(b) = b_rs_c.as_mut() { b[k] = s.rs_c; }
      if let Some(b) = b_rp_c.as_mut() { b[k] = s.rp_c; }
      if let Some(b) = b_ts_c.as_mut() { b[k] = s.ts_c; }
      if let Some(b) = b_tp_c.as_mut() { b[k] = s.tp_c; }
      if let Some(b) = b_rbs_c.as_mut() { b[k] = s.rbs_c; }
      if let Some(b) = b_rbp_c.as_mut() { b[k] = s.rbp_c; }
      if let Some(b) = b_tbs_c.as_mut() { b[k] = s.tbs_c; }
      if let Some(b) = b_tbp_c.as_mut() { b[k] = s.tbp_c; }
      if let Some(b) = b_cross_r.as_mut() { b[k] = s.cross_r; }
      if let Some(b) = b_cross_t.as_mut() { b[k] = s.cross_t; }
    }

    let omega: Vec<f64> =
      self.wavls.iter().map(|&l| 2.0 * PI * C_NM_PER_FS / l).collect();
    let disp = |phi: &Option<Vec<f64>>| -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
      phi.as_ref().map(|p| dispersion_channel(p, &omega, num_angles, num_wavs))
    };
    let disp_r_s = if requested & REQ_DISP_R_S != 0 { disp(&b_phi_rs) } else { None };
    let disp_r_p = if requested & REQ_DISP_R_P != 0 { disp(&b_phi_rp) } else { None };
    let disp_t_s = if requested & REQ_DISP_T_S != 0 { disp(&b_phi_ts) } else { None };
    let disp_t_p = if requested & REQ_DISP_T_P != 0 { disp(&b_phi_tp) } else { None };

    let mut f64maps: Vec<(String, Vec<f64>)> = Vec::new();
    macro_rules! keep_f64 {
      ($name:expr, $buf:expr) => {
        if let Some(b) = $buf {
          f64maps.push(($name.to_string(), b));
        }
      };
    }
    keep_f64!("Rs", b_rs);
    keep_f64!("Rp", b_rp);
    keep_f64!("Ts", b_ts);
    keep_f64!("Tp", b_tp);
    keep_f64!("R_avg", b_ravg);
    keep_f64!("T_avg", b_tavg);
    keep_f64!("A_s", b_as);
    keep_f64!("A_p", b_ap);
    keep_f64!("A_avg", b_aavg);
    keep_f64!("Psi_R", b_psi_r);
    keep_f64!("Psi_T", b_psi_t);
    keep_f64!("Delta_R", b_delta_r);
    keep_f64!("Delta_T", b_delta_t);
    keep_f64!("DOP_R", b_dop_r);
    keep_f64!("DOP_T", b_dop_t);
    keep_f64!("Diattenuation_R", b_diatt_r);
    keep_f64!("Diattenuation_T", b_diatt_t);
    keep_f64!("S0_R", b_s0r);
    keep_f64!("S1_R", b_s1r);
    keep_f64!("S2_R", b_s2r);
    keep_f64!("S3_R", b_s3r);
    keep_f64!("S0_T", b_s0t);
    keep_f64!("S1_T", b_s1t);
    keep_f64!("S2_T", b_s2t);
    keep_f64!("S3_T", b_s3t);
    keep_f64!("Retardance_R", b_retard_r);
    keep_f64!("Retardance_T", b_retard_t);
    // Phases surface only when explicitly requested (not merely
    // dispersion-needed) — mirrors the binding contract.
    if requested & REQ_PHI_RS != 0 { keep_f64!("phi_rs", b_phi_rs); }
    if requested & REQ_PHI_RP != 0 { keep_f64!("phi_rp", b_phi_rp); }
    if requested & REQ_PHI_TS != 0 { keep_f64!("phi_ts", b_phi_ts); }
    if requested & REQ_PHI_TP != 0 { keep_f64!("phi_tp", b_phi_tp); }
    if requested & REQ_PHI_RBS != 0 { keep_f64!("phi_rbs", b_phi_rbs); }
    if requested & REQ_PHI_RBP != 0 { keep_f64!("phi_rbp", b_phi_rbp); }
    if requested & REQ_PHI_TBS != 0 { keep_f64!("phi_tbs", b_phi_tbs); }
    if requested & REQ_PHI_TBP != 0 { keep_f64!("phi_tbp", b_phi_tbp); }

    let mut c64maps: Vec<(String, Vec<Complex64>)> = Vec::new();
    macro_rules! keep_c {
      ($name:expr, $buf:expr) => {
        if let Some(b) = $buf {
          c64maps.push(($name.to_string(), b));
        }
      };
    }
    keep_c!("rs_c", b_rs_c);
    keep_c!("rp_c", b_rp_c);
    keep_c!("ts_c", b_ts_c);
    keep_c!("tp_c", b_tp_c);
    keep_c!("rbs_c", b_rbs_c);
    keep_c!("rbp_c", b_rbp_c);
    keep_c!("tbs_c", b_tbs_c);
    keep_c!("tbp_c", b_tbp_c);
    keep_c!("cross_R", b_cross_r);
    keep_c!("cross_T", b_cross_t);

    let mut dispmaps: Vec<(String, [Vec<f64>; 4])> = Vec::new();
    // NOTE: dispersion key names are assembled by the caller (binding)
    // from the 4-tuple order [GD, GDD, TOD, FOD]; see `DISP_SUFFIXES`.
    if let Some(t) = disp_r_s { dispmaps.push(("R_s".to_string(), [t.0, t.1, t.2, t.3])); }
    if let Some(t) = disp_r_p { dispmaps.push(("R_p".to_string(), [t.0, t.1, t.2, t.3])); }
    if let Some(t) = disp_t_s { dispmaps.push(("T_s".to_string(), [t.0, t.1, t.2, t.3])); }
    if let Some(t) = disp_t_p { dispmaps.push(("T_p".to_string(), [t.0, t.1, t.2, t.3])); }

    Ok(Solution { n_angles: num_angles, n_wavs: num_wavs, f64maps, c64maps, dispmaps })
  }
}

/// Solved output: flat angle-major buffers plus grid shape.
pub struct Solution {
  pub n_angles: usize,
  pub n_wavs: usize,
  pub f64maps: Vec<(String, Vec<f64>)>,
  pub c64maps: Vec<(String, Vec<Complex64>)>,
  /// `(channel, [GD, GDD, TOD, FOD])`; key suffixes per `DISP_SUFFIXES`.
  pub dispmaps: Vec<(String, [Vec<f64>; 4])>,
}

/// Key suffixes for the dispersion 4-tuple, in order.
pub const DISP_SUFFIXES: [(&str, usize); 4] =
  [("GD", 0), ("GDD", 1), ("TOD", 2), ("FOD", 3)];

/// Solve pre-expanded arrays (native `solve_structure` core): grid check,
/// half-space convention warning, one solve. `indices` is `(n_rows,
/// n_wavs)` row-major complex; returns `(solution, warnings)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_arrays(
  indices: &[Complex64],
  thicknesses: &[f64],
  incoherent: &[bool],
  rough_types: &[i32],
  rough_vals: &[f64],
  wavelengths: &[f64],
  angles: &[f64],
  radians: bool,
  requested: u64,
  coherence_mode: i32,
) -> Result<(Solution, Vec<String>), String> {
  let n_rows = thicknesses.len();
  if indices.len() != n_rows * wavelengths.len() {
    return Err(format!(
      "solve_arrays: provider grid mismatch: solver arrays vs {n_rows} layers x {} wavelengths.",
      wavelengths.len()
    ));
  }
  let mut warnings = Vec::new();
  if n_rows >= 2 && (thicknesses[0] != 0.0 || thicknesses[n_rows - 1] != 0.0) {
    warnings.push(
      "solve_arrays: first/last thickness is not 0 (ambient/substrate convention); \
       the engine treats row 0/last as half-spaces."
        .to_string(),
    );
  }
  let inc: Vec<i32> = incoherent.iter().map(|b| i32::from(*b)).collect();
  let solver = Solver::from_raw(
    wavelengths,
    angles,
    radians,
    indices,
    n_rows,
    Some(thicknesses),
    Some(&inc),
    Some(rough_types),
    Some(rough_vals),
    coherence_mode,
  )?;
  Ok((solver.solve(requested)?, warnings))
}

// ---------------------------------------------------------------------------
// Needle gradients (moved verbatim from the PyO3 binding)
// ---------------------------------------------------------------------------

/// Needle-gradient output: flat `[n_points × n_depths]` buffers plus shape.
pub struct NeedleSolution {
  pub n_points: usize,
  pub n_depths: usize,
  pub maps: Vec<(String, Vec<f64>)>,
}

#[allow(clippy::too_many_arguments)]
pub fn needle_gradient(
    wavls: &[f64],
    sin_theta: &[f64],
    n_layers: usize,
    n_stack_cache: &[f64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    needle_n_per_wav: &[Complex64],
    z_grid: &[f64],
    requested: u64,
    incoherent_flags: Option<&[i32]>,
    targets_r: Option<&[f64]>,
    weights_r: Option<&[f64]>,
    targets_t: Option<&[f64]>,
    weights_t: Option<&[f64]>,
    targets_a: Option<&[f64]>,
    weights_a: Option<&[f64]>,
    targets_phi: Option<&[f64]>,
    weights_phi: Option<&[f64]>,
    targets_tb: Option<&[f64]>,
    weights_tb: Option<&[f64]>,
    targets_rb: Option<&[f64]>,
    weights_rb: Option<&[f64]>,
    targets_ab: Option<&[f64]>,
    weights_ab: Option<&[f64]>,
    start_idx: usize,
    end_idx: Option<usize>,
    channel: usize,
    calc_s: bool,
    calc_p: bool,
    host_mask: Option<&[bool]>,
    gain_shift_phi: f64,
) -> Result<NeedleSolution, String>
{
    if requested == 0 {
        return Err(String::from("empty request mask"));
    }

    let num_wavs = wavls.len();
    let num_angles = sin_theta.len();
    let total_points = num_wavs * num_angles;
    let nz = z_grid.len();

    let nl = n_layers;
    if !(0..nl).contains(&start_idx) {
        return Err(String::from("start_idx out of range"));
    }
    let idx_end = end_idx.unwrap_or(nl - 1);
    if idx_end < start_idx + 2 || idx_end >= nl {
        return Err(String::from(
            "end_idx must leave at least one host layer inside [start_idx, end_idx]",
        ));
    }
    if num_wavs == 0 || num_angles == 0 || nz == 0 {
        return Err(String::from("empty grid"));
    }
    if needle_n_per_wav.len() != num_wavs {
        return Err(String::from(
            "needle_n_per_wav must have one complex index per wavelength",
        ));
    }
    if n_stack_cache.len() != num_wavs * nl * 2 {
        return Err(String::from("n_stack_cache layout mismatch"));
    }
    let want_p = requested & NREQ_P != 0;
    let want_pmb = requested & NREQ_P_MB != 0;
    let want_pmb_t = requested & NREQ_P_MB_T != 0;
    let want_pmb_a = requested & NREQ_P_MB_A != 0;
    let want_ptb = requested & NREQ_P_TB != 0;
    let want_prb = requested & NREQ_P_RB != 0;
    let want_pab = requested & NREQ_P_AB != 0;
    let want_pmb_tb = requested & NREQ_P_MB_TB != 0;
    let want_pmb_rb = requested & NREQ_P_MB_RB != 0;
    let want_pmb_ab = requested & NREQ_P_MB_AB != 0;
    let want_pt = requested & NREQ_P_T != 0;
    let want_pa = requested & NREQ_P_A != 0;
    let want_pphi = requested & NREQ_P_PHI != 0;
    let want_disp = max_disp_order(requested).is_some();
    if !calc_s && !calc_p {
        return Err(String::from("no polarization branch enabled"));
    }
    if channel > 3 {
        return Err(String::from("channel must be 0..=3"));
    }

    // Optional per-point merit inputs (default: target 0, weight 1).
    // Scalars broadcast; full vectors are angle-major.
    let load_pair = |a: &Option<&[f64]>, name: &str| -> Result<Option<Vec<f64>>, String> {
        match a {
            Some(arr) => {
                if arr.len() == 1 {
                    Ok(Some(vec![arr[0]; total_points]))
                } else if arr.len() == total_points {
                    Ok(Some(arr.to_vec()))
                } else {
                    Err(String::from(format!(
                        "{name} must be a scalar or have num_angles*num_wavs entries (angle-major)",
                    )))
                }
            }
            None => Ok(None),
        }
    };
    let tgt = load_pair(&targets_r, "targets_r")?;
    let wgt = load_pair(&weights_r, "weights_r")?;
    let target_of = |k: usize| tgt.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_of = |k: usize| wgt.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let tgt_t = load_pair(&targets_t, "targets_t")?;
    let wgt_t = load_pair(&weights_t, "weights_t")?;
    let tgt_a = load_pair(&targets_a, "targets_a")?;
    let wgt_a = load_pair(&weights_a, "weights_a")?;
    let tgt_phi = load_pair(&targets_phi, "targets_phi")?;
    let wgt_phi = load_pair(&weights_phi, "weights_phi")?;
    let target_t_of = |k: usize| tgt_t.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_t_of = |k: usize| wgt_t.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_a_of = |k: usize| tgt_a.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_a_of = |k: usize| wgt_a.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_phi_of = |k: usize| tgt_phi.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_phi_of = |k: usize| wgt_phi.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let tgt_tb = load_pair(&targets_tb, "targets_tb")?;
    let wgt_tb = load_pair(&weights_tb, "weights_tb")?;
    let tgt_rb = load_pair(&targets_rb, "targets_rb")?;
    let wgt_rb = load_pair(&weights_rb, "weights_rb")?;
    let tgt_ab = load_pair(&targets_ab, "targets_ab")?;
    let wgt_ab = load_pair(&weights_ab, "weights_ab")?;
    let target_tb_of = |k: usize| tgt_tb.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_tb_of = |k: usize| wgt_tb.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_rb_of = |k: usize| tgt_rb.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_rb_of = |k: usize| wgt_rb.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_ab_of = |k: usize| tgt_ab.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_ab_of = |k: usize| wgt_ab.as_ref().map(|t| t[k]).unwrap_or(1.0);

    // Incoherent flags only needed for the multiblock path.
    let want_any_pmb =
        want_pmb || want_pmb_t || want_pmb_a || want_pmb_tb || want_pmb_rb || want_pmb_ab;
    let inc = match (&incoherent_flags, want_any_pmb) {
        (_, false) => None,
        (None, true) => {
            return Err(String::from(
                "NREQ_P_MB requires incoherent_flags",
            ))
        }
        (Some(a), true) => {
            let v = a;
            if v.len() != nl {
                return Err(String::from(
                    "incoherent_flags must have n_layers entries",
                ));
            }
            Some(v.to_vec())
        }
    };
    let mask = match &host_mask {
        Some(a) => {
            let v = a;
            if v.len() != nl {
                return Err(String::from(
                    "host_mask must have n_layers entries",
                ));
            }
            Some(v.to_vec())
        }
        None => None,
    };

    // Host maps are geometry-only: compute once, share across all points.
    let mb_locs = match &inc {
        Some(flags) => Some(locate_hosts_multiblock(thicknesses, flags, z_grid, mask.as_deref())
            ?),
        None => None,
    };
    let coh_locs: Vec<(usize, f64)> =
        if want_p || want_pt || want_pa || want_pphi || want_ptb || want_prb || want_pab || want_disp {
        z_grid
            .iter()
            .map(|&z| locate_depth_in(thicknesses, start_idx, idx_end, z))
            .collect()
    } else {
        Vec::new()
    };

    struct PointOut {
        p: [Option<Vec<f64>>; 2],
        pmb: [Option<Vec<f64>>; 2],
        q: [Option<Vec<f64>>; 2], // Q rows (order 0), flattened nz
        pt: [Option<Vec<f64>>; 2],
        pa: [Option<Vec<f64>>; 2],
        pphi: [Option<Vec<f64>>; 2],
        pmb_t: [Option<Vec<f64>>; 2],
        pmb_a: [Option<Vec<f64>>; 2],
        ptb: [Option<Vec<f64>>; 2],
        prb: [Option<Vec<f64>>; 2],
        pab: [Option<Vec<f64>>; 2],
        pmb_tb: [Option<Vec<f64>>; 2],
        pmb_rb: [Option<Vec<f64>>; 2],
        pmb_ab: [Option<Vec<f64>>; 2],
    }
    impl PointOut {
        fn empty() -> Self {
            PointOut {
                p: [None, None], pmb: [None, None], q: [None, None],
                pt: [None, None], pa: [None, None], pphi: [None, None],
                pmb_t: [None, None], pmb_a: [None, None],
                ptb: [None, None], prb: [None, None], pab: [None, None],
                pmb_tb: [None, None], pmb_rb: [None, None], pmb_ab: [None, None],
            }
        }
    }

    let pol_on = [calc_s, calc_p];

    // ── Phase A: everything expressible per point, in parallel ──
    let outs: Vec<PointOut> =
        (0..total_points)
            .into_par_iter()
            .map(|k| {
                let a = k / num_wavs;
                let w = k % num_wavs;
                let lam = wavls[w];
                let sin_t = sin_theta[a];
                let base = w * nl * 2;
                let ns: Vec<Complex64> = (0..nl)
                    .map(|l| Complex64::new(n_stack_cache[base + l * 2], n_stack_cache[base + l * 2 + 1]))
                    .collect();
                let nsin_fi = ns[0] * Complex64::new(sin_t, 0.0);
                let np_c = needle_n_per_wav[w];
                let tgt_k = target_of(k);
                let wgt_k = weight_of(k);

                let mut o = PointOut::empty();

                // Coherent observables share ONE fields build per polarization.
                if want_p || want_pt || want_pa || want_pphi || want_ptb || want_prb || want_pab || want_disp {
                    for (pi, &on) in pol_on.iter().enumerate() {
                        if !on {
                            continue;
                        }
                        let pol = pi as i32;
                        let fields = build_stack_fields_range(
                            start_idx, idx_end, &ns, thicknesses, rough_vals, rough_types,
                            lam, nsin_fi, pol,
                        );
                        if want_p {
                            o.p[pi] = Some(p_coherent_from_fields(
                                &fields, nsin_fi, lam, pol, np_c, tgt_k, wgt_k,
                                thicknesses, start_idx, idx_end, z_grid,
                            ));
                        }
                        if want_pt {
                            o.pt[pi] = Some(p_coherent_t_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_t_of(k), weight_t_of(k),
                                thicknesses, start_idx, idx_end, z_grid,
                            ));
                        }
                        if want_pa {
                            o.pa[pi] = Some(p_coherent_a_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_a_of(k), weight_a_of(k),
                                thicknesses, start_idx, idx_end, z_grid,
                            ));
                        }
                        if want_pphi {
                            o.pphi[pi] = Some(p_coherent_phi_from_fields(
                                &fields, nsin_fi, lam, pol, np_c, channel,
                                target_phi_of(k), weight_phi_of(k),
                                thicknesses, start_idx, idx_end, z_grid,
                            ));
                        }
                        if want_ptb {
                            o.ptb[pi] = Some(p_coherent_tb_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_tb_of(k), weight_tb_of(k),
                                thicknesses, start_idx, idx_end, z_grid,
                            ));
                        }
                        if want_prb {
                            o.prb[pi] = Some(p_coherent_rb_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_rb_of(k), weight_rb_of(k),
                                thicknesses, start_idx, idx_end, z_grid,
                            ));
                        }
                        if want_pab {
                            o.pab[pi] = Some(p_coherent_ab_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_ab_of(k), weight_ab_of(k),
                                thicknesses, start_idx, idx_end, z_grid,
                            ));
                        }
                        if want_disp {
                            let m = fields.s_left[idx_end];
                            let amp = [m.0, m.1, m.2, m.3][channel];
                            let r2 = amp.norm_sqr();
                            let mut qv = vec![0.0_f64; nz];
                            if r2 > 1e-20 {
                                for (zi, &(j, xi)) in coh_locs.iter().enumerate() {
                                    // Per-channel slope (channel-0 here would
                                    // mix r-motion into t-phase).
                                    let da = needle_slopes4_ddz(
                                        &fields, nsin_fi, j, xi, np_c, pol, lam)[channel];
                                    qv[zi] = (amp.conj() * da).im / r2;
                                }
                            }
                            o.q[pi] = Some(qv);
                        }
                    }
                }

                if let (Some(flags), Some(locs)) = (&inc, &mb_locs) {
                    for (pi, &on) in pol_on.iter().enumerate() {
                        if !on {
                            continue;
                        }
                        if want_pmb {
                            o.pmb[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thicknesses, flags, rough_vals, rough_types,
                                np_c, PmbQuantity::R, tgt_k, wgt_k, locs, pi as i32,
                            ));
                        }
                        if want_pmb_t {
                            o.pmb_t[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thicknesses, flags, rough_vals, rough_types,
                                np_c, PmbQuantity::T,
                                target_t_of(k), weight_t_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_a {
                            o.pmb_a[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thicknesses, flags, rough_vals, rough_types,
                                np_c, PmbQuantity::A,
                                target_a_of(k), weight_a_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_tb {
                            o.pmb_tb[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thicknesses, flags, rough_vals, rough_types,
                                np_c, PmbQuantity::TB,
                                target_tb_of(k), weight_tb_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_rb {
                            o.pmb_rb[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thicknesses, flags, rough_vals, rough_types,
                                np_c, PmbQuantity::RB,
                                target_rb_of(k), weight_rb_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_ab {
                            o.pmb_ab[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thicknesses, flags, rough_vals, rough_types,
                                np_c, PmbQuantity::AB,
                                target_ab_of(k), weight_ab_of(k), locs, pi as i32,
                            ));
                        }
                    }
                }

                o
            })
            .collect::<Vec<_>>();

    // ── Phase B: spectral differentiation chain (crosses wavelengths) ──
    let max_order = max_disp_order(requested);
    // chains[pol][order][k*nz+zi]
    let disp_chain: Vec<Option<Vec<Vec<Vec<f64>>>>> = match max_order {
        None => vec![None, None],
        Some(mo) => {
            let omega: Vec<f64> =
                wavls.iter().map(|&l| 2.0 * std::f64::consts::PI * C_NM_PER_FS / l).collect();
            pol_on
                .iter()
                .enumerate()
                .map(|(pi, &on)| {
                    if !on || !want_disp {
                        return None;
                    }
                    if outs.iter().any(|o| o.q[pi].is_none()) {
                        return None;
                    }
                    let q0: Vec<Vec<f64>> =
                        outs.iter().map(|o| o.q[pi].clone().unwrap()).collect();
                    let mut chain = vec![q0.clone()];
                    for _ in 0..mo {
                        let prev = chain.last().unwrap();
                        chain.push(spectral_gradient_step(prev, &omega, num_wavs, num_angles, nz));
                    }
                    Some(chain)
                })
                .collect()
        }
    };
    let _ = channel;

    // ── Assemble dict ──
    let mut maps: Vec<(String, Vec<f64>)> = Vec::new();

    macro_rules! emit {
        ($name:expr, $field:ident, $pi:expr) => {{
            let name: String = $name;
            let mut flat: Vec<f64> = Vec::with_capacity(total_points * nz);
            for o in &outs {
                match &o.$field[$pi] {
                    Some(v) => flat.extend_from_slice(v),
                    None => {
                        return Err(String::from(
                            "internal error: missing output buffer",
                        ))
                    }
                }
            }
            if gain_shift_phi != 0.0 && name.starts_with("P_PHI") {
                for v in flat.iter_mut() {
                    *v -= gain_shift_phi;
                }
            }
            maps.push((name, flat));
        }};
    }

    let pol_suffix = |pi: usize| if pi == 0 { "s" } else { "p" };
    if want_p {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_{}", pol_suffix(pi)), p, pi);
            }
        }
    }
    if want_pt {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_T_{}", pol_suffix(pi)), pt, pi);
            }
        }
    }
    if want_pa {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_A_{}", pol_suffix(pi)), pa, pi);
            }
        }
    }
    if want_pphi {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_PHI_{}", pol_suffix(pi)), pphi, pi);
            }
        }
    }
    if want_pmb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_{}", pol_suffix(pi)), pmb, pi);
            }
        }
    }
    if want_pmb_t {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_T_{}", pol_suffix(pi)), pmb_t, pi);
            }
        }
    }
    if want_pmb_a {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_A_{}", pol_suffix(pi)), pmb_a, pi);
            }
        }
    }
    if want_ptb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_TB_{}", pol_suffix(pi)), ptb, pi);
            }
        }
    }
    if want_prb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_RB_{}", pol_suffix(pi)), prb, pi);
            }
        }
    }
    if want_pab {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_AB_{}", pol_suffix(pi)), pab, pi);
            }
        }
    }
    if want_pmb_tb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_TB_{}", pol_suffix(pi)), pmb_tb, pi);
            }
        }
    }
    if want_pmb_rb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_RB_{}", pol_suffix(pi)), pmb_rb, pi);
            }
        }
    }
    if want_pmb_ab {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_AB_{}", pol_suffix(pi)), pmb_ab, pi);
            }
        }
    }
    const DISP_KEYS: [&str; 5] = ["dphi", "dgd", "dgdd", "dtod", "dfod"];
    if let Some(mo) = max_order {
        for pi in 0..2 {
            if !pol_on[pi] {
                continue;
            }
            if let Some(chain) = &disp_chain[pi] {
                for order in 0..=mo {
                    let key = format!("{}_{}", DISP_KEYS[order], pol_suffix(pi));
                    let mut flat: Vec<f64> = Vec::with_capacity(total_points * nz);
                    for row in &chain[order] {
                        flat.extend_from_slice(row);
                    }
                    maps.push((key, flat));
                }
            }
        }
    }

    Ok(NeedleSolution { n_points: total_points, n_depths: nz, maps })
}

// ---------------------------------------------------------------------------
// Eigenmode tools (moved verbatim from the PyO3 binding)
// ---------------------------------------------------------------------------

impl Solver {
  /// `(lam, per-layer complex indices)` for one wavelength: explicit
  /// index wins, else nearest grid point, else the single grid point.
  pub fn index_column(
    &self,
    wavelength: Option<f64>,
    wav_index: Option<usize>,
  ) -> Result<(f64, Vec<Complex64>), String> {
    let w = match wav_index {
      Some(i) => {
        if i >= self.wavls.len() {
          return Err(format!("wav_index {i} out of range."));
        }
        i
      }
      None => match wavelength {
        Some(lam) => {
          let mut best = 0;
          for (i, w) in self.wavls.iter().enumerate() {
            if (w - lam).abs() < (self.wavls[best] - lam).abs() {
              best = i;
            }
          }
          best
        }
        None => {
          if self.wavls.len() != 1 {
            return Err(format!(
              "specify wavelength or wav_index (grid has {} wavelengths).",
              self.wavls.len()
            ));
          }
          0
        }
      },
    };
    Ok((self.wavls[w], self.n_cache[w].clone()))
  }

  /// Scan `|1/r(n_eff)|^2` over a complex effective-index box.
  /// Returns `(real_vals, imag_vals, flat imag-major values)`.
  pub fn landscape(
    &self,
    real_range: (f64, f64),
    imag_range: (f64, f64),
    points_real: usize,
    points_imag: usize,
    pol: i32,
    wavelength: Option<f64>,
    wav_index: Option<usize>,
  ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let (lam, col) = self.index_column(wavelength, wav_index)?;
    Ok(scan_box(
      &col,
      &self.thicknesses,
      &self.rough_types,
      &self.rough_vals,
      lam,
      pol,
      real_range.0,
      real_range.1,
      imag_range.0,
      imag_range.1,
      points_real,
      points_imag,
    ))
  }

  /// Coarse minima of a landscape flat (imag-major `n_imag × n_real`).
  pub fn local_minima(
    flat: &[f64],
    n_real: usize,
    n_imag: usize,
    real_vals: &[f64],
    imag_vals: &[f64],
    median_factor: f64,
  ) -> Vec<(f64, f64)> {
    find_minima(flat, n_real, n_imag, real_vals, imag_vals, median_factor)
  }

  /// Nelder-Mead refine of one complex eigenmode guess.
  /// Returns `(n_eff, characteristic_value)`.
  pub fn refine_mode(
    &self,
    guess: Complex64,
    pol: i32,
    wavelength: Option<f64>,
    wav_index: Option<usize>,
    step: f64,
    tol: f64,
    max_iter: usize,
  ) -> Result<(Complex64, f64), String> {
    let (lam, col) = self.index_column(wavelength, wav_index)?;
    let (re, im, val) = nelder_refine(
      &col,
      &self.thicknesses,
      &self.rough_types,
      &self.rough_vals,
      lam,
      pol,
      (guess.re, guess.im),
      step,
      tol,
      max_iter,
    );
    Ok((Complex64::new(re, im), val))
  }

  /// Scan, locate coarse minima, optionally refine each.
  pub fn find_eigenmodes(
    &self,
    real_range: (f64, f64),
    imag_range: (f64, f64),
    points_real: usize,
    points_imag: usize,
    median_factor: f64,
    refine: bool,
    pol: i32,
    wavelength: Option<f64>,
    wav_index: Option<usize>,
  ) -> Result<Vec<Complex64>, String> {
    let (real_vals, imag_vals, flat) =
      self.landscape(real_range, imag_range, points_real, points_imag, pol, wavelength, wav_index)?;
    let seeds = Self::local_minima(&flat, points_real, points_imag, &real_vals, &imag_vals, median_factor);
    if !refine {
      return Ok(seeds.into_iter().map(|(re, im)| Complex64::new(re, im)).collect());
    }
    let mut out = Vec::with_capacity(seeds.len());
    for (re, im) in seeds {
      let (n_eff, _) =
        self.refine_mode(Complex64::new(re, im), pol, wavelength, wav_index, 1e-3, 1e-9, 200)?;
      out.push(n_eff);
    }
    Ok(out)
  }

  /// `|E(z)|` profile for one eigenmode: `(z, E, start, end, layer_n)`.
  pub fn field_profile(
    &self,
    n_eff: Complex64,
    pol: i32,
    wavelength: Option<f64>,
    wav_index: Option<usize>,
    points_per_layer: usize,
  ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Complex64>), String> {
    let (lam, col) = self.index_column(wavelength, wav_index)?;
    field_prof(
      &col,
      &self.thicknesses,
      &self.rough_types,
      &self.rough_vals,
      lam,
      n_eff,
      pol,
      points_per_layer,
    )
  }

} // impl Solver (eigen drivers)

/// Landscape scan over the complex effective-index box.
/// Returns `(real_vals, imag_vals, flat imag-major values)`.
pub fn scan_box(
    n_stack: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    pol: i32,
    real_min: f64,
    real_max: f64,
    imag_min: f64,
    imag_max: f64,
    points_real: usize,
    points_imag: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n_slice, d_slice, rt_slice, rv_slice) = (n_stack, thicknesses, rough_types, rough_vals);

    // Reciprocals computed ONCE per wavelength and shared read-only across all
    // grid points (and all rayon threads). Previously this Vec was allocated
    // inside every char_func call — thousands of heap allocations per scan.
    let inv_n: Vec<Complex64> = n_slice.iter().map(|n| n.recip()).collect();

    let real_vals: Vec<f64> = (0..points_real)
        .map(|i| real_min + (i as f64) * (real_max - real_min) / ((points_real - 1) as f64))
        .collect();
    let imag_vals: Vec<f64> = (0..points_imag)
        .map(|i| imag_min + (i as f64) * (imag_max - imag_min) / ((points_imag - 1) as f64))
        .collect();

    let landscape: Vec<f64> = (0..points_imag * points_real)
            .into_par_iter()
            .map(|idx| {
                let i = idx / points_real;
                let j = idx % points_real;
                let nr = real_vals[j];
                let ni = imag_vals[i];
                let n_eff = Complex64::new(nr, ni);
                char_func(n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, n_eff, pol)
            })
            .collect();
    (real_vals, imag_vals, landscape)
}

/// Coarse local minima below `median_factor * median(values)`.
/// `flat` is imag-major with shape `(n_imag, n_real)`; the first/last
/// real columns are skipped (matches the reference sweep).
pub fn find_minima(
    flat: &[f64],
    n_real: usize,
    n_imag: usize,
    real_vals: &[f64],
    imag_vals: &[f64],
    median_factor: f64,
) -> Vec<(f64, f64)> {
    let at = |i: usize, j: usize| flat[i * n_real + j];
    let mut land_vec: Vec<f64> = flat.to_vec();
    let land = land_vec.as_slice();

    // True median of the landscape (the previous code averaged, which the
    // `median_factor` name and the Python reference (`np.median`) do not).
    // Sentinel 1e30 cells sort to the top and so don't perturb the median,
    // whereas they badly skewed the mean.
    let mut sorted: Vec<f64> = land_vec.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    let median = if len == 0 {
        0.0
    } else if len % 2 == 1 {
        sorted[len / 2]
    } else {
        0.5 * (sorted[len / 2 - 1] + sorted[len / 2])
    };
    let threshold = median * median_factor;

    let mut candidates = Vec::new();
    if n_real < 2 {
        return candidates;
    }
    for i in 0..n_imag {
        // Skip the first/last real columns, matching the reference
        // (`for j in range(1, len(Nr) - 1)`); all imag rows are scanned so
        // lossless modes on the Im=0 edge are still detected.
        for j in 1..n_real - 1 {
            let val = at(i, j);
            if val >= threshold {
                continue;
            }
            let i0 = i.saturating_sub(1);
            let i1 = (i + 1).min(n_imag - 1);
            let j0 = j.saturating_sub(1);
            let j1 = (j + 1).min(n_real - 1);
            let mut is_min = true;
            'neighbors: for ii in i0..=i1 {
                for jj in j0..=j1 {
                    if ii == i && jj == j {
                        continue;
                    }
                    if at(ii, jj) <= val {
                        is_min = false;
                        break 'neighbors;
                    }
                }
            }
            if is_min {
                candidates.push((real_vals[j], imag_vals[i]));
            }
        }
    }
    candidates
}

/// Nelder-Mead refine of one complex eigenmode guess.
/// Returns `(re, im, characteristic_value)`.
pub fn nelder_refine(
    n_stack: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    pol: i32,
    x0: (f64, f64),
    step: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64, f64) {
    let (n_slice, d_slice, rt_slice, rv_slice) = (n_stack, thicknesses, rough_types, rough_vals);

    // Reciprocals computed once and reused across every simplex evaluation.
    let inv_n: Vec<Complex64> = n_slice.iter().map(|n| n.recip()).collect();

    let mut simplex = vec![
        [x0.0, x0.1],
        [x0.0 + step, x0.1],
        [x0.0, x0.1 + step * 0.1],
    ];
    let mut values: Vec<f64> = simplex
        .iter()
        .map(|x| char_func_xy(x, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol))
        .collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;
    let mut iter = 0;

    loop {
        let mut indices: Vec<usize> = (0..3).collect();
        indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
        let (best, good, worst) = (indices[0], indices[1], indices[2]);

        let centroid = [
            (simplex[best][0] + simplex[good][0]) / 2.0,
            (simplex[best][1] + simplex[good][1]) / 2.0,
        ];
        let reflected = [
            centroid[0] + alpha * (centroid[0] - simplex[worst][0]),
            centroid[1] + alpha * (centroid[1] - simplex[worst][1]),
        ];
        let f_ref = char_func_xy(&reflected, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);

        if f_ref < values[best] {
            let expanded = [
                centroid[0] + gamma * (reflected[0] - centroid[0]),
                centroid[1] + gamma * (reflected[1] - centroid[1]),
            ];
            let f_exp = char_func_xy(&expanded, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);
            if f_exp < f_ref {
                simplex[worst] = expanded;
                values[worst] = f_exp;
            } else {
                simplex[worst] = reflected;
                values[worst] = f_ref;
            }
        } else if f_ref < values[good] {
            simplex[worst] = reflected;
            values[worst] = f_ref;
        } else {
            let contracted = [
                centroid[0] + rho * (simplex[worst][0] - centroid[0]),
                centroid[1] + rho * (simplex[worst][1] - centroid[1]),
            ];
            let f_con = char_func_xy(&contracted, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);
            if f_con < values[worst] {
                simplex[worst] = contracted;
                values[worst] = f_con;
            } else {
                for i in 0..3 {
                    if i != best {
                        simplex[i][0] = simplex[best][0] + sigma * (simplex[i][0] - simplex[best][0]);
                        simplex[i][1] = simplex[best][1] + sigma * (simplex[i][1] - simplex[best][1]);
                        values[i] = char_func_xy(&simplex[i], n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);
                    }
                }
            }
        }

        iter += 1;
        let size = ((simplex[0][0] - simplex[1][0]).powi(2) + (simplex[0][1] - simplex[1][1]).powi(2)).sqrt()
                + ((simplex[1][0] - simplex[2][0]).powi(2) + (simplex[1][1] - simplex[2][1]).powi(2)).sqrt()
                + ((simplex[2][0] - simplex[0][0]).powi(2) + (simplex[2][1] - simplex[0][1]).powi(2)).sqrt();
        if size < tol || iter >= max_iter {
            break;
        }
    }

    let best_idx = (0..3).min_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap()).unwrap();
    (simplex[best_idx][0], simplex[best_idx][1], values[best_idx])
}

/// Per-layer data for the field-profile sweep.
struct LayerData {
  n: Complex64,
  cos: Complex64,
  thickness: f64,
}

/// `|E(z)|` profile through the stack for one eigenmode.
/// Returns `(z, E, layer_start, layer_end, layer_index)`.
pub fn field_prof(
    n_stack: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    n_eff: Complex64,
    pol: i32,
    points_per_layer: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Complex64>), String>
{
    let (n_slice, d_slice, rt_slice, rv_slice) = (n_stack, thicknesses, rough_types, rough_vals);

    let n_layers = n_slice.len();
    if n_layers < 2 {
        return Err("field profile needs at least 2 layers".to_string());
    }

    let two_pi_lam = 2.0 * PI / lam;

    // Precompute layer data: n, cosθ, thickness
    let mut layers: Vec<LayerData> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let n = n_slice[i];
        let r0 = n_eff * n.recip();
        let v = Complex64::new(1.0, 0.0) - r0 * r0;
        let mut cos = v.sqrt();
        if cos.im < 0.0 {
            cos = -cos;
        }
        layers.push(LayerData {
            n,
            cos,
            thickness: d_slice[i],
        });
    }

    // Helper for Fresnel + roughness at an interface (i -> i+1)
    let interface_props = |i: usize| -> (Complex64, Complex64, Complex64, Complex64) {
        let n_curr = layers[i].n;
        let cos_curr = layers[i].cos;
        let y_curr = if pol == 0 {
            n_curr * cos_curr
        } else {
            let c = if cos_curr.norm() < 1e-12 { Complex64::new(1e-12, 0.0) } else { cos_curr };
            n_curr / c
        };
        let n_next = layers[i+1].n;
        let cos_next = layers[i+1].cos;
        let y_next = if pol == 0 {
            n_next * cos_next
        } else {
            let c = if cos_next.norm() < 1e-12 { Complex64::new(1e-12, 0.0) } else { cos_next };
            n_next / c
        };

        let den = y_curr + y_next;
        let den_safe = if den.norm() < 1e-100 { Complex64::new(1e-100, 1e-100) } else { den };
        let inv_den = den_safe.recip();
        let r12 = (y_curr - y_next) * inv_den;
        let t12 = y_curr * 2.0 * inv_den;
        let t21 = y_next * 2.0 * inv_den;
        let r21 = -r12;

        let sigma = rv_slice[i+1];
        let rtype = rt_slice[i+1];
        if rtype != 0 && sigma > 0.0 {
            let kz1 = two_pi_lam * n_curr * cos_curr;
            let kz2 = two_pi_lam * n_next * cos_next;
            if rtype == 5 {
                let f = (-2.0 * kz1 * kz2 * sigma * sigma).exp();
                (r12 * f, r21 * f, t12 * f, t21 * f)
            } else {
                let al = w_function_inner(2.0 * kz1 * sigma, rtype);
                let be = w_function_inner(2.0 * kz2 * sigma, rtype);
                let ga = w_function_inner((kz1 - kz2) * sigma, rtype);
                (r12 * al, r21 * be, t12 * ga, t21 * ga)
            }
        } else {
            (r12, r21, t12, t21)
        }
    };

    // Propagation phase through a layer (i)
    let prop_phase = |i: usize| -> Complex64 {
        let d = layers[i].thickness;
        if d <= 1e-12 {
            return Complex64::new(1.0, 0.0);
        }
        let mut beta = two_pi_lam * d * layers[i].n * layers[i].cos;
        if beta.im < 0.0 {
            beta = Complex64::new(beta.re, -beta.im);
        }
        (Complex64::new(0.0, 1.0) * beta).exp()
    };

    // ---------- Build left and right S‑matrices ----------
    // S_left[i] = S‑matrix from ambient up to the left side of layer i (i from 1 to n_layers-1)
    let mut s_left: Vec<(Complex64, Complex64, Complex64, Complex64)> = Vec::with_capacity(n_layers);
    s_left.push((Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))); // identity before ambient

    for i in 0..n_layers-1 {
        let mut sg = s_left.last().unwrap().clone();
        if i > 0 && layers[i].thickness > 1e-12 {
            let phi = prop_phase(i);
            sg = redheffer_product_complex_field_inner(
                sg.0, sg.1, sg.2, sg.3,
                Complex64::new(0.0, 0.0), phi, phi, Complex64::new(0.0, 0.0),
            );
        }
        let iface = interface_props(i);
        sg = redheffer_product_complex_field_inner(sg.0, sg.1, sg.2, sg.3, iface.0, iface.1, iface.2, iface.3);
        s_left.push(sg);
    }

    // S_right[i] = S‑matrix from substrate up to the right side of layer i (i from n_layers-2 down to 0)
    let mut s_right: Vec<Option<(Complex64, Complex64, Complex64, Complex64)>> = vec![None; n_layers];
    s_right[n_layers-1] = Some((Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)));

    for i in (0..n_layers-1).rev() {
        let mut sg = s_right[i+1].unwrap();
        if i+1 < n_layers-1 && layers[i+1].thickness > 1e-12 {
            let phi = prop_phase(i+1);
            sg = redheffer_product_complex_field_inner(
                Complex64::new(0.0, 0.0), phi, phi, Complex64::new(0.0, 0.0),
                sg.0, sg.1, sg.2, sg.3,
            );
        }
        let iface = interface_props(i);
        sg = redheffer_product_complex_field_inner(iface.0, iface.1, iface.2, iface.3, sg.0, sg.1, sg.2, sg.3);
        s_right[i] = Some(sg);
    }

    // ---------- Compute field inside each layer ----------
    let mut z_pos = Vec::new();
    let mut e_mag = Vec::new();
    let mut layer_start = Vec::new();
    let mut layer_end = Vec::new();
    let mut layer_n = Vec::new();

    let mut z_cursor = 0.0;

    for i in 1..n_layers-1 {
        let d = layers[i].thickness;
        if d <= 1e-12 {
            continue;
        }
        let sl = &s_left[i];
        let sr = s_right[i].as_ref().unwrap();
        let denom = Complex64::new(1.0, 0.0) - sl.3 * sr.0;
        let denom_safe = if denom.norm() < 1e-100 {
            Complex64::new(1e-100, 1e-100)
        } else {
            denom
        };
        let inv_denom = denom_safe.recip();
        let e_plus = sl.2 * inv_denom;
        let e_minus = sr.0 * e_plus;
        let mut beta = two_pi_lam * d * layers[i].n * layers[i].cos;
        if beta.im < 0.0 {
            beta = Complex64::new(beta.re, -beta.im);
        }

        let step = d / (points_per_layer as f64);
        for k in 0..=points_per_layer {
            let zz = k as f64 * step;
            let xi = zz / d;
            let e_z = e_plus * (Complex64::new(0.0, 1.0) * beta * xi).exp()
                    + e_minus * (-Complex64::new(0.0, 1.0) * beta * xi).exp();
            z_pos.push(z_cursor + zz);
            e_mag.push(e_z.norm());
        }
        layer_start.push(z_cursor);
        layer_end.push(z_cursor + d);
        layer_n.push(layers[i].n);
        z_cursor += d;
    }

    // Normalise E‑field to max = 1
    let max_e = e_mag.iter().copied().fold(0.0, f64::max);
    if max_e > 0.0 {
        for val in &mut e_mag {
            *val /= max_e;
        }
    }

    Ok((z_pos, e_mag, layer_start, layer_end, layer_n))
}

// ---------------------------------------------------------------------------
// Tests (standalone: no Python)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::super::core_engine::{REQ_R_AVG, REQ_RS, REQ_T_AVG};
  use super::*;

  /// (n_layers, n_wavs) row-major complex indices.
  fn ambient_substrate(n_amb: f64, n_sub: f64, nw: usize) -> Vec<Complex64> {
    let mut out = Vec::with_capacity(2 * nw);
    for _ in 0..nw {
      out.push(Complex64::new(n_amb, 0.0));
    }
    for _ in 0..nw {
      out.push(Complex64::new(n_sub, 0.0));
    }
    out
  }

  #[test]
  fn bare_interface_fresnel_hex() {
    // Air (n=1) → glass (n=1.5): Rs(0°) = ((1-1.5)/(1+1.5))² = 0.04.
    let wl = vec![500.0, 600.0];
    let s = Solver::new(
      &wl,
      &[0.0],
      &ambient_substrate(1.0, 1.5, 2),
      2,
      &[0.0, 0.0],
      &[0, 0],
      &[0, 0],
      &[0.0, 0.0],
      2,
    )
    .unwrap();
    let sol = s.solve(REQ_RS | REQ_R_AVG).unwrap();
    let rs = sol.f64maps.iter().find(|(k, _)| k == "Rs").unwrap().1.clone();
    for v in &rs {
      assert!((v - 0.04).abs() < 1e-15, "Rs must be 0.04, got {v}");
    }
    let ra = sol.f64maps.iter().find(|(k, _)| k == "R_avg").unwrap().1.clone();
    for v in &ra {
      assert!((v - 0.04).abs() < 1e-15);
    }
  }

  #[test]
  fn view_masks_match_python() {
    use super::super::core_engine::{
      REQ_A_AVG, REQ_A_P, REQ_A_S, REQ_DELTA_R, REQ_DOP_R, REQ_PSI_R, REQ_RP, REQ_RS,
      REQ_R_AVG, REQ_S0_R, REQ_S1_R, REQ_S2_R, REQ_S3_R, REQ_TS,
    };
    assert_eq!(super::rt_request("s").unwrap(), REQ_R_AVG | REQ_T_AVG | REQ_RS | REQ_TS);
    assert_eq!(
      super::rt_request("u").unwrap(),
      REQ_R_AVG | REQ_T_AVG | REQ_RS | REQ_TS | REQ_RP | super::super::core_engine::REQ_TP
    );
    assert!(super::rt_request("x").is_err());
    assert_eq!(
      super::ellipsometry_request(false),
      REQ_PSI_R | REQ_DELTA_R | REQ_DOP_R | REQ_RS | REQ_RP | REQ_R_AVG
    );
    assert_eq!(super::absorption_request(), REQ_A_S | REQ_A_P | REQ_A_AVG);
    assert_eq!(
      super::stokes_request(true, false).unwrap(),
      REQ_S0_R | REQ_S1_R | REQ_S2_R | REQ_S3_R
    );
    assert!(super::stokes_request(false, false).is_err());
    assert!(super::dispersion_request(false, false, false, false).is_err());
  }

  #[test]
  fn energy_conservation_values() {
    let e = super::energy_conservation(&[0.04], &[0.04], &[0.96], &[0.96]).unwrap();
    assert!((e[0] - 0.0).abs() < 1e-15);
    let e = super::energy_conservation(&[0.5], &[0.4], &[0.3], &[0.4]).unwrap();
    assert!((e[0] - 0.2).abs() < 1e-15);
    assert!(super::energy_conservation(&[0.5], &[0.4], &[0.3], &[]).is_err());
  }

  #[test]
  fn from_raw_broadcast_matches_explicit() {
    let wl = vec![500.0, 600.0];
    let per_layer = vec![Complex64::new(1.0, 0.0), Complex64::new(1.5, 0.0)];
    let a = super::Solver::from_raw(&wl, &[0.0], false, &per_layer, 2, None, None, None, None, 2)
      .unwrap();
    let full = vec![
      Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0),
      Complex64::new(1.5, 0.0), Complex64::new(1.5, 0.0),
    ];
    let b = super::Solver::new(&wl, &[0.0], &full, 2, &[0.0, 0.0], &[0, 0], &[0, 0], &[0.0, 0.0], 2)
      .unwrap();
    use super::super::core_engine::REQ_RS;
    let ra = a.solve(REQ_RS).unwrap();
    let rb = b.solve(REQ_RS).unwrap();
    let fa = ra.f64maps.iter().find(|(k, _)| k == "Rs").unwrap().1.clone();
    let fb = rb.f64maps.iter().find(|(k, _)| k == "Rs").unwrap().1.clone();
    assert_eq!(fa.len(), fb.len());
    for (x, y) in fa.iter().zip(fb.iter()) {
      assert_eq!(x.to_bits(), y.to_bits());
    }
  }

  #[test]
  fn needle_gradient_standalone() {
    use super::super::needle_engine::NREQ_P;
    let wl = vec![500.0, 600.0];
    let idx = vec![
      Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0),
      Complex64::new(2.0, 0.0), Complex64::new(2.0, 0.0),
      Complex64::new(1.5, 0.0), Complex64::new(1.5, 0.0),
    ];
    let s = super::Solver::new(
      &wl, &[0.0], &idx, 3, &[0.0, 100.0, 0.0], &[0, 0, 0], &[0, 0, 0], &[0.0, 0.0, 0.0], 2,
    )
    .unwrap();
    let sol = s
      .needle_gradient(
        &[Complex64::new(2.1, 0.0), Complex64::new(2.1, 0.0)],
        &[10.0, 50.0, 90.0],
        NREQ_P,
        None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None,
        0, Some(2), 0, true, true, None, 0.0,
      )
      .unwrap();
    assert_eq!(sol.n_depths, 3);
    assert_eq!(sol.n_points, 2);
    let ps = sol.maps.iter().find(|(k, _)| k == "P_s").expect("P_s").1.clone();
    assert_eq!(ps.len(), 6);
    assert!(ps.iter().all(|v| v.is_finite()));
  }

  #[test]
  fn eigen_tools_standalone() {
    // 3-layer waveguide-ish stack at one wavelength.
    let col = vec![
      Complex64::new(1.0, 0.0),
      Complex64::new(2.0, 0.0),
      Complex64::new(1.5, 0.0),
    ];
    let th = vec![0.0, 500.0, 0.0];
    let rt = vec![0, 0, 0];
    let rv = vec![0.0, 0.0, 0.0];
    let (re, im, flat) = super::scan_box(&col, &th, &rt, &rv, 600.0, 0, 1.5, 2.0, 0.0, 0.05, 4, 3);
    assert_eq!(re.len(), 4);
    assert_eq!(im.len(), 3);
    assert_eq!(flat.len(), 12);
    assert!(flat.iter().all(|v| v.is_finite()));
    // Synthetic valley: minimum at (re[1], im[1]).
    let synth = vec![5.0, 5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 5.0];
    let rr = vec![1.0, 2.0, 3.0];
    let ii = vec![0.0, 0.1, 0.2];
    let mins = super::find_minima(&synth, 3, 3, &rr, &ii, 0.5);
    assert_eq!(mins, vec![(2.0, 0.1)]);
    let (r, i, v) = super::nelder_refine(&col, &th, &rt, &rv, 600.0, 0, (1.7, 0.01), 1e-3, 1e-9, 50);
    assert!(r.is_finite() && i.is_finite() && v.is_finite());
    let prof = super::field_prof(&col, &th, &rt, &rv, 600.0, Complex64::new(1.7, 0.01), 0, 4);
    assert!(prof.is_ok());
    let (z, e, _, _, _) = prof.unwrap();
    assert_eq!(z.len(), e.len());
    assert!(!z.is_empty());
  }

  #[test]
  fn solve_arrays_contract() {
    use super::super::core_engine::REQ_RS;
    let wl = vec![500.0, 600.0];
    let idx = vec![
      Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0),
      Complex64::new(1.5, 0.0), Complex64::new(1.5, 0.0),
    ];
    // Grid mismatch refused.
    assert!(super::solve_arrays(&idx, &[0.0, 0.0], &[false, false], &[0, 0], &[0.0, 0.0],
      &[500.0], &[0.0], false, REQ_RS, 2).is_err());
    // Clean half-spaces: no warnings, Rs = 0.04.
    let (sol, warns) = super::solve_arrays(&idx, &[0.0, 0.0], &[false, false], &[0, 0], &[0.0, 0.0],
      &wl, &[0.0], false, REQ_RS, 2).unwrap();
    assert!(warns.is_empty());
    let rs = sol.f64maps.iter().find(|(k, _)| k == "Rs").unwrap().1.clone();
    assert!((rs[0] - 0.04).abs() < 1e-15);
    // Nonzero half-spaces: warned, not refused.
    let (_, warns) = super::solve_arrays(&idx, &[5.0, 0.0], &[false, false], &[0, 0], &[0.0, 0.0],
      &wl, &[0.0], false, REQ_RS, 2).unwrap();
    assert_eq!(warns.len(), 1);
  }

  #[test]
  fn validation_refuses() {
    let wl = vec![500.0];
    let idx = vec![Complex64::new(1.0, 0.0)];
    assert!(Solver::new(&[], &[0.0], &idx, 1, &[0.0], &[0], &[0], &[0.0], 2).is_err());
    assert!(Solver::new(&wl, &[], &idx, 1, &[0.0], &[0], &[0], &[0.0], 2).is_err());
    assert!(Solver::new(&wl, &[0.0], &idx, 1, &[0.0], &[0], &[0], &[0.0], 2).is_err());
    assert!(Solver::new(&wl, &[0.0], &idx, 2, &[0.0], &[0], &[0], &[0.0], 5).is_err());
    let s = Solver::new(&wl, &[0.0], &[Complex64::new(1.0, 0.0), Complex64::new(1.5, 0.0)], 2,
      &[0.0, 0.0], &[0, 0], &[0, 0], &[0.0, 0.0], 2).unwrap();
    assert!(s.solve(0).is_err());
  }
}
