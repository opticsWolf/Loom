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
  thicknesses: Vec<f64>,
  incoherent_flags: Vec<i32>,
  rough_types: Vec<i32>,
  rough_vals: Vec<f64>,
  coherence_mode: i32,
  n_layers: usize,
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
      thicknesses: thicknesses.to_vec(),
      incoherent_flags: incoherent_flags.to_vec(),
      rough_types: rough_types.to_vec(),
      rough_vals: rough_vals.to_vec(),
      coherence_mode,
      n_layers,
    })
  }

  pub fn n_angles(&self) -> usize {
    self.sin_theta.len()
  }

  pub fn n_wavs(&self) -> usize {
    self.wavls.len()
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
