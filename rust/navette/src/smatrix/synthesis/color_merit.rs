//! synthesis::color_merit — colorimetric demand kernel (Option B plan D2).
//!
//! P1 serves `Lab | XyY` × `DeltaE2000 | DeltaE76 | Channels`. The P2
//! variants (`LCh | Oklab | Y`) exist in the enums so the schema is stable,
//! but evaluate through them is refused here until 0.4.27.
//!
//! All fns are `pub(crate)`: the whole demand lifecycle (compile → merit →
//! needle fold) lives in-crate; the PyO3 surface binds those arms, never
//! this kernel directly (exposure lint stays green by construction).

use serde::{Deserialize, Serialize};

use crate::color::common::{xyz_to_lab, REF_WHITE_D65};
use crate::color::func_02::{lab_to_lch, lch_to_lab};
use crate::color::func_04::xyz_to_oklab;
use crate::color::func_08::adapt;
use crate::color::func_01::xyz_to_xyy;
use crate::color::func_09::delta_e_76_single;
use crate::color::func_16::delta_e_2000_single;

/// P1: Lab | XyY. P2 adds LCh | Oklab | Y (scalar) — refused in `new`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorQuantity {
  Lab,
  XyY,
  LCh,
  Oklab,
  Y,
}

/// P1: all three live; XyY×ΔE and Oklab×ΔE are refused (see matrix).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorDistance {
  DeltaE2000,
  DeltaE76,
  Channels,
}

/// Triple reference — or scalar for `Y` (luminance-only, P2).
/// Untagged (`[62, 18, -34]` vs `12.5`); validated against the quantity
/// in `new`. Malformed shapes never reach this type: the compile arm
/// catches them via `ReferenceJson::Other` with a named refusal.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ColorReference {
  Triple([f64; 3]),
  Scalar(f64),
}

/// One compiled color demand. Tables ride NATIVE grids (resampled per eval
/// into scratch — no memo, keeps future holders `Send+Sync`). `white` is
/// the demand illuminant's own white, integrated once here at construction
/// (never adapt foreign-white numbers).
#[derive(Clone, Debug)]
pub struct ColorDemand {
  /// Index into `MeritSpec::keys` (set at compile; groups the demand for
  /// missing-penalty + residual ordering, exactly like pointwise targets).
  pub key_idx: u32,
  pub cmf: Vec<[f64; 3]>,
  pub cmf_wl: Vec<f64>,
  pub illuminant: Vec<f64>,
  pub illum_wl: Vec<f64>,
  pub white: [f64; 3],
  pub quantity: ColorQuantity,
  pub reference: ColorReference,
  pub distance: ColorDistance,
  /// Per-channel tolerances (Channels mode). Unit default (documented).
  pub tol: [f64; 3],
  pub weight: f64,
}

fn check_grid(name: &str, wl: &[f64], cols: usize) -> Result<(), String> {
  if wl.is_empty() {
    return Err(format!("color: {name} wavelength grid is empty."));
  }
  if wl.windows(2).any(|w| !(w[0] < w[1])) {
    return Err(format!("color: {name} wavelengths not strictly increasing."));
  }
  if wl.iter().any(|v| !v.is_finite()) {
    return Err(format!("color: {name} has non-finite wavelengths."));
  }
  let _ = cols;
  Ok(())
}

impl ColorDemand {
  pub(crate) fn new(
    key_idx: u32,
    cmf: Vec<[f64; 3]>,
    cmf_wl: Vec<f64>,
    illuminant: Vec<f64>,
    illum_wl: Vec<f64>,
    quantity: ColorQuantity,
    reference: ColorReference,
    distance: ColorDistance,
    weight: f64,
  ) -> Result<Self, String> {
    // Reference shape gates on quantity (both directions).
    let want_scalar = quantity == ColorQuantity::Y;
    let got_scalar = matches!(reference, ColorReference::Scalar(_));
    if want_scalar != got_scalar {
      return Err(if want_scalar {
        "color: quantity 'Y' needs a scalar reference.".to_string()
      } else {
        "color: scalar reference needs quantity 'Y'.".to_string()
      });
    }
    // Compat matrix (§D2): ΔE is Lab-space (XyY takes Channels);
    // Oklab-ΔE would double-count (equal-tol Channels is mathematically
    // identical to unweighted Euclidean); Y is scalar (Channels only).
    match (quantity, distance) {
      (ColorQuantity::XyY, ColorDistance::DeltaE2000) | (ColorQuantity::XyY, ColorDistance::DeltaE76) => {
        return Err(format!(
          "color: quantity 'XyY' with distance '{distance:?}' refused (DeltaE is Lab-space; use Channels)."
        ))
      }
      (ColorQuantity::Oklab, ColorDistance::DeltaE2000) | (ColorQuantity::Oklab, ColorDistance::DeltaE76) => {
        return Err(format!(
          "color: quantity 'Oklab' with distance '{distance:?}' refused (use equal-tol Channels)."
        ))
      }
      (ColorQuantity::Y, ColorDistance::DeltaE2000) | (ColorQuantity::Y, ColorDistance::DeltaE76) => {
        return Err(format!(
          "color: quantity 'Y' with distance '{distance:?}' refused (scalar demand takes Channels)."
        ))
      }
      _ => {}
    }
    check_grid("CMF", &cmf_wl, cmf.len())?;
    check_grid("illuminant", &illum_wl, illuminant.len())?;
    if cmf.len() != cmf_wl.len() || illuminant.len() != illum_wl.len() {
      return Err("color: table values length != grid length.".to_string());
    }
    if cmf.iter().any(|c| c.iter().any(|v| !v.is_finite()))
      || illuminant.iter().any(|v| !v.is_finite())
    {
      return Err("color: non-finite CMF/illuminant table value.".to_string());
    }
    if !weight.is_finite() || weight < 0.0 {
      return Err(format!("color: weight must be finite and >= 0, got {weight}."));
    }
    if let ColorReference::Triple(t) = &reference {
      if t.iter().any(|v| !v.is_finite()) {
        return Err("color: non-finite reference triple.".to_string());
      }
    }
    // Own-white rule: integrate the illuminant as a perfect diffuser on
    // its NATIVE grid (no resample involved — both tables native here).
    let ones = vec![1.0; illuminant.len()];
    let white = xyz_of_spectrum(&ones, &illum_wl, &cmf, &cmf_wl, &illuminant, &illum_wl)?;
    if white.iter().any(|v| !v.is_finite()) {
      return Err("color: degenerate illuminant (non-finite white point).".to_string());
    }
    Ok(ColorDemand {
      key_idx,
      cmf,
      cmf_wl,
      illuminant,
      illum_wl,
      white,
      quantity,
      reference,
      distance,
      tol: [1.0, 1.0, 1.0],
      weight,
    })
  }
}

/// Linear interpolation of a native table at `x`; `None` outside coverage.
fn resample(tbl_wl: &[f64], tbl: &[f64], x: f64) -> Option<f64> {
  if x < tbl_wl[0] || x > tbl_wl[tbl_wl.len() - 1] {
    return None;
  }
  let mut lo = 0usize;
  let mut hi = tbl_wl.len() - 1;
  while hi - lo > 1 {
    let mid = (lo + hi) / 2;
    if tbl_wl[mid] <= x {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  let t = (x - tbl_wl[lo]) / (tbl_wl[hi] - tbl_wl[lo]);
  Some(tbl[lo] + t * (tbl[hi] - tbl[lo]))
}

/// Integration workspace: resampled tables on the covered sim points +
/// the forward-difference Δλ weights. On a uniform fully-covered grid the
/// weights are a constant interval and the sums below reproduce
/// `func_13` summation op-for-op (R1 pins this).
struct XyzWorkspace {
  /// Covered sim indices (ascending) parallel to dw/e_res/cmf_res.
  idx: Vec<usize>,
  xyz: [f64; 3],
  k: f64,
  dw: Vec<f64>,
  e_res: Vec<f64>,
  cmf_res: Vec<[f64; 3]>,
}

fn xyz_workspace(
  sim_row: &[f64],
  sim_wl: &[f64],
  cmf: &[[f64; 3]],
  cmf_wl: &[f64],
  illum: &[f64],
  illum_wl: &[f64],
) -> Result<XyzWorkspace, String> {
  if sim_row.len() != sim_wl.len() || sim_row.is_empty() {
    return Err("color: spectrum length != grid length (or empty).".to_string());
  }
  if sim_row.iter().any(|v| !v.is_finite()) {
    return Err("color: non-finite spectrum value.".to_string());
  }
  let cmf_c: Vec<Vec<f64>> = (0..3).map(|c| cmf.iter().map(|t| t[c]).collect()).collect();
  let mut idx = Vec::new();
  let mut e_res = Vec::new();
  let mut cmf_res = Vec::new();
  for (i, &w) in sim_wl.iter().enumerate() {
    let e = resample(illum_wl, illum, w);
    let x = resample(cmf_wl, &cmf_c[0], w);
    let y = resample(cmf_wl, &cmf_c[1], w);
    let z = resample(cmf_wl, &cmf_c[2], w);
    if let (Some(e), Some(x), Some(y), Some(z)) = (e, x, y, z) {
      idx.push(i);
      e_res.push(e);
      cmf_res.push([x, y, z]);
    }
  }
  if idx.is_empty() {
    return Err("color: no overlap between sim grid and CMF/illuminant tables.".to_string());
  }
  if idx.len() < 2 {
    return Err("color: table overlap is a single point (need >= 2).".to_string());
  }
  // Forward-difference Δλ on the covered subset (uniform grid → constant).
  let wl: Vec<f64> = idx.iter().map(|&i| sim_wl[i]).collect();
  let m = wl.len();
  let mut dw = vec![0.0; m];
  for i in 0..m - 1 {
    dw[i] = wl[i + 1] - wl[i];
  }
  dw[m - 1] = wl[m - 1] - wl[m - 2];
  if dw.iter().any(|&d| !(d > 0.0)) {
    return Err("color: sim grid not strictly increasing over the overlap.".to_string());
  }
  let denom: f64 = (0..m).map(|i| e_res[i] * cmf_res[i][1] * dw[i]).sum();
  if !(denom > 1e-300) {
    return Err("color: degenerate illuminant overlap (k normalization failed).".to_string());
  }
  let k = 1.0 / denom;
  let mut xyz = [0.0; 3];
  for i in 0..m {
    let r = sim_row[idx[i]];
    let w = r * e_res[i] * k * dw[i];
    xyz[0] += w * cmf_res[i][0];
    xyz[1] += w * cmf_res[i][1];
    xyz[2] += w * cmf_res[i][2];
  }
  Ok(XyzWorkspace { idx, xyz, k, dw, e_res, cmf_res })
}

/// `XYZ = Σ R·E·cmf·k·Δλ`, `k = 1/ΣE·ȳ·Δλ` (perfect diffuser ⇒ Y ≡ 1).
pub(crate) fn xyz_of_spectrum(
  sim_row: &[f64],
  sim_wl: &[f64],
  cmf: &[[f64; 3]],
  cmf_wl: &[f64],
  illum: &[f64],
  illum_wl: &[f64],
) -> Result<[f64; 3], String> {
  xyz_workspace(sim_row, sim_wl, cmf, cmf_wl, illum, illum_wl).map(|w| w.xyz)
}

/// XYZ → quantity triple. Lab white = the demand illuminant's own white.
pub(crate) fn color_of_xyz(
  xyz: &[f64; 3],
  white: &[f64; 3],
  quantity: ColorQuantity,
) -> Result<[f64; 3], String> {
  if xyz.iter().any(|v| !v.is_finite()) || white.iter().any(|v| !v.is_finite()) {
    return Err("color: non-finite XYZ/white in quantity map.".to_string());
  }
  match quantity {
    ColorQuantity::Lab => {
      let mut out = [[0.0; 3]];
      xyz_to_lab(&[*xyz], white, &mut out);
      Ok(out[0])
    }
    ColorQuantity::XyY => {
      let mut out = [[0.0; 3]];
      xyz_to_xyy(&[*xyz], &mut out);
      Ok(out[0])
    }
    ColorQuantity::LCh => {
      let mut lab = [[0.0; 3]];
      xyz_to_lab(&[*xyz], white, &mut lab);
      let mut out = [[0.0; 3]];
      lab_to_lch(&lab, &mut out);
      Ok(out[0])
    }
    ColorQuantity::Oklab => {
      // Oklab is D65-defined: Bradford-adapt non-D65 XYZ first via the
      // same `adapt` the bindings (and func_13) use — no clipping, so
      // the FD gradient keeps the smooth map.
      let mut adapted = [[0.0; 3]];
      adapt(&[*xyz], white, &REF_WHITE_D65, false, &mut adapted);
      let mut out = [[0.0; 3]];
      xyz_to_oklab(&adapted, &mut out);
      Ok(out[0])
    }
    ColorQuantity::Y => Err("color: quantity 'Y' is scalar (no triple form).".to_string()),
  }
}

/// Wrap a hue difference in degrees to [-180, 180] BEFORE scaling
/// (179 vs -179 is 2 deg, not 358).
pub(crate) fn wrap_deg(d: f64) -> f64 {
  d - 360.0 * (d / 360.0).round()
}

/// Scalar objective `F(xyz)`: `w·ΔE²`, resp. `w·Σ((c−c_t)/tol)²`.
fn objective_of_xyz(demand: &ColorDemand, xyz: &[f64; 3]) -> Result<f64, String> {
  // Scalar Y: single residual off tol[0] (no triple form involved).
  if demand.quantity == ColorQuantity::Y {
    let ColorReference::Scalar(t) = &demand.reference else {
      return Err("color: quantity 'Y' needs a scalar reference.".to_string());
    };
    let r = (xyz[1] - t) / demand.tol[0];
    return Ok(demand.weight * r * r);
  }
  let c = color_of_xyz(xyz, &demand.white, demand.quantity)?;
  match demand.distance {
    ColorDistance::DeltaE2000 | ColorDistance::DeltaE76 => {
      let ColorReference::Triple(t) = &demand.reference else {
        return Err("color: scalar reference needs quantity 'Y'.".to_string());
      };
      // ΔE lives in Lab: the LCh ref converts (exact map), and so does
      // the op point (recomputed from XYZ — no trig roundtrip).
      let t_lab;
      let t_ref: &[f64; 3] = if demand.quantity == ColorQuantity::LCh {
        let mut lab = [[0.0; 3]];
        lch_to_lab(&[*t], &mut lab);
        t_lab = lab[0];
        &t_lab
      } else {
        t
      };
      let c_lab_hold;
      let c_lab: &[f64; 3] = if demand.quantity == ColorQuantity::LCh {
        let mut lab = [[0.0; 3]];
        xyz_to_lab(&[*xyz], &demand.white, &mut lab);
        c_lab_hold = lab[0];
        &c_lab_hold
      } else {
        &c
      };
      let d = match demand.distance {
        ColorDistance::DeltaE2000 => delta_e_2000_single(c_lab, t_ref, 1.0, 1.0, 1.0),
        _ => delta_e_76_single(c_lab, t_ref),
      };
      Ok(demand.weight * d * d)
    }
    ColorDistance::Channels => {
      let ColorReference::Triple(t) = &demand.reference else {
        return Err("color: scalar reference needs quantity 'Y'.".to_string());
      };
      let mut s = 0.0;
      for i in 0..3 {
        let mut diff = c[i] - t[i];
        // Hue wraps BEFORE scaling (LCh channel 2, degrees).
        if demand.quantity == ColorQuantity::LCh && i == 2 {
          diff = wrap_deg(diff);
        }
        let r = diff / demand.tol[i];
        s += r * r;
      }
      Ok(demand.weight * s)
    }
  }
}

/// `(residual = √F, grad = ∂F/∂R(λ))`: analytic `dXYZ/dR` (exact, linear)
/// chained with a central 3-pt FD of the XYZ→objective map
/// (`h = 1e-6·(1+|XYZ|)`). Cost: 1 XYZ + 6 tiny map evals.
/// `(residual = √F, covered)` with covered sim indices: the needle fold
/// deposits each gradient value at its own sim point (solver-mapped).
/// `eval_color` is the dense projection (covered order = ascending).
pub(crate) fn eval_color_covered(
  demand: &ColorDemand,
  sim_row: &[f64],
  sim_wl: &[f64],
) -> Result<(f64, Vec<(usize, f64)>), String> {
  let ws = xyz_workspace(sim_row, sim_wl, &demand.cmf, &demand.cmf_wl, &demand.illuminant, &demand.illum_wl)?;
  let f0 = objective_of_xyz(demand, &ws.xyz)?;
  if !f0.is_finite() {
    return Err("color: non-finite objective at op point.".to_string());
  }
  // Central FD of F over XYZ.
  let mut df = [0.0; 3];
  for j in 0..3 {
    let h = 1e-6 * (1.0 + ws.xyz[j].abs());
    let mut xp = ws.xyz;
    xp[j] += h;
    let mut xm = ws.xyz;
    xm[j] -= h;
    let fp = objective_of_xyz(demand, &xp)?;
    let fm = objective_of_xyz(demand, &xm)?;
    if !(fp.is_finite() && fm.is_finite()) {
      return Err("color: non-finite objective under FD step (singular map point).".to_string());
    }
    df[j] = (fp - fm) / (2.0 * h);
  }
  // Chain with analytic dXYZ/dR(λᵢ) = E·cmf·k·Δλ.
  let mut grad = Vec::with_capacity(ws.dw.len());
  for (j, &sim_idx) in ws.idx.iter().enumerate() {
    let s = ws.e_res[j] * ws.k * ws.dw[j];
    let g = s * (df[0] * ws.cmf_res[j][0] + df[1] * ws.cmf_res[j][1] + df[2] * ws.cmf_res[j][2]);
    grad.push((sim_idx, g));
  }
  Ok((f0.sqrt(), grad))
}

pub(crate) fn eval_color(
  demand: &ColorDemand,
  sim_row: &[f64],
  sim_wl: &[f64],
) -> Result<(f64, Vec<f64>), String> {
  eval_color_covered(demand, sim_row, sim_wl)
    .map(|(r, v)| (r, v.into_iter().map(|(_, g)| g).collect()))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::color::tables::default_tables;

  /// Tiny native toy tables (uniform 10 nm): independent of the embed.
  fn toy() -> (Vec<f64>, Vec<[f64; 3]>, Vec<f64>, Vec<f64>) {
    let wl: Vec<f64> = (0..8).map(|i| 500.0 + 10.0 * i as f64).collect();
    let cmf: Vec<[f64; 3]> = (0..8)
      .map(|i| [0.10 + 0.01 * i as f64, 0.20 + 0.01 * i as f64, 0.05 + 0.005 * i as f64])
      .collect();
    let illum = vec![1.0; 8];
    (wl.clone(), cmf, wl, illum)
  }

  fn toy_demand() -> ColorDemand {
    let (cmf_wl, cmf, illum_wl, illuminant) = toy();
    ColorDemand::new(
      0,
      cmf,
      cmf_wl,
      illuminant,
      illum_wl,
      ColorQuantity::Lab,
      ColorReference::Triple([60.0, 10.0, -20.0]),
      ColorDistance::DeltaE2000,
      2.0,
    )
    .unwrap()
  }

  #[test]
  fn white_identity_y_exactly_one() {
    // k-normalization self-check: perfect diffuser ⇒ Y ≡ 1 (same sums,
    // same order, numerator and denominator — bitwise, not approximate).
    let d = toy_demand();
    let ones = vec![1.0; 8];
    let (cmf_wl, cmf, illum_wl, illuminant) = toy();
    let xyz = xyz_of_spectrum(&ones, &cmf_wl, &cmf, &cmf_wl, &illuminant, &illum_wl).unwrap();
    // 1-ulp, not bitwise: the per-term k multiplication rounds once.
    assert!((xyz[1] - 1.0).abs() < 1e-15);
    assert!((d.white[1] - 1.0).abs() < 1e-15);
    assert!(xyz[0].is_finite() && xyz[2].is_finite());
  }

  #[test]
  fn ramp_matches_independent_trapezoid() {
    // Independent reimplementation (trapezoid, reverse summation): same
    // math, different formula AND different rounding path — agreement
    // bounds both formula and summation-order error.
    let (wl, cmf, _, illum) = toy();
    let ramp: Vec<f64> = (0..8).map(|i| 0.2 + 0.05 * i as f64).collect();
    let xyz = xyz_of_spectrum(&ramp, &wl, &cmf, &wl, &illum, &wl).unwrap();
    let h: f64 = 10.0;
    let denom: f64 = (0..8).rev().map(|i| illum[i] * cmf[i][1] * h).sum();
    let k = 1.0 / denom;
    for c in 0..3 {
      // Trapezoid differs (endpoint halves) — bound, don't equal.
      let trap_sum: f64 = (0..8)
        .map(|i| {
          let w: f64 = if i == 0 || i == 7 { 0.5 } else { 1.0 };
          w * ramp[i] * illum[i] * cmf[i][c]
        })
        .sum();
      let trap = h * trap_sum;
      // Coarse 8-pt grid: endpoint-halving legitimately differs ~15%
      // (the rectangular reverse twin below is the strict one).
      let rel = ((xyz[c] / k) - trap).abs() / trap.abs().max(1e-300);
      assert!(rel < 0.20, "c={c} rel={rel}");
    }
    // Rectangular twin (same formula, reverse order): tight.
    for c in 0..3 {
      let rect: f64 = (0..8).rev().map(|i| ramp[i] * illum[i] * cmf[i][c] * h).sum();
      let rel = ((xyz[c] / k) - rect).abs() / rect.abs().max(1e-300);
      assert!(rel < 1e-12, "c={c} rel={rel}");
    }
  }

  #[test]
  fn quantity_wrappers_are_bitwise() {
    let d = toy_demand();
    let xyz = [0.3, 0.4, 0.2];
    let mut lab = [[0.0; 3]];
    xyz_to_lab(&[xyz], &d.white, &mut lab);
    assert_eq!(color_of_xyz(&xyz, &d.white, ColorQuantity::Lab).unwrap(), lab[0]);
    let mut xyy = [[0.0; 3]];
    xyz_to_xyy(&[xyz], &mut xyy);
    assert_eq!(color_of_xyz(&xyz, &d.white, ColorQuantity::XyY).unwrap(), xyy[0]);
  }

  #[test]
  fn de_wrappers_are_bitwise() {
    let d = toy_demand();
    let xyz = [0.3, 0.4, 0.2];
    let lab = color_of_xyz(&xyz, &d.white, ColorQuantity::Lab).unwrap();
    let t = [60.0, 10.0, -20.0];
    assert_eq!(delta_e_2000_single(&lab, &t, 1.0, 1.0, 1.0), {
      let f = objective_of_xyz(&d, &xyz).unwrap();
      (f / 2.0).sqrt()
    });
  }

  #[test]
  fn gradient_matches_brute_force_bump() {
    // Analytic-Jacobian+3pt-FD g(λ) vs per-λ bumped full evals.
    let d = toy_demand();
    let (wl, _, _, _) = toy();
    let row = vec![0.5, 0.55, 0.6, 0.65, 0.6, 0.55, 0.5, 0.45];
    let (resid, grad) = eval_color(&d, &row, &wl).unwrap();
    assert!(resid.is_finite() && resid > 0.0);
    assert_eq!(grad.len(), 8);
    let delta = 1e-5;
    for i in 0..8 {
      let mut rp = row.clone();
      rp[i] += delta;
      let mut rm = row.clone();
      rm[i] -= delta;
      let fp = objective_of_xyz(&d, &xyz_of_spectrum(&rp, &wl, &d.cmf, &d.cmf_wl, &d.illuminant, &d.illum_wl).unwrap()).unwrap();
      let fm = objective_of_xyz(&d, &xyz_of_spectrum(&rm, &wl, &d.cmf, &d.cmf_wl, &d.illuminant, &d.illum_wl).unwrap()).unwrap();
      let brute = (fp - fm) / (2.0 * delta);
      let rel = (grad[i] - brute).abs() / brute.abs().max(1e-300);
      assert!(rel < 1e-6, "λ{i} rel={rel}");
    }
  }

  #[test]
  fn refusals_name_the_culprit() {
    let (cmf_wl, cmf, illum_wl, illuminant) = toy();
    let mk = |q, r, dist| {
      ColorDemand::new(
        0, cmf.clone(), cmf_wl.clone(), illuminant.clone(), illum_wl.clone(), q, r, dist, 1.0,
      )
    };
    assert!(mk(ColorQuantity::XyY, ColorReference::Triple([0.3, 0.3, 0.5]), ColorDistance::DeltaE2000)
      .unwrap_err()
      .contains("XyY"));
    // P2 quantities construct (compat matrix below gates the pairs).
    for q in [ColorQuantity::LCh, ColorQuantity::Oklab] {
      assert!(mk(q, ColorReference::Triple([0.0; 3]), ColorDistance::Channels).is_ok());
    }
    assert!(mk(ColorQuantity::Y, ColorReference::Scalar(0.5), ColorDistance::Channels).is_ok());
    assert!(mk(ColorQuantity::Oklab, ColorReference::Triple([0.0; 3]), ColorDistance::DeltaE2000)
      .unwrap_err()
      .contains("Oklab"));
    assert!(mk(ColorQuantity::Y, ColorReference::Scalar(0.5), ColorDistance::DeltaE76)
      .unwrap_err()
      .contains("'Y'"));
    // Shape gate fires before the P2 gate (malformed regardless of phase).
    assert!(mk(ColorQuantity::Y, ColorReference::Triple([0.0; 3]), ColorDistance::Channels)
      .unwrap_err()
      .contains("scalar"));
    assert!(mk(ColorQuantity::Lab, ColorReference::Scalar(1.0), ColorDistance::Channels)
      .unwrap_err()
      .contains("scalar reference"));
    // Empty overlap / single point / non-finite.
    let d = toy_demand();
    let far: Vec<f64> = (0..8).map(|i| 1000.0 + 10.0 * i as f64).collect();
    assert!(xyz_of_spectrum(&vec![0.5; 8], &far, &d.cmf, &d.cmf_wl, &d.illuminant, &d.illum_wl)
      .unwrap_err()
      .contains("no overlap"));
    let mut bad = vec![0.5; 8];
    bad[3] = f64::NAN;
    assert!(eval_color(&d, &bad, &toy().0).unwrap_err().contains("non-finite"));
  }

  #[test]
  fn embedded_defaults_recover_d65_white() {
    // End-to-end embed sanity: D65 white from embedded tables ≈ the
    // textbook D65 chromaticity (Y ≡ 1 by construction).
    let t = default_tables();
    assert_eq!(t.cmf_wl.len(), 471);
    assert_eq!(t.cmf_wl[0], 360.0);
    assert_eq!(t.cmf_wl[470], 830.0);
    let ones = vec![1.0; t.illum_wl.len()];
    let white =
      xyz_of_spectrum(&ones, &t.illum_wl, &t.cmf_xyz, &t.cmf_wl, &t.illum, &t.illum_wl).unwrap();
    assert!((white[1] - 1.0).abs() < 1e-12);
    assert!((white[0] - 0.9505).abs() < 1e-3, "X={}", white[0]);
    assert!((white[2] - 1.0890).abs() < 1e-3, "Z={}", white[2]);
  }

  fn p2_demand(q: ColorQuantity, r: ColorReference) -> ColorDemand {
    let (cmf_wl, cmf, illum_wl, illuminant) = toy();
    ColorDemand::new(0, cmf, cmf_wl, illuminant, illum_wl, q, r, ColorDistance::Channels, 1.0)
      .unwrap()
  }

  #[test]
  fn lch_wrappers_are_bitwise() {
    let d = p2_demand(ColorQuantity::LCh, ColorReference::Triple([60.0, 20.0, 100.0]));
    let xyz = [0.3, 0.4, 0.2];
    let mut lab = [[0.0; 3]];
    xyz_to_lab(&[xyz], &d.white, &mut lab);
    let mut lch = [[0.0; 3]];
    crate::color::func_02::lab_to_lch(&lab, &mut lch);
    assert_eq!(color_of_xyz(&xyz, &d.white, ColorQuantity::LCh).unwrap(), lch[0]);
    // ΔE path converts the LCh ref to Lab (roundtrip identity pins it).
    let mut back = [[0.0; 3]];
    crate::color::func_02::lch_to_lab(&[lch[0]], &mut back);
    for i in 0..3 {
      assert!((back[0][i] - lab[0][i]).abs() < 1e-12);
    }
  }

  #[test]
  fn oklab_wrapper_is_bitwise_and_white_is_unit() {
    use crate::color::common::REF_WHITE_D65;
    let d = p2_demand(ColorQuantity::Oklab, ColorReference::Triple([0.5, 0.01, -0.02]));
    let xyz = [0.3, 0.4, 0.2];
    let mut adapted = [[0.0; 3]];
    adapt(&[xyz], &d.white, &REF_WHITE_D65, false, &mut adapted);
    let mut ok = [[0.0; 3]];
    crate::color::func_04::xyz_to_oklab(&adapted, &mut ok);
    assert_eq!(color_of_xyz(&xyz, &d.white, ColorQuantity::Oklab).unwrap(), ok[0]);
    // Full Bradford path (clearly non-D65 toy white): maps onto D65 to
    // rounding. (Near-D65 whites take adapt's copy short-circuit instead —
    // the common D65-demand regime, exact by construction.)
    let (twl, tcmf, tewl, till) = toy();
    let ones = vec![1.0; 8];
    let tw = xyz_of_spectrum(&ones, &tewl, &tcmf, &twl, &till, &tewl).unwrap();
    let mut w65 = [[0.0; 3]];
    adapt(&[tw], &tw, &REF_WHITE_D65, false, &mut w65);
    for i in 0..3 {
      assert!((w65[0][i] - REF_WHITE_D65[i]).abs() < 1e-12, "{w65:?}");
    }
    // D65 white is Oklab near-neutral — up to a PRE-EXISTING in-tree
    // constant/matrix mismatch (REF_WHITE_D65 is the 2-dp-chromaticity
    // derivation [0.95045593, 1, 1.08905775]; the Oklab matrix is
    // native-white, hence b ≈ -1.2e-4). Systematic, ~1e-4 in b, far below
    // demand relevance; kernel and oracle agree bitwise regardless
    // (proven via the bound `_color` twins in R3).
    let mut wok = [[0.0; 3]];
    crate::color::func_04::xyz_to_oklab(&[REF_WHITE_D65], &mut wok);
    assert!((wok[0][0] - 1.0).abs() < 3e-6, "{wok:?}");
    assert!(wok[0][1].abs() < 3e-4 && wok[0][2].abs() < 3e-4, "{wok:?}");
  }

  #[test]
  fn lch_de_measures_in_lab() {
    // The ΔE arm converts BOTH sides to Lab (regression: c once leaked
    // through in LCh — the R3 twin caught it).
    let (cmf_wl, cmf, illum_wl, illuminant) = toy();
    let wl = cmf_wl.clone();
    let d = ColorDemand::new(
      0, cmf, cmf_wl, illuminant, illum_wl,
      ColorQuantity::LCh,
      ColorReference::Triple([60.0, 20.0, 100.0]),
      ColorDistance::DeltaE76,
      1.0,
    )
    .unwrap();
    let row = vec![0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25];
    let (r, _) = eval_color(&d, &row, &wl).unwrap();
    let xyz = xyz_of_spectrum(&row, &wl, &d.cmf, &d.cmf_wl, &d.illuminant, &d.illum_wl).unwrap();
    let mut lab = [[0.0; 3]];
    xyz_to_lab(&[xyz], &d.white, &mut lab);
    let mut ref_lab = [[0.0; 3]];
    lch_to_lab(&[[60.0, 20.0, 100.0]], &mut ref_lab);
    assert_eq!(r, delta_e_76_single(&lab[0], &ref_lab[0]));
  }

  #[test]
  fn y_scalar_is_exact_and_hue_wraps() {
    // wrap unit: 179 vs -179 differ by 2 deg, not 358.
    assert_eq!(wrap_deg(358.0), -2.0);
    assert_eq!(wrap_deg(-358.0), 2.0);
    // Y residual == hand formula, bitwise.
    let d = p2_demand(ColorQuantity::Y, ColorReference::Scalar(0.5));
    let (wl, _, _, _) = toy();
    let row = vec![0.6; 8];
    let (r, g) = eval_color(&d, &row, &wl).unwrap();
    let xyz = xyz_of_spectrum(&row, &wl, &d.cmf, &d.cmf_wl, &d.illuminant, &d.illum_wl).unwrap();
    let e = ((xyz[1] - 0.5) / 1.0).abs();
    assert_eq!(r, e);
    assert_eq!(g.len(), 8);
    // Hue wrap through the objective: h=179 vs ref h=-179 -> 2 deg gap.
    let dh = p2_demand(ColorQuantity::LCh, ColorReference::Triple([60.0, 20.0, -179.0]));
    let xyz_h = [0.3, 0.4, 0.2];
    let f = objective_of_xyz(&dh, &xyz_h).unwrap();
    let c = color_of_xyz(&xyz_h, &dh.white, ColorQuantity::LCh).unwrap();
    let gap = (wrap_deg(c[2] - -179.0) / 1.0).powi(2);
    let rest = ((c[0] - 60.0).powi(2) + (c[1] - 20.0).powi(2)).max(0.0);
    assert!((f - (rest + gap)).abs() < 1e-12, "{f}");
  }

  #[test]
  fn achromatic_gradients_stay_finite() {
    // C=0 singularity (LCh) + cbrt neutrals (Oklab): uniform spectra land
    // on the achromatic axis — gradients must stay finite (FD-twin class).
    let (wl, _, _, _) = toy();
    let row = vec![0.5; 8];
    for (q, r) in [
      (ColorQuantity::LCh, ColorReference::Triple([50.0, 5.0, 10.0])),
      (ColorQuantity::Oklab, ColorReference::Triple([0.5, 0.0, 0.0])),
      (ColorQuantity::Y, ColorReference::Scalar(0.4)),
    ] {
      let (cmf_wl, cmf, illum_wl, illuminant) = toy();
      let d = ColorDemand::new(0, cmf, cmf_wl, illuminant, illum_wl, q, r, ColorDistance::Channels, 1.0)
        .unwrap();
      let (resid, grad) = eval_color(&d, &row, &wl).unwrap();
      assert!(resid.is_finite());
      assert!(grad.iter().all(|v| v.is_finite()), "{q:?}");
    }
  }
}
