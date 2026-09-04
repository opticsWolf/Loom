//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::needle_pass — analytic needle insertion pass.
//!
//! Replaces the brute-force `compute_p_function` from needle_synthesis.py
//! (which inserted a 1 nm test needle and re-solved the FULL stack at every
//! scan position). Here one sweep of the analytic operator yields P(z)
//! everywhere at once:
//!
//!   P(z) = Σ_k 2·w_k·(R_k − Rt_k)·Re{conj(r_k)·∂r_k/∂δ}      (half-gradient)
//!
//! summed over spectral points (angle-major k = a·num_wavs + w) and over the
//! enabled polarization branches. The most-negative-P site is exactly the
//! argmin-MF-after-insertion site in the δ→0 limit — without the 1 nm
//! test-thickness bias.
//!
//! Scan-grid contract (mirrors compute_p_function verbatim):
//!   * per admissible film: interior positions k·step, k = 1 .. int(d/step)−1
//!   * non-admissible films advance the cumulative depth only
//!
//! Merit coupling: `build_needle_targets` folds a MeritSpec into flat
//! per-solver-point (raw target, folded weight) arrays. Multiple entries
//! overlapping one solver point fold EXACTLY (not approximately): since
//! every merit term is quadratic in R with positive weight,
//!   Σ_e w_e·(R − t_e) = W·(R − t_eff),  W = Σw_e, t_eff = Σ(w_e t_e)/W.
//! Weights carry the normalization/tolerance folding w = nf²/tol² so the
//! descent direction matches dF/dδ of the spectralweave merit function.

use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;

use crate::needle_operator::{build_stack_fields_range, p_coherent_from_fields};
use crate::synthesis::merit::{CurveId, MeritSpec, SimCurves};

// ---------------------------------------------------------------------------
// Scan candidates (Python-parity grid)
// ---------------------------------------------------------------------------

/// One candidate insertion site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanSite {
    /// Film index (0-based within `DesignStack::films`).
    pub film_idx: usize,
    /// Depth below the top of that film (nm).
    pub depth_into_layer_nm: f64,
    /// Cumulative physical depth from the top of the first host-range film.
    pub z_nm: f64,
}

/// Build the candidate list exactly like `compute_p_function`: interior
/// multiples of `scan_step_nm` inside each admissible film.
///
/// `films` are the film layers; host admissibility = `layer.needle`.
pub fn build_scan_sites(films: &[crate::synthesis::structure::LayerSpec], scan_step_nm: f64) -> Vec<ScanSite> {
    assert!(scan_step_nm > 0.0, "scan_step_nm must be positive");
    let mut sites = Vec::new();
    let mut cumulative = 0.0f64;

    for (film_idx, layer) in films.iter().enumerate() {
        let d = layer.d_nm;
        if layer.needle && d > 0.0 {
            let n_steps = (d / scan_step_nm) as i32;
            for k in 1..n_steps {
                let pos_in_layer = k as f64 * scan_step_nm;
                sites.push(ScanSite {
                    film_idx,
                    depth_into_layer_nm: pos_in_layer,
                    z_nm: cumulative + pos_in_layer,
                });
            }
        }
        cumulative += d;
    }
    sites
}

// ---------------------------------------------------------------------------
// MeritSpec → flat needle targets/weights
// ---------------------------------------------------------------------------

/// Folded needle inputs: one `(targets, weights)` pair per quantity, all in
/// angle-major layout (k = a·num_wavs + w). All-zero pairs mean "no demands
/// of that quantity" (the engine skips zero-weight points naturally) and
/// feed straight into `needle_gradient`'s `targets_r/weights_r` (`.r`),
/// `targets_t/weights_t` (`.t`), `targets_a/weights_a` (`.a`), the
/// back-incidence siblings (`.rb`, `.tb`, `.ab`), and one pair per
/// S-matrix channel for phase demands (`.phi[0..=3]` → separate `P_PHI`
/// calls, since the engine takes a single channel per call).
pub struct NeedleTargets {
    pub r: (Vec<f64>, Vec<f64>),
    pub t: (Vec<f64>, Vec<f64>),
    pub a: (Vec<f64>, Vec<f64>),
    pub rb: (Vec<f64>, Vec<f64>),
    pub tb: (Vec<f64>, Vec<f64>),
    pub ab: (Vec<f64>, Vec<f64>),
    pub phi: [(Vec<f64>, Vec<f64>); 4],
}

/// Fold a [`MeritSpec`] into flat per-quantity needle inputs.
///
/// * Intensity demands (front and back R/T) fold to their own quantity
///   bucket; absorption demands (front and back) derive A = 1 − R − T from
///   the companion curves and fold to the absorption buckets.
/// * Phase demands fold to the `.phi[channel]` pair of their element
///   (see `CurveId::phase_channel`); emit one `P_PHI` call per used channel.
/// * Intensity/absorption demands require **linear** normalization;
///   phase demands accept linear or phase (wrapped residuals mirror the
///   evaluator). Anything else returns Err.
/// * `current_sim`, when given, activates the one-sided and banded kinds at
///   the current operating point: satisfied Above/Below points and in-band
///   Range points contribute nothing; Range/CenterBand violations fold to
///   the nearest band edge, CenterBand interiors fold to the centre with
///   reduced weight `nf²/bw²` (matching calculate_merit's masking). Without
///   `current_sim` every kind folds conservatively (Range/CenterBand as
///   Exact at the centre).
/// * The CenterBand `+1` merit level is a constant offset: it shifts merit
///   values but not needle gradients, so the fold drops it by design —
///   folded merit reads lower than `calculate_merit` by exactly the count
///   of currently-violated `c` points (`M_true = M_folded + N_outside`).
///   See `docs/spectralweave-target-kinds.md` for the full kind/fold reference.
pub fn build_needle_targets(
    spec: &MeritSpec,
    angles: &[f64],
    wavelengths: &[f64],
    current_sim: Option<&SimCurves>,
) -> Result<NeedleTargets, String> {
    let na = angles.len();
    let nw = wavelengths.len();
    let zero = || (vec![0.0f64; na * nw], vec![0.0f64; na * nw]);
    let mut buckets: [(Vec<f64>, Vec<f64>); 6] =
        [zero(), zero(), zero(), zero(), zero(), zero()];
    let mut phi: [(Vec<f64>, Vec<f64>); 4] = [zero(), zero(), zero(), zero()];

    for t in spec.targets() {
        let key = &spec.keys()[t.key_idx as usize];
        // Demand bucket: absorption derives from companions, intensities
        // fold to their own quantity, phase demands to their element pair.
        // (index-based so the loop holds no &mut across iterations)
        enum BucketKind { Pair(usize), Phi(usize) }
        let bucket = if t.phase {
            match key.curve.phase_channel() {
                Some(ch) => BucketKind::Phi(ch),
                None => return Err(format!(
                    "phase demand on {:?}: no S-matrix element", key.curve)),
            }
        } else {
            BucketKind::Pair(match key.curve {
                CurveId::Rs | CurveId::Rp | CurveId::Ru => 0,
                CurveId::Ts | CurveId::Tp | CurveId::Tu => 1,
                CurveId::As | CurveId::Ap | CurveId::Au => 2,
                CurveId::RBs | CurveId::RBp | CurveId::RBu => 3,
                CurveId::TBs | CurveId::TBp | CurveId::TBu => 4,
                CurveId::ABs | CurveId::ABp | CurveId::ABu => 5,
            })
        };
        if t.phase {
            if !matches!(t.transform,
                crate::synthesis::merit::SimTransform::Linear |
                crate::synthesis::merit::SimTransform::Phase)
            {
                return Err(format!(
                    "phase demands need linear/phase normalization; got {:?}",
                    t.transform
                ));
            }
        } else if t.transform != crate::synthesis::merit::SimTransform::Linear {
            return Err(format!(
                "needle pass requires linear-normalized targets; got {:?}",
                t.transform
            ));
        }

        // Angle row: argmin(|angles − key.angle|), first minimum wins —
        // same semantics as SimCurves::angle_row.
        let mut row = 0usize;
        let mut best_d = f64::INFINITY;
        for (a, &ang) in angles.iter().enumerate() {
            let dd = (ang - key.angle).abs();
            if dd < best_d {
                best_d = dd;
                row = a;
            }
        }

        // Operating-point rows for activation (None before the first
        // simulation, or when a companion/complex row is missing — then
        // banded/one-sided kinds fold conservatively). Rows are sliced to
        // the demand angle (the old code sampled the whole row-major
        // array, misreading multi-angle sims).
        enum OpRows<'a> {
            Intensity(&'a [f64]),
            Absorption(&'a [f64], &'a [f64]),
            Phase(&'a [Complex64]),
        }
        let op: Option<OpRows> = match current_sim {
            None => None,
            Some(sim) => {
                let ar = sim.angle_row(key.angle);
                let nws = sim.wavelengths.len();
                if nws == 0 {
                    None
                } else {
                    let irow = |id: CurveId| -> Option<&[f64]> {
                        let arc = if id.is_back() { sim.back_curve(id) } else { sim.curve(id) };
                        arc.map(|c| sim_row(c, ar, nws))
                    };
                    if t.phase {
                        // Registration guarantees a phaseable key.
                        let crow = if key.curve.is_back() {
                            sim.complex_back_curve(key.curve)
                        } else {
                            sim.complex_curve(key.curve)
                        };
                        crow.map(|c| OpRows::Phase(&c[ar * nws..(ar + 1) * nws]))
                    } else {
                        match key.curve.absorption_companions() {
                            Some((rc, tc)) => match (irow(rc), irow(tc)) {
                                (Some(r), Some(t2)) => Some(OpRows::Absorption(r, t2)),
                                _ => None,
                            },
                            None => irow(key.curve).map(OpRows::Intensity),
                        }
                    }
                }
            },
        };

        let twl: &[f64] = &t.wavelengths;
        for &twl_i in twl.iter() {
            // Interpolate normalized target, tolerance and band half-width
            // onto this solver wavelength (edge-clamped; ascending grids).
            let tgt_norm = interp_clamped(twl, &t.normalized_targets, twl_i);
            let tol = interp_clamped(twl, &t.tolerances, twl_i).max(1e-300);
            let bw = if t.band.is_empty() { 0.0 } else { interp_clamped(twl, &t.band, twl_i).max(0.0) };
            let raw_target = tgt_norm / t.norm_factor;
            let nf2 = t.norm_factor * t.norm_factor;

            // Operating-point residual for activation (None before the
            // first simulation). Absorption samples A = 1 − R − T from
            // the companion rows; phase samples arg() of the complex row,
            // wrapped exactly when the evaluator would wrap (Phase mode).
            let wrap_phase = t.phase
                && t.transform == crate::synthesis::merit::SimTransform::Phase;
            let d_opt: Option<f64> = match (current_sim, &op) {
                (Some(sim), Some(rows)) => {
                    let v = match rows {
                        OpRows::Intensity(r) => interp_clamped(&sim.wavelengths, r, twl_i),
                        OpRows::Absorption(r, t2) => {
                            1.0 - interp_clamped(&sim.wavelengths, r, twl_i)
                                - interp_clamped(&sim.wavelengths, t2, twl_i)
                        },
                        OpRows::Phase(c) => {
                            cinterp_clamped(&sim.wavelengths, c, twl_i).arg()
                        },
                    };
                    let mut d = v * t.norm_factor - tgt_norm;
                    if wrap_phase {
                        d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
                    }
                    Some(d)
                },
                _ => None,
            };

            // Fold to an equivalent (raw_target, weight) quadratic, or skip.
            let folded: Option<(f64, f64)> = match t.kind {
                crate::synthesis::merit::ConstraintKind::Exact => {
                    Some((raw_target, nf2 / (tol * tol)))
                },
                crate::synthesis::merit::ConstraintKind::Above => match d_opt {
                    Some(d) if d >= 0.0 => None,
                    _ => Some((raw_target, nf2 / (tol * tol))),
                },
                crate::synthesis::merit::ConstraintKind::Below => match d_opt {
                    Some(d) if d <= 0.0 => None,
                    _ => Some((raw_target, nf2 / (tol * tol))),
                },
                crate::synthesis::merit::ConstraintKind::Range => {
                    let bw_eff = if bw <= 0.0 { tol } else { bw };
                    match d_opt {
                        // No sim yet: conservative exact fold at the centre.
                        None => Some((raw_target, nf2 / (tol * tol))),
                        Some(d) if d.abs() <= bw_eff => None,
                        Some(d) => {
                            let edge_raw = (tgt_norm + d.signum() * bw_eff) / t.norm_factor;
                            Some((edge_raw, nf2 / (tol * tol)))
                        },
                    }
                },
                crate::synthesis::merit::ConstraintKind::CenterBand => {
                    if bw <= 0.0 {
                        Some((raw_target, nf2 / (tol * tol)))
                    } else {
                        match d_opt {
                            // No sim yet: exact fold at the centre.
                            None => Some((raw_target, nf2 / (tol * tol))),
                            // Inside: reduced weight nf²/bw² at the centre.
                            Some(d) if d.abs() <= bw => Some((raw_target, nf2 / (bw * bw))),
                            // Outside: nearest edge drives (the +1 merit
                            // level is gradient-free, dropped by design).
                            Some(d) => {
                                let edge_raw = (tgt_norm + d.signum() * bw) / t.norm_factor;
                                Some((edge_raw, nf2 / (tol * tol)))
                            },
                        }
                    }
                },
            };
            let Some((rt, w)) = folded else { continue };

            let k = row * nw + solver_wav_index(wavelengths, twl_i);
            match bucket {
                BucketKind::Pair(i) => {
                    buckets[i].1[k] += w;
                    buckets[i].0[k] += w * rt;
                },
                BucketKind::Phi(ch) => {
                    phi[ch].1[k] += w;
                    phi[ch].0[k] += w * rt;
                },
            }
        }
    }

    // Accumulate W and W·t separately, then divide once (exact fold).
    for (bt, bw_) in buckets.iter_mut().map(|(t, w)| (t, w))
        .chain(phi.iter_mut().map(|(t, w)| (t, w))) {
        for k in 0..bt.len() {
            if bw_[k] > 0.0 {
                bt[k] /= bw_[k];
            }
        }
    }
    let [r, t, a, rb, tb, ab] = buckets;
    Ok(NeedleTargets { r, t, a, rb, tb, ab, phi })
}

/// One angle row of a row-major [n_angles, n_wav] sim curve.
fn sim_row(curve: &Arc<[f64]>, angle_row: usize, n_wav: usize) -> &[f64] {
    &curve[angle_row * n_wav..(angle_row + 1) * n_wav]
}

/// Complex twin of [`interp_clamped`] for phase operating points.
fn cinterp_clamped(xs: &[f64], ys: &[Complex64], x: f64) -> Complex64 {
    debug_assert_eq!(xs.len(), ys.len());
    if xs.is_empty() {
        return Complex64::new(0.0, 0.0);
    }
    if x <= xs[0] {
        return ys[0];
    }
    let last = xs.len() - 1;
    if x >= xs[last] {
        return ys[last];
    }
    let hi = xs.partition_point(|&v| v < x).max(1);
    let lo = hi - 1;
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + (ys[hi] - ys[lo]) * t
}

fn solver_wav_index(wavelengths: &[f64], x: f64) -> usize {
    // Nearest solver wavelength to the target point (grids usually align
    // bit-exactly; nearest keeps misaligned cases well-defined).
    let mut best = 0usize;
    let mut bd = f64::INFINITY;
    for (i, &v) in wavelengths.iter().enumerate() {
        let d = (v - x).abs();
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

fn interp_clamped(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    let last = xs.len() - 1;
    if x >= xs[last] {
        return ys[last];
    }
    let hi = xs.partition_point(|&v| v < x).max(1);
    let lo = hi - 1;
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NeedlePassResult {
    /// Candidate sites, in scan order.
    pub sites: Vec<ScanSite>,
    /// P accumulated over spectral points and enabled polarizations,
    /// aligned with `sites`. Negative ⇒ predicted merit decrease when a
    /// seed of the contrast material is inserted there.
    pub p_profile: Vec<f64>,
}

impl NeedlePassResult {
    /// Most-negative-P site, if any candidate is negative.
    pub fn best(&self) -> Option<(&ScanSite, f64)> {
        let (i, &p) = self
            .p_profile
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))?;
        if p >= 0.0 {
            None
        } else {
            Some((&self.sites[i], p))
        }
    }
}

/// Inputs for one analytic needle sweep (fully coherent block path).
pub struct NeedlePassInput<'a> {
    /// Flat n_stack_cache (wav-major, re/im interleaved) — see
    /// `DesignStack::solver_arrays`.
    pub n_stack_cache: &'a [f64],
    pub thicknesses: &'a [f64],
    pub rough_types: &'a [i32],
    pub rough_vals: &'a [f64],
    pub n_layers: usize,
    pub wavls: &'a [f64],
    /// Sines of the incidence angles (solver convention).
    pub sin_theta: &'a [f64],
    /// Raw reflectance targets, angle-major (see `build_needle_targets`).
    pub targets_r: &'a [f64],
    /// Folded merit weights, angle-major.
    pub weights_r: &'a [f64],
    /// Complex index of the contrast material, per wavelength.
    pub needle_n_per_wav: &'a [Complex64],
    /// Coherent sub-block `[start_idx, end_idx)` (absolute indices);
    /// hosts are the interior layers. Fully-coherent stacks: (0, nl−1).
    pub start_idx: usize,
    pub end_idx: usize,
    pub calc_s: bool,
    pub calc_p: bool,
}

/// Run the analytic P(z) sweep. Sites come from `build_scan_sites`; the
/// profile accumulates the half-gradient over all spectral points and both
/// polarization branches (caller folds U-targets by enabling both branches).
pub fn needle_pass_scan(input: &NeedlePassInput<'_>, sites: &[ScanSite]) -> Result<NeedlePassResult, String> {
    let nl = input.n_layers;
    let nw = input.wavls.len();
    let na = input.sin_theta.len();
    let nz = sites.len();

    if nw == 0 || na == 0 {
        return Err("empty spectral/angular grid".into());
    }
    if input.n_stack_cache.len() != nw * nl * 2 {
        return Err("n_stack_cache layout mismatch".into());
    }
    if input.thicknesses.len() != nl || input.rough_types.len() != nl || input.rough_vals.len() != nl {
        return Err("per-layer array length mismatch".into());
    }
    if input.targets_r.len() != na * nw || input.weights_r.len() != na * nw {
        return Err("targets/weights must be angle-major na*nw".into());
    }
    if input.needle_n_per_wav.len() != nw {
        return Err("needle_n_per_wav must have one index per wavelength".into());
    }
    if !(input.start_idx < input.end_idx && input.end_idx < nl) {
        return Err("invalid block range".into());
    }
    if nz == 0 {
        return Ok(NeedlePassResult { sites: Vec::new(), p_profile: Vec::new() });
    }
    let z_grid: Vec<f64> = sites.iter().map(|s| s.z_nm).collect();

    // Per-point P rows computed in parallel (fields built once per pol),
    // then folded in k order for bit-reproducible accumulation.
    let total_points = na * nw;
    let pol_on = [input.calc_s, input.calc_p];
    if !pol_on[0] && !pol_on[1] {
        return Err("no polarization branch enabled".into());
    }

    let rows: Vec<Vec<f64>> = (0..total_points)
        .into_par_iter()
        .map(|k| {
            let a = k / nw;
            let w = k % nw;
            let lam = input.wavls[w];
            let sin_t = input.sin_theta[a];
            let base = w * nl * 2;
            let ns: Vec<Complex64> = (0..nl)
                .map(|l| Complex64::new(input.n_stack_cache[base + l * 2], input.n_stack_cache[base + l * 2 + 1]))
                .collect();
            let nsin_fi = ns[0] * Complex64::new(sin_t, 0.0);
            let np_c = input.needle_n_per_wav[w];
            let tgt = input.targets_r[k];
            let wgt = input.weights_r[k];

            let mut acc = vec![0.0f64; nz];
            for (pi, &on) in pol_on.iter().enumerate() {
                if !on {
                    continue;
                }
                let fields = build_stack_fields_range(
                    input.start_idx,
                    input.end_idx,
                    &ns,
                    input.thicknesses,
                    input.rough_vals,
                    input.rough_types,
                    lam,
                    nsin_fi,
                    pi as i32,
                );
                let contrib = p_coherent_from_fields(
                    &fields,
                    nsin_fi,
                    lam,
                    pi as i32,
                    np_c,
                    tgt,
                    wgt,
                    input.thicknesses,
                    input.start_idx,
                    input.end_idx,
                    &z_grid,
                );
                for (zi, v) in contrib.into_iter().enumerate() {
                    acc[zi] += v;
                }
            }
            acc
        })
        .collect::<Vec<_>>();

    let mut p_profile = vec![0.0f64; nz];
    for row in &rows {
        for zi in 0..nz {
            p_profile[zi] += row[zi];
        }
    }

    Ok(NeedlePassResult { sites: sites.to_vec(), p_profile })
}

/// Convenience wrapper: scan sites + selection in one call.
pub fn run_needle_pass(input: &NeedlePassInput<'_>, films: &[crate::synthesis::structure::LayerSpec], scan_step_nm: f64) -> Result<NeedlePassResult, String> {
    let sites = build_scan_sites(films, scan_step_nm);
    needle_pass_scan(input, &sites)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics_core::cplx;
    use crate::synthesis::merit::{
        ConstraintKind, MeritKey, MeritTarget, SimTransform,
    };
    use crate::synthesis::structure::LayerSpec;

    const NW: usize = 6;
    const NL: usize = 5; // air, H, L, H, sub

    const TEST_WAVLS: [f64; NW] = [400.0, 500.0, 600.0, 700.0, 800.0, 900.0];

    fn nk_const(n_re: f64) -> Arc<[Complex64]> {
        vec![cplx(n_re, 0.0); NW].into()
    }

    /// air | H(40) L(30) H(50) | substrate
    fn test_stack_arrays() -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>) {
        let films_d = [40.0, 30.0, 50.0];
        let mut thicknesses = vec![0.0f64]; // ambient
        thicknesses.extend_from_slice(&films_d);
        thicknesses.push(0.0); // substrate

        let n_re = [1.0, 2.35, 1.46, 2.35, 1.52];
        let mut cache = Vec::with_capacity(NW * NL * 2);
        for _w in 0..NW {
            for l in 0..NL {
                cache.push(n_re[l]);
                cache.push(0.0);
            }
        }
        (cache, thicknesses, vec![0; NL], vec![0.0; NL])
    }

    fn pass_input<'a>(
        cache: &'a [f64],
        d: &'a [f64],
        rt: &'a [i32],
        rv: &'a [f64],
        targets: &'a [f64],
        weights: &'a [f64],
        needle_n: &'a [Complex64],
    ) -> NeedlePassInput<'a> {
        NeedlePassInput {
            n_stack_cache: cache,
            thicknesses: d,
            rough_types: rt,
            rough_vals: rv,
            n_layers: NL,
            wavls: &TEST_WAVLS,
            sin_theta: &[0.0],
            targets_r: targets,
            weights_r: weights,
            needle_n_per_wav: needle_n,
            start_idx: 0,
            end_idx: NL - 1,
            calc_s: true,
            calc_p: false,
        }
    }

    #[test]
    fn scan_sites_mirror_python_loop() {
        let films = vec![
            LayerSpec::constant("H", 2.35, 0.0, 10.0, NW), // step 2 → 4 sites
            { let mut l = LayerSpec::constant("X", 1.0, 0.0, 7.0, NW); l.needle = false; l }, // skipped, advances 7
            LayerSpec::constant("L", 1.46, 0.0, 5.0, NW),  // step 2 → 1 site (k=1: 2<5; k=2: 4 ≥ int(5/2)=2 stops)
        ];
        let sites = build_scan_sites(&films, 2.0);
        let got: Vec<(usize, f64, f64)> = sites
            .iter()
            .map(|s| (s.film_idx, s.depth_into_layer_nm, s.z_nm))
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 2.0, 2.0),
                (0, 4.0, 4.0),
                (0, 6.0, 6.0),
                (0, 8.0, 8.0),
                (2, 2.0, 19.0), // cumulative 10+7 → z = 17+2
            ]
        );
    }

    #[test]
    fn profile_matches_p_function_oracle_bitwise() {
        // Independent oracle: needle_operator::p_function (the FD-validated
        // grid driver) called directly on the same inputs.
        let (cache, d, rt, rv) = test_stack_arrays();
        let wavls: Vec<f64> = (0..NW).map(|i| 400.0 + 100.0 * i as f64).collect();
        let targets = vec![0.05f64; NW]; // one angle → na*nw = nw
        let weights = vec![100.0f64; NW];
        let needle_n = nk_const(1.46);

        let films = vec![
            LayerSpec::constant("H", 2.35, 0.0, 40.0, NW),
            LayerSpec::constant("L", 1.46, 0.0, 30.0, NW),
            LayerSpec::constant("H", 2.35, 0.0, 50.0, NW),
        ];
        let res = run_needle_pass(
            &pass_input(&cache, &d, &rt, &rv, &targets, &weights, &needle_n),
            &films,
            2.5,
        )
        .unwrap();

        let z_grid: Vec<f64> = res.sites.iter().map(|s| s.z_nm).collect();
        let oracle = crate::needle_operator::p_function(
            &wavls,
            &[0.0],
            0,
            NL - 1,
            NL,
            &cache,
            &d,
            &rt,
            &rv,
            &needle_n,
            &targets,
            &weights,
            &z_grid,
            0,
        )
        .unwrap();

        assert_eq!(res.p_profile.len(), oracle.len());
        for (a, b) in res.p_profile.iter().zip(&oracle) {
            let scale = a.abs().max(b.abs()).max(1e-30);
            assert!((a - b).abs() / scale < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn dual_pol_equals_sum_of_branches() {
        let (cache, d, rt, rv) = test_stack_arrays();
        let targets = vec![0.02f64; NW];
        let weights = vec![50.0f64; NW];
        let needle_n = nk_const(1.46);
        let films = vec![
            LayerSpec::constant("H", 2.35, 0.0, 40.0, NW),
            LayerSpec::constant("L", 1.46, 0.0, 30.0, NW),
            LayerSpec::constant("H", 2.35, 0.0, 50.0, NW),
        ];

        let mut inp = pass_input(&cache, &d, &rt, &rv, &targets, &weights, &needle_n);
        let s_only = run_needle_pass(&inp, &films, 3.0).unwrap();
        inp.calc_s = false;
        inp.calc_p = true;
        let p_only = run_needle_pass(&inp, &films, 3.0).unwrap();
        inp.calc_s = true;
        let both = run_needle_pass(&inp, &films, 3.0).unwrap();

        for i in 0..both.p_profile.len() {
            let sum = s_only.p_profile[i] + p_only.p_profile[i];
            assert!((both.p_profile[i] - sum).abs() < 1e-9 * sum.abs().max(1e-30) + 1e-18);
        }
        // A mismatched stack should show negative P somewhere (improvable).
        assert!(s_only.best().is_some());
    }

    #[test]
    fn best_selects_most_negative_site() {
        let (cache, d, rt, rv) = test_stack_arrays();
        let targets = vec![0.01f64; NW];
        let weights = vec![200.0f64; NW];
        let needle_n = nk_const(1.46);
        let films = vec![
            LayerSpec::constant("H", 2.35, 0.0, 40.0, NW),
            LayerSpec::constant("L", 1.46, 0.0, 30.0, NW),
            LayerSpec::constant("H", 2.35, 0.0, 50.0, NW),
        ];
        let res = run_needle_pass(
            &pass_input(&cache, &d, &rt, &rv, &targets, &weights, &needle_n),
            &films,
            2.0,
        )
        .unwrap();
        let (site, p) = res.best().expect("expected an improving site");
        assert!(p < 0.0);
        assert_eq!(p, res.p_profile.iter().cloned().fold(f64::INFINITY, f64::min));
        // Site must lie inside its claimed film.
        let film_top: f64 = films[..site.film_idx].iter().map(|f| f.d_nm).sum();
        assert!(site.depth_into_layer_nm > 0.0 && site.depth_into_layer_nm < films[site.film_idx].d_nm);
        assert!((site.z_nm - (film_top + site.depth_into_layer_nm)).abs() < 1e-12);
    }

    // -- build_needle_targets ------------------------------------------------

    fn spec_single(
        angle: f64,
        curve: CurveId,
        wl: &[f64],
        norm_targets: &[f64],
        tols: &[f64],
        nf: f64,
        kind: ConstraintKind,
    ) -> MeritSpec {
        spec_banded(angle, curve, wl, norm_targets, tols, &[], nf, kind)
    }

    fn spec_single_phase(
        angle: f64,
        curve: CurveId,
        wl: &[f64],
        norm_targets: &[f64],
        tols: &[f64],
        nf: f64,
        kind: ConstraintKind,
    ) -> MeritSpec {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle, curve });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: wl.to_vec().into(),
            kind,
            transform: SimTransform::Phase,
            norm_factor: nf,
            normalized_targets: norm_targets.to_vec().into(),
            tolerances: tols.to_vec().into(),
            band: vec![].into(),
            phase: true,
        })
        .unwrap();
        spec
    }

    fn entry_for_phase(key_idx: u32) -> MeritTarget {
        MeritTarget {
            key_idx,
            wavelengths: vec![400.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Phase,
            norm_factor: 1.0,
            normalized_targets: vec![1.0].into(),
            tolerances: vec![0.1].into(),
            band: vec![].into(),
            phase: true,
        }
    }

    fn spec_banded(
        angle: f64,
        curve: CurveId,
        wl: &[f64],
        norm_targets: &[f64],
        tols: &[f64],
        band: &[f64],
        nf: f64,
        kind: ConstraintKind,
    ) -> MeritSpec {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle, curve });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: wl.to_vec().into(),
            kind,
            transform: SimTransform::Linear,
            norm_factor: nf,
            normalized_targets: norm_targets.to_vec().into(),
            tolerances: tols.to_vec().into(),
            band: band.to_vec().into(),
            phase: false,
        })
        .unwrap();
        spec
    }

    fn sim_single(angle: f64, wl: f64, curve: CurveId, val: f64) -> SimCurves {
        let mut sim = SimCurves {
            angles: vec![angle].into(),
            wavelengths: vec![wl].into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
        };
        sim.curves[curve.index()] = Some(vec![val].into());
        sim
    }

    #[test]
    fn targets_builder_linear_fold_exact() {
        // nf = 4/3, raw targets [0.5, 1.0] on solver wavelengths 400/500.
        let nf = 4.0f64 / 3.0;
        let spec = spec_single(
            0.0,
            CurveId::Rs,
            &[400.0, 500.0],
            &[0.5 * nf, 1.0 * nf],
            &[0.01, 0.02],
            nf,
            ConstraintKind::Exact,
        );
        let angles = [0.0];
        let wavs = [400.0, 500.0, 600.0];
        let (tgt, wgt) = build_needle_targets(&spec, &angles, &wavs, None).unwrap().r;
        assert_eq!(tgt.len(), 3);
        assert!((tgt[0] - 0.5).abs() < 1e-14); // raw target recovered
        assert!((tgt[1] - 1.0).abs() < 1e-14);
        assert_eq!(tgt[2], 0.0); // no demand → zero target AND weight
        assert_eq!(wgt[2], 0.0);
        // folded weight = nf²/tol²
        assert!((wgt[0] - nf * nf / (0.01 * 0.01)).abs() < 1e-9);
        assert!((wgt[1] - nf * nf / (0.02 * 0.02)).abs() < 1e-6);
    }

    #[test]
    fn targets_builder_exact_overlap_fold() {
        // Two entries demanding different raw targets at the SAME solver
        // point: quadratic forms fold exactly to the weighted mean.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rp });
        let mk = |norm_t: f64, tol: f64, nf: f64| MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![500.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Linear,
            norm_factor: nf,
            normalized_targets: vec![norm_t].into(),
            tolerances: vec![tol].into(),
            band: vec![].into(),
            phase: false,
        };
        let (nfa, nfb) = (2.0_f64, 3.0_f64);
        spec.add_target(mk(0.4 * nfa, 0.1, nfa)).unwrap();
        spec.add_target(mk(0.6 * nfb, 0.2, nfb)).unwrap();

        let (tgt, wgt) = build_needle_targets(&spec, &[0.0], &[500.0], None).unwrap().r;
        let wa = nfa * nfa / 0.01;
        let wb = nfb * nfb / 0.04;
        let expect_t = (wa * 0.4 + wb * 0.6) / (wa + wb);
        assert!((tgt[0] - expect_t).abs() < 1e-12);
        assert!((wgt[0] - (wa + wb)).abs() < 1e-9);
    }

    #[test]
    fn targets_builder_above_masking_uses_current_sim() {
        // Above-target constraint, already satisfied → masked out.
        let nf = 1.0;
        let spec = spec_single(
            0.0,
            CurveId::Ru,
            &[400.0],
            &[0.5], // normalized target (== raw, nf=1)
            &[0.1],
            nf,
            ConstraintKind::Above,
        );
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: vec![400.0].into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
        };
        sim.curves[CurveId::Ru.index()] = Some(vec![0.7f64].into()); // above → satisfied

        let (tgt, wgt) =
            build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim)).unwrap().r;
        assert_eq!(wgt[0], 0.0);
        assert_eq!(tgt[0], 0.0);

        sim.curves[CurveId::Ru.index()] = Some(vec![0.3f64].into()); // violated → active
        let (tgt, wgt) =
            build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim)).unwrap().r;
        assert!(wgt[0] > 0.0);
        assert!((tgt[0] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn targets_builder_range_folds_to_edge_or_masks() {
        // Centre 0.5 (nf=1), tol 0.1, band 0.05 → edges at 0.45/0.55, w=100.
        let spec = spec_banded(0.0, CurveId::Rs, &[400.0], &[0.5], &[0.1],
            &[0.05], 1.0, ConstraintKind::Range);
        // In-band operating point → masked out.
        let sim_in = sim_single(0.0, 400.0, CurveId::Rs, 0.52);
        let (_, w) = build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim_in)).unwrap().r;
        assert_eq!(w[0], 0.0);
        // Violated above → nearest (upper) edge drives.
        let sim_hi = sim_single(0.0, 400.0, CurveId::Rs, 0.6);
        let (t, w) = build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim_hi)).unwrap().r;
        assert!((t[0] - 0.55).abs() < 1e-14);
        assert!((w[0] - 100.0).abs() < 1e-9);
        // Violated below → lower edge drives.
        let sim_lo = sim_single(0.0, 400.0, CurveId::Rs, 0.3);
        let (t, w) = build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim_lo)).unwrap().r;
        assert!((t[0] - 0.45).abs() < 1e-14);
        assert!((w[0] - 100.0).abs() < 1e-9);
        // No sim yet → conservative exact fold at the centre.
        let (t, w) = build_needle_targets(&spec, &[0.0], &[400.0], None).unwrap().r;
        assert!((t[0] - 0.5).abs() < 1e-14);
        assert!((w[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn targets_builder_centerband_reduced_weight_inside() {
        // Centre 0.5 (nf=1), tol 0.1, band 0.05: inside w = 1/0.05² = 400.
        let spec = spec_banded(0.0, CurveId::Rs, &[400.0], &[0.5], &[0.1],
            &[0.05], 1.0, ConstraintKind::CenterBand);
        let sim_in = sim_single(0.0, 400.0, CurveId::Rs, 0.52);
        let (t, w) = build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim_in)).unwrap().r;
        assert!((t[0] - 0.5).abs() < 1e-14);
        assert!((w[0] - 400.0).abs() < 1e-9);
        // Outside → upper edge with the outer weight.
        let sim_hi = sim_single(0.0, 400.0, CurveId::Rs, 0.6);
        let (t, w) = build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim_hi)).unwrap().r;
        assert!((t[0] - 0.55).abs() < 1e-14);
        assert!((w[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn targets_builder_transmission_folds_to_t_bucket() {
        // T demand folds exactly like R but lands in the t bucket.
        let spec = spec_single(0.0, CurveId::Ts, &[400.0], &[0.3], &[0.1], 1.0,
            ConstraintKind::Exact);
        let nt = build_needle_targets(&spec, &[0.0], &[400.0], None).unwrap();
        assert!((nt.t.0[0] - 0.3).abs() < 1e-14);
        assert!((nt.t.1[0] - 100.0).abs() < 1e-9);
        assert_eq!(nt.r.1[0], 0.0);
        assert_eq!(nt.a.1[0], 0.0);
    }

    #[test]
    fn targets_builder_absorption_folds_from_companions() {
        // A demand 0.2 (nf=1), tol 0.1; sim R=0.6/T=0.3 → A=0.1, shortfall
        // of 0.1 → active with the R-style weight at the centre.
        let spec = spec_single(0.0, CurveId::As, &[400.0], &[0.2], &[0.1], 1.0,
            ConstraintKind::Exact);
        let mut sim = sim_single(0.0, 400.0, CurveId::Rs, 0.6);
        sim.curves[CurveId::Ts.index()] = Some(vec![0.3f64].into());
        let nt = build_needle_targets(&spec, &[0.0], &[400.0], Some(&sim)).unwrap();
        assert!((nt.a.0[0] - 0.2).abs() < 1e-14);
        assert!((nt.a.1[0] - 100.0).abs() < 1e-9);
        assert_eq!(nt.r.1[0], 0.0);
        // Satisfied Above on A → masked.
        let spec2 = spec_single(0.0, CurveId::As, &[400.0], &[0.05], &[0.1], 1.0,
            ConstraintKind::Above);
        let nt2 = build_needle_targets(&spec2, &[0.0], &[400.0], Some(&sim)).unwrap();
        assert_eq!(nt2.a.1[0], 0.0);
    }

    #[test]
    fn targets_builder_back_buckets() {
        // RBs demand folds to the rb bucket (mirrors the R path).
        let spec = spec_single(0.0, CurveId::RBs, &[400.0], &[0.4], &[0.1], 1.0,
            ConstraintKind::Exact);
        let nt = build_needle_targets(&spec, &[0.0], &[400.0], None).unwrap();
        assert!((nt.rb.0[0] - 0.4).abs() < 1e-14);
        assert!((nt.rb.1[0] - 100.0).abs() < 1e-9);
        assert_eq!(nt.r.1[0], 0.0);
    }

    #[test]
    fn targets_builder_phase_channel_pairs() {
        // Phase demand on Rs folds to phi[0] with the outer weight;
        // on Ts to phi[2]. No sim → conservative exact fold at centre.
        for (curve, ch) in [(CurveId::Rs, 0), (CurveId::Ts, 2)] {
            let spec = spec_single_phase(0.0, curve, &[400.0], &[1.0], &[0.1], 1.0,
                ConstraintKind::Exact);
            let nt = build_needle_targets(&spec, &[0.0], &[400.0], None).unwrap();
            assert!((nt.phi[ch].0[0] - 1.0).abs() < 1e-14);
            assert!((nt.phi[ch].1[0] - 100.0).abs() < 1e-9);
            for (i, pair) in nt.phi.iter().enumerate() {
                if i != ch {
                    assert_eq!(pair.1[0], 0.0);
                }
            }
        }
        // Phase demand with a log transform cannot fold.
        let mut spec_log = MeritSpec::new();
        let k = spec_log.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        let mut tgt = entry_for_phase(k as u32);
        tgt.transform = SimTransform::Log;
        spec_log.add_target(tgt).unwrap();
        assert!(build_needle_targets(&spec_log, &[0.0], &[400.0], None).is_err());
    }

    #[test]
    fn targets_builder_rejects_log() {
        // Log transform cannot fold → must be rejected.
        let mut spec_log = MeritSpec::new();
        let k = spec_log.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec_log.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![400.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Log,
            norm_factor: 1.0,
            normalized_targets: vec![1.0].into(),
            tolerances: vec![0.01].into(),
            band: vec![].into(),
            phase: false,
        })
        .unwrap();
        assert!(build_needle_targets(&spec_log, &[0.0], &[400.0], None).is_err());
    }

    #[test]
    fn interp_clamped_edges() {
        assert_eq!(super::interp_clamped(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], 0.5), 10.0);
        assert_eq!(super::interp_clamped(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], 9.0), 30.0);
        assert!((super::interp_clamped(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], 2.5) - 25.0).abs() < 1e-14);
    }

    #[test]
    fn invalid_inputs_rejected() {
        let (cache, d, rt, rv) = test_stack_arrays();
        let targets = vec![0.0f64; NW];
        let weights = vec![1.0f64; NW];
        let needle_n = nk_const(1.46);
        let mut inp = pass_input(&cache, &d, &rt, &rv, &targets, &weights, &needle_n);
        inp.calc_s = false;
        inp.calc_p = false;
        let films = vec![LayerSpec::constant("H", 2.35, 0.0, 40.0, NW)];
        assert!(run_needle_pass(&inp, &films, 2.0).is_err());

        let inp2 = pass_input(&cache, &d, &rt, &rv, &[0.0], &weights, &needle_n);
        assert!(run_needle_pass(&inp2, &films, 2.0).is_err()); // bad targets len
    }
}
