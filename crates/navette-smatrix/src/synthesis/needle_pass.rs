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
use crate::synthesis::merit::{CurveId, MeritKey, MeritSpec, MeritTarget, SimCurves};

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
    /// Exact `dM/dD` correction per phase channel for differential demands:
    /// inserting thickness δ anywhere grows the reference by the same δ,
    /// so `dM/dδ` picks up `Σ −2·kz·w·(s−rt)` over folded points (kz =
    /// `passes·reference_wavenumber`, `w`/`rt` the emitted pair). Uniform in
    /// z — subtract from the assembled `P_PHI` gradient; `argmax` (the site)
    /// is provably unaffected, only predicted-gain bookkeeping shifts.
    /// Zero for absolute-phase (and all non-phase) demand sets.
    pub phi_gain_shift: [f64; 4],
}

/// Demand bucket: absorption derives from companions, intensities fold to
/// their own quantity, phase demands to their element pair.
#[derive(Clone, Copy)]
enum BucketKind {
    Pair(usize),
    Phi(usize),
}

/// Operating-point rows for activation, sliced to the demand angle.
enum OpRows<'a> {
    Intensity(&'a [f64]),
    Absorption(&'a [f64], &'a [f64]),
    Phase(&'a [Complex64]),
}

/// Operating-point sample in demand space (Δφ with the reference subtracted
/// for PD demands). Shared by the pointwise and integral folds.
fn sample_op_value(
    rows: &OpRows,
    sim: &SimCurves,
    twl_i: f64,
    t: &MeritTarget,
    key: &MeritKey,
    n_inc: Option<f64>,
) -> f64 {
    match rows {
        OpRows::Intensity(r) => interp_clamped(&sim.wavelengths, r, twl_i),
        OpRows::Absorption(r, t2) => {
            1.0 - interp_clamped(&sim.wavelengths, r, twl_i)
                - interp_clamped(&sim.wavelengths, t2, twl_i)
        },
        OpRows::Phase(c) => {
            let mut a = cinterp_clamped(&sim.wavelengths, c, twl_i).arg();
            if let Some(passes) = t.differential_passes {
                a -= crate::optics_core::reference_phase(
                    twl_i,
                    n_inc.unwrap_or(1.0),
                    key.angle,
                    sim.total_d,
                    passes,
                );
            }
            a
        },
    }
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
    let mut phi_gain_shift = [0.0f64; 4];

    for t in spec.targets() {
        let key = &spec.keys()[t.key_idx as usize];
        // (index-based so the loop holds no &mut across iterations)
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
        // Loop-invariant: the demand's incidence medium (None without sim).
        // Hoisted — the per-point loop below must stay lean for
        // intensity-only demand sets.
        let n_inc = current_sim.map(|sim| {
            if key.curve.is_back() { sim.n_back_re } else { sim.n_front_re }
        });
        // Integral demands fold the MEAN (own two-pass block below): the
        // mean-form merit is non-diagonal, so the fold matches its UNIFORM
        // gradient (exact at the op point, values up to dropped constants —
        // same loss class as the +1/overlap terms). Pointwise continues.
        if t.integral {
            fold_integral_demand(t, key, twl, current_sim, &op, n_inc,
                row, wavelengths, nw, bucket, &mut buckets, &mut phi,
                &mut phi_gain_shift)?;
            continue;
        }
        for &twl_i in twl.iter() {
            // Interpolate normalized target, tolerance and band half-width
            // onto this solver wavelength (edge-clamped; ascending grids).
            let tgt_norm = interp_clamped(twl, &t.normalized_targets, twl_i);
            let tol = interp_clamped(twl, &t.tolerances, twl_i).max(1e-300);
            let bw = if t.band.is_empty() { 0.0 } else { interp_clamped(twl, &t.band, twl_i).max(0.0) };
            let raw_target = tgt_norm / t.norm_factor;
            let nf2 = t.norm_factor * t.norm_factor;

            // Operating-point sim value in demand space (None before the
            // first simulation). Absorption samples A = 1 − R − T from
            // the companion rows; phase samples arg() of the complex row
            // minus the differential reference (`PDts`/`PDtp`) when set.
            // Residuals wrap exactly when the evaluator would (Phase mode).
            let wrap_phase = t.phase
                && t.transform == crate::synthesis::merit::SimTransform::Phase;
            let s_op: Option<f64> = match (current_sim, &op) {
                (Some(sim), Some(rows)) => {
                    Some(sample_op_value(rows, sim, twl_i, t, key, n_inc))
                },
                _ => None,
            };
            let d_opt: Option<f64> = s_op.map(|v| {
                let mut d = v * t.norm_factor - tgt_norm;
                if wrap_phase {
                    d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
                }
                d
            });

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
            // User weight + count normalization, applied once at emission:
            // folded merit scales by weight/count exactly, and the gain
            // shift below inherits the scaled `w` consistently. Defaults
            // (1.0/None) are the identity — legacy folds bit-safe.
            let w = w * t.weight / t.count_norm.unwrap_or(1.0);

            let k = row * nw + solver_wav_index(wavelengths, twl_i);
            match bucket {
                BucketKind::Pair(i) => {
                    buckets[i].1[k] += w;
                    buckets[i].0[k] += w * rt;
                },
                BucketKind::Phi(ch) => {
                    phi[ch].1[k] += w;
                    phi[ch].0[k] += w * rt;
                    // Differential gain shift, exact at the op point:
                    // dM/dD contribution −2·kz·w·(s−rt) with the emitted
                    // pair (holds for every kind arm — verified per-arm in
                    // the `phi_gain_shift_matches_fd` test). Skipped points
                    // (folded None) and the no-sim arm (s unknown) add 0.
                    if let (Some(passes), Some(s), Some(n)) =
                        (t.differential_passes, s_op, n_inc)
                    {
                        let kz = passes
                            * crate::optics_core::reference_wavenumber(
                                twl_i, n, key.angle,
                            );
                        phi_gain_shift[ch] += -2.0 * kz * w * (s - rt);
                    }
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
    Ok(NeedleTargets { r, t, a, rb, tb, ab, phi, phi_gain_shift })
}

/// How an integral demand emits after mean activation.
enum IntGap {
    Skip,
    /// Centre form at the op mean (Exact / violated one-sided).
    CentreMean(f64),
    /// Reduced-weight centre form (CenterBand inside).
    InsideMean(f64),
    /// Edge form at the op mean (Range / CenterBand outside).
    EdgeMean(f64, f64),
    /// No sim yet: per-point raw targets, Exact-like weight.
    Conservative,
}

/// Fold ONE integral demand (see the call site for the derivation): the
/// mean-form merit `W·(m−T)²` is non-diagonal, so the fold matches its
/// UNIFORM gradient — exact at the operating point — with per-point pairs
/// `w_i = W_eff/N²`, `t_i = s_i − N·G` (`G` = raw gap to the centre/edge
/// mean, `N` = demand count). Values differ by dropped constants (same
/// loss class as the +1/overlap terms); gradients — what the needle
/// consumes — are exact, and overlapping integral frames superpose
/// gradient-exactly. Without sim, conservative centre form.
#[allow(clippy::too_many_arguments)]
fn fold_integral_demand(
    t: &MeritTarget,
    key: &MeritKey,
    twl: &[f64],
    current_sim: Option<&SimCurves>,
    op: &Option<OpRows>,
    n_inc: Option<f64>,
    row: usize,
    wavelengths: &[f64],
    nw: usize,
    bucket: BucketKind,
    buckets: &mut [(Vec<f64>, Vec<f64>); 6],
    phi: &mut [(Vec<f64>, Vec<f64>); 4],
    phi_gain_shift: &mut [f64; 4],
) -> Result<(), String> {
    use crate::synthesis::merit::{ConstraintKind, SimTransform};
    let n = twl.len() as f64;
    let nf = t.norm_factor;
    let wrap_phase =
        t.phase && t.transform == SimTransform::Phase;
    // Pass 1: means of normalized target / tol / band.
    let mut t_bar = 0.0;
    let mut tol_sum = 0.0;
    let mut bw_sum = 0.0;
    for &twl_i in twl {
        t_bar += interp_clamped(twl, &t.normalized_targets, twl_i);
        tol_sum += interp_clamped(twl, &t.tolerances, twl_i).max(1e-300);
        bw_sum += if t.band.is_empty() {
            0.0
        } else {
            interp_clamped(twl, &t.band, twl_i).max(0.0)
        };
    }
    t_bar /= n;
    let tol_bar = (tol_sum / n).max(1e-300);
    let bw_bar = bw_sum / n;
    let t_raw = t_bar / nf;
    // Pass 2: op means (None before the first simulation).
    let mut s_vals: Vec<f64> = Vec::new();
    let opm: Option<(f64, f64)> = match (current_sim, op) {
        (Some(sim), Some(rows)) => {
            let mut m = 0.0;
            let mut d = 0.0;
            s_vals.reserve(twl.len());
            for &twl_i in twl {
                let s = sample_op_value(rows, sim, twl_i, t, key, n_inc);
                s_vals.push(s);
                m += s;
                let tgt_j = interp_clamped(twl, &t.normalized_targets, twl_i);
                let mut dj = s * nf - tgt_j;
                if wrap_phase {
                    dj -= std::f64::consts::TAU * (dj / std::f64::consts::TAU).round();
                }
                d += dj;
            }
            Some((m / n, d / n))
        },
        _ => None,
    };
    let w_centre = t.weight * nf * nf / (tol_bar * tol_bar);
    // Kind activation on the mean residual R = d_bar / tol_bar.
    let gap = match t.kind {
        ConstraintKind::Exact => match opm {
            Some((m, _)) => IntGap::CentreMean(m),
            None => IntGap::Conservative,
        },
        ConstraintKind::Above => match opm {
            Some((m, d_bar)) if d_bar / tol_bar >= 0.0 => IntGap::Skip,
            Some((m, _)) => IntGap::CentreMean(m),
            None => IntGap::Conservative,
        },
        ConstraintKind::Below => match opm {
            Some((m, d_bar)) if d_bar / tol_bar <= 0.0 => IntGap::Skip,
            Some((m, _)) => IntGap::CentreMean(m),
            None => IntGap::Conservative,
        },
        ConstraintKind::Range => {
            let bw_eff = if bw_bar <= 0.0 { tol_bar } else { bw_bar };
            match opm {
                None => IntGap::Conservative,
                Some((m, d_bar)) if d_bar.abs() <= bw_eff => IntGap::Skip,
                Some((m, _)) => IntGap::EdgeMean(m, bw_eff),
            }
        },
        ConstraintKind::CenterBand => {
            if bw_bar <= 0.0 {
                // Degrades to Exact (mirrors the pointwise rule).
                match opm {
                    Some((m, _)) => IntGap::CentreMean(m),
                    None => IntGap::Conservative,
                }
            } else {
                match opm {
                    None => IntGap::Conservative,
                    Some((m, d_bar)) if d_bar.abs() <= bw_bar => IntGap::InsideMean(m),
                    Some((m, _)) => IntGap::EdgeMean(m, bw_bar),
                }
            }
        },
    };
    if matches!(gap, IntGap::Skip) {
        return Ok(());
    }
    let w_inside = t.weight * nf * nf / (bw_bar * bw_bar);
    // d_bar for the edge gap (opm is Some whenever we emit edge forms).
    let d_bar = opm.map(|(_, d)| d).unwrap_or(0.0);
    for (j, &twl_i) in twl.iter().enumerate() {
        let (rt_j, w_j) = match &gap {
            IntGap::Skip => unreachable!("returned above"),
            IntGap::Conservative => {
                let tj = interp_clamped(twl, &t.normalized_targets, twl_i) / nf;
                (tj, w_centre / (n * n))
            },
            IntGap::CentreMean(m) => (s_vals[j] - n * (m - t_raw), w_centre / (n * n)),
            IntGap::InsideMean(m) => (s_vals[j] - n * (m - t_raw), w_inside / (n * n)),
            IntGap::EdgeMean(_, bw_eff) => {
                let g = d_bar.signum() * (d_bar.abs() - bw_eff) / nf;
                (s_vals[j] - n * g, w_centre / (n * n))
            },
        };
        let k = row * nw + solver_wav_index(wavelengths, twl_i);
        match bucket {
            BucketKind::Pair(i) => {
                buckets[i].1[k] += w_j;
                buckets[i].0[k] += w_j * rt_j;
            },
            BucketKind::Phi(ch) => {
                phi[ch].1[k] += w_j;
                phi[ch].0[k] += w_j * rt_j;
                // Gain shift with the EMITTED pair (exact dM/dD at the op
                // point — the uniform-gradient construction makes the
                // per-emission formula land on −2·W·(m−T)·mean(kz)).
                // No-sim arm has no op value → contributes 0.
                if let (Some(passes), Some(nn), Some(&s)) =
                    (t.differential_passes, n_inc, s_vals.get(j))
                {
                    let kz = passes
                        * crate::optics_core::reference_wavenumber(twl_i, nn, key.angle);
                    phi_gain_shift[ch] += -2.0 * kz * w_j * (s - rt_j);
                }
            },
        }
    }
    Ok(())
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
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
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
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
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
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
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
            ..Default::default()
        };
        sim.curves[curve.index()] = Some(vec![val].into());
        sim
    }

    #[test]
    fn phi_gain_shift_matches_fd() {
        // Ts complex phase flat 0.3, D = 100 nm air, θ = 0°: kz(400) = 2π/400,
        // kz(500) = 2π/500. PD targets sit 0.01 above Δφ (tol 0.05) →
        // r = −0.2/point. dM/dD = Σ 2·r·(−kz/tol) must equal the fold's
        // phi_gain_shift[2] (Ts → channel 2) and a D finite difference.
        use std::f64::consts::{PI, TAU};
        let ref_of = |wl: f64| TAU * 100.0 / wl;
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        let tgt: Vec<f64> = [400.0, 500.0].iter().map(|&w| 0.3 - ref_of(w) + 0.01).collect();
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![400.0, 500.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Phase,
            norm_factor: 1.0,
            normalized_targets: tgt.into(),
            tolerances: vec![0.05, 0.05].into(),
            band: vec![].into(),
            phase: true,
            differential_passes: Some(1.0),
            integral: false,
            weight: 1.0,
            count_norm: None,
        })
        .unwrap();
        let mk_sim = |d: f64| {
            let mut sim = SimCurves {
                angles: vec![0.0].into(),
                wavelengths: vec![400.0, 500.0].into(),
                curves: [None, None, None, None, None, None, None, None, None],
                back: [None, None, None, None, None, None],
                cplx: [None, None, None, None, None, None],
                cplx_back: [None, None, None, None],
                total_d: d,
                n_front_re: 1.0,
                n_back_re: 1.0,
            };
            sim.cplx[3] = Some(vec![
                Complex64::from_polar(0.7, 0.3),
                Complex64::from_polar(0.7, 0.3),
            ].into());
            sim
        };
        let angles = [0.0];
        let wavs = [400.0, 500.0];
        let nt = build_needle_targets(&spec, &angles, &wavs, Some(&mk_sim(100.0))).unwrap();
        // Hand calc: 2·(−0.2)·(−kz/0.05) per point = 8·kz summed.
        let expect = 8.0 * (TAU / 400.0 + TAU / 500.0);
        assert!((nt.phi_gain_shift[2] - expect).abs() < 1e-9,
            "shift={} expect={}", nt.phi_gain_shift[2], expect);
        assert!(nt.phi_gain_shift[0] == 0.0 && nt.phi_gain_shift[1] == 0.0
            && nt.phi_gain_shift[3] == 0.0);
        // Finite difference of the true merit over D.
        let h = 1e-3;
        let m_hi = spec.merit(&mk_sim(100.0 + h), 1e6);
        let m_lo = spec.merit(&mk_sim(100.0 - h), 1e6);
        let fd = (m_hi - m_lo) / (2.0 * h);
        assert!((fd - expect).abs() < 1e-6, "fd={fd} expect={expect}");
        assert!((nt.phi_gain_shift[2] - fd).abs() < 1e-6);
        // Sanity: merit itself is 2·(−0.2)² = 0.08, and π-scale check
        // ref(400) = π/2 exactly.
        assert!((spec.merit(&mk_sim(100.0), 1e6) - 0.08).abs() < 1e-12);
        assert!((ref_of(400.0) - PI / 2.0).abs() < 1e-15);
    }

    #[test]
    fn fold_applies_weight_and_count() {
        // Exact Rs demand, nf = 1, tol 0.1 → base folded w = 100 at 400.
        // weight 3 + count 2 → 150; raw target untouched.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![400.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Linear,
            norm_factor: 1.0,
            normalized_targets: vec![0.5].into(),
            tolerances: vec![0.1].into(),
            band: vec![].into(),
            phase: false,
            differential_passes: None,
            integral: false,
            weight: 3.0,
            count_norm: Some(2.0),
        })
        .unwrap();
        let nt = build_needle_targets(&spec, &[0.0], &[400.0], None).unwrap();
        assert!((nt.r.1[0] - 150.0).abs() < 1e-9, "w={}", nt.r.1[0]);
        assert!((nt.r.0[0] - 0.5).abs() < 1e-14);
        // Gain shift inherits the same factor (PD demand, weight 2 →
        // shift doubles vs the unweighted `phi_gain_shift_matches_fd`).
        use std::f64::consts::TAU;
        let ref_of = |wl: f64| TAU * 100.0 / wl;
        let mut pspec = MeritSpec::new();
        let pk = pspec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        let tgt: Vec<f64> = [400.0].iter().map(|&w| 0.3 - ref_of(w) + 0.01).collect();
        pspec.add_target(MeritTarget {
            key_idx: pk as u32,
            wavelengths: vec![400.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Phase,
            norm_factor: 1.0,
            normalized_targets: tgt.into(),
            tolerances: vec![0.05].into(),
            band: vec![].into(),
            phase: true,
            differential_passes: Some(1.0),
            integral: false,
            weight: 2.0,
            count_norm: None,
        })
        .unwrap();
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: vec![400.0].into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            total_d: 100.0,
            n_front_re: 1.0,
            n_back_re: 1.0,
        };
        sim.cplx[3] = Some(vec![Complex64::from_polar(0.7, 0.3)].into());
        let pnt = build_needle_targets(&pspec, &[0.0], &[400.0], Some(&sim)).unwrap();
        // Hand: −2·kz·w·Δ with w = 2/0.05² = 800, Δ = −0.01.
        let expect = -2.0 * (TAU / 400.0) * 800.0 * (-0.01);
        assert!((pnt.phi_gain_shift[2] - expect).abs() < 1e-9,
            "shift={} expect={}", pnt.phi_gain_shift[2], expect);
    }

    fn spec_integral(angle: f64, kind: ConstraintKind) -> MeritSpec {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle, curve: CurveId::Rs });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![400.0, 500.0, 600.0].into(),
            kind,
            transform: SimTransform::Linear,
            norm_factor: 1.0,
            normalized_targets: vec![0.5, 0.5, 0.5].into(),
            tolerances: vec![0.1, 0.1, 0.1].into(),
            band: vec![].into(),
            phase: false,
            differential_passes: None,
            weight: 1.0,
            count_norm: None,
            integral: true,
        })
        .unwrap();
        spec
    }

    fn sim_rs(vals: [f64; 3]) -> SimCurves {
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: vec![400.0, 500.0, 600.0].into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            ..Default::default()
        };
        sim.curves[CurveId::Rs.index()] = Some(vec![vals[0], vals[1], vals[2]].into());
        sim
    }

    #[test]
    fn integral_fold_uniform_gradient() {
        // Exact integral, sim [0.7, 0.6, 0.5]: m = 0.6, T = 0.5, N = 3,
        // W = 1/0.01 = 100. True per-point gradient g = 2·W·(m−T)/N =
        // 200·0.1/3 ≈ 6.6667 — UNIFORM across the band. Folded pairs must
        // reproduce exactly that: 2·w_i·(s_i−t_i) == g for every i.
        let spec = spec_integral(0.0, ConstraintKind::Exact);
        let sim = sim_rs([0.7, 0.6, 0.5]);
        let nt = build_needle_targets(&spec, &[0.0], &[400.0, 500.0, 600.0],
            Some(&sim)).unwrap();
        let g = 2.0 * 100.0 * 0.1 / 3.0;
        // Direct: folded w_i = W/N² = 100/9, t_i = s_i − N·G = s_i − 0.3.
        for i in 0..3 {
            assert!((nt.r.1[i] - 100.0 / 9.0).abs() < 1e-9, "w[{}]={}", i, nt.r.1[i]);
            let expect_t = [0.7, 0.6, 0.5][i] - 0.3;
            assert!((nt.r.0[i] - expect_t).abs() < 1e-12, "t[{}]={}", i, nt.r.0[i]);
            // Per-point folded gradient 2·w·(s−t) equals the uniform g.
            let gi = 2.0 * nt.r.1[i] * ([0.7, 0.6, 0.5][i] - nt.r.0[i]);
            assert!((gi - g).abs() < 1e-9, "g[{i}]={gi} expect={g}");
        }
        // Uniform-shift FD: dM/dε for s→s+ε must equal Σ folded gradients.
        let h = 1e-6;
        let m_hi = spec.merit(&sim_rs([0.7 + h, 0.6 + h, 0.5 + h]), 1e6);
        let m_lo = spec.merit(&sim_rs([0.7 - h, 0.6 - h, 0.5 - h]), 1e6);
        let fd = (m_hi - m_lo) / (2.0 * h);
        assert!((fd - 3.0 * g).abs() < 1e-6, "fd={fd} expect={}", 3.0 * g);
    }

    #[test]
    fn integral_range_skips_inband_mean() {
        // Range integral, band 0.05: mean diff 0.04 → all weights zero.
        let mut spec = spec_integral(0.0, ConstraintKind::Range);
        // (spec_integral has empty band → bare-r fallback ±tol = ±0.1;
        // mean diff 0.04 is inside → skip.)
        let sim = sim_rs([0.54, 0.54, 0.54]);
        let nt = build_needle_targets(&spec, &[0.0], &[400.0, 500.0, 600.0],
            Some(&sim)).unwrap();
        assert!(nt.r.1.iter().all(|&w| w == 0.0));
        // Mean diff 0.2 → violated: uniform edge pairs, w = 100/9 each.
        let sim2 = sim_rs([0.7, 0.7, 0.7]);
        let nt2 = build_needle_targets(&spec, &[0.0], &[400.0, 500.0, 600.0],
            Some(&sim2)).unwrap();
        for i in 0..3 {
            assert!((nt2.r.1[i] - 100.0 / 9.0).abs() < 1e-9);
            // edge gap G = (0.2−0.1)/1 = 0.1 → t = 0.7 − 0.3 = 0.4.
            assert!((nt2.r.0[i] - 0.4).abs() < 1e-12, "t[{}]={}", i, nt2.r.0[i]);
        }
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
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
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
            ..Default::default()
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
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
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
