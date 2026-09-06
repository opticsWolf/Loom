//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::cycle — one needle pass: repeated analytic insertions.
//!
//! Port of NeedleSynthesizer.run() with compute_p_function replaced by the
//! analytic sweep (needle_pass). The Python convergence test measured the
//! MF drop from inserting a 1 nm TEST needle:
//!     improvement_py = MF_now − MF(test needle, 1 nm) ≈ −P(z*)·1 nm
//! (first-order, since P = dF/dδ). We generalize to the seed thickness:
//!     predicted_improvement = −P_best · δ_seed
//! and stop when it falls below `convergence_threshold`. Thresholds are in
//! merit-units-per-nm-of-seed — recalibrate when porting configs.

use std::collections::HashMap;
use std::sync::Arc;

use num_complex::Complex64;

use crate::synthesis::context::DesignContext;
use crate::synthesis::needle_pass::{
    build_scan_sites, run_needle_pass, NeedlePassInput, NeedleTargets,
};
use crate::synthesis::pipeline::SpectralInputs;
use crate::synthesis::structure::{DesignStack, LayerSpec};

/// Knobs for the inner insertion loop (Python `NeedleConfig` subset).
#[derive(Clone, Debug)]
pub struct NeedleCycleConfig {
    /// Max insertions per pass (pipeline forwards `needles_per_cycle`).
    pub max_needles: usize,
    /// Stop when predicted improvement drops below this (merit units).
    pub convergence_threshold: f64,
    /// Seed layer thickness (nm).
    pub needle_seed_thickness_nm: f64,
    /// Scan resolution (nm).
    pub scan_step_nm: f64,
    /// Re-fold needle targets against the live sim each cycle (op-point
    /// masking for one-sided/banded/integral kinds + live PD gain shift).
    /// `false` keeps the static conservative fold (A/B benchmark switch).
    pub refold_per_cycle: bool,
}

impl Default for NeedleCycleConfig {
    fn default() -> Self {
        NeedleCycleConfig {
            max_needles: 10,
            convergence_threshold: 1e-4,
            needle_seed_thickness_nm: 5.0,
            scan_step_nm: 2.0,
            refold_per_cycle: true,
        }
    }
}

/// Record of one insertion cycle — mirrors Python `NeedleCycleResult`
/// with the scan-MF replaced by its analytic analog.
#[derive(Clone, Debug)]
pub struct NeedleCycleResult {
    pub cycle: usize,
    pub merit_before: f64,
    pub merit_after: f64,
    /// Most-negative P value at the chosen site (dF/dδ there).
    pub best_p: Option<f64>,
    /// −best_p · seed_thickness — the recalibrated convergence metric.
    pub predicted_improvement: Option<f64>,
    pub layer_count: usize,
    pub insertion: Option<Insertion>,
}

#[derive(Clone, Debug)]
pub struct Insertion {
    pub film_idx: usize,
    pub depth_into_layer_nm: f64,
    pub material: Arc<str>,
}

/// Contrast-material table: film material name → needle LayerSpec template.
/// (nk arrays must be evaluated on the simulation grid; thickness/flags of
/// the template are ignored — the seed is built fresh.)
pub type ContrastMap = HashMap<Arc<str>, LayerSpec>;

/// Run one full needle pass on `stack`.
///
/// Mirrors `synth.run(max_needles = cfg.max_needles)`:
/// initial optimization → [re-fold → scan → select → insert → optimize]×N.
///
/// The fold refreshes against the live sim each cycle (when
/// `cfg.refold_per_cycle` and the context simulates): one-sided/banded
/// demands mask at the operating point and PD gain shifts go live, so
/// insertion tracks the merit the optimizer actually sees. Any simulate
/// or fold failure keeps the previous fold (mock contexts without a
/// solver run the static conservative fold throughout).
pub fn run_needle_cycles<C: DesignContext + ?Sized>(
    ctx: &mut C,
    stack: &mut DesignStack,
    spectral: &SpectralInputs,
    contrast: &ContrastMap,
    cfg: &NeedleCycleConfig,
) -> Result<Vec<NeedleCycleResult>, String> {
    use crate::synthesis::needle_pass::build_needle_targets;

    let wavls = &spectral.wavls;
    let sin_theta = &spectral.sin_theta;
    let mut fold = spectral.fold.clone();
    let mut history = Vec::new();

    // Initial optimization.
    ctx.optimize_thicknesses(stack)?;

    for cycle in 0..cfg.max_needles {
        // Refresh the fold against the live operating point.
        if cfg.refold_per_cycle {
            if let Ok(sim) = ctx.simulate(stack) {
                if let Ok(f) = build_needle_targets(
                    &spectral.spec,
                    &spectral.angles_deg,
                    &spectral.wavls,
                    Some(&sim),
                ) {
                    fold = f;
                }
            }
        }
        // 1. Build candidate sites restricted to films whose material has a
        //    contrast entry (Python skips layers without a mapping) AND
        //    that are needle hosts. The flag check is load-bearing:
        //    interface slices and pinned graded rows share their carrier's
        //    material (hence a contrast entry) but must never host seeds.
        let sites = build_scan_sites(stack.films(), cfg.scan_step_nm);
        let sites: Vec<_> = sites
            .into_iter()
            .filter(|s| {
                stack
                    .films()
                    .get(s.film_idx)
                    .map(|l| l.needle && contrast.contains_key(&l.material))
                    .unwrap_or(false)
            })
            .collect();
        if sites.is_empty() {
            break; // "Stack too thin for needle insertion."
        }

        // 2. Analytic sweep (both polarizations summed) per DISTINCT contrast
        //    material among admissible hosts; global best kept across sweeps.
        let sa = stack.solver_arrays();

        // Distinct contrast materials among admissible hosts.
        let mats: Vec<&LayerSpec> = {
            let mut seen: Vec<Arc<str>> = Vec::new();
            let mut out = Vec::new();
            for l in stack.films() {
                if contrast.contains_key(&l.material) && !seen.contains(&l.material) {
                    seen.push(l.material.clone());
                    out.push(contrast.get(&l.material).unwrap());
                }
            }
            out
        };

        let mut best: Option<(Insertion, f64)> = None;
        for mat in &mats {
            let input = NeedlePassInput {
                n_stack_cache: &sa.n_stack_cache,
                thicknesses: &sa.thicknesses,
                rough_types: &sa.rough_types,
                rough_vals: &sa.rough_vals,
                n_layers: sa.n_layers as usize,
                wavls: wavls.as_slice(),
                sin_theta: sin_theta.as_slice(),
                fold: &fold,
                needle_n_per_wav: &mat.nk,
                start_idx: 0,
                end_idx: (sa.n_layers - 1) as usize,
                calc_s: true,
                calc_p: true,
            };
            let res = run_needle_pass(&input, stack.films(), cfg.scan_step_nm)?;
            if let Some((site, p)) = res.best() {
                let better = match &best {
                    None => true,
                    Some((_, bp)) => p < *bp,
                };
                if better {
                    best = Some((
                        Insertion {
                            film_idx: site.film_idx,
                            depth_into_layer_nm: site.depth_into_layer_nm,
                            material: mat.material.clone(),
                        },
                        p,
                    ));
                }
            }
        }

        // 3. Convergence check on the predicted improvement.
        let current_mf = ctx.evaluate_merit(stack)?;
        let result = match best {
            None => {
                // No improving site anywhere.
                history.push(NeedleCycleResult {
                    cycle: cycle + 1,
                    merit_before: current_mf,
                    merit_after: current_mf,
                    best_p: None,
                    predicted_improvement: None,
                    layer_count: stack.films().len(),
                    insertion: None,
                });
                break;
            }
            Some((ins, p)) => {
                // Inserting a seed of thickness δ also grows the
                // equivalent-medium reference by δ, so the true slope is
                // P(z*) + Σ gain_shift (0 for absolute-phase demand sets;
                // the site itself is unaffected, only this bookkeeping).
                let gtot: f64 = fold.phi_gain_shift.iter().sum();
                let predicted = -(p + gtot) * cfg.needle_seed_thickness_nm;
                if predicted < cfg.convergence_threshold {
                    history.push(NeedleCycleResult {
                        cycle: cycle + 1,
                        merit_before: current_mf,
                        merit_after: current_mf,
                        best_p: Some(p),
                        predicted_improvement: Some(predicted),
                        layer_count: stack.films().len(),
                        insertion: None,
                    });
                    break; // "Convergence reached — stopping."
                }

                // 4. Insert the seed (split host, Python `_insert_needle`).
                let seed_nk: Arc<[Complex64]> = contrast
                    .get(&stack.films()[ins.film_idx].material)
                    .map(|m| m.nk.clone())
                    .unwrap_or_else(|| vec![Complex64::new(1.0, 0.0); wavls.len()].into());
                let seed = LayerSpec {
                    material: ins.material.clone(),
                    nk: seed_nk,
                    d_nm: cfg.needle_seed_thickness_nm,
                    coherent: true,
                    rough_type: 0,
                    rough_val: 0.0,
                    optimize: true,
                    needle: true,
                };
                stack.insert_needle_seed(ins.film_idx, ins.depth_into_layer_nm, seed)?;

                // 5. Re-optimize.
                let new_mf = ctx.optimize_thicknesses(stack)?;

                NeedleCycleResult {
                    cycle: cycle + 1,
                    merit_before: current_mf,
                    merit_after: new_mf,
                    best_p: Some(p),
                    predicted_improvement: Some(predicted),
                    layer_count: stack.films().len(),
                    insertion: Some(ins),
                }
            }
        };
        history.push(result);
    }

    Ok(history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::merit::{MeritSpec, SimCurves};
    use std::sync::Arc;

    /// Frozen context: merit constant, optimization a no-op (keeps the
    /// analytic AR condition intact so the scan sees exact zeros).
    struct StillCtx;
    impl DesignContext for StillCtx {
        fn evaluate_merit(&self, _s: &DesignStack) -> Result<f64, String> {
            Ok(0.0)
        }
        fn simulate(&self, _s: &DesignStack) -> Result<SimCurves, String> {
            Err("mock context has no simulator".into())
        }
        fn optimize_thicknesses(&mut self, s: &mut DesignStack) -> Result<f64, String> {
            self.evaluate_merit(s)
        }
    }

    /// air | G(200 nm, n = 1.52, host) | glass: all matched, so R is
    /// exactly the single-interface value ((n−1)/(n+1))² ≈ 0.04258 and
    /// T = 1 − R (lossless) at normal incidence.
    /// (A lossless R = 0 point would NOT discriminate: T = 1 − R
    /// identically there, so every intensity gradient vanishes with R's.)
    fn glass_stack() -> (DesignStack, f64) {
        let nw = 1;
        let r0 = ((1.52_f64 - 1.0) / (1.52_f64 + 1.0)).powi(2);
        let mut ambient = LayerSpec::constant("air", 1.0, 0.0, 0.0, nw);
        ambient.optimize = false;
        ambient.needle = false;
        let mut substrate = LayerSpec::constant("sub", 1.52, 0.0, 0.0, nw);
        substrate.optimize = false;
        substrate.needle = false;
        let stack = DesignStack::with_films(
            ambient,
            substrate,
            vec![LayerSpec::constant("G", 1.52, 0.0, 200.0, nw)],
        )
        .unwrap();
        (stack, r0)
    }

    fn fold_with(bucket: &str, target: f64) -> NeedleTargets {
        let zero = || (vec![0.0f64; 1], vec![0.0f64; 1]);
        let demand = || (vec![target; 1], vec![1.0f64; 1]);
        let (r, t) = match bucket {
            "r" => (demand(), zero()),
            "t" => (zero(), demand()),
            _ => unreachable!(),
        };
        NeedleTargets {
            r,
            t,
            a: zero(),
            rb: zero(),
            tb: zero(),
            ab: zero(),
            phi: [zero(), zero(), zero(), zero()],
            phi_gain_shift: [0.0; 4],
        }
    }

    fn contrast_h() -> ContrastMap {
        let mut m = ContrastMap::new();
        m.insert(Arc::from("G"), LayerSpec::constant("H", 2.35, 0.0, 0.0, 1));
        m
    }

    fn cycle_cfg() -> NeedleCycleConfig {
        NeedleCycleConfig {
            max_needles: 2,
            convergence_threshold: 1e-4,
            needle_seed_thickness_nm: 5.0,
            scan_step_nm: 2.0,
            refold_per_cycle: true,
        }
    }

    /// SpectralInputs wrapping a hand-built fold (empty spec — the
    /// StillCtx simulator errors, so the static fold stands).
    fn spectral_of(fold: NeedleTargets) -> SpectralInputs {
        SpectralInputs {
            wavls: vec![1000.0],
            sin_theta: vec![0.0],
            fold,
            spec: MeritSpec::new(),
            angles_deg: vec![0.0],
        }
    }

    /// Non-host rows (needle=false) are never insertion sites even when
    /// their material has a contrast entry (interface slices, pinned
    /// graded spans). The run breaks with the stack untouched.
    #[test]
    fn non_host_rows_never_seed() {
        let nw = 1;
        let mut ambient = LayerSpec::constant("air", 1.0, 0.0, 0.0, nw);
        ambient.optimize = false;
        ambient.needle = false;
        let mut substrate = LayerSpec::constant("sub", 1.52, 0.0, 0.0, nw);
        substrate.optimize = false;
        substrate.needle = false;
        let mut host = LayerSpec::constant("G", 1.52, 0.0, 200.0, nw);
        host.needle = false; // contrast entry exists, host flag refuses
        let mut stack =
            DesignStack::with_films(ambient, substrate, vec![host]).unwrap();
        let mut ctx = StillCtx;
        let spectral = spectral_of(fold_with("t", 2.0));
        let hist = run_needle_cycles(&mut ctx, &mut stack, &spectral, &contrast_h(), &cycle_cfg())
            .unwrap();
        assert!(hist.is_empty());
        assert_eq!(stack.films().len(), 1);
        assert!((stack.films()[0].d_nm - 200.0).abs() < 1e-12);
    }

    #[test]
    fn satisfied_r_demand_inserts_nothing() {
        // R demanded at its exact value → residual ~1e-16 → dust profile,
        // convergence gate stops: no insertion, film count untouched.
        let (mut stack, r0) = glass_stack();
        let mut ctx = StillCtx;
        let spectral = spectral_of(fold_with("r", r0));
        let hist = run_needle_cycles(
            &mut ctx,
            &mut stack,
            &spectral,
            &contrast_h(),
            &cycle_cfg(),
        )
        .unwrap();
        assert_eq!(hist.len(), 1);
        assert!(hist[0].insertion.is_none());
        assert!(hist[0].best_p.map(|p| p > -1e-9).unwrap_or(true));
        assert_eq!(stack.films().len(), 1);
    }

    #[test]
    fn live_refold_masks_satisfied_above_demand() {
        // Glass R ≈ 0.0426 against Above 0.01: satisfied (sim ≥ target)
        // at the operating point. The static conservative fold would
        // insert (Above folds active without a sim); the per-cycle live
        // re-fold masks it, so nothing is inserted. Real solver context.
        use crate::synthesis::evaluator::SmatrixContext;
        use crate::synthesis::merit::{
            ConstraintKind, MeritKey, MeritTarget, SimTransform,
        };
        use crate::synthesis::thick_opt::LmConfig;

        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: crate::synthesis::merit::CurveId::Rs });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![1000.0].into(),
            kind: ConstraintKind::Above,
            transform: SimTransform::Linear,
            norm_factor: 1.0,
            normalized_targets: vec![0.01].into(),
            tolerances: vec![0.05].into(),
            band: vec![].into(),
            phase: false,
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
        })
        .unwrap();
        let mut ctx = SmatrixContext {
            wavls: vec![1000.0],
            sin_theta: vec![0.0],
            spec: spec.clone(),
            clamp_min_nm: 2.0,
            clamp_max_nm: 1000.0,
            lm: LmConfig::default(),
        };
        let spectral = SpectralInputs::from_spec(&spec, &[0.0], &[1000.0]).unwrap();
        let (mut stack, _) = glass_stack();
        let hist = run_needle_cycles(
            &mut ctx,
            &mut stack,
            &spectral,
            &contrast_h(),
            &cycle_cfg(),
        )
        .unwrap();
        assert!(hist.iter().all(|h| h.insertion.is_none()));
        assert_eq!(stack.films().len(), 1);
    }

    #[test]
    fn violated_t_demand_drives_insertion() {
        // Same stack, T ≈ 0.957 against a T = 0 demand: inserting H
        // disrupts the matching and lowers T, so the scan must find a
        // negative-P site and the host film must split.
        let (mut stack, _) = glass_stack();
        let mut ctx = StillCtx;
        let spectral = spectral_of(fold_with("t", 0.0));
        let hist = run_needle_cycles(
            &mut ctx,
            &mut stack,
            &spectral,
            &contrast_h(),
            &cycle_cfg(),
        )
        .unwrap();
        // max_needles = 2 and T stays violated → two insertions
        // (1 → 3 → 5 films), both H into the original host lineage.
        assert_eq!(hist.len(), 2);
        for h in &hist {
            let ins = h.insertion.as_ref().expect("expected an insertion");
            assert!(h.best_p.unwrap() < 0.0);
            assert_eq!(ins.material.as_ref(), "H");
        }
        assert_eq!(stack.films().len(), 5);
    }
}
