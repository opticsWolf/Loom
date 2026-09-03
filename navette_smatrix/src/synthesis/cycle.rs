// Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//
// synthesis::cycle — one needle pass: repeated analytic insertions.
//
// Port of NeedleSynthesizer.run() with compute_p_function replaced by the
// analytic sweep (needle_pass). The Python convergence test measured the
// MF drop from inserting a 1 nm TEST needle:
//     improvement_py = MF_now − MF(test needle, 1 nm) ≈ −P(z*)·1 nm
// (first-order, since P = dF/dδ). We generalize to the seed thickness:
//     predicted_improvement = −P_best · δ_seed
// and stop when it falls below `convergence_threshold`. Thresholds are in
// merit-units-per-nm-of-seed — recalibrate when porting configs.

use std::collections::HashMap;
use std::sync::Arc;

use num_complex::Complex64;

use crate::synthesis::context::DesignContext;
use crate::synthesis::needle_pass::{
    build_scan_sites, run_needle_pass, NeedlePassInput,
};
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
}

impl Default for NeedleCycleConfig {
    fn default() -> Self {
        NeedleCycleConfig {
            max_needles: 10,
            convergence_threshold: 1e-4,
            needle_seed_thickness_nm: 5.0,
            scan_step_nm: 2.0,
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
/// initial optimization → [scan → select → insert → optimize]×N.
pub fn run_needle_cycles<C: DesignContext + ?Sized>(
    ctx: &mut C,
    stack: &mut DesignStack,
    wavls: &[f64],
    sin_theta: &[f64],
    targets_r: &[f64],
    weights_r: &[f64],
    contrast: &ContrastMap,
    cfg: &NeedleCycleConfig,
) -> Result<Vec<NeedleCycleResult>, String> {
    let mut history = Vec::new();

    // Initial optimization.
    ctx.optimize_thicknesses(stack)?;

    for cycle in 0..cfg.max_needles {
        // 1. Build candidate sites restricted to films whose material has a
        //    contrast entry (Python skips layers without a mapping).
        let sites = build_scan_sites(stack.films(), cfg.scan_step_nm);
        let sites: Vec<_> = sites
            .into_iter()
            .filter(|s| {
                stack
                    .films()
                    .get(s.film_idx)
                    .map(|l| contrast.contains_key(&l.material))
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
                wavls,
                sin_theta,
                targets_r,
                weights_r,
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
                let predicted = -p * cfg.needle_seed_thickness_nm;
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
