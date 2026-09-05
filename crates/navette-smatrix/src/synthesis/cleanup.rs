//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::cleanup — post-synthesis design cleanup.
//!
//! Verbatim port of needle_synthesis.py: `remove_thin_layers` and
//! `cleanup_design`. Decision logic (candidate ranking, re-optimization
//! points, merge passes) is kept exactly comparable to the Python so
//! trajectories can be cross-validated; only the evaluation/optimize calls
//! are abstracted behind [`DesignContext`].
//!
//! Sequence (cleanup_design):
//!   1. merge adjacent same-material films
//!   2. iterative impact-ranked removal of thin layers (re-opt after each)
//!   3. merge again (pruning exposes new same-material neighbours)
//!   4. optional final re-optimization on the simplified topology

use crate::synthesis::context::DesignContext;
use crate::synthesis::structure::DesignStack;

/// Record of one cleanup pass — mirrors Python `CleanupResult`.
#[derive(Clone, Copy, Debug)]
pub struct CleanupResult {
    pub merit_before: f64,
    pub merit_after: f64,
    pub layers_before: usize,
    pub layers_after: usize,
    pub layers_removed_thin: usize,
    pub layers_merged: usize,
}

/// Iteratively remove the lowest-impact thin film layers.
///
/// Per iteration: collect films below `min_thickness`, trial-remove each
/// on a CLONE, remove the candidate with the LOWEST resulting MF,
/// re-optimize the survivors. Repeats until no thin layers remain or the
/// removal budget is exhausted.
///
/// `max_removals = None` means uncapped (Python default: layer_count).
pub fn remove_thin_layers<C: DesignContext + ?Sized>(
    ctx: &mut C,
    stack: &mut DesignStack,
    min_thickness: Option<f64>,
    max_removals: Option<usize>,
) -> Result<usize, String> {
    let threshold = min_thickness.unwrap_or(0.5); // cfg.min_layer_thickness default
    let budget = max_removals.unwrap_or(stack.films().len());
    let mut removed = 0usize;

    while removed < budget {
        // 1. Collect candidates (thin layers), in film order.
        let candidates: Vec<usize> = stack
            .films()
            .iter()
            .enumerate()
            .filter(|(_, l)| l.d_nm < threshold)
            .map(|(i, _)| i)
            .collect();
        if candidates.is_empty() {
            break;
        }

        // 2. Trial-remove each candidate; lower trial MF wins.
        let mut best_idx: Option<usize> = None;
        let mut best_mf = f64::INFINITY;
        for &film_idx in &candidates {
            let mut trial = stack.clone();
            trial.remove_film(film_idx)?;
            let mf = ctx.evaluate_merit(&trial)?;
            if mf < best_mf {
                best_mf = mf;
                best_idx = Some(film_idx);
            }
        }
        let Some(best_idx) = best_idx else { break };

        // 3. Perform the removal.
        stack.remove_film(best_idx)?;
        removed += 1;

        // 4. Re-optimize after removal.
        if !stack.films().is_empty() {
            ctx.optimize_thicknesses(stack)?;
        }
    }

    Ok(removed)
}

/// Full cleanup: merge → prune → merge → optional final re-optimize.
///
/// Mirrors `cleanup_design(min_thickness, max_removals, reoptimize)` with
/// Python's defaults filled by the caller (the pipeline passes its own
/// config values).
pub fn cleanup_design<C: DesignContext + ?Sized>(
    ctx: &mut C,
    stack: &mut DesignStack,
    min_thickness: Option<f64>,
    max_removals: Option<usize>,
    reoptimize: bool,
) -> Result<CleanupResult, String> {
    let mf_before = ctx.evaluate_merit(stack)?;
    let n_before = stack.films().len();

    // Pass 1: merge adjacent same-material.
    let merges_1 = stack.merge_adjacent();

    // Pass 2: iterative impact-ranked removal.
    let removed = remove_thin_layers(ctx, stack, min_thickness, max_removals)?;

    // Pass 3: merge again.
    let merges_2 = stack.merge_adjacent();
    let total_merged = merges_1 + merges_2;

    // Final re-optimize on the cleaned stack.
    let mf_after = if reoptimize && !stack.films().is_empty() {
        ctx.optimize_thicknesses(stack)?
    } else {
        ctx.evaluate_merit(stack)?
    };

    Ok(CleanupResult {
        merit_before: mf_before,
        merit_after: mf_after,
        layers_before: n_before,
        layers_after: stack.films().len(),
        layers_removed_thin: removed,
        layers_merged: total_merged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::merit::SimCurves;
    use crate::synthesis::structure::{DesignStack, LayerSpec};

    const NW: usize = 4;

    fn air() -> LayerSpec {
        let mut l = LayerSpec::constant("air", 1.0, 0.0, 0.0, NW);
        l.optimize = false;
        l.needle = false;
        l
    }

    fn sub() -> LayerSpec {
        let mut l = LayerSpec::constant("sub", 1.52, 0.0, 0.0, NW);
        l.optimize = false;
        l.needle = false;
        l
    }

    /// Mock context: MF(stack) = Σ_films w_i·(d_i − target_i)² over ALL
    /// films (optimize flag irrelevant for evaluation). Optimization is a
    /// perfect oracle: every optimize-flagged film jumps to its target.
    struct MockCtx {
        targets: Vec<f64>, // indexed by FILM SLOT (position-based)
        n_opt_calls: usize,
    }

    impl DesignContext for MockCtx {
        fn evaluate_merit(&self, stack: &DesignStack) -> Result<f64, String> {
            Ok(stack
                .films()
                .iter()
                .zip(&self.targets)
                .map(|(l, t)| (l.d_nm - t).powi(2))
                .sum())
        }

        fn simulate(&self, _stack: &DesignStack) -> Result<SimCurves, String> {
            Err("mock context has no simulator".into())
        }

        fn optimize_thicknesses(
            &mut self,
            stack: &mut DesignStack,
        ) -> Result<f64, String> {
            self.n_opt_calls += 1;
            for i in 0..stack.films().len() {
                if stack.films()[i].optimize && i < self.targets.len() {
                    let t = self.targets[i];
                    stack.set_thickness(i, t)?;
                }
            }
            self.evaluate_merit(stack)
        }
    }

    fn film(name: &str, d: f64) -> LayerSpec {
        LayerSpec::constant(name, 2.0, 0.0, d, NW)
    }

    #[test]
    fn removes_least_impactful_first_and_reopts_between() {
        // Films H(1) L(50) H(2); targets [30, 30, 40]; threshold 3.
        // Thin candidates: idx0 H(1) and idx2 H(2).
        // Trial MFs (survivors scored against slot targets [30, 30]):
        //   drop idx0 → [L50 H2] → (50−30)² + (2−30)² = 400+784 = 1184
        //   drop idx2 → [H1 L50] → (1−30)² + (50−30)² = 841+400 = 1241
        // → idx0 removed first (lowest trial MF), then re-opt pulls the
        //   survivors to their slot targets → no thin layers remain.
        let mut stack =
            DesignStack::with_films(air(), sub(), vec![film("H", 1.0), film("L", 50.0), film("H", 2.0)])
                .unwrap();
        let mut ctx = MockCtx { targets: vec![30.0, 30.0, 40.0], n_opt_calls: 0 };

        let removed =
            remove_thin_layers(&mut ctx, &mut stack, Some(3.0), None).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(ctx.n_opt_calls, 1);
        let fs = stack.films();
        assert_eq!(fs.len(), 2);
        assert_eq!(fs[0].material.as_ref(), "L"); // idx0 (the H(1)) was removed
        assert_eq!(fs[1].material.as_ref(), "H");
        assert!((fs[0].d_nm - 30.0).abs() < 1e-12); // re-opt pulled to targets
        assert!((fs[1].d_nm - 30.0).abs() < 1e-12);
    }

    #[test]
    fn max_removals_budget_respected() {
        // Non-optimizable films stay thin across re-opt passes, so the loop
        // keeps finding candidates until the budget cuts it off at 2.
        let mk_thin = |d: f64| {
            let mut l = film("H", d);
            l.optimize = false;
            l
        };
        let mut stack = DesignStack::with_films(
            air(), sub(),
            vec![mk_thin(1.0), mk_thin(1.5), mk_thin(1.7)],
        )
        .unwrap();
        let mut ctx = MockCtx { targets: vec![10.0; 3], n_opt_calls: 0 };

        let removed =
            remove_thin_layers(&mut ctx, &mut stack, Some(3.0), Some(2)).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(stack.films().len(), 1); // budget stopped before third
    }

    #[test]
    fn no_candidates_noop() {
        let mut stack =
            DesignStack::with_films(air(), sub(), vec![film("H", 30.0)]).unwrap();
        let mut ctx = MockCtx { targets: vec![30.0], n_opt_calls: 0 };
        let removed = remove_thin_layers(&mut ctx, &mut stack, Some(3.0), None).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(ctx.n_opt_calls, 0);
    }

    #[test]
    fn full_cleanup_sequence_merge_prune_merge() {
        // Topology: L(30) H(0.8) H(40) — merge pass 1 collapses the two H's
        // (first-of-pair props win), leaving no thin layers; second merge is
        // a no-op. CleanupResult counts one merged pair.
        let mut first_h = film("H", 0.8);
        first_h.optimize = true;
        let mut stack = DesignStack::with_films(
            air(),
            sub(),
            vec![film("L", 30.0), first_h, film("H", 40.0)],
        )
        .unwrap();
        // After merge: L(30) H(40.8) — nothing below threshold 3.
        let mut ctx = MockCtx { targets: vec![30.0, 40.0], n_opt_calls: 0 };

        let res = cleanup_design(&mut ctx, &mut stack, Some(3.0), None, false).unwrap();
        assert_eq!(res.layers_merged, 1);
        assert_eq!(res.layers_removed_thin, 0);
        assert_eq!(res.layers_before, 3);
        assert_eq!(res.layers_after, 2);
        assert_eq!(ctx.n_opt_calls, 0); // reoptimize=false
        // Merged pair keeps FIRST layer's thickness sum and identity.
        assert_eq!(stack.films()[1].material.as_ref(), "H");
        assert!((stack.films()[1].d_nm - 40.8).abs() < 1e-12);
        // Merit evaluated before/after (reoptimize=false → pure evals).
        assert!((res.merit_after - 0.64).abs() < 1e-12);
    }

    #[test]
    fn cleanup_with_final_reoptimize() {
        let mut stack =
            DesignStack::with_films(air(), sub(), vec![film("L", 25.0), film("H", 45.0)])
                .unwrap();
        let mut ctx = MockCtx { targets: vec![30.0, 40.0], n_opt_calls: 0 };
        let res = cleanup_design(&mut ctx, &mut stack, Some(3.0), None, true).unwrap();
        assert_eq!(ctx.n_opt_calls, 1); // final re-opt ran
        assert!((res.merit_after - 0.0).abs() < 1e-12); // perfect mock optimizer
    }

    #[test]
    fn empty_stack_edge() {
        let mut stack = DesignStack::with_films(air(), sub(), vec![]).unwrap();
        let mut ctx = MockCtx { targets: vec![], n_opt_calls: 0 };
        // No films: cleanup must not call optimize (Python guard
        // `layer_count > 0`) and must not panic.
        let res = cleanup_design(&mut ctx, &mut stack, Some(3.0), None, true).unwrap();
        assert_eq!(res.layers_removed_thin, 0);
        assert_eq!(ctx.n_opt_calls, 0);
    }
}
