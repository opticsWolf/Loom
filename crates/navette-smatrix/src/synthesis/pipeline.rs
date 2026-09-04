//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::pipeline — NeedlePipeline macro-loop.
//!
//! Verbatim port of needle_pipeline.py section 6. Each macro-cycle:
//!   1. Needle pass   (cycle.rs: analytic insertions + optimize)
//!   2. Cleanup       (optional; threshold defaults to clamp_min;
//!                     post-cleanup clamp sweep)
//!   3. Inflate       (optional; clamps BEFORE and after re-optimize —
//!                     the ClampedNeedleSynthesizer overrides)
//!   4. Stagnation    record mf_end → check (divergence → oscillation → plateau)
//!
//! Budget checks run pre-flight, after the needle phase, and post-cycle.
//! The loop ends with a final optimize + clamp sweep + evaluation.
//! User abort = callback returning Err (the KeyboardInterrupt analog).

use crate::synthesis::cleanup::{cleanup_design, CleanupResult};
use crate::synthesis::config::{PipelineConfig, TerminationReason};
use crate::synthesis::context::DesignContext;
use crate::synthesis::cycle::{run_needle_cycles, ContrastMap, NeedleCycleConfig, NeedleCycleResult};
use crate::synthesis::inflate::{inflate_design, InflateResult};
use crate::synthesis::stagnation::StagnationDetector;
use crate::synthesis::structure::DesignStack;

/// Record of one macro-cycle — mirrors Python `PipelinePhaseResult`.
#[derive(Clone, Debug)]
pub struct PipelinePhaseResult {
    pub macro_cycle: usize,
    pub mf_after_needle: f64,
    pub mf_after_cleanup: Option<f64>,
    pub mf_after_inflate: Option<f64>,
    pub mf_end: f64,
    pub layer_count: usize,
    pub total_thickness_nm: f64,
    pub needle_results: Vec<NeedleCycleResult>,
    pub cleanup_result: Option<CleanupResult>,
    pub inflate_result: Option<InflateResult>,
}

/// Full output of a pipeline run — mirrors Python `PipelineResult`.
#[derive(Clone, Debug)]
pub struct PipelineResult {
    pub phases: Vec<PipelinePhaseResult>,
    pub termination: TerminationReason,
    pub final_mf: f64,
    pub final_layer_count: usize,
    pub final_total_thickness_nm: f64,
    pub stagnation_detail: Option<String>,
}

/// Continuous iterative needle synthesis pipeline.
pub struct NeedlePipeline {
    pub stack: DesignStack,
    pub cfg: PipelineConfig,
    pub needle_cfg: NeedleCycleConfig,
    pub contrast: ContrastMap,
    /// Fixed spectral problem definition for the needle pass.
    pub spectral: SpectralInputs,
    detector: StagnationDetector,
}

impl NeedlePipeline {
    pub fn new(
        stack: DesignStack,
        spectral: SpectralInputs,
        cfg: PipelineConfig,
        needle_cfg: NeedleCycleConfig,
        contrast: ContrastMap,
    ) -> Result<Self, String> {
        let cfg = cfg.validated()?;
        let detector = StagnationDetector::new(
            cfg.stagnation_window,
            cfg.stagnation_gradient_tol,
            cfg.stagnation_oscillation_ratio,
            cfg.stagnation_divergence_count,
        );
        Ok(NeedlePipeline { stack, spectral, cfg, needle_cfg, contrast, detector })
    }

    fn check_budgets<C: DesignContext + ?Sized>(
        &self,
        ctx: &C,
    ) -> Result<Option<TerminationReason>, String> {
        if self.stack.films().len() >= self.cfg.max_film_layers {
            return Ok(Some(TerminationReason::LayerBudgetReached));
        }
        let total: f64 = self.stack.films().iter().map(|l| l.d_nm).sum();
        if total >= self.cfg.max_total_thickness_nm {
            return Ok(Some(TerminationReason::ThicknessBudgetReached));
        }
        if self.cfg.merit_target > 0.0 {
            let mf = ctx.evaluate_merit(&self.stack)?;
            if mf <= self.cfg.merit_target {
                return Ok(Some(TerminationReason::MeritTargetReached));
            }
        }
        Ok(None)
    }

    /// Execute the pipeline.
    ///
    /// `callback(macro_cycle, &phase, &detector)` runs after each completed
    /// cycle; an `Err` return aborts the run as [`TerminationReason::UserAbort`].
    pub fn run<C: DesignContext + ?Sized>(
        &mut self,
        ctx: &mut C,
        mut callback: impl FnMut(usize, &PipelinePhaseResult, &StagnationDetector) -> Result<(), String>,
    ) -> Result<PipelineResult, String> {
        let mut phases: Vec<PipelinePhaseResult> = Vec::new();
        self.detector.reset();

        // Per-cycle needle budget.
        let mut needle_cfg = self.needle_cfg.clone();
        needle_cfg.max_needles = self.cfg.needles_per_cycle;

        let mut termination = TerminationReason::MaxIterationsReached;
        let mut stag_detail: Option<String> = None;
        let mut user_abort = false;

        'main: for cycle_i in 1..=self.cfg.max_macro_cycles {
            // -- Pre-flight budget check --
            if let Some(reason) = self.check_budgets(ctx)? {
                termination = reason;
                break;
            }

            // ── Phase 1: Needle pass ──
            let needle_results = run_needle_cycles(
                ctx,
                &mut self.stack,
                &self.spectral.wavls,
                &self.spectral.sin_theta,
                &self.spectral.targets_r,
                &self.spectral.weights_r,
                &self.contrast,
                &needle_cfg,
            )?;
            let mf_needle = ctx.evaluate_merit(&self.stack)?;

            // Budget check after needle.
            if let Some(reason) = self.check_budgets(ctx)? {
                let total: f64 = self.stack.films().iter().map(|l| l.d_nm).sum();
                let phase = PipelinePhaseResult {
                    macro_cycle: cycle_i,
                    mf_after_needle: mf_needle,
                    mf_after_cleanup: None,
                    mf_after_inflate: None,
                    mf_end: mf_needle,
                    layer_count: self.stack.films().len(),
                    total_thickness_nm: total,
                    needle_results,
                    cleanup_result: None,
                    inflate_result: None,
                };
                phases.push(phase);
                let phase = phases.last().unwrap();
                self.detector.record(phase.mf_end);
                // KeyboardInterrupt analog: callback failure aborts the loop.
                if callback(cycle_i, phase, &self.detector).is_err() {
                    user_abort = true;
                }
                termination = reason;
                break 'main;
            }

            // ── Phase 2: Cleanup (optional) ──
            let cleanup_result = if self.cfg.enable_cleanup {
                let r = cleanup_design(
                    ctx,
                    &mut self.stack,
                    self.cfg.cleanup_min_nm,
                    self.cfg.cleanup_max_removals,
                    true,
                )?;
                // Post-cleanup clamp (Clamped override semantics).
                self.stack.clamp_all(self.cfg.clamp_min_nm, self.cfg.clamp_max_nm);
                Some(r)
            } else {
                None
            };
            let mf_cleanup = cleanup_result.as_ref().map(|r| r.merit_after);

            // ── Phase 3: Inflate (optional) ──
            let inflate_result = if self.cfg.enable_inflate {
                let r = inflate_design(
                    ctx,
                    &mut self.stack,
                    &self.spectral.wavls,
                    self.cfg.inflate_addon_qwot,
                    self.cfg.inflate_reference_wl,
                    self.cfg.inflate_max_layers,
                    true,
                )?;
                // Clamp BEFORE re-optimize happened in Python before the call
                // ordering above; enforce the AFTER clamp here too.
                self.stack.clamp_all(self.cfg.clamp_min_nm, self.cfg.clamp_max_nm);
                Some(r)
            } else {
                None
            };
            let mf_inflate = inflate_result.as_ref().map(|r| r.merit_after);

            let mf_cleanup_s = mf_cleanup;
            let mf_inflate_s = mf_inflate;

            // ── Record phase ──
            let total: f64 = self.stack.films().iter().map(|l| l.d_nm).sum();
            let mf_end = mf_inflate.or(mf_cleanup).unwrap_or(mf_needle);
            phases.push(PipelinePhaseResult {
                macro_cycle: cycle_i,
                mf_after_needle: mf_needle,
                mf_after_cleanup: mf_cleanup_s,
                mf_after_inflate: mf_inflate_s,
                mf_end,
                layer_count: self.stack.films().len(),
                total_thickness_nm: total,
                needle_results,
                cleanup_result,
                inflate_result,
            });

            // ── Stagnation check ──
            self.detector.record(mf_end);
            {
                let phase = phases.last().unwrap();
                if callback(cycle_i, phase, &self.detector).is_err() {
                    // KeyboardInterrupt analog: skip straight to finalization.
                    user_abort = true;
                    break 'main;
                }
            }

            if let Some(stag) = self.detector.check() {
                termination = stag;
                stag_detail = Some(self.detector.summary());
                break 'main;
            }

            // Post-cycle budget check.
            if let Some(reason) = self.check_budgets(ctx)? {
                termination = reason;
                break 'main;
            }
        }

        // Exception-style abort overrides any other reason (Python catches
        // KeyboardInterrupt around the whole loop).
        if user_abort {
            termination = TerminationReason::UserAbort;
            stag_detail = None;
        }

        // ── Final optimisation + clamp sweep ──
        ctx.optimize_thicknesses(&mut self.stack)?;
        self.stack.clamp_all(self.cfg.clamp_min_nm, self.cfg.clamp_max_nm);
        let final_mf = ctx.evaluate_merit(&self.stack)?;

        Ok(PipelineResult {
            phases,
            termination,
            final_mf,
            final_layer_count: self.stack.films().len(),
            final_total_thickness_nm: self.stack.films().iter().map(|l| l.d_nm).sum(),
            stagnation_detail: stag_detail,
        })
    }
}

/// Spectral problem definition needed by the needle pass.
#[derive(Clone, Debug)]
pub struct SpectralInputs {
    pub wavls: Vec<f64>,
    /// Sines of incidence angles.
    pub sin_theta: Vec<f64>,
    /// Raw reflectance targets, angle-major (see needle_pass builder).
    pub targets_r: Vec<f64>,
    /// Folded weights, angle-major.
    pub weights_r: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::structure::LayerSpec;

    const NW: usize = 3;

    fn dummy_spectral() -> SpectralInputs {
        SpectralInputs {
            wavls: vec![500.0; NW],
            sin_theta: vec![0.0],
            targets_r: vec![0.0; NW],
            weights_r: vec![1.0; NW],
        }
    }

    fn air() -> LayerSpec {
        let mut l = LayerSpec::constant("air", 1.0, 0.0, 0.0, NW);
        l.optimize = false;
        l
    }

    fn sub() -> LayerSpec {
        let mut l = LayerSpec::constant("sub", 1.52, 0.0, 0.0, NW);
        l.optimize = false;
        l
    }

    /// No-op context: MF constant (2.5), optimize does nothing.
    struct FlatCtx;
    impl DesignContext for FlatCtx {
        fn evaluate_merit(&self, _s: &DesignStack) -> Result<f64, String> {
            Ok(2.5)
        }
        fn optimize_thicknesses(&mut self, s: &mut DesignStack) -> Result<f64, String> {
            self.evaluate_merit(s)
        }
    }

    fn pipeline(cfg_over: impl FnOnce(&mut PipelineConfig)) -> NeedlePipeline {
        let mut cfg = PipelineConfig::default();
        cfg_over(&mut cfg);
        let stack =
            DesignStack::with_films(air(), sub(), vec![LayerSpec::constant("H", 2.35, 0.0, 100.0, NW)])
                .unwrap();
        NeedlePipeline::new(
            stack,
            dummy_spectral(),
            cfg,
            NeedleCycleConfig::default(),
            ContrastMap::new(), // no contrast → needle pass inserts nothing
        )
        .unwrap()
    }

    #[test]
    fn max_macro_cycles_is_default_termination() {
        let mut p = pipeline(|c| c.max_macro_cycles = 2);
        let res = p.run(&mut FlatCtx, |_, _, _| Ok(())).unwrap();
        assert_eq!(res.termination, TerminationReason::MaxIterationsReached);
        assert_eq!(res.phases.len(), 2);
        assert!((res.final_mf - 2.5).abs() < 1e-12);
    }

    #[test]
    fn layer_budget_fires_preflight() {
        let mut p = pipeline(|c| {
            c.max_macro_cycles = 5;
            c.max_film_layers = 1; // 1 film present ≥ 1
        });
        let res = p.run(&mut FlatCtx, |_, _, _| Ok(())).unwrap();
        assert_eq!(res.termination, TerminationReason::LayerBudgetReached);
        assert!(res.phases.is_empty()); // fired BEFORE cycle 1
    }

    #[test]
    fn merit_target_fires_preflight() {
        let mut p = pipeline(|c| {
            c.max_macro_cycles = 5;
            c.merit_target = 3.0; // initial MF 2.5 ≤ 3 → stop immediately
        });
        let res = p.run(&mut FlatCtx, |_, _, _| Ok(())).unwrap();
        assert_eq!(res.termination, TerminationReason::MeritTargetReached);
        assert!(res.phases.is_empty());
    }

    #[test]
    fn thickness_budget_fires() {
        let mut p = pipeline(|c| {
            c.max_total_thickness_nm = 50.0; // stack is 100 nm thick already
        });
        let res = p.run(&mut FlatCtx, |_, _, _| Ok(())).unwrap();
        assert_eq!(res.termination, TerminationReason::ThicknessBudgetReached);
    }

    #[test]
    fn callback_error_aborts_as_user_abort() {
        let mut p = pipeline(|c| {
            // Need one completed phase before callback fires post-cycle:
            // disable cleanup/inflate so phases complete; but pre-flight
            // budgets must pass: defaults ok.
            c.enable_cleanup = false;
            c.stagnation_window = usize::MAX; // never fire stagnation first
        });
        let res = p.run(&mut FlatCtx, |_, _, _| Err("stop!".into())).unwrap();
        assert_eq!(res.termination, TerminationReason::UserAbort);
    }

    #[test]
    fn plateau_terminates_flat_trajectory() {
        let mut p = pipeline(|c| {
            c.enable_cleanup = false;
            c.stagnation_window = 2;
            c.max_macro_cycles = 10;
        });
        let res = p.run(&mut FlatCtx, |_, _, _| Ok(())).unwrap();
        assert_eq!(res.termination, TerminationReason::StagnationPlateau);
        assert!(res.stagnation_detail.is_some());
        // Window of 2 samples recorded then plateau detected at cycle 2.
        assert_eq!(res.phases.len(), 2);
    }

    #[test]
    fn final_optimize_and_clamp_run_after_loop() {
        // Context whose optimizer caps thicknesses via recorded calls.
        struct CountingCtx {
            opt_calls: usize,
        }
        impl DesignContext for CountingCtx {
            fn evaluate_merit(&self, _: &DesignStack) -> Result<f64, String> {
                Ok(1.0)
            }
            fn optimize_thicknesses(
                &mut self,
                s: &mut DesignStack,
            ) -> Result<f64, String> {
                self.opt_calls += 1;
                s.set_thickness(0, 5000.0)?;
                self.evaluate_merit(s)
            }
        }
        // clamp_max small → final clamp sweep must cap the 5000 nm film.
        let mut p = pipeline(|c| {
            c.clamp_max_nm = 800.0;
            c.max_macro_cycles = 1;
            c.enable_cleanup = false;
            c.stagnation_window = usize::MAX;
        });
        let mut ctx = CountingCtx { opt_calls: 0 };
        let res = p.run(&mut ctx, |_, _, _| Ok(())).unwrap();
        // needle pass optimize (initial) + per-cycle + FINAL = counted
        assert!(ctx.opt_calls >= 2);
        assert!(res.final_total_thickness_nm <= 800.0);
    }
}
