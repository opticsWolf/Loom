//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::context — abstraction seam between the synthesis algorithms
//! (cleanup / inflate / pipeline) and the numeric machinery (TMM solve +
//! MeritSpec residuals + bounded LM).
//!
//! The algorithms in cleanup.rs / inflate.rs / pipeline.rs are written
//! against this trait so they are unit-testable with analytic mock
//! evaluators today, and wired to core_engine + thick_opt in the pyo3 shell
//! tomorrow without touching their decision logic (which must stay
//! byte-comparable to needle_synthesis.py).

use crate::synthesis::merit::SimCurves;
use crate::synthesis::structure::DesignStack;

pub trait DesignContext {
    /// Merit function of the given stack (pure evaluation, no mutation).
    ///
    /// Mirrors `NeedleSynthesizer.evaluate_merit(structure)`.
    fn evaluate_merit(&self, stack: &DesignStack) -> Result<f64, String>;

    /// Full simulation on the context grid (rows + PD metadata).
    ///
    /// Used for per-cycle needle-target re-folds; contexts without a
    /// solver return `Err` and callers fall back to the static fold.
    fn simulate(&self, stack: &DesignStack) -> Result<SimCurves, String>;

    /// Bounded thickness re-optimization IN PLACE.
    ///
    /// Mirrors `ClampedNeedleSynthesizer.optimize_thicknesses`: optimizes
    /// all films flagged `optimize`, enforces bounds, removes films driven
    /// below the minimum. Returns the post-optimization merit.
    fn optimize_thicknesses(&mut self, stack: &mut DesignStack)
        -> Result<f64, String>;
}
