//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::config — PipelineConfig / TerminationReason / cycle knobs.
//!
//! Verbatim port of needle_pipeline.py section 1–2 (defaults included).

// ---------------------------------------------------------------------------
// TerminationReason
// ---------------------------------------------------------------------------

/// Why the pipeline stopped — mirrors Python `TerminationReason`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerminationReason {
    LayerBudgetReached,
    ThicknessBudgetReached,
    StagnationPlateau,
    StagnationOscillation,
    StagnationDivergence,
    MaxIterationsReached,
    MeritTargetReached,
    UserAbort,
}

impl TerminationReason {
    pub fn name(self) -> &'static str {
        match self {
            TerminationReason::LayerBudgetReached => "LAYER_BUDGET_REACHED",
            TerminationReason::ThicknessBudgetReached => "THICKNESS_BUDGET_REACHED",
            TerminationReason::StagnationPlateau => "STAGNATION_PLATEAU",
            TerminationReason::StagnationOscillation => "STAGNATION_OSCILLATION",
            TerminationReason::StagnationDivergence => "STAGNATION_DIVERGENCE",
            TerminationReason::MaxIterationsReached => "MAX_ITERATIONS_REACHED",
            TerminationReason::MeritTargetReached => "MERIT_TARGET_REACHED",
            TerminationReason::UserAbort => "USER_ABORT",
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------

/// All loop-control parameters — mirrors Python `PipelineConfig`.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    // -- budgets --
    pub max_film_layers: usize,
    pub max_total_thickness_nm: f64,
    pub max_macro_cycles: usize,
    pub merit_target: f64,

    // -- clamping --
    pub clamp_min_nm: f64,
    pub clamp_max_nm: f64,

    // -- needle --
    pub needles_per_cycle: usize,

    // -- cleanup --
    pub enable_cleanup: bool,
    /// `None` → falls back to `clamp_min_nm` (`validated()` resolves it).
    pub cleanup_min_nm: Option<f64>,
    /// `None` → uncapped.
    pub cleanup_max_removals: Option<usize>,

    // -- inflate --
    pub enable_inflate: bool,
    pub inflate_addon_qwot: f64,
    pub inflate_reference_wl: f64,
    /// `None` → all layers.
    pub inflate_max_layers: Option<usize>,

    // -- stagnation --
    pub stagnation_window: usize,
    pub stagnation_gradient_tol: f64,
    pub stagnation_oscillation_ratio: f64,
    pub stagnation_divergence_count: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            max_film_layers: 40,
            max_total_thickness_nm: 5000.0,
            max_macro_cycles: 50,
            merit_target: 0.0,
            clamp_min_nm: 2.0,
            clamp_max_nm: 1000.0,
            needles_per_cycle: 3,
            enable_cleanup: true,
            cleanup_min_nm: None, // resolved to clamp_min_nm by validated()
            cleanup_max_removals: None,
            enable_inflate: false,
            inflate_addon_qwot: 2.0,
            inflate_reference_wl: 550.0,
            inflate_max_layers: None,
            stagnation_window: 5,
            stagnation_gradient_tol: 1e-4,
            stagnation_oscillation_ratio: 0.75,
            stagnation_divergence_count: 3,
        }
    }
}

impl PipelineConfig {
    /// Apply the `__post_init__` semantics: resolve `cleanup_min_nm`
    /// fallback and validate invariants. Returns an owned, resolved copy.
    pub fn validated(mut self) -> Result<Self, String> {
        if self.cleanup_min_nm.is_none() {
            self.cleanup_min_nm = Some(self.clamp_min_nm);
        }
        if self.clamp_min_nm < 0.0 {
            return Err("clamp_min_nm must be non-negative.".into());
        }
        if self.clamp_max_nm <= self.clamp_min_nm {
            return Err("clamp_max_nm must be greater than clamp_min_nm.".into());
        }
        if self.stagnation_window < 2 {
            return Err("stagnation_window must be ≥ 2.".into());
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let c = PipelineConfig::default();
        assert_eq!(c.max_film_layers, 40);
        assert!((c.max_total_thickness_nm - 5000.0).abs() < 1e-12);
        assert_eq!(c.max_macro_cycles, 50);
        assert_eq!(c.merit_target, 0.0);
        assert!((c.clamp_min_nm - 2.0).abs() < 1e-12);
        assert!((c.clamp_max_nm - 1000.0).abs() < 1e-12);
        assert_eq!(c.needles_per_cycle, 3);
        assert!(c.enable_cleanup && !c.enable_inflate);
        assert!((c.inflate_addon_qwot - 2.0).abs() < 1e-12);
        assert!((c.inflate_reference_wl - 550.0).abs() < 1e-12);
        assert_eq!(c.stagnation_window, 5);
        assert!((c.stagnation_gradient_tol - 1e-4).abs() < 1e-15);
        assert!((c.stagnation_oscillation_ratio - 0.75).abs() < 1e-12);
        assert_eq!(c.stagnation_divergence_count, 3);
    }

    #[test]
    fn validated_resolves_cleanup_fallback_and_checks_invariants() {
        let c = PipelineConfig::default().validated().unwrap();
        assert!((c.cleanup_min_nm.unwrap() - c.clamp_min_nm).abs() < 1e-12);

        assert!(PipelineConfig { clamp_min_nm: -1.0, ..Default::default() }
            .validated()
            .is_err());
        assert!(PipelineConfig {
            clamp_min_nm: 5.0,
            clamp_max_nm: 5.0,
            ..Default::default()
        }
        .validated()
        .is_err());
        assert!(PipelineConfig { stagnation_window: 1, ..Default::default() }
            .validated()
            .is_err());
    }
}
