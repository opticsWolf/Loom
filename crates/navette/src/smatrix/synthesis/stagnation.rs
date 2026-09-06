//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::stagnation — StagnationDetector.
//!
//! Verbatim port of needle_pipeline.py section 4. Three independent failure
//! modes over a sliding window of MF samples:
//!   1. Plateau      — |normalized regression gradient| < tol  OR  grad > 0
//!   2. Oscillation  — sign-alternation fraction of consecutive deltas ≥ ratio
//!   3. Divergence   — ≥ N consecutive MF increases (>= comparisons)
//! Check order: divergence → oscillation → plateau (most urgent first).

/// Why the trajectory is stuck — reuses `TerminationReason` stagnation
/// variants via [`StagnationDetector::check`].

#[derive(Clone, Debug)]
pub struct StagnationDetector {
    window: usize,
    gradient_tol: f64,
    oscillation_ratio: f64,
    divergence_count: usize,
    history: Vec<f64>,
}

impl StagnationDetector {
    pub fn new(
        window: usize,
        gradient_tol: f64,
        oscillation_ratio: f64,
        divergence_count: usize,
    ) -> Self {
        StagnationDetector {
            window: window.max(2),
            gradient_tol,
            oscillation_ratio,
            divergence_count: divergence_count.max(2),
            history: Vec::new(),
        }
    }

    /// Build from PipelineConfig fields.
    pub fn from_params(
        window: usize,
        gradient_tol: f64,
        oscillation_ratio: f64,
        divergence_count: usize,
    ) -> Self {
        Self::new(window, gradient_tol, oscillation_ratio, divergence_count)
    }

    pub fn record(&mut self, mf: f64) {
        self.history.push(mf);
    }

    pub fn count(&self) -> usize {
        self.history.len()
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    fn recent(&self) -> &[f64] {
        let start = self.history.len().saturating_sub(self.window);
        &self.history[start..]
    }

    /// Linear-regression slope of the recent window, normalized by the mean
    /// MF. `< 0` improving, `≈ 0` plateau, `> 0` diverging.
    /// Returns `-inf` for insufficient data — mirrors Python exactly
    /// (including the `y_mean == 0` and zero-variance guards returning 0).
    pub fn normalised_gradient(&self) -> f64 {
        if self.count() < 2 {
            return f64::NEG_INFINITY;
        }
        let recent = self.recent();
        let n = recent.len() as f64;
        let x_mean = (n - 1.0) / 2.0; // mean of 0..n-1
        let y_mean: f64 = recent.iter().sum::<f64>() / n;
        if y_mean == 0.0 {
            return 0.0;
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in recent.iter().enumerate() {
            let dx = i as f64 - x_mean;
            num += dx * (y - y_mean);
            den += dx * dx;
        }
        if den == 0.0 {
            return 0.0;
        }
        (num / den) / y_mean.abs()
    }

    /// Fraction of consecutive deltas (within the window) that alternate in
    /// sign; zeros are removed before counting. 0.0 when insufficient data.
    pub fn oscillation_fraction(&self) -> f64 {
        if self.count() < 3 {
            return 0.0;
        }
        let recent = self.recent();
        // NOTE: f64::signum(+0.0) == 1.0 in Rust (unlike numpy.sign), so an
        // explicit three-way mapping is required to treat flat deltas as
        // "no change" exactly like the Python reference.
        let signs: Vec<f64> = recent
            .windows(2)
            .map(|w| {
                let d = w[1] - w[0];
                if d > 0.0 {
                    1.0
                } else if d < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .filter(|&s| s != 0.0)
            .collect();
        if signs.len() < 2 {
            return 0.0;
        }
        let alternations = signs.windows(2).filter(|w| w[0] != w[1]).count();
        let max_possible = signs.len() - 1;
        alternations as f64 / max_possible as f64
    }

    /// Consecutive MF increases counting back from the newest sample
    /// (`>=` comparisons — flat segments count as increases, like Python).
    pub fn consecutive_increases(&self) -> usize {
        if self.count() < 2 {
            return 0;
        }
        let mut streak = 0;
        for i in (1..self.history.len()).rev() {
            if self.history[i] >= self.history[i - 1] {
                streak += 1;
            } else {
                break;
            }
        }
        streak
    }

    /// Run all detectors: `Some(reason)` to stop, `None` to continue.
    /// Requires a full window of samples before it will fire.
    pub fn check(&self) -> Option<crate::smatrix::synthesis::config::TerminationReason> {
        use crate::smatrix::synthesis::config::TerminationReason::*;
        if self.count() < self.window {
            return None;
        }
        // 1. Divergence
        if self.consecutive_increases() >= self.divergence_count {
            return Some(StagnationDivergence);
        }
        // 2. Oscillation
        if self.oscillation_fraction() >= self.oscillation_ratio {
            return Some(StagnationOscillation);
        }
        // 3. Plateau: near-zero gradient or any positive trend
        let grad = self.normalised_gradient();
        if grad.abs() < self.gradient_tol || grad > 0.0 {
            return Some(StagnationPlateau);
        }
        None
    }

    /// Human-readable state summary (mirrors `summary()`).
    pub fn summary(&self) -> String {
        format!(
            "StagnationDetector(samples={}, gradient={:+.2e}, oscillation={:.0}%, consecutive_up={})",
            self.count(),
            self.normalised_gradient(),
            self.oscillation_fraction() * 100.0,
            self.consecutive_increases()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det() -> StagnationDetector {
        StagnationDetector::new(5, 1e-4, 0.75, 3)
    }

    #[test]
    fn gradient_linear_decrease_hand_computed() {
        let mut d = det();
        for v in [10.0, 8.0, 6.0, 4.0, 2.0] {
            d.record(v);
        }
        // slope = −2, mean = 6 → −2/6
        assert!((d.normalised_gradient() - (-1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn gradient_edge_cases_match_python() {
        let mut d = det();
        assert_eq!(d.normalised_gradient(), f64::NEG_INFINITY); // < 2 samples
        d.record(5.0);
        assert_eq!(d.normalised_gradient(), f64::NEG_INFINITY); // still < 2
        d.record(5.0);
        // constant series: slope 0, mean 5 → 0
        assert_eq!(d.normalised_gradient(), 0.0);
        // zero-mean series: guard returns 0.0
        d.reset();
        d.record(-1.0);
        d.record(0.0);
        d.record(1.0);
        // window [−1,0,1]: mean 0 → returns 0.0 per the Python guard
        assert_eq!(d.normalised_gradient(), 0.0);
    }

    #[test]
    fn oscillation_fraction_alternations() {
        let mut d = det();
        assert_eq!(d.oscillation_fraction(), 0.0); // < 3 samples
        for v in [1.0, 3.0, 0.5, 2.5, 0.7] {
            d.record(v);
        }
        // deltas: +2, −2.5, +2, −1.8 → all alternate → 3/3 = 1.0
        assert!((d.oscillation_fraction() - 1.0).abs() < 1e-12);

        d.reset();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            d.record(v);
        }
        // monotone: no alternations → 0/3
        assert!((d.oscillation_fraction() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn oscillation_ignores_zero_deltas() {
        let mut d = det();
        for v in [1.0, 1.0, 3.0, 3.0, 0.5] {
            d.record(v);
        }
        assert!((d.oscillation_fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn consecutive_increases_counts_flat_as_increase() {
        let mut d = det();
        for v in [5.0, 4.0, 3.0] {
            d.record(v);
        }
        assert_eq!(d.consecutive_increases(), 0);
        d.record(3.5);
        assert_eq!(d.consecutive_increases(), 1);
        d.record(3.5); // equal counts (>= semantics)
        assert_eq!(d.consecutive_increases(), 2);
    }

    #[test]
    fn check_requires_full_window() {
        let mut d = StagnationDetector::new(5, 1e-4, 0.75, 3);
        for v in [9.0, 9.0, 9.0] {
            d.record(v); // blatantly stagnant but window not full
        }
        assert!(d.check().is_none());
    }

    #[test]
    fn check_detects_divergence_first() {
        let mut d = det();
        for v in [10.0, 11.0, 12.0, 13.0, 14.0] {
            d.record(v);
        }
        use crate::smatrix::synthesis::config::TerminationReason::*;
        // Also oscillation-free & positive-gradient — divergence wins by order.
        assert_eq!(d.check(), Some(StagnationDivergence));
    }

    #[test]
    fn check_detects_oscillation() {
        let mut d = det();
        for v in [10.0, 14.0, 10.0, 14.0, 10.0] {
            d.record(v);
        }
        use crate::smatrix::synthesis::config::TerminationReason::*;
        // consecutive increases back-from-newest: 14→10 decrease → 0 streak;
        // oscillation fraction 1.0 ≥ 0.75 fires.
        assert_eq!(d.check(), Some(StagnationOscillation));
    }

    #[test]
    fn flat_and_rising_series_diverge_first_python_faithful() {
        use crate::smatrix::synthesis::config::TerminationReason::*;
        // IMPORTANT: Python's consecutive_increases uses >= comparisons, so
        // FLAT and slowly-RISING trajectories trip DIVERGENCE (streak ≥ 3)
        // before the plateau detector ever runs. The port is faithful.
        let mut d = det();
        for v in [5.0, 5.0, 5.0, 5.0, 5.0] {
            d.record(v);
        }
        assert_eq!(d.check(), Some(StagnationDivergence));

        let mut d_up = det();
        for i in 0..5 {
            d_up.record(50.0 + 1e-6 * i as f64); // "slow worsening"
        }
        assert_eq!(d_up.check(), Some(StagnationDivergence));
    }

    #[test]
    fn check_detects_plateau_on_slow_improvement() {
        use crate::smatrix::synthesis::config::TerminationReason::*;
        // Strictly DECREASING too slowly: no increase-streak, no
        // oscillation, |normalized gradient| < tol → Plateau.
        let mut d = det();
        for i in 0..5 {
            d.record(50.0 - 1e-6 * i as f64);
        }
        assert_eq!(d.check(), Some(StagnationPlateau));

        // Healthy improvement continues: strictly decreasing fast enough
        // (slope −10 on mean 80 → grad ≈ −0.125, |grad| > 1e-4).
        let mut d2 = det();
        for v in [100.0, 90.0, 80.0, 70.0, 60.0] {
            d2.record(v);
        }
        assert_eq!(d2.check(), None);
    }

    #[test]
    fn summary_format_smoke() {
        let mut d = det();
        d.record(1.0);
        let s = d.summary();
        assert!(s.contains("samples=1"));
        assert!(s.contains("consecutive_up=0"));
    }
}
