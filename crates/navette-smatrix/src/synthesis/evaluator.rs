//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::evaluator — SmatrixContext: the REAL DesignContext.
//!
//! Wires together, all in-process (no Python round trip):
//!   * coherent_block::solve_coherent_block_fields_dual  — one dual-pol
//!     interface sweep per spectral point (rayon over angle×λ grid)
//!   * MeritSpec::merit / ::residuals                    — targets engine
//!   * thick_opt::levenberg_marquardt                    — bounded LM
//!
//! optimize_thicknesses mirrors ClampedNeedleSynthesizer.optimize_thicknesses:
//! LM over optimize-flagged films with bounds [0, clamp_max], then a
//! clamp_all(clamp_min, clamp_max) sweep that REMOVES sub-min layers.

use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;

use crate::coherent_block::solve_coherent_block_fields_dual;
use crate::synthesis::context::DesignContext;
use crate::synthesis::merit::{CurveId, MeritSpec, SimCurves};
use crate::synthesis::structure::{DesignStack, SolverArrays};
use crate::synthesis::thick_opt::{levenberg_marquardt, LmConfig};

/// Solver + merit context for one synthesis problem.
#[derive(Clone)]
pub struct SmatrixContext {
    pub wavls: Vec<f64>,
    /// Sines of incidence angles (solver convention).
    pub sin_theta: Vec<f64>,
    pub spec: MeritSpec,
    pub clamp_min_nm: f64,
    pub clamp_max_nm: f64,
    pub lm: LmConfig,
}

impl SmatrixContext {
    /// Simulate the stack on the fixed grid → SimCurves.
    ///
    /// Fully-coherent path: block = [0, nl−1) with the substrate as
    /// half-space. Rs/Rp/Ts/Tp assembled row-major (k = a·nw + w).
    pub fn simulate(&self, stack: &DesignStack) -> Result<SimCurves, String> {
        let sa: SolverArrays = stack.solver_arrays();
        let nl = sa.n_layers as usize;
        if nl < 2 {
            return Err("degenerate stack".into());
        }
        let nw = self.wavls.len();
        let na = self.sin_theta.len();
        let start = 0usize;
        let end = nl - 1;
        if end <= start {
            return Err("stack needs at least ambient + substrate".into());
        }

        // Per-point (r, t_back) intensities via the dual solver.
        struct Pt {
            rs: f64,
            rp: f64,
            ts: f64,
            tp: f64,
        }
        let pts: Vec<Pt> = (0..na * nw)
            .into_par_iter()
            .map(|k| {
                let a = k / nw;
                let w = k % nw;
                let lam = self.wavls[w];
                let base = w * nl * 2;
                let n_slice: Vec<Complex64> = (0..nl)
                    .map(|l| Complex64::new(sa.n_stack_cache[base + l * 2], sa.n_stack_cache[base + l * 2 + 1]))
                    .collect();
                let inv_n_slice: Vec<Complex64> =
                    n_slice.iter().map(|&n| 1.0 / n).collect();
                let nsin_fi = n_slice[0] * Complex64::new(self.sin_theta[a], 0.0);
                let (s_res, p_res) = solve_coherent_block_fields_dual(
                    start,
                    end,
                    &n_slice,
                    &inv_n_slice,
                    &sa.thicknesses,
                    &sa.rough_vals,
                    &sa.rough_types,
                    lam,
                    nsin_fi,
                );
                Pt { rs: s_res.4, rp: p_res.4, ts: s_res.5, tp: p_res.5 }
            })
            .collect();

        let mk = |id: CurveId| -> Arc<[f64]> {
            let idx = id.index();
            let pick = |p: &Pt| match id {
                CurveId::Rs => p.rs,
                CurveId::Rp => p.rp,
                CurveId::Ts => p.ts,
                CurveId::Tp => p.tp,
                _ => unreachable!("only R/T s/p curves produced"),
            };
            let _ = idx;
            let v: Vec<f64> = pts.iter().map(pick).collect();
            v.into()
        };
        let mut curves = [None, None, None, None, None, None, None, None];
        curves[CurveId::Rs.index()] = Some(mk(CurveId::Rs));
        curves[CurveId::Rp.index()] = Some(mk(CurveId::Rp));
        curves[CurveId::Ts.index()] = Some(mk(CurveId::Ts));
        curves[CurveId::Tp.index()] = Some(mk(CurveId::Tp));

        Ok(SimCurves {
            angles: self.sin_theta.clone().into(),
            wavelengths: self.wavls.clone().into(),
            curves,
        })
    }
}

impl DesignContext for SmatrixContext {
    fn evaluate_merit(&self, stack: &DesignStack) -> Result<f64, String> {
        let sim = self.simulate(stack)?;
        Ok(self.spec.merit(&sim, 1e6))
    }

    fn optimize_thicknesses(
        &mut self,
        stack: &mut DesignStack,
    ) -> Result<f64, String> {
        // Collect optimize-flagged film indices and their starting values.
        let opt_indices: Vec<usize> = stack
            .films()
            .iter()
            .enumerate()
            .filter(|(_, l)| l.optimize)
            .map(|(i, _)| i)
            .collect();
        if opt_indices.is_empty() {
            return self.evaluate_merit(stack);
        }

        // Owned copies for the (Send+Sync) residual closure.
        let base_stack = stack.clone();
        let spec = self.spec.clone();
        let ctx_self = self.clone();
        let indices = opt_indices.clone();

        let residuals = move |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            let mut st = base_stack.clone();
            for (j, &i) in indices.iter().enumerate() {
                st.set_thickness(i, x[j])?;
            }
            let sim = ctx_self.simulate(&st)?;
            spec.residuals(&sim, out).map_err(|c| format!("missing curve {c:?}"))
        };

        let x0: Vec<f64> = opt_indices
            .iter()
            .map(|&i| stack.films()[i].d_nm.clamp(0.0, self.clamp_max_nm))
            .collect();
        let lb = vec![0.0f64; x0.len()];
        let ub = vec![self.clamp_max_nm; x0.len()];

        let res = levenberg_marquardt(&residuals, &x0, &lb, &ub, &self.lm)?;

        // Write back, then clamp sweep (removes sub-min, caps above-max).
        for (j, &i) in opt_indices.iter().enumerate() {
            if i < stack.films().len() {
                stack.set_thickness(i, res.x[j])?;
            }
        }
        stack.clamp_all(self.clamp_min_nm, self.clamp_max_nm);

        self.evaluate_merit(stack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics_core::cplx;
    use crate::synthesis::structure::LayerSpec;
    use crate::synthesis::merit::{ConstraintKind, MeritKey, MeritTarget, SimTransform};

    fn ar_spec(angle: f64, wl: f64) -> MeritSpec {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle, curve: CurveId::Rs });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![wl].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Linear,
            norm_factor: 1.0,
            normalized_targets: vec![0.0].into(), // R = 0 demanded
            tolerances: vec![0.01].into(),
            band: vec![].into(),
        })
        .unwrap();
        spec
    }

    /// air | L(d, n=1.2329) | sub(n=1.52) — ideal single-layer AR.
    fn ar_stack(d: f64) -> DesignStack {
        let nw = 3;
        let n_l = 1.52_f64.sqrt(); // perfect AR index at normal incidence
        let mut ambient = LayerSpec::constant("air", 1.0, 0.0, 0.0, nw);
        ambient.optimize = false;
        ambient.needle = false;
        let mut substrate = LayerSpec::constant("sub", 1.52, 0.0, 0.0, nw);
        substrate.optimize = false;
        substrate.needle = false;
        DesignStack::with_films(
            ambient,
            substrate,
            vec![LayerSpec::constant("L", n_l, 0.0, d, nw)],
        )
        .unwrap()
    }

    fn ar_ctx(clamp_max: f64) -> SmatrixContext {
        SmatrixContext {
            wavls: vec![900.0, 1000.0, 1100.0],
            sin_theta: vec![0.0],
            spec: ar_spec(0.0, 1000.0),
            clamp_min_nm: 2.0,
            clamp_max_nm: clamp_max,
            lm: LmConfig::default(),
        }
    }

    #[test]
    fn energy_conservation_lossless_stack() {
        // R + T == 1 for a lossless coherent stack at normal incidence.
        let ctx = ar_ctx(1000.0);
        let stack = ar_stack(200.0);
        let sim = ctx.simulate(&stack).unwrap();

        let rs = sim.curve(CurveId::Rs).unwrap();
        let ts = sim.curve(CurveId::Ts).unwrap();
        for k in 0..rs.len() {
            assert!((rs[k] + ts[k] - 1.0).abs() < 1e-10, "k={k} R={} T={}", rs[k], ts[k]);
        }
        // All values physical.
        for arr in [sim.curve(CurveId::Rs).unwrap(), sim.curve(CurveId::Rp).unwrap()] {
            for &v in arr.iter() {
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn quarter_wave_ar_is_zero_reflection() {
        // Exact analytic anchor: d = λ/(4n) gives R = 0 for n₁ = √(n₀n_s).
        let n_l = 1.52_f64.sqrt();
        let d_qw = 1000.0 / (4.0 * n_l);
        let ctx = ar_ctx(1000.0);
        let stack = ar_stack(d_qw);
        let mf = ctx.evaluate_merit(&stack).unwrap();
        assert!(mf < 1e-16, "mf={mf}");

        // Off-quarter-wave has nonzero reflection.
        let stack_off = ar_stack(d_qw * 0.7);
        assert!(ctx.evaluate_merit(&stack_off).unwrap() > 1e-4);
    }

    #[test]
    fn optimizer_recovers_quarter_wave_from_bad_start() {
        // Start far off (350 nm); LM must find the quarter-wave thickness.
        let n_l = 1.52_f64.sqrt();
        let d_qw = 1000.0 / (4.0 * n_l);
        let mut ctx = ar_ctx(1000.0);
        let mut stack = ar_stack(350.0);

        let mf = ctx.optimize_thicknesses(&mut stack).unwrap();
        assert!(mf < 1e-6, "mf={mf}");
        let d_final = stack.films()[0].d_nm;
        // FD-Jacobian LM lands within a fraction of a nm here (smooth 1-D).
        assert!(
            (d_final - d_qw).abs() < 1.0,
            "d_final={d_final}, expected {d_qw}"
        );
    }
}
