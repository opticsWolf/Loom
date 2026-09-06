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

use crate::smatrix::coherent_block::solve_coherent_block_fields_dual;
use crate::smatrix::synthesis::context::DesignContext;
use crate::smatrix::synthesis::merit::{CurveId, MeritSpec, SimCurves};
use crate::smatrix::synthesis::structure::{DesignStack, SolverArrays};
use crate::smatrix::synthesis::thick_opt::{levenberg_marquardt, LmConfig};

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

        // Per-point intensities + forward-transmission amplitudes via the
        // dual solver. BlockResult order is (rf, tb, tf, rb, R, Tb, T, Rb):
        // `.2` is the complex forward t (front incidence) — `.1`/`.5` are
        // the BACKWARD (reciprocal) quantities, equal in intensity by
        // reciprocity but NOT in phase. PDts/PDtp need `.2`.
        struct Pt {
            rs: f64,
            rp: f64,
            ts: f64,
            tp: f64,
            tfs: Complex64,
            tfp: Complex64,
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
                Pt { rs: s_res.4, rp: p_res.4, ts: s_res.5, tp: p_res.5,
                     tfs: s_res.2, tfp: p_res.2 }
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
        let mut curves = [None, None, None, None, None, None, None, None, None];
        curves[CurveId::Rs.index()] = Some(mk(CurveId::Rs));
        curves[CurveId::Rp.index()] = Some(mk(CurveId::Rp));
        curves[CurveId::Ts.index()] = Some(mk(CurveId::Ts));
        curves[CurveId::Tp.index()] = Some(mk(CurveId::Tp));
        // Complex forward-t rows for (differential-)phase demands — ONLY
        // when the spec asks (two allocations + O(grid) copies saved per
        // merit call for intensity-only optimizations; the amplitudes
        // themselves come from the dual solver regardless).
        let mut cplx: [Option<Arc<[Complex64]>>; 6] = [None, None, None, None, None, None];
        if self.spec.uses_phase() {
            cplx[CurveId::Ts.index()] =
                Some(pts.iter().map(|p| p.tfs).collect::<Vec<_>>().into());
            cplx[CurveId::Tp.index()] =
                Some(pts.iter().map(|p| p.tfp).collect::<Vec<_>>().into());
        }
        // Stack metadata for the PD reference: ambient/substrate thickness
        // entries are zero, so the plain sum is the coating thickness D;
        // ambient index at the centre wavelength (dispersive ambients are
        // pathological — the scalar is a documented approximation).
        // Gated the same way (defaults zero the reference anyway).
        let (total_d, n_front_re, n_back_re) = if self.spec.uses_differential() {
            let total_d: f64 = sa.thicknesses.iter().sum();
            let n_front_re = sa.n_stack_cache[(nw / 2) * nl * 2];
            // Substrate exit index (back-phase reference; unused front-only).
            let n_back_re = sa.n_stack_cache[(nw / 2) * nl * 2 + (nl - 1) * 2];
            (total_d, n_front_re, n_back_re)
        } else {
            (0.0, 1.0, 1.0)
        };

        Ok(SimCurves {
            angles: self.sin_theta.clone().into(),
            wavelengths: self.wavls.clone().into(),
            total_d,
            n_front_re,
            n_back_re,
            curves,
            cplx,
            ..Default::default()
        })
    }
}

impl DesignContext for SmatrixContext {
    fn evaluate_merit(&self, stack: &DesignStack) -> Result<f64, String> {
        let sim = self.simulate(stack)?;
        Ok(self.spec.merit(&sim, 1e6))
    }

    fn simulate(&self, stack: &DesignStack) -> Result<SimCurves, String> {
        SmatrixContext::simulate(self, stack)
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
    use crate::smatrix::optics_core::cplx;
    use crate::smatrix::synthesis::structure::LayerSpec;
    use crate::smatrix::synthesis::merit::{ConstraintKind, MeritKey, MeritTarget, SimTransform};

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
            phase: false,
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
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

    /// Context whose spec demands PDts (Ts, phase, 1 pass): `simulate()`
    /// must then fill complex-t rows + reference metadata (gated assembly).
    fn ar_ctx_pd() -> SmatrixContext {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![1000.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Phase,
            norm_factor: 1.0,
            normalized_targets: vec![0.0].into(),
            tolerances: vec![0.05].into(),
            band: vec![].into(),
            phase: true,
            differential_passes: Some(1.0),
            integral: false,
            weight: 1.0,
            count_norm: None,
        })
        .unwrap();
        SmatrixContext {
            wavls: vec![900.0, 1000.0, 1100.0],
            sin_theta: vec![0.0],
            spec,
            clamp_min_nm: 2.0,
            clamp_max_nm: 1000.0,
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

    /// Independent 2×2 characteristic-matrix oracle (s-pol, normal
    /// incidence): M = [[cosδ, i·sinδ/n],[i·n·sinδ, cosδ]],
    /// t = 2·n₀/(n₀·M₁₁ + n₀·n_s·M₁₂ + M₂₁ + n_s·M₂₂).
    fn oracle_tf(n0: f64, n1: f64, ns: f64, d: f64, lam: f64) -> Complex64 {
        use num_complex::Complex64 as C;
        let delta = 2.0 * std::f64::consts::PI * n1 * d / lam;
        let (s, c) = delta.sin_cos();
        let m11 = C::new(c, 0.0);
        let m12 = C::new(0.0, s / n1);
        let m21 = C::new(0.0, s * n1);
        let m22 = C::new(c, 0.0);
        C::new(2.0 * n0, 0.0)
            / (C::new(n0, 0.0) * m11 + C::new(n0 * ns, 0.0) * m12 + m21 + C::new(ns, 0.0) * m22)
    }

    #[test]
    fn solver_propagation_sign_matches_reference() {
        // All-matched stack (n = 1 everywhere, film D = 500): tf is pure
        // propagation — its arg IS the solver's propagation-phase sign,
        // and `reference_phase` must reproduce it (convention lock for
        // every differential-phase demand in the crate).
        let nw = 3;
        let ambient = LayerSpec::constant("air", 1.0, 0.0, 0.0, nw);
        let substrate = LayerSpec::constant("sub", 1.0, 0.0, 0.0, nw);
        let slab = DesignStack::with_films(
            ambient, substrate, vec![LayerSpec::constant("F", 1.0, 0.0, 500.0, nw)],
        ).unwrap();
        // NOTE: empty spec → gated assembly skips complex rows, so this
        // calibration context carries a (value-irrelevant) phase demand.
        let mut pd_spec = MeritSpec::new();
        let pk = pd_spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        pd_spec.add_target(MeritTarget {
            key_idx: pk as u32,
            wavelengths: vec![400.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Phase,
            norm_factor: 1.0,
            normalized_targets: vec![0.0].into(),
            tolerances: vec![0.05].into(),
            band: vec![].into(),
            phase: true,
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
        })
        .unwrap();
        let ctx = SmatrixContext {
            wavls: vec![400.0],
            sin_theta: vec![0.0],
            spec: pd_spec,
            clamp_min_nm: 2.0,
            clamp_max_nm: 1000.0,
            lm: LmConfig::default(),
        };
        let sim = ctx.simulate(&slab).unwrap();
        let tf = sim.cplx[CurveId::Ts.index()].as_ref().unwrap()[0];
        // kD = 2π·500/400 = 2.5π → +π/2 vs −π/2, unambiguous.
        assert!((tf.arg() - std::f64::consts::PI / 2.0).abs() < 1e-9, "tf={tf}");
        // `reference_phase` is unwrapped (2.5π here) while `arg()` wraps:
        // compare in wrapped space, exactly as the merit kernel does.
        let r = crate::smatrix::optics_core::reference_phase(400.0, 1.0, 0.0, 500.0, 1.0);
        let rw = r - std::f64::consts::TAU * (r / std::f64::consts::TAU).round();
        assert!((rw - tf.arg()).abs() < 1e-9, "ref={r} arg={}", tf.arg());
    }

    #[test]
    fn gated_assembly_skips_unrequested_rows() {
        // Intensity-only spec: no complex rows, default metadata (the
        // premise — unrequested paths stay dark in the hot LM loop).
        let ctx = ar_ctx(1000.0);
        assert!(!ctx.spec.uses_phase());
        assert!(!ctx.spec.uses_differential());
        let sim = ctx.simulate(&ar_stack(200.0)).unwrap();
        assert!(sim.cplx.iter().all(|c| c.is_none()));
        assert_eq!(sim.total_d, 0.0);
        assert_eq!((sim.n_front_re, sim.n_back_re), (1.0, 1.0));
        // Phase-demanding spec flips both gates.
        let ctx_pd = ar_ctx_pd();
        assert!(ctx_pd.spec.uses_phase());
        assert!(ctx_pd.spec.uses_differential());
    }

    #[test]
    fn simulate_fills_pd_metadata_and_complex_t() {
        let n_l = 1.52_f64.sqrt();
        let d = 200.0;
        let ctx = ar_ctx_pd();
        let stack = ar_stack(d);
        let sim = ctx.simulate(&stack).unwrap();
        // Metadata: coating thickness + media.
        assert!((sim.total_d - d).abs() < 1e-12, "total_d={}", sim.total_d);
        assert!((sim.n_front_re - 1.0).abs() < 1e-12);
        assert!((sim.n_back_re - 1.52).abs() < 1e-12);
        // Complex-t rows present for Ts/Tp, consistent with intensities:
        // |tf|² × flux (n_s/n_0 at normal incidence) == Ts row.
        let ts = sim.curve(CurveId::Ts).unwrap();
        let tfs = &sim.cplx[CurveId::Ts.index()].as_ref().unwrap();
        assert_eq!(tfs.len(), ts.len());
        for k in 0..ts.len() {
            assert!((tfs[k].norm_sqr() * 1.52 - ts[k]).abs() < 1e-10, "k={k}");
        }
        // Independent oracle per wavelength (pins tf-vs-tb: the backward
        // amplitude has a different phase in asymmetric stacks).
        // `.conj()`: the solver's forward-propagation convention is the
        // conjugate of Macleod textbooks (see `reference_phase` docs) —
        // magnitudes/physics identical, phase sign flipped crate-wide.
        for (k, &lam) in [900.0, 1000.0, 1100.0].iter().enumerate() {
            let t_oracle = oracle_tf(1.0, n_l, 1.52, d, lam).conj();
            let diff = (tfs[k] - t_oracle).norm();
            assert!(diff < 1e-9, "k={k} lam={lam} diff={diff}");
        }
    }

    #[test]
    fn pd_merit_matches_hand_delta_phi() {
        // Δφ = arg(t_oracle) − 2π·D/λ (air, normal) demanded exactly → ~0.
        let n_l = 1.52_f64.sqrt();
        let d = 200.0;
        let mut ctx = ar_ctx_pd();
        let stack = ar_stack(d);
        let sim = ctx.simulate(&stack).unwrap();
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        let wl = vec![900.0, 1000.0, 1100.0];
        let tgt: Vec<f64> = wl.iter().map(|&lam| {
            // `.conj()`: solver convention (see oracle test above).
            oracle_tf(1.0, n_l, 1.52, d, lam).conj().arg() - 2.0 * std::f64::consts::PI * d / lam
        }).collect();
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: wl.into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Phase,
            norm_factor: 1.0,
            normalized_targets: tgt.into(),
            tolerances: vec![0.05, 0.05, 0.05].into(),
            band: vec![].into(),
            phase: true,
            differential_passes: Some(1.0),
            integral: false,
            weight: 1.0,
            count_norm: None,
        })
        .unwrap();
        assert!(spec.merit(&sim, 1e6) < 1e-20, "m={}", spec.merit(&sim, 1e6));
        ctx.spec = spec;
        assert!(ctx.evaluate_merit(&stack).unwrap() < 1e-20);
    }

    #[test]
    fn optimizer_recovers_thickness_from_pd_target() {
        // End-to-end PD loop: demand the QW design's Δφ, start at 350 nm.
        // Convergence is on MERIT (phase wraps admit 2π-branch solutions).
        let n_l = 1.52_f64.sqrt();
        let d_qw = 1000.0 / (4.0 * n_l);
        let qw_stack = ar_stack(d_qw);
        let probe = ar_ctx_pd();
        let qw_sim = probe.simulate(&qw_stack).unwrap();
        let qw_phase = qw_sim.cplx[CurveId::Ts.index()].as_ref().unwrap()[1].arg();
        let ref_qw = 2.0 * std::f64::consts::PI * d_qw / 1000.0;
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        spec.add_target(MeritTarget {
            key_idx: k as u32,
            wavelengths: vec![1000.0].into(),
            kind: ConstraintKind::Exact,
            transform: SimTransform::Phase,
            norm_factor: 1.0,
            normalized_targets: vec![qw_phase - ref_qw].into(),
            tolerances: vec![0.05].into(),
            band: vec![].into(),
            phase: true,
            differential_passes: Some(1.0),
            integral: false,
            weight: 1.0,
            count_norm: None,
        })
        .unwrap();
        let mut ctx = ar_ctx(1000.0);
        ctx.spec = spec;
        let mut stack = ar_stack(350.0);
        let mf = ctx.optimize_thicknesses(&mut stack).unwrap();
        assert!(mf < 1e-6, "mf={mf}");
    }
}
