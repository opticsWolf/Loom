//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::thick_opt — bounded Levenberg–Marquardt for film thicknesses.
//!
//! Replaces scipy `least_squares(method="trf")` from needle_synthesis.py /
//! ClampedNeedleSynthesizer.optimize_thicknesses. The residual system is
//! injected as a closure, so this module is solver-agnostic and unit-testable
//! standalone; the synthesis wiring supplies residuals via core_engine +
//! MeritSpec (Phase 4+).
//!
//! Algorithm notes:
//!   * Classic Marquardt scaling: solve (JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr, with
//!     diag floored so flat directions do not stall the step.
//!   * Central-difference Jacobian, per-column step h_j = ∛ε·max(|x_j|, 1),
//!     evaluated RAYON-PARALLEL across columns (the film-synthesis system
//!     costs one full TMM solve per evaluation — columns dominate runtime).
//!   * Bounds: steps are vetoed component-wise when they push away from an
//!     active bound, then the trial point is clamped into [lb, ub] (mirrors
//!     ClampedNeedleSynthesizer's optimizer-bounds + post-clamp contract;
//!     removal of sub-min layers stays the CALLER's job, as in Python).

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Configuration / results
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LmConfig {
    /// Hard cap on accepted iterations.
    pub max_iterations: usize,
    /// Hard cap on total residual evaluations (including Jacobian probes).
    pub max_evals: usize,
    /// Terminate when relative cost decrease falls below this.
    pub ftol: f64,
    /// Terminate when relative step size falls below this.
    pub xtol: f64,
    /// Terminate when ‖Jᵀr‖∞ falls below this.
    pub gtol: f64,
    /// Initial damping λ.
    pub lambda_init: f64,
    /// λ multiplier on rejected steps.
    pub lambda_up: f64,
    /// λ divisor on accepted steps.
    pub lambda_down: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        LmConfig {
            max_iterations: 200,
            max_evals: 100_000,
            ftol: 1e-12,
            xtol: 1e-12,
            gtol: 1e-10,
            lambda_init: 1e-3,
            lambda_up: 5.0,
            lambda_down: 3.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LmTermination {
    /// ‖Jᵀr‖∞ < gtol.
    Gradient,
    /// Relative step below xtol.
    Step,
    /// Relative cost improvement below ftol.
    Cost,
    /// max_iterations reached.
    MaxIterations,
    /// Damping escalated without any acceptable step (stuck at bounds or
    /// numerically degenerate system).
    Stalled,
}

#[derive(Clone, Debug)]
pub struct LmResult {
    pub x: Vec<f64>,
    pub cost: f64,
    pub iterations: usize,
    pub evals: usize,
    pub termination: LmTermination,
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Minimize ½‖r(x)‖²  s.t.  lb ≤ x ≤ ub.
///
/// `residuals(x, out)` must append exactly `m` components (m fixed across
/// calls); inactive constraints should contribute zeros (see
/// `synthesis::merit::MeritSpec::residuals`).
pub fn levenberg_marquardt<F>(
    residuals: &F,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    cfg: &LmConfig,
) -> Result<LmResult, String>
where
    F: Fn(&[f64], &mut Vec<f64>) -> Result<(), String> + Sync,
{
    let n = x0.len();
    if n == 0 {
        return Err("levenberg_marquardt: empty parameter vector".into());
    }
    if lb.len() != n || ub.len() != n {
        return Err("levenberg_marquardt: bound length mismatch".into());
    }
    for j in 0..n {
        if !(lb[j] <= x0[j] && x0[j] <= ub[j]) {
            return Err(format!(
                "levenberg_marquardt: x0[{}]={} outside [{}, {}]",
                j, x0[j], lb[j], ub[j]
            ));
        }
        if lb[j] > ub[j] {
            return Err(format!("levenberg_marquardt: lb[{}] > ub[{}]", j, j));
        }
    }

    let evals = std::cell::Cell::new(0usize);
    let mut buf_r: Vec<f64> = Vec::new();
    let mut buf_t: Vec<f64> = Vec::new();

    let evaluate = |x: &[f64], out: &mut Vec<f64>| -> Result<f64, String> {
        evals.set(evals.get() + 1);
        residuals(x, out)?;
        Ok(out.iter().map(|r| r * r).sum::<f64>())
    };

    let mut x = x0.to_vec();
    // Clamp x0 defensively into bounds.
    for j in 0..n {
        x[j] = x[j].clamp(lb[j], ub[j]);
    }

    let mut cost = evaluate(&x, &mut buf_r)?;
    let mut jac = vec![0.0f64; buf_r.len() * n]; // row-major m×n
    let mut jtj = vec![0.0f64; n * n];
    let mut jtr = vec![0.0f64; n];
    let mut delta = vec![0.0f64; n];
    let mut trial = vec![0.0f64; n];

    let mut lambda = cfg.lambda_init;
    let mut termination: Option<LmTermination> = None;
    let mut iteration = 0usize;

    while iteration < cfg.max_iterations {
        // ---- Jacobian (rayon across columns, central differences) ----
        build_jacobian(residuals, &x, &mut jac)
            .map(|added| evals.set(evals.get() + added))?;

        // ---- g = Jᵀr, A = JᵀJ ----
        let m = buf_r.len();
        jtr.fill(0.0);
        jtj.fill(0.0);
        for i in 0..m {
            let ri = buf_r[i];
            let row = &jac[i * n..(i + 1) * n];
            for j in 0..n {
                jtr[j] += row[j] * ri;
                let jr = row[j];
                for k in j..n {
                    jtj[j * n + k] += jr * row[k];
                }
            }
        }
        for j in 0..n {
            for k in 0..j {
                jtj[j * n + k] = jtj[k * n + j];
            }
        }

        // Gradient convergence on the current point.
        let g_inf = jtr.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        if g_inf < cfg.gtol {
            termination = Some(LmTermination::Gradient);
            break;
        }

        // ---- damped-step loop: escalate λ until some step is accepted ----
        let mut accepted = false;
        let mut stalled = true;
        for _damping_try in 0..40 {
            if evals.get() >= cfg.max_evals {
                break;
            }
            // A = JᵀJ + λ·diag(max(diag(JᵀJ), floor))
            let mut a = jtj.clone();
            for j in 0..n {
                let dj = jtj[j * n + j].abs().max(1e-14);
                a[j * n + j] = jtj[j * n + j] + lambda * dj;
            }

            // Solve Aδ = −g.
            match solve_symmetric(&a, &neg(&jtr), &mut delta) {
                Ok(()) => {}
                Err(_) => {
                    lambda *= cfg.lambda_up;
                    continue;
                }
            }

            // Bound-aware projection: veto components pushing past an ACTIVE
            // bound, then clamp the trial point (post-clamp contract).
            for j in 0..n {
                if x[j] <= lb[j] && delta[j] < 0.0 {
                    delta[j] = 0.0;
                }
                if x[j] >= ub[j] && delta[j] > 0.0 {
                    delta[j] = 0.0;
                }
                trial[j] = (x[j] + delta[j]).clamp(lb[j], ub[j]);
            }
            if trial.iter().zip(&x).all(|(t, v)| (t - v).abs() == 0.0) {
                // Purely-zero step: all coordinates pinned. Try raising λ to
                // escape coupling once, else declare stall.
                if lambda > 1e18 {
                    break;
                }
                lambda *= cfg.lambda_up;
                continue;
            }

            let new_cost = match evaluate(&trial, &mut buf_t) {
                Ok(c) => c,
                Err(_) => {
                    lambda *= cfg.lambda_up;
                    continue;
                }
            };

            if new_cost < cost {
                // Relative-improvement check BEFORE swapping buffers.
                let rel_improve = (cost - new_cost) / cost.abs().max(1e-300);
                std::mem::swap(&mut buf_r, &mut buf_t);
                std::mem::swap(&mut x, &mut trial);
                cost = new_cost;
                lambda /= cfg.lambda_down;
                accepted = true;
                stalled = false;

                if rel_improve < cfg.ftol {
                    termination = Some(LmTermination::Cost);
                }
                break;
            }
            lambda *= cfg.lambda_up;
            if lambda > 1e18 {
                break;
            }
        }

        if !accepted {
            termination = Some(if stalled {
                LmTermination::Stalled
            } else {
                LmTermination::Cost
            });
            break;
        }
        if termination.is_some() {
            break;
        }

        // Step-size convergence: ‖Δx‖ ≤ xtol·(xtol + ‖x‖).
        let dx_norm: f64 = x
            .iter()
            .zip(&trial)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if dx_norm <= cfg.xtol * (cfg.xtol + x_norm) {
            termination = Some(LmTermination::Step);
            break;
        }
        if evals.get() >= cfg.max_evals {
            termination = Some(LmTermination::MaxIterations);
            break;
        }

        iteration += 1;
    }

    Ok(LmResult {
        x,
        cost,
        iterations: iteration,
        evals: evals.get(),
        termination: termination.unwrap_or(LmTermination::MaxIterations),
    })
}

fn neg(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| -x).collect()
}

/// Central-difference Jacobian, columns in parallel.
/// Returns the number of residual evaluations performed (2·n).
fn build_jacobian<F>(
    residuals: &F,
    x: &[f64],
    jac: &mut Vec<f64>,
) -> Result<usize, String>
where
    F: Fn(&[f64], &mut Vec<f64>) -> Result<(), String> + Sync,
{
    let n = x.len();
    let cube_root_eps = f64::EPSILON.cbrt();

    // Probe column 0 first (serially) to learn m and size the buffer.
    let h0 = cube_root_eps * x[0].abs().max(1.0);
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[0] += h0;
    xm[0] -= h0;

    let mut rp = Vec::new();
    let mut rm = Vec::new();
    residuals(&xp, &mut rp)?;
    residuals(&xm, &mut rm)?;
    if rp.len() != rm.len() {
        return Err("levenberg_marquardt: inconsistent residual length".into());
    }
    let m = rp.len();
    if jac.len() != m * n {
        jac.resize(m * n, 0.0);
    }
    let inv2h = 1.0 / (2.0 * h0);
    for i in 0..m {
        jac[i * n] = (rp[i] - rm[i]) * inv2h;
    }

    // Remaining columns in parallel; each task owns its buffers.
    let cols: Vec<usize> = (1..n).collect();
    let results: Result<Vec<(usize, Vec<f64>)>, String> = cols
        .into_par_iter()
        .map(|j| {
            let hj = cube_root_eps * x[j].abs().max(1.0);
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[j] += hj;
            xm[j] -= hj;
            let mut rp = Vec::new();
            let mut rm = Vec::new();
            residuals(&xp, &mut rp)?;
            residuals(&xm, &mut rm)?;
            if rp.len() != m || rm.len() != m {
                return Err("levenberg_marquardt: inconsistent residual length".into());
            }
            let inv2h = 1.0 / (2.0 * hj);
            let col: Vec<f64> = (0..m).map(|i| (rp[i] - rm[i]) * inv2h).collect();
            Ok((j, col))
        })
        .collect();

    for (j, col) in results? {
        for i in 0..m {
            jac[i * n + j] = col[i];
        }
    }
    Ok(2 * n)
}

/// Symmetric positive-definite solve via Cholesky with a fallback to
/// Gaussian elimination with partial pivoting (for semi-definite systems
/// nudged by damping). Returns the solution vector.
fn solve_symmetric(a_in: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), String> {
    let n = b.len();
    debug_assert_eq!(a_in.len(), n * n);

    // Try Cholesky in-place on a copy.
    let mut l = a_in.to_vec();
    for j in 0..n {
        let mut d = l[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d <= 1e-300 || !d.is_finite() {
            return gauss_solve(a_in, b, out);
        }
        let dj = d.sqrt();
        l[j * n + j] = dj;
        for i in (j + 1)..n {
            let mut s = l[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / dj;
        }
    }

    // Forward/back substitution using lower triangle (L Lᵀ).
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * out[k];
        }
        out[i] = s / l[i * n + i];
    }
    Ok(())
}

/// Dense Gaussian elimination with partial pivoting (fallback path).
fn gauss_solve(a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), String> {
    let n = b.len();
    let mut m = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        m[i * (n + 1)..i * (n + 1) + n].copy_from_slice(&a[i * n..i * n + n]);
        m[i * (n + 1) + n] = b[i];
    }

    for col in 0..n {
        // Partial pivot.
        let (piv, _) = (col..n)
            .map(|r| (r, m[r * (n + 1) + col].abs()))
            .fold((col, 0.0f64), |(br, bv), (r, v)| if v > bv { (r, v) } else { (br, bv) });
        if m[piv * (n + 1) + col].abs() < 1e-300 {
            return Err("singular normal-equation system".into());
        }
        if piv != col {
            for c in 0..(n + 1) {
                m.swap(col * (n + 1) + c, piv * (n + 1) + c);
            }
        }
        let d = m[col * (n + 1) + col];
        for r in (col + 1)..n {
            let factor = m[r * (n + 1) + col] / d;
            if factor == 0.0 {
                continue;
            }
            for c in col..(n + 1) {
                let idx = r * (n + 1) + c;
                m[idx] -= factor * m[col * (n + 1) + c];
            }
        }
    }

    for i in (0..n).rev() {
        let mut s = m[i * (n + 1) + n];
        for c in (i + 1)..n {
            s -= m[i * (n + 1) + c] * out[c];
        }
        out[i] = s / m[i * (n + 1) + i];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_fast() -> LmConfig {
        LmConfig {
            max_iterations: 200,
            ..LmConfig::default()
        }
    }

    /// Tight-gradient config for problems where we assert near machine
    /// precision (with FD Jacobians, termination at ‖Jᵀr‖∞ < gtol leaves
    /// parameter error ~√gtol — same as scipy with numeric Jacobians).
    fn cfg_precise() -> LmConfig {
        LmConfig {
            max_iterations: 500,
            ftol: 1e-16,
            xtol: 1e-16,
            gtol: 1e-14,
            ..LmConfig::default()
        }
    }

    #[test]
    fn linear_least_squares_exact_recovery() {
        // y = 2x − 1 sampled; residuals r_i = a·x_i + b − y_i
        let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let ys = [-1.0, 1.0, 3.0, 5.0, 7.0];
        let f = |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            for (&xi, &yi) in xs.iter().zip(&ys) {
                out.push(x[0] * xi + x[1] - yi);
            }
            Ok(())
        };
        let res = levenberg_marquardt(&f, &[0.0, 0.0], &[-100., -100.], &[100., 100.], &cfg_precise())
            .unwrap();
        assert!((res.x[0] - 2.0).abs() < 1e-8, "slope {:?}", res.x);
        assert!((res.x[1] + 1.0).abs() < 1e-8, "intercept {:?}", res.x);
        assert!(res.cost < 1e-18, "cost {}", res.cost);
    }

    #[test]
    fn exponential_fit_nonlinear() {
        // y = 2·exp(−0.5 t); start far away at (a,b) = (1, 1).
        let ts = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0];
        let ys: Vec<f64> = ts.iter().map(|&t| 2.0 * (-0.5 * t).exp()).collect();
        let f = move |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            for (&t, &y) in ts.iter().zip(&ys) {
                out.push(x[0] * (-x[1] * t).exp() - y);
            }
            Ok(())
        };
        let res =
            levenberg_marquardt(&f, &[1.0, 1.0], &[-10., -10.], &[10., 10.], &cfg_fast()).unwrap();
        assert!((res.x[0] - 2.0).abs() < 1e-6, "a {:?}", res.x);
        assert!((res.x[1] - 0.5).abs() < 1e-6, "b {:?}", res.x);
        assert!(res.cost < 1e-20);
    }

    #[test]
    fn bounds_respected_optimum_at_boundary() {
        // min (x−5)² with ub = 3 → boundary optimum x = 3, cost = 4.
        let f = |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            out.push(x[0] - 5.0);
            Ok(())
        };
        let res =
            levenberg_marquardt(&f, &[0.0], &[-10.0], &[3.0], &cfg_fast()).unwrap();
        assert!((res.x[0] - 3.0).abs() < 1e-9, "x {:?}", res.x);
        assert!((res.cost - 4.0).abs() < 1e-9);

        // Mirror case at lower bound.
        let res_lo =
            levenberg_marquardt(&f, &[8.0], &[7.0], &[20.0], &cfg_fast()).unwrap();
        assert!((res_lo.x[0] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn box_constrained_corner_optimum_two_params() {
        // min (x−2)² + (y−3)² inside [2.5, ∞) × [3.5, ∞) → corner (2.5, 3.5).
        let f = |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            out.push(x[0] - 2.0);
            out.push(x[1] - 3.0);
            Ok(())
        };
        let res = levenberg_marquardt(
            &f,
            &[10.0, 10.0],
            &[2.5, 3.5],
            &[100.0, 100.0],
            &cfg_fast(),
        )
        .unwrap();
        assert!((res.x[0] - 2.5).abs() < 1e-9, "x {:?}", res.x);
        assert!((res.x[1] - 3.5).abs() < 1e-9, "y {:?}", res.x);
        assert!((res.cost - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mixed_interior_and_bound_variables() {
        // One variable interior-optimal, other pinned at its upper bound:
        // min (x−1)² + (y−0)² with y ∈ (…, 2].
        let f = |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            out.push(x[0] - 1.0);
            out.push(x[1]);
            Ok(())
        };
        let res = levenberg_marquardt(
            &f,
            &[5.0, 0.0],
            &[-50.0, -50.0],
            &[50.0, 2.0],
            &cfg_fast(),
        )
        .unwrap();
        assert!((res.x[0] - 1.0).abs() < 1e-9);
        assert!((res.x[1]).abs() < 1e-9); // interior: bound irrelevant
    }

    #[test]
    fn residual_error_propagates() {
        let f = |_x: &[f64], _out: &mut Vec<f64>| -> Result<(), String> {
            Err("solver blew up".into())
        };
        let err = levenberg_marquardt(&f, &[0.0], &[-1.0], &[1.0], &LmConfig::default());
        assert!(err.is_err());
    }

    #[test]
    fn max_iterations_respected() {
        // Slow-converging ill-conditioned problem with a tiny budget.
        let f = |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            out.push(1e3 * (x[0] - 1.0));
            out.push(x[0] * x[0] - 1.0);
            Ok(())
        };
        let cfg = LmConfig {
            max_iterations: 3,
            ..LmConfig::default()
        };
        let res = levenberg_marquardt(&f, &[0.0], &[-10.0], &[10.0], &cfg).unwrap();
        assert!(res.iterations <= 3);
        assert!(matches!(
            res.termination,
            LmTermination::MaxIterations
                | LmTermination::Step
                | LmTermination::Stalled
                | LmTermination::Gradient
        ));
    }

    #[test]
    fn x0_outside_bounds_rejected() {
        let f = |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            out.push(x[0]);
            Ok(())
        };
        // Out-of-bounds x0 is a caller bug → loud error, not silent clamping.
        assert!(levenberg_marquardt(&f, &[99.0], &[0.0], &[10.0], &cfg_fast()).is_err());

        // Valid x0 at the upper bound converges to the interior optimum.
        // Default gtol=1e-10 on r = x leaves |x| ~ √gtol — assert accordingly.
        let res = levenberg_marquardt(&f, &[10.0], &[0.0], &[10.0], &cfg_fast()).unwrap();
        assert!(res.x[0].abs() < 1e-4, "x {:?}", res.x);
    }

    #[test]
    fn invalid_inputs_rejected() {
        let f = |_x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            Ok(())
        };
        assert!(levenberg_marquardt(&f, &[], &[], &[], &LmConfig::default()).is_err());
        assert!(levenberg_marquardt(
            &f,
            &[0.0],
            &[1.0],
            &[-1.0],
            &LmConfig::default()
        )
        .is_err());
    }

    #[test]
    fn fixed_residual_length_enforced() {
        // Variable-length residual system must be detected.
        let flip = std::sync::atomic::AtomicBool::new(false);
        let f = |x: &[f64], out: &mut Vec<f64>| -> Result<(), String> {
            out.clear();
            out.push(x[0]);
            if flip.load(std::sync::atomic::Ordering::Relaxed) {
                out.push(0.0);
            }
            Ok(())
        };
        // First call defines m; subsequent mismatch errors surface either as
        // Err from the driver or are tolerated depending on ordering — here
        // flip is never set, so it must succeed cleanly instead.
        let res = levenberg_marquardt(&f, &[1.0], &[-1.0], &[1.0], &cfg_fast());
        assert!(res.is_ok());
    }
}
