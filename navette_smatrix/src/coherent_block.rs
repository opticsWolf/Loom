use num_complex::{Complex64, ComplexFloat};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::f64::consts::PI;

use crate::optics_core::{cexp_fast, csqrt_fast, redheffer_product_complex_field_inner, w_function_inner};

const POL_S: i32 = 0;
const LOG_MIN: f64 = 1e-100;
const EPS_COS: f64 = 1e-12;

/// Result of a coherent block solve:
/// (r_front, t_back, t_fwd, r_back, R_front, T_back, T_fwd, R_back)
pub type BlockResult = (
    Complex64,
    Complex64,
    Complex64,
    Complex64,
    f64,
    f64,
    f64,
    f64,
);

/// Fast algebraic principal complex square root (`re >= 0` branch).
///
/// `num_complex`'s general-case `sqrt` routes through `to_polar`/`from_polar`,
/// i.e. an `atan2` plus a `sin`/`cos` — three transcendentals to take a root.
/// This computes the same principal value from `|z|` and two real `sqrt`s,
/// using the larger component first to avoid cancellation. Differs from the
/// polar result by at most ~1 ULP, and `cos θ` is taken every interface so the
/// saving compounds. Both solvers normalize the sign afterwards (`im >= 0`),
/// so the branch convention matches the reference exactly.
///
/// `|z|` is taken via `hypot`, which is correctly rounded and avoids the
/// `a*a + b*b` intermediate (a couple of ULP of error plus over/underflow
/// risk). The earlier naive form was chosen to mirror the reference's
/// Single-polarization coherent block solver, monomorphized on polarization.
///
/// `IS_S` is a const generic, so the s/p admittance choice is resolved at
/// compile time: each instantiation is a branch-free loop with the dead
/// polarization's code (notably the p-only `cos` magnitude guard) eliminated
/// outright. This is what lets the single-pol path reach the same per-pol
/// codegen as the dual solver (whose admittance calls already take literal
/// bools and so constant-fold). Arithmetic is byte-identical to the reference;
/// indexing matches the `@njit` implementation exactly.
#[inline]
fn solve_pol_specialized<const IS_S: bool>(
    start_idx: usize,
    end_idx: usize,
    n_slice: &[Complex64],
    inv_n_slice: &[Complex64],
    d_slice: &[f64],
    rv_slice: &[f64],
    rt_slice: &[i32],
    lam: f64,
    nsin_fi: Complex64,
) -> BlockResult {
    let two_pi_lam = 2.0 * PI / lam;

    #[inline(always)]
    fn admittance<const S: bool>(n: Complex64, cos: Complex64) -> Complex64 {
        if S {
            n * cos
        } else {
            let c = if cos.norm() < EPS_COS {
                Complex64::new(EPS_COS, 0.0)
            } else {
                cos
            };
            n / c
        }
    }

    let mut sg_rf = Complex64::new(0.0, 0.0);
    let mut sg_tb = Complex64::new(1.0, 0.0);
    let mut sg_tf = Complex64::new(1.0, 0.0);
    let mut sg_rb = Complex64::new(0.0, 0.0);

    let mut n_curr = n_slice[start_idx];
    let mut cos_curr = {
        let r0 = nsin_fi * inv_n_slice[start_idx];
        let v = Complex64::new(1.0, 0.0) - r0 * r0;
        let c = csqrt_fast(v);
        if c.im < 0.0 { -c } else { c }
    };
    let mut y_curr = admittance::<IS_S>(n_curr, cos_curr);
    let y_first = y_curr;

    let mut idx = start_idx;
    while idx < end_idx {
        let i_next = idx + 1;
        let n_next = n_slice[i_next];
        let sigma = rv_slice[i_next];
        let rtype = rt_slice[i_next];

        let cos_next = {
            let rr = nsin_fi * inv_n_slice[i_next];
            let v = Complex64::new(1.0, 0.0) - rr * rr;
            let c = csqrt_fast(v);
            if c.im < 0.0 { -c } else { c }
        };
        let y_next = admittance::<IS_S>(n_next, cos_next);

        let den = y_curr + y_next;
        let den_safe = if den.norm() < LOG_MIN {
            Complex64::new(LOG_MIN, LOG_MIN)
        } else {
            den
        };
        let inv_den = den_safe.recip();

        let r12 = (y_curr - y_next) * inv_den;
        let t12 = y_curr * 2.0 * inv_den;
        let t21 = y_next * 2.0 * inv_den;
        let r21 = -r12;

        let (r12_mod, r21_mod, t12_mod, t21_mod) = if rtype == 0 {
            (r12, r21, t12, t21)
        } else if rtype == 5 {
            let kz1 = two_pi_lam * n_curr * cos_curr;
            let kz2 = two_pi_lam * n_next * cos_next;
            let f = (-2.0 * kz1 * kz2 * sigma * sigma).exp();
            (r12 * f, r21 * f, t12 * f, t21 * f)
        } else {
            let kz1 = two_pi_lam * n_curr * cos_curr;
            let kz2 = two_pi_lam * n_next * cos_next;
            let al = w_function_inner(2.0 * kz1 * sigma, rtype);
            let be = w_function_inner(2.0 * kz2 * sigma, rtype);
            let ga = w_function_inner((kz1 - kz2) * sigma, rtype);
            (r12 * al, r21 * be, t12 * ga, t21 * ga)
        };

        let (new_rf, new_tb, new_tf, new_rb) = redheffer_product_complex_field_inner(
            sg_rf, sg_tb, sg_tf, sg_rb, r12_mod, t21_mod, t12_mod, r21_mod,
        );
        sg_rf = new_rf;
        sg_tb = new_tb;
        sg_tf = new_tf;
        sg_rb = new_rb;

        if idx + 1 < end_idx {
            let d = d_slice[i_next];
            if d > 1e-12 {
                let mut beta = two_pi_lam * d * n_next * cos_next;
                if beta.im < 0.0 {
                    beta = Complex64::new(beta.re, -beta.im);
                }
                let phi = cexp_fast(Complex64::new(0.0, 1.0) * beta);
                sg_rb = sg_rb * phi * phi;
                sg_tb *= phi;
                sg_tf *= phi;
            }
        }

        n_curr = n_next;
        cos_curr = cos_next;
        y_curr = y_next;
        idx += 1;
    }

    let r_front = sg_rf.norm_sqr();
    let r_back = sg_rb.norm_sqr();

    let mut real_y_first = y_first.re;
    let mut real_y_last = y_curr.re;
    if real_y_first < 1e-15 {
        real_y_first = 0.0;
    }
    if real_y_last < 1e-15 {
        real_y_last = 0.0;
    }
    let factor_fwd = if real_y_first > 1e-15 {
        real_y_last / real_y_first
    } else {
        0.0
    };
    let factor_back = if real_y_last > 1e-15 {
        real_y_first / real_y_last
    } else {
        0.0
    };

    let t_fwd = sg_tf.norm_sqr() * factor_fwd;
    let t_back = sg_tb.norm_sqr() * factor_back;

    (sg_rf, sg_tb, sg_tf, sg_rb, r_front, t_back, t_fwd, r_back)
}

/// Pure-Rust coherent block solver (single polarization).
///
/// Thin dispatcher over the polarization-monomorphized kernel. `n_slice`,
/// `d_slice`, `rv_slice`, `rt_slice` are the FULL per-wavelength arrays; the
/// block spans interfaces `start_idx -> start_idx+1` through `end_idx-1 ->
/// end_idx`. No per-call allocation, no NumPy round-trip.
#[inline]
pub fn solve_coherent_block_fields_inner(
    start_idx: usize,
    end_idx: usize,
    n_slice: &[Complex64],
    inv_n_slice: &[Complex64],
    d_slice: &[f64],
    rv_slice: &[f64],
    rt_slice: &[i32],
    lam: f64,
    nsin_fi: Complex64,
    pol: i32,
) -> BlockResult {
    if pol == POL_S {
        solve_pol_specialized::<true>(
            start_idx, end_idx, n_slice, inv_n_slice, d_slice, rv_slice, rt_slice, lam, nsin_fi,
        )
    } else {
        solve_pol_specialized::<false>(
            start_idx, end_idx, n_slice, inv_n_slice, d_slice, rv_slice, rt_slice, lam, nsin_fi,
        )
    }
}

/// Dual-polarization coherent block solver.
///
/// Produces the s- and p-polarization results in a SINGLE interface sweep.
/// The polarization-independent work — branch-safe `cos θ`, the layer
/// propagation phase `phi`, and the roughness form factors — is computed once
/// and shared, roughly halving the transcendental load versus calling the
/// single-pol solver twice. The per-polarization arithmetic is byte-identical
/// to two separate calls.
///
/// Returns `(s_result, p_result)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn solve_coherent_block_fields_dual(
    start_idx: usize,
    end_idx: usize,
    n_slice: &[Complex64],
    inv_n_slice: &[Complex64],
    d_slice: &[f64],
    rv_slice: &[f64],
    rt_slice: &[i32],
    lam: f64,
    nsin_fi: Complex64,
) -> (BlockResult, BlockResult) {
    let two_pi_lam = 2.0 * PI / lam;

    // S-matrix accumulators for both polarizations.
    let (mut s_rf, mut s_tb, mut s_tf, mut s_rb) = (
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    );
    let (mut p_rf, mut p_tb, mut p_tf, mut p_rb) = (
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    );

    let mut n_curr = n_slice[start_idx];
    let mut cos_curr = {
        let r0 = nsin_fi * inv_n_slice[start_idx];
        let v = Complex64::new(1.0, 0.0) - r0 * r0;
        let c = csqrt_fast(v);
        if c.im < 0.0 { -c } else { c }
    };

    #[inline(always)]
    fn admittance(is_s: bool, n: Complex64, cos: Complex64) -> Complex64 {
        if is_s {
            n * cos
        } else {
            let c = if cos.norm() < EPS_COS {
                Complex64::new(EPS_COS, 0.0)
            } else {
                cos
            };
            n / c
        }
    }

    let mut ys_curr = admittance(true, n_curr, cos_curr);
    let mut yp_curr = admittance(false, n_curr, cos_curr);
    let ys_first = ys_curr;
    let yp_first = yp_curr;

    let mut idx = start_idx;
    while idx < end_idx {
        let i_next = idx + 1;
        let n_next = n_slice[i_next];
        let sigma = rv_slice[i_next];
        let rtype = rt_slice[i_next];

        let cos_next = {
            let rr = nsin_fi * inv_n_slice[i_next];
            let v = Complex64::new(1.0, 0.0) - rr * rr;
            let c = csqrt_fast(v);
            if c.im < 0.0 { -c } else { c }
        };

        let ys_next = admittance(true, n_next, cos_next);
        let yp_next = admittance(false, n_next, cos_next);

        // Shared roughness factors (polarization-independent).
        // (rough_r12, rough_r21, rough_t) multipliers applied below.
        let (rg_r12, rg_r21, rg_t): (Complex64, Complex64, Complex64) = if rtype == 0 {
            (
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            )
        } else if rtype == 5 {
            let kz1 = two_pi_lam * n_curr * cos_curr;
            let kz2 = two_pi_lam * n_next * cos_next;
            let f = (-2.0 * kz1 * kz2 * sigma * sigma).exp();
            (f, f, f)
        } else {
            let kz1 = two_pi_lam * n_curr * cos_curr;
            let kz2 = two_pi_lam * n_next * cos_next;
            let al = w_function_inner(2.0 * kz1 * sigma, rtype);
            let be = w_function_inner(2.0 * kz2 * sigma, rtype);
            let ga = w_function_inner((kz1 - kz2) * sigma, rtype);
            (al, be, ga)
        };

        // Shared propagation phase for the next layer.
        let phi = if idx + 1 < end_idx {
            let d = d_slice[i_next];
            if d > 1e-12 {
                let mut beta = two_pi_lam * d * n_next * cos_next;
                if beta.im < 0.0 {
                    beta = Complex64::new(beta.re, -beta.im);
                }
                Some(cexp_fast(Complex64::new(0.0, 1.0) * beta))
            } else {
                None
            }
        } else {
            None
        };

        // --- s-polarization interface + propagation ---
        {
            let den = ys_curr + ys_next;
            let den_safe = if den.norm() < LOG_MIN {
                Complex64::new(LOG_MIN, LOG_MIN)
            } else {
                den
            };
            let inv = den_safe.recip();
            let r12 = (ys_curr - ys_next) * inv * rg_r12;
            let r21 = -((ys_curr - ys_next) * inv) * rg_r21;
            let t12 = ys_curr * 2.0 * inv * rg_t;
            let t21 = ys_next * 2.0 * inv * rg_t;
            let (a, b, c, d) = redheffer_product_complex_field_inner(
                s_rf, s_tb, s_tf, s_rb, r12, t21, t12, r21,
            );
            s_rf = a; s_tb = b; s_tf = c; s_rb = d;
            if let Some(phi) = phi {
                s_rb = s_rb * phi * phi;
                s_tb *= phi;
                s_tf *= phi;
            }
        }
        // --- p-polarization interface + propagation ---
        {
            let den = yp_curr + yp_next;
            let den_safe = if den.norm() < LOG_MIN {
                Complex64::new(LOG_MIN, LOG_MIN)
            } else {
                den
            };
            let inv = den_safe.recip();
            let r12 = (yp_curr - yp_next) * inv * rg_r12;
            let r21 = -((yp_curr - yp_next) * inv) * rg_r21;
            let t12 = yp_curr * 2.0 * inv * rg_t;
            let t21 = yp_next * 2.0 * inv * rg_t;
            let (a, b, c, d) = redheffer_product_complex_field_inner(
                p_rf, p_tb, p_tf, p_rb, r12, t21, t12, r21,
            );
            p_rf = a; p_tb = b; p_tf = c; p_rb = d;
            if let Some(phi) = phi {
                p_rb = p_rb * phi * phi;
                p_tb *= phi;
                p_tf *= phi;
            }
        }

        n_curr = n_next;
        cos_curr = cos_next;
        ys_curr = ys_next;
        yp_curr = yp_next;
        idx += 1;
    }

    let finalize = |rf: Complex64, tb: Complex64, tf: Complex64, rb: Complex64,
                    y_first: Complex64, y_last: Complex64| -> BlockResult {
        let r_front = rf.norm_sqr();
        let r_back = rb.norm_sqr();
        let mut ry0 = y_first.re;
        let mut ry1 = y_last.re;
        if ry0 < 1e-15 { ry0 = 0.0; }
        if ry1 < 1e-15 { ry1 = 0.0; }
        let f_fwd = if ry0 > 1e-15 { ry1 / ry0 } else { 0.0 };
        let f_back = if ry1 > 1e-15 { ry0 / ry1 } else { 0.0 };
        (rf, tb, tf, rb, r_front, tb.norm_sqr() * f_back, tf.norm_sqr() * f_fwd, r_back)
    };

    let s_res = finalize(s_rf, s_tb, s_tf, s_rb, ys_first, ys_curr);
    let p_res = finalize(p_rf, p_tb, p_tf, p_rb, yp_first, yp_curr);
    (s_res, p_res)
}

/// Python-facing wrapper. Same contract as the reference `@njit` function:
/// pass the FULL per-wavelength arrays plus absolute `start_idx`/`end_idx`.
#[pyfunction]
#[pyo3(name = "solve_coherent_block_fields")]
#[allow(clippy::too_many_arguments)]
pub fn solve_coherent_block_fields(
    start_idx: i32,
    end_idx: i32,
    n_stack: PyReadonlyArray1<Complex64>,
    d_stack: PyReadonlyArray1<f64>,
    rough_vals: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    lam: f64,
    nsin_fi: Complex64,
    pol: i32,
) -> PyResult<BlockResult> {
    let n_slice = n_stack.as_slice()?;
    let d_slice = d_stack.as_slice()?;
    let rv_slice = rough_vals.as_slice()?;
    let rt_slice = rough_types.as_slice()?;

    // Per-layer reciprocals (1/n). In the engines these are precomputed once
    // per wavelength and reused across all angles; here it is a single call.
    let inv_n: Vec<Complex64> = n_slice.iter().map(|n| n.recip()).collect();

    Ok(solve_coherent_block_fields_inner(
        start_idx as usize,
        end_idx as usize,
        n_slice,
        &inv_n,
        d_slice,
        rv_slice,
        rt_slice,
        lam,
        nsin_fi,
        pol,
    ))
}
