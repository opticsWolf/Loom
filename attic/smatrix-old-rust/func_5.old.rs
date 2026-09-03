// func_5.rs
use num_complex::Complex64;
use num_complex::ComplexFloat;
use numpy::{PyArray, PyArrayMethods};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::func_2::redheffer_product_real_inner;
use crate::func_3::{solve_coherent_block_fields_dual, solve_coherent_block_fields_inner};

/// Combine a coherent block's intensity matrix into the running global matrix,
/// then apply an incoherent gap if the block boundary is flagged. Returns the
/// updated (R_front,T_back,T_fwd,R_back) accumulator.
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_block(
    ig: (f64, f64, f64, f64),
    rf: f64,
    tb: f64,
    tf: f64,
    rb: f64,
) -> (f64, f64, f64, f64) {
    redheffer_product_real_inner(ig.0, ig.1, ig.2, ig.3, rf, tb, tf, rb)
}

/// Solve all coherent blocks + incoherent gaps for a single polarization.
/// Returns (R_front, T_fwd).
#[inline]
#[allow(clippy::too_many_arguments)]
fn solve_one_pol(
    idx_n: usize,
    lam: f64,
    nsinfi: Complex64,
    n_stack: &[Complex64],
    inv_n_stack: &[Complex64],
    thick_slice: &[f64],
    inc_flags_slice: &[i32],
    rough_types_slice: &[i32],
    rough_vals_slice: &[f64],
    pol: i32,
) -> (f64, f64) {
    let mut ig = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut current_idx = 0usize;

    while current_idx < idx_n {
        let mut next_incoh = current_idx + 1;
        while next_incoh < idx_n && inc_flags_slice[next_incoh] == 0 {
            next_incoh += 1;
        }

        let (_, _, _, _, rf, tb, tf, rb) = solve_coherent_block_fields_inner(
            current_idx,
            next_incoh,
            n_stack,
            inv_n_stack,
            thick_slice,
            rough_vals_slice,
            rough_types_slice,
            lam,
            nsinfi,
            pol,
        );

        ig = redheffer_product_real_inner(ig.0, ig.1, ig.2, ig.3, rf, tb, tf, rb);

        if next_incoh < idx_n && inc_flags_slice[next_incoh] == 1 {
            let n_inc = n_stack[next_incoh];
            let d_inc = thick_slice[next_incoh];
            let rinc = nsinfi * inv_n_stack[next_incoh];
            let val_inc = Complex64::new(1.0, 0.0) - rinc * rinc;
            let mut cos_inc = val_inc.sqrt();
            if cos_inc.im < 0.0 {
                cos_inc = -cos_inc;
            }
            let beta_imag = (2.0 * PI * d_inc / lam) * (n_inc * cos_inc).im;
            let beta_imag = if beta_imag < 0.0 { 0.0 } else { beta_imag };
            let trans_factor = (-2.0 * beta_imag).exp();

            ig = redheffer_product_real_inner(
                ig.0, ig.1, ig.2, ig.3, 0.0, trans_factor, trans_factor, 0.0,
            );
        }

        current_idx = next_incoh;
    }

    (ig.0, ig.2)
}

/// Solve s AND p in one block sweep (used for unpolarized 'u' mode).
/// Returns (rs, rp, ts, tp).
#[inline]
#[allow(clippy::too_many_arguments)]
fn solve_both_pol(
    idx_n: usize,
    lam: f64,
    nsinfi: Complex64,
    n_stack: &[Complex64],
    inv_n_stack: &[Complex64],
    thick_slice: &[f64],
    inc_flags_slice: &[i32],
    rough_types_slice: &[i32],
    rough_vals_slice: &[f64],
) -> (f64, f64, f64, f64) {
    let mut ig_s = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut ig_p = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut current_idx = 0usize;

    while current_idx < idx_n {
        let mut next_incoh = current_idx + 1;
        while next_incoh < idx_n && inc_flags_slice[next_incoh] == 0 {
            next_incoh += 1;
        }

        let (s_res, p_res) = solve_coherent_block_fields_dual(
            current_idx, next_incoh, n_stack, inv_n_stack, thick_slice, rough_vals_slice,
            rough_types_slice, lam, nsinfi,
        );
        let (_, _, _, _, rf_s, tb_s, tf_s, rb_s) = s_res;
        let (_, _, _, _, rf_p, tb_p, tf_p, rb_p) = p_res;
        ig_s = accumulate_block(ig_s, rf_s, tb_s, tf_s, rb_s);
        ig_p = accumulate_block(ig_p, rf_p, tb_p, tf_p, rb_p);

        if next_incoh < idx_n && inc_flags_slice[next_incoh] == 1 {
            let n_inc = n_stack[next_incoh];
            let d_inc = thick_slice[next_incoh];
            let rinc = nsinfi * inv_n_stack[next_incoh];
            let val_inc = Complex64::new(1.0, 0.0) - rinc * rinc;
            let mut cos_inc = val_inc.sqrt();
            if cos_inc.im < 0.0 {
                cos_inc = -cos_inc;
            }
            let beta_imag = (2.0 * PI * d_inc / lam) * (n_inc * cos_inc).im;
            let beta_imag = if beta_imag < 0.0 { 0.0 } else { beta_imag };
            let tfac = (-2.0 * beta_imag).exp();
            ig_s = accumulate_block(ig_s, 0.0, tfac, tfac, 0.0);
            ig_p = accumulate_block(ig_p, 0.0, tfac, tfac, 0.0);
        }
        current_idx = next_incoh;
    }
    (ig_s.0, ig_p.0, ig_s.2, ig_p.2)
}

#[pyfunction]
#[pyo3(name = "core_engine_photometry_only")]
#[allow(clippy::too_many_arguments)]
pub fn core_engine_photometry_only(
    py: Python<'_>,
    wavls: PyReadonlyArray1<f64>,
    sin_theta_arr: PyReadonlyArray1<f64>,
    n_layers: i32,
    n_stack_cache: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    incoherent_flags: PyReadonlyArray1<i32>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    calc_s: i32,
    calc_p: i32,
) -> PyResult<Py<PyTuple>> {
    let wav_slice = wavls.as_slice()?;
    let sin_theta_slice = sin_theta_arr.as_slice()?;
    let n_stack_slice = n_stack_cache.as_slice()?;
    let thick_slice = thicknesses.as_slice()?;
    let inc_flags_slice = incoherent_flags.as_slice()?;
    let rough_types_slice = rough_types.as_slice()?;
    let rough_vals_slice = rough_vals.as_slice()?;

    let num_wavs = wav_slice.len();
    let num_angles = sin_theta_slice.len();
    let total_points = num_wavs * num_angles;
    let idx_n = (n_layers - 1) as usize;
    let n_layers_us = n_layers as usize;
    let do_s = calc_s != 0;
    let do_p = calc_p != 0;

    let mut n_cache: Vec<Vec<Complex64>> = Vec::with_capacity(num_wavs);
    let mut inv_n_cache: Vec<Vec<Complex64>> = Vec::with_capacity(num_wavs);
    for ww in 0..num_wavs {
        let base = ww * n_layers_us * 2;
        let mut layer_n = Vec::with_capacity(n_layers_us);
        let mut layer_inv = Vec::with_capacity(n_layers_us);
        for l in 0..n_layers_us {
            let nv = Complex64::new(n_stack_slice[base + l * 2], n_stack_slice[base + l * 2 + 1]);
            layer_n.push(nv);
            layer_inv.push(nv.recip());
        }
        n_cache.push(layer_n);
        inv_n_cache.push(layer_inv);
    }

    // (rs, rp, ts, tp) per point.
    let points: Vec<(f64, f64, f64, f64)> = py.detach(|| {
        (0..total_points)
            .into_par_iter()
            .map(|k| {
                let a = k / num_wavs;
                let w = k % num_wavs;
                let lam = wav_slice[w];
                let sin_theta = sin_theta_slice[a];
                let n_stack = &n_cache[w];
                let inv_n_stack = &inv_n_cache[w];
                let nsinfi = n_stack[0] * Complex64::new(sin_theta, 0.0);

                if do_s && do_p {
                    return solve_both_pol(
                        idx_n, lam, nsinfi, n_stack, inv_n_stack, thick_slice, inc_flags_slice,
                        rough_types_slice, rough_vals_slice,
                    );
                }

                let (rs, ts) = if do_s {
                    solve_one_pol(
                        idx_n, lam, nsinfi, n_stack, inv_n_stack, thick_slice, inc_flags_slice,
                        rough_types_slice, rough_vals_slice, 0,
                    )
                } else {
                    (0.0, 0.0)
                };
                let (rp, tp) = if do_p {
                    solve_one_pol(
                        idx_n, lam, nsinfi, n_stack, inv_n_stack, thick_slice, inc_flags_slice,
                        rough_types_slice, rough_vals_slice, 1,
                    )
                } else {
                    (0.0, 0.0)
                };
                (rs, rp, ts, tp)
            })
            .collect()
    });

    let mut rs_out = vec![0.0; total_points];
    let mut rp_out = vec![0.0; total_points];
    let mut ts_out = vec![0.0; total_points];
    let mut tp_out = vec![0.0; total_points];
    for (k, p) in points.iter().enumerate() {
        rs_out[k] = p.0;
        rp_out[k] = p.1;
        ts_out[k] = p.2;
        tp_out[k] = p.3;
    }

    let shape = [num_angles, num_wavs];
    let rs_arr = PyArray::from_vec(py, rs_out).reshape(shape)?;
    let rp_arr = PyArray::from_vec(py, rp_out).reshape(shape)?;
    let ts_arr = PyArray::from_vec(py, ts_out).reshape(shape)?;
    let tp_arr = PyArray::from_vec(py, tp_out).reshape(shape)?;

    let result = PyTuple::new(py, vec![rs_arr, rp_arr, ts_arr, tp_arr])?;
    Ok(result.into())
}
