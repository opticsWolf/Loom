// func_4.rs
use num_complex::Complex64;
use num_complex::ComplexFloat;
use numpy::{PyArray, PyArrayMethods};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::func_2::redheffer_product_real_inner;
use crate::func_3::solve_coherent_block_fields_dual;

/// Per-point ellipsometry outputs, in output-array order.
struct EllipPoint {
    psi_r: f64,
    delta_r: f64,
    dop_r: f64,
    rs: f64,
    rp: f64,
    r_avg: f64,
    psi_t: f64,
    delta_t: f64,
    dop_t: f64,
    ts: f64,
    tp: f64,
    t_avg: f64,
    conservation: f64,
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_point_ellip(
    k: usize,
    num_wavs: usize,
    idx_n: usize,
    wav_slice: &[f64],
    sin_theta_slice: &[f64],
    n_cache: &[Vec<Complex64>],
    inv_n_cache: &[Vec<Complex64>],
    thick_slice: &[f64],
    inc_flags_slice: &[i32],
    rough_types_slice: &[i32],
    rough_vals_slice: &[f64],
    debug_flag: bool,
) -> EllipPoint {
    let a = k / num_wavs;
    let w = k % num_wavs;
    let lam = wav_slice[w];
    let sin_theta = sin_theta_slice[a];
    let current_n_stack = &n_cache[w];
    let current_inv_n = &inv_n_cache[w];
    let n0 = current_n_stack[0];
    let nsinfi = n0 * Complex64::new(sin_theta, 0.0);

    let mut ig_s = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut ig_p = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut rs0_c = Complex64::new(0.0, 0.0);
    let mut rp0_c = Complex64::new(0.0, 0.0);
    let mut first_block_processed = false;
    let mut cross_t_acc = Complex64::new(1.0, 0.0);
    let mut current_idx = 0usize;

    while current_idx < idx_n {
        let mut next_incoh = current_idx + 1;
        while next_incoh < idx_n && inc_flags_slice[next_incoh] == 0 {
            next_incoh += 1;
        }

        let (s_res, p_res) = solve_coherent_block_fields_dual(
            current_idx,
            next_incoh,
            current_n_stack,
            current_inv_n,
            thick_slice,
            rough_vals_slice,
            rough_types_slice,
            lam,
            nsinfi,
        );
        // BlockResult = (rf, tb, tf, rb, R_front, T_back, T_fwd, R_back)
        let (rs_f, _, ts_f, _, rs_rf, rs_tb, rs_tf, rs_rb) = s_res;
        let (rp_f, _, tp_f, _, rp_rf, rp_tb, rp_tf, rp_rb) = p_res;

        if !first_block_processed {
            rs0_c = rs_f;
            rp0_c = rp_f;
            first_block_processed = true;
        }

        // Transmission cross-term uses the forward-transmission FIELD amplitudes.
        cross_t_acc *= tp_f * ts_f.conj();

        ig_s = redheffer_product_real_inner(
            ig_s.0, ig_s.1, ig_s.2, ig_s.3, rs_rf, rs_tb, rs_tf, rs_rb,
        );
        ig_p = redheffer_product_real_inner(
            ig_p.0, ig_p.1, ig_p.2, ig_p.3, rp_rf, rp_tb, rp_tf, rp_rb,
        );

        if next_incoh < idx_n && inc_flags_slice[next_incoh] == 1 {
            let n_inc = current_n_stack[next_incoh];
            let d_inc = thick_slice[next_incoh];
            let rinc = nsinfi * current_inv_n[next_incoh];
            let val_inc = Complex64::new(1.0, 0.0) - rinc * rinc;
            let mut cos_inc = val_inc.sqrt();
            if cos_inc.im < 0.0 {
                cos_inc = -cos_inc;
            }
            let beta_imag = (2.0 * PI * d_inc / lam) * (n_inc * cos_inc).im;
            let beta_imag = if beta_imag < 0.0 { 0.0 } else { beta_imag };
            let trans_factor = (-2.0 * beta_imag).exp();

            ig_s = redheffer_product_real_inner(
                ig_s.0, ig_s.1, ig_s.2, ig_s.3, 0.0, trans_factor, trans_factor, 0.0,
            );
            ig_p = redheffer_product_real_inner(
                ig_p.0, ig_p.1, ig_p.2, ig_p.3, 0.0, trans_factor, trans_factor, 0.0,
            );
            cross_t_acc *= trans_factor;
        }

        current_idx = next_incoh;
    }

    let val_rs = ig_s.0;
    let val_ts = ig_s.2;
    let val_rp = ig_p.0;
    let val_tp = ig_p.2;

    let conservation = if debug_flag {
        let err_s = (1.0 - val_rs - val_ts).abs();
        let err_p = (1.0 - val_rp - val_tp).abs();
        err_s.max(err_p)
    } else {
        0.0
    };

    // ---- Reflection Stokes / ellipsometry ----
    let s0_r = val_rp + val_rs;
    let s1_r = val_rp - val_rs;
    let cross_r = rp0_c * rs0_c.conj();
    let s2_r = (-2.0 * cross_r.re) + 0.0; // flush -0.0 -> +0.0
    let s3_r = (-2.0 * cross_r.im) + 0.0;

    let dop_r = (s1_r * s1_r + s2_r * s2_r + s3_r * s3_r).sqrt() / (s0_r + 1e-20);

    let (psi_r, delta_r) = if val_rs < 1e-12 {
        (PI / 2.0, 0.0)
    } else {
        ((val_rp / val_rs).sqrt().atan(), s3_r.atan2(s2_r))
    };

    // ---- Transmission Stokes / ellipsometry ----
    let s0_t = val_tp + val_ts;
    let s1_t = val_tp - val_ts;
    let s2_t = (2.0 * cross_t_acc.re) + 0.0;
    let s3_t = (2.0 * cross_t_acc.im) + 0.0;

    let raw_dop_t = (s1_t * s1_t + s2_t * s2_t + s3_t * s3_t).sqrt() / (s0_t + 1e-20);
    let dop_t = raw_dop_t.min(1.0);

    let (psi_t, delta_t) = if val_ts < 1e-20 {
        (PI / 2.0, 0.0)
    } else {
        ((val_tp / val_ts).sqrt().atan(), s3_t.atan2(s2_t))
    };

    EllipPoint {
        psi_r,
        delta_r,
        dop_r,
        rs: val_rs,
        rp: val_rp,
        r_avg: 0.5 * (val_rs + val_rp),
        psi_t,
        delta_t,
        dop_t,
        ts: val_ts,
        tp: val_tp,
        t_avg: 0.5 * (val_ts + val_tp),
        conservation,
    }
}

#[pyfunction]
#[pyo3(name = "core_engine_rigorous_ellipsometry")]
#[allow(clippy::too_many_arguments)]
pub fn core_engine_rigorous_ellipsometry(
    py: Python<'_>,
    wavls: PyReadonlyArray1<f64>,
    sin_theta_arr: PyReadonlyArray1<f64>,
    n_layers: i32,
    n_stack_cache: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    incoherent_flags: PyReadonlyArray1<i32>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    debug_flag: i32,
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
    let debug = debug_flag != 0;

    // Build complex refractive-index cache (n_wavs x n_layers).
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

    // Heavy numeric work: release the GIL and run the point loop in parallel.
    let points: Vec<EllipPoint> = py.detach(|| {
        (0..total_points)
            .into_par_iter()
            .map(|k| {
                compute_point_ellip(
                    k,
                    num_wavs,
                    idx_n,
                    wav_slice,
                    sin_theta_slice,
                    &n_cache,
                    &inv_n_cache,
                    thick_slice,
                    inc_flags_slice,
                    rough_types_slice,
                    rough_vals_slice,
                    debug,
                )
            })
            .collect()
    });

    // Scatter struct-of-arrays into flat output buffers (cheap, O(N)).
    let mut psi_r = vec![0.0; total_points];
    let mut delta_r = vec![0.0; total_points];
    let mut dop_r = vec![0.0; total_points];
    let mut rs_out = vec![0.0; total_points];
    let mut rp_out = vec![0.0; total_points];
    let mut r_avg = vec![0.0; total_points];
    let mut psi_t = vec![0.0; total_points];
    let mut delta_t = vec![0.0; total_points];
    let mut dop_t = vec![0.0; total_points];
    let mut ts_out = vec![0.0; total_points];
    let mut tp_out = vec![0.0; total_points];
    let mut t_avg = vec![0.0; total_points];
    let mut conservation = vec![0.0; total_points];

    for (k, p) in points.iter().enumerate() {
        psi_r[k] = p.psi_r;
        delta_r[k] = p.delta_r;
        dop_r[k] = p.dop_r;
        rs_out[k] = p.rs;
        rp_out[k] = p.rp;
        r_avg[k] = p.r_avg;
        psi_t[k] = p.psi_t;
        delta_t[k] = p.delta_t;
        dop_t[k] = p.dop_t;
        ts_out[k] = p.ts;
        tp_out[k] = p.tp;
        t_avg[k] = p.t_avg;
        conservation[k] = p.conservation;
    }

    let shape = [num_angles, num_wavs];
    let arrs = vec![
        PyArray::from_vec(py, psi_r).reshape(shape)?,
        PyArray::from_vec(py, delta_r).reshape(shape)?,
        PyArray::from_vec(py, dop_r).reshape(shape)?,
        PyArray::from_vec(py, rs_out).reshape(shape)?,
        PyArray::from_vec(py, rp_out).reshape(shape)?,
        PyArray::from_vec(py, r_avg).reshape(shape)?,
        PyArray::from_vec(py, psi_t).reshape(shape)?,
        PyArray::from_vec(py, delta_t).reshape(shape)?,
        PyArray::from_vec(py, dop_t).reshape(shape)?,
        PyArray::from_vec(py, ts_out).reshape(shape)?,
        PyArray::from_vec(py, tp_out).reshape(shape)?,
        PyArray::from_vec(py, t_avg).reshape(shape)?,
        PyArray::from_vec(py, conservation).reshape(shape)?,
    ];

    let result = PyTuple::new(py, arrs)?;
    Ok(result.into())
}
