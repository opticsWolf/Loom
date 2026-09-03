// func_4.rs
use num_complex::Complex64;
use numpy::{PyArray, PyArrayMethods};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::func_6::solve_point;

// Coherence modes (0 front_block, 1 coherency_matrix, 2 fully_coherent) are
// handled entirely inside the shared kernel `solve_point`; func_4 only validates
// the value and forwards it. Ellipsometry always solves both polarizations and
// the p-s coherency channel, so it calls solve_point(.., true, true, true).

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
    // Complex first-block (Modes A/B) or global (Mode C) field amplitudes,
    // and the p-s cross-coherence terms. r_p is in the admittance convention
    // (Born-Wolf phase = arg(-rp)); cross_r already carries the sign fix.
    rs_c: Complex64,
    rp_c: Complex64,
    ts_c: Complex64,
    tp_c: Complex64,
    cross_r_c: Complex64,
    cross_t_c: Complex64,
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
    coherence_mode: i32,
) -> EllipPoint {
    let a = k / num_wavs;
    let w = k % num_wavs;

    // Solve stage: shared kernel. Ellipsometry needs both pols + the coherency
    // channel, so need_s = need_p = need_cross = true.
    let st = solve_point(
        idx_n,
        wav_slice[w],
        sin_theta_slice[a],
        &n_cache[w],
        &inv_n_cache[w],
        thick_slice,
        inc_flags_slice,
        rough_types_slice,
        rough_vals_slice,
        coherence_mode,
        true,
        true,
        true,
    );

    let val_rs = st.rs;
    let val_ts = st.ts;
    let val_rp = st.rp;
    let val_tp = st.tp;
    let cross_r = st.cross_r;
    let cross_t = st.cross_t;

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
    let s2_t = (2.0 * cross_t.re) + 0.0;
    let s3_t = (2.0 * cross_t.im) + 0.0;

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
        rs_c: st.rs_c,
        rp_c: st.rp_c,
        ts_c: st.ts_c,
        tp_c: st.tp_c,
        cross_r_c: cross_r,
        cross_t_c: cross_t,
    }
}

#[pyfunction]
#[pyo3(name = "core_engine_rigorous_ellipsometry")]
#[pyo3(signature = (
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, debug_flag, coherence_mode=0
))]
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
    coherence_mode: i32,
) -> PyResult<Py<PyTuple>> {
    let wav_slice = wavls.as_slice()?;
    let sin_theta_slice = sin_theta_arr.as_slice()?;
    let n_stack_slice = n_stack_cache.as_slice()?;
    let thick_slice = thicknesses.as_slice()?;
    let inc_flags_slice = incoherent_flags.as_slice()?;
    let rough_types_slice = rough_types.as_slice()?;
    let rough_vals_slice = rough_vals.as_slice()?;

    if !(0..=2).contains(&coherence_mode) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coherence_mode must be 0 (front_block), 1 (coherency_matrix), or 2 (fully_coherent).",
        ));
    }

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
                    coherence_mode,
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
    // New complex outputs (appended after the 13 real arrays).
    let mut rs_c = vec![Complex64::new(0.0, 0.0); total_points];
    let mut rp_c = vec![Complex64::new(0.0, 0.0); total_points];
    let mut ts_c = vec![Complex64::new(0.0, 0.0); total_points];
    let mut tp_c = vec![Complex64::new(0.0, 0.0); total_points];
    let mut cross_r_c = vec![Complex64::new(0.0, 0.0); total_points];
    let mut cross_t_c = vec![Complex64::new(0.0, 0.0); total_points];

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
        rs_c[k] = p.rs_c;
        rp_c[k] = p.rp_c;
        ts_c[k] = p.ts_c;
        tp_c[k] = p.tp_c;
        cross_r_c[k] = p.cross_r_c;
        cross_t_c[k] = p.cross_t_c;
    }

    let shape = [num_angles, num_wavs];
    // Output order: 13 real arrays (unchanged), then 6 complex arrays appended:
    //   [13] rs_c  [14] rp_c  [15] ts_c  [16] tp_c  [17] cross_R  [18] cross_T
    let arrs: Vec<Bound<'_, pyo3::PyAny>> = vec![
        PyArray::from_vec(py, psi_r).reshape(shape)?.into_any(),
        PyArray::from_vec(py, delta_r).reshape(shape)?.into_any(),
        PyArray::from_vec(py, dop_r).reshape(shape)?.into_any(),
        PyArray::from_vec(py, rs_out).reshape(shape)?.into_any(),
        PyArray::from_vec(py, rp_out).reshape(shape)?.into_any(),
        PyArray::from_vec(py, r_avg).reshape(shape)?.into_any(),
        PyArray::from_vec(py, psi_t).reshape(shape)?.into_any(),
        PyArray::from_vec(py, delta_t).reshape(shape)?.into_any(),
        PyArray::from_vec(py, dop_t).reshape(shape)?.into_any(),
        PyArray::from_vec(py, ts_out).reshape(shape)?.into_any(),
        PyArray::from_vec(py, tp_out).reshape(shape)?.into_any(),
        PyArray::from_vec(py, t_avg).reshape(shape)?.into_any(),
        PyArray::from_vec(py, conservation).reshape(shape)?.into_any(),
        PyArray::from_vec(py, rs_c).reshape(shape)?.into_any(),
        PyArray::from_vec(py, rp_c).reshape(shape)?.into_any(),
        PyArray::from_vec(py, ts_c).reshape(shape)?.into_any(),
        PyArray::from_vec(py, tp_c).reshape(shape)?.into_any(),
        PyArray::from_vec(py, cross_r_c).reshape(shape)?.into_any(),
        PyArray::from_vec(py, cross_t_c).reshape(shape)?.into_any(),
    ];

    let result = PyTuple::new(py, arrs)?;
    Ok(result.into())
}
