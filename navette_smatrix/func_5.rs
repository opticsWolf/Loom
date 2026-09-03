// func_5.rs
//
// Photometry-only engine. Now a thin wrapper over the shared per-point kernel
// `func_6::solve_point`: it resolves the polarization branches from calc_s /
// calc_p, never runs the coherency channel (need_cross = false), and packages
// the four intensities into the legacy (Rs, Rp, Ts, Tp) tuple. The block-loop /
// incoherent-gap logic lives in one place (solve_point) shared with func_4 and
// func_6. A `coherence_mode` argument (default 0 = front_block) is accepted so
// that fully-coherent photometry (mode 2) is available without a second path.

use num_complex::Complex64;
use num_complex::ComplexFloat;
use numpy::{PyArray, PyArrayMethods};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

use crate::func_6::solve_point;

#[pyfunction]
#[pyo3(name = "core_engine_photometry_only")]
#[pyo3(signature = (
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, calc_s, calc_p, coherence_mode=0
))]
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
    coherence_mode: i32,
) -> PyResult<Py<PyTuple>> {
    if !(0..=2).contains(&coherence_mode) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coherence_mode must be 0 (front_block), 1 (coherency_matrix), or 2 (fully_coherent).",
        ));
    }

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

    let need_s = calc_s != 0;
    let need_p = calc_p != 0;

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

    // (rs, rp, ts, tp) per point; 0.0 for any polarization not requested.
    let points: Vec<(f64, f64, f64, f64)> = py.detach(|| {
        (0..total_points)
            .into_par_iter()
            .map(|k| {
                let a = k / num_wavs;
                let w = k % num_wavs;
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
                    need_s,
                    need_p,
                    false, // photometry: no p-s coherency channel
                );
                (
                    if need_s { st.rs } else { 0.0 },
                    if need_p { st.rp } else { 0.0 },
                    if need_s { st.ts } else { 0.0 },
                    if need_p { st.tp } else { 0.0 },
                )
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
