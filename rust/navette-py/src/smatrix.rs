//! Thin PyO3 bindings for the Navette S-matrix engine.
//!
//! No physics here: every kernel lives in the pure-Rust
//! `navette-smatrix` core. Wrappers own NumPy inputs, release
//! the GIL while rayon-parallel kernels run, and return NumPy.

use num_complex::{Complex64, ComplexFloat};
use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::f64::consts::PI;

use navette::smatrix::coherent_block::*;
use navette::smatrix::needle_engine::{max_disp_order, NREQ_DFOD, NREQ_DGDD, NREQ_DGD, NREQ_DPHI, NREQ_DTOD, NREQ_P, NREQ_P_A, NREQ_P_AB, NREQ_P_MB, NREQ_P_MB_A, NREQ_P_MB_AB, NREQ_P_MB_RB, NREQ_P_MB_T, NREQ_P_MB_TB, NREQ_P_PHI, NREQ_P_RB, NREQ_P_T, NREQ_P_TB};
use navette::smatrix::needle_operator::*;
use navette::smatrix::optics_core::*;
use navette::smatrix::optimizer::*;

// ---- roughness / redheffer (trivial, over optics_core) ----
#[pyfunction]
pub fn w_function(q: Complex64, rough_type: i32) -> PyResult<Complex64> {
    Ok(w_function_inner(q, rough_type))
}

#[pyfunction]
pub fn redheffer_product_complex_field(
    r_a_front: Complex64, t_a_back: Complex64, t_a_fwd: Complex64, r_a_back: Complex64,
    r_b_front: Complex64, t_b_back: Complex64, t_b_fwd: Complex64, r_b_back: Complex64,
) -> PyResult<(Complex64, Complex64, Complex64, Complex64)> {
    Ok(redheffer_product_complex_field_inner(
        r_a_front, t_a_back, t_a_fwd, r_a_back, r_b_front, t_b_back, t_b_fwd, r_b_back,
    ))
}

#[pyfunction]
pub fn redheffer_product_real(
    ra_rf: f64, ra_tb: f64, ra_tf: f64, ra_rb: f64,
    rb_rf: f64, rb_tb: f64, rb_tf: f64, rb_rb: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(redheffer_product_real_inner(ra_rf, ra_tb, ra_tf, ra_rb, rb_rf, rb_tb, rb_tf, rb_rb))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn redheffer_product_cross(
    a_cf: Complex64, a_db: Complex64, a_df: Complex64, a_cb: Complex64,
    b_cf: Complex64, b_db: Complex64, b_df: Complex64, b_cb: Complex64,
) -> PyResult<(Complex64, Complex64, Complex64, Complex64)> {
    Ok(redheffer_product_cross_inner(a_cf, a_db, a_df, a_cb, b_cf, b_db, b_df, b_cb))
}

// ---- coherent_block wrapper (verbatim from core) ----
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

// ---- shared solution emit (used by core_engine + PySolver + structure) ----
pub(crate) fn solution_to_dict(
    py: Python<'_>,
    sol: &navette::smatrix::solver::Solution,
) -> PyResult<Py<PyDict>> {
    let shape = [sol.n_angles, sol.n_wavs];
    let out = PyDict::new(py);
    for (k, b) in &sol.f64maps {
        out.set_item(k, PyArray::from_vec(py, b.clone()).reshape(shape)?)?;
    }
    for (k, b) in &sol.c64maps {
        out.set_item(k, PyArray::from_vec(py, b.clone()).reshape(shape)?)?;
    }
    for (chan, quads) in &sol.dispmaps {
        let (g, gg, t, f) = match chan.as_str() {
            "R_s" => ("GD_R_s", "GDD_R_s", "TOD_R_s", "FOD_R_s"),
            "R_p" => ("GD_R_p", "GDD_R_p", "TOD_R_p", "FOD_R_p"),
            "T_s" => ("GD_T_s", "GDD_T_s", "TOD_T_s", "FOD_T_s"),
            _ => ("GD_T_p", "GDD_T_p", "TOD_T_p", "FOD_T_p"),
        };
        out.set_item(g, PyArray::from_vec(py, quads[0].clone()).reshape(shape)?)?;
        out.set_item(gg, PyArray::from_vec(py, quads[1].clone()).reshape(shape)?)?;
        out.set_item(t, PyArray::from_vec(py, quads[2].clone()).reshape(shape)?)?;
        out.set_item(f, PyArray::from_vec(py, quads[3].clone()).reshape(shape)?)?;
    }
    Ok(out.into())
}

// ---- native Solver (Rust-first ScatterMatrix engine) ----
#[pyclass(name = "Solver")]
pub struct PySolver {
    inner: navette::smatrix::solver::Solver,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (wavelengths, angles, indices, n_layers, thicknesses=None, incoherent_flags=None, roughness_types=None, roughness_values=None, coherence_mode=0, angles_in_radians=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        wavelengths: PyReadonlyArray1<f64>,
        angles: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<Complex64>,
        n_layers: usize,
        thicknesses: Option<PyReadonlyArray1<f64>>,
        incoherent_flags: Option<PyReadonlyArray1<i32>>,
        roughness_types: Option<PyReadonlyArray1<i32>>,
        roughness_values: Option<PyReadonlyArray1<f64>>,
        coherence_mode: i32,
        angles_in_radians: bool,
    ) -> PyResult<Self> {
        let opt_f = |o: &Option<PyReadonlyArray1<f64>>| o.as_ref().map(|a| a.as_slice().unwrap().to_vec());
        let opt_i = |o: &Option<PyReadonlyArray1<i32>>| o.as_ref().map(|a| a.as_slice().unwrap().to_vec());
        let tf = opt_f(&thicknesses);
        let cf = opt_i(&incoherent_flags);
        let rt = opt_i(&roughness_types);
        let rv = opt_f(&roughness_values);
        navette::smatrix::solver::Solver::from_raw(
            wavelengths.as_slice()?,
            angles.as_slice()?,
            angles_in_radians,
            indices.as_slice()?,
            n_layers,
            tf.as_deref(),
            cf.as_deref(),
            rt.as_deref(),
            rv.as_deref(),
            coherence_mode,
        )
        .map(|inner| PySolver { inner })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn solve(&self, py: Python<'_>, requested: u64) -> PyResult<Py<PyDict>> {
        let sol = py
            .detach(|| self.inner.solve(requested))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        solution_to_dict(py, &sol)
    }

    #[getter]
    fn n_angles(&self) -> usize {
        self.inner.n_angles()
    }

    #[getter]
    fn n_wavs(&self) -> usize {
        self.inner.n_wavs()
    }

    /// Scan `|1/r(n_eff)|^2` over an effective-index box.
    /// Returns `(real_vals, imag_vals, flat imag-major values)`.
    #[pyo3(signature = (real_range, imag_range, points_real, points_imag, pol, wavelength=None, wav_index=None))]
    fn landscape(
        &self,
        real_range: (f64, f64),
        imag_range: (f64, f64),
        points_real: usize,
        points_imag: usize,
        pol: i32,
        wavelength: Option<f64>,
        wav_index: Option<usize>,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        self.inner
            .landscape(real_range, imag_range, points_real, points_imag, pol, wavelength, wav_index)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Nelder-Mead refine of one eigenmode guess → `(n_eff, value)`.
    #[pyo3(signature = (guess, pol, wavelength=None, wav_index=None, step=1e-3, tol=1e-9, max_iter=200))]
    fn refine_mode(
        &self,
        guess: Complex64,
        pol: i32,
        wavelength: Option<f64>,
        wav_index: Option<usize>,
        step: f64,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<(Complex64, f64)> {
        self.inner
            .refine_mode(guess, pol, wavelength, wav_index, step, tol, max_iter)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Scan, locate minima, optionally refine each → eigenmode list.
    #[pyo3(signature = (real_range, imag_range, points=(200, 200), median_factor=0.1, refine=true, pol=0, wavelength=None, wav_index=None))]
    fn find_eigenmodes(
        &self,
        real_range: (f64, f64),
        imag_range: (f64, f64),
        points: (usize, usize),
        median_factor: f64,
        refine: bool,
        pol: i32,
        wavelength: Option<f64>,
        wav_index: Option<usize>,
    ) -> PyResult<Vec<Complex64>> {
        self.inner
            .find_eigenmodes(
                real_range, imag_range, points.0, points.1, median_factor, refine, pol,
                wavelength, wav_index,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// `|E(z)|` profile dict parts for one eigenmode.
    #[pyo3(signature = (n_eff, pol, wavelength=None, wav_index=None, points_per_layer=50))]
    fn field_profile(
        &self,
        n_eff: Complex64,
        pol: i32,
        wavelength: Option<f64>,
        wav_index: Option<usize>,
        points_per_layer: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Complex64>)> {
        self.inner
            .field_profile(n_eff, pol, wavelength, wav_index, points_per_layer)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    #[pyo3(signature = (needle_n_per_wav, z_grid, requested, incoherent_flags=None, targets_r=None, weights_r=None, targets_t=None, weights_t=None, targets_a=None, weights_a=None, targets_phi=None, weights_phi=None, targets_tb=None, weights_tb=None, targets_rb=None, weights_rb=None, targets_ab=None, weights_ab=None, start_idx=0, end_idx=None, channel=0, calc_s=true, calc_p=true, host_mask=None, gain_shift_phi=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn needle_gradient(
        &self,
        py: Python<'_>,
        needle_n_per_wav: PyReadonlyArray1<Complex64>,
        z_grid: PyReadonlyArray1<f64>,
        requested: u64,
        incoherent_flags: Option<PyReadonlyArray1<i32>>,
        targets_r: Option<PyReadonlyArray1<f64>>,
        weights_r: Option<PyReadonlyArray1<f64>>,
        targets_t: Option<PyReadonlyArray1<f64>>,
        weights_t: Option<PyReadonlyArray1<f64>>,
        targets_a: Option<PyReadonlyArray1<f64>>,
        weights_a: Option<PyReadonlyArray1<f64>>,
        targets_phi: Option<PyReadonlyArray1<f64>>,
        weights_phi: Option<PyReadonlyArray1<f64>>,
        targets_tb: Option<PyReadonlyArray1<f64>>,
        weights_tb: Option<PyReadonlyArray1<f64>>,
        targets_rb: Option<PyReadonlyArray1<f64>>,
        weights_rb: Option<PyReadonlyArray1<f64>>,
        targets_ab: Option<PyReadonlyArray1<f64>>,
        weights_ab: Option<PyReadonlyArray1<f64>>,
        start_idx: usize,
        end_idx: Option<usize>,
        channel: usize,
        calc_s: bool,
        calc_p: bool,
        host_mask: Option<PyReadonlyArray1<bool>>,
        gain_shift_phi: f64,
    ) -> PyResult<Py<PyDict>> {
        let opt = |o: &Option<PyReadonlyArray1<f64>>| -> PyResult<Option<Vec<f64>>> {
            o.as_ref()
                .map(|a| a.as_slice().map(|v| v.to_vec()).map_err(|e| e.into()))
                .transpose()
        };
        let t = |o: &Option<PyReadonlyArray1<f64>>| opt(o).unwrap();
        let (npn, zg) = (
            needle_n_per_wav.as_slice()?.to_vec(),
            z_grid.as_slice()?.to_vec(),
        );
        let inc = incoherent_flags
            .as_ref()
            .map(|a| a.as_slice().map(|v| v.to_vec()))
            .transpose()
            .map_err(|e| -> pyo3::PyErr { e.into() })?;
        let hm = host_mask
            .as_ref()
            .map(|a| a.as_slice().map(|v| v.to_vec()))
            .transpose()
            .map_err(|e| -> pyo3::PyErr { e.into() })?;
        let (tr, wr, tt, wt, ta, wa, tp, wp, ttb, wtb, trb, wrb, tab, wab) = (
            t(&targets_r), t(&weights_r), t(&targets_t), t(&weights_t), t(&targets_a),
            t(&weights_a), t(&targets_phi), t(&weights_phi), t(&targets_tb),
            t(&weights_tb), t(&targets_rb), t(&weights_rb), t(&targets_ab), t(&weights_ab),
        );
        let sol = py
            .detach(|| {
                self.inner.needle_gradient(
                    &npn, &zg, requested, inc.as_deref(),
                    tr.as_deref(), wr.as_deref(), tt.as_deref(), wt.as_deref(),
                    ta.as_deref(), wa.as_deref(), tp.as_deref(), wp.as_deref(),
                    ttb.as_deref(), wtb.as_deref(), trb.as_deref(), wrb.as_deref(),
                    tab.as_deref(), wab.as_deref(),
                    start_idx, end_idx, channel, calc_s, calc_p, hm.as_deref(),
                    gain_shift_phi,
                )
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let shape = [sol.n_points, sol.n_depths];
        let out = PyDict::new(py);
        for (k, b) in &sol.maps {
            out.set_item(k, PyArray::from_vec(py, b.clone()).reshape(shape)?)?;
        }
        Ok(out.into())
    }}


// ---- view request masks + energy kernel (thin over solver) ----
#[pyfunction]
fn solver_rt_request(pol: &str) -> PyResult<u64> {
    navette::smatrix::solver::rt_request(pol).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn solver_ellipsometry_request(transmission: bool) -> u64 {
    navette::smatrix::solver::ellipsometry_request(transmission)
}

#[pyfunction]
fn solver_absorption_request() -> u64 {
    navette::smatrix::solver::absorption_request()
}

#[pyfunction]
fn solver_amplitudes_request() -> u64 {
    navette::smatrix::solver::amplitudes_request()
}

#[pyfunction]
fn solver_stokes_request(reflection: bool, transmission: bool) -> PyResult<u64> {
    navette::smatrix::solver::stokes_request(reflection, transmission)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn solver_dispersion_request(
    reflection: bool,
    transmission: bool,
    s_pol: bool,
    p_pol: bool,
) -> PyResult<u64> {
    navette::smatrix::solver::dispersion_request(reflection, transmission, s_pol, p_pol)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn solver_energy_conservation(
    py: Python<'_>,
    rs: PyReadonlyArray1<f64>,
    rp: PyReadonlyArray1<f64>,
    ts: PyReadonlyArray1<f64>,
    tp: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let e = navette::smatrix::solver::energy_conservation(
        rs.as_slice()?,
        rp.as_slice()?,
        ts.as_slice()?,
        tp.as_slice()?,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyArray::from_vec(py, e).into())
}

// ---- core_engine wrapper (verbatim from core) ----
#[pyfunction]
#[pyo3(name = "core_engine")]
#[pyo3(signature = (
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, coherence_mode, requested
))]
#[allow(clippy::too_many_arguments)]
pub fn core_engine(
    py: Python<'_>,
    wavls: PyReadonlyArray1<f64>,
    sin_theta_arr: PyReadonlyArray1<f64>,
    n_layers: i32,
    n_stack_cache: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    incoherent_flags: PyReadonlyArray1<i32>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    coherence_mode: i32,
    requested: u64,
) -> PyResult<Py<PyDict>> {
    // Thin over the native Solver: transpose the wav-major flat cache to
    // layer-major, solve, emit arrays. No physics here.
    use navette::smatrix::solver::Solver;
    let wav_slice = wavls.as_slice()?;
    let sin_theta_slice = sin_theta_arr.as_slice()?;
    let n_stack_slice = n_stack_cache.as_slice()?;
    let num_wavs = wav_slice.len();
    let num_angles = sin_theta_slice.len();
    let n_layers_us = n_layers as usize;
    let mut layer_major = Vec::with_capacity(n_layers_us * num_wavs);
    for l in 0..n_layers_us {
        for w in 0..num_wavs {
            let base = w * n_layers_us * 2;
            layer_major.push(Complex64::new(
                n_stack_slice[base + l * 2],
                n_stack_slice[base + l * 2 + 1],
            ));
        }
    }
    let solver = Solver::new(
        wav_slice,
        sin_theta_slice,
        &layer_major,
        n_layers_us,
        thicknesses.as_slice()?,
        incoherent_flags.as_slice()?,
        rough_types.as_slice()?,
        rough_vals.as_slice()?,
        coherence_mode,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let sol = py
        .detach(|| solver.solve(requested))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    solution_to_dict(py, &sol)
}

// ---- needle_engine wrapper (verbatim from core) ----
#[pyfunction]
#[pyo3(name = "needle_engine")]
#[pyo3(signature = (
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    rough_types, rough_vals, needle_n_per_wav, z_grid,
    requested,
    incoherent_flags=None, targets_r=None, weights_r=None,
    targets_t=None, weights_t=None, targets_a=None, weights_a=None,
    targets_phi=None, weights_phi=None,
    targets_tb=None, weights_tb=None, targets_rb=None, weights_rb=None,
    targets_ab=None, weights_ab=None,
    start_idx=0, end_idx=None, channel=0,
    calc_s=true, calc_p=true, host_mask=None
))]
#[allow(clippy::too_many_arguments)]
pub fn needle_engine<'py>(
    py: Python<'py>,
    wavls: PyReadonlyArray1<f64>,
    sin_theta_arr: PyReadonlyArray1<f64>,
    n_layers: i32,
    n_stack_cache: PyReadonlyArray1<f64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    needle_n_per_wav: PyReadonlyArray1<Complex64>,
    z_grid: PyReadonlyArray1<f64>,
    requested: u64,
    incoherent_flags: Option<PyReadonlyArray1<i32>>,
    targets_r: Option<PyReadonlyArray1<f64>>,
    weights_r: Option<PyReadonlyArray1<f64>>,
    targets_t: Option<PyReadonlyArray1<f64>>,
    weights_t: Option<PyReadonlyArray1<f64>>,
    targets_a: Option<PyReadonlyArray1<f64>>,
    weights_a: Option<PyReadonlyArray1<f64>>,
    targets_phi: Option<PyReadonlyArray1<f64>>,
    weights_phi: Option<PyReadonlyArray1<f64>>,
    targets_tb: Option<PyReadonlyArray1<f64>>,
    weights_tb: Option<PyReadonlyArray1<f64>>,
    targets_rb: Option<PyReadonlyArray1<f64>>,
    weights_rb: Option<PyReadonlyArray1<f64>>,
    targets_ab: Option<PyReadonlyArray1<f64>>,
    weights_ab: Option<PyReadonlyArray1<f64>>,
    start_idx: usize,
    end_idx: Option<usize>,
    channel: usize,
    calc_s: bool,
    calc_p: bool,
    host_mask: Option<PyReadonlyArray1<bool>>,
) -> PyResult<Py<PyDict>> {
    // Thin over the native needle_gradient: extract slices, solve
    // detached, reshape flat buffers. No physics here.
    use navette::smatrix::solver::needle_gradient as core_needle;
    let opt = |o: &Option<PyReadonlyArray1<f64>>| -> PyResult<Option<Vec<f64>>> {
        o.as_ref()
            .map(|a| a.as_slice().map(|v| v.to_vec()).map_err(|e| e.into()))
            .transpose()
    };
    let (wv, st, cache, th, rt, rv, npn, zg) = (
        wavls.as_slice()?.to_vec(),
        sin_theta_arr.as_slice()?.to_vec(),
        n_stack_cache.as_slice()?.to_vec(),
        thicknesses.as_slice()?.to_vec(),
        rough_types.as_slice()?.to_vec(),
        rough_vals.as_slice()?.to_vec(),
        needle_n_per_wav.as_slice()?.to_vec(),
        z_grid.as_slice()?.to_vec(),
    );
    let inc = incoherent_flags
        .as_ref()
        .map(|a| a.as_slice().map(|v| v.to_vec()))
        .transpose()
        .map_err(|e| -> pyo3::PyErr { e.into() })?;
    let hm = host_mask
        .as_ref()
        .map(|a| a.as_slice().map(|v| v.to_vec()))
        .transpose()
        .map_err(|e| -> pyo3::PyErr { e.into() })?;
    let t = |o: &Option<PyReadonlyArray1<f64>>| opt(o).unwrap();
    let (tr, wr, tt, wt, ta, wa, tp, wp, ttb, wtb, trb, wrb, tab, wab) = (
        t(&targets_r), t(&weights_r), t(&targets_t), t(&weights_t), t(&targets_a),
        t(&weights_a), t(&targets_phi), t(&weights_phi), t(&targets_tb),
        t(&weights_tb), t(&targets_rb), t(&weights_rb), t(&targets_ab), t(&weights_ab),
    );
    let sol = py
        .detach(|| {
            core_needle(
                &wv, &st, n_layers as usize, &cache, &th, &rt, &rv, &npn, &zg,
                requested, inc.as_deref(),
                tr.as_deref(), wr.as_deref(), tt.as_deref(), wt.as_deref(),
                ta.as_deref(), wa.as_deref(), tp.as_deref(), wp.as_deref(),
                ttb.as_deref(), wtb.as_deref(), trb.as_deref(), wrb.as_deref(),
                tab.as_deref(), wab.as_deref(),
                start_idx, end_idx, channel, calc_s, calc_p, hm.as_deref(), 0.0,
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let shape = [sol.n_points, sol.n_depths];
    let out = PyDict::new(py);
    for (k, b) in &sol.maps {
        out.set_item(k, PyArray::from_vec(py, b.clone()).reshape(shape)?)?;
    }
    Ok(out.into())
}

// ---- optimizer wrappers (verbatim from core; see note above) ----
// -----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(name = "scan_landscape")]
pub fn scan_landscape(
    py: Python<'_>,
    n_stack: PyReadonlyArray1<Complex64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    lam: f64,
    pol: i32,
    real_min: f64,
    real_max: f64,
    imag_min: f64,
    imag_max: f64,
    points_real: usize,
    points_imag: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Py<PyArray2<f64>>)> {
    // Thin over core scan_box.
    let (ns, th, rt, rv) = (
        n_stack.as_slice()?.to_vec(),
        thicknesses.as_slice()?.to_vec(),
        rough_types.as_slice()?.to_vec(),
        rough_vals.as_slice()?.to_vec(),
    );
    let (real_vals, imag_vals, flat) = py.detach(|| {
        navette::smatrix::solver::scan_box(
            &ns, &th, &rt, &rv, lam, pol, real_min, real_max, imag_min, imag_max,
            points_real, points_imag,
        )
    });
    let land_arr =
        PyArray1::from_vec(py, flat).reshape([points_imag, points_real]).unwrap();
    Ok((real_vals, imag_vals, land_arr.into()))
}

// -----------------------------------------------------------------------------
// Find local minima on the coarse grid
// -----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(name = "find_local_minima")]
pub fn find_local_minima(
    landscape: PyReadonlyArray2<f64>,
    real_vals: Vec<f64>,
    imag_vals: Vec<f64>,
    median_factor: f64,
) -> Vec<(f64, f64)> {
    // Thin over core find_minima.
    let land = landscape.as_array();
    let (n_imag, n_real) = (land.shape()[0], land.shape()[1]);
    let flat: Vec<f64> = land.iter().copied().collect();
    navette::smatrix::solver::find_minima(
        &flat, n_real, n_imag, &real_vals, &imag_vals, median_factor,
    )
}

// -----------------------------------------------------------------------------
// Nelder‑Mead minimiser (2D, adaptive)
// -----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(name = "nelder_mead")]
pub fn nelder_mead(
    n_stack: PyReadonlyArray1<Complex64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    lam: f64,
    pol: i32,
    x0: (f64, f64),
    step: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64, f64) {
    // Thin over core nelder_refine.
    navette::smatrix::solver::nelder_refine(
        n_stack.as_slice().unwrap(),
        thicknesses.as_slice().unwrap(),
        rough_types.as_slice().unwrap(),
        rough_vals.as_slice().unwrap(),
        lam,
        pol,
        x0,
        step,
        tol,
        max_iter,
    )
}

// -----------------------------------------------------------------------------
// Field profile: |E(z)| through the stack for a given eigenmode (thin)
// -----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(name = "field_profile")]
pub fn field_profile(
    n_stack: PyReadonlyArray1<Complex64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    lam: f64,
    n_eff: Complex64,
    pol: i32,
    points_per_layer: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Complex64>)> {
    // Thin over core field_prof.
    navette::smatrix::solver::field_prof(
        n_stack.as_slice()?,
        thicknesses.as_slice()?,
        rough_types.as_slice()?,
        rough_vals.as_slice()?,
        lam,
        n_eff,
        pol,
        points_per_layer,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)
}
/// Register the S-matrix engine submodule (`navette._navette._smatrix`).
#[pymodule]
pub fn _smatrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(w_function, m)?)?;
    m.add_function(wrap_pyfunction!(redheffer_product_complex_field, m)?)?;
    m.add_function(wrap_pyfunction!(redheffer_product_real, m)?)?;
    m.add_function(wrap_pyfunction!(redheffer_product_cross, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coherent_block_fields, m)?)?;
    m.add_function(wrap_pyfunction!(core_engine, m)?)?;
    m.add_function(wrap_pyfunction!(scan_landscape, m)?)?;
    m.add_function(wrap_pyfunction!(find_local_minima, m)?)?;
    m.add_function(wrap_pyfunction!(nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(field_profile, m)?)?;
    m.add_function(wrap_pyfunction!(needle_engine, m)?)?;
    m.add_class::<crate::synthesis_merit::PySimCurves>()?;
    m.add_class::<crate::synthesis_merit::PyMeritSpec>()?;
    m.add_function(wrap_pyfunction!(crate::synthesis_merit::build_needle_targets, m)?)?;
    m.add_function(wrap_pyfunction!(crate::synthesis_merit::reference_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(crate::synthesis_pipeline::run_design, m)?)?;
    m.add_class::<crate::synthesis_pipeline::PyLayerSpec>()?;
    m.add_class::<crate::synthesis_pipeline::PyDesignStack>()?;
    m.add_class::<crate::synthesis_pipeline::PySmatrixContext>()?;
    m.add_class::<crate::synthesis_pipeline::PyLmConfig>()?;
    m.add_class::<crate::synthesis_pipeline::PyPipelineConfig>()?;
    m.add_class::<crate::synthesis_pipeline::PyNeedleCycleConfig>()?;
    m.add_class::<crate::synthesis_pipeline::PyNeedlePipeline>()?;
    m.add_class::<PySolver>()?;
    m.add_function(wrap_pyfunction!(solver_rt_request, m)?)?;
    m.add_function(wrap_pyfunction!(solver_ellipsometry_request, m)?)?;
    m.add_function(wrap_pyfunction!(solver_absorption_request, m)?)?;
    m.add_function(wrap_pyfunction!(solver_amplitudes_request, m)?)?;
    m.add_function(wrap_pyfunction!(solver_stokes_request, m)?)?;
    m.add_function(wrap_pyfunction!(solver_dispersion_request, m)?)?;
    m.add_function(wrap_pyfunction!(solver_energy_conservation, m)?)?;
    m.add("NREQ_P", NREQ_P)?;
    m.add("NREQ_P_MB", NREQ_P_MB)?;
    m.add("NREQ_P_T", NREQ_P_T)?;
    m.add("NREQ_P_A", NREQ_P_A)?;
    m.add("NREQ_P_PHI", NREQ_P_PHI)?;
    m.add("NREQ_P_MB_T", NREQ_P_MB_T)?;
    m.add("NREQ_P_MB_A", NREQ_P_MB_A)?;
    m.add("NREQ_P_TB", NREQ_P_TB)?;
    m.add("NREQ_P_RB", NREQ_P_RB)?;
    m.add("NREQ_P_AB", NREQ_P_AB)?;
    m.add("NREQ_P_MB_TB", NREQ_P_MB_TB)?;
    m.add("NREQ_P_MB_RB", NREQ_P_MB_RB)?;
    m.add("NREQ_P_MB_AB", NREQ_P_MB_AB)?;
    m.add("NREQ_DPHI", NREQ_DPHI)?;
    m.add("NREQ_DGD", NREQ_DGD)?;
    m.add("NREQ_DGDD", NREQ_DGDD)?;
    m.add("NREQ_DTOD", NREQ_DTOD)?;
    m.add("NREQ_DFOD", NREQ_DFOD)?;
    Ok(())
}
