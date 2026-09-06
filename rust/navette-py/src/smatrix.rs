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

// ---- shared solution emit (used by core_engine + PySolver) ----
fn solution_to_dict(
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
}

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
    if requested == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("empty request mask"));
    }
    let wav_slice = wavls.as_slice()?;
    let sin_slice = sin_theta_arr.as_slice()?;
    let thick_slice = thicknesses.as_slice()?;
    let rt_slice = rough_types.as_slice()?;
    let rv_slice = rough_vals.as_slice()?;
    let np_slice = needle_n_per_wav.as_slice()?;
    let z_slice = z_grid.as_slice()?;
    let cache_slice = n_stack_cache.as_slice()?;

    let num_wavs = wav_slice.len();
    let num_angles = sin_slice.len();
    let total_points = num_wavs * num_angles;
    let nl = n_layers as usize;
    let nz = z_slice.len();

    if !(0..nl).contains(&start_idx) {
        return Err(pyo3::exceptions::PyValueError::new_err("start_idx out of range"));
    }
    let idx_end = end_idx.unwrap_or(nl - 1);
    if idx_end < start_idx + 2 || idx_end >= nl {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "end_idx must leave at least one host layer inside [start_idx, end_idx]",
        ));
    }
    if num_wavs == 0 || num_angles == 0 || nz == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("empty grid"));
    }
    if np_slice.len() != num_wavs {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "needle_n_per_wav must have one complex index per wavelength",
        ));
    }
    if cache_slice.len() != num_wavs * nl * 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("n_stack_cache layout mismatch"));
    }
    let want_p = requested & NREQ_P != 0;
    let want_pmb = requested & NREQ_P_MB != 0;
    let want_pmb_t = requested & NREQ_P_MB_T != 0;
    let want_pmb_a = requested & NREQ_P_MB_A != 0;
    let want_ptb = requested & NREQ_P_TB != 0;
    let want_prb = requested & NREQ_P_RB != 0;
    let want_pab = requested & NREQ_P_AB != 0;
    let want_pmb_tb = requested & NREQ_P_MB_TB != 0;
    let want_pmb_rb = requested & NREQ_P_MB_RB != 0;
    let want_pmb_ab = requested & NREQ_P_MB_AB != 0;
    let want_pt = requested & NREQ_P_T != 0;
    let want_pa = requested & NREQ_P_A != 0;
    let want_pphi = requested & NREQ_P_PHI != 0;
    let want_disp = max_disp_order(requested).is_some();
    if !calc_s && !calc_p {
        return Err(pyo3::exceptions::PyValueError::new_err("no polarization branch enabled"));
    }
    if channel > 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("channel must be 0..=3"));
    }

    // Optional per-point merit inputs (default: target 0, weight 1).
    let tgt = match &targets_r {
        Some(a) => {
            let v = a.as_slice()?;
            if v.len() != total_points {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "targets_r must have num_angles*num_wavs entries (angle-major)",
                ));
            }
            Some(v.to_vec())
        }
        None => None,
    };
    let wgt = match &weights_r {
        Some(a) => {
            let v = a.as_slice()?;
            if v.len() != total_points {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "weights_r must have num_angles*num_wavs entries (angle-major)",
                ));
            }
            Some(v.to_vec())
        }
        None => None,
    };
    let target_of = |k: usize| tgt.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_of = |k: usize| wgt.as_ref().map(|t| t[k]).unwrap_or(1.0);
    // Optional per-point merit inputs for the T/A/phase gradients
    // (default: target 0, weight 1 — same as the R pair).
    let load_pair = |a: &Option<PyReadonlyArray1<f64>>, name: &str| -> PyResult<Option<Vec<f64>>> {
        match a {
            Some(arr) => {
                let v = arr.as_slice()?;
                if v.len() != total_points {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{name} must have num_angles*num_wavs entries (angle-major)",
                    )));
                }
                Ok(Some(v.to_vec()))
            }
            None => Ok(None),
        }
    };
    let tgt_t = load_pair(&targets_t, "targets_t")?;
    let wgt_t = load_pair(&weights_t, "weights_t")?;
    let tgt_a = load_pair(&targets_a, "targets_a")?;
    let wgt_a = load_pair(&weights_a, "weights_a")?;
    let tgt_phi = load_pair(&targets_phi, "targets_phi")?;
    let wgt_phi = load_pair(&weights_phi, "weights_phi")?;
    let target_t_of = |k: usize| tgt_t.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_t_of = |k: usize| wgt_t.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_a_of = |k: usize| tgt_a.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_a_of = |k: usize| wgt_a.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_phi_of = |k: usize| tgt_phi.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_phi_of = |k: usize| wgt_phi.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let tgt_tb = load_pair(&targets_tb, "targets_tb")?;
    let wgt_tb = load_pair(&weights_tb, "weights_tb")?;
    let tgt_rb = load_pair(&targets_rb, "targets_rb")?;
    let wgt_rb = load_pair(&weights_rb, "weights_rb")?;
    let tgt_ab = load_pair(&targets_ab, "targets_ab")?;
    let wgt_ab = load_pair(&weights_ab, "weights_ab")?;
    let target_tb_of = |k: usize| tgt_tb.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_tb_of = |k: usize| wgt_tb.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_rb_of = |k: usize| tgt_rb.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_rb_of = |k: usize| wgt_rb.as_ref().map(|t| t[k]).unwrap_or(1.0);
    let target_ab_of = |k: usize| tgt_ab.as_ref().map(|t| t[k]).unwrap_or(0.0);
    let weight_ab_of = |k: usize| wgt_ab.as_ref().map(|t| t[k]).unwrap_or(1.0);

    // Incoherent flags only needed for the multiblock path.
    let want_any_pmb =
        want_pmb || want_pmb_t || want_pmb_a || want_pmb_tb || want_pmb_rb || want_pmb_ab;
    let inc = match (&incoherent_flags, want_any_pmb) {
        (_, false) => None,
        (None, true) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "NREQ_P_MB requires incoherent_flags",
            ))
        }
        (Some(a), true) => {
            let v = a.as_slice()?;
            if v.len() != nl {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "incoherent_flags must have n_layers entries",
                ));
            }
            Some(v.to_vec())
        }
    };
    let mask = match &host_mask {
        Some(a) => {
            let v = a.as_slice()?;
            if v.len() != nl {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "host_mask must have n_layers entries",
                ));
            }
            Some(v.to_vec())
        }
        None => None,
    };

    // Host maps are geometry-only: compute once, share across all points.
    let mb_locs = match &inc {
        Some(flags) => Some(locate_hosts_multiblock(thick_slice, flags, z_slice, mask.as_deref())
            .map_err(pyo3::exceptions::PyValueError::new_err)?),
        None => None,
    };
    let coh_locs: Vec<(usize, f64)> =
        if want_p || want_pt || want_pa || want_pphi || want_ptb || want_prb || want_pab || want_disp {
        z_slice
            .iter()
            .map(|&z| locate_depth_in(thick_slice, start_idx, idx_end, z))
            .collect()
    } else {
        Vec::new()
    };

    struct PointOut {
        p: [Option<Vec<f64>>; 2],
        pmb: [Option<Vec<f64>>; 2],
        q: [Option<Vec<f64>>; 2], // Q rows (order 0), flattened nz
        pt: [Option<Vec<f64>>; 2],
        pa: [Option<Vec<f64>>; 2],
        pphi: [Option<Vec<f64>>; 2],
        pmb_t: [Option<Vec<f64>>; 2],
        pmb_a: [Option<Vec<f64>>; 2],
        ptb: [Option<Vec<f64>>; 2],
        prb: [Option<Vec<f64>>; 2],
        pab: [Option<Vec<f64>>; 2],
        pmb_tb: [Option<Vec<f64>>; 2],
        pmb_rb: [Option<Vec<f64>>; 2],
        pmb_ab: [Option<Vec<f64>>; 2],
    }
    impl PointOut {
        fn empty() -> Self {
            PointOut {
                p: [None, None], pmb: [None, None], q: [None, None],
                pt: [None, None], pa: [None, None], pphi: [None, None],
                pmb_t: [None, None], pmb_a: [None, None],
                ptb: [None, None], prb: [None, None], pab: [None, None],
                pmb_tb: [None, None], pmb_rb: [None, None], pmb_ab: [None, None],
            }
        }
    }

    let pol_on = [calc_s, calc_p];

    // ── Phase A: everything expressible per point, in parallel ──
    let outs: Vec<PointOut> = py.detach(|| {
        (0..total_points)
            .into_par_iter()
            .map(|k| {
                let a = k / num_wavs;
                let w = k % num_wavs;
                let lam = wav_slice[w];
                let sin_t = sin_slice[a];
                let base = w * nl * 2;
                let ns: Vec<Complex64> = (0..nl)
                    .map(|l| Complex64::new(cache_slice[base + l * 2], cache_slice[base + l * 2 + 1]))
                    .collect();
                let nsin_fi = ns[0] * Complex64::new(sin_t, 0.0);
                let np_c = np_slice[w];
                let tgt_k = target_of(k);
                let wgt_k = weight_of(k);

                let mut o = PointOut::empty();

                // Coherent observables share ONE fields build per polarization.
                if want_p || want_pt || want_pa || want_pphi || want_ptb || want_prb || want_pab || want_disp {
                    for (pi, &on) in pol_on.iter().enumerate() {
                        if !on {
                            continue;
                        }
                        let pol = pi as i32;
                        let fields = build_stack_fields_range(
                            start_idx, idx_end, &ns, thick_slice, rv_slice, rt_slice,
                            lam, nsin_fi, pol,
                        );
                        if want_p {
                            o.p[pi] = Some(p_coherent_from_fields(
                                &fields, nsin_fi, lam, pol, np_c, tgt_k, wgt_k,
                                thick_slice, start_idx, idx_end, z_slice,
                            ));
                        }
                        if want_pt {
                            o.pt[pi] = Some(p_coherent_t_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_t_of(k), weight_t_of(k),
                                thick_slice, start_idx, idx_end, z_slice,
                            ));
                        }
                        if want_pa {
                            o.pa[pi] = Some(p_coherent_a_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_a_of(k), weight_a_of(k),
                                thick_slice, start_idx, idx_end, z_slice,
                            ));
                        }
                        if want_pphi {
                            o.pphi[pi] = Some(p_coherent_phi_from_fields(
                                &fields, nsin_fi, lam, pol, np_c, channel,
                                target_phi_of(k), weight_phi_of(k),
                                thick_slice, start_idx, idx_end, z_slice,
                            ));
                        }
                        if want_ptb {
                            o.ptb[pi] = Some(p_coherent_tb_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_tb_of(k), weight_tb_of(k),
                                thick_slice, start_idx, idx_end, z_slice,
                            ));
                        }
                        if want_prb {
                            o.prb[pi] = Some(p_coherent_rb_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_rb_of(k), weight_rb_of(k),
                                thick_slice, start_idx, idx_end, z_slice,
                            ));
                        }
                        if want_pab {
                            o.pab[pi] = Some(p_coherent_ab_from_fields(
                                &fields, nsin_fi, lam, pol, np_c,
                                target_ab_of(k), weight_ab_of(k),
                                thick_slice, start_idx, idx_end, z_slice,
                            ));
                        }
                        if want_disp {
                            let m = fields.s_left[idx_end];
                            let amp = [m.0, m.1, m.2, m.3][channel];
                            let r2 = amp.norm_sqr();
                            let mut qv = vec![0.0_f64; nz];
                            if r2 > 1e-20 {
                                for (zi, &(j, xi)) in coh_locs.iter().enumerate() {
                                    // Per-channel slope (channel-0 here would
                                    // mix r-motion into t-phase).
                                    let da = needle_slopes4_ddz(
                                        &fields, nsin_fi, j, xi, np_c, pol, lam)[channel];
                                    qv[zi] = (amp.conj() * da).im / r2;
                                }
                            }
                            o.q[pi] = Some(qv);
                        }
                    }
                }

                if let (Some(flags), Some(locs)) = (&inc, &mb_locs) {
                    for (pi, &on) in pol_on.iter().enumerate() {
                        if !on {
                            continue;
                        }
                        if want_pmb {
                            o.pmb[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thick_slice, flags, rv_slice, rt_slice,
                                np_c, PmbQuantity::R, tgt_k, wgt_k, locs, pi as i32,
                            ));
                        }
                        if want_pmb_t {
                            o.pmb_t[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thick_slice, flags, rv_slice, rt_slice,
                                np_c, PmbQuantity::T,
                                target_t_of(k), weight_t_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_a {
                            o.pmb_a[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thick_slice, flags, rv_slice, rt_slice,
                                np_c, PmbQuantity::A,
                                target_a_of(k), weight_a_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_tb {
                            o.pmb_tb[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thick_slice, flags, rv_slice, rt_slice,
                                np_c, PmbQuantity::TB,
                                target_tb_of(k), weight_tb_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_rb {
                            o.pmb_rb[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thick_slice, flags, rv_slice, rt_slice,
                                np_c, PmbQuantity::RB,
                                target_rb_of(k), weight_rb_of(k), locs, pi as i32,
                            ));
                        }
                        if want_pmb_ab {
                            o.pmb_ab[pi] = Some(p_multiblock_point(
                                lam, sin_t, &ns, thick_slice, flags, rv_slice, rt_slice,
                                np_c, PmbQuantity::AB,
                                target_ab_of(k), weight_ab_of(k), locs, pi as i32,
                            ));
                        }
                    }
                }

                o
            })
            .collect::<Vec<_>>()
    });

    // ── Phase B: spectral differentiation chain (crosses wavelengths) ──
    let max_order = max_disp_order(requested);
    // chains[pol][order][k*nz+zi]
    let disp_chain: Vec<Option<Vec<Vec<Vec<f64>>>>> = match max_order {
        None => vec![None, None],
        Some(mo) => {
            let omega: Vec<f64> =
                wav_slice.iter().map(|&l| 2.0 * std::f64::consts::PI * C_NM_PER_FS / l).collect();
            pol_on
                .iter()
                .enumerate()
                .map(|(pi, &on)| {
                    if !on || !want_disp {
                        return None;
                    }
                    if outs.iter().any(|o| o.q[pi].is_none()) {
                        return None;
                    }
                    let q0: Vec<Vec<f64>> =
                        outs.iter().map(|o| o.q[pi].clone().unwrap()).collect();
                    let mut chain = vec![q0.clone()];
                    for _ in 0..mo {
                        let prev = chain.last().unwrap();
                        chain.push(spectral_gradient_step(prev, &omega, num_wavs, num_angles, nz));
                    }
                    Some(chain)
                })
                .collect()
        }
    };
    let _ = channel;

    // ── Assemble dict ──
    let shape = [total_points, nz];
    let out = PyDict::new(py);

    macro_rules! emit {
        ($name:expr, $field:ident, $pi:expr) => {{
            let name: String = $name;
            let mut flat: Vec<f64> = Vec::with_capacity(total_points * nz);
            for o in &outs {
                match &o.$field[$pi] {
                    Some(v) => flat.extend_from_slice(v),
                    None => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "internal error: missing output buffer",
                        ))
                    }
                }
            }
            out.set_item(name.as_str(), PyArray::from_vec(py, flat).reshape(shape)?)?;
        }};
    }

    let pol_suffix = |pi: usize| if pi == 0 { "s" } else { "p" };
    if want_p {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_{}", pol_suffix(pi)), p, pi);
            }
        }
    }
    if want_pt {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_T_{}", pol_suffix(pi)), pt, pi);
            }
        }
    }
    if want_pa {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_A_{}", pol_suffix(pi)), pa, pi);
            }
        }
    }
    if want_pphi {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_PHI_{}", pol_suffix(pi)), pphi, pi);
            }
        }
    }
    if want_pmb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_{}", pol_suffix(pi)), pmb, pi);
            }
        }
    }
    if want_pmb_t {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_T_{}", pol_suffix(pi)), pmb_t, pi);
            }
        }
    }
    if want_pmb_a {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_A_{}", pol_suffix(pi)), pmb_a, pi);
            }
        }
    }
    if want_ptb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_TB_{}", pol_suffix(pi)), ptb, pi);
            }
        }
    }
    if want_prb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_RB_{}", pol_suffix(pi)), prb, pi);
            }
        }
    }
    if want_pab {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("P_AB_{}", pol_suffix(pi)), pab, pi);
            }
        }
    }
    if want_pmb_tb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_TB_{}", pol_suffix(pi)), pmb_tb, pi);
            }
        }
    }
    if want_pmb_rb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_RB_{}", pol_suffix(pi)), pmb_rb, pi);
            }
        }
    }
    if want_pmb_ab {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_AB_{}", pol_suffix(pi)), pmb_ab, pi);
            }
        }
    }
    const DISP_KEYS: [&str; 5] = ["dphi", "dgd", "dgdd", "dtod", "dfod"];
    if let Some(mo) = max_order {
        for pi in 0..2 {
            if !pol_on[pi] {
                continue;
            }
            if let Some(chain) = &disp_chain[pi] {
                for order in 0..=mo {
                    let key = format!("{}_{}", DISP_KEYS[order], pol_suffix(pi));
                    let mut flat: Vec<f64> = Vec::with_capacity(total_points * nz);
                    for row in &chain[order] {
                        flat.extend_from_slice(row);
                    }
                    out.set_item(key, PyArray::from_vec(py, flat).reshape(shape)?)?;
                }
            }
        }
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
    let n_slice = n_stack.as_slice()?;
    let d_slice = thicknesses.as_slice()?;
    let rt_slice = rough_types.as_slice()?;
    let rv_slice = rough_vals.as_slice()?;

    // Reciprocals computed ONCE per wavelength and shared read-only across all
    // grid points (and all rayon threads). Previously this Vec was allocated
    // inside every char_func call — thousands of heap allocations per scan.
    let inv_n: Vec<Complex64> = n_slice.iter().map(|n| n.recip()).collect();

    let real_vals: Vec<f64> = (0..points_real)
        .map(|i| real_min + (i as f64) * (real_max - real_min) / ((points_real - 1) as f64))
        .collect();
    let imag_vals: Vec<f64> = (0..points_imag)
        .map(|i| imag_min + (i as f64) * (imag_max - imag_min) / ((points_imag - 1) as f64))
        .collect();

    let landscape: Vec<f64> = py.detach(|| {
        (0..points_imag * points_real)
            .into_par_iter()
            .map(|idx| {
                let i = idx / points_real;
                let j = idx % points_real;
                let nr = real_vals[j];
                let ni = imag_vals[i];
                let n_eff = Complex64::new(nr, ni);
                char_func(n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, n_eff, pol)
            })
            .collect()
    });

    let land_arr = PyArray1::from_vec(py, landscape).reshape([points_imag, points_real]).unwrap();
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
    let land = landscape.as_array();
    let (n_imag, n_real) = (land.shape()[0], land.shape()[1]);

    // True median of the landscape (the previous code averaged, which the
    // `median_factor` name and the Python reference (`np.median`) do not).
    // Sentinel 1e30 cells sort to the top and so don't perturb the median,
    // whereas they badly skewed the mean.
    let mut sorted: Vec<f64> = land.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    let median = if len == 0 {
        0.0
    } else if len % 2 == 1 {
        sorted[len / 2]
    } else {
        0.5 * (sorted[len / 2 - 1] + sorted[len / 2])
    };
    let threshold = median * median_factor;

    let mut candidates = Vec::new();
    if n_real < 2 {
        return candidates;
    }
    for i in 0..n_imag {
        // Skip the first/last real columns, matching the reference
        // (`for j in range(1, len(Nr) - 1)`); all imag rows are scanned so
        // lossless modes on the Im=0 edge are still detected.
        for j in 1..n_real - 1 {
            let val = land[[i, j]];
            if val >= threshold {
                continue;
            }
            let i0 = i.saturating_sub(1);
            let i1 = (i + 1).min(n_imag - 1);
            let j0 = j.saturating_sub(1);
            let j1 = (j + 1).min(n_real - 1);
            let mut is_min = true;
            'neighbors: for ii in i0..=i1 {
                for jj in j0..=j1 {
                    if ii == i && jj == j {
                        continue;
                    }
                    if land[[ii, jj]] <= val {
                        is_min = false;
                        break 'neighbors;
                    }
                }
            }
            if is_min {
                candidates.push((real_vals[j], imag_vals[i]));
            }
        }
    }
    candidates
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
    let n_slice = n_stack.as_slice().unwrap();
    let d_slice = thicknesses.as_slice().unwrap();
    let rt_slice = rough_types.as_slice().unwrap();
    let rv_slice = rough_vals.as_slice().unwrap();

    // Reciprocals computed once and reused across every simplex evaluation.
    let inv_n: Vec<Complex64> = n_slice.iter().map(|n| n.recip()).collect();

    let mut simplex = vec![
        [x0.0, x0.1],
        [x0.0 + step, x0.1],
        [x0.0, x0.1 + step * 0.1],
    ];
    let mut values: Vec<f64> = simplex
        .iter()
        .map(|x| char_func_xy(x, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol))
        .collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;
    let mut iter = 0;

    loop {
        let mut indices: Vec<usize> = (0..3).collect();
        indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
        let (best, good, worst) = (indices[0], indices[1], indices[2]);

        let centroid = [
            (simplex[best][0] + simplex[good][0]) / 2.0,
            (simplex[best][1] + simplex[good][1]) / 2.0,
        ];
        let reflected = [
            centroid[0] + alpha * (centroid[0] - simplex[worst][0]),
            centroid[1] + alpha * (centroid[1] - simplex[worst][1]),
        ];
        let f_ref = char_func_xy(&reflected, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);

        if f_ref < values[best] {
            let expanded = [
                centroid[0] + gamma * (reflected[0] - centroid[0]),
                centroid[1] + gamma * (reflected[1] - centroid[1]),
            ];
            let f_exp = char_func_xy(&expanded, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);
            if f_exp < f_ref {
                simplex[worst] = expanded;
                values[worst] = f_exp;
            } else {
                simplex[worst] = reflected;
                values[worst] = f_ref;
            }
        } else if f_ref < values[good] {
            simplex[worst] = reflected;
            values[worst] = f_ref;
        } else {
            let contracted = [
                centroid[0] + rho * (simplex[worst][0] - centroid[0]),
                centroid[1] + rho * (simplex[worst][1] - centroid[1]),
            ];
            let f_con = char_func_xy(&contracted, n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);
            if f_con < values[worst] {
                simplex[worst] = contracted;
                values[worst] = f_con;
            } else {
                for i in 0..3 {
                    if i != best {
                        simplex[i][0] = simplex[best][0] + sigma * (simplex[i][0] - simplex[best][0]);
                        simplex[i][1] = simplex[best][1] + sigma * (simplex[i][1] - simplex[best][1]);
                        values[i] = char_func_xy(&simplex[i], n_slice, &inv_n, d_slice, rt_slice, rv_slice, lam, pol);
                    }
                }
            }
        }

        iter += 1;
        let size = ((simplex[0][0] - simplex[1][0]).powi(2) + (simplex[0][1] - simplex[1][1]).powi(2)).sqrt()
                + ((simplex[1][0] - simplex[2][0]).powi(2) + (simplex[1][1] - simplex[2][1]).powi(2)).sqrt()
                + ((simplex[2][0] - simplex[0][0]).powi(2) + (simplex[2][1] - simplex[0][1]).powi(2)).sqrt();
        if size < tol || iter >= max_iter {
            break;
        }
    }

    let best_idx = (0..3).min_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap()).unwrap();
    (simplex[best_idx][0], simplex[best_idx][1], values[best_idx])
}

// -----------------------------------------------------------------------------
// Field profile: |E(z)| through the stack for a given eigenmode
// -----------------------------------------------------------------------------
/// Data needed to compute fields inside a layer
struct LayerData {
    n: Complex64,
    cos: Complex64,
    thickness: f64,
    // inv_n no longer needed (was unused)
}

/// Compute the field profile inside the entire stack.
/// Returns (z_positions, |E|_values, layer_bounds_start, layer_bounds_end, layer_indices).
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
    let n_slice = n_stack.as_slice()?;
    let d_slice = thicknesses.as_slice()?;
    let rt_slice = rough_types.as_slice()?;
    let rv_slice = rough_vals.as_slice()?;

    let n_layers = n_slice.len();
    if n_layers < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("At least two layers required."));
    }

    let two_pi_lam = 2.0 * PI / lam;

    // Precompute layer data: n, cosθ, thickness
    let mut layers: Vec<LayerData> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let n = n_slice[i];
        let r0 = n_eff * n.recip();
        let v = Complex64::new(1.0, 0.0) - r0 * r0;
        let mut cos = v.sqrt();
        if cos.im < 0.0 {
            cos = -cos;
        }
        layers.push(LayerData {
            n,
            cos,
            thickness: d_slice[i],
        });
    }

    // Helper for Fresnel + roughness at an interface (i -> i+1)
    let interface_props = |i: usize| -> (Complex64, Complex64, Complex64, Complex64) {
        let n_curr = layers[i].n;
        let cos_curr = layers[i].cos;
        let y_curr = if pol == 0 {
            n_curr * cos_curr
        } else {
            let c = if cos_curr.norm() < 1e-12 { Complex64::new(1e-12, 0.0) } else { cos_curr };
            n_curr / c
        };
        let n_next = layers[i+1].n;
        let cos_next = layers[i+1].cos;
        let y_next = if pol == 0 {
            n_next * cos_next
        } else {
            let c = if cos_next.norm() < 1e-12 { Complex64::new(1e-12, 0.0) } else { cos_next };
            n_next / c
        };

        let den = y_curr + y_next;
        let den_safe = if den.norm() < 1e-100 { Complex64::new(1e-100, 1e-100) } else { den };
        let inv_den = den_safe.recip();
        let r12 = (y_curr - y_next) * inv_den;
        let t12 = y_curr * 2.0 * inv_den;
        let t21 = y_next * 2.0 * inv_den;
        let r21 = -r12;

        let sigma = rv_slice[i+1];
        let rtype = rt_slice[i+1];
        if rtype != 0 && sigma > 0.0 {
            let kz1 = two_pi_lam * n_curr * cos_curr;
            let kz2 = two_pi_lam * n_next * cos_next;
            if rtype == 5 {
                let f = (-2.0 * kz1 * kz2 * sigma * sigma).exp();
                (r12 * f, r21 * f, t12 * f, t21 * f)
            } else {
                let al = w_function_inner(2.0 * kz1 * sigma, rtype);
                let be = w_function_inner(2.0 * kz2 * sigma, rtype);
                let ga = w_function_inner((kz1 - kz2) * sigma, rtype);
                (r12 * al, r21 * be, t12 * ga, t21 * ga)
            }
        } else {
            (r12, r21, t12, t21)
        }
    };

    // Propagation phase through a layer (i)
    let prop_phase = |i: usize| -> Complex64 {
        let d = layers[i].thickness;
        if d <= 1e-12 {
            return Complex64::new(1.0, 0.0);
        }
        let mut beta = two_pi_lam * d * layers[i].n * layers[i].cos;
        if beta.im < 0.0 {
            beta = Complex64::new(beta.re, -beta.im);
        }
        (Complex64::new(0.0, 1.0) * beta).exp()
    };

    // ---------- Build left and right S‑matrices ----------
    // S_left[i] = S‑matrix from ambient up to the left side of layer i (i from 1 to n_layers-1)
    let mut s_left: Vec<(Complex64, Complex64, Complex64, Complex64)> = Vec::with_capacity(n_layers);
    s_left.push((Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))); // identity before ambient

    for i in 0..n_layers-1 {
        let mut sg = s_left.last().unwrap().clone();
        if i > 0 && layers[i].thickness > 1e-12 {
            let phi = prop_phase(i);
            sg = redheffer_product_complex_field_inner(
                sg.0, sg.1, sg.2, sg.3,
                Complex64::new(0.0, 0.0), phi, phi, Complex64::new(0.0, 0.0),
            );
        }
        let iface = interface_props(i);
        sg = redheffer_product_complex_field_inner(sg.0, sg.1, sg.2, sg.3, iface.0, iface.1, iface.2, iface.3);
        s_left.push(sg);
    }

    // S_right[i] = S‑matrix from substrate up to the right side of layer i (i from n_layers-2 down to 0)
    let mut s_right: Vec<Option<(Complex64, Complex64, Complex64, Complex64)>> = vec![None; n_layers];
    s_right[n_layers-1] = Some((Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)));

    for i in (0..n_layers-1).rev() {
        let mut sg = s_right[i+1].unwrap();
        if i+1 < n_layers-1 && layers[i+1].thickness > 1e-12 {
            let phi = prop_phase(i+1);
            sg = redheffer_product_complex_field_inner(
                Complex64::new(0.0, 0.0), phi, phi, Complex64::new(0.0, 0.0),
                sg.0, sg.1, sg.2, sg.3,
            );
        }
        let iface = interface_props(i);
        sg = redheffer_product_complex_field_inner(iface.0, iface.1, iface.2, iface.3, sg.0, sg.1, sg.2, sg.3);
        s_right[i] = Some(sg);
    }

    // ---------- Compute field inside each layer ----------
    let mut z_pos = Vec::new();
    let mut e_mag = Vec::new();
    let mut layer_start = Vec::new();
    let mut layer_end = Vec::new();
    let mut layer_n = Vec::new();

    let mut z_cursor = 0.0;

    for i in 1..n_layers-1 {
        let d = layers[i].thickness;
        if d <= 1e-12 {
            continue;
        }
        let sl = &s_left[i];
        let sr = s_right[i].as_ref().unwrap();
        let denom = Complex64::new(1.0, 0.0) - sl.3 * sr.0;
        let denom_safe = if denom.norm() < 1e-100 {
            Complex64::new(1e-100, 1e-100)
        } else {
            denom
        };
        let inv_denom = denom_safe.recip();
        let e_plus = sl.2 * inv_denom;
        let e_minus = sr.0 * e_plus;
        let mut beta = two_pi_lam * d * layers[i].n * layers[i].cos;
        if beta.im < 0.0 {
            beta = Complex64::new(beta.re, -beta.im);
        }

        let step = d / (points_per_layer as f64);
        for k in 0..=points_per_layer {
            let zz = k as f64 * step;
            let xi = zz / d;
            let e_z = e_plus * (Complex64::new(0.0, 1.0) * beta * xi).exp()
                    + e_minus * (-Complex64::new(0.0, 1.0) * beta * xi).exp();
            z_pos.push(z_cursor + zz);
            e_mag.push(e_z.norm());
        }
        layer_start.push(z_cursor);
        layer_end.push(z_cursor + d);
        layer_n.push(layers[i].n);
        z_cursor += d;
    }

    // Normalise E‑field to max = 1
    let max_e = e_mag.iter().copied().fold(0.0, f64::max);
    if max_e > 0.0 {
        for val in &mut e_mag {
            *val /= max_e;
        }
    }

    Ok((z_pos, e_mag, layer_start, layer_end, layer_n))
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
