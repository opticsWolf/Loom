// func_needle_engine.rs
//
// Request-driven, rayon-parallel Python API for the analytic needle operator.
// Mirrors the conventions of func_4's `core_engine`: one Rust entry point
// evaluates every requested observable per (wavelength, angle) point — with
// the coherent-block partial matrices built ONCE per point and shared across
// observables (the "single-sweep" property) — then returns a dict of ndarrays.
//
// What runs is decided entirely by the request bitmask (mirror these in a
// Python `NeedleRequest(IntFlag)`):
//   * NREQ_P     — coherent merit gradient P(z)      (sub-block confined)
//   * NREQ_P_MB  — incoherent-aware P(z)             (Modes A/B, needs flags)
//   * NREQ_DPHI / NREQ_DGD / NREQ_DGDD / NREQ_DTOD / NREQ_DFOD
//                — phase-dispersion sensitivities    (∂φ/∂δ … ∂FOD/∂δ)
//
// Polarization branches are NOT inputs; they are resolved from `calc_s` /
// `calc_p` flags so both polarizations can ride one parallel sweep.
//
// Merit convention: P uses targets/weights per spectral point,
// P_point = 2·w·(R − R_target)·Re{conj(r)·∂r/∂δ} accumulated over points into
// each z. Dispersion channels are emitted RAW (per point, per z); aggregate
// merit gradients at the call site:
//   ∂F/∂δ(z) = Σ_k 2·w_k·(GDD_k − GDD_t_k)·dGDD[k][z]

use numpy::{PyArray, PyArrayMethods, PyReadonlyArray1};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const C_NM_PER_FS: f64 = 299.792458;

use crate::needle_operator::{
    build_stack_fields_range, locate_depth_in, locate_hosts_multiblock, needle_dr_ddz,
    p_coherent_from_fields, p_multiblock_point, spectral_gradient_step,
};

// ─── Request bits ────────────────────────────────────────────────────────────
pub const NREQ_P: u64 = 1 << 0; // coherent P(z)
pub const NREQ_P_MB: u64 = 1 << 1; // multiblock P(z) through intensity cascade
pub const NREQ_DPHI: u64 = 1 << 2; // ∂φ/∂δ
pub const NREQ_DGD: u64 = 1 << 3; // ∂GD/∂δ
pub const NREQ_DGDD: u64 = 1 << 4; // ∂GDD/∂δ
pub const NREQ_DTOD: u64 = 1 << 5; // ∂TOD/∂δ
pub const NREQ_DFOD: u64 = 1 << 6; // ∂FOD/∂δ

/// Highest dispersion derivative order implied by the request mask.
fn max_disp_order(requested: u64) -> Option<usize> {
    let orders = [
        (NREQ_DPHI, 0usize),
        (NREQ_DGD, 1),
        (NREQ_DGDD, 2),
        (NREQ_DTOD, 3),
        (NREQ_DFOD, 4),
    ];
    orders
        .iter()
        .filter(|(bit, _)| requested & bit != 0)
        .map(|(_, order)| *order)
        .max()
}

#[pyfunction]
#[pyo3(name = "needle_engine")]
#[pyo3(signature = (
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    rough_types, rough_vals, needle_n_per_wav, z_grid,
    requested,
    incoherent_flags=None, targets_r=None, weights_r=None,
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
    if idx_end <= start_idx + 2 || idx_end >= nl {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "end_idx must leave at least one interior host layer inside [start_idx, end_idx]",
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

    // Incoherent flags only needed for the multiblock path.
    let inc = match (&incoherent_flags, want_pmb) {
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
    let coh_locs: Vec<(usize, f64)> = if want_p || want_disp {
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
    }
    impl PointOut {
        fn empty() -> Self {
            PointOut { p: [None, None], pmb: [None, None], q: [None, None] }
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
                if want_p || want_disp {
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
                        if want_disp {
                            let m = fields.s_left[idx_end];
                            let amp = [m.0, m.1, m.2, m.3][channel];
                            let r2 = amp.norm_sqr();
                            let mut qv = vec![0.0_f64; nz];
                            if r2 > 1e-20 {
                                for (zi, &(j, xi)) in coh_locs.iter().enumerate() {
                                    let dr =
                                        needle_dr_ddz(&fields, nsin_fi, j, xi, np_c, pol, lam);
                                    qv[zi] = (amp.conj() * dr).im / r2;
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
                        o.pmb[pi] = Some(p_multiblock_point(
                            lam, sin_t, &ns, thick_slice, flags, rv_slice, rt_slice,
                            np_c, tgt_k, wgt_k, locs, pi as i32,
                        ));
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
    if want_pmb {
        for pi in 0..2 {
            if pol_on[pi] {
                emit!(format!("Pmb_{}", pol_suffix(pi)), pmb, pi);
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
