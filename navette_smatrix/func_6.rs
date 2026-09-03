// func_6.rs
//
// Unified, request-driven engine. One Rust entry point solves the optical
// problem once per (wavelength, angle) point into an `OpticalState`, then
// derives only the observables the caller asked for and returns them as a
// dict { name -> ndarray }.
//
// What runs is decided entirely by the request bitmask:
//   * which polarization branches solve (s, p, or both),
//   * whether the complex p-s coherency channel runs (Mode B),
//   * which derived observables are computed,
//   * whether the cross-wavelength dispersion post-pass runs.
// `calc_s` / `calc_p` are NOT inputs; they are resolved from the request.

use num_complex::Complex64;
use num_complex::ComplexFloat;
use numpy::{PyArray, PyArrayMethods};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::func_2::{redheffer_product_cross_inner, redheffer_product_real_inner};
use crate::func_3::{solve_coherent_block_fields_dual, solve_coherent_block_fields_inner};

const POL_S: i32 = 0;
const POL_P: i32 = 1;

const MODE_A: i32 = 0; // front_block
const MODE_B: i32 = 1; // coherency_matrix
const MODE_C: i32 = 2; // fully_coherent

// Speed of light in nm/fs, so that with wavelength in nm: GD -> fs, GDD -> fs^2.
const C_NM_PER_FS: f64 = 299.792458;

// Smallest s-channel intensity for which Psi/Delta are well-defined. Below this
// the p/s ratio is treated as degenerate and (Psi, Delta) = (PI/2, 0). The
// transmission floor is far smaller because T can be vanishingly small deep in
// an absorbing stack. (Matches the original ellipsometry engine.)
const RS_FLOOR: f64 = 1e-12;
const TS_FLOOR: f64 = 1e-20;

/// Ellipsometric (Psi, Delta) from the p/s polarization intensities and the p-s
/// Stokes components. Guards a vanishing s-channel: below `floor` the ratio is
/// undefined, so it returns the degenerate (PI/2, 0).
#[inline]
fn psi_delta(num_p: f64, den_s: f64, floor: f64, s2: f64, s3: f64) -> (f64, f64) {
    if den_s < floor {
        (PI / 2.0, 0.0)
    } else {
        ((num_p / den_s).sqrt().atan(), s3.atan2(s2))
    }
}

// ─── Request bits (mirror these in the Python `Request(IntFlag)`) ────────────
pub const REQ_RS: u64 = 1 << 0;
pub const REQ_RP: u64 = 1 << 1;
pub const REQ_TS: u64 = 1 << 2;
pub const REQ_TP: u64 = 1 << 3;
pub const REQ_R_AVG: u64 = 1 << 4;
pub const REQ_T_AVG: u64 = 1 << 5;
pub const REQ_A_S: u64 = 1 << 6;
pub const REQ_A_P: u64 = 1 << 7;
pub const REQ_A_AVG: u64 = 1 << 8;
pub const REQ_PSI_R: u64 = 1 << 9;
pub const REQ_PSI_T: u64 = 1 << 10;
pub const REQ_DELTA_R: u64 = 1 << 11;
pub const REQ_DELTA_T: u64 = 1 << 12;
pub const REQ_DOP_R: u64 = 1 << 13;
pub const REQ_DOP_T: u64 = 1 << 14;
pub const REQ_DIATT_R: u64 = 1 << 15;
pub const REQ_DIATT_T: u64 = 1 << 16;
pub const REQ_S0_R: u64 = 1 << 17;
pub const REQ_S1_R: u64 = 1 << 18;
pub const REQ_S2_R: u64 = 1 << 19;
pub const REQ_S3_R: u64 = 1 << 20;
pub const REQ_S0_T: u64 = 1 << 21;
pub const REQ_S1_T: u64 = 1 << 22;
pub const REQ_S2_T: u64 = 1 << 23;
pub const REQ_S3_T: u64 = 1 << 24;
pub const REQ_PHI_RS: u64 = 1 << 25;
pub const REQ_PHI_RP: u64 = 1 << 26;
pub const REQ_PHI_TS: u64 = 1 << 27;
pub const REQ_PHI_TP: u64 = 1 << 28;
pub const REQ_RS_C: u64 = 1 << 29;
pub const REQ_RP_C: u64 = 1 << 30;
pub const REQ_TS_C: u64 = 1 << 31;
pub const REQ_TP_C: u64 = 1 << 32;
pub const REQ_CROSS_R: u64 = 1 << 33;
pub const REQ_CROSS_T: u64 = 1 << 34;
pub const REQ_RETARD_R: u64 = 1 << 35;
pub const REQ_RETARD_T: u64 = 1 << 36;
pub const REQ_DISP_R_S: u64 = 1 << 37; // emits GD/GDD/TOD/FOD_R_s
pub const REQ_DISP_R_P: u64 = 1 << 38;
pub const REQ_DISP_T_S: u64 = 1 << 39;
pub const REQ_DISP_T_P: u64 = 1 << 40;

// ─── Dependency classes (OR-reduced over the request to resolve needs) ───────
const NEEDS_S_ONLY: u64 = REQ_RS | REQ_TS | REQ_A_S | REQ_PHI_RS | REQ_PHI_TS
    | REQ_RS_C | REQ_TS_C | REQ_DISP_R_S | REQ_DISP_T_S;
const NEEDS_P_ONLY: u64 = REQ_RP | REQ_TP | REQ_A_P | REQ_PHI_RP | REQ_PHI_TP
    | REQ_RP_C | REQ_TP_C | REQ_DISP_R_P | REQ_DISP_T_P;
const NEEDS_BOTH_INT: u64 = REQ_R_AVG | REQ_T_AVG | REQ_A_AVG | REQ_PSI_R | REQ_PSI_T
    | REQ_DIATT_R | REQ_DIATT_T | REQ_S0_R | REQ_S1_R | REQ_S0_T | REQ_S1_T;
const NEEDS_CROSS: u64 = REQ_DELTA_R | REQ_DELTA_T | REQ_DOP_R | REQ_DOP_T
    | REQ_S2_R | REQ_S3_R | REQ_S2_T | REQ_S3_T
    | REQ_CROSS_R | REQ_CROSS_T | REQ_RETARD_R | REQ_RETARD_T;

/// Minimal, physically-complete solved state at one (wavelength, angle) point.
/// Intensities are totals (all incoherent echoes). Complex amplitudes are the
/// first coherent block (Modes A/B) or the whole stack (Mode C). cross_* is the
/// mode-resolved p-s coherency. Fields for unsolved pols are NaN.
#[derive(Clone, Copy)]
pub(crate) struct OpticalState {
    pub(crate) rs: f64,
    pub(crate) rp: f64,
    pub(crate) ts: f64,
    pub(crate) tp: f64,
    pub(crate) rs_c: Complex64,
    pub(crate) rp_c: Complex64,
    pub(crate) ts_c: Complex64,
    pub(crate) tp_c: Complex64,
    pub(crate) cross_r: Complex64,
    pub(crate) cross_t: Complex64,
}

/// Solve one point. Runs only the requested polarization branch(es) and the
/// coherency channel only when `need_cross`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_point(
    idx_n: usize,
    lam: f64,
    sin_theta: f64,
    n_stack: &[Complex64],
    inv_n_stack: &[Complex64],
    thick_slice: &[f64],
    inc_flags_slice: &[i32],
    rough_types_slice: &[i32],
    rough_vals_slice: &[f64],
    coherence_mode: i32,
    need_s: bool,
    need_p: bool,
    need_cross: bool,
) -> OpticalState {
    let c0 = Complex64::new(0.0, 0.0);
    let c1 = Complex64::new(1.0, 0.0);
    let nsinfi = n_stack[0] * Complex64::new(sin_theta, 0.0);

    let single_block = coherence_mode == MODE_C;
    let track_cross_channel = need_cross && coherence_mode == MODE_B;

    let mut ig_s = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut ig_p = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut cg = (c0, c1, c1, c0);
    let mut cross_t_acc = c1;

    let mut rs0 = c0;
    let mut rp0 = c0;
    let mut ts0 = c0;
    let mut tp0 = c0;
    let mut first = false;

    let mut current_idx = 0usize;
    while current_idx < idx_n {
        let next_incoh = if single_block {
            idx_n
        } else {
            let mut ni = current_idx + 1;
            while ni < idx_n && inc_flags_slice[ni] == 0 {
                ni += 1;
            }
            ni
        };

        if need_s && need_p {
            let (s_res, p_res) = solve_coherent_block_fields_dual(
                current_idx, next_incoh, n_stack, inv_n_stack, thick_slice,
                rough_vals_slice, rough_types_slice, lam, nsinfi,
            );
            let (s_rf, s_tb, s_tf, s_rb, s_rfi, s_tbi, s_tfi, s_rbi) = s_res;
            let (p_rf, p_tb, p_tf, p_rb, p_rfi, p_tbi, p_tfi, p_rbi) = p_res;

            ig_s = redheffer_product_real_inner(ig_s.0, ig_s.1, ig_s.2, ig_s.3, s_rfi, s_tbi, s_tfi, s_rbi);
            ig_p = redheffer_product_real_inner(ig_p.0, ig_p.1, ig_p.2, ig_p.3, p_rfi, p_tbi, p_tfi, p_rbi);

            if !first {
                rs0 = s_rf;
                ts0 = s_tf;
                rp0 = p_rf;
                tp0 = p_tf;
                first = true;
            }

            if track_cross_channel {
                let c_rf = p_rf * s_rf.conj();
                let c_tb = p_tb * s_tb.conj();
                let c_tf = p_tf * s_tf.conj();
                let c_rb = p_rb * s_rb.conj();
                cg = redheffer_product_cross_inner(cg.0, cg.1, cg.2, cg.3, c_rf, c_tb, c_tf, c_rb);
            } else if need_cross {
                cross_t_acc *= p_tf * s_tf.conj();
            }
        } else if need_s {
            let (s_rf, _s_tb, s_tf, _s_rb, s_rfi, s_tbi, s_tfi, s_rbi) =
                solve_coherent_block_fields_inner(
                    current_idx, next_incoh, n_stack, inv_n_stack, thick_slice,
                    rough_vals_slice, rough_types_slice, lam, nsinfi, POL_S,
                );
            ig_s = redheffer_product_real_inner(ig_s.0, ig_s.1, ig_s.2, ig_s.3, s_rfi, s_tbi, s_tfi, s_rbi);
            if !first {
                rs0 = s_rf;
                ts0 = s_tf;
                first = true;
            }
        } else if need_p {
            let (p_rf, _p_tb, p_tf, _p_rb, p_rfi, p_tbi, p_tfi, p_rbi) =
                solve_coherent_block_fields_inner(
                    current_idx, next_incoh, n_stack, inv_n_stack, thick_slice,
                    rough_vals_slice, rough_types_slice, lam, nsinfi, POL_P,
                );
            ig_p = redheffer_product_real_inner(ig_p.0, ig_p.1, ig_p.2, ig_p.3, p_rfi, p_tbi, p_tfi, p_rbi);
            if !first {
                rp0 = p_rf;
                tp0 = p_tf;
                first = true;
            }
        }

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
            let tau = (-2.0 * beta_imag).exp();

            if need_s {
                ig_s = redheffer_product_real_inner(ig_s.0, ig_s.1, ig_s.2, ig_s.3, 0.0, tau, tau, 0.0);
            }
            if need_p {
                ig_p = redheffer_product_real_inner(ig_p.0, ig_p.1, ig_p.2, ig_p.3, 0.0, tau, tau, 0.0);
            }
            if track_cross_channel {
                let tf = Complex64::new(tau, 0.0);
                cg = redheffer_product_cross_inner(cg.0, cg.1, cg.2, cg.3, c0, tf, tf, c0);
            } else if need_cross {
                cross_t_acc *= tau;
            }
        }

        current_idx = next_incoh;
    }

    let nan = f64::NAN;
    let (cross_r, cross_t) = if need_cross {
        if track_cross_channel {
            (cg.0, cg.2)
        } else {
            (rp0 * rs0.conj(), cross_t_acc)
        }
    } else {
        (c0, c0)
    };

    OpticalState {
        rs: if need_s { ig_s.0 } else { nan },
        rp: if need_p { ig_p.0 } else { nan },
        ts: if need_s { ig_s.2 } else { nan },
        tp: if need_p { ig_p.2 } else { nan },
        rs_c: rs0,
        rp_c: rp0,
        ts_c: ts0,
        tp_c: tp0,
        cross_r,
        cross_t,
    }
}

// ─── Dispersion helpers (cross-wavelength post-pass) ─────────────────────────

/// Unwrap a phase row (radians) along its length.
fn unwrap(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }
    out[0] = y[0];
    let two_pi = 2.0 * PI;
    for i in 1..n {
        let mut delta = y[i] - y[i - 1];
        delta = ((delta + PI).rem_euclid(two_pi)) - PI;
        out[i] = out[i - 1] + delta;
    }
    out
}

/// Second-order non-uniform central-difference derivative dy/dx (numpy.gradient).
fn gradient(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut d = vec![0.0; n];
    if n < 2 {
        return d;
    }
    d[0] = (y[1] - y[0]) / (x[1] - x[0]);
    d[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    for i in 1..n - 1 {
        let hd = x[i] - x[i - 1];
        let hs = x[i + 1] - x[i];
        d[i] = (-hs / (hd * (hd + hs))) * y[i - 1]
            + ((hs - hd) / (hd * hs)) * y[i]
            + (hd / (hs * (hd + hs))) * y[i + 1];
    }
    d
}

/// GD/GDD/TOD/FOD of one phase channel by successive differentiation w.r.t. omega.
/// NOTE: only physically meaningful for coherent stacks (Mode C / single block).
/// Higher orders (TOD/FOD) amplify the solver's ~1-ULP transcendental noise and
/// the repeated-gradient error; validate against a spline fit on a fine grid.
#[allow(clippy::type_complexity)]
fn dispersion_channel(
    phi: &[f64],
    omega: &[f64],
    num_angles: usize,
    num_wavs: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let total = num_angles * num_wavs;
    let mut gd = vec![0.0; total];
    let mut gdd = vec![0.0; total];
    let mut tod = vec![0.0; total];
    let mut fod = vec![0.0; total];
    for a in 0..num_angles {
        let lo = a * num_wavs;
        let hi = lo + num_wavs;
        let u = unwrap(&phi[lo..hi]);
        let d1 = gradient(&u, omega); // GD = dphi/domega
        let d2 = gradient(&d1, omega);
        let d3 = gradient(&d2, omega);
        let d4 = gradient(&d3, omega);
        gd[lo..hi].copy_from_slice(&d1);
        gdd[lo..hi].copy_from_slice(&d2);
        tod[lo..hi].copy_from_slice(&d3);
        fod[lo..hi].copy_from_slice(&d4);
    }
    (gd, gdd, tod, fod)
}

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
    if !(MODE_A..=MODE_C).contains(&coherence_mode) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coherence_mode must be 0 (front_block), 1 (coherency_matrix), or 2 (fully_coherent).",
        ));
    }
    if requested == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("empty request mask"));
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

    // ── Resolve what must be solved, purely from the request ──
    let need_cross = requested & NEEDS_CROSS != 0;
    let need_s = need_cross || requested & (NEEDS_S_ONLY | NEEDS_BOTH_INT) != 0;
    let need_p = need_cross || requested & (NEEDS_P_ONLY | NEEDS_BOTH_INT) != 0;

    // Phase buffers are needed if a phase OR a dispersion observable asks for them.
    let want_phi_rs = requested & (REQ_PHI_RS | REQ_DISP_R_S) != 0;
    let want_phi_rp = requested & (REQ_PHI_RP | REQ_DISP_R_P) != 0;
    let want_phi_ts = requested & (REQ_PHI_TS | REQ_DISP_T_S) != 0;
    let want_phi_tp = requested & (REQ_PHI_TP | REQ_DISP_T_P) != 0;

    // Build complex index cache.
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

    // ── Solve every point in parallel ──
    let states: Vec<OpticalState> = py.detach(|| {
        (0..total_points)
            .into_par_iter()
            .map(|k| {
                let a = k / num_wavs;
                let w = k % num_wavs;
                solve_point(
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
                    need_cross,
                )
            })
            .collect()
    });

    // ── Per-point derive into buffers (only for requested keys) ──
    macro_rules! f64buf {
        ($name:ident, $cond:expr) => {
            let mut $name: Option<Vec<f64>> =
                if $cond { Some(vec![0.0; total_points]) } else { None };
        };
    }
    macro_rules! cbuf {
        ($name:ident, $bit:expr) => {
            let mut $name: Option<Vec<Complex64>> = if requested & $bit != 0 {
                Some(vec![Complex64::new(0.0, 0.0); total_points])
            } else {
                None
            };
        };
    }
    macro_rules! put {
        ($buf:ident, $k:expr, $val:expr) => {
            if let Some(b) = $buf.as_mut() {
                b[$k] = $val;
            }
        };
    }

    f64buf!(b_rs, requested & REQ_RS != 0);
    f64buf!(b_rp, requested & REQ_RP != 0);
    f64buf!(b_ts, requested & REQ_TS != 0);
    f64buf!(b_tp, requested & REQ_TP != 0);
    f64buf!(b_ravg, requested & REQ_R_AVG != 0);
    f64buf!(b_tavg, requested & REQ_T_AVG != 0);
    f64buf!(b_as, requested & REQ_A_S != 0);
    f64buf!(b_ap, requested & REQ_A_P != 0);
    f64buf!(b_aavg, requested & REQ_A_AVG != 0);
    f64buf!(b_psi_r, requested & REQ_PSI_R != 0);
    f64buf!(b_psi_t, requested & REQ_PSI_T != 0);
    f64buf!(b_delta_r, requested & REQ_DELTA_R != 0);
    f64buf!(b_delta_t, requested & REQ_DELTA_T != 0);
    f64buf!(b_dop_r, requested & REQ_DOP_R != 0);
    f64buf!(b_dop_t, requested & REQ_DOP_T != 0);
    f64buf!(b_diatt_r, requested & REQ_DIATT_R != 0);
    f64buf!(b_diatt_t, requested & REQ_DIATT_T != 0);
    f64buf!(b_s0r, requested & REQ_S0_R != 0);
    f64buf!(b_s1r, requested & REQ_S1_R != 0);
    f64buf!(b_s2r, requested & REQ_S2_R != 0);
    f64buf!(b_s3r, requested & REQ_S3_R != 0);
    f64buf!(b_s0t, requested & REQ_S0_T != 0);
    f64buf!(b_s1t, requested & REQ_S1_T != 0);
    f64buf!(b_s2t, requested & REQ_S2_T != 0);
    f64buf!(b_s3t, requested & REQ_S3_T != 0);
    f64buf!(b_retard_r, requested & REQ_RETARD_R != 0);
    f64buf!(b_retard_t, requested & REQ_RETARD_T != 0);
    // phase buffers: allocated if phase OR dispersion wants them
    f64buf!(b_phi_rs, want_phi_rs);
    f64buf!(b_phi_rp, want_phi_rp);
    f64buf!(b_phi_ts, want_phi_ts);
    f64buf!(b_phi_tp, want_phi_tp);

    cbuf!(b_rs_c, REQ_RS_C);
    cbuf!(b_rp_c, REQ_RP_C);
    cbuf!(b_ts_c, REQ_TS_C);
    cbuf!(b_tp_c, REQ_TP_C);
    cbuf!(b_cross_r, REQ_CROSS_R);
    cbuf!(b_cross_t, REQ_CROSS_T);

    for (k, s) in states.iter().enumerate() {
        let rs = s.rs;
        let rp = s.rp;
        let ts = s.ts;
        let tp = s.tp;

        put!(b_rs, k, rs);
        put!(b_rp, k, rp);
        put!(b_ts, k, ts);
        put!(b_tp, k, tp);
        put!(b_ravg, k, 0.5 * (rs + rp));
        put!(b_tavg, k, 0.5 * (ts + tp));
        put!(b_as, k, 1.0 - rs - ts);
        put!(b_ap, k, 1.0 - rp - tp);
        put!(b_aavg, k, 1.0 - 0.5 * (rs + rp) - 0.5 * (ts + tp));

        // Stokes (reflection)
        let s0r = rp + rs;
        let s1r = rp - rs;
        let s2r = -2.0 * s.cross_r.re + 0.0;
        let s3r = -2.0 * s.cross_r.im + 0.0;
        put!(b_s0r, k, s0r);
        put!(b_s1r, k, s1r);
        put!(b_s2r, k, s2r);
        put!(b_s3r, k, s3r);
        // Stokes (transmission)
        let s0t = tp + ts;
        let s1t = tp - ts;
        let s2t = 2.0 * s.cross_t.re + 0.0;
        let s3t = 2.0 * s.cross_t.im + 0.0;
        put!(b_s0t, k, s0t);
        put!(b_s1t, k, s1t);
        put!(b_s2t, k, s2t);
        put!(b_s3t, k, s3t);

        put!(b_diatt_r, k, s1r / (s0r + 1e-20));
        put!(b_diatt_t, k, s1t / (s0t + 1e-20));

        put!(b_dop_r, k, (s1r * s1r + s2r * s2r + s3r * s3r).sqrt() / (s0r + 1e-20));
        put!(b_dop_t, k, ((s1t * s1t + s2t * s2t + s3t * s3t).sqrt() / (s0t + 1e-20)).min(1.0));

        let (psi_r, delta_r) = psi_delta(rp, rs, RS_FLOOR, s2r, s3r);
        put!(b_psi_r, k, psi_r);
        put!(b_delta_r, k, delta_r);
        let (psi_t, delta_t) = psi_delta(tp, ts, TS_FLOOR, s2t, s3t);
        put!(b_psi_t, k, psi_t);
        put!(b_delta_t, k, delta_t);

        // Retardance == arg(cross): identical quantity to Delta (BW convention aside).
        put!(b_retard_r, k, s.cross_r.arg());
        put!(b_retard_t, k, s.cross_t.arg());

        // Absolute phases (admittance convention; phi_rp differs by pi from BW).
        put!(b_phi_rs, k, s.rs_c.arg());
        put!(b_phi_rp, k, s.rp_c.arg());
        put!(b_phi_ts, k, s.ts_c.arg());
        put!(b_phi_tp, k, s.tp_c.arg());

        if let Some(b) = b_rs_c.as_mut() { b[k] = s.rs_c; }
        if let Some(b) = b_rp_c.as_mut() { b[k] = s.rp_c; }
        if let Some(b) = b_ts_c.as_mut() { b[k] = s.ts_c; }
        if let Some(b) = b_tp_c.as_mut() { b[k] = s.tp_c; }
        if let Some(b) = b_cross_r.as_mut() { b[k] = s.cross_r; }
        if let Some(b) = b_cross_t.as_mut() { b[k] = s.cross_t; }
    }

    // ── Dispersion post-pass (cross-wavelength) ──
    let omega: Vec<f64> = wav_slice.iter().map(|&l| 2.0 * PI * C_NM_PER_FS / l).collect();
    let disp = |phi: &Option<Vec<f64>>| -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        phi.as_ref().map(|p| dispersion_channel(p, &omega, num_angles, num_wavs))
    };
    let disp_r_s = if requested & REQ_DISP_R_S != 0 { disp(&b_phi_rs) } else { None };
    let disp_r_p = if requested & REQ_DISP_R_P != 0 { disp(&b_phi_rp) } else { None };
    let disp_t_s = if requested & REQ_DISP_T_S != 0 { disp(&b_phi_ts) } else { None };
    let disp_t_p = if requested & REQ_DISP_T_P != 0 { disp(&b_phi_tp) } else { None };

    // ── Assemble dict (only requested keys) ──
    let shape = [num_angles, num_wavs];
    let out = PyDict::new(py);

    macro_rules! emit_f64 {
        ($name:expr, $buf:expr) => {
            if let Some(b) = $buf {
                out.set_item($name, PyArray::from_vec(py, b).reshape(shape)?)?;
            }
        };
    }
    macro_rules! emit_c {
        ($name:expr, $buf:expr) => {
            if let Some(b) = $buf {
                out.set_item($name, PyArray::from_vec(py, b).reshape(shape)?)?;
            }
        };
    }

    emit_f64!("Rs", b_rs);
    emit_f64!("Rp", b_rp);
    emit_f64!("Ts", b_ts);
    emit_f64!("Tp", b_tp);
    emit_f64!("R_avg", b_ravg);
    emit_f64!("T_avg", b_tavg);
    emit_f64!("A_s", b_as);
    emit_f64!("A_p", b_ap);
    emit_f64!("A_avg", b_aavg);
    emit_f64!("Psi_R", b_psi_r);
    emit_f64!("Psi_T", b_psi_t);
    emit_f64!("Delta_R", b_delta_r);
    emit_f64!("Delta_T", b_delta_t);
    emit_f64!("DOP_R", b_dop_r);
    emit_f64!("DOP_T", b_dop_t);
    emit_f64!("Diattenuation_R", b_diatt_r);
    emit_f64!("Diattenuation_T", b_diatt_t);
    emit_f64!("S0_R", b_s0r);
    emit_f64!("S1_R", b_s1r);
    emit_f64!("S2_R", b_s2r);
    emit_f64!("S3_R", b_s3r);
    emit_f64!("S0_T", b_s0t);
    emit_f64!("S1_T", b_s1t);
    emit_f64!("S2_T", b_s2t);
    emit_f64!("S3_T", b_s3t);
    emit_f64!("Retardance_R", b_retard_r);
    emit_f64!("Retardance_T", b_retard_t);

    // phases emitted only if explicitly requested (not merely needed for dispersion)
    if requested & REQ_PHI_RS != 0 { emit_f64!("phi_rs", b_phi_rs); }
    if requested & REQ_PHI_RP != 0 { emit_f64!("phi_rp", b_phi_rp); }
    if requested & REQ_PHI_TS != 0 { emit_f64!("phi_ts", b_phi_ts); }
    if requested & REQ_PHI_TP != 0 { emit_f64!("phi_tp", b_phi_tp); }

    emit_c!("rs_c", b_rs_c);
    emit_c!("rp_c", b_rp_c);
    emit_c!("ts_c", b_ts_c);
    emit_c!("tp_c", b_tp_c);
    emit_c!("cross_R", b_cross_r);
    emit_c!("cross_T", b_cross_t);

    macro_rules! emit_disp {
        ($d:expr, $g:expr, $gg:expr, $t:expr, $f:expr) => {
            if let Some((gd, gdd, tod, fod)) = $d {
                out.set_item($g, PyArray::from_vec(py, gd).reshape(shape)?)?;
                out.set_item($gg, PyArray::from_vec(py, gdd).reshape(shape)?)?;
                out.set_item($t, PyArray::from_vec(py, tod).reshape(shape)?)?;
                out.set_item($f, PyArray::from_vec(py, fod).reshape(shape)?)?;
            }
        };
    }
    emit_disp!(disp_r_s, "GD_R_s", "GDD_R_s", "TOD_R_s", "FOD_R_s");
    emit_disp!(disp_r_p, "GD_R_p", "GDD_R_p", "TOD_R_p", "FOD_R_p");
    emit_disp!(disp_t_s, "GD_T_s", "GDD_T_s", "TOD_T_s", "FOD_T_s");
    emit_disp!(disp_t_p, "GD_T_p", "GDD_T_p", "TOD_T_p", "FOD_T_p");

    Ok(out.into())
}
