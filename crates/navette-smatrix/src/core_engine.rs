//! core_engine.rs
//!
//! Unified, request-driven engine. One Rust entry point solves the optical
//! problem once per (wavelength, angle) point into an `OpticalState`, then
//! derives only the observables the caller asked for and returns them as a
//! dict { name -> ndarray }.
//!
//! What runs is decided entirely by the request bitmask:
//!   * which polarization branches solve (s, p, or both),
//!   * whether the complex p-s coherency channel runs (Mode B),
//!   * which derived observables are computed,
//!   * whether the cross-wavelength dispersion post-pass runs.
//! `calc_s` / `calc_p` are NOT inputs; they are resolved from the request.

use num_complex::Complex64;
use num_complex::ComplexFloat;
use std::f64::consts::PI;

use crate::coherent_block::{solve_coherent_block_fields_dual, solve_coherent_block_fields_inner};
use crate::optics_core::{
    grad_nonuniform as gradient, redheffer_product_cross_inner, redheffer_product_real_inner,
};

/// s-polarization branch selector.
pub const POL_S: i32 = 0;
/// p-polarization branch selector.
pub const POL_P: i32 = 1;

/// Mode A: first coherent block only (front block).
pub const MODE_A: i32 = 0;
/// Mode B: coherency-matrix cascade over incoherent echoes.
pub const MODE_B: i32 = 1;
/// Mode C: fully coherent whole-stack solve.
pub const MODE_C: i32 = 2;

// Speed of light in nm/fs, so that with wavelength in nm: GD -> fs, GDD -> fs^2.

// Smallest s-channel intensity for which Psi/Delta are well-defined. Below this
// the p/s ratio is treated as degenerate and (Psi, Delta) = (PI/2, 0). The
// transmission floor is far smaller because T can be vanishingly small deep in
// an absorbing stack. (Matches the original ellipsometry engine.)
/// Reflectance floor guarding Ψ/Δ extraction at near-zero signal.
pub const RS_FLOOR: f64 = 1e-12;
/// Transmittance floor guarding Ψ/Δ extraction at near-zero signal.
pub const TS_FLOOR: f64 = 1e-20;

// ─── Request bits (mirror these in the Python `Request(IntFlag)`) ────────────
/// Request s-polarized reflectance Rs.
pub const REQ_RS: u64 = 1 << 0;
/// Request p-polarized reflectance Rp.
pub const REQ_RP: u64 = 1 << 1;
/// Request s-polarized transmittance Ts.
pub const REQ_TS: u64 = 1 << 2;
/// Request p-polarized transmittance Tp.
pub const REQ_TP: u64 = 1 << 3;
/// Request unpolarized reflectance (Rs + Rp) / 2.
pub const REQ_R_AVG: u64 = 1 << 4;
/// Request unpolarized transmittance (Ts + Tp) / 2.
pub const REQ_T_AVG: u64 = 1 << 5;
/// Request s-polarized absorptance As = 1 − Rs − Ts.
pub const REQ_A_S: u64 = 1 << 6;
/// Request p-polarized absorptance Ap = 1 − Rp − Tp.
pub const REQ_A_P: u64 = 1 << 7;
/// Request unpolarized absorptance.
pub const REQ_A_AVG: u64 = 1 << 8;
/// Request reflection ellipsometric Ψ (tan Ψ = |rp/rs|).
pub const REQ_PSI_R: u64 = 1 << 9;
/// Request transmission ellipsometric Ψ.
pub const REQ_PSI_T: u64 = 1 << 10;
/// Request reflection ellipsometric Δ (needs the p-s coherency channel).
pub const REQ_DELTA_R: u64 = 1 << 11;
/// Request transmission ellipsometric Δ (needs the coherency channel).
pub const REQ_DELTA_T: u64 = 1 << 12;
/// Request reflected degree of polarization.
pub const REQ_DOP_R: u64 = 1 << 13;
/// Request transmitted degree of polarization.
pub const REQ_DOP_T: u64 = 1 << 14;
/// Request reflection diattenuation.
pub const REQ_DIATT_R: u64 = 1 << 15;
/// Request transmission diattenuation.
pub const REQ_DIATT_T: u64 = 1 << 16;
/// Request reflected Stokes S0 (total intensity).
pub const REQ_S0_R: u64 = 1 << 17;
/// Request reflected Stokes S1.
pub const REQ_S1_R: u64 = 1 << 18;
/// Request reflected Stokes S2 (needs the coherency channel).
pub const REQ_S2_R: u64 = 1 << 19;
/// Request reflected Stokes S3 (needs the coherency channel).
pub const REQ_S3_R: u64 = 1 << 20;
/// Request transmitted Stokes S0.
pub const REQ_S0_T: u64 = 1 << 21;
/// Request transmitted Stokes S1.
pub const REQ_S1_T: u64 = 1 << 22;
/// Request transmitted Stokes S2 (needs the coherency channel).
pub const REQ_S2_T: u64 = 1 << 23;
/// Request transmitted Stokes S3 (needs the coherency channel).
pub const REQ_S3_T: u64 = 1 << 24;
/// Request absolute phase of reflected s amplitude (needs complex amps).
pub const REQ_PHI_RS: u64 = 1 << 25;
/// Request absolute phase of reflected p amplitude.
pub const REQ_PHI_RP: u64 = 1 << 26;
/// Request absolute phase of transmitted s amplitude.
pub const REQ_PHI_TS: u64 = 1 << 27;
/// Request absolute phase of transmitted p amplitude.
pub const REQ_PHI_TP: u64 = 1 << 28;
/// Request complex reflected s amplitude (first coherent block).
pub const REQ_RS_C: u64 = 1 << 29;
/// Request complex reflected p amplitude.
pub const REQ_RP_C: u64 = 1 << 30;
/// Request complex transmitted s amplitude.
pub const REQ_TS_C: u64 = 1 << 31;
/// Request complex transmitted p amplitude.
pub const REQ_TP_C: u64 = 1 << 32;
/// Request reflected p-s cross-coherency (forces the cross channel).
pub const REQ_CROSS_R: u64 = 1 << 33;
/// Request transmitted p-s cross-coherency.
pub const REQ_CROSS_T: u64 = 1 << 34;
/// Request reflection retardance.
pub const REQ_RETARD_R: u64 = 1 << 35;
/// Request transmission retardance.
pub const REQ_RETARD_T: u64 = 1 << 36;
pub const REQ_DISP_R_S: u64 = 1 << 37; // emits GD/GDD/TOD/FOD_R_s
pub const REQ_DISP_R_P: u64 = 1 << 38;
pub const REQ_DISP_T_S: u64 = 1 << 39;
pub const REQ_DISP_T_P: u64 = 1 << 40;

// ─── Dependency resolution (request mask → what must run) ────────────────────
//
// Two orthogonal axes are resolved from the request, each by a single OR-reduce:
//
//   * polarization — which branch(es) must solve at all: s, p, or both.
//   * compute level — how much each solved branch must produce. The levels are
//     nested supersets, so the engine runs at the maximum level any requested
//     observable demands:
//        Intensities : real |r|^2,|t|^2 accumulators only   (cheapest; func_5 path)
//        ComplexAmps : + first-block complex amplitudes      (phases, dispersion, *_c)
//        Cross       : + p-s coherency channel               (Delta, DOP, S2/S3, retardance)
//
// The cross channel intrinsically couples s and p, so a Cross-level request
// forces both polarizations regardless of the per-polarization usage masks.

// Polarization usage: which branch each observable reads from.
pub const USES_S: u64 = REQ_RS | REQ_TS | REQ_A_S | REQ_PHI_RS | REQ_PHI_TS
    | REQ_RS_C | REQ_TS_C | REQ_DISP_R_S | REQ_DISP_T_S;
pub const USES_P: u64 = REQ_RP | REQ_TP | REQ_A_P | REQ_PHI_RP | REQ_PHI_TP
    | REQ_RP_C | REQ_TP_C | REQ_DISP_R_P | REQ_DISP_T_P;
pub const USES_BOTH: u64 = REQ_R_AVG | REQ_T_AVG | REQ_A_AVG | REQ_PSI_R | REQ_PSI_T
    | REQ_DIATT_R | REQ_DIATT_T | REQ_S0_R | REQ_S1_R | REQ_S0_T | REQ_S1_T;

// Compute-level demand. NEEDS_COMPLEX: observables that need the first-block
// complex amplitudes (absolute phase, dispersion, raw complex coefficients).
// NEEDS_CROSS: observables that need the p-s coherency channel.
pub const NEEDS_COMPLEX: u64 = REQ_PHI_RS | REQ_PHI_RP | REQ_PHI_TS | REQ_PHI_TP
    | REQ_RS_C | REQ_RP_C | REQ_TS_C | REQ_TP_C
    | REQ_DISP_R_S | REQ_DISP_R_P | REQ_DISP_T_S | REQ_DISP_T_P;
pub const NEEDS_CROSS: u64 = REQ_DELTA_R | REQ_DELTA_T | REQ_DOP_R | REQ_DOP_T
    | REQ_S2_R | REQ_S3_R | REQ_S2_T | REQ_S3_T
    | REQ_CROSS_R | REQ_CROSS_T | REQ_RETARD_R | REQ_RETARD_T;

/// How much each solved polarization branch must compute. Strictly increasing:
/// each level is a superset of the one before, so the resolved level is the max
/// over all requested observables.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Real |r|^2/|t|^2 accumulators only (cheapest).
    Intensities,
    /// Plus first-block complex amplitudes (phases, dispersion).
    ComplexAmps,
    /// Plus the p-s coherency channel (Delta, DOP, S2/S3, retardance).
    Cross,
}

/// Everything `core_engine` needs to steer the solve, derived once from the mask.
pub struct Plan {
    /// Solve the s branch.
    pub need_s: bool,
    /// Solve the p branch.
    pub need_p: bool,
    /// Track the p-s coherency channel (forces both branches).
    pub need_cross: bool,
    /// Maximum compute level over all requested observables.
    pub level: Level,
}

/// Resolve the request mask into a concrete solve plan. Single source of truth:
/// polarization branches and compute level both fall out of the masks above.
pub fn resolve_plan(requested: u64) -> Plan {
    let need_cross = requested & NEEDS_CROSS != 0;
    let need_s = need_cross || requested & (USES_S | USES_BOTH) != 0;
    let need_p = need_cross || requested & (USES_P | USES_BOTH) != 0;
    let level = if need_cross {
        Level::Cross
    } else if requested & NEEDS_COMPLEX != 0 {
        Level::ComplexAmps
    } else {
        Level::Intensities
    };
    Plan { need_s, need_p, need_cross, level }
}

/// Minimal, physically-complete solved state at one (wavelength, angle) point.
/// Intensities are totals (all incoherent echoes). Complex amplitudes are the
/// first coherent block (Modes A/B) or the whole stack (Mode C). cross_* is the
/// mode-resolved p-s coherency. Fields for unsolved pols are NaN.
#[derive(Clone, Copy)]
pub struct OpticalState {
    /// Total s reflectance (all incoherent echoes).
    pub rs: f64,
    /// Total p reflectance.
    pub rp: f64,
    /// Total s transmittance.
    pub ts: f64,
    /// Total p transmittance.
    pub tp: f64,
    /// Complex reflected s amplitude (first block / whole stack in Mode C).
    pub rs_c: Complex64,
    /// Complex reflected p amplitude.
    pub rp_c: Complex64,
    /// Complex transmitted s amplitude.
    pub ts_c: Complex64,
    /// Complex transmitted p amplitude.
    pub tp_c: Complex64,
    /// Mode-resolved reflected p-s coherency.
    pub cross_r: Complex64,
    /// Mode-resolved transmitted p-s coherency.
    pub cross_t: Complex64,
}

/// Solve one point. Runs only the requested polarization branch(es) and the
/// coherency channel only when `need_cross`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn solve_point(
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

/// Solve one point at the `Intensities` level: real |r|^2,|t|^2 accumulators
/// only, for whichever polarization(s) are requested. No complex amplitudes are
/// captured and the coherency channel never runs — this is the lean photometric
/// path (cf. the legacy `core_engine_photometry_only`). The returned state has
/// real intensities for solved pols (NaN for unsolved ones) and NaN/zero complex
/// fields, which are guaranteed unread at this level (requesting any of them
/// would have lifted the plan to `ComplexAmps`/`Cross`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn solve_point_intensity(
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
) -> OpticalState {
    let nsinfi = n_stack[0] * Complex64::new(sin_theta, 0.0);
    let single_block = coherence_mode == MODE_C;

    let mut ig_s = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut ig_p = (0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);

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
            let (_, _, _, _, s_rfi, s_tbi, s_tfi, s_rbi) = s_res;
            let (_, _, _, _, p_rfi, p_tbi, p_tfi, p_rbi) = p_res;
            ig_s = redheffer_product_real_inner(ig_s.0, ig_s.1, ig_s.2, ig_s.3, s_rfi, s_tbi, s_tfi, s_rbi);
            ig_p = redheffer_product_real_inner(ig_p.0, ig_p.1, ig_p.2, ig_p.3, p_rfi, p_tbi, p_tfi, p_rbi);
        } else if need_s {
            let (_, _, _, _, s_rfi, s_tbi, s_tfi, s_rbi) = solve_coherent_block_fields_inner(
                current_idx, next_incoh, n_stack, inv_n_stack, thick_slice,
                rough_vals_slice, rough_types_slice, lam, nsinfi, POL_S,
            );
            ig_s = redheffer_product_real_inner(ig_s.0, ig_s.1, ig_s.2, ig_s.3, s_rfi, s_tbi, s_tfi, s_rbi);
        } else if need_p {
            let (_, _, _, _, p_rfi, p_tbi, p_tfi, p_rbi) = solve_coherent_block_fields_inner(
                current_idx, next_incoh, n_stack, inv_n_stack, thick_slice,
                rough_vals_slice, rough_types_slice, lam, nsinfi, POL_P,
            );
            ig_p = redheffer_product_real_inner(ig_p.0, ig_p.1, ig_p.2, ig_p.3, p_rfi, p_tbi, p_tfi, p_rbi);
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
        }

        current_idx = next_incoh;
    }

    let nan = f64::NAN;
    let cnan = Complex64::new(nan, nan);
    OpticalState {
        rs: if need_s { ig_s.0 } else { nan },
        rp: if need_p { ig_p.0 } else { nan },
        ts: if need_s { ig_s.2 } else { nan },
        tp: if need_p { ig_p.2 } else { nan },
        rs_c: cnan,
        rp_c: cnan,
        ts_c: cnan,
        tp_c: cnan,
        cross_r: Complex64::new(0.0, 0.0),
        cross_t: Complex64::new(0.0, 0.0),
    }
}

// ─── Dispersion helpers (cross-wavelength post-pass) ─────────────────────────

/// Unwrap a phase row (radians) along its length.
pub fn unwrap(y: &[f64]) -> Vec<f64> {
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

/// GD/GDD/TOD/FOD of one phase channel by successive differentiation w.r.t. omega.
/// NOTE: only physically meaningful for coherent stacks (Mode C / single block).
/// Higher orders (TOD/FOD) amplify the solver's ~1-ULP transcendental noise and
/// the repeated-gradient error; validate against a spline fit on a fine grid.
#[allow(clippy::type_complexity)]
pub fn dispersion_channel(
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
