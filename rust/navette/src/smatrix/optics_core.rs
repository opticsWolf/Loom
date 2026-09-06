//! Pure-Rust shared primitives for the whole crate: constants, fast complex
//! kernels, the roughness form factor, the three Redheffer star products, and
//! the non-uniform spectral differentiation operator.
//!
//! This module is the SINGLE SOURCE OF TRUTH for everything that used to be
//! duplicated across func_0/func_1/func_2/func_4 and the needle operator.
//! It deliberately carries no pyo3 / numpy dependencies so it can be compiled
//! standalone (e.g. into the needle verification crate via `#[path]`).
//!
//! Numerical behaviour is byte-identical to the code that was moved here;
//! nothing was re-derived or "cleaned up".

use num_complex::Complex64;
use num_complex::ComplexFloat;

/// √3 (roughness/field-profile closed forms).
pub const SQRT3: f64 = 1.73205080757;
/// Denominator regularization floor shared by the field-amplitude star product.
pub const LOG_MIN: f64 = 1e-100;
/// p-pol cosθ magnitude guard (matches coherent_block).
pub const EPS_COS: f64 = 1e-12;
/// Machine epsilon used by the intensity/cross star regularization (inv := 0).
pub const DBL_EPS: f64 = 2.22e-16;
/// Speed of light in nm/fs — ω = 2π·c/λ with λ in nm gives rad/fs.
pub const C_NM_PER_FS: f64 = 299.792458;

/// Propagation phase of the differential-phase reference: the phase a plane
/// wave accumulates crossing an equivalent layer of incidence medium —
/// thickness `total_d`, real index `n_inc_re` — at incidence angle
/// `theta_inc_deg` (degrees, in the incidence medium, so no Snell step):
/// `passes · 2π · n · d · cosθ / λ`.
///
/// `wavelength` and `total_d` must share length units (nm in this crate).
/// `passes` is 1 for transmitted (single traversal, `PDts`/`PDtp`) and 2
/// for a reflection round trip. The incidence medium is assumed lossless;
/// pass `Re(n)` — any extinction only attenuates, it never phases.
/// Pure function (no solves); cost is one `cos` per call — hoist per angle
/// in hot loops via [`reference_wavenumber`].
///
/// Sign convention: `+kD` matches this crate's forward-propagation phase
/// (pinned by the `solver_propagation_sign_matches_reference` test — an
/// all-matched slab of D simulated by the solver has `arg(tf) = +kD`).
/// That is the conjugate of Macleod/`e^{+iωt}` textbooks; the crate is
/// self-consistent (all phase demands, needle `P_PHI` and GD/GDD share it),
/// so differential and absolute demands agree with each other exactly —
/// only textbook-imported target numbers need conjugating.
#[inline]
pub fn reference_phase(
    wavelength: f64,
    n_inc_re: f64,
    theta_inc_deg: f64,
    total_d: f64,
    passes: f64,
) -> f64 {
    passes * reference_wavenumber(wavelength, n_inc_re, theta_inc_deg) * total_d
}

/// Axial wavenumber part of [`reference_phase`]: `2π · n · cosθ / λ`
/// (radians per unit thickness; the caller scales by `passes · D`).
/// The needle gain correction for differential
/// demands is built from this: `dM/dD = Σ −2·kz·w·Δ/tol²` over folded points
/// (exact pre-fold data), a position-independent shift of `P(z)` that never
/// moves the needle site `argmax` — only the predicted-gain bookkeeping.
#[inline]
pub fn reference_wavenumber(wavelength: f64, n_inc_re: f64, theta_inc_deg: f64) -> f64 {
    2.0 * std::f64::consts::PI * n_inc_re * theta_inc_deg.to_radians().cos() / wavelength
}

#[inline(always)]
/// Construct a complex value without importing the trait soup at call sites.
pub fn cplx(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Fast algebraic principal complex square root (`re >= 0` branch).
#[inline(always)]
pub fn csqrt_fast(z: Complex64) -> Complex64 {
    let a = z.re;
    let b = z.im;
    if a == 0.0 && b == 0.0 {
        return cplx(0.0, 0.0);
    }
    let m = a.hypot(b); // |z|, correctly-rounded (accurate, no a^2+b^2 intermediate)
    if a >= 0.0 {
        let re = ((m + a) * 0.5).sqrt();
        Complex64::new(re, b / (2.0 * re))
    } else {
        let im = ((m - a) * 0.5).sqrt().copysign(b);
        Complex64::new(b / (2.0 * im), im)
    }
}

/// Fast complex exponential. Same formula as `num_complex`
/// (`e^re · (cos im, sin im)`) but via a single `sin_cos`, which shares the
/// argument reduction between sine and cosine.
#[inline(always)]
pub fn cexp_fast(z: Complex64) -> Complex64 {
    let e = z.re.exp();
    let (s, c) = z.im.sin_cos();
    Complex64::new(e * c, e * s)
}

/// Roughness form factor W(q). No Python/PyO3 overhead so it can be called
/// millions of times from hot loops and inlined by the compiler.
#[inline(always)]
pub fn w_function_inner(q: Complex64, rough_type: i32) -> Complex64 {
    match rough_type {
        0 => Complex64::new(1.0, 0.0),
        1 => {
            let val = q * SQRT3;
            if val.norm() < 1e-9 {
                Complex64::new(1.0, 0.0)
            } else {
                val.sin() / val
            }
        }
        2 => q.cos(),
        3 => {
            let denom = Complex64::new(1.0, 0.0) + (q * q) * 0.5;
            Complex64::new(1.0, 0.0) / denom
        }
        4 => (-(q * q) * 0.5).exp(),
        _ => Complex64::new(1.0, 0.0),
    }
}

/// Complex (field-amplitude) Redheffer star product. Pure Rust, inlined.
#[inline(always)]
pub fn redheffer_product_complex_field_inner(
    r_a_front: Complex64,
    t_a_back: Complex64,
    t_a_fwd: Complex64,
    r_a_back: Complex64,
    r_b_front: Complex64,
    t_b_back: Complex64,
    t_b_fwd: Complex64,
    r_b_back: Complex64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let mut denom = Complex64::new(1.0, 0.0) - r_a_back * r_b_front;
    if denom.abs() < LOG_MIN {
        // Phase-preserving regularization (matches the reference implementation).
        let phase = denom / (denom.abs() + 1e-300);
        denom = Complex64::new(LOG_MIN, 0.0) * phase + 1e-300;
    }
    let inv_denom = denom.recip();

    let s_r_front = r_a_front + t_a_back * r_b_front * t_a_fwd * inv_denom;
    let s_t_back = t_a_back * t_b_back * inv_denom;
    let s_t_fwd = t_b_fwd * t_a_fwd * inv_denom;
    let s_r_back = r_b_back + t_b_fwd * r_a_back * t_b_back * inv_denom;

    (s_r_front, s_t_back, s_t_fwd, s_r_back)
}

/// Real-valued (intensity) Redheffer star product. Pure Rust, inlined.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn redheffer_product_real_inner(
    ra_rf: f64,
    ra_tb: f64,
    ra_tf: f64,
    ra_rb: f64,
    rb_rf: f64,
    rb_tb: f64,
    rb_tf: f64,
    rb_rb: f64,
) -> (f64, f64, f64, f64) {
    let denom = 1.0 - ra_rb * rb_rf;
    let inv_denom = if denom.abs() < DBL_EPS { 0.0 } else { 1.0 / denom };

    let rf = ra_rf + ra_tb * rb_rf * ra_tf * inv_denom;
    let tb = ra_tb * rb_tb * inv_denom;
    let tf = rb_tf * ra_tf * inv_denom;
    let rb = rb_rb + rb_tf * ra_rb * rb_tb * inv_denom;

    (rf, tb, tf, rb)
}

/// Complex (coherency) Redheffer star product over the p-s cross-amplitudes
/// C = (p-field)·conj(s-field). Structurally identical to
/// `redheffer_product_real_inner`; the denominator `1 - C_Ab·C_Bf` is the
/// n=m term of the incoherent multiple-reflection geometric series in the
/// cross channel (different bounce orders are mutually incoherent). On the
/// diagonal (C = |field|²) this collapses exactly onto the real product, which
/// is why R/T are unaffected by the coherency mode.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn redheffer_product_cross_inner(
    a_cf: Complex64,
    a_db: Complex64,
    a_df: Complex64,
    a_cb: Complex64,
    b_cf: Complex64,
    b_db: Complex64,
    b_df: Complex64,
    b_cb: Complex64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let denom = Complex64::new(1.0, 0.0) - a_cb * b_cf;
    let inv = if denom.norm() < DBL_EPS {
        Complex64::new(0.0, 0.0)
    } else {
        denom.recip()
    };

    let cf = a_cf + a_db * b_cf * a_df * inv;
    let db = a_db * b_db * inv;
    let df = b_df * a_df * inv;
    let cb = b_cb + b_df * a_cb * b_db * inv;

    (cf, db, df, cb)
}

/// Second-order non-uniform central-difference derivative dy/dx
/// (numpy.gradient semantics; one-sided at endpoints). Single source of truth
/// for the dispersion post-passes of both the core engine and the needle
/// operator's spectral sensitivity chain.
pub fn grad_nonuniform(y: &[f64], x: &[f64]) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_phase_hand_calc() {
        // λ = 500 nm, n = 1, θ = 0°, D = 100 nm, 1 pass:
        // 2π·1·100·1/500 = 2π/5 = 1.2566370614...
        let p = reference_phase(500.0, 1.0, 0.0, 100.0, 1.0);
        assert!((p - 2.0 * std::f64::consts::PI / 5.0).abs() < 1e-12, "p={p}");
        // Oblique: θ = 60° halves the axial projection.
        let po = reference_phase(500.0, 1.0, 60.0, 100.0, 1.0);
        assert!((po - p / 2.0).abs() < 1e-12, "po={po}");
        // Two passes double; zero thickness kills the reference.
        assert!((reference_phase(500.0, 1.0, 0.0, 100.0, 2.0) - 2.0 * p).abs() < 1e-12);
        assert_eq!(reference_phase(500.0, 1.5, 10.0, 0.0, 1.0), 0.0);
        // Wavenumber × D × passes reconstructs the phase.
        let kz = reference_wavenumber(500.0, 1.5, 10.0);
        assert!((1.0 * kz * 100.0 - reference_phase(500.0, 1.5, 10.0, 100.0, 1.0)).abs() < 1e-15);
    }
}
