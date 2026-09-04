//! Core constants, matrix/vector utilities, and CIE Lab transfer functions.
//!
//! This module provides the shared building blocks for all colour
//! transformations in the crate. All constants and functions are kept in
//! strict parity with the Python reference implementation
//! (`navette_colorengine.py`).

use std::f64::consts::PI;

// -----------------------------------------------------------------------------
// Angular conversion constants
// -----------------------------------------------------------------------------
pub const DEG2RAD: f64 = PI / 180.0;
pub const RAD2DEG: f64 = 180.0 / PI;

// -----------------------------------------------------------------------------
// CIE 1976 Lab constants (exact rational definitions)
// -----------------------------------------------------------------------------
/// (6/29)³ ≈ 0.008856 – threshold where the Lab f(t) switches from cubic to linear.
pub const LAB_EPSILON: f64 = (6.0 / 29.0) * (6.0 / 29.0) * (6.0 / 29.0);
/// (6/29) – used in the inverse Lab f⁻¹(t) condition.
pub const LAB_DELTA: f64 = 6.0 / 29.0;
/// (116·29²)/(3·6²) ≈ 903.296 – slope of the linear part of f(t).
pub const LAB_KAPPA: f64 = (116.0 * 29.0 * 29.0) / (3.0 * 6.0 * 6.0);

// -----------------------------------------------------------------------------
// Reference white points (Y = 1.0) derived from the exact xy chromaticities
// used by colour‑science 0.4.7 (CIE 1931 2° standard observer).
// -----------------------------------------------------------------------------
/// CIE standard illuminant D65 (daylight, ~6504K).
/// xy = (0.31271667705511086, 0.3290176878026649)
/// XYZ = (x/y, 1, (1−x−y)/y)
pub const REF_WHITE_D65: [f64; 3] = [
    0.9504559270516716, // X = x/y
    1.0,
    1.0890577507598784, // Z = (1−x−y)/y
];
/// CIE standard illuminant D50 (horizon light, ~5003K), used in printing.
/// xy = (0.34570, 0.35850) — the standard locus point, matching
/// `colour-science` (`CCS_ILLUMINANTS`) and the loom reference engine.
/// (A previous revision carried a slightly different locus point
/// (0.34570291…, 0.35853859…); it skewed every D50 Bradford matrix by
/// ~2e-4 and disagreed with the golden vectors, so it was corrected.)
pub const REF_WHITE_D50: [f64; 3] = [
    0.9642956764295677, // X = x/y
    1.0,
    0.8251046025104605, // Z = (1−x−y)/y
];

// -----------------------------------------------------------------------------
// 3×3 matrix utilities (row‑major storage)
// -----------------------------------------------------------------------------

/// Multiply a 3×3 matrix (rows) by a 3‑element column vector.
///
/// # Formula
/// `out[i] = M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2]`
#[inline]
pub fn mat3_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Multiply two 3×3 matrices (row‑major).
///
/// `out = a · b`  (matrix multiplication, not element‑wise).
#[inline]
pub fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            let aik = a[i][k];
            if aik != 0.0 {
                for j in 0..3 {
                    out[i][j] += aik * b[k][j];
                }
            }
        }
    }
    out
}

// -----------------------------------------------------------------------------
// Sign‑preserving power function
// -----------------------------------------------------------------------------

/// Raise a number to a power, preserving the sign of the input.
///
/// Returns `sign(x) * |x|^p`. For `x == 0.0`, returns `0.0` (the sign is 0).
/// This matches the behaviour of `np.sign(x) * np.abs(x)**p` in NumPy.
#[inline]
pub fn signed_pow(x: f64, p: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x.signum() * x.abs().powf(p)
    }
}

// -----------------------------------------------------------------------------
// CIE 1976 Lab forward / inverse transfer functions f(t) and f⁻¹(t)
// -----------------------------------------------------------------------------

/// CIE 1976 non‑linear transfer function `f(t)`.
///
/// Used in the conversion between XYZ and CIELAB.
/// - For `t > ε`  (ε = LAB_EPSILON):   `f(t) = t^(1/3)`
/// - Otherwise:                        `f(t) = (κ·t + 16) / 116`
/// where `κ = LAB_KAPPA`, `ε = LAB_EPSILON`.
///
/// **Note:** The condition uses `t <= ε` to exactly match the behaviour
/// of colour‑science (which uses `np.where(xyz <= eps, …)`). This avoids
/// branch‑mismatch when `t` equals the threshold due to rounding.
#[inline]
pub fn lab_f(t: f64) -> f64 {
    if t <= LAB_EPSILON {
        (LAB_KAPPA * t + 16.0) / 116.0
    } else {
        t.powf(1.0 / 3.0)
    }
}

/// Inverse of the CIE 1976 transfer function `f⁻¹(t)`.
///
/// - For `t > δ` (δ = LAB_DELTA):  `f⁻¹(t) = t³`
/// - Otherwise:                    `f⁻¹(t) = (116·t - 16) / κ`
#[inline]
pub fn lab_f_inv(t: f64) -> f64 {
    if t > LAB_DELTA {
        t * t * t
    } else {
        (116.0 * t - 16.0) / LAB_KAPPA
    }
}

// -----------------------------------------------------------------------------
// CIE 1976 (u′, v′) chromaticity coordinates from XYZ
// -----------------------------------------------------------------------------

/// Compute the CIE 1976 (u′, v′) chromaticity coordinates from CIE XYZ.
///
/// Formulas:
/// ```text
/// u′ = 4X / (X + 15Y + 3Z)
/// v′ = 9Y / (X + 15Y + 3Z)
/// ```
/// If the denominator is zero (black pixel), returns (0,0).
#[inline]
pub fn xyz_to_uv_prime(xyz: &[f64; 3]) -> (f64, f64) {
    let denom = xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2];
    if denom > 1e-12 {
        let inv = 1.0 / denom;
        (4.0 * xyz[0] * inv, 9.0 * xyz[1] * inv)
    } else {
        (0.0, 0.0)
    }
}

// -----------------------------------------------------------------------------
// sRGB transfer functions (IEC 61966-2-1) and [0,1] clamp
// -----------------------------------------------------------------------------

/// sRGB OETF (linear → gamma-encoded).
#[inline(always)]
pub fn gamma_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB EOTF (gamma-encoded → linear).
#[inline(always)]
pub fn inverse_gamma_srgb(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Clamp a value to the closed unit interval `[0, 1]`.
#[inline(always)]
pub fn clip01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

// -----------------------------------------------------------------------------
// Base conversions (the engine's func_01–func_05 core)
// -----------------------------------------------------------------------------

/// sRGB → XYZ. When `clip` is set, encoded RGB is clamped to `[0,1]` first
/// (display-referred).
pub fn srgb_to_xyz(rgb: &[[f64; 3]], clip: bool, out: &mut [[f64; 3]]) {
    use crate::matrices::M_SRGB_TO_XYZ;
    for (rgb, o) in rgb.iter().zip(out.iter_mut()) {
        let lin = [
            inverse_gamma_srgb(if clip { clip01(rgb[0]) } else { rgb[0] }),
            inverse_gamma_srgb(if clip { clip01(rgb[1]) } else { rgb[1] }),
            inverse_gamma_srgb(if clip { clip01(rgb[2]) } else { rgb[2] }),
        ];
        *o = mat3_mul_vec(&M_SRGB_TO_XYZ, &lin);
    }
}

/// XYZ → sRGB. When `clip` is set, linear RGB is clamped to `[0,1]` before
/// gamma encoding (display-referred).
pub fn xyz_to_srgb(xyz: &[[f64; 3]], clip: bool, out: &mut [[f64; 3]]) {
    use crate::matrices::M_XYZ_TO_SRGB;
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let mut lin = mat3_mul_vec(&M_XYZ_TO_SRGB, xyz);
        if clip {
            lin = [clip01(lin[0]), clip01(lin[1]), clip01(lin[2])];
        }
        *o = [gamma_srgb(lin[0]), gamma_srgb(lin[1]), gamma_srgb(lin[2])];
    }
}

/// XYZ → CIELAB under `illuminant`.
pub fn xyz_to_lab(xyz: &[[f64; 3]], illuminant: &[f64; 3], out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let fx = lab_f(xyz[0] / illuminant[0]);
        let fy = lab_f(xyz[1] / illuminant[1]);
        let fz = lab_f(xyz[2] / illuminant[2]);
        o[0] = 116.0 * fy - 16.0;
        o[1] = 500.0 * (fx - fy);
        o[2] = 200.0 * (fy - fz);
    }
}

/// CIELAB → XYZ under `illuminant`.
pub fn lab_to_xyz(lab: &[[f64; 3]], illuminant: &[f64; 3], out: &mut [[f64; 3]]) {
    for (lab, o) in lab.iter().zip(out.iter_mut()) {
        let fy = (lab[0] + 16.0) / 116.0;
        let fx = lab[1] / 500.0 + fy;
        let fz = fy - lab[2] / 200.0;
        o[0] = lab_f_inv(fx) * illuminant[0];
        o[1] = lab_f_inv(fy) * illuminant[1];
        o[2] = lab_f_inv(fz) * illuminant[2];
    }
}

// -----------------------------------------------------------------------------
// Vectorised versions (for batch processing, called by the func_* modules)
// -----------------------------------------------------------------------------

/// Batch version of `lab_f` – applies to each element of a slice.
pub fn lab_f_batch(t: &[f64], out: &mut [f64]) {
    for (ti, o) in t.iter().zip(out.iter_mut()) {
        *o = lab_f(*ti);
    }
}

/// Batch version of `lab_f_inv` – applies to each element of a slice.
pub fn lab_f_inv_batch(t: &[f64], out: &mut [f64]) {
    for (ti, o) in t.iter().zip(out.iter_mut()) {
        *o = lab_f_inv(*ti);
    }
}

/// Batch version of `xyz_to_uv_prime` – writes to a slice of (u′, v′) pairs.
pub fn xyz_to_uv_prime_batch(xyz: &[[f64; 3]], out: &mut [(f64, f64)]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        *o = xyz_to_uv_prime(xyz);
    }
}

// -----------------------------------------------------------------------------
// Tests (consistency with the Python reference)
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_f_constants_match_python() {
        // LAB_EPSILON is exactly (6/29)^3; the rounded value is ~0.008856.
        let eps_exact = (6.0_f64 / 29.0).powi(3);
        assert!((LAB_EPSILON - eps_exact).abs() < 1e-12);
        assert!((LAB_DELTA - 0.20689655172413793).abs() < 1e-12);
        assert!((LAB_KAPPA - 903.2962962962963).abs() < 1e-12);
    }

    #[test]
    fn signed_pow_parity() {
        assert_eq!(signed_pow(0.0, 3.0), 0.0);
        assert_eq!(signed_pow(8.0, 1.0 / 3.0), 2.0);
        assert_eq!(signed_pow(-8.0, 1.0 / 3.0), -2.0);
        assert_eq!(signed_pow(-0.5, 2.0), -0.25);
    }

    #[test]
    fn lab_f_parity() {
        // At the epsilon threshold, both branches should be continuous.
        let t = LAB_EPSILON;
        let cubic = t.powf(1.0 / 3.0);
        let linear = (LAB_KAPPA * t + 16.0) / 116.0;
        assert!((cubic - linear).abs() < 1e-12);

        // Check a few known values (relative to Python output)
        assert!((lab_f(0.0) - (16.0 / 116.0)).abs() < 1e-12);
        assert!((lab_f(1.0) - 1.0).abs() < 1e-12);
        assert!((lab_f(0.5) - 0.5f64.powf(1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn lab_f_inv_parity() {
        // Inverse of f(t) should recover t for positive inputs.
        let test_vals = [0.0, 0.1, LAB_EPSILON, 0.5, 1.0];
        for &t in &test_vals {
            let ft = lab_f(t);
            let t_back = lab_f_inv(ft);
            assert!((t_back - t).abs() < 1e-12, "t={} failed", t);
        }
    }

    #[test]
    fn xyz_to_uv_prime_parity() {
        let white = REF_WHITE_D65;
        let (up, vp) = xyz_to_uv_prime(&white);
        // D65 u' ≈ 0.1978, v' ≈ 0.4683
        assert!((up - 0.1978).abs() < 1e-4);
        assert!((vp - 0.4683).abs() < 1e-4);

        let black = [0.0, 0.0, 0.0];
        let (up_b, vp_b) = xyz_to_uv_prime(&black);
        assert_eq!(up_b, 0.0);
        assert_eq!(vp_b, 0.0);
    }

    #[test]
    fn mat3_mul_vec_works() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let v = [1.0, 2.0, 3.0];
        let res = mat3_mul_vec(&m, &v);
        assert_eq!(res, [1.0*1.+2.*2.+3.*3., 4.*1.+5.*2.+6.*3., 7.*1.+8.*2.+9.*3.]);
    }

    #[test]
    fn mat3_mul_works() {
        let a = [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]];
        let prod = mat3_mul(&a, &b);
        assert_eq!(prod[0], [4.0, 10.0, 18.0]);
        assert_eq!(prod[1], [0.0, 5.0, 0.0]);
        assert_eq!(prod[2], [0.0, 0.0, 6.0]);
    }
}