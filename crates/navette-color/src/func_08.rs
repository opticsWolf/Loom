// src/func_08.rs
//! Chromatic adaptation using the Bradford transform (CIE 1994).
//!
//! The Bradford method transforms XYZ tristimulus values from a source
//! illuminant to a destination illuminant using a von Kries‑type gain in
//! a “sharpened” cone response space (LMS).
//!
//! This implementation matches the Python reference exactly:
//! - Gains are clamped: `|src_lms[i]| < 1e-12 → 1e-12`
//! - Identity short‑circuit if `allclose(src_white, dst_white)` (rtol=1e-5, atol=1e-8)
//! - Optional negative clamping (default true): values `< -1e-6` are set to `0.0`

use crate::common::{mat3_mul, mat3_mul_vec};
use crate::matrices::{M_BRADFORD, M_BRADFORD_INV};

/// Check if two white points are nearly equal (NumPy `allclose` defaults).
#[inline]
fn allclose3(a: &[f64; 3], b: &[f64; 3]) -> bool {
    let rtol = 1e-5;
    let atol = 1e-8;
    for i in 0..3 {
        let tol = atol + rtol * b[i].abs();
        if (a[i] - b[i]).abs() > tol {
            return false;
        }
    }
    true
}

/// Compute the von Kries gains in LMS space for the given white points.
///
/// Gains are `dst_lms[i] / src_lms[i]` with a floor of `1e-12` on each
/// `src_lms[i]` to avoid division by zero.
fn bradford_gains(src_white: &[f64; 3], dst_white: &[f64; 3]) -> [f64; 3] {
    let mut src_lms = mat3_mul_vec(&M_BRADFORD, src_white);
    let dst_lms = mat3_mul_vec(&M_BRADFORD, dst_white);
    for c in &mut src_lms {
        if c.abs() < 1e-12 {
            *c = 1e-12;
        }
    }
    [
        dst_lms[0] / src_lms[0],
        dst_lms[1] / src_lms[1],
        dst_lms[2] / src_lms[2],
    ]
}

/// Multiply a row vector by a 3×3 matrix: `result = v · M`.
///
/// This is the row‑vector convention. It is the correct way to apply a matrix
/// produced by [`calc_transform_matrix`], which is stored in row‑vector form.
///
/// **Do not** apply such a matrix with [`mat3_mul_vec`]: that helper computes
/// `M · v` (column‑vector convention). Because the Bradford composite is not
/// symmetric, the two disagree by a large, *silent* margin (≈0.07 on a white
/// point) — no panic, just wrong colours. This helper exists so there is a
/// single, clearly‑named code path for applying row‑vector matrices.
#[inline]
/// Multiply a row vector by a 3×3 matrix: `out[j] = Σᵢ v[i]·m[i][j]`.
///
/// This is the convention used throughout the engine (row-major batches,
/// row-vector maths); [`crate::common::mat3_mul_vec`] is the column-vector
/// counterpart. Used to apply adaptation matrices from
/// [`calc_transform_matrix`].
pub fn vec_mul_mat3(v: &[f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    ]
}

/// Calculate the 3×3 Bradford adaptation matrix for row‑vector multiplication.
///
/// The composite matrix is: `M_adapt = M_BRADFORDᵀ · diag(gains) · M_BRADFORD_INVᵀ`
/// which is equivalent to `(M_BRADFORD_INV · diag(gains) · M_BRADFORD)ᵀ` for row vectors.
///
/// # Applying the result
/// The returned matrix is in **row‑vector** form. Apply it with
/// [`vec_mul_mat3`] (`v · M`) or use [`adapt`], which performs the equivalent
/// two‑step transform directly. Applying it with [`mat3_mul_vec`] (`M · v`) is
/// a convention mismatch and yields silently incorrect results.
pub fn calc_transform_matrix(src_white: &[f64; 3], dst_white: &[f64; 3]) -> [[f64; 3]; 3] {
    let gains = bradford_gains(src_white, dst_white);
    let gain_mat = [
        [gains[0], 0.0, 0.0],
        [0.0, gains[1], 0.0],
        [0.0, 0.0, gains[2]],
    ];
    // Compose: M = (M_BRADFORD_INV · gain_mat · M_BRADFORD)ᵀ
    let tmp = mat3_mul(&M_BRADFORD_INV, &gain_mat);
    let m = mat3_mul(&tmp, &M_BRADFORD);
    // Transpose because we store row-major but the composition is for column vectors.
    // For row-vector multiplication, the effective matrix is the transpose.
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Adapt XYZ colours from a source illuminant to a destination illuminant.
///
/// # Parameters
/// - `xyz`: Input XYZ values (N×3) in the source white point.
/// - `src_white`: Source white point XYZ.
/// - `dst_white`: Destination white point XYZ.
/// - `clip_negative`: If true (default), clamp any resulting component `< -1e-6` to `0.0`.
/// - `out`: Output buffer, same length as `xyz`.
///
/// # Panics
/// If input and output slices have different lengths.
///
/// # Short‑circuit
/// If `src_white` and `dst_white` are close (rtol=1e-5, atol=1e-8), the function
/// copies the input to output and only applies negative clipping if requested.
pub fn adapt(
    xyz: &[[f64; 3]],
    src_white: &[f64; 3],
    dst_white: &[f64; 3],
    clip_negative: bool,
    out: &mut [[f64; 3]],
) {
    assert_eq!(xyz.len(), out.len(), "input and output length mismatch");

    let clip = |v: f64| if v < -1e-6 { 0.0 } else { v };

    // Identity short‑circuit
    if allclose3(src_white, dst_white) {
        for (x, o) in xyz.iter().zip(out.iter_mut()) {
            if clip_negative {
                *o = [clip(x[0]), clip(x[1]), clip(x[2])];
            } else {
                *o = *x;
            }
        }
        return;
    }

    let gains = bradford_gains(src_white, dst_white);

    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        // Convert XYZ → LMS (row vector multiplication with M_BRADFORD)
        let mut lms = mat3_mul_vec(&M_BRADFORD, xyz);
        // Apply gains
        for k in 0..3 {
            lms[k] *= gains[k];
        }
        // Convert back LMS → XYZ
        let res = mat3_mul_vec(&M_BRADFORD_INV, &lms);

        if clip_negative {
            *o = [clip(res[0]), clip(res[1]), clip(res[2])];
        } else {
            *o = res;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{REF_WHITE_D50, REF_WHITE_D65};

    #[test]
    fn identity_short_circuit() {
        let xyz = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]];
        let mut out = [[0.0; 3]; 2];
        adapt(&xyz, &REF_WHITE_D65, &REF_WHITE_D65, true, &mut out);
        assert_eq!(out, xyz);
    }

    #[test]
    fn negative_clipping() {
        let xyz = [[-0.1, 0.5, 0.5]];
        let mut out = [[0.0; 3]];
        adapt(&xyz, &REF_WHITE_D65, &REF_WHITE_D50, true, &mut out);
        assert_eq!(out[0][0], 0.0);
        assert!(out[0][1] > 0.0);
        assert!(out[0][2] > 0.0);
    }

    #[test]
    fn d65_to_d50_consistency() {
        // Known D65 white point adapted to D50 should give D50 white point.
        let xyz = [REF_WHITE_D65];
        let mut adapted = [[0.0; 3]];
        adapt(&xyz, &REF_WHITE_D65, &REF_WHITE_D50, true, &mut adapted);
        // The adapted white point should be very close to REF_WHITE_D50.
        assert!((adapted[0][0] - REF_WHITE_D50[0]).abs() < 1e-6);
        assert!((adapted[0][1] - REF_WHITE_D50[1]).abs() < 1e-6);
        assert!((adapted[0][2] - REF_WHITE_D50[2]).abs() < 1e-6);
    }

    #[test]
    fn calc_matrix_round_trip() {
        let m = calc_transform_matrix(&REF_WHITE_D65, &REF_WHITE_D50);
        // The matrix is in row‑vector form, so it must be applied as `v · M`
        // via `vec_mul_mat3` — NOT `mat3_mul_vec` (`M · v`), which would apply
        // the transpose and miss D50 by ~0.07. With the correct convention the
        // round trip reproduces D50 to ~2e-16, so the strict 1e-12 bound holds.
        let adapted_white = vec_mul_mat3(&REF_WHITE_D65, &m);
        assert!((adapted_white[0] - REF_WHITE_D50[0]).abs() < 1e-12);
        assert!((adapted_white[1] - REF_WHITE_D50[1]).abs() < 1e-12);
        assert!((adapted_white[2] - REF_WHITE_D50[2]).abs() < 1e-12);
    }
}