// src/func_04.rs
//! Oklab colour space – direct XYZ pipeline.
//!
//! Uses the standard Oklab matrices (M1: XYZ → LMS, M2: LMS^(1/3) → Lab).
//! The forward transform applies a sign‑preserving cube root: `sign(x)·|x|^(1/3)`.
//! The inverse uses the third power. These operations are implemented by
//! `crate::color::common::signed_pow` to guarantee bit‑exact parity with NumPy.

use crate::color::common::{mat3_mul_vec, signed_pow};
use crate::color::matrices::{
    M1_LMS_TO_XYZ_OKLAB, M1_XYZ_TO_LMS_OKLAB, M2_LAB_TO_LMS_OKLAB, M2_LMS_TO_LAB_OKLAB,
};

/// Convert CIE XYZ to Oklab.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_04::xyz_to_oklab;
/// let xyz = [[0.95047, 1.00000, 1.08883]]; // D65 white
/// let mut oklab = [[0.0; 3]];
/// xyz_to_oklab(&xyz, &mut oklab);
/// // White point maps to L ≈ 1.0, a = b ≈ 0. Not bit-exact: the published
/// // Oklab matrices leave D65 at L≈0.9999998, a≈-1e-5 (same rounding as func_05).
/// assert!((oklab[0][0] - 1.0).abs() < 1e-4);
/// assert!(oklab[0][1].abs() < 1e-4);
/// ```
pub fn xyz_to_oklab(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let lms = mat3_mul_vec(&M1_XYZ_TO_LMS_OKLAB, xyz);
        let lms_p = [
            signed_pow(lms[0], 1.0 / 3.0),
            signed_pow(lms[1], 1.0 / 3.0),
            signed_pow(lms[2], 1.0 / 3.0),
        ];
        *o = mat3_mul_vec(&M2_LMS_TO_LAB_OKLAB, &lms_p);
    }
}

/// Convert Oklab back to CIE XYZ.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_04::oklab_to_xyz;
/// let oklab = [[1.0, 0.0, 0.0]];
/// let mut xyz = [[0.0; 3]];
/// oklab_to_xyz(&oklab, &mut xyz);
/// // Recover the D65 white point (to ~1e-7: [1,0,0] is not exactly D65's Oklab
/// // under the published matrices).
/// assert!((xyz[0][0] - 0.95047).abs() < 1e-6);
/// ```
pub fn oklab_to_xyz(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lab, o) in lab.iter().zip(out.iter_mut()) {
        let lms_p = mat3_mul_vec(&M2_LAB_TO_LMS_OKLAB, lab);
        let lms = [
            signed_pow(lms_p[0], 3.0),
            signed_pow(lms_p[1], 3.0),
            signed_pow(lms_p[2], 3.0),
        ];
        *o = mat3_mul_vec(&M1_LMS_TO_XYZ_OKLAB, &lms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let xyz_in = [
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [0.95047, 1.0, 1.08883],
            [-0.1, 0.5, 0.2], // negative values allowed
        ];
        let mut oklab = [[0.0; 3]; 4];
        let mut xyz_out = [[0.0; 3]; 4];
        xyz_to_oklab(&xyz_in, &mut oklab);
        oklab_to_xyz(&oklab, &mut xyz_out);
        for (i, (a, b)) in xyz_in.iter().zip(xyz_out.iter()).enumerate() {
            for j in 0..3 {
                let diff = (a[j] - b[j]).abs();
                assert!(diff < 1e-12, "mismatch at [{i}][{j}]: {diff}");
            }
        }
    }
}