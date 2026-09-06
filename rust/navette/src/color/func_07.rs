// src/func_07.rs
//! CIE 1960 UCS (linear UVW tristimulus) and chromaticity conversions.
//!
//! This module provides four kernels:
//! - Linear transform between XYZ and UCS (U, V, W).
//! - XYZ → CIE 1960 chromaticity (u, v).
//! - CIE 1976 (u', v') → CIE 1931 (x, y).
//! - CIE 1960 (u, v) → CIE 1931 (x, y).

use crate::color::common::xyz_to_uv_prime;

/// Convert CIE XYZ to CIE 1960 UCS (U, V, W).
///
/// Formulas:
/// U = (2/3) X
/// V = Y
/// W = 0.5 * (-X + 3Y + Z)
///
/// # Panics
/// None. Input and output slices must have the same length.
pub fn xyz_to_ucs(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
        o[0] = (2.0 / 3.0) * x;
        o[1] = y;
        o[2] = 0.5 * (-x + 3.0 * y + z);
    }
}

/// Convert CIE 1960 UCS (U, V, W) back to CIE XYZ.
///
/// Inverse formulas:
/// X = 1.5 U
/// Y = V
/// Z = 1.5 U - 3 V + 2 W
///
/// # Panics
/// None. Input and output slices must have the same length.
pub fn ucs_to_xyz(ucs: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (ucs, o) in ucs.iter().zip(out.iter_mut()) {
        let (u, v, w) = (ucs[0], ucs[1], ucs[2]);
        o[0] = 1.5 * u;
        o[1] = v;
        o[2] = 1.5 * u - 3.0 * v + 2.0 * w;
    }
}

/// Convert CIE XYZ to CIE 1960 chromaticity coordinates (u, v).
///
/// u = u' (CIE 1976), v = (2/3) v'.
///
/// # Panics
/// None. Input and output slices must have the same length.
pub fn xyz_to_ucs_uv(xyz: &[[f64; 3]], out: &mut [[f64; 2]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let (up, vp) = xyz_to_uv_prime(xyz);
        o[0] = up;
        o[1] = (2.0 / 3.0) * vp;
    }
}

/// Convert CIE 1976 (u', v') chromaticity to CIE 1931 (x, y).
///
/// Denominator: 6u' - 16v' + 12.
/// x = 9u' / denom, y = 4v' / denom.
/// If |denom| < 1e-12, returns (0, 0).
///
/// # Panics
/// None. Input and output slices must have the same length.
pub fn uv1976_to_xy(uv_prime: &[[f64; 2]], out: &mut [[f64; 2]]) {
    for (uv, o) in uv_prime.iter().zip(out.iter_mut()) {
        let (up, vp) = (uv[0], uv[1]);
        let denom = 6.0 * up - 16.0 * vp + 12.0;
        if denom.abs() < 1e-12 {
            *o = [0.0, 0.0];
        } else {
            let inv = 1.0 / denom;
            o[0] = 9.0 * up * inv;
            o[1] = 4.0 * vp * inv;
        }
    }
}

/// Convert CIE 1960 (u, v) chromaticity to CIE 1931 (x, y).
///
/// Denominator: 2u - 8v + 4.
/// x = 3u / denom, y = 2v / denom.
/// If |denom| < 1e-12, returns (0, 0).
///
/// # Panics
/// None. Input and output slices must have the same length.
pub fn uv1960_to_xy(uv: &[[f64; 2]], out: &mut [[f64; 2]]) {
    for (uv, o) in uv.iter().zip(out.iter_mut()) {
        let (u, v) = (uv[0], uv[1]);
        let denom = 2.0 * u - 8.0 * v + 4.0;
        if denom.abs() < 1e-12 {
            *o = [0.0, 0.0];
        } else {
            let inv = 1.0 / denom;
            o[0] = 3.0 * u * inv;
            o[1] = 2.0 * v * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::common::REF_WHITE_D65;

    #[test]
    fn ucs_round_trip() {
        let xyz_in = [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.95047, 1.0, 1.08883]];
        let mut ucs = [[0.0; 3]; 3];
        let mut xyz_out = [[0.0; 3]; 3];
        xyz_to_ucs(&xyz_in, &mut ucs);
        ucs_to_xyz(&ucs, &mut xyz_out);
        for (i, (a, b)) in xyz_in.iter().zip(xyz_out.iter()).enumerate() {
            for j in 0..3 {
                let diff = (a[j] - b[j]).abs();
                assert!(diff < 1e-12, "mismatch at [{i}][{j}]: {diff}");
            }
        }
    }

    #[test]
    fn uv_chromaticity_consistency() {
        // D65 white point
        let xyz_d65 = REF_WHITE_D65;
        let mut uv1960 = [[0.0; 2]];
        xyz_to_ucs_uv(&[xyz_d65], &mut uv1960);
        let (up, vp) = crate::color::common::xyz_to_uv_prime(&xyz_d65);
        // v1960 should equal (2/3) vp
        assert!((uv1960[0][0] - up).abs() < 1e-12);
        assert!((uv1960[0][1] - (2.0 / 3.0) * vp).abs() < 1e-12);
    }

    #[test]
    fn uv1976_to_xy_known() {
        // D65 u',v' -> should give D65 x,y
        let uv_d65 = [[0.1978, 0.4683]];
        let mut xy = [[0.0; 2]];
        uv1976_to_xy(&uv_d65, &mut xy);
        assert!((xy[0][0] - 0.3127).abs() < 1e-4);
        assert!((xy[0][1] - 0.3290).abs() < 1e-4);
    }
}