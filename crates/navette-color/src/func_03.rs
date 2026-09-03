// src/func_03.rs
//! XYZ ↔ CIELUV conversions (CIE 1976 L*u*v*).
//!
//! This color space is useful for emitted light and white‑point estimation.
//! The forward transform uses the same `f(t)` function as CIELAB; the inverse
//! contains two independent guards: one to recover (u',v') when L > 1e-12,
//! and another to recover X and Z when both L > 1e-12 and v' > 1e-12.

use crate::common::{lab_f, lab_f_inv, xyz_to_uv_prime, REF_WHITE_D65};

/// Convert CIE XYZ to CIELUV under a given illuminant.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette_color::func_03::xyz_to_luv;
/// use navette_color::common::REF_WHITE_D65;
/// // The reference white maps to L* = 100, u* = v* = 0 exactly. We feed the
/// // crate's own D65 constant (which tracks colour-science, not loom's rounded
/// // [0.95047, 1, 1.08883]) so input and illuminant are identical.
/// let xyz = [REF_WHITE_D65];
/// let mut luv = [[0.0; 3]];
/// xyz_to_luv(&xyz, &REF_WHITE_D65, &mut luv);
/// assert!((luv[0][0] - 100.0).abs() < 1e-12);
/// assert!(luv[0][1].abs() < 1e-12);
/// assert!(luv[0][2].abs() < 1e-12);
/// ```
pub fn xyz_to_luv(xyz: &[[f64; 3]], illuminant: &[f64; 3], out: &mut [[f64; 3]]) {
    let (un, vn) = xyz_to_uv_prime(illuminant);
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let (up, vp) = xyz_to_uv_prime(xyz);
        let l = 116.0 * lab_f(xyz[1] / illuminant[1]) - 16.0;
        o[0] = l;
        o[1] = 13.0 * l * (up - un);
        o[2] = 13.0 * l * (vp - vn);
    }
}

/// Convert CIELUV back to CIE XYZ under a given illuminant.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette_color::func_03::luv_to_xyz;
/// use navette_color::common::REF_WHITE_D65;
/// let luv = [[100.0, 0.0, 0.0]];
/// let mut xyz = [[0.0; 3]];
/// luv_to_xyz(&luv, &REF_WHITE_D65, &mut xyz);
/// // Recovers the crate's D65 white point (assert against the constant, not a
/// // hardcoded literal — navette's D65 tracks colour-science).
/// assert!((xyz[0][0] - REF_WHITE_D65[0]).abs() < 1e-12);
/// ```
pub fn luv_to_xyz(luv: &[[f64; 3]], illuminant: &[f64; 3], out: &mut [[f64; 3]]) {
    let (un, vn) = xyz_to_uv_prime(illuminant);
    for (luv, o) in luv.iter().zip(out.iter_mut()) {
        let (l, u, v) = (luv[0], luv[1], luv[2]);

        // Recover (u', v') from L*, u*, v*.  If L ≈ 0, fall back to the white point.
        let (up, vp) = if l > 1e-12 {
            let inv = 1.0 / (13.0 * l);
            (u * inv + un, v * inv + vn)
        } else {
            (un, vn)
        };

        let fy = (l + 16.0) / 116.0;
        let big_y = lab_f_inv(fy) * illuminant[1];

        // Compute X and Z only when both Y > 0 and v' is non‑zero.
        let (mut x, mut z) = (0.0, 0.0);
        if vp > 1e-12 && l > 1e-12 {
            let inv4v = 1.0 / (4.0 * vp);
            x = big_y * 9.0 * up * inv4v;
            z = big_y * (12.0 - 3.0 * up - 20.0 * vp) * inv4v;
        }

        *o = [x, big_y, z];
    }
}

/// Convenience wrapper that uses D65 as the reference white point.
pub fn xyz_to_luv_d65(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    xyz_to_luv(xyz, &REF_WHITE_D65, out);
}

/// Convenience wrapper that uses D65 as the reference white point.
pub fn luv_to_xyz_d65(luv: &[[f64; 3]], out: &mut [[f64; 3]]) {
    luv_to_xyz(luv, &REF_WHITE_D65, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::REF_WHITE_D65;

    #[test]
    fn black_pixel() {
        let mut out = [[0.0; 3]];
        xyz_to_luv(&[[0.0, 0.0, 0.0]], &REF_WHITE_D65, &mut out);
        // L = 0, u* = v* = 0 (the fallback produces (0,0) because L=0)
        assert_eq!(out[0][0], 0.0);
        assert_eq!(out[0][1], 0.0);
        assert_eq!(out[0][2], 0.0);
    }

    #[test]
    fn round_trip() {
        let xyz_in = [
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [0.95047, 1.0, 1.08883],
        ];
        let mut luv = [[0.0; 3]; 3];
        let mut xyz_out = [[0.0; 3]; 3];
        xyz_to_luv(&xyz_in, &REF_WHITE_D65, &mut luv);
        luv_to_xyz(&luv, &REF_WHITE_D65, &mut xyz_out);
        for (i, (a, b)) in xyz_in.iter().zip(xyz_out.iter()).enumerate() {
            for j in 0..3 {
                let diff = (a[j] - b[j]).abs();
                assert!(diff < 1e-12, "mismatch at [{i}][{j}]: {diff}");
            }
        }
    }
}