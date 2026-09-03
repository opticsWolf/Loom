// src/func_06.rs
//! CIE 1964 U*V*W* colour space.
//!
//! This space transforms XYZ (D65‑relative, range [0,1]) into a uniform
//! chromaticity scale. The engine internally scales XYZ to [0,100] as required
//! by the standard. The white‑point anchor `(un, vn)` is the CIE 1960
//! chromaticity of the illuminant: `un = u'`, `vn = (2/3) v'`.

use crate::common::xyz_to_uv_prime;

/// Compute the CIE 1960 white‑point chromaticity `(u, v)` for a given illuminant XYZ.
///
/// `u` = `u'` (CIE 1976), `v` = `(2/3) v'`.
pub fn white_point_uv1960(illuminant: &[f64; 3]) -> (f64, f64) {
    let (up, vp) = xyz_to_uv_prime(illuminant);
    (up, (2.0 / 3.0) * vp)
}

/// Convert CIE XYZ (range [0,1]) to CIE 1964 U*V*W*.
///
/// # Parameters
/// - `xyz`: Input XYZ tristimuli (N×3, values in [0,1]).
/// - `un`, `vn`: CIE 1960 white‑point coordinates (e.g. from `white_point_uv1960`).
/// - `out`: Output buffer, same length as `xyz`.
///
/// # Panics
/// If input and output slice lengths differ.
pub fn xyz_to_uvw(xyz: &[[f64; 3]], un: f64, vn: f64, out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let x = xyz[0] * 100.0;
        let y = xyz[1] * 100.0;
        let z = xyz[2] * 100.0;

        let denom = x + 15.0 * y + 3.0 * z;
        
        // OPTIMIZATION: Branchless masking. Instead of returning tuples from 
        // an if/else block (which blocks vectorization), we conditionally set the 
        // inverse to 0.0. LLVM converts this to a fast `select` instruction.
        let inv_denom = if denom < 1e-12 { 0.0 } else { 1.0 / denom };
        
        let u = 4.0 * x * inv_denom;
        let v = 6.0 * y * inv_denom;

        // OPTIMIZATION: Reverted to .powf. It maps cleanly to LLVM intrinsics 
        // whereas .cbrt() often makes an opaque libm call that stops SIMD.
        let w_star = 25.0 * y.powf(1.0 / 3.0) - 17.0;
        
        o[0] = 13.0 * w_star * (u - un);
        o[1] = 13.0 * w_star * (v - vn);
        o[2] = w_star;
    }
}

/// Convert CIE 1964 U*V*W* back to CIE XYZ (range [0,1]).
///
/// # Parameters
/// - `uvw`: Input U*V*W* values (N×3).
/// - `un`, `vn`: Same white‑point coordinates used in forward transform.
/// - `out`: Output buffer, same length as `uvw`.
///
/// # Panics
/// If input and output slice lengths differ.
pub fn uvw_to_xyz(uvw: &[[f64; 3]], un: f64, vn: f64, out: &mut [[f64; 3]]) {
    for (uvw, o) in uvw.iter().zip(out.iter_mut()) {
        let (u_star, v_star, w_star) = (uvw[0], uvw[1], uvw[2]);

        // OPTIMIZATION: Reverted back to .powi(3). Modern rustc recognizes 
        // this and perfectly unrolls it without breaking vectorization context.
        let y_base = (w_star + 17.0) / 25.0;
        let y100 = y_base.powi(3);
        let y = y100 / 100.0;

        // Branchless masking for U/V fallback
        let inv_13w = if w_star.abs() < 1e-12 { 0.0 } else { 1.0 / (13.0 * w_star) };
        let u = if w_star.abs() < 1e-12 { un } else { u_star * inv_13w + un };
        let v = if w_star.abs() < 1e-12 { vn } else { v_star * inv_13w + vn };

        let denom_uv = 2.0 * u - 8.0 * v + 4.0;
        
        // Branchless masking for Chromaticity
        let inv_den = if denom_uv.abs() < 1e-12 { 0.0 } else { 1.0 / denom_uv };
        let x = 3.0 * u * inv_den;
        let y_chrom = 2.0 * v * inv_den;

        let factor = if y_chrom.abs() < 1e-12 { 0.0 } else { y100 / y_chrom };

        o[0] = (x * factor) / 100.0;
        o[1] = y;
        o[2] = ((1.0 - x - y_chrom) * factor) / 100.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::REF_WHITE_D65;

    #[test]
    fn white_point_uv1960_d65() {
        let (un, vn) = white_point_uv1960(&REF_WHITE_D65);
        // D65 u'=0.1978, v'=0.4683 → v = 2/3 v' ≈ 0.3122
        assert!((un - 0.1978).abs() < 1e-4);
        assert!((vn - 0.3122).abs() < 1e-4);
    }

    #[test]
    fn round_trip() {
        let xyz_in = [
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [0.95047, 1.0, 1.08883],
        ];
        let (un, vn) = white_point_uv1960(&REF_WHITE_D65);
        let mut uvw = [[0.0; 3]; 3];
        let mut xyz_out = [[0.0; 3]; 3];
        xyz_to_uvw(&xyz_in, un, vn, &mut uvw);
        uvw_to_xyz(&uvw, un, vn, &mut xyz_out);
        for (i, (a, b)) in xyz_in.iter().zip(xyz_out.iter()).enumerate() {
            for j in 0..3 {
                let diff = (a[j] - b[j]).abs();
                assert!(diff < 1e-12, "mismatch at [{i}][{j}]: {diff}");
            }
        }
    }
}