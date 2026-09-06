// src/func_01.rs
//! XYZ ↔ xyY conversions (CIE 1931 chromaticity + luminance).
//!
//! Black‑pixel convention: zero‑luminance pixels map to all zeros
//! (x=y=Y=0), matching colour‑science's `XYZ_to_xyY`. The inverse likewise
//! returns all zeros when y < 1e-12.

/// Convert CIE XYZ tristimulus values to xyY chromaticity coordinates.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_01::xyz_to_xyy;
/// let xyz = [[0.95047, 1.00000, 1.08883]]; // D65 white
/// let mut xyY = [[0.0; 3]];
/// xyz_to_xyy(&xyz, &mut xyY);
/// // D65 chromaticity x ≈ 0.31273; the literal below is rounded to 4 dp.
/// assert!((xyY[0][0] - 0.3127).abs() < 1e-4);
/// ```
pub fn xyz_to_xyy(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let sum = xyz[0] + xyz[1] + xyz[2];
        if sum > 1e-12 {
            let inv = 1.0 / sum;
            o[0] = xyz[0] * inv;
            o[1] = xyz[1] * inv;
            o[2] = xyz[1];
        } else {
            // Black pixel: match colour‑science, which returns zeros for
            // zero‑luminance input rather than substituting a chromaticity.
            *o = [0.0, 0.0, 0.0];
        }
    }
}

/// Convert xyY chromaticity coordinates back to CIE XYZ.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_01::xyy_to_xyz;
/// let xyY = [[0.3127, 0.3290, 1.0]];
/// let mut xyz = [[0.0; 3]];
/// xyy_to_xyz(&xyY, &mut xyz);
/// assert!((xyz[0][1] - 1.0).abs() < 1e-12);
/// ```
pub fn xyy_to_xyz(xyy: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyy, o) in xyy.iter().zip(out.iter_mut()) {
        let (x, y, big_y) = (xyy[0], xyy[1], xyy[2]);
        if y > 1e-12 {
            let factor = big_y / y;
            o[0] = x * factor;
            o[1] = big_y;
            o[2] = (1.0 - x - y) * factor;
        } else {
            *o = [0.0, 0.0, 0.0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_pixel_convention() {
        // Black maps to all zeros (parity with colour‑science XYZ_to_xyY).
        let mut out = [[0.0; 3]];
        xyz_to_xyy(&[[0.0, 0.0, 0.0]], &mut out);
        assert_eq!(out[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn round_trip() {
        let xyz_in = [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.95047, 1.0, 1.08883]];
        let mut xyY = [[0.0; 3]; 3];
        let mut xyz_out = [[0.0; 3]; 3];
        xyz_to_xyy(&xyz_in, &mut xyY);
        xyy_to_xyz(&xyY, &mut xyz_out);
        for (i, (a, b)) in xyz_in.iter().zip(xyz_out.iter()).enumerate() {
            for j in 0..3 {
                assert!((a[j] - b[j]).abs() < 1e-12, "mismatch at [{i}][{j}]");
            }
        }
    }
}