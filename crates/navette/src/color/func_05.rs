// src/func_05.rs
//! Oklab colour space – legacy sRGB pipeline.
//!
//! This module provides conversions between sRGB and Oklab using the original
//! matrices from Ottosson's blog post, which bake the sRGB linearisation into
//! the first matrix.  The output is display‑referred (clipped to [0,1]).
//!
//! **Note:** For a more general pipeline that starts from CIE XYZ, use
//! `func_04::xyz_to_oklab` / `oklab_to_xyz`.

use crate::color::common::{clip01, gamma_srgb, inverse_gamma_srgb, mat3_mul_vec, signed_pow};
use crate::color::matrices::{
    M1_OKLAB_SRGB, M1_OKLAB_SRGB_INV, M2_OKLAB_SRGB, M2_OKLAB_SRGB_INV,
};

/// Convert sRGB (gamma‑encoded, [0,1]) to Oklab.
///
/// The input is clamped to [0,1] before linearisation.  The result is the
/// Oklab representation (L in [0,1] for in‑gamut colours, a and b typically
/// in roughly [-0.4, 0.4] for sRGB colours).
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_05::srgb_to_oklab;
/// let srgb = [[0.5, 0.5, 0.5]];   // neutral grey
/// let mut oklab = [[0.0; 3]];
/// srgb_to_oklab(&srgb, &mut oklab);
/// // For neutral grey, a and b are ~0. They are not bit-exact zero: the
/// // published Oklab M2 b-row sums to +3.73e-8, so neutrals carry b ≈ 3.73e-8·L.
/// assert!((oklab[0][1]).abs() < 1e-7);
/// assert!((oklab[0][2]).abs() < 1e-7);
/// ```
pub fn srgb_to_oklab(rgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (rgb, o) in rgb.iter().zip(out.iter_mut()) {
        // 1. Clamp and linearise each channel
        let lin = [
            inverse_gamma_srgb(clip01(rgb[0])),
            inverse_gamma_srgb(clip01(rgb[1])),
            inverse_gamma_srgb(clip01(rgb[2])),
        ];
        // 2. Convert linear RGB to LMS (M1)
        let lms = mat3_mul_vec(&M1_OKLAB_SRGB, &lin);
        // 3. Sign‑preserving cube root
        let lms_cube = [
            signed_pow(lms[0], 1.0 / 3.0),
            signed_pow(lms[1], 1.0 / 3.0),
            signed_pow(lms[2], 1.0 / 3.0),
        ];
        // 4. Convert LMS^(1/3) to Oklab (M2)
        *o = mat3_mul_vec(&M2_OKLAB_SRGB, &lms_cube);
    }
}

/// Convert Oklab back to sRGB (gamma‑encoded, [0,1]).
///
/// The inverse transform clamps the intermediate linear RGB to [0,1] before
/// gamma encoding, making the result display‑referred.  Out‑of‑gamut Oklab
/// colours will be clipped.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_05::oklab_to_srgb;
/// let oklab = [[0.5, 0.0, 0.0]];   // neutral grey
/// let mut srgb = [[0.0; 3]];
/// oklab_to_srgb(&oklab, &mut srgb);
/// // For neutral grey, srgb is equal in all channels up to matrix-rounding
/// // (~1e-7); the published Oklab matrices are not perfectly normalised.
/// assert!((srgb[0][0] - srgb[0][1]).abs() < 1e-7);
/// assert!((srgb[0][1] - srgb[0][2]).abs() < 1e-7);
/// ```
pub fn oklab_to_srgb(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lab, o) in lab.iter().zip(out.iter_mut()) {
        // 1. Convert Oklab → LMS^(1/3) (inverse M2)
        let lms_p = mat3_mul_vec(&M2_OKLAB_SRGB_INV, lab);
        // 2. Cube (sign‑preserving)
        let lms_lin = [
            signed_pow(lms_p[0], 3.0),
            signed_pow(lms_p[1], 3.0),
            signed_pow(lms_p[2], 3.0),
        ];
        // 3. Convert LMS → linear RGB (inverse M1)
        let lin = mat3_mul_vec(&M1_OKLAB_SRGB_INV, &lms_lin);
        // 4. Clip to [0,1] and gamma‑encode
        *o = [
            gamma_srgb(clip01(lin[0])),
            gamma_srgb(clip01(lin[1])),
            gamma_srgb(clip01(lin[2])),
        ];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_in_gamut() {
        // Test a set of in‑gamut sRGB colours
        let rgb_in = [
            [0.0, 0.0, 0.0],     // black
            [1.0, 1.0, 1.0],     // white
            [0.5, 0.5, 0.5],     // grey
            [0.2, 0.6, 0.9],     // some blueish colour
            [0.9, 0.1, 0.3],     // reddish
        ];
        let mut oklab = [[0.0; 3]; 5];
        let mut rgb_out = [[0.0; 3]; 5];
        srgb_to_oklab(&rgb_in, &mut oklab);
        oklab_to_srgb(&oklab, &mut rgb_out);
        for (i, (a, b)) in rgb_in.iter().zip(rgb_out.iter()).enumerate() {
            for j in 0..3 {
                let diff = (a[j] - b[j]).abs();
                assert!(diff < 1e-12, "mismatch at [{i}][{j}]: {diff}");
            }
        }
    }

    #[test]
    fn neutral_grey_property() {
        // Neutral greys are achromatic only up to the rounding baked into the
        // published Oklab matrices: M2's b-row sums to +3.73e-8 (not 0), so a
        // perfect grey produces b ≈ 3.73e-8·L, and M1's sRGB row sums leave a
        // residual a ≈ 6e-11. Both are well under 1e-7; a 1e-12 bound is not
        // achievable with these coefficients and would be a wrong test, not a
        // real defect. (If exact-zero neutrals are ever required, the matrices
        // must be renormalised — which then diverges from the published values.)
        let mut oklab = [[0.0; 3]];
        srgb_to_oklab(&[[0.4, 0.4, 0.4]], &mut oklab);
        assert!((oklab[0][1]).abs() < 1e-7);
        assert!((oklab[0][2]).abs() < 1e-7);

        srgb_to_oklab(&[[0.7, 0.7, 0.7]], &mut oklab);
        assert!((oklab[0][1]).abs() < 1e-7);
        assert!((oklab[0][2]).abs() < 1e-7);
    }

    #[test]
    fn clip_behaviour() {
        // Out‑of‑range input should be clamped before conversion
        let rgb_overshoot = [[1.5, -0.2, 0.8]];
        let mut oklab1 = [[0.0; 3]];
        let mut oklab2 = [[0.0; 3]];
        srgb_to_oklab(&rgb_overshoot, &mut oklab1);
        srgb_to_oklab(&[[1.0, 0.0, 0.8]], &mut oklab2);
        assert_eq!(oklab1, oklab2);
    }
}