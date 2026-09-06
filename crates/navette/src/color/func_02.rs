// src/func_02.rs
//! CIELAB ↔ CIELCh conversions (cylindrical representation).
//!
//! Chroma is Euclidean distance `hypot(a, b)`. Hue is computed in degrees
//! with `atan2(b, a)` and wrapped to the range `[0, 360)`.

use crate::color::common::{DEG2RAD, RAD2DEG};

/// Convert CIELAB to cylindrical CIELCh (Lightness, Chroma, hue angle in degrees).
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_02::lab_to_lch;
/// let lab = [[50.0, 10.0, 5.0]];
/// let mut lch = [[0.0; 3]];
/// lab_to_lch(&lab, &mut lch);
/// assert!((lch[0][0] - 50.0).abs() < 1e-12);
/// assert!(lch[0][1] > 0.0);
/// assert!(lch[0][2] >= 0.0 && lch[0][2] < 360.0);
/// ```
pub fn lab_to_lch(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lab, o) in lab.iter().zip(out.iter_mut()) {
        let (l, a, b) = (lab[0], lab[1], lab[2]);
        let c = a.hypot(b);
        let mut h = b.atan2(a) * RAD2DEG;
        if h < 0.0 {
            h += 360.0;
        }
        *o = [l, c, h];
    }
}

/// Convert cylindrical CIELCh back to CIELAB.
///
/// # Panics
/// None. Input and output slices must have the same length.
///
/// # Examples
/// ```
/// use navette::color::func_02::lch_to_lab;
/// let lch = [[50.0, 10.0, 30.0]];
/// let mut lab = [[0.0; 3]];
/// lch_to_lab(&lch, &mut lab);
/// assert!((lab[0][0] - 50.0).abs() < 1e-12);
/// ```
pub fn lch_to_lab(lch: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lch, o) in lch.iter().zip(out.iter_mut()) {
        let (l, c, h_deg) = (lch[0], lch[1], lch[2]);
        let h = h_deg * DEG2RAD;
        *o = [l, c * h.cos(), c * h.sin()];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let lab_in = [[50.0, 10.0, 5.0], [0.0, 0.0, 0.0], [100.0, -20.0, 30.0]];
        let mut lch = [[0.0; 3]; 3];
        let mut lab_out = [[0.0; 3]; 3];
        lab_to_lch(&lab_in, &mut lch);
        lch_to_lab(&lch, &mut lab_out);
        for (i, (a, b)) in lab_in.iter().zip(lab_out.iter()).enumerate() {
            for j in 0..3 {
                assert!((a[j] - b[j]).abs() < 1e-12, "mismatch at [{i}][{j}]");
            }
        }
    }

    #[test]
    fn hue_wrapping() {
        let lab = [[50.0, 0.0, -1.0]]; // atan2(-1,0) = -90°
        let mut lch = [[0.0; 3]];
        lab_to_lch(&lab, &mut lch);
        assert!((lch[0][2] - 270.0).abs() < 1e-12);
    }
}