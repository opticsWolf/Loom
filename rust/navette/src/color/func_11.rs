// src/func_11.rs
//! Delta E CMC(l:c) – CMC colour difference formula (1984).
//!
//! This metric is **asymmetric**: the first argument (`lab1`) is the reference
//! (standard) and its lightness, chroma and hue determine the weighting factors.
//! The parameters `pl` (lightness weight) and `pc` (chroma weight) are typically
//! set to `(2.0, 1.0)` for acceptability (default) or `(1.0, 1.0)` for
//! imperceptibility.
//!
//! The formula is:
//! ```text
//! ΔE_CMC = √[(ΔL* / (pl·S_L))² + (ΔC* / (pc·S_C))² + (ΔH* / S_H)²]
//! ```
//! with the reference‑dependent weights `S_L`, `S_C`, `S_H` defined below.

use crate::color::common::DEG2RAD;

/// Single‑pixel CMC colour difference.
///
/// # Parameters
/// - `lab1`: Reference colour (used for weights).
/// - `lab2`: Sample colour.
/// - `pl`: Lightness parametric factor (default 2.0 for acceptability).
/// - `pc`: Chroma parametric factor (default 1.0 for acceptability).
#[inline(always)]
pub fn delta_e_cmc_single(lab1: &[f64; 3], lab2: &[f64; 3], pl: f64, pc: f64) -> f64 {
    let dl = lab1[0] - lab2[0];
    let a1 = lab1[1];
    let b1 = lab1[2];
    let a2 = lab2[1];
    let b2 = lab2[2];

    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let dc = c1 - c2;

    let da = a1 - a2;
    let db = b1 - b2;
    // dH² = da² + db² - dC², clamp to avoid negative due to FP error
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);

    // Hue angle of reference in degrees, wrapped to [0,360)
    let h1 = b1.atan2(a1).to_degrees().rem_euclid(360.0);

    // Lightness weight S_L
    let sl = if lab1[0] < 16.0 {
        0.511
    } else {
        (0.040975 * lab1[0]) / (1.0 + 0.01765 * lab1[0])
    };

    // Chroma weight S_C
    let sc = 0.0638 * c1 / (1.0 + 0.0131 * c1) + 0.638;

    // T factor: depends on hue angle
    let t = if (164.0..=345.0).contains(&h1) {
        0.56 + (0.2 * ((h1 + 168.0) * DEG2RAD).cos()).abs()
    } else {
        0.36 + (0.4 * ((h1 + 35.0) * DEG2RAD).cos()).abs()
    };

    // F factor
    let c1_4 = c1.powi(4);
    let f = (c1_4 / (c1_4 + 1900.0)).sqrt();

    // Hue weight S_H
    let sh = sc * (f * t + 1.0 - f);

    let term_l = dl / (pl * sl);
    let term_c = dc / (pc * sc);
    let term_h = dh_sq.sqrt() / sh;

    (term_l * term_l + term_c * term_c + term_h * term_h).sqrt()
}

/// Compute the CMC colour difference between two batches of CIELAB colours.
///
/// # Broadcasting
/// The function supports NumPy‑style broadcasting: if one batch has length 1
/// and the other has length N, the single value is paired with every element
/// of the other batch. Both batches must have length 1 or equal length,
/// otherwise the function panics.
///
/// # Asymmetry
/// The first argument `lab1` is the **reference** batch; the second `lab2` is
/// the **sample** batch. Swapping arguments changes the result.
///
/// # Parameters
/// - `pl`: Lightness parametric factor (default 2.0 for acceptability).
/// - `pc`: Chroma parametric factor (default 1.0 for acceptability).
///
/// # Panics
/// Panics if the two slices have different lengths and neither length is 1.
///
/// # Examples
/// ```
/// use navette::color::func_11::delta_e_cmc;
/// let refs = [[50.0, 10.0, 5.0]];
/// let samples = [[52.0, 11.0, 6.0]];
/// let de = delta_e_cmc(&refs, &samples, 2.0, 1.0);
/// ```
pub fn delta_e_cmc(lab1: &[[f64; 3]], lab2: &[[f64; 3]], pl: f64, pc: f64) -> Vec<f64> {
    crate::color::metrics::map_pairs(lab1, lab2, |a, b| delta_e_cmc_single(a, b, pl, pc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_pure_lightness() {
        // Only lightness differs: all weights affect only dL term.
        let refs = [50.0, 0.0, 0.0];
        let samp = [55.0, 0.0, 0.0];
        // For L < 16 case not triggered.
        let sl = (0.040975 * 50.0) / (1.0 + 0.01765 * 50.0);
        let expected = 5.0 / (2.0 * sl);
        let de = delta_e_cmc_single(&refs, &samp, 2.0, 1.0);
        assert!((de - expected).abs() < 1e-12);
    }

    #[test]
    fn test_t_factor_branch_boundaries() {
        // Edge cases at 164° and 345°
        let mut lab = [50.0, 0.0, 0.0];
        // h=164° → branch 1 (164..=345 inclusive)
        lab[1] = 1.0; // a>0
        lab[2] = (164.0_f64).to_radians().tan() * 1.0; // b = tan(164°)*a → approx negative
        // Actually easier: set a,b such that atan2 gives exactly 164°
        let h = 164.0;
        let rad = h * DEG2RAD;
        let a = rad.cos();
        let b = rad.sin();
        lab[1] = a;
        lab[2] = b;
        let de1 = delta_e_cmc_single(&lab, &lab, 2.0, 1.0);
        assert!(de1.abs() < 1e-12); // zero difference

        // h=345° still in first branch
        let h2 = 345.0;
        let rad2 = h2 * DEG2RAD;
        lab[1] = rad2.cos();
        lab[2] = rad2.sin();
        let de2 = delta_e_cmc_single(&lab, &lab, 2.0, 1.0);
        assert!(de2.abs() < 1e-12);
    }

    #[test]
    fn test_sl_branch() {
        let refs_low = [15.0, 10.0, 5.0];
        let refs_high = [16.0, 10.0, 5.0];
        let samp = [15.5, 10.0, 5.0];
        let de_low = delta_e_cmc_single(&refs_low, &samp, 2.0, 1.0);
        let de_high = delta_e_cmc_single(&refs_high, &samp, 2.0, 1.0);
        // Different S_L -> different result
        assert!((de_low - de_high).abs() > 1e-6);
    }

    #[test]
    fn test_broadcast() {
        let refs = [[50.0, 10.0, 5.0]];
        let samples = [[51.0, 11.0, 6.0], [52.0, 12.0, 7.0]];
        let res = delta_e_cmc(&refs, &samples, 2.0, 1.0);
        assert_eq!(res.len(), 2);
        let expected0 = delta_e_cmc_single(&refs[0], &samples[0], 2.0, 1.0);
        let expected1 = delta_e_cmc_single(&refs[0], &samples[1], 2.0, 1.0);
        assert!((res[0] - expected0).abs() < 1e-12);
        assert!((res[1] - expected1).abs() < 1e-12);
    }

    #[test]
    fn test_equal_length() {
        let a = [[50.0, 10.0, 20.0], [60.0, 0.0, 0.0]];
        let b = [[55.0, 11.0, 22.0], [65.0, 0.0, 0.0]];
        let res = delta_e_cmc(&a, &b, 2.0, 1.0);
        assert_eq!(res.len(), 2);
        let expected0 = delta_e_cmc_single(&a[0], &b[0], 2.0, 1.0);
        let expected1 = delta_e_cmc_single(&a[1], &b[1], 2.0, 1.0);
        assert!((res[0] - expected0).abs() < 1e-12);
        assert!((res[1] - expected1).abs() < 1e-12);
    }
}