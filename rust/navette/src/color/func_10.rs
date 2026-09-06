// src/func_10.rs
//! Delta E 94 – CIE 1994 colour difference.
//!
//! This metric is **asymmetric**: the first argument (`lab1`) is treated as the
//! reference, and its chroma is used for the weighting factors `S_C` and `S_H`.
//! Swapping the arguments yields a different result.
//!
//! The formula is:
//! ```text
//! ΔE₉₄ = √[(ΔL* / (k_L·S_L))² + (ΔC* / (k_C·S_C))² + (ΔH* / (k_H·S_H))²]
//! ```
//! where `S_L = 1`, `S_C = 1 + K₁·C₁`, `S_H = 1 + K₂·C₁`, and
//! `ΔH* = √(Δa² + Δb² – ΔC²)` clamped to non‑negative.

/// Parameters for CIE 1994 Delta E.
///
/// - `k_l`: Lightness parametric factor (default 1.0 for graphic arts).
/// - `k1`: Chroma parametric factor (default 0.045 for graphic arts).
/// - `k2`: Hue parametric factor (default 0.015 for graphic arts).
#[derive(Clone, Copy, Debug)]
pub struct De94Params {
    pub k_l: f64,
    pub k1: f64,
    pub k2: f64,
}

impl De94Params {
    /// Graphic arts standard (default) parameters: `k_l = 1.0`, `k1 = 0.045`, `k2 = 0.015`.
    pub const GRAPHIC: Self = Self {
        k_l: 1.0,
        k1: 0.045,
        k2: 0.015,
    };
    /// Textile industry parameters: `k_l = 2.0`, `k1 = 0.048`, `k2 = 0.014`.
    pub const TEXTILES: Self = Self {
        k_l: 2.0,
        k1: 0.048,
        k2: 0.014,
    };
}

/// Single‑pixel Delta E 94 with the given parameters.
///
/// # Asymmetry
/// `lab1` is the **reference**, `lab2` is the **sample**.
#[inline(always)]
pub fn delta_e_94_single(lab1: &[f64; 3], lab2: &[f64; 3], params: De94Params) -> f64 {
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
    // dH² = da² + db² - dC², clamp to zero to avoid negative due to FP error
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);

    let sc = 1.0 + params.k1 * c1;
    let sh = 1.0 + params.k2 * c1;
    let sl = 1.0;

    let term_l = dl / (params.k_l * sl);
    let term_c = dc / sc;
    let term_h = dh_sq.sqrt() / sh;

    (term_l * term_l + term_c * term_c + term_h * term_h).sqrt()
}

/// Compute the CIE 1994 colour difference between two batches of CIELAB colours.
///
/// # Broadcasting
/// The function supports NumPy‑style broadcasting: if one batch has length 1
/// and the other has length N, the single value is paired with every element
/// of the other batch. Both batches must have length 1 or equal length,
/// otherwise the function panics.
///
/// # Asymmetry
/// The first argument `lab1` is the **reference** batch; the second `lab2` is
/// the **sample** batch. Swapping arguments will generally change the result.
///
/// # Panics
/// Panics if the two slices have different lengths and neither length is 1.
///
/// # Examples
/// ```
/// use navette::color::func_10::{delta_e_94, De94Params};
/// let refs = [[50.0, 0.0, 0.0]];
/// let samples = [[55.0, 0.0, 0.0], [60.0, 0.0, 0.0]];
/// let de = delta_e_94(&refs, &samples, De94Params::GRAPHIC);
/// // Pure lightness difference: ΔE = ΔL / k_L = 5.0 / 1.0 = 5.0
/// assert!((de[0] - 5.0).abs() < 1e-12);
/// // For textiles: k_L = 2.0 => ΔE = 5.0 / 2.0 = 2.5
/// let de_tex = delta_e_94(&refs, &samples, De94Params::TEXTILES);
/// assert!((de_tex[0] - 2.5).abs() < 1e-12);
/// ```
pub fn delta_e_94(lab1: &[[f64; 3]], lab2: &[[f64; 3]], params: De94Params) -> Vec<f64> {
    crate::color::metrics::map_pairs(lab1, lab2, |a, b| delta_e_94_single(a, b, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_pure_lightness() {
        let refs = [50.0, 0.0, 0.0];
        let samp = [55.0, 0.0, 0.0];
        let de = delta_e_94_single(&refs, &samp, De94Params::GRAPHIC);
        assert!((de - 5.0).abs() < 1e-12);
        let de_tex = delta_e_94_single(&refs, &samp, De94Params::TEXTILES);
        assert!((de_tex - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_single_pure_chroma() {
        // Reference has chroma 10, sample has chroma 15, both at same hue (a>0,b=0)
        let refs = [50.0, 10.0, 0.0];
        let samp = [50.0, 15.0, 0.0];
        let de = delta_e_94_single(&refs, &samp, De94Params::GRAPHIC);
        // ΔC = -5 (ref - sample) → absolute value? Formula uses dC = C1 - C2 = -5, then squared -> 25
        // S_C = 1 + 0.045*10 = 1.45, term_C = 5/1.45 ≈ 3.4483, ΔE = term_C
        let expected = 5.0 / (1.0 + 0.045 * 10.0);
        assert!((de - expected).abs() < 1e-12);
    }

    #[test]
    fn test_asymmetry() {
        let a = [50.0, 10.0, 0.0];
        let b = [50.0, 15.0, 0.0];
        let de_ab = delta_e_94_single(&a, &b, De94Params::GRAPHIC);
        let de_ba = delta_e_94_single(&b, &a, De94Params::GRAPHIC);
        // S_C depends on reference chroma: for a->b it's 1+0.045*10=1.45, for b->a it's 1+0.045*15=1.675
        // ΔC = C1-C2 gives -5 vs +5 → squared same, term_C = 5/1.45 vs 5/1.675, not equal
        assert!((de_ab - de_ba).abs() > 1e-6);
    }

    #[test]
    fn test_broadcast() {
        let refs = [[50.0, 0.0, 0.0]];
        let samples = [[55.0, 0.0, 0.0], [60.0, 0.0, 0.0]];
        let res = delta_e_94(&refs, &samples, De94Params::GRAPHIC);
        assert_eq!(res.len(), 2);
        assert!((res[0] - 5.0).abs() < 1e-12);
        assert!((res[1] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_equal_length() {
        let a = [[50.0, 10.0, 20.0], [60.0, 0.0, 0.0]];
        let b = [[55.0, 11.0, 22.0], [65.0, 0.0, 0.0]];
        let res = delta_e_94(&a, &b, De94Params::GRAPHIC);
        assert_eq!(res.len(), 2);
        let expected0 = delta_e_94_single(&a[0], &b[0], De94Params::GRAPHIC);
        let expected1 = delta_e_94_single(&a[1], &b[1], De94Params::GRAPHIC);
        assert!((res[0] - expected0).abs() < 1e-12);
        assert!((res[1] - expected1).abs() < 1e-12);
    }
}