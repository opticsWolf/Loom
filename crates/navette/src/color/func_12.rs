// src/func_12.rs
//! DIN99 colour difference (DIN 6176).
//!
//! The DIN99 colour space is a Euclideanisation of CIELAB via a 16° rotation
//! of the a,b plane and logarithmic compression of lightness and chroma.
//! The final ΔE is the Euclidean distance in this space.
//!
//! Two parameter presets exist:
//! - Graphics (default): `kE = 1.0`, `kCH = 1.0`
//! - Textiles: `kE = 2.0`, `kCH = 0.5`

use crate::color::common::DEG2RAD;

/// DIN99 lightness compression coefficient.
///
/// colour-science (following Cui et al. 2002) uses `105.509`. The DIN 6176
/// standard rounds this to `105.51`; using the rounded value introduces a
/// ~2e-4 parity error against the reference, so the precise constant is used.
const DIN99_L_COEFF: f64 = 105.509;

/// Convert a single CIELAB colour to DIN99 coordinates.
///
/// # Parameters
/// - `lab`: CIELAB colour.
/// - `ke`: Lightness scaling factor (graphics: 1.0, textiles: 2.0).
/// - `kch`: Chroma/hue scaling factor (graphics: 1.0, textiles: 0.5).
/// - `cos16`, `sin16`: Pre‑computed cosine and sine of 16°.
#[inline(always)]
fn din99_coords(lab: &[f64; 3], ke: f64, kch: f64, cos16: f64, sin16: f64) -> (f64, f64, f64) {
    let l = lab[0];
    let a = lab[1];
    let b = lab[2];

    // Lightness scales WITH ke (colour-science / DIN 6176 convention): L99 ∝ ke.
    let l99 = DIN99_L_COEFF * (0.0158 * l).ln_1p() * ke;

    let e = a * cos16 + b * sin16;
    let f = 0.7 * (-a * sin16 + b * cos16);

    let g = (e * e + f * f).sqrt();
    let (c99, h99) = if g < 1e-12 {
        (0.0, 0.0)
    } else {
        // Chroma compression: C99 = ln(1 + 0.045·G) / (0.045·kch·ke).
        let c = (0.045 * g).ln_1p() / (0.045 * kch * ke);
        let h = f.atan2(e);
        (c, h)
    };

    let a99 = c99 * h99.cos();
    let b99 = c99 * h99.sin();
    (l99, a99, b99)
}

/// Single‑pixel DIN99 colour difference.
///
/// Both colours are transformed into DIN99 space and then Euclidean distance
/// is taken.
#[inline(always)]
pub fn delta_e_din99_single(lab1: &[f64; 3], lab2: &[f64; 3], ke: f64, kch: f64) -> f64 {
    let cos16 = (16.0 * DEG2RAD).cos();
    let sin16 = (16.0 * DEG2RAD).sin();

    let (l1, a1, b1) = din99_coords(lab1, ke, kch, cos16, sin16);
    let (l2, a2, b2) = din99_coords(lab2, ke, kch, cos16, sin16);

    let dl = l1 - l2;
    let da = a1 - a2;
    let db = b1 - b2;
    (dl * dl + da * da + db * db).sqrt()
}

/// Compute the DIN99 colour difference between two batches of CIELAB colours.
///
/// # Broadcasting
/// The function supports NumPy‑style broadcasting: if one batch has length 1
/// and the other has length N, the single value is paired with every element
/// of the other batch. Both batches must have length 1 or equal length,
/// otherwise the function panics.
///
/// # Parameters
/// - `ke`: Lightness scaling factor (graphics: 1.0, textiles: 2.0).
/// - `kch`: Chroma/hue scaling factor (graphics: 1.0, textiles: 0.5).
///
/// # Panics
/// Panics if the two slices have different lengths and neither length is 1.
///
/// # Examples
/// ```
/// use navette::color::func_12::delta_e_din99;
/// let refs = [[50.0, 0.0, 0.0]];
/// let samples = [[55.0, 0.0, 0.0]];
/// let de = delta_e_din99(&refs, &samples, 1.0, 1.0);
/// ```
pub fn delta_e_din99(lab1: &[[f64; 3]], lab2: &[[f64; 3]], ke: f64, kch: f64) -> Vec<f64> {
    crate::color::metrics::map_pairs(lab1, lab2, |a, b| delta_e_din99_single(a, b, ke, kch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_pure_lightness() {
        // Only lightness differs, a=b=0
        let refs = [50.0, 0.0, 0.0];
        let samp = [55.0, 0.0, 0.0];
        let de = delta_e_din99_single(&refs, &samp, 1.0, 1.0);
        // L99 = DIN99_L_COEFF * ln(1 + 0.0158*L)
        let l1 = DIN99_L_COEFF * (0.0158 * 50.0_f64).ln_1p();
        let l2 = DIN99_L_COEFF * (0.0158 * 55.0_f64).ln_1p();
        let expected = (l1 - l2).abs();
        assert!((de - expected).abs() < 1e-12);
    }

    #[test]
    fn test_pure_chroma_achromatic_branch() {
        // Both colours have a=b=0 -> G=0 -> C99=0, h99=0 -> a99=b99=0
        let refs = [50.0, 0.0, 0.0];
        let samp = [50.0, 0.0, 0.0];
        let de = delta_e_din99_single(&refs, &samp, 1.0, 1.0);
        assert!(de.abs() < 1e-12);
    }

    #[test]
    fn test_textile_preset() {
        // Only lightness, textiles: ke=2.0, kch=0.5
        let refs = [50.0, 0.0, 0.0];
        let samp = [55.0, 0.0, 0.0];
        let de_gfx = delta_e_din99_single(&refs, &samp, 1.0, 1.0);
        let de_tex = delta_e_din99_single(&refs, &samp, 2.0, 0.5);
        // For pure lightness, kch irrelevant. ke doubles the difference.
        assert!((de_tex / de_gfx - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_and_flattening() {
        // Colour with a=1,b=0 after rotation e = cos16, f = -0.7*sin16
        let lab = [50.0, 1.0, 0.0];
        let cos16 = (16.0 * DEG2RAD).cos();
        let sin16 = (16.0 * DEG2RAD).sin();
        let (_l, a99, b99) = din99_coords(&lab, 1.0, 1.0, cos16, sin16);
        // Since G is small, C99 ~ ln(1+0.045*G) ~ 0.045*G (small)
        // We only check that a99,b99 are not simply (1,0) – the transform changed them.
        assert!(a99.abs() < 1.0);
        assert!(b99.abs() < 1.0);
    }

    #[test]
    fn test_broadcast() {
        let refs = [[50.0, 10.0, 5.0]];
        let samples = [[51.0, 11.0, 6.0], [52.0, 12.0, 7.0]];
        let res = delta_e_din99(&refs, &samples, 1.0, 1.0);
        assert_eq!(res.len(), 2);
        let expected0 = delta_e_din99_single(&refs[0], &samples[0], 1.0, 1.0);
        let expected1 = delta_e_din99_single(&refs[0], &samples[1], 1.0, 1.0);
        assert!((res[0] - expected0).abs() < 1e-12);
        assert!((res[1] - expected1).abs() < 1e-12);
    }

    #[test]
    fn test_equal_length() {
        let a = [[50.0, 10.0, 20.0], [60.0, 0.0, 0.0]];
        let b = [[55.0, 11.0, 22.0], [65.0, 0.0, 0.0]];
        let res = delta_e_din99(&a, &b, 1.0, 1.0);
        assert_eq!(res.len(), 2);
        let expected0 = delta_e_din99_single(&a[0], &b[0], 1.0, 1.0);
        let expected1 = delta_e_din99_single(&a[1], &b[1], 1.0, 1.0);
        assert!((res[0] - expected0).abs() < 1e-12);
        assert!((res[1] - expected1).abs() < 1e-12);
    }
}
