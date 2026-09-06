// src/func_09.rs
//! Delta E 76 – Euclidean distance in CIELAB.

/// Single-pair CIE 1976 colour difference: Euclidean distance in CIELAB.
///
/// Symmetric in its arguments; the batch [`delta_e_76`] maps this over pairs.
#[inline(always)]
pub fn delta_e_76_single(lab1: &[f64; 3], lab2: &[f64; 3]) -> f64 {
    let dl = lab1[0] - lab2[0];
    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// Compute the CIE 1976 colour difference (Delta E 76) between two batches of
/// CIELAB colours.
///
/// # Broadcasting
/// The function supports NumPy‑style broadcasting: if one batch has length 1
/// and the other has length N, the single value is paired with every element
/// of the other batch. Both batches must have length 1 or equal length,
/// otherwise the function panics.
///
/// # Panics
/// Panics if the two slices have different lengths and neither length is 1.
///
/// # Examples
/// ```
/// use navette::color::func_09::delta_e_76;
/// let refs = [[50.0, 0.0, 0.0]];
/// let samples = [[55.0, 0.0, 0.0], [60.0, 0.0, 0.0]];
/// let de = delta_e_76(&refs, &samples);
/// assert!((de[0] - 5.0).abs() < 1e-12);
/// assert!((de[1] - 10.0).abs() < 1e-12);
/// ```
pub fn delta_e_76(lab1: &[[f64; 3]], lab2: &[[f64; 3]]) -> Vec<f64> {
    crate::color::metrics::map_pairs(lab1, lab2, delta_e_76_single)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single() {
        let a = [50.0, 0.0, 0.0];
        let b = [55.0, 0.0, 0.0];
        assert!((delta_e_76_single(&a, &b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_broadcast() {
        let refs = [[50.0, 0.0, 0.0]];
        let samples = [[55.0, 0.0, 0.0], [60.0, 0.0, 0.0]];
        let res = delta_e_76(&refs, &samples);
        assert_eq!(res.len(), 2);
        assert!((res[0] - 5.0).abs() < 1e-12);
        assert!((res[1] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_equal_length() {
        let a = [[50.0, 10.0, 20.0], [60.0, 0.0, 0.0]];
        let b = [[55.0, 11.0, 22.0], [65.0, 0.0, 0.0]];
        let res = delta_e_76(&a, &b);
        assert_eq!(res.len(), 2);
        // Computed manually (or trust the single function)
        let dl0 = a[0][0] - b[0][0];
        let da0 = a[0][1] - b[0][1];
        let db0 = a[0][2] - b[0][2];
        let expected0 = (dl0 * dl0 + da0 * da0 + db0 * db0).sqrt();
        assert!((res[0] - expected0).abs() < 1e-12);
    }
}