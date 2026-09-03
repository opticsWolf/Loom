//! Wavelength-grid generation from (start, step) segments.
//!
//! Port of `generate_wavelength_array_from_steps`. Given points
//! [(s0, step0), (s1, step1), …, (sN, _)], each consecutive pair (sᵢ, sᵢ₊₁)
//! is filled at `stepᵢ` (half-open: the segment end is excluded), and the
//! final endpoint sN is appended once. The last point's step is ignored.

use ndarray::Array1;

/// Build the concatenated wavelength grid. `points` must have length ≥ 2.
pub fn generate_from_steps(points: &[(f64, f64)]) -> Result<Array1<f64>, String> {
    if points.len() < 2 {
        return Err("At least two points are needed to define ranges.".into());
    }

    // Count total points first (mirrors the two-pass Numba implementation).
    let mut total = 0usize;
    for i in 0..points.len() - 1 {
        let start = points[i].0;
        let end = points[i + 1].0;
        let step = points[i].1;
        let count = ((end - start) / step).floor() as usize;
        total += count;
    }
    total += 1; // final endpoint

    let mut out = Vec::with_capacity(total);
    for i in 0..points.len() - 1 {
        let start = points[i].0;
        let end = points[i + 1].0;
        let step = points[i].1;
        let n_steps = ((end - start) / step).floor() as usize;
        for j in 0..n_steps {
            out.push(start + (j as f64) * step);
        }
    }
    out.push(points[points.len() - 1].0);

    Ok(Array1::from(out))
}
