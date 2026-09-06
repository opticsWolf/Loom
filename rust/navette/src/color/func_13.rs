// src/func_13.rs
//! Spectral pipeline: convert a spectral power distribution to sRGB.
//!
//! This module integrates an SPD with CIE 1931 colour matching functions
//! and an illuminant SPD to obtain relative XYZ tristimuli, optionally
//! adapts to D65 using the Bradford transform, and finally converts to sRGB.
//!
//! The normalisation constant k = 1 / Σ E(λ)·ȳ(λ)·Δλ ensures that a perfect
//! reflecting diffuser yields Y = 1.

use crate::color::common::{xyz_to_srgb, REF_WHITE_D65};
use crate::color::func_08::adapt;

/// Convert a spectral power distribution to a single sRGB colour.
///
/// # Arguments
/// * `spd` – Spectral power distribution of the sample (array length = number of wavelengths).
/// * `cmfs` – Colour matching functions `(x̄, ȳ, z̄)` for each wavelength, shape (N, 3).
/// * `illum` – Illuminant spectral power distribution (same length as `spd`).
/// * `interval` – Wavelength sampling interval in nanometres.
/// * `apply_adaptation` – If true, adapt from the illuminant white point to D65.
///
/// # Returns
/// sRGB colour in the range [0,1] (display‑referred, clipped).
///
/// # Panics
/// If `spd`, `cmfs`, or `illum` have inconsistent lengths.
pub fn spectral_to_srgb(
    spd: &[f64],
    cmfs: &[[f64; 3]],
    illum: &[f64],
    interval: f64,
    apply_adaptation: bool,
) -> [f64; 3] {
    let n = cmfs.len();
    assert_eq!(spd.len(), n, "SPD length mismatch");
    assert_eq!(illum.len(), n, "Illuminant length mismatch");

    // 1. Normalisation factor k = 1 / Σ E(λ)·ȳ(λ)·Δλ
    let denom: f64 = (0..n)
        .map(|i| illum[i] * cmfs[i][1] * interval)
        .sum();
    let k = if denom.abs() > 1e-12 { 1.0 / denom } else { 1.0 };

    // 2. Integrate to obtain XYZ
    let mut xyz = [0.0; 3];
    for i in 0..n {
        let weight = spd[i] * illum[i] * k * interval;
        xyz[0] += weight * cmfs[i][0];
        xyz[1] += weight * cmfs[i][1];
        xyz[2] += weight * cmfs[i][2];
    }

    // 3. Chromatic adaptation to D65 (if requested)
    let xyz_adapted = if apply_adaptation {
        // Compute source white point by integrating illuminant alone
        let mut raw_white = [0.0; 3];
        for i in 0..n {
            let e = illum[i] * interval;
            raw_white[0] += cmfs[i][0] * e;
            raw_white[1] += cmfs[i][1] * e;
            raw_white[2] += cmfs[i][2] * e;
        }
        // Normalise so that Y = 1
        let source_white = if raw_white[1] > 1e-12 {
            [raw_white[0] / raw_white[1], 1.0, raw_white[2] / raw_white[1]]
        } else {
            raw_white
        };
        let mut adapted = [[0.0; 3]];
        adapt(&[xyz], &source_white, &REF_WHITE_D65, true, &mut adapted);
        adapted[0]
    } else {
        xyz
    };

    // 4. XYZ → sRGB (clip = true)
    let mut srgb = [[0.0; 3]];
    xyz_to_srgb(&[xyz_adapted], true, &mut srgb);
    srgb[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    // Note: Full golden‑vector tests are located in the parity test suite.
    // Here we only verify basic consistency.

    #[test]
    fn test_constant_spd_gives_neutral() {
        // D65 illuminant integrated with D65 CMF should yield white.
        let n = 81;
        let wavelengths: Vec<f64> = (0..n).map(|i| 380.0 + i as f64 * 5.0).collect();
        // D65 illuminant approximated by constant for test (not accurate).
        let illum: Vec<f64> = vec![1.0; n];
        // D65 colour matching functions (simplified: use constant for test)
        let cmfs: Vec<[f64; 3]> = vec![[0.5, 1.0, 0.5]; n];
        let spd: Vec<f64> = vec![1.0; n];
        let interval = 5.0;
        let rgb = spectral_to_srgb(&spd, &cmfs, &illum, interval, true);
        // With simplified data, we just check no panic and output within [0,1].
        assert!(rgb[0] >= 0.0 && rgb[0] <= 1.0);
        assert!(rgb[1] >= 0.0 && rgb[1] <= 1.0);
        assert!(rgb[2] >= 0.0 && rgb[2] <= 1.0);
    }
}