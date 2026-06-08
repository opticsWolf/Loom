// src/func_14.rs
//! Photometry engine: photopic, scotopic, and mesopic luminous flux.
//!
//! This module provides a high‑performance calculator for luminous flux
//! using the CIE spectral luminous efficiency functions for photopic (V(λ))
//! and scotopic (V'(λ)) vision, with standard efficacy constants.

/// Photometry engine holding pre‑computed V(λ) and V'(λ) curves.
pub struct PhotometryEngine {
    vp: Vec<f64>,      // photopic V(λ)
    vs: Vec<f64>,      // scotopic V'(λ)
    pub km_p: f64,     // photopic efficacy (default 683.002 lm/W)
    pub km_s: f64,     // scotopic efficacy (default 1700.05 lm/W)
}

/// Vision type for flux calculation.
pub enum Vision {
    /// Photopic (daylight) vision: uses V(λ).
    Photopic,
    /// Scotopic (night) vision: uses V'(λ).
    Scotopic,
    /// Mesopic (twilight) vision: blends photopic and scotopic.
    /// The blending factor `m` (0 = scotopic, 1 = photopic) is provided
    /// at call time via `calculate_flux`.
    Mesopic,
}

impl PhotometryEngine {
    /// Create a new photometry engine with default efficacy constants.
    ///
    /// # Arguments
    /// * `v_photopic` – Photopic luminous efficiency V(λ) as a vector.
    /// * `v_scotopic` – Scotopic luminous efficiency V'(λ) as a vector.
    ///
    /// Both slices must have the same length. The constants used are
    /// `Km_p = 683.002` and `Km_s = 1700.05`.
    pub fn new(v_photopic: Vec<f64>, v_scotopic: Vec<f64>) -> Self {
        Self::with_constants(v_photopic, v_scotopic, 683.002, 1700.05)
    }

    /// Create a new photometry engine with custom efficacy constants.
    pub fn with_constants(
        v_photopic: Vec<f64>,
        v_scotopic: Vec<f64>,
        km_p: f64,
        km_s: f64,
    ) -> Self {
        assert_eq!(
            v_photopic.len(),
            v_scotopic.len(),
            "Photopic and scotopic curve lengths must match"
        );
        Self {
            vp: v_photopic,
            vs: v_scotopic,
            km_p,
            km_s,
        }
    }

    /// Core flux kernel: compute Σ spd[i] · (vp[i]·w_p + vs[i]·w_s) · Δλ.
    #[inline]
    fn flux_kernel(&self, spd: &[f64], w_p: f64, w_s: f64, interval: f64) -> f64 {
        let mut total = 0.0;
        for i in 0..spd.len() {
            total += spd[i] * (self.vp[i] * w_p + self.vs[i] * w_s);
        }
        total * interval
    }

    /// Calculate luminous flux for a given SPD.
    ///
    /// # Arguments
    /// * `spd` – Spectral power distribution (same length as V(λ) curves).
    /// * `vision` – Type of vision (`Photopic`, `Scotopic`, or `Mesopic`).
    /// * `m` – Mesopic adaptation factor (ignored for photopic/scotopic).
    ///         `m = 1` gives pure photopic, `m = 0` pure scotopic.
    /// * `interval` – Wavelength sampling interval in nanometres.
    ///
    /// # Returns
    /// Luminous flux in lumens.
    pub fn calculate_flux(
        &self,
        spd: &[f64],
        vision: Vision,
        m: f64,
        interval: f64,
    ) -> f64 {
        let (w_p, w_s) = match vision {
            Vision::Photopic => (self.km_p, 0.0),
            Vision::Scotopic => (0.0, self.km_s),
            Vision::Mesopic => (m * self.km_p, (1.0 - m) * self.km_s),
        };
        self.flux_kernel(spd, w_p, w_s, interval)
    }

    /// Calculate the S/P ratio (scotopic / photopic flux).
    ///
    /// Returns 0.0 if the photopic flux is below `1e-12`.
    pub fn calculate_sp_ratio(&self, spd: &[f64], interval: f64) -> f64 {
        let photopic = self.flux_kernel(spd, self.km_p, 0.0, interval);
        if photopic < 1e-12 {
            return 0.0;
        }
        let scotopic = self.flux_kernel(spd, 0.0, self.km_s, interval);
        scotopic / photopic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photopic_flux() {
        let vp = vec![1.0, 0.5, 0.0];
        let vs = vec![0.0, 0.2, 0.8];
        let engine = PhotometryEngine::new(vp, vs);
        let spd = vec![10.0, 20.0, 30.0];
        let interval = 1.0;
        let flux = engine.calculate_flux(&spd, Vision::Photopic, 0.0, interval);
        // w_p = 683.002, w_s = 0
        let expected = (10.0 * 683.002 * 1.0 + 20.0 * 683.002 * 0.5 + 30.0 * 683.002 * 0.0) * 1.0;
        assert!((flux - expected).abs() < 1e-9);
    }

    #[test]
    fn test_scotopic_flux() {
        let vp = vec![1.0, 0.5, 0.0];
        let vs = vec![0.0, 0.2, 0.8];
        let engine = PhotometryEngine::new(vp, vs);
        let spd = vec![10.0, 20.0, 30.0];
        let interval = 1.0;
        let flux = engine.calculate_flux(&spd, Vision::Scotopic, 0.0, interval);
        // w_p = 0, w_s = 1700.05
        let expected = (10.0 * 1700.05 * 0.0 + 20.0 * 1700.05 * 0.2 + 30.0 * 1700.05 * 0.8) * 1.0;
        assert!((flux - expected).abs() < 1e-9);
    }

    #[test]
    fn test_mesopic_flux() {
        let vp = vec![1.0, 0.5];
        let vs = vec![0.0, 0.2];
        let engine = PhotometryEngine::new(vp, vs);
        let spd = vec![10.0, 20.0];
        let interval = 1.0;
        let m = 0.3;
        let flux = engine.calculate_flux(&spd, Vision::Mesopic, m, interval);
        let w_p = m * 683.002;
        let w_s = (1.0 - m) * 1700.05;
        let expected = (10.0 * (1.0 * w_p + 0.0 * w_s) + 20.0 * (0.5 * w_p + 0.2 * w_s)) * interval;
        assert!((flux - expected).abs() < 1e-9);
    }

    #[test]
    fn test_sp_ratio() {
        let vp = vec![1.0, 0.0];
        let vs = vec![0.0, 1.0];
        let engine = PhotometryEngine::new(vp, vs);
        let spd = vec![100.0, 200.0];
        let interval = 1.0;
        let ratio = engine.calculate_sp_ratio(&spd, interval);
        let photopic = 100.0 * 683.002 * 1.0 + 200.0 * 683.002 * 0.0;
        let scotopic = 100.0 * 1700.05 * 0.0 + 200.0 * 1700.05 * 1.0;
        let expected = scotopic / photopic;
        assert!((ratio - expected).abs() < 1e-9);
    }
}