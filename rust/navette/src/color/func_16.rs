// src/func_16.rs
//! CIEDE2000 Color Difference.

use crate::color::common::DEG2RAD;

const C25_7: f64 = 6103515625.0; // 25.0^7

/// Single-pair CIEDE2000 colour difference with parametric weights
/// `k_l` (lightness), `k_c` (chroma), `k_h` (hue); use 1.0/1.0/1.0 for
/// reference conditions. Includes the rotation term for the blue region.
///
/// See the module docs for the full formula; the batch [`delta_e_2000`]
/// maps this over pairs with NumPy-style broadcasting.
#[inline(always)]
pub fn delta_e_2000_single(lab1: &[f64; 3], lab2: &[f64; 3], k_l: f64, k_c: f64, k_h: f64) -> f64 {
    let l1 = lab1[0]; let a1 = lab1[1]; let b1 = lab1[2];
    let l2 = lab2[0]; let a2 = lab2[1]; let b2 = lab2[2];

    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let c_bar = (c1 + c2) * 0.5;
    let c_bar_7 = c_bar.powi(7);
    
    let g = 0.5 * (1.0 - (c_bar_7 / (c_bar_7 + C25_7)).sqrt());
    let scale = 1.0 + g;
    
    let a1_p = scale * a1;
    let a2_p = scale * a2;
    
    let c1_p = a1_p.hypot(b1);
    let c2_p = a2_p.hypot(b2);
    
    let h1_p = b1.atan2(a1_p).to_degrees().rem_euclid(360.0);
    let h2_p = b2.atan2(a2_p).to_degrees().rem_euclid(360.0);
    
    let dl_p = l2 - l1;
    let dc_p = c2_p - c1_p;
    
    let mut dh_p = 0.0;
    if c1_p * c2_p > 1e-12 {
        let diff = h2_p - h1_p;
        if diff.abs() <= 180.0 { dh_p = diff; }
        else if diff > 180.0 { dh_p = diff - 360.0; }
        else { dh_p = diff + 360.0; }
    }
    let dh_p_cap = 2.0 * (c1_p * c2_p).sqrt() * ((dh_p * DEG2RAD) * 0.5).sin();
    
    let l_bar_p = (l1 + l2) * 0.5;
    let c_bar_p = (c1_p + c2_p) * 0.5;
    
    let mut h_bar_p = h1_p + h2_p;
    if c1_p * c2_p > 1e-12 {
        if (h1_p - h2_p).abs() <= 180.0 { h_bar_p *= 0.5; }
        else if h_bar_p < 360.0 { h_bar_p = (h_bar_p + 360.0) * 0.5; }
        else { h_bar_p = (h_bar_p - 360.0) * 0.5; }
    }
    
    let t = 1.0 - 0.17 * ((h_bar_p - 30.0) * DEG2RAD).cos()
                + 0.24 * ((2.0 * h_bar_p) * DEG2RAD).cos()
                + 0.32 * ((3.0 * h_bar_p + 6.0) * DEG2RAD).cos()
                - 0.20 * ((4.0 * h_bar_p - 63.0) * DEG2RAD).cos();
                
    let d_theta = 30.0 * (-((h_bar_p - 275.0) / 25.0).powi(2)).exp();
    let c_bar_p_7 = c_bar_p.powi(7);
    let rc = 2.0 * (c_bar_p_7 / (c_bar_p_7 + C25_7)).sqrt();
    let rt = -((2.0 * d_theta) * DEG2RAD).sin() * rc;
    
    let l_term = (l_bar_p - 50.0).powi(2);
    let sl = 1.0 + (0.015 * l_term) / (20.0 + l_term).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    
    let val_l = dl_p / (k_l * sl);
    let val_c = dc_p / (k_c * sc);
    let val_h = dh_p_cap / (k_h * sh);
    
    (val_l.powi(2) + val_c.powi(2) + val_h.powi(2) + rt * val_c * val_h).sqrt()
}

/// Batch CIEDE2000 over two CIELAB batches with NumPy-style broadcasting
/// (lengths must match or one side must have length 1).
///
/// # Panics
/// Panics if the two batches have different lengths and neither length is 1.
pub fn delta_e_2000(lab1: &[[f64; 3]], lab2: &[[f64; 3]], k_l: f64, k_c: f64, k_h: f64) -> Vec<f64> {
    crate::color::metrics::map_pairs(lab1, lab2, |a, b| delta_e_2000_single(a, b, k_l, k_c, k_h))
}