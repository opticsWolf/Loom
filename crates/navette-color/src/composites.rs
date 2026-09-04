// src/composites.rs
//! Convenience composite functions and Gamut Mapping.

use crate::common::{srgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_srgb, REF_WHITE_D65, clip01};
use crate::func_01::{xyz_to_xyy, xyy_to_xyz};
use crate::func_02::{lab_to_lch, lch_to_lab};
use crate::func_03::{xyz_to_luv, luv_to_xyz};

/// sRGB → CIELAB (D65) via XYZ. Input is gamma-encoded sRGB in [0, 1];
/// values are clipped (display-referred) before conversion.
pub fn srgb_to_lab(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; srgb.len()];
    srgb_to_xyz(srgb, true, &mut xyz);
    xyz_to_lab(&xyz, &REF_WHITE_D65, out);
}

/// CIELAB (D65) → sRGB via XYZ, gamma-encoded and clipped to [0, 1].
pub fn lab_to_srgb(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; lab.len()];
    lab_to_xyz(lab, &REF_WHITE_D65, &mut xyz);
    xyz_to_srgb(&xyz, true, out);
}

/// sRGB → cylindrical CIELCh (D65): `[L, C, h°]` with hue in [0, 360).
pub fn srgb_to_lch(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut lab = vec![[0.0; 3]; srgb.len()];
    srgb_to_lab(srgb, &mut lab);
    lab_to_lch(&lab, out);
}

/// Cylindrical CIELCh (D65) → sRGB, gamma-encoded and clipped to [0, 1].
pub fn lch_to_srgb(lch: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut lab = vec![[0.0; 3]; lch.len()];
    lch_to_lab(lch, &mut lab);
    lab_to_srgb(&lab, out);
}

/// sRGB → CIELUV (D65) via XYZ.
pub fn srgb_to_luv(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; srgb.len()];
    srgb_to_xyz(srgb, true, &mut xyz);
    xyz_to_luv(&xyz, &REF_WHITE_D65, out);
}

/// CIELUV (D65) → sRGB via XYZ, gamma-encoded and clipped to [0, 1].
pub fn luv_to_srgb(luv: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; luv.len()];
    luv_to_xyz(luv, &REF_WHITE_D65, &mut xyz);
    xyz_to_srgb(&xyz, true, out);
}

/// sRGB → xyY chromaticity (D65) via XYZ; `out` holds `[x, y, Y]`.
pub fn srgb_to_xyy_bound(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; srgb.len()];
    srgb_to_xyz(srgb, true, &mut xyz);
    xyz_to_xyy(&xyz, out);
}

/// xyY chromaticity → sRGB (D65) via XYZ, gamma-encoded and clipped.
pub fn xyy_to_srgb(xyy: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; xyy.len()];
    xyy_to_xyz(xyy, &mut xyz);
    xyz_to_srgb(&xyz, true, out);
}

/// Clamp every channel of an RGB batch into [0, 1] (gamut-clip helper).
pub fn clip_absolute(rgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (r, o) in rgb.iter().zip(out.iter_mut()) {
        *o = [clip01(r[0]), clip01(r[1]), clip01(r[2])];
    }
}