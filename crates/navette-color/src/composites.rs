// src/composites.rs
//! Convenience composite functions and Gamut Mapping.

use crate::common::{srgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_srgb, REF_WHITE_D65, clip01};
use crate::func_01::{xyz_to_xyy, xyy_to_xyz};
use crate::func_02::{lab_to_lch, lch_to_lab};
use crate::func_03::{xyz_to_luv, luv_to_xyz};

pub fn srgb_to_lab(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; srgb.len()];
    srgb_to_xyz(srgb, true, &mut xyz);
    xyz_to_lab(&xyz, &REF_WHITE_D65, out);
}

pub fn lab_to_srgb(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; lab.len()];
    lab_to_xyz(lab, &REF_WHITE_D65, &mut xyz);
    xyz_to_srgb(&xyz, true, out);
}

pub fn srgb_to_lch(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut lab = vec![[0.0; 3]; srgb.len()];
    srgb_to_lab(srgb, &mut lab);
    lab_to_lch(&lab, out);
}

pub fn lch_to_srgb(lch: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut lab = vec![[0.0; 3]; lch.len()];
    lch_to_lab(lch, &mut lab);
    lab_to_srgb(&lab, out);
}

pub fn srgb_to_luv(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; srgb.len()];
    srgb_to_xyz(srgb, true, &mut xyz);
    xyz_to_luv(&xyz, &REF_WHITE_D65, out);
}

pub fn luv_to_srgb(luv: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; luv.len()];
    luv_to_xyz(luv, &REF_WHITE_D65, &mut xyz);
    xyz_to_srgb(&xyz, true, out);
}

pub fn srgb_to_xyy_bound(srgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; srgb.len()];
    srgb_to_xyz(srgb, true, &mut xyz);
    xyz_to_xyy(&xyz, out);
}

pub fn xyy_to_srgb(xyy: &[[f64; 3]], out: &mut [[f64; 3]]) {
    let mut xyz = vec![[0.0; 3]; xyy.len()];
    xyy_to_xyz(xyy, &mut xyz);
    xyz_to_srgb(&xyz, true, out);
}

pub fn clip_absolute(rgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (r, o) in rgb.iter().zip(out.iter_mut()) {
        *o = [clip01(r[0]), clip01(r[1]), clip01(r[2])];
    }
}