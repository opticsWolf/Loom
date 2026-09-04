//! # navette_color
//!
//! Rust rewrite of the Loom unified color engine — a parity port of
//! `navette_colorengine.py`. Color batches are `&[[f64; 3]]` (stride-of-3 for
//! aggressive auto-vectorization); spectra are `&[f64]`.
//!
//! Module map:
//! - [`common`]      core constants, `mat3_mul_vec`, transfer fns, sRGB/Lab core
//! - [`matrices`]    generated matrix constants (natural form + numpy inverses)
//! - [`composites`]  chained convenience pipelines (e.g. sRGB <-> Lab) and gamut
//! - [`func_01`]     XYZ <-> xyY
//! - [`func_02`]     Lab <-> LCh
//! - [`func_03`]     XYZ <-> CIELUV
//! - [`func_04`]     XYZ <-> Oklab (direct XYZ matrices)
//! - [`func_05`]     sRGB <-> Oklab (legacy sRGB matrices)
//! - [`func_06`]     CIE 1964 U*V*W*
//! - [`func_07`]     CIE 1960 UCS & chromaticity
//! - [`func_08`]     Bradford chromatic adaptation
//! - [`func_09`]     Delta E 76
//! - [`func_10`]     Delta E 94
//! - [`func_11`]     Delta E CMC(l:c)
//! - [`func_12`]     DIN99
//! - [`func_13`]     spectral pipeline
//! - [`func_14`]     photometry engine
//! - [`func_15`]     shape handling & broadcasting
//! - [`func_16`]     CIEDE2000
//!
//! The Python bindings live in the `navette-py` aggregator crate and expose
//! these kernels as the `navette._color` extension submodule (one of five in
//! the single `navette._navette` module built with `maturin develop` from the
//! workspace root). This crate itself is pure Rust: no pyo3, no I/O.

pub mod common;
pub mod matrices;
pub mod metrics;
pub mod composites;

pub mod func_01;
pub mod func_02;
pub mod func_03;
pub mod func_04;
pub mod func_05;
pub mod func_06;
pub mod func_07;
pub mod func_08;
pub mod func_09;
pub mod func_10;
pub mod func_11;
pub mod func_12;
pub mod func_13;
pub mod func_14;
pub mod func_15;
pub mod func_16;

pub mod prelude {
    pub use crate::common::{REF_WHITE_D50, REF_WHITE_D65};
    pub use crate::composites::*;
    pub use crate::func_01::{xyy_to_xyz, xyz_to_xyy};
    pub use crate::func_02::{lab_to_lch, lch_to_lab};
    pub use crate::func_03::{luv_to_xyz, xyz_to_luv};
    pub use crate::func_04::{oklab_to_xyz, xyz_to_oklab};
    pub use crate::func_05::{oklab_to_srgb, srgb_to_oklab};
    pub use crate::func_06::{uvw_to_xyz, white_point_uv1960, xyz_to_uvw};
    pub use crate::func_07::{ucs_to_xyz, uv1960_to_xy, uv1976_to_xy, xyz_to_ucs, xyz_to_ucs_uv};
    pub use crate::func_08::{adapt, calc_transform_matrix};
    pub use crate::func_09::delta_e_76;
    pub use crate::func_10::{delta_e_94, De94Params};
    pub use crate::func_11::delta_e_cmc;
    pub use crate::func_12::delta_e_din99;
    pub use crate::func_13::spectral_to_srgb;
    pub use crate::func_14::{PhotometryEngine, Vision};
    pub use crate::func_16::delta_e_2000;
}

// ============================================================================
// Python bindings (numpy-based). Enabled with `--features python`.
// Targets pyo3 0.28 + rust-numpy 0.28.
// ============================================================================
