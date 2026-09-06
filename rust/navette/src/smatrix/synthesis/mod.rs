//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis — automated thin-film design synthesis (needle method).
//!
//! Sub-module map:
//!   structure   — DesignStack / LayerSpec: stack model and layer ops
//!                 (split-insert, merge_adjacent, remove, clamp) plus
//!                 solver-array materialization for core_engine /
//!                 needle_engine layouts.
//!
//! Design notes:
//!   * Pure Rust, no pyo3 in this tree — Python access goes through a thin
//!     shell added later (mirrors needle_operator / needle_engine split).
//!   * No optics math lives here; solver arrays produced by `structure`
//!     are consumed verbatim by coherent_block / core_engine /
//!     needle_engine. Conventions inherited from optics_core.
//!
//! Reference:
//!   Tikhonravov, Trubetskov, DeBell, "Application of the needle
//!   optimization technique to the design of optical coatings,"
//!   Appl. Opt. 35(28), 5493–5508 (1996).

pub mod structure;
pub mod merit;
pub mod thick_opt;
pub mod needle_pass;
pub mod context;
pub mod cleanup;
pub mod inflate;
pub mod config;
pub mod stagnation;
pub mod evaluator;
pub mod cycle;
pub mod design_config;
pub mod driver;
pub mod pipeline;
