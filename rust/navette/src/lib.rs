//! Navette — unified optical thin-film engine (pure Rust, no Python).
//!
//! Single crate, six modules. Rust consumers depend on this one crate;
//! the Python bindings in `navette-py` build on it.
//!
//! ```rust,no_run
//! // S-matrix core through the umbrella:
//! let _ = navette::smatrix::core_engine::REQ_RS;
//! ```

/// CIE color science: spaces, conversions, delta-E, whites.
pub mod color;
/// Univariate interpolation (linear / pchip / makima).
pub mod interpolate;
/// Optical dispersion models (Cauchy … UBF, tables, EMA, KK).
pub mod materials;
/// S-matrix thin-film engine + needle synthesis pipeline.
pub mod smatrix;
/// Spectral weaving: resampling, merit, targets.
pub mod spectralweave;
/// Stack model: layers, groups, providers, expansion, architect.
pub mod structure;
/// Program documents: versioned envelopes + section assembly.
pub mod config;
