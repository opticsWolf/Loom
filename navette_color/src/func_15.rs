// src/func_15.rs
//! Shape handling and broadcasting (re‑export of `metrics::map_pairs`).
//!
//! This unit provides the runtime broadcasting logic required by all
//! Delta‑E metrics and other pairwise functions. The implementation
//! lives in `crate::metrics`.

pub use crate::metrics::map_pairs;