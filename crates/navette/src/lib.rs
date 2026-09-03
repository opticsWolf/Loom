//! Navette — unified optical thin-film engine (pure Rust, no Python).
//!
//! Umbrella over the five engine cores. Rust consumers depend on this
//! single crate; the Python bindings in `navette-py` build on it.
//!
//! ```rust,no_run
//! // S-matrix core through the umbrella:
//! let _ = navette::smatrix::core_engine::REQ_RS;
//! ```

pub use navette_color as color;
pub use navette_interpolate as interpolate;
pub use navette_materials as materials;
pub use navette_smatrix as smatrix;
pub use navette_spectralweave as spectralweave;
