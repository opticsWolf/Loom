//! Navette materials — pure-Rust optical dispersion core.
//!
//! No Python, no I/O. Every kernel is a free function over `ndarray` views
//! taking the wavelength in **nanometres** and returning the complex
//! refractive index n + ik as `Array1<Complex64>`. Unit conversions live in
//! [`units`]. The binding crate `navette-materials-py` wraps these for Python;
//! this crate is independently `cargo test`/`cargo bench`-able.
//!
//! Two surfaces:
//!   * free functions (e.g. [`lorentz::lorentz_nk`]) — the 1:1 kernel ports,
//!     used by the v1 Python wrapper, and
//!   * the [`Dispersion`] trait + [`Model`] enum — lets a whole material
//!     (eventually including composites) evaluate in one call, the seam for a
//!     future Rust-side fitting loop.

use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;

pub mod units;
pub mod common;
pub mod grid;

pub mod cauchy;
pub mod sellmeier;
pub mod lorentz;
pub mod drude;
pub mod ema;

// Next-phase scaffolds (documented contracts, no behaviour yet).
pub mod forouhi_bloomer;
pub mod cody_lorentz;
pub mod ubf;
pub mod tauc_lorentz;
pub mod table;
pub mod kk;

/// Shared evaluation interface: complex refractive index at wavelengths [nm].
///
/// Implementors evaluate a whole wavelength grid in one call and return
/// `n + ik` per point. Element-wise kernels parallelise over points with
/// rayon above [`crate::common::PAR_THRESHOLD`]; the mapping is
/// point-independent, so parallel and serial results are bit-identical.
pub trait Dispersion {
    /// Evaluate `n + ik` on every wavelength of `wavelength_nm` [nm].
    fn nk(&self, wavelength_nm: ArrayView1<f64>) -> Array1<Complex64>;
}

/// Mixing rule for composite (EMA) materials.
///
/// Each rule blends inclusion/host **permittivities** at volume fraction
/// `f`; see [`ema`] for the closed forms. The caller applies √ to obtain n̂.
#[derive(Clone, Copy, Debug)]
pub enum MixRule {
    /// Bruggeman symmetric-medium root (Newton–Raphson per point).
    /// `max_iter` caps iterations, `tol` is the |Δε| stop threshold.
    Bruggeman { max_iter: usize, tol: f64 },
    /// Maxwell-Garnett dilute-sphere limit (explicit, host-background).
    MaxwellGarnett,
    /// Landau–Lifshitz–Looyenga cube-root rule (good all-rounder).
    Looyenga,
    /// Lichtenecker logarithmic rule (also the α → 0 power-law limit).
    Lichtenecker,
    /// Mori–Tanaka for ellipsoidal inclusions; `l` is the depolarisation
    /// factor along the field (1/3 recovers spheres).
    MoriTanaka { l: f64 },
    /// Birchak general power law with exponent `alpha`.
    PowerLaw { alpha: f64 },
}

/// A material model. Arms delegate to the free-function kernels; composite
/// arms (added in phase 3/4) evaluate their children then mix.
#[derive(Clone, Debug)]
pub enum Model {
    /// Cauchy polynomial; `a, b, c` with λ in µm (`n = A + B/λ² + C/λ⁴`), k = 0.
    Cauchy { a: f64, b: f64, c: f64 },
    /// Cauchy `n` plus Urbach tail `k` (`alpha0` [1/cm], `eu` [eV], `lambda_g` [nm]).
    CauchyUrbach { a: f64, b: f64, c: f64, alpha0: f64, eu: f64, lambda_g: f64 },
    /// Up-to-three-term Sellmeier; `b/c` coefficient triples, λ in µm.
    Sellmeier { b: [f64; 3], c: [f64; 3] },
    /// Sellmeier `n` plus Urbach tail `k` (same tail params as above).
    SellmeierUrbach { b: [f64; 3], c: [f64; 3], alpha0: f64, eu: f64, lambda_g: f64 },
    /// Lorentz oscillators; `osc` is (N, 3) rows of (E0, Γ, f) in eV.
    Lorentz { osc: Array2<f64>, eps_inf: f64 },
    /// Drude free carriers; `omega_p` plasma energy, `gamma` damping [eV].
    Drude { omega_p: f64, gamma: f64, eps_inf: f64 },
    /// Combined Drude + Lorentz terms (metals with interband structure).
    DrudeLorentz { omega_p: f64, gamma_d: f64, eps_inf: f64, osc: Array2<f64> },
    // Effective { host, inclusion, fraction, rule } — phase 3 (see kk/cody phase 4).
}

impl Dispersion for Model {
    fn nk(&self, wl: ArrayView1<f64>) -> Array1<Complex64> {
        match self {
            Model::Cauchy { a, b, c } => cauchy::cauchy_nk(wl, *a, *b, *c),
            Model::CauchyUrbach { a, b, c, alpha0, eu, lambda_g } => {
                cauchy::cauchy_urbach_nk(wl, *a, *b, *c, *alpha0, *eu, *lambda_g)
            }
            Model::Sellmeier { b, c } => {
                sellmeier::sellmeier_nk(wl, b[0], c[0], b[1], c[1], b[2], c[2])
            }
            Model::SellmeierUrbach { b, c, alpha0, eu, lambda_g } => {
                sellmeier::sellmeier_urbach_nk(
                    wl, b[0], c[0], b[1], c[1], b[2], c[2], *alpha0, *eu, *lambda_g,
                )
            }
            Model::Lorentz { osc, eps_inf } => lorentz::lorentz_nk(wl, osc.view(), *eps_inf),
            Model::Drude { omega_p, gamma, eps_inf } => drude::drude_nk(wl, *omega_p, *gamma, *eps_inf),
            Model::DrudeLorentz { omega_p, gamma_d, eps_inf, osc } => {
                drude::drude_lorentz_nk(wl, *omega_p, *gamma_d, *eps_inf, osc.view())
            }
        }
    }
}
