//! Request-driven needle sensitivities over the analytic needle operator.
//!
//! Pure-Rust core (no Python): mirrors the conventions of `core_engine` —
//! one entry point evaluates every requested observable per (wavelength,
//! angle) point, with the coherent-block partial matrices built ONCE per
//! point and shared across observables (the "single-sweep" property).
//! The rayon/Python API lives in the `navette-py` aggregator crate.
//!
//! What runs is decided entirely by the request bitmask (mirror these in a
//! Python `NeedleRequest(IntFlag)`):
//!   * NREQ_P     — coherent merit gradient P(z)      (sub-block confined)
//!   * NREQ_P_MB  — incoherent-aware P(z)             (Modes A/B, needs flags)
//!   * NREQ_DPHI / NREQ_DGD / NREQ_DGDD / NREQ_DTOD / NREQ_DFOD
//!                — phase-dispersion sensitivities    (∂φ/∂δ … ∂FOD/∂δ)
//!
//! Polarization branches are NOT inputs; they are resolved from `calc_s` /
//! `calc_p` flags so both polarizations can ride one parallel sweep.
//!
//! Merit convention: P uses targets/weights per spectral point,
//! P_point = 2·w·(R − R_target)·Re{conj(r)·∂r/∂δ} accumulated over points into
//! each z. Dispersion channels are emitted RAW (per point, per z); aggregate
//! merit gradients at the call site:
//!   ∂F/∂δ(z) = Σ_k 2·w_k·(GDD_k − GDD_t_k)·dGDD[k][z]


/// Speed of light in nm/fs (group-delay conversions).
pub const C_NM_PER_FS: f64 = 299.792458;


// ─── Request bits ────────────────────────────────────────────────────────────
/// Request coherent absorptance profile P(z).
pub const NREQ_P: u64 = 1 << 0;
/// Request multiblock absorptance profile through the intensity cascade.
pub const NREQ_P_MB: u64 = 1 << 1;
/// Request phase sensitivity to optical-path perturbation (dispersion-order needle channel).
pub const NREQ_DPHI: u64 = 1 << 2;
/// Request group-delay sensitivity (dispersion-order needle channel).
pub const NREQ_DGD: u64 = 1 << 3;
/// Request GDD sensitivity (dispersion-order needle channel).
pub const NREQ_DGDD: u64 = 1 << 4;
/// Request TOD sensitivity (dispersion-order needle channel).
pub const NREQ_DTOD: u64 = 1 << 5;
/// Request FOD sensitivity (dispersion-order needle channel).
pub const NREQ_DFOD: u64 = 1 << 6;
/// Request coherent transmission-merit gradient P_T(z) (t_fwd + flux).
pub const NREQ_P_T: u64 = 1 << 7;
/// Request coherent absorption-merit gradient P_A(z) (A = 1 − R − T).
pub const NREQ_P_A: u64 = 1 << 8;
/// Request coherent phase-merit gradient P_PHI(z) for the selected channel.
pub const NREQ_P_PHI: u64 = 1 << 9;
/// Request multiblock transmission-merit gradient Pmb_T(z).
pub const NREQ_P_MB_T: u64 = 1 << 10;
/// Request multiblock absorption-merit gradient Pmb_A(z).
pub const NREQ_P_MB_A: u64 = 1 << 11;
/// Request coherent back-transmission-merit gradient P_TB(z) (t_back + flux).
pub const NREQ_P_TB: u64 = 1 << 12;
/// Request coherent back-reflection-merit gradient P_RB(z).
pub const NREQ_P_RB: u64 = 1 << 13;
/// Request coherent back-absorption-merit gradient P_AB(z).
pub const NREQ_P_AB: u64 = 1 << 14;
/// Request multiblock back-transmission-merit gradient Pmb_TB(z).
pub const NREQ_P_MB_TB: u64 = 1 << 15;
/// Request multiblock back-reflection-merit gradient Pmb_RB(z).
pub const NREQ_P_MB_RB: u64 = 1 << 16;
/// Request multiblock back-absorption-merit gradient Pmb_AB(z).
pub const NREQ_P_MB_AB: u64 = 1 << 17;

/// Highest dispersion derivative order implied by the request mask.
pub fn max_disp_order(requested: u64) -> Option<usize> {
    let orders = [
        (NREQ_DPHI, 0usize),
        (NREQ_DGD, 1),
        (NREQ_DGDD, 2),
        (NREQ_DTOD, 3),
        (NREQ_DFOD, 4),
    ];
    orders
        .iter()
        .filter(|(bit, _)| requested & bit != 0)
        .map(|(_, order)| *order)
        .max()
}

