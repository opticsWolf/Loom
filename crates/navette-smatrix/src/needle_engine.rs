// func_needle_engine.rs
//
// Request-driven, rayon-parallel Python API for the analytic needle operator.
// Mirrors the conventions of func_4's `core_engine`: one Rust entry point
// evaluates every requested observable per (wavelength, angle) point — with
// the coherent-block partial matrices built ONCE per point and shared across
// observables (the "single-sweep" property) — then returns a dict of ndarrays.
//
// What runs is decided entirely by the request bitmask (mirror these in a
// Python `NeedleRequest(IntFlag)`):
//   * NREQ_P     — coherent merit gradient P(z)      (sub-block confined)
//   * NREQ_P_MB  — incoherent-aware P(z)             (Modes A/B, needs flags)
//   * NREQ_DPHI / NREQ_DGD / NREQ_DGDD / NREQ_DTOD / NREQ_DFOD
//                — phase-dispersion sensitivities    (∂φ/∂δ … ∂FOD/∂δ)
//
// Polarization branches are NOT inputs; they are resolved from `calc_s` /
// `calc_p` flags so both polarizations can ride one parallel sweep.
//
// Merit convention: P uses targets/weights per spectral point,
// P_point = 2·w·(R − R_target)·Re{conj(r)·∂r/∂δ} accumulated over points into
// each z. Dispersion channels are emitted RAW (per point, per z); aggregate
// merit gradients at the call site:
//   ∂F/∂δ(z) = Σ_k 2·w_k·(GDD_k − GDD_t_k)·dGDD[k][z]


pub const C_NM_PER_FS: f64 = 299.792458;


// ─── Request bits ────────────────────────────────────────────────────────────
pub const NREQ_P: u64 = 1 << 0; // coherent P(z)
pub const NREQ_P_MB: u64 = 1 << 1; // multiblock P(z) through intensity cascade
pub const NREQ_DPHI: u64 = 1 << 2; // ∂φ/∂δ
pub const NREQ_DGD: u64 = 1 << 3; // ∂GD/∂δ
pub const NREQ_DGDD: u64 = 1 << 4; // ∂GDD/∂δ
pub const NREQ_DTOD: u64 = 1 << 5; // ∂TOD/∂δ
pub const NREQ_DFOD: u64 = 1 << 6; // ∂FOD/∂δ

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

