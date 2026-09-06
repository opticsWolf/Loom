//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::merit — MeritSpec: flat optimization targets + residual kernel.
//!
//! Bridge between navette_spectralweave's TargetWeaver and the synthesis
//! loop. The residual math below is lifted VERBATIM from
//! `calculate_merit` (navette_spectralweave/src/targetweaver.rs):
//!   * Exact/Above/Below/Range/CenterBand activation (`kind`)
//!   * Linear/Log/Phase/Complex sim-side transforms (`transform`)
//!   * bit-exact aligned-grid fast path + two-pointer monotone interpolation
//!   * overlap-skip and missing-key penalty semantics
//!
//! Residual space is the square root of merit space: `merit()` sums `r²`,
//! so the Range/CenterBand arms return `0`/`(ad-bw)/tol` and `d/bw` /
//! `sqrt(((ad-bw)/tol)² + 1)` respectively — the `+1` under the root is the
//! band-edge continuity level from `calculate_merit`.
//! Normalization itself is NOT re-implemented: it happened at ingestion in
//! TargetWeaver::register_metadata, and the Python converter copies the
//! finished (normalized_targets, norm_factor, floored tolerances) over.
//!
//! Differences from calculate_merit (by design):
//!   * No OpticalWeaver: simulation curves arrive as SimCurves — plain
//!     [n_angles, n_wav] row-major slices indexed by (pol, channel).
//!   * Entries are pre-flattened into (key-group, target) lists; the
//!     missing-curve penalty is applied ONCE PER KEY GROUP, matching the
//!     per-key `continue` in calculate_merit.
//!   * Angle rows resolved with argmin(|angles − key_angle|), identical to
//!     `_collect_target_angles` + row selection in needle_synthesis.py.

use std::sync::Arc;

use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Vocabulary types
// ---------------------------------------------------------------------------

/// Polarization vocabulary shared with `_RESULT_KEY_MAP`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pol {
    S,
    P,
    U,
}

/// Spectral channel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Channel {
    R,
    T,
}

/// Identifies one simulated curve (mirrors `_RESULT_KEY_MAP`), one derived
/// absorptance demand, or one back-incidence demand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CurveId {
    Rs,
    Rp,
    Ru,
    Ts,
    Tp,
    Tu,
    /// Absorptance demand (s): derived as A = 1 − Rs − Ts, never stored.
    As,
    /// Absorptance demand (p): derived as A = 1 − Rp − Tp, never stored.
    Ap,
    /// Absorptance demand (unpolarized): A = 1 − Ru − Tu, never stored.
    Au,
    /// Back-reflectance (s): back-incidence experiment, `back` rows.
    RBs,
    /// Back-reflectance (p).
    RBp,
    /// Back-reflectance (unpolarized).
    RBu,
    /// Back-transmittance (s).
    TBs,
    /// Back-transmittance (p).
    TBp,
    /// Back-transmittance (unpolarized).
    TBu,
    /// Back-absorptance (s): A = 1 − RBs − TBs, never stored.
    ABs,
    /// Back-absorptance (p).
    ABp,
    /// Back-absorptance (unpolarized).
    ABu,
}

impl CurveId {
    pub const ALL: [CurveId; 18] = [
        CurveId::Rs,
        CurveId::Rp,
        CurveId::Ru,
        CurveId::Ts,
        CurveId::Tp,
        CurveId::Tu,
        CurveId::As,
        CurveId::Ap,
        CurveId::Au,
        CurveId::RBs,
        CurveId::RBp,
        CurveId::RBu,
        CurveId::TBs,
        CurveId::TBp,
        CurveId::TBu,
        CurveId::ABs,
        CurveId::ABp,
        CurveId::ABu,
    ];

    pub fn new(pol: Pol, channel: Channel) -> Self {
        match (pol, channel) {
            (Pol::S, Channel::R) => CurveId::Rs,
            (Pol::P, Channel::R) => CurveId::Rp,
            (Pol::U, Channel::R) => CurveId::Ru,
            (Pol::S, Channel::T) => CurveId::Ts,
            (Pol::P, Channel::T) => CurveId::Tp,
            (Pol::U, Channel::T) => CurveId::Tu,
        }
    }

    /// True for the derived absorptance demands (`As`/`Ap`/`Au`/`ABs`/`ABp`/`ABu`).
    pub fn is_absorption(self) -> bool {
        matches!(self,
            CurveId::As | CurveId::Ap | CurveId::Au |
            CurveId::ABs | CurveId::ABp | CurveId::ABu)
    }

    /// True for back-incidence demands (`RB*`/`TB*`/`AB*`).
    pub fn is_back(self) -> bool {
        matches!(self,
            CurveId::RBs | CurveId::RBp | CurveId::RBu |
            CurveId::TBs | CurveId::TBp | CurveId::TBu |
            CurveId::ABs | CurveId::ABp | CurveId::ABu)
    }

    /// Companion intensity curves an absorption demand derives from.
    /// `None` for plain simulated curves.
    pub fn absorption_companions(self) -> Option<(CurveId, CurveId)> {
        match self {
            CurveId::As => Some((CurveId::Rs, CurveId::Ts)),
            CurveId::Ap => Some((CurveId::Rp, CurveId::Tp)),
            CurveId::Au => Some((CurveId::Ru, CurveId::Tu)),
            CurveId::ABs => Some((CurveId::RBs, CurveId::TBs)),
            CurveId::ABp => Some((CurveId::RBp, CurveId::TBp)),
            CurveId::ABu => Some((CurveId::RBu, CurveId::TBu)),
            _ => None,
        }
    }

    /// Index into `SimCurves::curves`
    /// ([Rs, Rp, Ru, Ts, Tp, Tu, As, Ap, Au]; absorption slots stay `None`).
    pub fn index(self) -> usize {
        match self {
            CurveId::Rs => 0,
            CurveId::Rp => 1,
            CurveId::Ru => 2,
            CurveId::Ts => 3,
            CurveId::Tp => 4,
            CurveId::Tu => 5,
            CurveId::As => 6,
            CurveId::Ap => 7,
            CurveId::Au => 8,
            _ => panic!("back-incidence curves live in SimCurves::back (see back_index)"),
        }
    }

    /// Index into `SimCurves::back` ([RBs, RBp, RBu, TBs, TBp, TBu]).
    /// `None` for front demands (see `index`) and absorption demands.
    pub fn back_index(self) -> Option<usize> {
        match self {
            CurveId::RBs => Some(0),
            CurveId::RBp => Some(1),
            CurveId::RBu => Some(2),
            CurveId::TBs => Some(3),
            CurveId::TBp => Some(4),
            CurveId::TBu => Some(5),
            _ => None,
        }
    }

    /// S-matrix element a phase demand on this curve tracks: front R → 0,
    /// front T → 2, back R → 3, back T → 1. `None` for absorption demands
    /// (phase of absorption is meaningless) and unpolarized keys
    /// (phase of averaged intensities is ill-defined).
    pub fn phase_channel(self) -> Option<usize> {
        match self {
            CurveId::Rs | CurveId::Rp => Some(0),
            CurveId::Ts | CurveId::Tp => Some(2),
            CurveId::RBs | CurveId::RBp => Some(3),
            CurveId::TBs | CurveId::TBp => Some(1),
            _ => None,
        }
    }

    /// Parse curve codes: front `Rs/Rp/Ru/Ts/Tp/Tu`, absorption
    /// `As/Ap/Au`, back `RBs/RBp/RBu/TBs/TBp/TBu` and `ABs/ABp/ABu`.
    /// None for anything else.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Rs" => Some(CurveId::Rs),
            "Rp" => Some(CurveId::Rp),
            "Ru" => Some(CurveId::Ru),
            "Ts" => Some(CurveId::Ts),
            "Tp" => Some(CurveId::Tp),
            "Tu" => Some(CurveId::Tu),
            "As" => Some(CurveId::As),
            "Ap" => Some(CurveId::Ap),
            "Au" => Some(CurveId::Au),
            "RBs" => Some(CurveId::RBs),
            "RBp" => Some(CurveId::RBp),
            "RBu" => Some(CurveId::RBu),
            "TBs" => Some(CurveId::TBs),
            "TBp" => Some(CurveId::TBp),
            "TBu" => Some(CurveId::TBu),
            "ABs" => Some(CurveId::ABs),
            "ABp" => Some(CurveId::ABp),
            "ABu" => Some(CurveId::ABu),
            _ => None,
        }
    }
}

/// Constraint activation — mirrors spectralweave `TargetKind`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConstraintKind {
    /// Always active.
    Exact,
    /// Active only while sim is BELOW target (driving it up).
    Above,
    /// Active only while sim is ABOVE target (driving it down).
    Below,
    /// Hard box of half-width `band`: zero inside, exceedance outside
    /// (paired `Above`/`Below` at centre∓band; bare band falls back to tol).
    Range,
    /// Soft box of half-width `band`: reduced `(d/band)` inside,
    /// exceedance plus continuity level outside (bare band falls back to exact).
    /// NOTE for the needle fold: the `+1` level is gradient-free and dropped
    /// there by design (see `docs/spectralweave-target-kinds.md`).
    CenterBand,
}

impl ConstraintKind {
    /// Parse `"e"/"a"/"b"/"r"/"c"` (spectralweave `TargetKind` codes);
    /// None for anything else.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "e" => Some(ConstraintKind::Exact),
            "a" => Some(ConstraintKind::Above),
            "b" => Some(ConstraintKind::Below),
            "r" => Some(ConstraintKind::Range),
            "c" => Some(ConstraintKind::CenterBand),
            _ => None,
        }
    }
}

/// Sim-side transform — mirrors spectralweave `ResolvedNormMode`.
///
/// Linear-mode folding note: `(nf·(sim − t)/tol)² ≡ ((sim − t)/(tol/nf))²`,
/// so purely-linear specs may equivalently store Linear + original targets
/// + eff_tol. We keep the spectralweave representation verbatim instead so
/// the parity test compares like-for-like.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimTransform {
    Linear,
    Log,
    Phase,
    Complex,
}

impl SimTransform {
    /// Parse `"linear"/"log"/"phase"/"complex"`; None for anything else.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "linear" => Some(SimTransform::Linear),
            "log" => Some(SimTransform::Log),
            "phase" => Some(SimTransform::Phase),
            "complex" => Some(SimTransform::Complex),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation curves
// ---------------------------------------------------------------------------

/// Simulated R/T curves on the fixed solver grid.
///
/// Each curve is row-major [n_angles, n_wavs]; `angles` / `wavelengths`
/// describe the axes. Arc slices make this cheap to share across rayon
/// workers during Jacobian assembly.
///
/// `total_d` / `n_front_re` / `n_back_re` describe the stack for
/// differential-phase (`PDts`/`PDtp`) demands: total coating thickness
/// (same units as `wavelengths`) and the real incidence/exit indices.
/// Defaults (0/1/1) zero the reference, i.e. differential ≡ absolute.
#[derive(Clone, Debug)]
pub struct SimCurves {
    pub angles: Arc<[f64]>,
    pub wavelengths: Arc<[f64]>,
    pub total_d: f64,
    pub n_front_re: f64,
    pub n_back_re: f64,
    /// Order: [Rs, Rp, Ru, Ts, Tp, Tu, As, Ap, Au]. Absorption slots stay
    /// `None`: absorptance is derived from the companion R/T curves.
    pub curves: [Option<Arc<[f64]>>; 9],
    /// Back-incidence intensity rows: [RBs, RBp, RBu, TBs, TBp, TBu].
    /// Back-absorptance derives from back companions; no slots of its own.
    pub back: [Option<Arc<[f64]>>; 6],
    /// Complex amplitudes for phase demands, front R/T curves only
    /// (same indexing as `curves`; row-major [n_angles, n_wavs]).
    pub cplx: [Option<Arc<[Complex64]>>; 6],
    /// Complex back amplitudes for back-phase demands: [RBs, RBp, TBs, TBp]
    /// (no unpolarized complex rows — back-phase needs s/p keys).
    pub cplx_back: [Option<Arc<[Complex64]>>; 4],
}

impl Default for SimCurves {
    fn default() -> Self {
        Self {
            angles: Arc::from(Vec::new()),
            wavelengths: Arc::from(Vec::new()),
            total_d: 0.0,
            n_front_re: 1.0,
            n_back_re: 1.0,
            curves: Default::default(),
            back: Default::default(),
            cplx: Default::default(),
            cplx_back: Default::default(),
        }
    }
}

impl SimCurves {
    pub fn curve(&self, id: CurveId) -> Option<&Arc<[f64]>> {
        // Back-incidence rows live in `back` (see back_curve); absorption
        // slots stay None (derived from companions).
        match id.back_index() {
            Some(_) => None,
            None => self.curves.get(id.index()).and_then(|c| c.as_ref()),
        }
    }

    /// Back-incidence intensity row, if supplied.
    pub fn back_curve(&self, id: CurveId) -> Option<&Arc<[f64]>> {
        id.back_index()
            .and_then(|i| self.back.get(i))
            .and_then(|c| c.as_ref())
    }

    /// Complex-amplitude row for phase demands (front R/T s/p curves).
    pub fn complex_curve(&self, id: CurveId) -> Option<&Arc<[Complex64]>> {
        if id.is_absorption() || id.is_back() {
            return None;
        }
        match id {
            CurveId::Ru | CurveId::Tu => None, // unpolarized phase is ill-defined
            _ => self.cplx.get(id.index()).and_then(|c| c.as_ref()),
        }
    }

    /// Complex back-amplitude row for back-phase demands (s/p only).
    pub fn complex_back_curve(&self, id: CurveId) -> Option<&Arc<[Complex64]>> {
        let i = match id {
            CurveId::RBs => 0,
            CurveId::RBp => 1,
            CurveId::TBs => 2,
            CurveId::TBp => 3,
            _ => return None,
        };
        self.cplx_back.get(i).and_then(|c| c.as_ref())
    }

    /// Row index for an angle value: argmin(|angles − a|), first minimum
    /// wins — identical to numpy argmin semantics used by
    /// `_simulate_to_weaver`.
    pub fn angle_row(&self, a: f64) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, &ang) in self.angles.iter().enumerate() {
            let d = (ang - a).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// MeritSpec
// ---------------------------------------------------------------------------

/// One residual from a scaled diff (signed forms), shared by the pointwise
/// path and the integral mean path (which calls it once with mean
/// diff/tol/band). Extracted verbatim — formulas bit-identical.
fn kind_residual(kind: ConstraintKind, scaled_diff: f64, tol: f64, bw: f64) -> f64 {
    match kind {
        ConstraintKind::Exact => scaled_diff / tol,
        ConstraintKind::Above if scaled_diff < 0.0 => scaled_diff / tol,
        ConstraintKind::Below if scaled_diff > 0.0 => scaled_diff / tol,
        ConstraintKind::Range => {
            // Bare `r` without a band falls back to the tolerance
            // as half-width (paired a/b at centre∓tol).
            let bw_eff = if bw <= 0.0 { tol } else { bw };
            let ad = scaled_diff.abs();
            if ad <= bw_eff { 0.0 } else { (ad - bw_eff) / tol }
        },
        ConstraintKind::CenterBand => {
            if bw <= 0.0 {
                scaled_diff / tol
            } else {
                let ad = scaled_diff.abs();
                if ad <= bw { scaled_diff / bw }
                else { (((ad - bw) / tol).powi(2) + 1.0).sqrt() }
            }
        },
        _ => 0.0,
    }
}

/// One unique (angle, curve) demand — mirrors an `OpticalKey`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeritKey {
    pub angle: f64,
    pub curve: CurveId,
}

/// One target frame flattened onto its own wavelength grid — mirrors a
/// `(frame uid, OpticalKey) → TargetEntry` pair in spectralweave.
#[derive(Clone, Debug)]
pub struct MeritTarget {
    /// Index into `MeritSpec::keys`.
    pub key_idx: u32,
    /// This entry's target wavelength grid (may be any sub/superset of the
    /// solver grid; two-pointer interpolation handles the rest).
    pub wavelengths: Arc<[f64]>,
    pub kind: ConstraintKind,
    pub transform: SimTransform,
    pub norm_factor: f64,
    /// Post-normalization targets from TargetWeaver.
    pub normalized_targets: Arc<[f64]>,
    /// Floored tolerances copied verbatim from the TargetEntry.
    pub tolerances: Arc<[f64]>,
    /// Scaled band half-widths for `Range`/`CenterBand`, copied verbatim
    /// from the TargetEntry (empty means all-zero = unused).
    pub band: Arc<[f64]>,
    /// Phase demand: sample `arg()` of the complex row for the key curve's
    /// element (see `CurveId::phase_channel`) instead of intensities.
    /// Absorption keys with `phase` are rejected at registration.
    pub phase: bool,
    /// Differential phase (`PDts`/`PDtp`): subtract `passes × reference_phase`
    /// (equivalent incidence-medium layer of `SimCurves::total_d`) from the
    /// sampled `arg()` before wrapping. `None` = absolute phase. Requires
    /// `phase` (rejected otherwise); `passes` is 1 for transmitted, 2 for a
    /// reflection round trip; must be finite and non-negative.
    pub differential_passes: Option<f64>,
    /// User weight (default 1): multiplies this frame's merit sum.
    /// Applied at the residual level (`r × √(weight/count)`) so values,
    /// LM Jacobians and the needle fold stay consistent by construction.
    pub weight: f64,
    /// Count-normalization divisor (default None = off): the frame's sum
    /// is divided by this (target-level point count — the converter
    /// resolves it, since angular targets span many single-point frames).
    /// `Some(n)` needs `n > 0` (rejected otherwise).
    pub count_norm: Option<f64>,
    /// Integral target: constrain the MEAN of the scaled diffs (single
    /// residual `R = mean(d)/mean(tol)`), not each point. Kinds apply once
    /// to the mean. Rejected with `count_norm` (the mean already is one).
    pub integral: bool,
}

/// Flat, immutable, Send+Sync target description.
#[derive(Clone, Debug, Default)]
pub struct MeritSpec {
    keys: Vec<MeritKey>,
    targets: Vec<MeritTarget>,
}

impl MeritSpec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a key group; returns its index.
    pub fn add_key(&mut self, key: MeritKey) -> usize {
        self.keys.push(key);
        self.keys.len() - 1
    }

    /// Append one target frame to key group `key_idx`.
    ///
    /// Wavelengths/targets/tolerances must have equal length; `band` must
    /// either be empty (all-zero = unused) or match that length.
    pub fn add_target(&mut self, target: MeritTarget) -> Result<(), String> {
        let n = target.wavelengths.len();
        if target.normalized_targets.len() != n || target.tolerances.len() != n {
            return Err(format!(
                "MeritTarget length mismatch: wl={} targets={} tols={}",
                n,
                target.normalized_targets.len(),
                target.tolerances.len()
            ));
        }
        if !target.band.is_empty() && target.band.len() != n {
            return Err(format!(
                "MeritTarget band mismatch: wl={} band={}",
                n,
                target.band.len()
            ));
        }
        if target.key_idx as usize >= self.keys.len() {
            return Err(format!(
                "key_idx {} out of range ({} keys registered)",
                target.key_idx,
                self.keys.len()
            ));
        }
        if target.phase && self.keys[target.key_idx as usize].curve.phase_channel().is_none() {
            return Err(format!(
                "phase demand on {:?}: absorption/unpolarized curves have no phase",
                self.keys[target.key_idx as usize].curve
            ));
        }
        // Mirror invariant (see spectralweave `register_metadata`): the phase
        // arm scales nothing, so phase demands must carry raw values with
        // norm_factor == 1 (converters: divide the resolved triple by nf).
        if target.transform == SimTransform::Phase
            && (target.norm_factor - 1.0).abs() > 1e-12
        {
            return Err(format!(
                "phase transform needs norm_factor == 1 (got {}); pass raw values",
                target.norm_factor
            ));
        }
        // Differential phase is phase-only (subtracting a propagation
        // reference from an intensity is meaningless) with a sane pass count.
        if let Some(passes) = target.differential_passes {
            if !target.phase {
                return Err("differential_passes without phase: PD demands are phase-only".into());
            }
            if !(passes >= 0.0) || !passes.is_finite() {
                return Err(format!("differential passes must be finite and >= 0 (got {passes})"));
            }
        }
        // Weight/count trust boundary (bindings + converter pass user
        // values straight through; garbage here means NaN merits).
        if !target.weight.is_finite() || target.weight < 0.0 {
            return Err(format!(
                "weight must be finite and >= 0 (got {})", target.weight
            ));
        }
        if let Some(n) = target.count_norm {
            if !n.is_finite() || n <= 0.0 {
                return Err(format!("count_norm must be finite and > 0 (got {n})"));
            }
        }
        // Integral targets already are means — a count divisor would
        // double-dilute silently.
        if target.integral && target.count_norm.is_some() {
            return Err("integral targets reject count_norm (the mean already is one)".into());
        }
        self.targets.push(target);
        Ok(())
    }

    /// True if any demand samples complex amplitudes (phase or
    /// differential-phase). The evaluator uses this to skip complex-row
    /// assembly for intensity-only specs — virtually-free values are always
    /// built, requested ones only on demand.
    pub fn uses_phase(&self) -> bool {
        self.targets.iter().any(|t| t.phase)
    }

    /// True if any demand subtracts the equivalent-medium reference.
    /// Gates the (trivial but non-zero) stack-metadata computation.
    pub fn uses_differential(&self) -> bool {
        self.targets.iter().any(|t| t.differential_passes.is_some())
    }

    pub fn keys(&self) -> &[MeritKey] {
        &self.keys
    }

    pub fn targets(&self) -> &[MeritTarget] {
        &self.targets
    }

    /// Total number of residual components (all target points, active or
    /// not — inactive ones contribute zeros so the residual vector length
    /// stays fixed for the optimizer).
    pub fn n_residuals(&self) -> usize {
        self.targets.iter().map(|t| t.wavelengths.len()).sum()
    }

    /// Scalar merit function: Σ residual² + missing-key penalties.
    ///
    /// Semantics identical to `calculate_merit(sim, tw, missing_penalty)`:
    /// a missing curve costs `missing_penalty` ONCE per key group, and
    /// target grids that do not overlap the simulated grid are skipped
    /// silently (zero contribution).
    pub fn merit(&self, sim: &SimCurves, missing_penalty: f64) -> f64 {
        let mut total = 0.0;
        let mut buf: Vec<f64> = Vec::new();
        for k in 0..self.keys.len() {
            buf.clear();
            match self.residuals_into(sim, k, &mut buf) {
                Ok(()) => total += buf.iter().map(|r| r * r).sum::<f64>(),
                Err(_) => total += missing_penalty,
            }
        }
        total
    }

    /// Fixed-length residual vector (zeros where constraints are inactive).
    ///
    /// Returns `Err(missing CurveId)` if a demanded curve was not supplied.
    /// Component order: keys in registration order, targets per key in
    /// insertion order, points along each target grid — deterministic,
    /// which the thickness optimizer relies on.
    pub fn residuals(&self, sim: &SimCurves, out: &mut Vec<f64>) -> Result<(), CurveId> {
        out.clear();
        out.reserve(self.n_residuals());
        for k in 0..self.keys.len() {
            self.residuals_into(sim, k, out)?;
        }
        Ok(())
    }

    // -- internals -----------------------------------------------------------

    /// Residuals for every target belonging to one key group.
    ///
    /// Inner loop is a verbatim lift of the calculate_merit body:
    /// overlap skip → aligned fast path / two-pointer interpolation →
    /// mode transform → kind activation. Absorption demands (`As`/`Ap`)
    /// derive A = 1 − R − T from the companion curves on the shared grid;
    /// a missing companion fails the whole key group like a missing curve.
    fn residuals_into(
        &self,
        sim: &SimCurves,
        key_idx: usize,
        out: &mut Vec<f64>,
    ) -> Result<(), CurveId> {
        let key = &self.keys[key_idx];
        let ang_row = sim.angle_row(key.angle);
        let sim_wl: &[f64] = &sim.wavelengths;
        let n_wav = sim_wl.len();
        // One intensity row, either side (front `curves` or `back`).
        let irow = |id: CurveId| -> Result<&[f64], CurveId> {
            let arc = if id.is_back() { sim.back_curve(id) } else { sim.curve(id) };
            arc.map(|c| &c[ang_row * n_wav..(ang_row + 1) * n_wav])
                .ok_or(key.curve)
        };
        for t in self.targets.iter().filter(|t| t.key_idx as usize == key_idx) {
            // Resolve this target's simulated input BEFORE pushing anything,
            // so missing rows leave `out` untouched. Phase demands sample
            // arg() of the complex row for the key's element.
            enum TargetInput<'a> {
                Intensity(&'a [f64]),
                Absorption(&'a [f64], &'a [f64]),
                Phase(&'a [Complex64]),
            }
            let input: TargetInput = if t.phase {
                let crow = if key.curve.is_back() {
                    sim.complex_back_curve(key.curve)
                } else {
                    sim.complex_curve(key.curve)
                };
                match crow {
                    Some(c) => TargetInput::Phase(&c[ang_row * n_wav..(ang_row + 1) * n_wav]),
                    None => return Err(key.curve),
                }
            } else if let Some((rc, tc)) = key.curve.absorption_companions() {
                TargetInput::Absorption(irow(rc)?, irow(tc)?)
            } else {
                TargetInput::Intensity(irow(key.curve)?)
            };
            let t_wl: &[f64] = &t.wavelengths;
            if t_wl.is_empty() {
                continue;
            }
            // Skip frames whose grid does not overlap the simulated curve.
            if sim_wl.last().map_or(true, |&l| l < t_wl[0])
                || sim_wl.first().zip(t_wl.last()).map_or(true, |(&f, &l)| f > l)
            {
                continue;
            }

            // Fast path: when the target grid coincides bit-for-bit with a
            // contiguous block of the simulated grid, read simulated values
            // directly — no interpolation, no per-point division.
            let offset = sim_wl.partition_point(|&x| x < t_wl[0]);
            let aligned = offset + t_wl.len() <= n_wav
                && t_wl
                    .iter()
                    .zip(&sim_wl[offset..offset + t_wl.len()])
                    .all(|(&a, &b)| a.to_bits() == b.to_bits());

            // Two-pointer state advances monotonically across the sorted
            // target grid (O(n + m), never reset inside this entry). The
            // sampler is shared by both companion rows (identical grids),
            // so absorption stays consistent point-for-point.
            let mut sim_idx = 0usize;
            let sample = |row: &[f64], i: usize, sim_idx: &mut usize| -> f64 {
                if aligned {
                    return row[offset + i];
                }
                let target_w = t_wl[i];
                while *sim_idx + 1 < n_wav && sim_wl[*sim_idx + 1] < target_w {
                    *sim_idx += 1;
                }
                if *sim_idx + 1 < n_wav && sim_wl[*sim_idx] <= target_w {
                    let w0 = sim_wl[*sim_idx];
                    let w1 = sim_wl[*sim_idx + 1];
                    let v0 = row[*sim_idx];
                    let v1 = row[*sim_idx + 1];
                    if (w1 - w0).abs() < 1e-14 {
                        v0
                    } else {
                        v0 + (target_w - w0) * (v1 - v0) / (w1 - w0)
                    }
                } else if *sim_idx < n_wav {
                    row[*sim_idx]
                } else {
                    row[n_wav - 1]
                }
            };
            // Complex twin for phase demands (shared two-pointer state is
            // safe: identical grids take identical paths).
            let sample_c = |row: &[Complex64], i: usize, sim_idx: &mut usize| -> Complex64 {
                if aligned {
                    return row[offset + i];
                }
                let target_w = t_wl[i];
                while *sim_idx + 1 < n_wav && sim_wl[*sim_idx + 1] < target_w {
                    *sim_idx += 1;
                }
                if *sim_idx + 1 < n_wav && sim_wl[*sim_idx] <= target_w {
                    let w0 = sim_wl[*sim_idx];
                    let w1 = sim_wl[*sim_idx + 1];
                    let v0 = row[*sim_idx];
                    let v1 = row[*sim_idx + 1];
                    if (w1 - w0).abs() < 1e-14 {
                        v0
                    } else {
                        v0 + (v1 - v0) * ((target_w - w0) / (w1 - w0))
                    }
                } else if *sim_idx < n_wav {
                    row[*sim_idx]
                } else {
                    row[n_wav - 1]
                }
            };
            // Differential-phase reference for this key (front/back medium
            // + total thickness from the sim metadata; None = absolute).
            // `key.angle` is degrees (converter convention); the reference
            // helper converts internally. The medium lookup stays INSIDE
            // the `if let` below so non-differential keys pay nothing.
            let diff_passes = t.differential_passes;
            // Weight + count normalization at the residual level (once per
            // frame): merit scales by weight/count exactly, and LM
            // Jacobians differentiate the scaled residuals consistently.
            // Defaults (1.0/None) are the identity — legacy paths bit-safe.
            let rscale = (t.weight / t.count_norm.unwrap_or(1.0)).sqrt();
            let mut acc_d = 0.0;
            let mut acc_tol = 0.0;
            let mut acc_bw = 0.0;
            for i in 0..t_wl.len() {
                let sim_raw = match &input {
                    TargetInput::Intensity(row) => sample(row, i, &mut sim_idx),
                    TargetInput::Absorption(r, tt) => {
                        1.0 - sample(r, i, &mut sim_idx) - sample(tt, i, &mut sim_idx)
                    },
                    TargetInput::Phase(crow) => {
                        let mut a = sample_c(crow, i, &mut sim_idx).arg();
                        if let Some(passes) = diff_passes {
                            let n_inc = if key.curve.is_back() {
                                sim.n_back_re
                            } else {
                                sim.n_front_re
                            };
                            a -= crate::smatrix::optics_core::reference_phase(
                                t_wl[i],
                                n_inc,
                                key.angle,
                                sim.total_d,
                                passes,
                            );
                        }
                        a
                    },
                };

                let target_scaled = t.normalized_targets[i];

                let scaled_diff = match t.transform {
                    SimTransform::Phase => {
                        // Wrap the residual to [-pi, pi] without trig — cheaper
                        // than the equivalent sin/cos/atan2 formulation.
                        let diff = sim_raw - target_scaled;
                        diff - std::f64::consts::TAU * (diff / std::f64::consts::TAU).round()
                    }
                    SimTransform::Log => {
                        sim_raw.max(1e-12).log10() * t.norm_factor - target_scaled
                    }
                    SimTransform::Linear | SimTransform::Complex => {
                        sim_raw * t.norm_factor - target_scaled
                    }
                };

                let tol = t.tolerances[i];
                let bw = t.band.get(i).copied().unwrap_or(0.0);

                if t.integral {
                    // Mean branch: accumulate raw ingredients; the kind
                    // applies ONCE to the mean after the loop.
                    acc_d += scaled_diff;
                    acc_tol += tol;
                    acc_bw += bw;
                } else {
                    out.push(kind_residual(t.kind, scaled_diff, tol, bw) * rscale);
                }
            }
            if t.integral {
                // Single mean residual R = mean(d)/mean(tol); kinds
                // constrain the MEAN (integral-`a` = lower bound on the
                // average). Weight multiplies (count rejected at intake).
                let n = t_wl.len() as f64;
                let tol_bar = (acc_tol / n).max(1e-300);
                out.push(kind_residual(t.kind, acc_d / n, tol_bar, acc_bw / n) * rscale);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests — pin the lifted kernel against hand-computed values
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NW: usize = 5;

    /// Sim grid 400..800 step 100 nm, single angle row, R_s supplied.
    fn sim_one_angle(vals: &[f64; NW]) -> SimCurves {
        SimCurves {
            angles: vec![0.0].into(),
            wavelengths: (0..NW).map(|i| 400.0 + 100.0 * i as f64).collect::<Vec<_>>().into(),
            curves: [Some(Arc::from(vals.to_vec())), None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            ..Default::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn entry(
        key_idx: u32,
        wl: Vec<f64>,
        targets: Vec<f64>,
        tols: Vec<f64>,
        kind: ConstraintKind,
        transform: SimTransform,
        norm_factor: f64,
    ) -> MeritTarget {
        entry_banded(key_idx, wl, targets, tols, vec![], kind, transform, norm_factor)
    }

    fn entry_banded(
        key_idx: u32,
        wl: Vec<f64>,
        targets: Vec<f64>,
        tols: Vec<f64>,
        band: Vec<f64>,
        kind: ConstraintKind,
        transform: SimTransform,
        norm_factor: f64,
    ) -> MeritTarget {
        MeritTarget {
            key_idx,
            wavelengths: wl.into(),
            kind,
            transform,
            norm_factor,
            normalized_targets: targets.into(),
            tolerances: tols.into(),
            band: band.into(),
            phase: false,
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
        }
    }

    fn entry_phase(
        key_idx: u32,
        wl: Vec<f64>,
        targets: Vec<f64>,
        tols: Vec<f64>,
        kind: ConstraintKind,
        transform: SimTransform,
        norm_factor: f64,
    ) -> MeritTarget {
        MeritTarget {
            key_idx,
            wavelengths: wl.into(),
            kind,
            transform,
            norm_factor,
            normalized_targets: targets.into(),
            tolerances: tols.into(),
            band: Vec::new().into(),
            phase: true,
            differential_passes: None,
            integral: false,
            weight: 1.0,
            count_norm: None,
        }
    }

    #[test]
    fn linear_fold_exact_zero() {
        // targets [0.5, 1.0]: avg = 0.75 → nf = 4/3 (register_metadata math)
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        let nf = 1.0 / 0.75;
        spec.add_target(entry(
            k as u32,
            vec![400.0, 500.0],
            vec![0.5 * nf, 1.0 * nf],
            vec![0.01, 0.01],
            ConstraintKind::Exact,
            SimTransform::Linear,
            nf,
        ))
        .unwrap();

        let sim = sim_one_angle(&[0.5, 1.0, 0.9, 0.8, 0.7]);
        assert_eq!(spec.merit(&sim, 1e6), 0.0);
    }

    #[test]
    fn exact_residual_hand_computed() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry(
            k as u32,
            vec![400.0],
            vec![0.5], // already-normalized target
            vec![0.1], // tol
            ConstraintKind::Exact,
            SimTransform::Linear,
            1.0,
        ))
        .unwrap();
        let sim = sim_one_angle(&[0.55, 0., 0., 0., 0.]);
        // r = (0.55 − 0.5)/0.1 = 0.5 → merit 0.25
        assert!((spec.merit(&sim, 1e6) - 0.25).abs() < 1e-14);
    }

    #[test]
    fn above_below_activation() {
        // Above: active only while sim < target; Below: mirror image.
        let mk = |kind| {
            let mut s = MeritSpec::new();
            let k = s.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
            s.add_target(entry(
                k as u32,
                vec![400.0],
                vec![0.5],
                vec![0.1],
                kind,
                SimTransform::Linear,
                1.0,
            ))
            .unwrap();
            s
        };
        let above = mk(ConstraintKind::Above);
        let below = mk(ConstraintKind::Below);

        let sim_hi = sim_one_angle(&[0.6, 0., 0., 0., 0.]); // sim > target
        let sim_lo = sim_one_angle(&[0.3, 0., 0., 0., 0.]); // sim < target

        assert_eq!(above.merit(&sim_hi, 0.0), 0.0); // satisfied → inactive
        assert!((above.merit(&sim_lo, 0.0) - 4.0).abs() < 1e-14); // ((0.3−0.5)/0.1)²
        assert_eq!(below.merit(&sim_lo, 0.0), 0.0); // satisfied → inactive
        assert!((below.merit(&sim_hi, 0.0) - 1.0).abs() < 1e-14); // ((0.6−0.5)/0.1)²
    }

    #[test]
    fn residual_vector_zeros_inactive_but_fixed_length() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry(
            k as u32,
            vec![400.0, 500.0],
            vec![0.5, 0.5],
            vec![0.1, 0.1],
            ConstraintKind::Above,
            SimTransform::Linear,
            1.0,
        ))
        .unwrap();
        let sim = sim_one_angle(&[0.6, 0.3, 0., 0., 0.]);
        let mut out = Vec::new();
        spec.residuals(&sim, &mut out).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 0.0); // Above satisfied at 400
        assert!((out[1] - (-2.0)).abs() < 1e-14); // active at 500
        assert_eq!(spec.n_residuals(), 2);
    }

    #[test]
    fn log_transform_matches_register_metadata() {
        // Log mode: nf = 1/avg(|log10 t|); scaled diff = log10(sim)·nf − tgt·nf
        let targets = [0.01_f64, 1.0];
        let log_sum: f64 = targets.iter().map(|v| v.max(1e-12).log10().abs()).sum();
        let avg = log_sum / 2.0;
        let nf = 1.0 / avg.max(1e-12);

        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry(
            k as u32,
            vec![400.0],
            vec![targets[0].max(1e-12).log10() * nf],
            vec![0.05],
            ConstraintKind::Exact,
            SimTransform::Log,
            nf,
        ))
        .unwrap();
        // sim == target exactly → zero residual
        let sim = sim_one_angle(&[0.01, 0., 0., 0., 0.]);
        assert!(spec.merit(&sim, 0.0).abs() < 1e-14);
        // sim off by ×10 in raw space → log10 diff = 1 · nf / tol
        let sim10 = sim_one_angle(&[0.1, 0., 0., 0., 0.]);
        let expect = (1.0 * nf / 0.05).powi(2);
        assert!((spec.merit(&sim10, 0.0) - expect).abs() < 1e-9 * expect);
    }

    #[test]
    fn phase_wrap_no_trig() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry(
            k as u32,
            vec![400.0],
            vec![0.0],
            vec![1.0],
            ConstraintKind::Exact,
            SimTransform::Phase,
            1.0,
        ))
        .unwrap();
        // sim = π + 0.1 → wrapped diff −(π − 0.1)
        let sim = sim_one_angle(&[std::f64::consts::PI + 0.1, 0., 0., 0., 0.]);
        let expect = (std::f64::consts::PI - 0.1).powi(2);
        assert!((spec.merit(&sim, 0.0) - expect).abs() < 1e-12);
    }

    #[test]
    fn misaligned_interpolation_two_pointer() {
        // Coarse target grid over a fine sim ramp — linear interp hits exactly.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry(
            k as u32,
            vec![450.0, 550.0],
            vec![0.5, 1.5],
            vec![1.0, 1.0],
            ConstraintKind::Exact,
            SimTransform::Linear,
            1.0,
        ))
        .unwrap();
        let sim = sim_one_angle(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(spec.merit(&sim, 0.0).abs() < 1e-14);
    }

    #[test]
    fn extrapolation_clamps_and_overlap_skips() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        // Below sim range → overlap holds (sim covers [400,800]) → clamps to first val 0.3.
        spec.add_target(entry(
            k as u32,
            vec![350.0],
            vec![0.3],
            vec![1.0],
            ConstraintKind::Exact,
            SimTransform::Linear,
            1.0,
        ))
        .unwrap();
        // Above sim range entirely (900 > 800) → skipped by overlap rule,
        // but still occupies its residual slot.
        spec.add_target(entry(
            k as u32,
            vec![900.0],
            vec![0.0],
            vec![1.0],
            ConstraintKind::Exact,
            SimTransform::Linear,
            1.0,
        ))
        .unwrap();

        let sim = sim_one_angle(&[0.3, 0., 0., 0., 0.]);
        assert_eq!(spec.merit(&sim, 0.0), 0.0);
        assert_eq!(spec.n_residuals(), 2);
    }

    #[test]
    fn missing_curve_penalty_once_per_key() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Tp }); // not supplied
        spec.add_target(entry(k as u32, vec![400.0], vec![0.0], vec![1.0],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        spec.add_target(entry(k as u32, vec![500.0], vec![0.0], vec![1.0],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let k2 = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs }); // supplied
        spec.add_target(entry(k2 as u32, vec![400.0], vec![0.0], vec![1.0],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();

        let sim = sim_one_angle(&[0.0, 0., 0., 0., 0.]);
        // penalty once for the Tp group + zero from Rs
        assert_eq!(spec.merit(&sim, 123.0), 123.0);
        assert!(matches!(spec.residuals(&sim, &mut Vec::new()), Err(CurveId::Tp)));
    }

    #[test]
    fn angle_row_argmin_semantics() {
        let mut sim = SimCurves {
            angles: vec![0.0, 30.0, 60.0].into(),
            wavelengths: vec![500.0].into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            ..Default::default()
        };
        sim.curves[CurveId::Ru.index()] = Some(Arc::from(vec![10.0, 20.0, 30.0]));

        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 25.0, curve: CurveId::Ru }); // → row 30°
        spec.add_target(entry(k as u32, vec![500.0], vec![20.0], vec![0.5],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();

        let mut out = Vec::new();
        spec.residuals(&sim, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].abs() < 1e-14); // picked the 30° row value 20
    }

    #[test]
    fn aligned_fast_path_agrees_with_interp_path() {
        let vals = [0.31, 0.52, 0.66, 0.71, 0.90];

        // Aligned: target wl exactly on the sim grid (bit-equal).
        let mut spec_a = MeritSpec::new();
        let ka = spec_a.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec_a.add_target(entry(ka as u32, vec![500.0], vec![0.42], vec![0.07],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();

        let sim = sim_one_angle(&vals);
        let mut out = Vec::new();
        spec_a.residuals(&sim, &mut out).unwrap();
        let r_aligned = out[0];

        // Interp: shift the sim grid by tiny per-point amounts so no value is
        // bit-equal, then request the same physical wavelength.
        let shifted: SimCurves = SimCurves {
            angles: sim.angles.clone(),
            wavelengths: (0..NW)
                .map(|i| 400.0 + 100.0 * i as f64 + 1e-11 * i as f64)
                .collect::<Vec<_>>()
                .into(),
            curves: sim.curves.clone(),
            back: sim.back.clone(),
            cplx: sim.cplx.clone(),
            cplx_back: sim.cplx_back.clone(),
            ..Default::default()
        };
        let mut spec_u = MeritSpec::new();
        let ku = spec_u.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec_u.add_target(entry(ku as u32, vec![500.0 + 1e-11], vec![0.42], vec![0.07],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();

        let mut out2 = Vec::new();
        spec_u.residuals(&shifted, &mut out2).unwrap();
        let r_interp = out2[0];

        assert!((r_aligned - r_interp).abs() < 1e-9);
    }

    #[test]
    fn constraint_kind_from_str() {
        assert_eq!(ConstraintKind::from_str("e"), Some(ConstraintKind::Exact));
        assert_eq!(ConstraintKind::from_str("a"), Some(ConstraintKind::Above));
        assert_eq!(ConstraintKind::from_str("b"), Some(ConstraintKind::Below));
        assert_eq!(ConstraintKind::from_str("r"), Some(ConstraintKind::Range));
        assert_eq!(ConstraintKind::from_str("c"), Some(ConstraintKind::CenterBand));
        assert_eq!(ConstraintKind::from_str("x"), None);
    }

    #[test]
    fn range_box_dead_band() {
        // target 0.5 (nf=1), tol 0.1, band 0.05.
        let mk = |band: Vec<f64>| {
            let mut s = MeritSpec::new();
            let k = s.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
            s.add_target(entry_banded(k as u32, vec![400.0], vec![0.5], vec![0.1],
                band, ConstraintKind::Range, SimTransform::Linear, 1.0)).unwrap();
            s
        };
        let spec = mk(vec![0.05]);
        assert_eq!(spec.merit(&sim_one_angle(&[0.5, 0., 0., 0., 0.]), 0.0), 0.0);
        assert_eq!(spec.merit(&sim_one_angle(&[0.53, 0., 0., 0., 0.]), 0.0), 0.0); // d=0.03 in band
        // d=0.1 → ((0.1−0.05)/0.1)² = 0.25
        assert!((spec.merit(&sim_one_angle(&[0.6, 0., 0., 0., 0.]), 0.0) - 0.25).abs() < 1e-14);
        // Bare band falls back to tol as half-width: d=0.05 inside → 0.
        let bare = mk(vec![]);
        assert_eq!(bare.merit(&sim_one_angle(&[0.55, 0., 0., 0., 0.]), 0.0), 0.0);
        // d=0.2 → ((0.2−0.1)/0.1)² = 1.
        assert!((bare.merit(&sim_one_angle(&[0.7, 0., 0., 0., 0.]), 0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn centerband_soft_box() {
        // target 0.5 (nf=1), tol 0.1, band 0.05.
        let mk = |band: Vec<f64>| {
            let mut s = MeritSpec::new();
            let k = s.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
            s.add_target(entry_banded(k as u32, vec![400.0], vec![0.5], vec![0.1],
                band, ConstraintKind::CenterBand, SimTransform::Linear, 1.0)).unwrap();
            s
        };
        let spec = mk(vec![0.05]);
        assert_eq!(spec.merit(&sim_one_angle(&[0.5, 0., 0., 0., 0.]), 0.0), 0.0);
        // Inside: (0.02/0.05)² = 0.16.
        assert!((spec.merit(&sim_one_angle(&[0.52, 0., 0., 0., 0.]), 0.0) - 0.16).abs() < 1e-14);
        // At the edge: 1.0 from both sides (continuity).
        assert!((spec.merit(&sim_one_angle(&[0.55, 0., 0., 0., 0.]), 0.0) - 1.0).abs() < 1e-14);
        // Outside: ((0.05)/0.1)² + 1 = 1.25.
        assert!((spec.merit(&sim_one_angle(&[0.6, 0., 0., 0., 0.]), 0.0) - 1.25).abs() < 1e-14);
        // Bare band degrades to exact: (0.1/0.1)² = 1.
        let bare = mk(vec![]);
        assert!((bare.merit(&sim_one_angle(&[0.6, 0., 0., 0., 0.]), 0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn merit_two_keys_no_double_count() {
        // Regression: merit() reused its scratch buffer across keys,
        // over-counting every key after the first.
        let mut spec = MeritSpec::new();
        let k0 = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry(k0 as u32, vec![400.0], vec![0.5], vec![0.1],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let k1 = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rp });
        spec.add_target(entry(k1 as u32, vec![400.0], vec![0.5], vec![0.1],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let mut sim = sim_one_angle(&[0.6, 0., 0., 0., 0.]);
        sim.curves[CurveId::Rp.index()] = Some(Arc::from(vec![0.6, 0., 0., 0., 0.]));
        // Each key: ((0.6−0.5)/0.1)² = 1 → total 2 (was 3 with stale buffer).
        assert!((spec.merit(&sim, 1e6) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn absorption_derived_from_companions() {
        // R row 0.6, T row 0.3 → A = 0.1; demand A = 0.1, tol 0.05 → 0.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::As });
        spec.add_target(entry(k as u32, vec![400.0], vec![0.1], vec![0.05],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: (0..NW).map(|i| 400.0 + 100.0 * i as f64).collect::<Vec<_>>().into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            ..Default::default()
        };
        sim.curves[CurveId::Rs.index()] = Some(Arc::from(vec![0.6, 0., 0., 0., 0.]));
        sim.curves[CurveId::Ts.index()] = Some(Arc::from(vec![0.3, 0., 0., 0., 0.]));
        assert!(spec.merit(&sim, 1e6) < 1e-28); // A = 1−0.6−0.3 ≈ 0.1
        let mut out = Vec::new();
        spec.residuals(&sim, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].abs() < 1e-14);
        // Missing companion → penalty once, residuals Err on the demand.
        sim.curves[CurveId::Ts.index()] = None;
        assert_eq!(spec.merit(&sim, 123.0), 123.0);
        assert!(matches!(spec.residuals(&sim, &mut Vec::new()), Err(CurveId::As)));
        // Unpolarized demand derives from the Ru/Tu companions the same way.
        let mut spec_u = MeritSpec::new();
        let ku = spec_u.add_key(MeritKey { angle: 0.0, curve: CurveId::Au });
        spec_u.add_target(entry(ku as u32, vec![400.0], vec![0.2], vec![0.1],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        sim.curves[CurveId::Ru.index()] = Some(Arc::from(vec![0.5, 0., 0., 0., 0.]));
        sim.curves[CurveId::Tu.index()] = Some(Arc::from(vec![0.3, 0., 0., 0., 0.]));
        assert!(spec_u.merit(&sim, 1e6) < 1e-28); // A = 1−0.5−0.3 ≈ 0.2
        sim.curves[CurveId::Ru.index()] = None;
        assert!(matches!(spec_u.residuals(&sim, &mut Vec::new()), Err(CurveId::Au)));
    }

    #[test]
    fn phase_demand_samples_argument() {
        // Complex row 0.5·e^{i·0.3}; demand phase 0.3 (nf=1) → zero.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry_phase(k as u32, vec![400.0], vec![0.3], vec![0.05],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: (0..NW).map(|i| 400.0 + 100.0 * i as f64).collect::<Vec<_>>().into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            ..Default::default()
        };
        sim.cplx[0] = Some(Arc::from(vec![
            Complex64::from_polar(0.5, 0.3), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]));
        assert!(spec.merit(&sim, 1e6) < 1e-28);
        // Missing complex row → penalty once, Err on the demand.
        sim.cplx[0] = None;
        assert_eq!(spec.merit(&sim, 123.0), 123.0);
        assert!(matches!(spec.residuals(&sim, &mut Vec::new()), Err(CurveId::Rs)));
    }

    #[test]
    fn phase_demand_wraps_in_phase_mode() {
        // Sim phase 0.3 + 2π − 0.01 vs target 0.3: wrapped diff −0.01.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        spec.add_target(entry_phase(k as u32, vec![400.0], vec![0.3], vec![0.05],
            ConstraintKind::Exact, SimTransform::Phase, 1.0)).unwrap();
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: (0..NW).map(|i| 400.0 + 100.0 * i as f64).collect::<Vec<_>>().into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            ..Default::default()
        };
        sim.cplx[3] = Some(Arc::from(vec![
            Complex64::from_polar(0.7, 0.3 - 0.01 + std::f64::consts::TAU),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]));
        // (−0.01/0.05)² = 0.04
        assert!((spec.merit(&sim, 1e6) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn phase_on_absorption_rejected() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::As });
        let mut tgt = entry_phase(k as u32, vec![400.0], vec![0.0], vec![0.1],
            ConstraintKind::Exact, SimTransform::Linear, 1.0);
        tgt.key_idx = k as u32;
        assert!(spec.add_target(tgt).is_err());
        let k2 = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ru });
        let tgt2 = entry_phase(k2 as u32, vec![400.0], vec![0.0], vec![0.1],
            ConstraintKind::Exact, SimTransform::Linear, 1.0);
        assert!(spec.add_target(tgt2).is_err());
    }

    /// Differential-phase entry: absolute-phase entry + `passes`.
    fn entry_pd(
        key_idx: u32,
        wl: Vec<f64>,
        targets: Vec<f64>,
        tols: Vec<f64>,
        kind: ConstraintKind,
        passes: f64,
    ) -> MeritTarget {
        let mut t = entry_phase(key_idx, wl, targets, tols, kind, SimTransform::Phase, 1.0);
        t.differential_passes = Some(passes);
        t
    }

    fn sim_pd() -> SimCurves {
        // One angle (0°), NW wavelengths from 400 nm; Ts complex row
        // 0.7·e^{i·0.3} at 400 nm; stack D = 100 nm of air (n = 1).
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: (0..NW).map(|i| 400.0 + 100.0 * i as f64).collect::<Vec<_>>().into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            total_d: 100.0,
            n_front_re: 1.0,
            n_back_re: 1.0,
        };
        sim.cplx[3] = Some(Arc::from(vec![
            Complex64::from_polar(0.7, 0.3), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)]));
        sim
    }

    #[test]
    fn differential_phase_subtracts_reference() {
        // λ = 400, D = 100, n = 1, θ = 0: ref = 2π·100/400 = π/2 ≈ 1.5707963.
        // Δφ = 0.3 − π/2; demanding exactly that with tol 0.05 → zero.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        let delta = 0.3 - std::f64::consts::PI / 2.0;
        spec.add_target(entry_pd(k as u32, vec![400.0], vec![delta], vec![0.05],
            ConstraintKind::Exact, 1.0)).unwrap();
        let sim = sim_pd();
        assert!(spec.merit(&sim, 1e6) < 1e-28);
        // Same demand as absolute (passes path off): residual is −π/2 →
        // (−π/2/0.05)² ≈ 986.96 — the reference is doing the work.
        let mut abs_spec = MeritSpec::new();
        let ka = abs_spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        abs_spec.add_target(entry_phase(ka as u32, vec![400.0], vec![delta], vec![0.05],
            ConstraintKind::Exact, SimTransform::Phase, 1.0)).unwrap();
        let m_abs = abs_spec.merit(&sim, 1e6);
        let expect = ((0.3 - delta) / 0.05).powi(2);
        assert!((m_abs - expect).abs() < 1e-9, "m_abs={m_abs} expect={expect}");
    }

    #[test]
    fn differential_phase_zero_d_is_absolute() {
        // D = 0 kills the reference: differential ≡ absolute bit-for-bit.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        spec.add_target(entry_pd(k as u32, vec![400.0], vec![0.3], vec![0.05],
            ConstraintKind::Exact, 1.0)).unwrap();
        let mut sim = sim_pd();
        sim.total_d = 0.0;
        assert!(spec.merit(&sim, 1e6) < 1e-28);
    }

    #[test]
    fn differential_phase_passes_scale() {
        // passes = 2 doubles the subtracted reference (round-trip).
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        let delta = 0.3 - std::f64::consts::PI; // 2 × π/2
        spec.add_target(entry_pd(k as u32, vec![400.0], vec![delta], vec![0.05],
            ConstraintKind::Exact, 2.0)).unwrap();
        assert!(spec.merit(&sim_pd(), 1e6) < 1e-28);
    }

    #[test]
    fn differential_validation() {
        // Without phase → Err; negative/NaN passes → Err.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Ts });
        let mut t = entry(k as u32, vec![400.0], vec![0.0], vec![0.1],
            ConstraintKind::Exact, SimTransform::Linear, 1.0);
        t.differential_passes = Some(1.0);
        assert!(spec.add_target(t).is_err());
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let mut t2 = entry_pd(k as u32, vec![400.0], vec![0.0], vec![0.1],
                ConstraintKind::Exact, bad);
            t2.differential_passes = Some(bad);
            assert!(spec.add_target(t2).is_err(), "passes={bad}");
        }
    }

    #[test]
    fn weight_and_count_scale_merit() {
        // Base: two Exact points, nf = 1, tol 0.1, sim 0.1 off →
        // r = ±1/point → merit 2.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry(k as u32, vec![400.0, 500.0], vec![0.5, 0.5],
            vec![0.1, 0.1], ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let mut sim = sim_one_angle(&[0.6, 0.4, 0.0, 0.0, 0.0]);
        assert!((spec.merit(&sim, 1e6) - 2.0).abs() < 1e-12);
        // weight 2 → merit 4 (residuals scale by √2).
        let mut sw = MeritSpec::new();
        let kw = sw.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        let mut tw = entry(kw as u32, vec![400.0, 500.0], vec![0.5, 0.5],
            vec![0.1, 0.1], ConstraintKind::Exact, SimTransform::Linear, 1.0);
        tw.weight = 2.0;
        sw.add_target(tw).unwrap();
        assert!((sw.merit(&sim, 1e6) - 4.0).abs() < 1e-12);
        let mut out = Vec::new();
        sw.residuals(&sim, &mut out).unwrap();
        assert!((out[0].abs() - 2.0f64.sqrt()).abs() < 1e-12);
        // count 2 → merit 1 (mean, not sum).
        let mut sc = MeritSpec::new();
        let kc = sc.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        let mut tc = entry(kc as u32, vec![400.0, 500.0], vec![0.5, 0.5],
            vec![0.1, 0.1], ConstraintKind::Exact, SimTransform::Linear, 1.0);
        tc.count_norm = Some(2.0);
        sc.add_target(tc).unwrap();
        assert!((sc.merit(&sim, 1e6) - 1.0).abs() < 1e-12);
        // weight 3 + count 2 → 3.
        let mut sb = MeritSpec::new();
        let kb = sb.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        let mut tb = entry(kb as u32, vec![400.0, 500.0], vec![0.5, 0.5],
            vec![0.1, 0.1], ConstraintKind::Exact, SimTransform::Linear, 1.0);
        tb.weight = 3.0;
        tb.count_norm = Some(2.0);
        sb.add_target(tb).unwrap();
        assert!((sb.merit(&sim, 1e6) - 3.0).abs() < 1e-12);
        // Trust boundary: negative/NaN weight, non-positive count rejected.
        for (w, c) in [(-1.0, None), (f64::NAN, None), (1.0, Some(0.0)), (1.0, Some(-2.0))] {
            let mut sx = MeritSpec::new();
            let kx = sx.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
            let mut tx = entry(kx as u32, vec![400.0], vec![0.5],
                vec![0.1], ConstraintKind::Exact, SimTransform::Linear, 1.0);
            tx.weight = w;
            tx.count_norm = c;
            assert!(sx.add_target(tx).is_err(), "w={w} c={c:?}");
        }
        let _ = &mut sim;
    }

    fn entry_integral(key_idx: u32, kind: ConstraintKind) -> MeritTarget {
        let mut t = entry(key_idx, vec![400.0, 500.0, 600.0], vec![0.5, 0.5, 0.5],
            vec![0.1, 0.1, 0.1], kind, SimTransform::Linear, 1.0);
        t.integral = true;
        t
    }

    #[test]
    fn integral_mean_single_residual() {
        // Sim [0.6, 0.5, 0.4] vs 0.5: mean diff 0 → merit 0, ONE residual.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry_integral(k as u32, ConstraintKind::Exact)).unwrap();
        let sim = sim_one_angle(&[0.6, 0.5, 0.4, 0.0, 0.0]);
        assert!(spec.merit(&sim, 1e6) < 1e-28);
        let mut out = Vec::new();
        spec.residuals(&sim, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        // Sim [0.7, 0.6, 0.5]: mean diff 0.1, tol 0.1 → R = 1 → merit 1.
        let sim2 = sim_one_angle(&[0.7, 0.6, 0.5, 0.0, 0.0]);
        assert!((spec.merit(&sim2, 1e6) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn integral_kinds_mask_the_mean() {
        // Above: mean above target → silent; mean below → active.
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        spec.add_target(entry_integral(k as u32, ConstraintKind::Above)).unwrap();
        // Mean 0.6 ≥ 0.5 → 0 (even though point 600 dips to 0.4!).
        let sim_hi = sim_one_angle(&[0.7, 0.7, 0.4, 0.0, 0.0]);
        assert!(spec.merit(&sim_hi, 1e6) < 1e-28);
        // Mean 0.4 < 0.5 → ((0.4−0.5)/0.1)² = 1.
        let sim_lo = sim_one_angle(&[0.4, 0.4, 0.4, 0.0, 0.0]);
        assert!((spec.merit(&sim_lo, 1e6) - 1.0).abs() < 1e-12);
        // Range with band 0.05 (raw, nf = 1): mean inside → 0.
        let mut spec_r = MeritSpec::new();
        let kr = spec_r.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        let mut tr = entry_integral(kr as u32, ConstraintKind::Range);
        tr.band = vec![0.05, 0.05, 0.05].into();
        spec_r.add_target(tr).unwrap();
        let sim_in = sim_one_angle(&[0.54, 0.54, 0.54, 0.0, 0.0]);
        assert!(spec_r.merit(&sim_in, 1e6) < 1e-28);
        // Mean 0.6: exceedance (0.1−0.05)/0.1 = 0.5 → 0.25.
        let sim_out = sim_one_angle(&[0.6, 0.6, 0.6, 0.0, 0.0]);
        assert!((spec_r.merit(&sim_out, 1e6) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn integral_rejects_count_norm() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        let mut t = entry_integral(k as u32, ConstraintKind::Exact);
        t.count_norm = Some(3.0);
        assert!(spec.add_target(t).is_err());
    }

    #[test]
    fn back_intensity_and_absorption() {
        // RBs row 0.4, TBs row 0.5: R demand 0.4 → 0; ABs demand 0.1 → 0.
        let mut spec = MeritSpec::new();
        let kr = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::RBs });
        spec.add_target(entry(kr as u32, vec![400.0], vec![0.4], vec![0.05],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let ka = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::ABs });
        spec.add_target(entry(ka as u32, vec![400.0], vec![0.1], vec![0.05],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).unwrap();
        let mut sim = SimCurves {
            angles: vec![0.0].into(),
            wavelengths: (0..NW).map(|i| 400.0 + 100.0 * i as f64).collect::<Vec<_>>().into(),
            curves: [None, None, None, None, None, None, None, None, None],
            back: [None, None, None, None, None, None],
            cplx: [None, None, None, None, None, None],
            cplx_back: [None, None, None, None],
            ..Default::default()
        };
        sim.back[0] = Some(Arc::from(vec![0.4, 0., 0., 0., 0.]));
        sim.back[3] = Some(Arc::from(vec![0.5, 0., 0., 0., 0.]));
        assert!(spec.merit(&sim, 1e6) < 1e-28); // 0 + (1−0.4−0.5−0.1)²
    }

    #[test]
    fn band_length_mismatch_rejected() {
        let mut spec = MeritSpec::new();
        let k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        assert!(spec.add_target(entry_banded(k as u32, vec![400.0], vec![0.0], vec![1.0],
            vec![0.1, 0.2], ConstraintKind::Range, SimTransform::Linear, 1.0)).is_err());
    }

    #[test]
    fn length_mismatch_and_bad_key_rejected() {
        let mut spec = MeritSpec::new();
        let _k = spec.add_key(MeritKey { angle: 0.0, curve: CurveId::Rs });
        assert!(spec.add_target(entry(0, vec![400.0], vec![0.0, 0.0], vec![1.0],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).is_err());
        assert!(spec.add_target(entry(7, vec![400.0], vec![0.0], vec![1.0],
            ConstraintKind::Exact, SimTransform::Linear, 1.0)).is_err());
    }
}
