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

/// Identifies one simulated curve (mirrors `_RESULT_KEY_MAP`) or one
/// derived absorptance demand.
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
}

impl CurveId {
    pub const ALL: [CurveId; 8] = [
        CurveId::Rs,
        CurveId::Rp,
        CurveId::Ru,
        CurveId::Ts,
        CurveId::Tp,
        CurveId::Tu,
        CurveId::As,
        CurveId::Ap,
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

    /// True for the derived absorptance demands (`As`/`Ap`).
    pub fn is_absorption(self) -> bool {
        matches!(self, CurveId::As | CurveId::Ap)
    }

    /// Companion intensity curves an absorption demand derives from.
    /// `None` for plain simulated curves.
    pub fn absorption_companions(self) -> Option<(CurveId, CurveId)> {
        match self {
            CurveId::As => Some((CurveId::Rs, CurveId::Ts)),
            CurveId::Ap => Some((CurveId::Rp, CurveId::Tp)),
            _ => None,
        }
    }

    /// Index into `SimCurves::curves`
    /// ([Rs, Rp, Ru, Ts, Tp, Tu, As, Ap]; absorption slots stay `None`).
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

// ---------------------------------------------------------------------------
// Simulation curves
// ---------------------------------------------------------------------------

/// Simulated R/T curves on the fixed solver grid.
///
/// Each curve is row-major [n_angles, n_wavs]; `angles` / `wavelengths`
/// describe the axes. Arc slices make this cheap to share across rayon
/// workers during Jacobian assembly.
#[derive(Clone, Debug, Default)]
pub struct SimCurves {
    pub angles: Arc<[f64]>,
    pub wavelengths: Arc<[f64]>,
    /// Order: [Rs, Rp, Ru, Ts, Tp, Tu, As, Ap]. Absorption slots stay
    /// `None`: absorptance is derived from the companion R/T curves.
    pub curves: [Option<Arc<[f64]>>; 8],
}

impl SimCurves {
    pub fn curve(&self, id: CurveId) -> Option<&Arc<[f64]>> {
        self.curves[id.index()].as_ref()
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
        self.targets.push(target);
        Ok(())
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
        // Resolve simulated rows BEFORE pushing anything, so a missing
        // companion leaves `out` untouched.
        let row_r: &[f64];
        let mut row_t: Option<&[f64]> = None;
        match key.curve.absorption_companions() {
            Some((rc, tc)) => {
                let (Some(r), Some(t)) = (sim.curve(rc), sim.curve(tc)) else {
                    return Err(key.curve);
                };
                row_r = &r[ang_row * n_wav..(ang_row + 1) * n_wav];
                row_t = Some(&t[ang_row * n_wav..(ang_row + 1) * n_wav]);
            },
            None => {
                let Some(curve) = sim.curve(key.curve) else {
                    return Err(key.curve);
                };
                row_r = &curve[ang_row * n_wav..(ang_row + 1) * n_wav];
            },
        }
        for t in self.targets.iter().filter(|t| t.key_idx as usize == key_idx) {
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
            for i in 0..t_wl.len() {
                let sim_raw = match row_t {
                    Some(rt) => 1.0 - sample(row_r, i, &mut sim_idx) - sample(rt, i, &mut sim_idx),
                    None => sample(row_r, i, &mut sim_idx),
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

                let r = match t.kind {
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
                };
                out.push(r);
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
            curves: [Some(Arc::from(vals.to_vec())), None, None, None, None, None, None, None],
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
            curves: [None, None, None, None, None, None, None, None],
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
            curves: [None, None, None, None, None, None, None, None],
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
