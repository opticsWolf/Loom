//! Optimization targets and merit evaluation over woven simulation curves.
//!
//! A [`TargetWeaver`] owns dedicated frames holding target curves plus
//! precomputed normalization metadata ([`TargetMetadata`]); the merit
//! function compares simulated weaves against them with Exact/Above/Below
//! activation and linear/log/phase/complex normalisation modes.

use crate::opticalweaver::{OpticalKey, OpticalWeaver, SpectralDataFrame};
use parking_lot::RwLock;
use ahash::AHashMap;
use std::sync::Arc;

/// How a target activates the residual: exact, one-sided, or banded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TargetKind {
    /// Penalise any deviation (residual = sim minus target).
    Exact,
    /// Penalise only shortfall (residual = max(0, target minus sim)).
    Above,
    /// Penalise only excess (residual = max(0, sim minus target)).
    Below,
    /// Hard box of half-width `band`: zero inside, quadratic exceedance
    /// outside (equivalent to paired `a`/`b` targets at centre∓band).
    Range,
    /// Soft box of half-width `band`: reduced quadratic `(d/band)^2`
    /// inside (i.e. exact penalisation scaled by `(tol/band)^2`), quadratic
    /// exceedance plus continuity offset outside.
    CenterBand,
}

impl TargetKind {
    /// Parse "e"/"a"/"b"/"r"/"c"; None for anything else.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "e" => Some(TargetKind::Exact),
            "a" => Some(TargetKind::Above),
            "b" => Some(TargetKind::Below),
            "r" => Some(TargetKind::Range),
            "c" => Some(TargetKind::CenterBand),
            _ => None,
        }
    }

    /// Canonical code for export (converters, round-trips).
    pub fn as_str(self) -> &'static str {
        match self {
            TargetKind::Exact => "e",
            TargetKind::Above => "a",
            TargetKind::Below => "b",
            TargetKind::Range => "r",
            TargetKind::CenterBand => "c",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// Normalisation applied before the residual: raw linear values, log10
/// magnitudes (wide-dynamic-range spectra), unwrapped phase, or complex
/// (real/imag jointly).
pub enum ResolvedNormMode {
    Linear,
    Log,
    Phase,
    Complex,
}

impl ResolvedNormMode {
    /// Canonical name for export (converters, round-trips).
    pub fn as_str(self) -> &'static str {
        match self {
            ResolvedNormMode::Linear => "linear",
            ResolvedNormMode::Log => "log",
            ResolvedNormMode::Phase => "phase",
            ResolvedNormMode::Complex => "complex",
        }
    }
}

#[derive(Clone)]
/// One ingested target curve: activation kind, normalisation, the
/// pre-normalized values plus per-point tolerances, and the scaled band
/// half-widths for the `r`/`c` kinds (zeros when unused).
pub struct TargetEntry {
    pub kind: TargetKind,
    pub resolved_mode: ResolvedNormMode,
    pub norm_factor: f64,
    pub normalized_targets: Arc<[f64]>,
    pub tolerances: Arc<[f64]>,
    pub band: Arc<[f64]>,
    /// User weight (default 1): multiplies the frame's merit sum.
    /// Validated at the bindings (finite, >= 0).
    pub weight: f64,
    /// Count normalization divisor (default None = off): the frame's sum
    /// is divided by this (target-level point count — the bindings resolve
    /// it, since angular targets span many single-point entries). None
    /// keeps the legacy pure sum.
    pub count_norm: Option<f64>,
    /// Integral target: merit constrains the MEAN of the scaled diffs
    /// (single residual `R = mean(d)/mean(tol)`), not each point. Kinds
    /// apply once to the mean (integral-`a` = lower bound on the average).
    /// Rejected in combination with `count_norm` (the mean already is one).
    pub integral: bool,
}

#[derive(Default, Clone)]
/// All target entries keyed by [`OpticalKey`], stored per frame UID.
pub struct TargetMetadata {
    pub entries: AHashMap<OpticalKey, TargetEntry>,
}

/// Target store for optimization: dedicated frames plus ingestion-time
/// normalization metadata. `tolerance_floor` clamps near-zero tolerances so
/// the merit can never divide by zero.
pub struct TargetWeaver {
    pub inner: Arc<OpticalWeaver>,
    pub target_metadata: RwLock<AHashMap<usize, TargetMetadata>>, // Keyed by Frame UID
    pub tolerance_floor: f64,
}

impl TargetWeaver {
    /// Create a weaver with an LRU plan cache of `cache_size` grids and a
    /// tolerance floor for merit denominators.
    pub fn new(cache_size: usize, tolerance_floor: f64) -> Self {
        TargetWeaver {
            inner: Arc::new(OpticalWeaver::new(cache_size)),
            target_metadata: RwLock::new(AHashMap::new()),
            tolerance_floor,
        }
    }

    /// Allocate a fresh frame for one target curve (targets never share
    /// frames with simulation data) and bump the generation.
    pub fn create_dedicated_frame(&self, wl: &[f64]) -> Result<Arc<SpectralDataFrame>, String> {
        let new_frame = Arc::new(SpectralDataFrame::new(wl)?);
        self.inner.inner.frames.write().push(new_frame.clone());
        self.inner.bump_generation();
        Ok(new_frame)
    }

    /// Resolve the normalization for one curve: mode + factor, shared by
    /// every point of the curve. Spectral curves resolve per call (one
    /// curve per call); angular targets MUST resolve once over the full
    /// angle curve and share the result across points — per-point
    /// resolution would weight each angle by its own magnitude.
    pub fn resolve_norm(raw_targets: &[f64], mode_str: &str) -> (ResolvedNormMode, f64) {
        let mut t_min = f64::MAX;
        let mut t_max = f64::MIN;
        let mut t_sum = 0.0;

        for &v in raw_targets {
            if v < t_min { t_min = v; }
            if v > t_max { t_max = v; }
            t_sum += v;
        }

        let resolved_mode = match mode_str {
            "phase" => ResolvedNormMode::Phase,
            "complex" => ResolvedNormMode::Complex,
            "log" => ResolvedNormMode::Log,
            "linear" => ResolvedNormMode::Linear,
            _ => {
                // "auto" default: require strictly positive data for log, and >= 100x dynamic range
                if t_min > 0.0 && (t_max / t_min) >= 100.0 {
                    ResolvedNormMode::Log
                } else {
                    ResolvedNormMode::Linear
                }
            }
        };

        let norm_factor =
            Self::norm_factor_for(&resolved_mode, raw_targets, t_min, t_max, t_sum);
        (resolved_mode, norm_factor)
    }

    /// Normalization factor for a resolved mode (see `resolve_norm`).
    /// Linear falls back to half-range scaling on zero-mean/cancelling
    /// data and to raw scale on constant data; log falls back to raw log
    /// scale on ~= 1 data. Everywhere else this is the legacy formula.
    fn norm_factor_for(
        resolved_mode: &ResolvedNormMode,
        raw_targets: &[f64],
        t_min: f64,
        t_max: f64,
        t_sum: f64,
    ) -> f64 {
        match resolved_mode {
            ResolvedNormMode::Phase | ResolvedNormMode::Complex => 1.0,
            ResolvedNormMode::Log => {
                let n = raw_targets.len() as f64;
                let mut log_min = f64::MAX;
                let mut log_max = f64::MIN;
                let mut log_sum = 0.0;
                for &v in raw_targets {
                    let lv = v.max(1e-12).log10().abs();
                    if lv < log_min { log_min = lv; }
                    if lv > log_max { log_max = lv; }
                    log_sum += lv;
                }
                let log_avg = log_sum / n;
                let log_scale = if log_avg <= 1e-9 * (log_max - log_min).max(1e-300) {
                    1.0
                } else {
                    log_avg
                };
                1.0 / log_scale.max(1e-300)
            },
            ResolvedNormMode::Linear => {
                let t_avg = (t_sum / raw_targets.len() as f64).abs();
                let spread = t_max - t_min;
                let scale = if t_avg <= 1e-9 * spread { spread / 2.0 } else { t_avg };
                if scale > 0.0 { 1.0 / scale } else { 1.0 }
            },
        }
    }

    /// Pre-calculates normalizations and transforms targets upon ingestion.
    /// `band` holds raw-unit half-widths for the `r`/`c` kinds (empty or
    /// all-zero when unused); it is scaled by the same `norm_factor` as the
    /// targets (per-point exact mapping in log mode, first-order otherwise).
    pub fn register_metadata(&self, uid: usize, key: OpticalKey, raw_targets: &[f64], tolerances: &[f64], kind: TargetKind, mode_str: &str, band: &[f64], weight: f64, count_norm: Option<f64>, integral: bool) {
        let (resolved_mode, norm_factor) = Self::resolve_norm(raw_targets, mode_str);
        self.register_metadata_resolved(uid, key, raw_targets, tolerances, kind, resolved_mode, norm_factor, band, weight, count_norm, integral)
    }

    /// `register_metadata` with a pre-resolved `(mode, factor)` — the
    /// angular path resolves once over the full curve and shares it.
    pub fn register_metadata_resolved(&self, uid: usize, key: OpticalKey, raw_targets: &[f64], tolerances: &[f64], kind: TargetKind, resolved_mode: ResolvedNormMode, norm_factor: f64, band: &[f64], weight: f64, count_norm: Option<f64>, integral: bool) {
        // Normalization itself lives in `norm_factor_for` (shared); here we
        // only apply it. Phase/Complex resolve to nf == 1 by construction.
        let mut normalized_targets = Vec::with_capacity(raw_targets.len());
        match resolved_mode {
            ResolvedNormMode::Phase | ResolvedNormMode::Complex => {
                normalized_targets.extend_from_slice(raw_targets);
            },
            ResolvedNormMode::Log => {
                for &v in raw_targets {
                    normalized_targets.push(v.max(1e-12).log10() * norm_factor);
                }
            },
            ResolvedNormMode::Linear => {
                for &v in raw_targets {
                    normalized_targets.push(v * norm_factor);
                }
            }
        }

        let floored_tols: Vec<f64> = tolerances
            .iter()
            .map(|&t| t.max(self.tolerance_floor))
            .collect();

        // Scale the raw band half-widths into the normalized residual space.
        let band_scaled: Vec<f64> = match resolved_mode {
            ResolvedNormMode::Linear | ResolvedNormMode::Phase | ResolvedNormMode::Complex => {
                raw_targets.iter().enumerate().map(|(i, _)| {
                    band.get(i).copied().unwrap_or(0.0).max(0.0) * norm_factor
                }).collect()
            },
            ResolvedNormMode::Log => {
                raw_targets.iter().enumerate().map(|(i, &t)| {
                    let b = band.get(i).copied().unwrap_or(0.0).max(0.0);
                    if b <= 0.0 { return 0.0; }
                    let t_pos = t.max(1e-12);
                    ((t_pos + b).max(1e-12).log10() - t_pos.log10()).abs() * norm_factor
                }).collect()
            },
        };

        let entry = TargetEntry {
            kind,
            resolved_mode,
            norm_factor,
            normalized_targets: Arc::from(normalized_targets),
            tolerances: Arc::from(floored_tols),
            band: Arc::from(band_scaled),
            weight,
            count_norm,
            integral,
        };

        let mut meta_guard = self.target_metadata.write();
        meta_guard.entry(uid).or_default().entries.insert(key, entry);
    }
}
