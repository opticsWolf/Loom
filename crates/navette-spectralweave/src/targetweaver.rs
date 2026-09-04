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

    /// Pre-calculates normalizations and transforms targets upon ingestion.
    /// `band` holds raw-unit half-widths for the `r`/`c` kinds (empty or
    /// all-zero when unused); it is scaled by the same `norm_factor` as the
    /// targets (per-point exact mapping in log mode, first-order otherwise).
    pub fn register_metadata(&self, uid: usize, key: OpticalKey, raw_targets: &[f64], tolerances: &[f64], kind: TargetKind, mode_str: &str, band: &[f64]) {
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

        let norm_factor: f64;
        let mut normalized_targets = Vec::with_capacity(raw_targets.len());

        match resolved_mode {
            ResolvedNormMode::Phase | ResolvedNormMode::Complex => {
                norm_factor = 1.0;
                normalized_targets.extend_from_slice(raw_targets);
            },
            ResolvedNormMode::Log => {
                let log_sum: f64 = raw_targets.iter().map(|&v| v.max(1e-12).log10().abs()).sum();
                let log_avg = log_sum / raw_targets.len() as f64;
                norm_factor = 1.0 / log_avg.max(1e-12);

                for &v in raw_targets {
                    normalized_targets.push(v.max(1e-12).log10() * norm_factor);
                }
            },
            ResolvedNormMode::Linear => {
                let t_avg = (t_sum / raw_targets.len() as f64).abs();
                norm_factor = 1.0 / t_avg.max(1e-12);

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
        };

        let mut meta_guard = self.target_metadata.write();
        meta_guard.entry(uid).or_default().entries.insert(key, entry);
    }
}
