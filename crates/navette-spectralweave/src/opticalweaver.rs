//! Fragment-weaving store: many overlapping spectral curves over shared grids.
//!
//! A [`SpectralDataFrame`] owns one wavelength grid plus the curves sampled on
//! it. An [`OpticalCollection`] groups frames by grid fingerprint so curves on
//! identical grids share storage, and an [`OpticalWeaver`] distributes long
//! input curves across frames (with an LRU-cached distribution plan) and
//! re-assembles continuous curves on demand. All state is `Arc` + interior
//! mutability, so weavers are cheap to clone and share across threads.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ahash::AHashMap;
use lru::LruCache;
use smallvec::SmallVec;
use parking_lot::RwLock;


// ---------------------------------------------------------------------------
// Unit system
// ---------------------------------------------------------------------------
/// Unit tag for spectral (wavelength) and intensity axes.
/// Conversions are currently identity (grids are stored in nm, data raw);
/// the tag exists so callers can be explicit and future units can convert.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unit {
    NM,
    RAW,
}

/// Convert `value` from one unit to another, borrowing when no-op.
/// Returns a borrowed view when `from == to`, otherwise an owned copy.
#[inline]
pub fn convert_unit<'a>(value: &'a [f64], from: Unit, to: Unit) -> Cow<'a, [f64]> {
    if from == to {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(value.to_vec())
    }
}

/// 128-bit fingerprint of a wavelength grid (xxh3 over the raw bits).
/// Used as the LRU/cache key so identical grids share frames and plans.
pub type WlSig = u128;

#[inline]
/// Fingerprint `wl` for frame/plan lookup.
pub fn wl_signature(wl: &[f64]) -> WlSig {
    xxhash_rust::xxh3::xxh3_128(bytemuck::cast_slice::<f64, u8>(wl))
}

#[inline]
/// Bit-exact grid equality (length + raw f64 bits, no tolerance).
/// Guards against xxh3 collisions before frames are shared.
pub fn wl_bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && bytemuck::cast_slice::<f64, u8>(a) == bytemuck::cast_slice::<f64, u8>(b)
}

#[inline]
/// Canonical display name of a unit (`"NM"` / `"RAW"`).
pub fn unit_to_str(u: Unit) -> &'static str {
    match u {
        Unit::NM => "NM",
        Unit::RAW => "RAW",
    }
}

#[inline]
/// Parse a spectral-unit label (defaults to nm).
pub fn parse_spectral(s: Option<&str>) -> Unit {
    match s {
        Some("NM") | None => Unit::NM,
        _ => Unit::NM,
    }
}

#[inline]
/// Parse an intensity-unit label (defaults to raw).
pub fn parse_intensity(s: Option<&str>) -> Unit {
    match s {
        Some("RAW") | None => Unit::RAW,
        _ => Unit::RAW,
    }
}

// ---------------------------------------------------------------------------
// Optical key
// ---------------------------------------------------------------------------
/// Identity of one curve: base wavelength plus data-type and polarisation labels.
/// Equality and hashing use the exact f64 bits of `wavelength`.
#[derive(Debug, Clone)]
pub struct OpticalKey {
    pub wavelength: f64,
    pub data_type: Arc<str>,
    pub polarisation: Arc<str>,
}

impl PartialEq for OpticalKey {
    fn eq(&self, other: &Self) -> bool {
        self.wavelength.to_bits() == other.wavelength.to_bits()
            && self.data_type == other.data_type
            && self.polarisation == other.polarisation
    }
}
impl Eq for OpticalKey {}

impl Hash for OpticalKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.wavelength.to_bits().hash(state);
        self.data_type.hash(state);
        self.polarisation.hash(state);
    }
}

impl From<(f64, String, String)> for OpticalKey {
    fn from(t: (f64, String, String)) -> Self {
        OpticalKey {
            wavelength: t.0,
            data_type: t.1.into(),
            polarisation: t.2.into(),
        }
    }
}

impl OpticalKey {
    #[inline]
    /// `(wavelength, data_type, polarisation)` triple for the Python boundary.
    pub fn as_tuple(&self) -> (f64, String, String) {
        (
            self.wavelength,
            self.data_type.to_string(),
            self.polarisation.to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// SpectralData & SpectralDataFrame
// ---------------------------------------------------------------------------
/// A (possibly strided) view onto a shared data buffer: `buf[start..start+len]`.
/// Contiguous slices can alias the source `Arc` (zero-copy); strided gathers copy.
/// Derefs to `[f64]` so callers treat it as a plain slice.
#[derive(Clone)]
pub struct SpectralData {
    buf: Arc<[f64]>,
    start: usize,
    len: usize,
}

impl SpectralData {
    #[inline]
    /// Wrap a whole owned buffer as one contiguous fragment.
    pub fn from_arc(buf: Arc<[f64]>) -> Self {
        let len = buf.len();
        SpectralData { buf, start: 0, len }
    }
}

impl std::ops::Deref for SpectralData {
    type Target = [f64];
    #[inline]
    fn deref(&self) -> &[f64] {
        &self.buf[self.start..self.start + self.len]
    }
}

/// One wavelength grid plus every curve sampled on it.
/// `uid` is process-unique; `set_data` rejects grids that differ bit-for-bit.
pub struct SpectralDataFrame {
    pub uid: usize,
    data: RwLock<AHashMap<OpticalKey, SpectralData>>,
    wavelength: Arc<[f64]>,
    wl_min: f64,
    wl_max: f64,
}

impl SpectralDataFrame {
    /// Create a frame on a strictly-increasing grid. Errors on empty or
    /// non-monotonic grids.
    pub fn new(wavelength: &[f64]) -> Result<Self, String> {
        static UID_GEN: AtomicUsize = AtomicUsize::new(0);
        let uid = UID_GEN.fetch_add(1, Ordering::SeqCst);

        if wavelength.is_empty() {
            return Err("SpectralDataFrame: wavelength array must be non-empty.".to_string());
        }
        for i in 0..wavelength.len() - 1 {
            if !(wavelength[i] < wavelength[i + 1]) {
                return Err("SpectralDataFrame: wavelength array must be strictly monotonically increasing.".to_string());
            }
        }

        let wl_min = wavelength[0];
        let wl_max = wavelength[wavelength.len() - 1];

        Ok(SpectralDataFrame {
            uid,
            data: RwLock::new(AHashMap::new()),
            wavelength: Arc::from(wavelength),
            wl_min,
            wl_max,
        })
    }

    /// Store `value` under `key`. Returns whether the key is new.
    /// Errors when `value` misaligns with the grid (or the optional grid
    /// conflicts with it).
    pub fn set_data(
        &self,
        key: OpticalKey,
        value: SpectralData,
        wavelength: Option<&[f64]>,
    ) -> Result<bool, String> {
        if let Some(wl) = wavelength {
            if !wl_bits_eq(&self.wavelength, wl) {
                return Err(format!(
                    "SpectralDataFrame(uid={}): wavelength grid conflict.",
                    self.uid
                ));
            }
        }
        if value.len() != self.wavelength.len() {
            return Err("Length mismatch".to_string());
        }

        let mut guard = self.data.write();
        let is_new = !guard.contains_key(&key);
        guard.insert(key, value);
        Ok(is_new)
    }

    /// Cloned view of the curve under `key`, if present.
    pub fn get_data(&self, key: &OpticalKey) -> Option<SpectralData> {
        self.data.read().get(key).cloned()
    }

    /// All keys stored in this frame.
    pub fn keys(&self) -> Vec<OpticalKey> {
        self.data.read().keys().cloned().collect()
    }

    /// Number of curves stored in this frame.
    pub fn len(&self) -> usize {
        self.data.read().len()
    }

    /// Borrowed grid shared by all curves in this frame.
    pub fn wavelength(&self) -> &[f64] {
        &self.wavelength
    }

    /// `(min, max)` grid endpoints (cached at construction).
    pub fn wl_bounds(&self) -> (f64, f64) {
        (self.wl_min, self.wl_max)
    }

    /// Drop the curve under `key`. Errors when the key is absent.
    pub fn remove(&self, key: &OpticalKey) -> Result<(), String> {
        if self.data.write().remove(key).is_none() {
            return Err("Key not found".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OpticalCollection
// ---------------------------------------------------------------------------
/// A set of frames indexed by grid fingerprint plus a key→frames map.
/// Curves on identical grids share one frame (fingerprints + bit-equality
/// guard); `key_map` tracks which frames hold fragments of each key.
pub struct OpticalCollection {
    pub frames: RwLock<Vec<Arc<SpectralDataFrame>>>,
    wl_fingerprints: RwLock<AHashMap<WlSig, Arc<SpectralDataFrame>>>,
    key_map: RwLock<AHashMap<OpticalKey, SmallVec<[Arc<SpectralDataFrame>; 2]>>>,
    display_spectral: RwLock<Unit>,
    display_intensity: RwLock<Unit>,
}

impl Default for OpticalCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl OpticalCollection {
    /// Empty collection with nm/raw display units.
    pub fn new() -> Self {
        OpticalCollection {
            frames: RwLock::new(Vec::new()),
            wl_fingerprints: RwLock::new(AHashMap::new()),
            key_map: RwLock::new(AHashMap::new()),
            display_spectral: RwLock::new(Unit::NM),
            display_intensity: RwLock::new(Unit::RAW),
        }
    }

    /// Display unit for wavelength axes returned to callers.
    pub fn set_display_spectral(&self, unit: Unit) {
        *self.display_spectral.write() = unit;
    }
    /// Current wavelength display unit.
    pub fn display_spectral(&self) -> Unit {
        *self.display_spectral.read()
    }
    /// Display unit for data values returned to callers.
    pub fn set_display_intensity(&self, unit: Unit) {
        *self.display_intensity.write() = unit;
    }
    /// Current data display unit.
    pub fn display_intensity(&self) -> Unit {
        *self.display_intensity.read()
    }
    /// Number of frames (distinct grids) held.
    pub fn frame_count(&self) -> usize {
        self.frames.read().len()
    }
    /// Number of distinct keys across all frames.
    pub fn len_keys(&self) -> usize {
        self.key_map.read().len()
    }
    /// All distinct keys across all frames.
    pub fn keys(&self) -> Vec<OpticalKey> {
        self.key_map.read().keys().cloned().collect()
    }
    /// True when any frame holds fragments of `key`.
    pub fn contains_key(&self, key: &OpticalKey) -> bool {
        self.key_map.read().contains_key(key)
    }
    /// Frame by insertion index, if in range.
    pub fn frame_at(&self, index: usize) -> Option<Arc<SpectralDataFrame>> {
        self.frames.read().get(index).cloned()
    }
    /// Cloned list of all frames (lock-free iteration for callers).
    pub fn frames_snapshot(&self) -> Vec<Arc<SpectralDataFrame>> {
        self.frames.read().clone()
    }
    /// Frames holding fragments of `key` (inline for the common ≤ 2 case).
    pub fn frames_for_key(&self, key: &OpticalKey) -> Option<SmallVec<[Arc<SpectralDataFrame>; 2]>> {
        self.key_map.read().get(key).cloned()
    }

    /// Per-fragment `(data, wavelength)` pairs under `key`, converted to the
    /// display units. Returns `None` when the key is unknown.
    pub fn get_converted(&self, key: &OpticalKey) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let frames = self.key_map.read().get(key)?.clone();
        let int_unit = *self.display_intensity.read();
        let spec_unit = *self.display_spectral.read();

        let mut data_list = Vec::with_capacity(frames.len());
        let mut wl_list = Vec::with_capacity(frames.len());
        for frm in frames {
            let data_raw = frm
                .get_data(key)
                .unwrap_or_else(|| SpectralData::from_arc(Arc::from(&[][..])));
            data_list.push(convert_unit(&data_raw, Unit::RAW, int_unit).into_owned());
            wl_list.push(convert_unit(frm.wavelength(), Unit::NM, spec_unit).into_owned());
        }
        Some((data_list, wl_list))
    }

    /// Record that `frame` holds fragments of `key`. Returns false when the
    /// mapping already existed.
    pub fn map_frame_to_key(&self, key: &OpticalKey, frame: &Arc<SpectralDataFrame>) -> bool {
        let mut key_map = self.key_map.write();
        let frames = key_map.entry(key.clone()).or_default();
        if frames.iter().any(|f| Arc::ptr_eq(f, frame)) {
            return false;
        }
        frames.push(frame.clone());
        true
    }

    pub fn set_data(
        &self,
        key: OpticalKey,
        value: &[f64],
        wavelength: &[f64],
        input_spectral: Unit,
        input_intensity: Unit,
    ) -> Result<bool, String> {
        let base_wl = convert_unit(wavelength, input_spectral, Unit::NM);
        let base_data = convert_unit(value, input_intensity, Unit::RAW);
        let (target_frame, frame_created) = self.get_or_create_frame(&base_wl)?;
        let data_arc: Arc<[f64]> = match base_data {
            Cow::Borrowed(b) => Arc::from(b),
            Cow::Owned(o) => Arc::from(o),
        };
        let is_new = target_frame.set_data(key.clone(), SpectralData::from_arc(data_arc), None)?;
        if is_new {
            self.map_frame_to_key(&key, &target_frame);
        }
        Ok(frame_created)
    }

    /// Fetch the frame for `wl_arr` (bit-exact grid match) or create it.
    /// Returns the frame plus whether it is newly created.
    fn get_or_create_frame(&self, wl_arr: &[f64]) -> Result<(Arc<SpectralDataFrame>, bool), String> {
        let sig = wl_signature(wl_arr);
        if let Some(frm) = self.wl_fingerprints.read().get(&sig) {
            if wl_bits_eq(frm.wavelength(), wl_arr) {
                return Ok((frm.clone(), false));
            }
        }
        let mut fp = self.wl_fingerprints.write();
        if let Some(frm) = fp.get(&sig) {
            if wl_bits_eq(frm.wavelength(), wl_arr) {
                return Ok((frm.clone(), false));
            }
        }
        let new_frame = Arc::new(SpectralDataFrame::new(wl_arr)?);
        self.frames.write().push(new_frame.clone());
        fp.insert(sig, new_frame.clone());
        Ok((new_frame, true))
    }
}

// ---------------------------------------------------------------------------
// Distribution plan & OpticalWeaver
// ---------------------------------------------------------------------------
#[derive(Clone)]
enum SliceOrIndices {
    Slice(usize, usize),
    Indices(Vec<usize>),
}
impl SliceOrIndices {
    /// True if this plan entry can use the zero-copy (buffer-sharing) path when
    /// a shared source `Arc<[f64]>` is available. Contiguous slices can be a
    /// view onto the source; strided index sets must be copied out.
    #[inline]
    fn wants_shared(&self) -> bool {
        matches!(self, SliceOrIndices::Slice(..))
    }

    /// Extract this entry's fragment from `data`. When `shared` is `Some` it must
    /// alias `data`; a contiguous slice then becomes a zero-copy view into that
    /// buffer (which keeps the whole source alive for the fragment's lifetime).
    /// Strided fragments, and the `shared == None` case, copy out.
    #[inline]
    fn gather(&self, data: &[f64], shared: Option<&Arc<[f64]>>) -> SpectralData {
        match self {
            SliceOrIndices::Slice(s, e) => match shared {
                Some(arc) => SpectralData { buf: Arc::clone(arc), start: *s, len: e - s },
                None => SpectralData::from_arc(Arc::from(&data[*s..*e])),
            },
            SliceOrIndices::Indices(idx) => {
                let vec: Vec<f64> = idx.iter().map(|&i| data[i]).collect();
                SpectralData::from_arc(Arc::from(vec))
            }
        }
    }
}

type DistributionPlan = Vec<(Arc<SpectralDataFrame>, SliceOrIndices)>;

/// Top-level weaving store: fragment distribution plus re-assembly.
/// `generation` bumps on every structural change so caches and Python-side
/// snapshots can detect staleness; `invalidate_cache` clears the LRU plan cache.
pub struct OpticalWeaver {
    pub inner: OpticalCollection,
    distribution_cache: RwLock<LruCache<WlSig, (usize, Arc<[f64]>, DistributionPlan)>>,
    generation: AtomicUsize,
}

impl OpticalWeaver {
    /// Create a weaver with an LRU distribution-plan cache of `cache_size` grids.
    pub fn new(cache_size: usize) -> Self {
        OpticalWeaver {
            inner: OpticalCollection::new(),
            distribution_cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(cache_size.max(1)).unwrap(),
            )),
            generation: AtomicUsize::new(0),
        }
    }

    /// Record a structural change (new frame or new key mapping).
    pub fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
    /// Monotonic structural-change counter for staleness checks.
    pub fn generation(&self) -> usize {
        self.generation.load(Ordering::SeqCst)
    }

    /// Ingest one curve: convert units, route to the matching frame
    /// (creating it when the grid is new), and bump the generation on
    /// structural change.
    pub fn set_data(
        &self,
        key: OpticalKey,
        value: &[f64],
        wavelength: &[f64],
        input_spectral: Unit,
        input_intensity: Unit,
    ) -> Result<(), String> {
        if self
            .inner
            .set_data(key, value, wavelength, input_spectral, input_intensity)?
        {
            self.bump_generation();
        }
        Ok(())
    }

    /// Re-assemble the continuous `(wavelength, data)` curve for `key` by
    /// concatenating its fragments in grid order. Errors on unknown keys.
    pub fn get_weaved(&self, key: &OpticalKey) -> Result<(Vec<f64>, Vec<f64>), String> {
        let frames = self
            .inner
            .frames_for_key(key)
            .ok_or_else(|| "Key not found.".to_string())?;
        let mut fragments: SmallVec<[(f64, &[f64], SpectralData); 4]> = SmallVec::new();
        for frm in &frames {
            if let Some(data) = frm.get_data(key) {
                fragments.push((frm.wl_bounds().0, frm.wavelength(), data));
            }
        }
        if fragments.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        fragments.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let total: usize = fragments.iter().map(|f| f.1.len()).sum();
        let mut all_wl = Vec::with_capacity(total);
        let mut all_data = Vec::with_capacity(total);
        for (_, wl, data) in fragments {
            all_wl.extend_from_slice(wl);
            all_data.extend_from_slice(&data);
        }
        Ok((all_wl, all_data))
    }

    /// All keys re-assembled, grouped by woven grid: one entry per distinct
    /// grid with the curves sampled on it.
    pub fn get_weaved_collections(&self) -> Vec<(Vec<f64>, AHashMap<OpticalKey, Vec<f64>>)> {
        let mut groups: AHashMap<WlSig, (Vec<f64>, AHashMap<OpticalKey, Vec<f64>>)> = AHashMap::new();
        for key in self.inner.keys() {
            if let Ok((wl, data)) = self.get_weaved(&key) {
                if wl.is_empty() {
                    continue;
                }
                let sig = wl_signature(&wl);
                let entry = groups.entry(sig).or_insert_with(|| (wl.clone(), AHashMap::new()));
                entry.1.insert(key, data);
            }
        }
        groups.into_values().collect()
    }

    /// Distribute one long curve across frames per the cached plan.
    /// Returns the number of fragments written.
    pub fn unweave(
        &self,
        key: OpticalKey,
        full_wavelength: &[f64],
        full_data: &[f64],
    ) -> Result<usize, String> {
        let plan = self.resolve_plan(full_wavelength)?;
        // Materialise the source once if any fragment is a contiguous slice; all
        // such fragments then view it with no copy. Skip it for all-strided plans.
        let shared_data: Option<Arc<[f64]>> =
            plan.iter().any(|(_, i)| i.wants_shared()).then(|| Arc::from(full_data));

        let mut updated = 0;
        for (frm, indices) in &plan {
            let subset = indices.gather(full_data, shared_data.as_ref());
            let is_new = frm.set_data(key.clone(), subset, None)?;
            if is_new {
                self.inner.map_frame_to_key(&key, frm);
            }
            updated += 1;
        }
        Ok(updated)
    }

    /// Distribute many curves sharing one grid, reusing a single plan.
    /// Returns the total fragments written.
    pub fn unweave_collection(
        &self,
        common_wavelength: &[f64],
        data_batch: AHashMap<OpticalKey, &[f64]>,
    ) -> Result<usize, String> {
        if data_batch.is_empty() {
            return Ok(0);
        }
        let plan = self.resolve_plan(common_wavelength)?;
        // The plan is shared across all keys, so decide once whether it contains
        // any contiguous slice worth materialising a per-key shared buffer for.
        let needs_shared = plan.iter().any(|(_, i)| i.wants_shared());
        let mut total = 0usize;
        for (key, full_data) in &data_batch {
            let shared_data: Option<Arc<[f64]>> =
                needs_shared.then(|| Arc::from(*full_data));
            for (frm, indices) in &plan {
                let subset = indices.gather(full_data, shared_data.as_ref());
                let is_new = frm.set_data(key.clone(), subset, None)?;
                if is_new {
                    self.inner.map_frame_to_key(key, frm);
                }
                total += 1;
            }
        }
        Ok(total)
    }

    /// Drop all cached distribution plans (grids re-resolve on next use).
    pub fn invalidate_cache(&self) {
        self.distribution_cache.write().clear();
    }

    /// LRU-cached distribution plan for `full_wavelength` (hit or rebuild).
    fn resolve_plan(&self, full_wavelength: &[f64]) -> Result<DistributionPlan, String> {
        let sig = wl_signature(full_wavelength);
        let current_gen = self.generation.load(Ordering::SeqCst);
        {
            let mut cache = self.distribution_cache.write();
            if let Some((cached_gen, cached_wl, plan)) = cache.get(&sig) {
                if *cached_gen == current_gen && wl_bits_eq(cached_wl, full_wavelength) {
                    return Ok(plan.clone());
                }
            }
            cache.pop(&sig);
        }
        let plan = self.build_distribution_plan(full_wavelength)?;
        self.distribution_cache.write().put(
            sig,
            (current_gen, Arc::from(full_wavelength), plan.clone()),
        );
        Ok(plan)
    }

    /// Fragment `full_wavelength` across existing frames (contiguous slices
    /// where grids overlap, strided gathers elsewhere).
    fn build_distribution_plan(&self, full_wavelength: &[f64]) -> Result<DistributionPlan, String> {
        let mut plan = Vec::new();
        if full_wavelength.is_empty() {
            return Ok(plan);
        }
        let (fw_min, fw_max) = (
            full_wavelength[0],
            full_wavelength[full_wavelength.len() - 1],
        );

        for frm in self.inner.frames_snapshot() {
            let frame_wl = frm.wavelength();
            let (f_min, f_max) = frm.wl_bounds();
            if f_min > fw_max || f_max < fw_min {
                continue;
            }

            let idx_lo = full_wavelength.partition_point(|&x| x < f_min);
            let idx_hi = full_wavelength.partition_point(|&x| x <= f_max);
            let candidate = &full_wavelength[idx_lo..idx_hi];

            if candidate.len() == frame_wl.len() && candidate == frame_wl {
                plan.push((frm, SliceOrIndices::Slice(idx_lo, idx_hi)));
                continue;
            }

            // Two-pointer sweep of two sorted grids (O(n+m)): `fw` advances
            // monotonically as we walk the (strictly increasing) frame grid,
            // collecting the exact-match positions. Same sweep shape as the merit
            // interpolation loop, minus the value math (this aligns indices only).
            let mut indices = Vec::with_capacity(frame_wl.len());
            let mut fw = idx_lo;
            for &target in frame_wl {
                while fw < full_wavelength.len() && full_wavelength[fw] < target {
                    fw += 1;
                }
                if fw < full_wavelength.len() && (full_wavelength[fw] - target).abs() < 1e-12 {
                    indices.push(fw);
                    fw += 1; // exact match consumed; next target is strictly greater
                }
            }
            if !indices.is_empty() {
                if indices
                    .iter()
                    .enumerate()
                    .all(|(i, &idx)| i == 0 || idx == indices[i - 1] + 1)
                {
                    plan.push((
                        frm,
                        SliceOrIndices::Slice(indices[0], indices[indices.len() - 1] + 1),
                    ));
                } else {
                    plan.push((frm, SliceOrIndices::Indices(indices)));
                }
            }
        }
        Ok(plan)
    }
}
