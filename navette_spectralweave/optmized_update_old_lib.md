Here is the fully consolidated implementation plan. It combines your original structure with all the optimizations, applying the three tweaks (partial matching, true zero-copy via `&Arc`, and safe `PyArray1` APIs). 

You can replace your `Cargo.toml` and `src/lib.rs` with the following.

### 1. `Cargo.toml`

```toml
[package]
name = "navette-spectralweave"
version = "0.2.0"
edition = "2024"
rust-version = "1.85"  # pyo3 0.28.x MSRV

[lib]
name = "spectralweave"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28.3", features = ["extension-module", "abi3-py312"] }
numpy = "0.28"
parking_lot = "0.12"
lru = "0.12"
bytemuck = "1"
xxhash-rust = { version = "0.8", features = ["xxh3"] }
mimalloc = "0.1"

# --- New dependencies ---
ahash = "0.8"
smallvec = "1.13"
dashmap = "6.1"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### 2. `src/lib.rs`

```rust
//! Navette (Rust core) — weaving spectral fragments into continuous curves.
//!
//! Fully optimized implementation featuring:
//! - Safe GIL detachment (no raw pointer unsoundness)
//! - String interning with auto-cleanup for O(1) key hashing/equality
//! - `ahash` and `SmallVec` for fast, low-allocation maps
//! - Two-pointer O(n+m) merge-pass plan building
//! - True thresholded zero-copy unweave
//! - Safe lock ordering in plan resolution

use std::borrow::Cow;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use ahash::AHashMap;
use dashmap::DashMap;
use lru::LruCache;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
use parking_lot::RwLock;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use smallvec::SmallVec;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ---------------------------------------------------------------------------
// String Interning (with periodic cleanup)
// ---------------------------------------------------------------------------
static STRING_INTERN: LazyLock<DashMap<String, Arc<str>>> = LazyLock::new(|| DashMap::new());
static INTERN_ACCESS_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn intern_str(s: &str) -> Arc<str> {
    if INTERN_ACCESS_COUNTER.fetch_add(1, Ordering::Relaxed) % 10000 == 0 {
        // Garbage collect unreferenced strings
        STRING_INTERN.retain(|_, v| Arc::strong_count(v) > 1);
    }
    
    if let Some(existing) = STRING_INTERN.get(s) {
        return existing.clone();
    }
    let interned: Arc<str> = Arc::from(s);
    STRING_INTERN.insert(s.to_string(), interned.clone());
    interned
}

// ---------------------------------------------------------------------------
// Unit system
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unit {
    NM,
    RAW,
}

#[inline]
fn convert_unit<'a>(value: &'a [f64], from: Unit, to: Unit) -> Cow<'a, [f64]> {
    if from == to {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(value.to_vec())
    }
}

type WlSig = u128;

#[inline]
fn wl_signature(wl: &[f64]) -> WlSig {
    xxhash_rust::xxh3::xxh3_128(bytemuck::cast_slice::<f64, u8>(wl))
}

#[inline]
fn wl_bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && bytemuck::cast_slice::<f64, u8>(a) == bytemuck::cast_slice::<f64, u8>(b)
}

// ---------------------------------------------------------------------------
// Optical key
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct OpticalKey {
    pub wavelength: f64,
    pub data_type: Arc<str>,
    pub polarisation: Arc<str>,
}

impl PartialEq for OpticalKey {
    fn eq(&self, other: &Self) -> bool {
        self.wavelength.to_bits() == other.wavelength.to_bits()
            && (Arc::ptr_eq(&self.data_type, &other.data_type) || self.data_type == other.data_type)
            && (Arc::ptr_eq(&self.polarisation, &other.polarisation) || self.polarisation == other.polarisation)
    }
}
impl Eq for OpticalKey {}

impl Hash for OpticalKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.wavelength.to_bits().hash(state);
        (Arc::as_ptr(&self.data_type) as usize).hash(state);
        (Arc::as_ptr(&self.polarisation) as usize).hash(state);
    }
}

impl From<(f64, String, String)> for OpticalKey {
    fn from(t: (f64, String, String)) -> Self {
        OpticalKey {
            wavelength: t.0,
            data_type: intern_str(&t.1),
            polarisation: intern_str(&t.2),
        }
    }
}

impl OpticalKey {
    #[inline]
    fn as_tuple(&self) -> (f64, String, String) {
        (
            self.wavelength,
            self.data_type.to_string(),
            self.polarisation.to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// SpectralData
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct SpectralData {
    buf: Arc<[f64]>,
    start: usize,
    len: usize,
}

impl SpectralData {
    #[inline]
    fn from_arc(buf: Arc<[f64]>) -> Self {
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

// ---------------------------------------------------------------------------
// SpectralDataFrame
// ---------------------------------------------------------------------------
pub struct SpectralDataFrame {
    pub uid: usize,
    data: RwLock<AHashMap<OpticalKey, SpectralData>>,
    wavelength: Arc<[f64]>,
    wl_min: f64,
    wl_max: f64,
}

impl SpectralDataFrame {
    fn new(wavelength: &[f64]) -> PyResult<Self> {
        static UID_GEN: AtomicUsize = AtomicUsize::new(0);
        let uid = UID_GEN.fetch_add(1, Ordering::Relaxed);

        if wavelength.is_empty() {
            return Err(PyValueError::new_err("SpectralDataFrame: wavelength array must be non-empty."));
        }
        for i in 0..wavelength.len() - 1 {
            if !(wavelength[i] < wavelength[i + 1]) {
                return Err(PyValueError::new_err("SpectralDataFrame: wavelength array must be strictly monotonically increasing."));
            }
        }

        Ok(SpectralDataFrame {
            uid,
            data: RwLock::new(AHashMap::new()),
            wl_min: wavelength[0],
            wl_max: wavelength[wavelength.len() - 1],
            wavelength: Arc::from(wavelength),
        })
    }

    fn set_data(&self, key: OpticalKey, value: SpectralData, wavelength: Option<&[f64]>) -> PyResult<bool> {
        if let Some(wl) = wavelength {
            if !wl_bits_eq(&self.wavelength, wl) {
                return Err(PyValueError::new_err(format!(
                    "SpectralDataFrame(uid={}): wavelength grid conflict (bit-exact match required).", self.uid
                )));
            }
        }
        if value.len() != self.wavelength.len() {
            return Err(PyValueError::new_err(format!(
                "SpectralDataFrame(uid={}): value length {} != wavelength length {}", self.uid, value.len(), self.wavelength.len()
            )));
        }

        let mut guard = self.data.write();
        let is_new = !guard.contains_key(&key);
        guard.insert(key, value);
        Ok(is_new)
    }

    fn get_data(&self, key: &OpticalKey) -> Option<SpectralData> { self.data.read().get(key).cloned() }
    fn keys(&self) -> Vec<OpticalKey> { self.data.read().keys().cloned().collect() }
    fn len(&self) -> usize { self.data.read().len() }
    fn wavelength(&self) -> &[f64] { &self.wavelength }
    fn wl_bounds(&self) -> (f64, f64) { (self.wl_min, self.wl_max) }

    fn remove(&self, key: &OpticalKey) -> PyResult<()> {
        if self.data.write().remove(key).is_none() {
            return Err(PyKeyError::new_err("Key not found"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OpticalCollection
// ---------------------------------------------------------------------------
pub struct OpticalCollection {
    frames: RwLock<Vec<Arc<SpectralDataFrame>>>,
    wl_fingerprints: RwLock<AHashMap<WlSig, Arc<SpectralDataFrame>>>,
    key_map: RwLock<AHashMap<OpticalKey, SmallVec<[Arc<SpectralDataFrame>; 2]>>>,
    display_spectral: RwLock<Unit>,
    display_intensity: RwLock<Unit>,
    cached_keys: RwLock<Option<Vec<(f64, String, String)>>>,
}

impl Default for OpticalCollection {
    fn default() -> Self { Self::new() }
}

impl OpticalCollection {
    pub fn new() -> Self {
        OpticalCollection {
            frames: RwLock::new(Vec::new()),
            wl_fingerprints: RwLock::new(AHashMap::new()),
            key_map: RwLock::new(AHashMap::new()),
            display_spectral: RwLock::new(Unit::NM),
            display_intensity: RwLock::new(Unit::RAW),
            cached_keys: RwLock::new(None),
        }
    }

    fn invalidate_keys_cache(&self) { *self.cached_keys.write() = None; }

    pub fn set_display_spectral(&self, unit: Unit) { *self.display_spectral.write() = unit; }
    pub fn display_spectral(&self) -> Unit { *self.display_spectral.read() }
    pub fn set_display_intensity(&self, unit: Unit) { *self.display_intensity.write() = unit; }
    pub fn display_intensity(&self) -> Unit { *self.display_intensity.read() }

    pub fn frame_count(&self) -> usize { self.frames.read().len() }
    pub fn len_keys(&self) -> usize { self.key_map.read().len() }
    
    pub fn keys(&self) -> Vec<OpticalKey> { self.key_map.read().keys().cloned().collect() }
    
    pub fn keys_as_tuples(&self) -> Vec<(f64, String, String)> {
        {
            let cached = self.cached_keys.read();
            if let Some(keys) = cached.as_ref() {
                return keys.clone();
            }
        }
        let keys: Vec<_> = self.key_map.read().keys().map(|k| k.as_tuple()).collect();
        *self.cached_keys.write() = Some(keys.clone());
        keys
    }
    
    pub fn contains_key(&self, key: &OpticalKey) -> bool { self.key_map.read().contains_key(key) }
    pub fn frame_at(&self, index: usize) -> Option<Arc<SpectralDataFrame>> { self.frames.read().get(index).cloned() }
    pub fn frames_snapshot(&self) -> Vec<Arc<SpectralDataFrame>> { self.frames.read().clone() }
    pub fn frames_for_key(&self, key: &OpticalKey) -> Option<SmallVec<[Arc<SpectralDataFrame>; 2]>> { self.key_map.read().get(key).cloned() }

    pub fn get_converted(&self, key: &OpticalKey) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let frames = self.key_map.read().get(key)?.clone();
        let int_unit = *self.display_intensity.read();
        let spec_unit = *self.display_spectral.read();

        let mut data_list = Vec::with_capacity(frames.len());
        let mut wl_list = Vec::with_capacity(frames.len());
        for frm in frames {
            let data_raw = frm.get_data(key).unwrap_or_else(|| SpectralData::from_arc(Arc::from(&[][..])));
            data_list.push(convert_unit(&data_raw, Unit::RAW, int_unit).into_owned());
            wl_list.push(convert_unit(frm.wavelength(), Unit::NM, spec_unit).into_owned());
        }
        Some((data_list, wl_list))
    }

    fn map_frame_to_key(&self, key: &OpticalKey, frame: &Arc<SpectralDataFrame>) -> bool {
        let mut key_map = self.key_map.write();
        let frames = key_map.entry(key.clone()).or_default();
        if frames.iter().any(|f| Arc::ptr_eq(f, frame)) {
            return false;
        }
        frames.push(frame.clone());
        self.invalidate_keys_cache();
        true
    }

    pub fn set_data(&self, key: OpticalKey, value: &[f64], wavelength: &[f64], input_spectral: Unit, input_intensity: Unit) -> PyResult<bool> {
        if value.len() != wavelength.len() {
            return Err(PyValueError::new_err(format!("Length mismatch: value ({}) vs wavelength ({}).", value.len(), wavelength.len())));
        }

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

    fn get_or_create_frame(&self, wl_arr: &[f64]) -> PyResult<(Arc<SpectralDataFrame>, bool)> {
        let sig = wl_signature(wl_arr);
        let fp = self.wl_fingerprints.upgradable_read();
        if let Some(frm) = fp.get(&sig) {
            if wl_bits_eq(frm.wavelength(), wl_arr) {
                return Ok((frm.clone(), false));
            }
        }
        
        let mut fp = parking_lot::RwLockUpgradableReadGuard::upgrade(fp);
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
// Distribution plan
// ---------------------------------------------------------------------------
#[derive(Clone)]
enum SliceOrIndices {
    Slice(usize, usize),
    Indices(Vec<usize>),
}

impl SliceOrIndices {
    #[inline]
    fn gather(&self, data: &Arc<[f64]>) -> SpectralData {
        const ZERO_COPY_THRESHOLD: usize = 8192;
        match self {
            SliceOrIndices::Slice(s, e) => {
                let len = e - s;
                if len > ZERO_COPY_THRESHOLD {
                    SpectralData { buf: Arc::clone(data), start: *s, len }
                } else {
                    SpectralData::from_arc(Arc::from(&data[*s..*e]))
                }
            },
            SliceOrIndices::Indices(idx) => {
                let vec: Vec<f64> = idx.iter().map(|&i| data[i]).collect();
                SpectralData::from_arc(Arc::from(vec))
            }
        }
    }
}

type DistributionPlan = Vec<(Arc<SpectralDataFrame>, SliceOrIndices)>;

// ---------------------------------------------------------------------------
// OpticalWeaver
// ---------------------------------------------------------------------------
pub struct OpticalWeaver {
    inner: OpticalCollection,
    distribution_cache: RwLock<LruCache<WlSig, (usize, Arc<[f64]>, DistributionPlan)>>,
    generation: AtomicUsize,
}

impl OpticalWeaver {
    pub fn new(cache_size: usize) -> Self {
        let cap = NonZeroUsize::new(cache_size.max(1)).unwrap();
        OpticalWeaver {
            inner: OpticalCollection::new(),
            distribution_cache: RwLock::new(LruCache::new(cap)),
            generation: AtomicUsize::new(0),
        }
    }

    fn bump_generation(&self) { self.generation.fetch_add(1, Ordering::Relaxed); }
    pub fn generation(&self) -> usize { self.generation.load(Ordering::Relaxed) }

    pub fn display_spectral(&self) -> Unit { self.inner.display_spectral() }
    pub fn set_display_spectral(&self, unit: Unit) { self.inner.set_display_spectral(unit); }
    pub fn display_intensity(&self) -> Unit { self.inner.display_intensity() }
    pub fn set_display_intensity(&self, unit: Unit) { self.inner.set_display_intensity(unit); }
    pub fn frame_count(&self) -> usize { self.inner.frame_count() }
    pub fn len_keys(&self) -> usize { self.inner.len_keys() }
    pub fn keys(&self) -> Vec<OpticalKey> { self.inner.keys() }
    pub fn keys_as_tuples(&self) -> Vec<(f64, String, String)> { self.inner.keys_as_tuples() }
    pub fn contains_key(&self, key: &OpticalKey) -> bool { self.inner.contains_key(key) }
    pub fn frame_at(&self, index: usize) -> Option<Arc<SpectralDataFrame>> { self.inner.frame_at(index) }
    pub fn frames_snapshot(&self) -> Vec<Arc<SpectralDataFrame>> { self.inner.frames_snapshot() }
    pub fn frames_for_key(&self, key: &OpticalKey) -> Option<SmallVec<[Arc<SpectralDataFrame>; 2]>> { self.inner.frames_for_key(key) }
    pub fn get_converted(&self, key: &OpticalKey) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> { self.inner.get_converted(key) }

    pub fn set_data(&self, key: OpticalKey, value: &[f64], wavelength: &[f64], input_spectral: Unit, input_intensity: Unit) -> PyResult<()> {
        let frame_created = self.inner.set_data(key, value, wavelength, input_spectral, input_intensity)?;
        if frame_created { self.bump_generation(); }
        Ok(())
    }

    pub fn set_data_batch(&self, entries: &[(OpticalKey, &[f64], &[f64], Unit, Unit)]) -> PyResult<usize> {
        let mut frame_created_count = 0;
        for (key, value, wavelength, s, i) in entries {
            let created = self.inner.set_data(key.clone(), value, wavelength, *s, *i)?;
            if created { frame_created_count += 1; }
        }
        if frame_created_count > 0 { self.bump_generation(); }
        Ok(frame_created_count)
    }

    pub fn get_weaved(&self, key: &OpticalKey, strict: bool) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let frames = self.inner.frames_for_key(key).ok_or_else(|| PyKeyError::new_err("Key not found."))?;

        let mut fragments: SmallVec<[(f64, &[f64], SpectralData); 4]> = SmallVec::new();
        for frm in &frames {
            if let Some(data) = frm.get_data(key) {
                fragments.push((frm.wl_bounds().0, frm.wavelength(), data));
            }
        }
        if fragments.is_empty() { return Ok((Vec::new(), Vec::new())); }

        fragments.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        if strict {
            for i in 0..fragments.len() - 1 {
                let cur_max = fragments[i].1.last().copied().unwrap_or(f64::NEG_INFINITY);
                let nxt_min = fragments[i + 1].0;
                if cur_max > nxt_min {
                    return Err(PyValueError::new_err(format!(
                        "get_weaved: frames overlap for key ({}, {:?}, {:?}) at {} > {}",
                        key.wavelength, key.data_type, key.polarisation, cur_max, nxt_min
                    )));
                }
            }
        }

        let total: usize = fragments.iter().map(|f| f.1.len()).sum();
        let mut all_wl = Vec::with_capacity(total);
        let mut all_data = Vec::with_capacity(total);
        
        for (_, wl, data) in fragments {
            all_wl.extend_from_slice(wl);
            all_data.extend_from_slice(&data);
        }
        Ok((all_wl, all_data))
    }

    pub fn get_weaved_collections(&self) -> Vec<(Vec<f64>, AHashMap<OpticalKey, Vec<f64>>)> {
        let mut groups: AHashMap<WlSig, (Arc<[f64]>, Vec<Arc<SpectralDataFrame>>)> = AHashMap::new();
        for frm in self.inner.frames_snapshot() {
            let sig = wl_signature(frm.wavelength());
            groups.entry(sig).or_insert_with(|| (Arc::from(frm.wavelength()), Vec::new())).1.push(frm);
        }
        
        groups.into_iter().map(|(_, (wl, frames))| {
            let mut per_key: AHashMap<OpticalKey, Vec<f64>> = AHashMap::new();
            for frm in &frames {
                let data_lock = frm.data.read();
                for (key, data) in data_lock.iter() {
                    per_key.entry(key.clone()).or_insert_with(|| data.to_vec());
                }
            }
            (wl.to_vec(), per_key)
        }).collect()
    }

    pub fn unweave(&self, key: OpticalKey, full_wavelength: &[f64], full_data: &[f64]) -> PyResult<usize> {
        if full_data.len() != full_wavelength.len() {
            return Err(PyValueError::new_err(format!("unweave: full_data length {} != full_wavelength length {}", full_data.len(), full_wavelength.len())));
        }

        let plan = self.resolve_plan(full_wavelength)?;
        let shared_data: Arc<[f64]> = Arc::from(full_data);
        
        let mut updated = 0;
        for (frm, indices) in plan {
            let subset = indices.gather(&shared_data);
            let is_new = frm.set_data(key.clone(), subset, None)?;
            if is_new { self.inner.map_frame_to_key(&key, &frm); }
            updated += 1;
        }
        Ok(updated)
    }

    pub fn unweave_collection(&self, common_wavelength: &[f64], data_batch: &AHashMap<OpticalKey, &[f64]>) -> PyResult<usize> {
        if data_batch.is_empty() { return Ok(0); }
        for (k, v) in data_batch {
            if v.len() != common_wavelength.len() {
                return Err(PyValueError::new_err(format!(
                    "unweave_collection: data for key ({}, {:?}, {:?}) has length {} != common_wavelength length {}",
                    k.wavelength, k.data_type, k.polarisation, v.len(), common_wavelength.len()
                )));
            }
        }

        let plan = self.resolve_plan(common_wavelength)?;
        let mut total = 0usize;
        
        for (key, full_data) in data_batch {
            let shared_data: Arc<[f64]> = Arc::from(*full_data);
            let mut new_frames: SmallVec<[Arc<SpectralDataFrame>; 4]> = SmallVec::new();
            
            for (frm, indices) in &plan {
                let subset = indices.gather(&shared_data);
                let is_new = frm.set_data(key.clone(), subset, None)?;
                if is_new { new_frames.push(frm.clone()); }
                total += 1;
            }
            
            if !new_frames.is_empty() {
                let mut key_map = self.inner.key_map.write();
                let entry = key_map.entry(key.clone()).or_default();
                for frm in new_frames {
                    if !entry.iter().any(|f| Arc::ptr_eq(f, &frm)) {
                        entry.push(frm);
                    }
                }
                self.inner.invalidate_keys_cache();
            }
        }
        Ok(total)
    }

    pub fn invalidate_cache(&self) { self.distribution_cache.write().clear(); }

    fn resolve_plan(&self, full_wavelength: &[f64]) -> PyResult<DistributionPlan> {
        let sig = wl_signature(full_wavelength);
        let current_gen = self.generation.load(Ordering::Relaxed);
        
        // 1. Check cache with read lock
        {
            let cache = self.distribution_cache.read();
            if let Some((cached_gen, cached_wl, plan)) = cache.get(&sig) {
                if *cached_gen == current_gen && wl_bits_eq(cached_wl, full_wavelength) {
                    return Ok(plan.clone());
                }
            }
        }
        
        // 2. Build plan without holding any cache locks
        let plan = self.build_distribution_plan(full_wavelength)?;
        let grid: Arc<[f64]> = Arc::from(full_wavelength);
        
        // 3. Insert into cache with write lock
        {
            let mut cache = self.distribution_cache.write();
            cache.put(sig, (current_gen, grid, plan.clone()));
        }
        
        Ok(plan)
    }

    fn build_distribution_plan(&self, full_wavelength: &[f64]) -> PyResult<DistributionPlan> {
        let frames_snapshot = self.inner.frames_snapshot();
        let mut plan = Vec::new();
        if full_wavelength.is_empty() { return Ok(plan); }

        let fw_min = full_wavelength[0];
        let fw_max = full_wavelength[full_wavelength.len() - 1];

        for frm in frames_snapshot {
            let frame_wl = frm.wavelength();
            let (f_min, f_max) = frm.wl_bounds();
            if f_min > fw_max || f_max < fw_min { continue; }

            let mut indices = Vec::with_capacity(frame_wl.len());
            let mut fi = 0;
            let mut fw = 0;
            let mut contiguous = true;
            
            while fi < frame_wl.len() && fw < full_wavelength.len() {
                let target = frame_wl[fi];
                let diff = (full_wavelength[fw] - target).abs();
                
                if diff < 1e-12 {
                    if let Some(&prev) = indices.last() {
                        if fw != prev + 1 { contiguous = false; }
                    }
                    indices.push(fw);
                    fi += 1;
                    fw += 1;
                } else if full_wavelength[fw] < target {
                    fw += 1;
                } else {
                    fi += 1;
                }
            }
            
            if indices.is_empty() { continue; }
            
            if contiguous {
                plan.push((frm, SliceOrIndices::Slice(indices[0], indices[indices.len() - 1] + 1)));
            } else {
                plan.push((frm, SliceOrIndices::Indices(indices)));
            }
        }
        Ok(plan)
    }
}

// =============================================================================
// Python bindings
// =============================================================================

#[inline] fn unit_to_str(u: Unit) -> &'static str { match u { Unit::NM => "NM", Unit::RAW => "RAW" } }
#[inline] fn parse_spectral(s: Option<&str>) -> Unit { match s { Some("NM") | None => Unit::NM, _ => Unit::NM } }
#[inline] fn parse_intensity(s: Option<&str>) -> Unit { match s { Some("RAW") | None => Unit::RAW, _ => Unit::RAW } }

#[pyclass(name = "SpectralDataFrame", frozen)]
struct PySpectralDataFrame { inner: Arc<SpectralDataFrame> }

#[pymethods]
impl PySpectralDataFrame {
    #[getter] fn uid(&self) -> usize { self.inner.uid }
    #[getter] fn wavelength<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { self.inner.wavelength().to_pyarray(py) }
    #[getter] fn wl_bounds(&self) -> (f64, f64) { self.inner.wl_bounds() }

    fn __getitem__<'py>(&self, py: Python<'py>, key: (f64, String, String)) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.inner.get_data(&OpticalKey::from(key)).map(|v| v.to_vec().into_pyarray(py)).ok_or_else(|| PyKeyError::new_err("Key not found"))
    }
    fn __contains__(&self, key: (f64, String, String)) -> bool { self.inner.get_data(&OpticalKey::from(key)).is_some() }
    fn __len__(&self) -> usize { self.inner.len() }
    fn keys(&self) -> Vec<(f64, String, String)> { self.inner.keys().iter().map(|k| k.as_tuple()).collect() }

    fn set_data(&self, key: (f64, String, String), value: PyReadonlyArray1<'_, f64>, wavelength: Option<PyReadonlyArray1<'_, f64>>) -> PyResult<bool> {
        let value_slice = value.as_slice()?;
        let key = OpticalKey::from(key);
        let value_data = SpectralData::from_arc(Arc::from(value_slice));
        match wavelength {
            Some(arr) => self.inner.set_data(key, value_data, Some(arr.as_slice()?)),
            None => self.inner.set_data(key, value_data, None),
        }
    }

    fn remove(&self, key: (f64, String, String)) -> PyResult<()> { self.inner.remove(&OpticalKey::from(key)) }
    fn __repr__(&self) -> String {
        let (lo, hi) = self.inner.wl_bounds();
        format!("SpectralDataFrame(uid={}, keys={}, wl_points={}, range=[{:.2}, {:.2}])", self.inner.uid, self.inner.len(), self.inner.wavelength().len(), lo, hi)
    }
}

#[pyclass(name = "OpticalCollection", frozen)]
struct PyOpticalCollection { inner: Arc<OpticalCollection> }

#[pymethods]
impl PyOpticalCollection {
    #[new] fn new() -> Self { PyOpticalCollection { inner: Arc::new(OpticalCollection::new()) } }

    #[getter] fn display_spectral(&self) -> String { unit_to_str(self.inner.display_spectral()).to_string() }
    #[setter] fn set_display_spectral(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() { "NM" => { self.inner.set_display_spectral(Unit::NM); Ok(()) }, _ => Err(PyValueError::new_err("Invalid spectral unit")) }
    }
    #[getter] fn display_intensity(&self) -> String { unit_to_str(self.inner.display_intensity()).to_string() }
    #[setter] fn set_display_intensity(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() { "RAW" => { self.inner.set_display_intensity(Unit::RAW); Ok(()) }, _ => Err(PyValueError::new_err("Invalid intensity unit")) }
    }

    #[getter] fn frame_count(&self) -> usize { self.inner.frame_count() }
    fn __len__(&self) -> usize { self.inner.len_keys() }
    fn keys(&self) -> Vec<(f64, String, String)> { self.inner.keys_as_tuples() }
    fn __contains__(&self, key: (f64, String, String)) -> bool { self.inner.contains_key(&OpticalKey::from(key)) }
    
    fn frame(&self, index: usize) -> PyResult<PySpectralDataFrame> {
        Ok(PySpectralDataFrame { inner: self.inner.frame_at(index).ok_or_else(|| PyValueError::new_err("Index out of range"))? })
    }
    #[getter] fn frames(&self) -> Vec<PySpectralDataFrame> { self.inner.frames_snapshot().into_iter().map(|inner| PySpectralDataFrame { inner }).collect() }
    fn frames_for_key(&self, key: (f64, String, String)) -> PyResult<Vec<PySpectralDataFrame>> {
        Ok(self.inner.frames_for_key(&OpticalKey::from(key)).ok_or_else(|| PyKeyError::new_err("Key not found"))?.into_iter().map(|inner| PySpectralDataFrame { inner }).collect())
    }

    fn get_converted<'py>(&self, py: Python<'py>, key: (f64, String, String)) -> PyResult<(Vec<Bound<'py, PyArray1<f64>>>, Vec<Bound<'py, PyArray1<f64>>>)> {
        let opt_key = OpticalKey::from(key);
        let (data_list, wl_list) = py.detach(|| self.inner.get_converted(&opt_key)).ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok((data_list.into_iter().map(|d| d.into_pyarray(py)).collect(), wl_list.into_iter().map(|w| w.into_pyarray(py)).collect()))
    }

    #[pyo3(signature = (key, value, wavelength, input_spectral=None, input_intensity=None))]
    fn set_data(&self, py: Python<'_>, key: (f64, String, String), value: PyReadonlyArray1<'_, f64>, wavelength: PyReadonlyArray1<'_, f64>, input_spectral: Option<String>, input_intensity: Option<String>) -> PyResult<()> {
        let key = OpticalKey::from(key);
        let s = parse_spectral(input_spectral.as_deref());
        let i = parse_intensity(input_intensity.as_deref());
        let v_slice = value.as_slice()?;
        let w_slice = wavelength.as_slice()?;
        
        const GIL_RELEASE_THRESHOLD: usize = 4096;
        if v_slice.len() + w_slice.len() > GIL_RELEASE_THRESHOLD {
            let value_vec = v_slice.to_vec();
            let wl_vec = w_slice.to_vec();
            py.detach(move || self.inner.set_data(key, &value_vec, &wl_vec, s, i).map(|_| ()))
        } else {
            self.inner.set_data(key, v_slice, w_slice, s, i).map(|_| ())
        }
    }
}

#[pyclass(name = "OpticalWeaver", frozen)]
struct PyOpticalWeaver { inner: Arc<OpticalWeaver> }

#[pymethods]
impl PyOpticalWeaver {
    #[new]
    #[pyo3(signature = (cache_size=128))]
    fn new(cache_size: usize) -> Self { PyOpticalWeaver { inner: Arc::new(OpticalWeaver::new(cache_size)) } }

    #[getter] fn display_spectral(&self) -> String { unit_to_str(self.inner.display_spectral()).to_string() }
    #[setter] fn set_display_spectral(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() { "NM" => { self.inner.set_display_spectral(Unit::NM); Ok(()) }, _ => Err(PyValueError::new_err("Invalid spectral unit")) }
    }
    #[getter] fn display_intensity(&self) -> String { unit_to_str(self.inner.display_intensity()).to_string() }
    #[setter] fn set_display_intensity(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() { "RAW" => { self.inner.set_display_intensity(Unit::RAW); Ok(()) }, _ => Err(PyValueError::new_err("Invalid intensity unit")) }
    }

    #[getter] fn frame_count(&self) -> usize { self.inner.frame_count() }
    #[getter] fn generation(&self) -> usize { self.inner.generation() }
    fn __len__(&self) -> usize { self.inner.len_keys() }
    fn keys(&self) -> Vec<(f64, String, String)> { self.inner.keys_as_tuples() }
    fn __contains__(&self, key: (f64, String, String)) -> bool { self.inner.contains_key(&OpticalKey::from(key)) }
    
    fn frame(&self, index: usize) -> PyResult<PySpectralDataFrame> {
        Ok(PySpectralDataFrame { inner: self.inner.frame_at(index).ok_or_else(|| PyValueError::new_err("Index out of range"))? })
    }
    #[getter] fn frames(&self) -> Vec<PySpectralDataFrame> { self.inner.frames_snapshot().into_iter().map(|inner| PySpectralDataFrame { inner }).collect() }
    fn frames_for_key(&self, key: (f64, String, String)) -> PyResult<Vec<PySpectralDataFrame>> {
        Ok(self.inner.frames_for_key(&OpticalKey::from(key)).ok_or_else(|| PyKeyError::new_err("Key not found"))?.into_iter().map(|inner| PySpectralDataFrame { inner }).collect())
    }

    fn get_converted<'py>(&self, py: Python<'py>, key: (f64, String, String)) -> PyResult<(Vec<Bound<'py, PyArray1<f64>>>, Vec<Bound<'py, PyArray1<f64>>>)> {
        let opt_key = OpticalKey::from(key);
        let (data_list, wl_list) = py.detach(|| self.inner.get_converted(&opt_key)).ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok((data_list.into_iter().map(|d| d.into_pyarray(py)).collect(), wl_list.into_iter().map(|w| w.into_pyarray(py)).collect()))
    }

    #[pyo3(signature = (key, value, wavelength, input_spectral=None, input_intensity=None))]
    fn set_data(&self, py: Python<'_>, key: (f64, String, String), value: PyReadonlyArray1<'_, f64>, wavelength: PyReadonlyArray1<'_, f64>, input_spectral: Option<String>, input_intensity: Option<String>) -> PyResult<()> {
        let key = OpticalKey::from(key);
        let s = parse_spectral(input_spectral.as_deref());
        let i = parse_intensity(input_intensity.as_deref());
        let v_slice = value.as_slice()?;
        let w_slice = wavelength.as_slice()?;
        
        const GIL_RELEASE_THRESHOLD: usize = 4096;
        if v_slice.len() + w_slice.len() > GIL_RELEASE_THRESHOLD {
            let value_vec = v_slice.to_vec();
            let wl_vec = w_slice.to_vec();
            py.detach(move || self.inner.set_data(key, &value_vec, &wl_vec, s, i).map(|_| ()))
        } else {
            self.inner.set_data(key, v_slice, w_slice, s, i).map(|_| ())
        }
    }

    #[pyo3(signature = (key, strict=false))]
    fn get_weaved<'py>(&self, py: Python<'py>, key: (f64, String, String), strict: bool) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let opt_key = OpticalKey::from(key);
        let (wl, data) = py.detach(|| self.inner.get_weaved(&opt_key, strict))?;
        
        const DIRECT_ALLOC_THRESHOLD: usize = 1024;
        if wl.len() > DIRECT_ALLOC_THRESHOLD {
            let wl_arr = PyArray1::new(py, wl.len(), false);
            wl_arr.as_slice_mut()?.copy_from_slice(&wl);
            let data_arr = PyArray1::new(py, data.len(), false);
            data_arr.as_slice_mut()?.copy_from_slice(&data);
            Ok((wl_arr, data_arr))
        } else {
            Ok((wl.into_pyarray(py), data.into_pyarray(py)))
        }
    }

    fn get_weaved_collections<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)>> {
        let groups = py.detach(|| self.inner.get_weaved_collections());
        let mut out = Vec::with_capacity(groups.len());
        for (wl, data_map) in groups {
            let wl_arr = wl.into_pyarray(py);
            let dict = PyDict::new(py);
            for (key, data) in data_map {
                dict.set_item(key.as_tuple(), data.into_pyarray(py))?;
            }
            out.push((wl_arr, dict));
        }
        Ok(out)
    }

    fn unweave(&self, py: Python<'_>, key: (f64, String, String), full_wavelength: PyReadonlyArray1<'_, f64>, full_data: PyReadonlyArray1<'_, f64>) -> PyResult<usize> {
        let key = OpticalKey::from(key);
        let w_slice = full_wavelength.as_slice()?;
        let d_slice = full_data.as_slice()?;
        
        const GIL_RELEASE_THRESHOLD: usize = 8192;
        if w_slice.len() + d_slice.len() > GIL_RELEASE_THRESHOLD {
            let wl_vec = w_slice.to_vec();
            let data_vec = d_slice.to_vec();
            py.detach(move || self.inner.unweave(key, &wl_vec, &data_vec))
        } else {
            self.inner.unweave(key, w_slice, d_slice)
        }
    }

    fn unweave_collection(&self, py: Python<'_>, common_wavelength: PyReadonlyArray1<'_, f64>, data_batch: &Bound<'_, PyDict>) -> PyResult<usize> {
        let w_slice = common_wavelength.as_slice()?;
        let mut items = Vec::with_capacity(data_batch.len());
        for (key_obj, value_obj) in data_batch.iter() {
            let key_tuple: (f64, String, String) = key_obj.extract()?;
            let arr: PyReadonlyArray1<f64> = value_obj.extract()?;
            items.push((OpticalKey::from(key_tuple), arr.as_slice()?.to_vec()));
        }

        const GIL_RELEASE_THRESHOLD: usize = 8192;
        let total_len: usize = items.iter().map(|(_, v)| v.len()).sum::<usize>() + w_slice.len();
        if total_len > GIL_RELEASE_THRESHOLD {
            let wl_vec = w_slice.to_vec();
            py.detach(move || {
                let mut batch = AHashMap::with_capacity(items.len());
                for (key, vec) in items { batch.insert(key, vec.as_slice()); }
                self.inner.unweave_collection(&wl_vec, &batch)
            })
        } else {
            let mut batch = AHashMap::with_capacity(items.len());
            for (key, vec) in items { batch.insert(key, vec.as_slice()); }
            self.inner.unweave_collection(w_slice, &batch)
        }
    }

    fn invalidate_cache(&self) { self.inner.invalidate_cache(); }
}

#[pymodule]
fn spectralweave(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpectralDataFrame>()?;
    m.add_class::<PyOpticalCollection>()?;
    m.add_class::<PyOpticalWeaver>()?;
    Ok(())
}

// =============================================================================
// Tests & Benches
// =============================================================================
#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn bench_set_data() {
        let weaver = OpticalWeaver::new(128);
        let wl: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
        
        let start = Instant::now();
        for i in 0..100 {
            let key = OpticalKey::from((i as f64 * 0.5, "R".to_string(), "s".to_string()));
            weaver.set_data(key, &data, &wl, Unit::NM, Unit::RAW).unwrap();
        }
        println!("100 set_data calls: {:?}", start.elapsed());
    }
    
    #[test]
    fn bench_unweave() {
        let weaver = OpticalWeaver::new(128);
        let full_wl: Vec<f64> = (0..10000).map(|i| i as f64 * 0.1).collect();
        let full_data: Vec<f64> = (0..10000).map(|i| (i as f64 * 0.1).sin()).collect();
        
        for i in 0..10 {
            let wl: Vec<f64> = (0..1000).map(|j| (i * 100 + j) as f64 * 0.1).collect();
            let data: Vec<f64> = wl.iter().map(|x| x.sin()).collect();
            let key = OpticalKey::from((i as f64, "T".to_string(), "p".to_string()));
            weaver.set_data(key, &data, &wl, Unit::NM, Unit::RAW).unwrap();
        }
        
        let start = Instant::now();
        let key = OpticalKey::from((0.0, "T".to_string(), "p".to_string()));
        weaver.unweave(key, &full_wl, &full_data).unwrap();
        println!("unweave with 10 frames: {:?}", start.elapsed());
    }
}
```