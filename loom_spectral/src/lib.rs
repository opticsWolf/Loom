//! Loom (Rust core) — weaving spectral fragments into continuous curves.
//!
//! Built against **pyo3 0.28** (Bound API, `Python::attach` / `Python::detach`).
//!
//! Design choice (improved over the Python original): frames own an
//! *immutable* wavelength grid set at construction. The grid is validated
//! before the frame is ever registered, removing a class of partial-state
//! bugs the Python version had (where a frame was appended to the registry
//! before `_apply_wavelength` got a chance to reject the grid).
//!
//! ### Performance notes (with improvements)
//! * Data inside a frame is stored as `Arc<[f64]>` – cloning is O(1) and
//!   read‑only access is allocation‑free.
//! * NumPy arrays are mapped directly to `&[f64]` inside detached GIL closures
//!   via raw pointer extraction for true zero-copy writes across thread boundaries.
//! * `get_weaved` concatenates directly from memory references, bypassing 
//!   intermediate vector cloning.
//! * Unit conversion uses `Cow` to skip allocations when units already match.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lru::LruCache;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
use parking_lot::RwLock;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

// ---------------------------------------------------------------------------
// Unit system (mock — drop in a real unit crate if desired)
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

#[inline]
fn wl_signature(wl: &[f64]) -> Vec<u8> {
    bytemuck::cast_slice::<f64, u8>(wl).to_vec()
}

// ---------------------------------------------------------------------------
// Optical key: (wavelength, type, polarisation)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct OpticalKey {
    pub wavelength: f64,
    pub data_type: String,
    pub polarisation: String,
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
            data_type: t.1,
            polarisation: t.2,
        }
    }
}

impl OpticalKey {
    #[inline]
    fn as_tuple(&self) -> (f64, String, String) {
        (self.wavelength, self.data_type.clone(), self.polarisation.clone())
    }
}

// ---------------------------------------------------------------------------
// SpectralDataFrame — thread-safe, immutable wavelength grid
// ---------------------------------------------------------------------------
pub struct SpectralDataFrame {
    pub uid: usize,
    data: RwLock<HashMap<OpticalKey, Arc<[f64]>>>,
    wavelength: Arc<[f64]>,
    wl_min: f64,
    wl_max: f64,
    wl_signature: Vec<u8>,
}

impl SpectralDataFrame {
    fn new(wavelength: &[f64]) -> PyResult<Self> {
        static UID_GEN: AtomicUsize = AtomicUsize::new(0);
        let uid = UID_GEN.fetch_add(1, Ordering::SeqCst);

        if wavelength.is_empty() {
            return Err(PyValueError::new_err(
                "SpectralDataFrame: wavelength array must be non-empty.",
            ));
        }
        for i in 0..wavelength.len() - 1 {
            if !(wavelength[i] < wavelength[i + 1]) {
                return Err(PyValueError::new_err(
                    "SpectralDataFrame: wavelength array must be strictly monotonically increasing.",
                ));
            }
        }

        let wl_min = wavelength[0];
        let wl_max = wavelength[wavelength.len() - 1];
        let wl_sig = wl_signature(wavelength);

        Ok(SpectralDataFrame {
            uid,
            data: RwLock::new(HashMap::new()),
            wavelength: Arc::from(wavelength),
            wl_min,
            wl_max,
            wl_signature: wl_sig,
        })
    }

    fn set_data(
        &self,
        key: OpticalKey,
        value: Arc<[f64]>,
        wavelength: Option<&[f64]>,
    ) -> PyResult<bool> {
        if let Some(wl) = wavelength {
            if self.wl_signature != wl_signature(wl) {
                return Err(PyValueError::new_err(format!(
                    "SpectralDataFrame(uid={}): wavelength grid conflict (bit-exact match required).",
                    self.uid
                )));
            }
        }
        if value.len() != self.wavelength.len() {
            return Err(PyValueError::new_err(format!(
                "SpectralDataFrame(uid={}): value length {} != wavelength length {}",
                self.uid,
                value.len(),
                self.wavelength.len()
            )));
        }

        let mut guard = self.data.write();
        let is_new = !guard.contains_key(&key);
        guard.insert(key, value);
        Ok(is_new)
    }

    fn get_data(&self, key: &OpticalKey) -> Option<Arc<[f64]>> {
        self.data.read().get(key).cloned()
    }

    fn keys(&self) -> Vec<OpticalKey> {
        self.data.read().keys().cloned().collect()
    }

    fn len(&self) -> usize {
        self.data.read().len()
    }

    fn wavelength(&self) -> &[f64] {
        &self.wavelength
    }

    fn wl_bounds(&self) -> (f64, f64) {
        (self.wl_min, self.wl_max)
    }

    fn remove(&self, key: &OpticalKey) -> PyResult<()> {
        if self.data.write().remove(key).is_none() {
            return Err(PyKeyError::new_err("Key not found"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OpticalCollection — frame manager with deduplication
// ---------------------------------------------------------------------------
pub struct OpticalCollection {
    frames: RwLock<Vec<Arc<SpectralDataFrame>>>,
    wl_fingerprints: RwLock<HashMap<Vec<u8>, Arc<SpectralDataFrame>>>,
    key_map: RwLock<HashMap<OpticalKey, Vec<Arc<SpectralDataFrame>>>>,
    key_uid_sets: RwLock<HashMap<OpticalKey, HashSet<usize>>>,
    display_spectral: RwLock<Unit>,
    display_intensity: RwLock<Unit>,
}

impl Default for OpticalCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl OpticalCollection {
    pub fn new() -> Self {
        OpticalCollection {
            frames: RwLock::new(Vec::new()),
            wl_fingerprints: RwLock::new(HashMap::new()),
            key_map: RwLock::new(HashMap::new()),
            key_uid_sets: RwLock::new(HashMap::new()),
            display_spectral: RwLock::new(Unit::NM),
            display_intensity: RwLock::new(Unit::RAW),
        }
    }

    pub fn set_display_spectral(&self, unit: Unit) {
        *self.display_spectral.write() = unit;
    }
    pub fn display_spectral(&self) -> Unit {
        *self.display_spectral.read()
    }
    pub fn set_display_intensity(&self, unit: Unit) {
        *self.display_intensity.write() = unit;
    }
    pub fn display_intensity(&self) -> Unit {
        *self.display_intensity.read()
    }

    pub fn frame_count(&self) -> usize {
        self.frames.read().len()
    }
    pub fn len_keys(&self) -> usize {
        self.key_map.read().len()
    }
    pub fn keys(&self) -> Vec<OpticalKey> {
        self.key_map.read().keys().cloned().collect()
    }
    pub fn contains_key(&self, key: &OpticalKey) -> bool {
        self.key_map.read().contains_key(key)
    }
    pub fn frame_at(&self, index: usize) -> Option<Arc<SpectralDataFrame>> {
        self.frames.read().get(index).cloned()
    }
    pub fn frames_snapshot(&self) -> Vec<Arc<SpectralDataFrame>> {
        self.frames.read().clone()
    }
    pub fn frames_for_key(&self, key: &OpticalKey) -> Option<Vec<Arc<SpectralDataFrame>>> {
        self.key_map.read().get(key).cloned()
    }

    pub fn get_converted(&self, key: &OpticalKey) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let frames = self.key_map.read().get(key)?.clone();
        let int_unit = *self.display_intensity.read();
        let spec_unit = *self.display_spectral.read();

        let mut data_list = Vec::with_capacity(frames.len());
        let mut wl_list = Vec::with_capacity(frames.len());
        for frm in frames {
            let data_raw = frm.get_data(key).unwrap_or_else(|| Arc::from(&[][..]));
            data_list.push(convert_unit(&data_raw, Unit::RAW, int_unit).into_owned());
            wl_list.push(convert_unit(frm.wavelength(), Unit::NM, spec_unit).into_owned());
        }
        Some((data_list, wl_list))
    }

    fn map_frame_to_key(&self, key: &OpticalKey, frame: &Arc<SpectralDataFrame>) -> bool {
        let mut key_map = self.key_map.write();
        let mut key_uids = self.key_uid_sets.write();
        let uid_set = key_uids.entry(key.clone()).or_default();
        if uid_set.insert(frame.uid) {
            key_map.entry(key.clone()).or_default().push(frame.clone());
            true
        } else {
            false
        }
    }

    pub fn set_data(
        &self,
        key: OpticalKey,
        value: &[f64],
        wavelength: &[f64],
        input_spectral: Unit,
        input_intensity: Unit,
    ) -> PyResult<bool> {
        if value.len() != wavelength.len() {
            return Err(PyValueError::new_err(format!(
                "Length mismatch: value ({}) vs wavelength ({}).",
                value.len(),
                wavelength.len()
            )));
        }

        let base_wl = convert_unit(wavelength, input_spectral, Unit::NM);
        let base_data = convert_unit(value, input_intensity, Unit::RAW);

        let (target_frame, frame_created) = self.get_or_create_frame(&base_wl)?;

        let data_arc: Arc<[f64]> = match base_data {
            Cow::Borrowed(b) => Arc::from(b),
            Cow::Owned(o) => Arc::from(o),
        };

        let is_new = target_frame.set_data(key.clone(), data_arc, Some(&base_wl))?;
        if is_new {
            self.map_frame_to_key(&key, &target_frame);
        }
        Ok(frame_created)
    }

    fn get_or_create_frame(&self, wl_arr: &[f64]) -> PyResult<(Arc<SpectralDataFrame>, bool)> {
        let sig = wl_signature(wl_arr);
        {
            let fp = self.wl_fingerprints.read();
            if let Some(frm) = fp.get(&sig) {
                return Ok((frm.clone(), false));
            }
        }
        let mut fp = self.wl_fingerprints.write();
        if let Some(frm) = fp.get(&sig) {
            return Ok((frm.clone(), false));
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
    fn gather(&self, data: &[f64]) -> Arc<[f64]> {
        match self {
            SliceOrIndices::Slice(s, e) => Arc::from(&data[*s..*e]),
            SliceOrIndices::Indices(idx) => {
                let vec: Vec<f64> = idx.iter().map(|&i| data[i]).collect();
                Arc::from(vec)
            }
        }
    }
}

type DistributionPlan = Vec<(Arc<SpectralDataFrame>, SliceOrIndices)>;

// ---------------------------------------------------------------------------
// OpticalWeaver — generation-based caching of distribution plans
// ---------------------------------------------------------------------------
pub struct OpticalWeaver {
    inner: OpticalCollection,
    distribution_cache: RwLock<LruCache<Vec<u8>, (usize, DistributionPlan)>>,
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

    fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
    pub fn generation(&self) -> usize {
        self.generation.load(Ordering::SeqCst)
    }

    pub fn display_spectral(&self) -> Unit {
        self.inner.display_spectral()
    }
    pub fn set_display_spectral(&self, unit: Unit) {
        self.inner.set_display_spectral(unit);
    }
    pub fn display_intensity(&self) -> Unit {
        self.inner.display_intensity()
    }
    pub fn set_display_intensity(&self, unit: Unit) {
        self.inner.set_display_intensity(unit);
    }
    pub fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }
    pub fn len_keys(&self) -> usize {
        self.inner.len_keys()
    }
    pub fn keys(&self) -> Vec<OpticalKey> {
        self.inner.keys()
    }
    pub fn contains_key(&self, key: &OpticalKey) -> bool {
        self.inner.contains_key(key)
    }
    pub fn frame_at(&self, index: usize) -> Option<Arc<SpectralDataFrame>> {
        self.inner.frame_at(index)
    }
    pub fn frames_snapshot(&self) -> Vec<Arc<SpectralDataFrame>> {
        self.inner.frames_snapshot()
    }
    pub fn frames_for_key(&self, key: &OpticalKey) -> Option<Vec<Arc<SpectralDataFrame>>> {
        self.inner.frames_for_key(key)
    }
    pub fn get_converted(&self, key: &OpticalKey) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        self.inner.get_converted(key)
    }

    pub fn set_data(
        &self,
        key: OpticalKey,
        value: &[f64],
        wavelength: &[f64],
        input_spectral: Unit,
        input_intensity: Unit,
    ) -> PyResult<()> {
        let frame_created =
            self.inner
                .set_data(key, value, wavelength, input_spectral, input_intensity)?;
        if frame_created {
            self.bump_generation();
        }
        Ok(())
    }

    pub fn get_weaved(&self, key: &OpticalKey) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let frames = self
            .inner
            .frames_for_key(key)
            .ok_or_else(|| PyKeyError::new_err("Key not found."))?;

        let mut fragments: Vec<(f64, &[f64], Arc<[f64]>)> = Vec::with_capacity(frames.len());
        for frm in &frames {
            if let Some(data) = frm.get_data(key) {
                fragments.push((frm.wl_bounds().0, frm.wavelength(), data));
            }
        }
        if fragments.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        fragments.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..fragments.len() - 1 {
            let cur_max = fragments[i].1.last().copied().unwrap_or(f64::NEG_INFINITY);
            let nxt_min = fragments[i + 1].0;
            if cur_max > nxt_min {
                eprintln!(
                    "Warning: get_weaved(key=({}, {:?}, {:?})): frames overlap at {} > {}; \
                     concatenated grid may be non-monotonic.",
                    key.wavelength, key.data_type, key.polarisation, cur_max, nxt_min
                );
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

    pub fn get_weaved_collections(&self) -> Vec<(Vec<f64>, HashMap<OpticalKey, Vec<f64>>)> {
        let mut groups: HashMap<Vec<u8>, (Vec<f64>, HashMap<OpticalKey, Vec<f64>>)> =
            HashMap::new();

        for key in self.inner.keys() {
            if let Ok((wl, data)) = self.get_weaved(&key) {
                if wl.is_empty() {
                    continue;
                }
                let sig = wl_signature(&wl);
                let entry = groups.entry(sig).or_insert_with(|| (wl.clone(), HashMap::new()));
                entry.1.insert(key, data);
            }
        }
        groups.into_values().collect()
    }

    pub fn unweave(
        &self,
        key: OpticalKey,
        full_wavelength: &[f64],
        full_data: &[f64],
    ) -> PyResult<usize> {
        if full_data.len() != full_wavelength.len() {
            return Err(PyValueError::new_err(format!(
                "unweave: full_data length {} != full_wavelength length {}",
                full_data.len(),
                full_wavelength.len()
            )));
        }

        let plan = self.resolve_plan(full_wavelength)?;
        let mut updated = 0;
        for (frm, indices) in plan {
            let subset = indices.gather(full_data);
            let is_new = frm.set_data(key.clone(), subset, None)?;
            if is_new {
                self.inner.map_frame_to_key(&key, &frm);
            }
            updated += 1;
        }
        Ok(updated)
    }

    pub fn unweave_collection(
        &self,
        common_wavelength: &[f64],
        data_batch: HashMap<OpticalKey, &[f64]>,
    ) -> PyResult<usize> {
        if data_batch.is_empty() {
            return Ok(0);
        }
        for (k, v) in &data_batch {
            if v.len() != common_wavelength.len() {
                return Err(PyValueError::new_err(format!(
                    "unweave_collection: data for key ({}, {:?}, {:?}) has length {} \
                     != common_wavelength length {}",
                    k.wavelength,
                    k.data_type,
                    k.polarisation,
                    v.len(),
                    common_wavelength.len()
                )));
            }
        }

        let plan = self.resolve_plan(common_wavelength)?;
        let mut total = 0;
        for (key, full_data) in data_batch {
            for (frm, indices) in &plan {
                let subset = indices.gather(full_data);
                let is_new = frm.set_data(key.clone(), subset, None)?;
                if is_new {
                    self.inner.map_frame_to_key(&key, frm);
                }
                total += 1;
            }
        }
        Ok(total)
    }

    pub fn invalidate_cache(&self) {
        self.distribution_cache.write().clear();
    }

    fn resolve_plan(&self, full_wavelength: &[f64]) -> PyResult<DistributionPlan> {
        let sig = wl_signature(full_wavelength);
        let current_gen = self.generation.load(Ordering::SeqCst);
        {
            let mut cache = self.distribution_cache.write();
            if let Some((cached_gen, plan)) = cache.get(&sig) {
                if *cached_gen == current_gen {
                    return Ok(plan.clone());
                }
            }
            cache.pop(&sig);
        }
        let plan = self.build_distribution_plan(full_wavelength)?;
        self.distribution_cache.write().put(sig, (current_gen, plan.clone()));
        Ok(plan)
    }

    fn build_distribution_plan(&self, full_wavelength: &[f64]) -> PyResult<DistributionPlan> {
        let frames_snapshot = self.inner.frames_snapshot();
        let mut plan = Vec::new();
        if full_wavelength.is_empty() {
            return Ok(plan);
        }

        let fw_min = full_wavelength[0];
        let fw_max = full_wavelength[full_wavelength.len() - 1];

        for frm in frames_snapshot {
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

            let mut indices = Vec::with_capacity(frame_wl.len());
            for &target in frame_wl {
                let pos = full_wavelength.partition_point(|&x| x < target);
                if pos < full_wavelength.len()
                    && (full_wavelength[pos] - target).abs() < 1e-12
                {
                    indices.push(pos);
                }
            }
            if indices.is_empty() {
                continue;
            }

            let contiguous = indices
                .iter()
                .enumerate()
                .all(|(i, &idx)| i == 0 || idx == indices[i - 1] + 1);
            if contiguous {
                plan.push((
                    frm,
                    SliceOrIndices::Slice(indices[0], indices[indices.len() - 1] + 1),
                ));
            } else {
                plan.push((frm, SliceOrIndices::Indices(indices)));
            }
        }
        Ok(plan)
    }
}

// =============================================================================
// Python bindings (pyo3 0.28 Bound API)
// =============================================================================

#[inline]
fn unit_to_str(u: Unit) -> &'static str {
    match u {
        Unit::NM => "NM",
        Unit::RAW => "RAW",
    }
}

#[inline]
fn parse_spectral(s: Option<&str>) -> Unit {
    match s {
        Some("NM") | None => Unit::NM,
        _ => Unit::NM,
    }
}

#[inline]
fn parse_intensity(s: Option<&str>) -> Unit {
    match s {
        Some("RAW") | None => Unit::RAW,
        _ => Unit::RAW,
    }
}

#[pyclass(frozen)]
struct PySpectralDataFrame {
    inner: Arc<SpectralDataFrame>,
}

#[pymethods]
impl PySpectralDataFrame {
    #[getter]
    fn uid(&self) -> usize {
        self.inner.uid
    }

    #[getter]
    fn wavelength<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.wavelength().to_pyarray(py)
    }

    #[getter]
    fn wl_bounds(&self) -> (f64, f64) {
        self.inner.wl_bounds()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: (f64, String, String),
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.inner
            .get_data(&OpticalKey::from(key))
            .map(|v| v.to_vec().into_pyarray(py))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))
    }

    fn __contains__(&self, key: (f64, String, String)) -> bool {
        self.inner.get_data(&OpticalKey::from(key)).is_some()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn keys(&self) -> Vec<(f64, String, String)> {
        self.inner.keys().iter().map(|k| k.as_tuple()).collect()
    }

    fn set_data(
        &self,
        key: (f64, String, String),
        value: PyReadonlyArray1<'_, f64>,
        wavelength: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<bool> {
        let value_slice = value.as_slice()?;
        let key = OpticalKey::from(key);
        let value_arc: Arc<[f64]> = Arc::from(value_slice);
        match wavelength {
            Some(arr) => {
                let wl = arr.as_slice()?;
                self.inner.set_data(key, value_arc, Some(wl))
            }
            None => self.inner.set_data(key, value_arc, None),
        }
    }

    fn remove(&self, key: (f64, String, String)) -> PyResult<()> {
        self.inner.remove(&OpticalKey::from(key))
    }

    fn __repr__(&self) -> String {
        let (lo, hi) = self.inner.wl_bounds();
        format!(
            "SpectralDataFrame(uid={}, keys={}, wl_points={}, range=[{:.2}, {:.2}])",
            self.inner.uid,
            self.inner.len(),
            self.inner.wavelength().len(),
            lo,
            hi
        )
    }
}

#[pyclass(frozen)]
struct PyOpticalCollection {
    inner: Arc<OpticalCollection>,
}

#[pymethods]
impl PyOpticalCollection {
    #[new]
    fn new() -> Self {
        PyOpticalCollection {
            inner: Arc::new(OpticalCollection::new()),
        }
    }

    #[getter]
    fn display_spectral(&self) -> String {
        unit_to_str(self.inner.display_spectral()).to_string()
    }
    #[setter]
    fn set_display_spectral(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() {
            "NM" => {
                self.inner.set_display_spectral(Unit::NM);
                Ok(())
            }
            _ => Err(PyValueError::new_err("Invalid spectral unit")),
        }
    }
    #[getter]
    fn display_intensity(&self) -> String {
        unit_to_str(self.inner.display_intensity()).to_string()
    }
    #[setter]
    fn set_display_intensity(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() {
            "RAW" => {
                self.inner.set_display_intensity(Unit::RAW);
                Ok(())
            }
            _ => Err(PyValueError::new_err("Invalid intensity unit")),
        }
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }
    fn __len__(&self) -> usize {
        self.inner.len_keys()
    }
    fn keys(&self) -> Vec<(f64, String, String)> {
        self.inner.keys().iter().map(|k| k.as_tuple()).collect()
    }
    fn __contains__(&self, key: (f64, String, String)) -> bool {
        self.inner.contains_key(&OpticalKey::from(key))
    }
    fn frame(&self, index: usize) -> PyResult<PySpectralDataFrame> {
        let frm = self
            .inner
            .frame_at(index)
            .ok_or_else(|| PyValueError::new_err("Index out of range"))?;
        Ok(PySpectralDataFrame { inner: frm })
    }
    #[getter]
    fn frames(&self) -> Vec<PySpectralDataFrame> {
        self.inner
            .frames_snapshot()
            .into_iter()
            .map(|inner| PySpectralDataFrame { inner })
            .collect()
    }
    fn frames_for_key(&self, key: (f64, String, String)) -> PyResult<Vec<PySpectralDataFrame>> {
        let frames = self
            .inner
            .frames_for_key(&OpticalKey::from(key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok(frames
            .into_iter()
            .map(|inner| PySpectralDataFrame { inner })
            .collect())
    }

    fn get_converted<'py>(
        &self,
        py: Python<'py>,
        key: (f64, String, String),
    ) -> PyResult<(Vec<Bound<'py, PyArray1<f64>>>, Vec<Bound<'py, PyArray1<f64>>>)> {
        let opt_key = OpticalKey::from(key);
        let (data_list, wl_list) = py
            .detach(|| self.inner.get_converted(&opt_key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok((
            data_list.into_iter().map(|d| d.into_pyarray(py)).collect(),
            wl_list.into_iter().map(|w| w.into_pyarray(py)).collect(),
        ))
    }

    #[pyo3(signature = (key, value, wavelength, input_spectral=None, input_intensity=None))]
    fn set_data(
        &self,
        py: Python<'_>,
        key: (f64, String, String),
        value: PyReadonlyArray1<'_, f64>,
        wavelength: PyReadonlyArray1<'_, f64>,
        input_spectral: Option<String>,
        input_intensity: Option<String>,
    ) -> PyResult<()> {
        let key = OpticalKey::from(key);
        let s = parse_spectral(input_spectral.as_deref());
        let i = parse_intensity(input_intensity.as_deref());
        
        // Extract raw pointers as `usize` (which is thread-safe to Send)
        let v_slice = value.as_slice()?;
        let v_ptr = v_slice.as_ptr() as usize; 
        let v_len = v_slice.len();

        let w_slice = wavelength.as_slice()?;
        let w_ptr = w_slice.as_ptr() as usize;
        let w_len = w_slice.len();
        
        py.detach(move || {
            // Cast back to *const f64 inside the detached thread
            let value_data = unsafe { std::slice::from_raw_parts(v_ptr as *const f64, v_len) };
            let wl_data = unsafe { std::slice::from_raw_parts(w_ptr as *const f64, w_len) };
            self.inner.set_data(key, value_data, wl_data, s, i).map(|_| ())
        })
    }
}

#[pyclass(frozen)]
struct PyOpticalWeaver {
    inner: Arc<OpticalWeaver>,
}

#[pymethods]
impl PyOpticalWeaver {
    #[new]
    #[pyo3(signature = (cache_size=128))]
    fn new(cache_size: usize) -> Self {
        PyOpticalWeaver {
            inner: Arc::new(OpticalWeaver::new(cache_size)),
        }
    }

    #[getter]
    fn display_spectral(&self) -> String {
        unit_to_str(self.inner.display_spectral()).to_string()
    }
    #[setter]
    fn set_display_spectral(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() {
            "NM" => {
                self.inner.set_display_spectral(Unit::NM);
                Ok(())
            }
            _ => Err(PyValueError::new_err("Invalid spectral unit")),
        }
    }
    #[getter]
    fn display_intensity(&self) -> String {
        unit_to_str(self.inner.display_intensity()).to_string()
    }
    #[setter]
    fn set_display_intensity(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() {
            "RAW" => {
                self.inner.set_display_intensity(Unit::RAW);
                Ok(())
            }
            _ => Err(PyValueError::new_err("Invalid intensity unit")),
        }
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }
    #[getter]
    fn generation(&self) -> usize {
        self.inner.generation()
    }
    fn __len__(&self) -> usize {
        self.inner.len_keys()
    }
    fn keys(&self) -> Vec<(f64, String, String)> {
        self.inner.keys().iter().map(|k| k.as_tuple()).collect()
    }
    fn __contains__(&self, key: (f64, String, String)) -> bool {
        self.inner.contains_key(&OpticalKey::from(key))
    }
    fn frame(&self, index: usize) -> PyResult<PySpectralDataFrame> {
        let frm = self
            .inner
            .frame_at(index)
            .ok_or_else(|| PyValueError::new_err("Index out of range"))?;
        Ok(PySpectralDataFrame { inner: frm })
    }
    #[getter]
    fn frames(&self) -> Vec<PySpectralDataFrame> {
        self.inner
            .frames_snapshot()
            .into_iter()
            .map(|inner| PySpectralDataFrame { inner })
            .collect()
    }
    fn frames_for_key(&self, key: (f64, String, String)) -> PyResult<Vec<PySpectralDataFrame>> {
        let frames = self
            .inner
            .frames_for_key(&OpticalKey::from(key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok(frames
            .into_iter()
            .map(|inner| PySpectralDataFrame { inner })
            .collect())
    }

    fn get_converted<'py>(
        &self,
        py: Python<'py>,
        key: (f64, String, String),
    ) -> PyResult<(Vec<Bound<'py, PyArray1<f64>>>, Vec<Bound<'py, PyArray1<f64>>>)> {
        let opt_key = OpticalKey::from(key);
        let (data_list, wl_list) = py
            .detach(|| self.inner.get_converted(&opt_key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok((
            data_list.into_iter().map(|d| d.into_pyarray(py)).collect(),
            wl_list.into_iter().map(|w| w.into_pyarray(py)).collect(),
        ))
    }

    #[pyo3(signature = (key, value, wavelength, input_spectral=None, input_intensity=None))]
    fn set_data(
        &self,
        py: Python<'_>,
        key: (f64, String, String),
        value: PyReadonlyArray1<'_, f64>,
        wavelength: PyReadonlyArray1<'_, f64>,
        input_spectral: Option<String>,
        input_intensity: Option<String>,
    ) -> PyResult<()> {
        let key = OpticalKey::from(key);
        let s = parse_spectral(input_spectral.as_deref());
        let i = parse_intensity(input_intensity.as_deref());
        
        let v_slice = value.as_slice()?;
        let v_ptr = v_slice.as_ptr() as usize;
        let v_len = v_slice.len();

        let w_slice = wavelength.as_slice()?;
        let w_ptr = w_slice.as_ptr() as usize;
        let w_len = w_slice.len();
        
        py.detach(move || {
            let value_data = unsafe { std::slice::from_raw_parts(v_ptr as *const f64, v_len) };
            let wl_data = unsafe { std::slice::from_raw_parts(w_ptr as *const f64, w_len) };
            self.inner.set_data(key, value_data, wl_data, s, i)
        })
    }

    fn get_weaved<'py>(
        &self,
        py: Python<'py>,
        key: (f64, String, String),
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let opt_key = OpticalKey::from(key);
        let (wl, data) = py.detach(|| self.inner.get_weaved(&opt_key))?;
        Ok((wl.into_pyarray(py), data.into_pyarray(py)))
    }

    fn get_weaved_collections<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)>> {
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

    fn unweave(
        &self,
        py: Python<'_>,
        key: (f64, String, String),
        full_wavelength: PyReadonlyArray1<'_, f64>,
        full_data: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<usize> {
        let key = OpticalKey::from(key);
        
        let w_slice = full_wavelength.as_slice()?;
        let w_ptr = w_slice.as_ptr() as usize;
        let w_len = w_slice.len();
        
        let d_slice = full_data.as_slice()?;
        let d_ptr = d_slice.as_ptr() as usize;
        let d_len = d_slice.len();

        py.detach(move || {
            let wl = unsafe { std::slice::from_raw_parts(w_ptr as *const f64, w_len) };
            let data = unsafe { std::slice::from_raw_parts(d_ptr as *const f64, d_len) };
            self.inner.unweave(key, wl, data)
        })
    }

    fn unweave_collection(
        &self,
        py: Python<'_>,
        common_wavelength: PyReadonlyArray1<'_, f64>,
        data_batch: &Bound<'_, PyDict>,
    ) -> PyResult<usize> {
        let w_slice = common_wavelength.as_slice()?;
        let w_ptr = w_slice.as_ptr() as usize;
        let w_len = w_slice.len();

        // Accumulate pointers (as usize) and length while holding the GIL
        let mut items = Vec::with_capacity(data_batch.len());
        let mut py_arrays = Vec::with_capacity(data_batch.len());
        
        for (key_obj, value_obj) in data_batch.iter() {
            let key_tuple: (f64, String, String) = key_obj.extract()?;
            let arr: PyReadonlyArray1<f64> = value_obj.extract()?;
            let slice = arr.as_slice()?;
            
            // Store pointer as usize
            items.push((key_tuple, slice.as_ptr() as usize, slice.len())); 
            py_arrays.push(arr);
        }

        py.detach(move || {
            let wl = unsafe { std::slice::from_raw_parts(w_ptr as *const f64, w_len) };
            let mut batch = HashMap::with_capacity(items.len());
            
            for (key_tuple, ptr_addr, len) in items {
                // Reconstruct from the usize address
                let slice = unsafe { std::slice::from_raw_parts(ptr_addr as *const f64, len) };
                batch.insert(OpticalKey::from(key_tuple), slice);
            }
            
            self.inner.unweave_collection(wl, batch)
        })
    }

    fn invalidate_cache(&self) {
        self.inner.invalidate_cache();
    }
}

// ---------------------------------------------------------------------------
// Module initialisation
// ---------------------------------------------------------------------------
#[pymodule]
fn loom_spectral(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpectralDataFrame>()?;
    m.add_class::<PyOpticalCollection>()?;
    m.add_class::<PyOpticalWeaver>()?;
    Ok(())
}