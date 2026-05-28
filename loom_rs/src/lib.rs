use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lru::LruCache;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use parking_lot::RwLock;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Unit system (mock – replace with real unit crate if needed)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unit {
    NM,
    RAW,
}

impl Unit {
    fn category(&self) -> &'static str {
        match self {
            Unit::NM => "spectral",
            Unit::RAW => "intensity",
        }
    }
}

/// Convert between units (mock: identity, but keeps API compatible)
fn convert_unit(value: &[f64], _from: Unit, _to: Unit) -> Vec<f64> {
    value.to_vec()
}

// ---------------------------------------------------------------------------
// Optical key: (wavelength, type, polarisation)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpticalKey {
    pub wavelength: f64,
    pub data_type: String,
    pub polarisation: String,
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

// ---------------------------------------------------------------------------
// SpectralDataFrame – thread‑safe, immutable wavelength grid
// ---------------------------------------------------------------------------
pub struct SpectralDataFrame {
    pub uid: usize,
    data: RwLock<HashMap<OpticalKey, Vec<f64>>>,
    wavelength: Arc<[f64]>, // immutable after creation
    wl_min: f64,
    wl_max: f64,
    wl_signature: Vec<u8>, // byte representation for exact comparison
}

impl SpectralDataFrame {
    fn new(wavelength: &[f64]) -> PyResult<Self> {
        static UID_GEN: AtomicUsize = AtomicUsize::new(0);
        let uid = UID_GEN.fetch_add(1, Ordering::SeqCst);

        // Validate strictly increasing
        if wavelength.len() > 1 {
            for i in 0..wavelength.len() - 1 {
                if wavelength[i] >= wavelength[i + 1] {
                    return Err(PyValueError::new_err(
                        "Wavelength must be strictly increasing",
                    ));
                }
            }
        }

        let wl_min = wavelength.first().copied().unwrap_or(f64::NEG_INFINITY);
        let wl_max = wavelength.last().copied().unwrap_or(f64::INFINITY);
        let wl_sig = wavelength.as_bytes().to_vec();

        Ok(SpectralDataFrame {
            uid,
            data: RwLock::new(HashMap::new()),
            wavelength: Arc::from(wavelength),
            wl_min,
            wl_max,
            wl_signature: wl_sig,
        })
    }

    /// Returns `true` if the key was newly inserted, `false` if updated.
    fn set_data(
        &self,
        key: OpticalKey,
        value: Vec<f64>,
        wavelength: Option<&[f64]>,
    ) -> PyResult<bool> {
        // Validate wavelength consistency
        if let Some(wl) = wavelength {
            let new_sig = wl.as_bytes().to_vec();
            if self.wl_signature != new_sig {
                return Err(PyValueError::new_err(
                    "SpectralDataFrame: wavelength grid conflict (bit-exact match required)",
                ));
            }
        }
        // Shape check
        if value.len() != self.wavelength.len() {
            return Err(PyValueError::new_err(format!(
                "Value length {} != wavelength length {}",
                value.len(),
                self.wavelength.len()
            )));
        }

        let mut data_guard = self.data.write();
        let is_new = !data_guard.contains_key(&key);
        data_guard.insert(key, value);
        Ok(is_new)
    }

    fn get_data(&self, key: &OpticalKey) -> Option<Vec<f64>> {
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
        let mut data_guard = self.data.write();
        if data_guard.remove(key).is_none() {
            return Err(PyKeyError::new_err("Key not found"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OpticalCollection – frame manager with deduplication
// ---------------------------------------------------------------------------
pub struct OpticalCollection {
    frames: RwLock<Vec<Arc<SpectralDataFrame>>>,
    wl_fingerprints: RwLock<HashMap<Vec<u8>, Arc<SpectralDataFrame>>>,
    key_map: RwLock<HashMap<OpticalKey, Vec<Arc<SpectralDataFrame>>>>,
    key_uid_sets: RwLock<HashMap<OpticalKey, HashSet<usize>>>,
    display_spectral: RwLock<Unit>,
    display_intensity: RwLock<Unit>,
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

    // ---- Display units ----
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

    // ---- Accessors ----
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

    pub fn frames_for_key(&self, key: &OpticalKey) -> Option<Vec<Arc<SpectralDataFrame>>> {
        self.key_map.read().get(key).cloned()
    }

    /// Get data with display‑unit conversion applied.
    pub fn get_converted(
        &self,
        key: &OpticalKey,
    ) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let frames = self.key_map.read().get(key)?.clone();
        let int_unit = *self.display_intensity.read();
        let spec_unit = *self.display_spectral.read();

        let mut data_list = Vec::with_capacity(frames.len());
        let mut wl_list = Vec::with_capacity(frames.len());

        for frm in frames {
            let data_raw = frm.get_data(key).unwrap();
            let data_converted = convert_unit(&data_raw, Unit::RAW, int_unit);
            let wl_converted = convert_unit(frm.wavelength(), Unit::NM, spec_unit);
            data_list.push(data_converted);
            wl_list.push(wl_converted);
        }
        Some((data_list, wl_list))
    }

    /// Public write path – creates a new frame if wavelength grid is new.
    pub fn set_data(
        &self,
        key: OpticalKey,
        value: Vec<f64>,
        wavelength: Vec<f64>,
        input_spectral: Unit,
        input_intensity: Unit,
    ) -> PyResult<()> {
        if value.len() != wavelength.len() {
            return Err(PyValueError::new_err("Length mismatch: value vs wavelength"));
        }

        // Convert to base units
        let base_wl = convert_unit(&wavelength, input_spectral, Unit::NM);
        let base_data = convert_unit(&value, input_intensity, Unit::RAW);

        let target_frame = self.get_or_create_frame(&base_wl)?;
        let is_new = target_frame.set_data(key.clone(), base_data, Some(&base_wl))?;

        if is_new {
            let mut key_map_guard = self.key_map.write();
            let mut key_uid_guard = self.key_uid_sets.write();
            let entry = key_map_guard.entry(key).or_insert_with(Vec::new);
            let uid_set = key_uid_guard.entry(key).or_insert_with(HashSet::new);
            if !uid_set.contains(&target_frame.uid) {
                entry.push(target_frame.clone());
                uid_set.insert(target_frame.uid);
            }
        }
        Ok(())
    }

    // ---- Internal helpers ----
    fn get_or_create_frame(&self, wl_arr: &[f64]) -> PyResult<Arc<SpectralDataFrame>> {
        let sig = wl_arr.as_bytes().to_vec();

        // Fast read
        {
            let fp_guard = self.wl_fingerprints.read();
            if let Some(frm) = fp_guard.get(&sig) {
                return Ok(frm.clone());
            }
        }

        // Write path (double‑checked)
        let mut fp_guard = self.wl_fingerprints.write();
        if let Some(frm) = fp_guard.get(&sig) {
            return Ok(frm.clone());
        }

        let new_frame = Arc::new(SpectralDataFrame::new(wl_arr)?);
        self.frames.write().push(new_frame.clone());
        fp_guard.insert(sig, new_frame.clone());
        Ok(new_frame)
    }
}

// ---------------------------------------------------------------------------
// Distribution plan – slice or explicit indices
// ---------------------------------------------------------------------------
#[derive(Clone)]
enum SliceOrIndices {
    Slice(usize, usize),
    Indices(Vec<usize>),
}

type DistributionPlan = Vec<(Arc<SpectralDataFrame>, SliceOrIndices)>;

// ---------------------------------------------------------------------------
// OpticalWeaver – adds generation‑based caching of distribution plans
// ---------------------------------------------------------------------------
pub struct OpticalWeaver {
    inner: OpticalCollection,
    distribution_cache: RwLock<LruCache<Vec<u8>, (usize, DistributionPlan)>>,
    generation: AtomicUsize,
    max_cache_size: usize,
}

impl OpticalWeaver {
    pub fn new(cache_size: usize) -> Self {
        OpticalWeaver {
            inner: OpticalCollection::new(),
            distribution_cache: RwLock::new(LruCache::new(cache_size.try_into().unwrap())),
            generation: AtomicUsize::new(0),
            max_cache_size: cache_size,
        }
    }

    fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    // ---- Delegated OpticalCollection methods ----
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
    pub fn frames_for_key(&self, key: &OpticalKey) -> Option<Vec<Arc<SpectralDataFrame>>> {
        self.inner.frames_for_key(key)
    }
    pub fn get_converted(
        &self,
        key: &OpticalKey,
    ) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        self.inner.get_converted(key)
    }

    // Override set_data to bump generation
    pub fn set_data(
        &self,
        key: OpticalKey,
        value: Vec<f64>,
        wavelength: Vec<f64>,
        input_spectral: Unit,
        input_intensity: Unit,
    ) -> PyResult<()> {
        let old_frame_count = self.inner.frame_count();
        self.inner.set_data(key, value, wavelength, input_spectral, input_intensity)?;
        if self.inner.frame_count() != old_frame_count {
            self.bump_generation();
        } else {
            // Even if no new frame, a new mapping for this key might have been added
            self.bump_generation();
        }
        Ok(())
    }

    // ---- Weave: reconstruct continuous curves ----
    pub fn get_weaved(&self, key: &OpticalKey) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let frames = self
            .inner
            .frames_for_key(key)
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;

        let mut fragments = Vec::new();
        for frm in frames {
            let wl = frm.wavelength().to_vec();
            let data = frm.get_data(key).unwrap(); // safe because frame contains key
            let min_wl = frm.wl_bounds().0;
            fragments.push((min_wl, wl, data));
        }

        if fragments.is_empty() {
            return Ok((vec![], vec![]));
        }

        fragments.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Overlap detection (warning in Python, we just print to stderr)
        for i in 0..fragments.len() - 1 {
            let cur_max = fragments[i].1.last().copied().unwrap_or(f64::NEG_INFINITY);
            let nxt_min = fragments[i + 1].0;
            if cur_max > nxt_min {
                eprintln!(
                    "Warning: get_weaved(key={:?}): frames overlap at {} > {}",
                    key, cur_max, nxt_min
                );
            }
        }

        let mut all_wl = Vec::new();
        let mut all_data = Vec::new();
        for (_, wl, data) in fragments {
            all_wl.extend(wl);
            all_data.extend(data);
        }
        Ok((all_wl, all_data))
    }

    pub fn get_weaved_collections(
        &self,
    ) -> Vec<(Vec<f64>, HashMap<OpticalKey, Vec<f64>>)> {
        let mut groups: HashMap<Vec<u8>, (Vec<f64>, HashMap<OpticalKey, Vec<f64>>)> =
            HashMap::new();

        for key in self.inner.keys() {
            if let Ok((wl, data)) = self.get_weaved(&key) {
                if wl.is_empty() {
                    continue;
                }
                let sig = wl.as_bytes().to_vec();
                let entry = groups.entry(sig).or_insert_with(|| (wl.clone(), HashMap::new()));
                entry.1.insert(key, data);
            }
        }
        groups.into_values().collect()
    }

    // ---- Unweave: distribute full curves into frames ----
    pub fn unweave(
        &self,
        key: OpticalKey,
        full_wavelength: &[f64],
        full_data: &[f64],
    ) -> PyResult<usize> {
        let plan = self.resolve_plan(full_wavelength)?;
        let mut updated = 0;

        for (frm, indices) in plan {
            let subset = match indices {
                SliceOrIndices::Slice(start, end) => full_data[start..end].to_vec(),
                SliceOrIndices::Indices(idx_vec) => idx_vec.iter().map(|&i| full_data[i]).collect(),
            };
            let is_new = frm.set_data(key.clone(), subset, Some(full_wavelength))?;
            if is_new {
                // Register mapping
                let mut key_map_guard = self.inner.key_map.write();
                let mut key_uid_guard = self.inner.key_uid_sets.write();
                let entry = key_map_guard.entry(key.clone()).or_insert_with(Vec::new);
                let uid_set = key_uid_guard.entry(key.clone()).or_insert_with(HashSet::new);
                if !uid_set.contains(&frm.uid) {
                    entry.push(frm.clone());
                    uid_set.insert(frm.uid);
                }
            }
            updated += 1;
        }
        Ok(updated)
    }

    pub fn unweave_collection(
        &self,
        common_wavelength: &[f64],
        data_batch: HashMap<OpticalKey, Vec<f64>>,
    ) -> PyResult<usize> {
        let plan = self.resolve_plan(common_wavelength)?;
        let mut total_updated = 0;

        for (key, full_data) in data_batch {
            for (frm, indices) in &plan {
                let subset = match indices {
                    SliceOrIndices::Slice(start, end) => &full_data[*start..*end],
                    SliceOrIndices::Indices(idx_vec) => {
                        let mut v = Vec::with_capacity(idx_vec.len());
                        for &i in idx_vec {
                            v.push(full_data[i]);
                        }
                        v
                    }
                };
                let is_new = frm.set_data(key.clone(), subset, Some(common_wavelength))?;
                if is_new {
                    let mut key_map_guard = self.inner.key_map.write();
                    let mut key_uid_guard = self.inner.key_uid_sets.write();
                    let entry = key_map_guard.entry(key.clone()).or_insert_with(Vec::new);
                    let uid_set = key_uid_guard.entry(key.clone()).or_insert_with(HashSet::new);
                    if !uid_set.contains(&frm.uid) {
                        entry.push(frm.clone());
                        uid_set.insert(frm.uid);
                    }
                }
                total_updated += 1;
            }
        }
        Ok(total_updated)
    }

    // ---- Cache management ----
    pub fn invalidate_cache(&self) {
        self.distribution_cache.write().clear();
    }

    // ---- Internal plan resolution ----
    fn resolve_plan(&self, full_wavelength: &[f64]) -> PyResult<DistributionPlan> {
        let sig = full_wavelength.as_bytes().to_vec();
        let gen = self.generation.load(Ordering::SeqCst);

        // Check cache
        {
            let mut cache = self.distribution_cache.write();
            if let Some((cached_gen, plan)) = cache.get(&sig) {
                if *cached_gen == gen {
                    // Promote to MRU (LruCache does this automatically on get)
                    return Ok(plan.clone());
                } else {
                    cache.pop(&sig);
                }
            }
        }

        // Build plan
        let plan = self.build_distribution_plan(full_wavelength)?;
        // Insert into cache
        {
            let mut cache = self.distribution_cache.write();
            cache.put(sig, (gen, plan.clone()));
        }
        Ok(plan)
    }

    fn build_distribution_plan(&self, full_wavelength: &[f64]) -> PyResult<DistributionPlan> {
        let frames_snapshot = self.inner.frames.read().clone();
        let mut plan = Vec::new();

        if full_wavelength.is_empty() {
            return Ok(plan);
        }

        let fw_min = full_wavelength[0];
        let fw_max = full_wavelength[full_wavelength.len() - 1];

        for frm in frames_snapshot {
            let frame_wl = frm.wavelength();
            let (f_min, f_max) = frm.wl_bounds();

            // Quick disjoint test
            if f_min > fw_max || f_max < fw_min {
                continue;
            }

            // Narrow search window via binary search
            let idx_lo = full_wavelength.partition_point(|&x| x < f_min);
            let idx_hi = full_wavelength.partition_point(|&x| x <= f_max);
            let candidate = &full_wavelength[idx_lo..idx_hi];

            // Fast path: exact contiguous match
            if candidate.len() == frame_wl.len() && candidate == frame_wl {
                plan.push((frm, SliceOrIndices::Slice(idx_lo, idx_hi)));
                continue;
            }

            // Slow path: match each point using searchsorted
            let mut indices = Vec::with_capacity(frame_wl.len());
            for &target in frame_wl {
                let pos = full_wavelength.partition_point(|&x| x < target);
                if pos < full_wavelength.len() && (full_wavelength[pos] - target).abs() < 1e-12 {
                    indices.push(pos);
                }
            }
            if indices.is_empty() {
                continue;
            }

            // Check contiguity
            let is_contiguous = indices
                .iter()
                .enumerate()
                .all(|(i, &idx)| i == 0 || idx == indices[i - 1] + 1);
            if is_contiguous {
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
// Python bindings
// =============================================================================

#[pyclass]
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
    fn wavelength(&self, py: Python) -> Option<Py<PyArray1<f64>>> {
        Some(self.inner.wavelength().to_pyarray(py).to_owned())
    }

    #[getter]
    fn wl_bounds(&self) -> (f64, f64) {
        self.inner.wl_bounds()
    }

    fn __getitem__(&self, key: (f64, String, String)) -> PyResult<Py<PyArray1<f64>>> {
        let opt_key = OpticalKey::from(key);
        self.inner
            .get_data(&opt_key)
            .map(|v| Python::with_gil(|py| v.to_pyarray(py).to_owned()))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))
    }

    fn __contains__(&self, key: (f64, String, String)) -> bool {
        self.inner.get_data(&OpticalKey::from(key)).is_some()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn keys(&self) -> Vec<(f64, String, String)> {
        self.inner
            .keys()
            .into_iter()
            .map(|k| (k.wavelength, k.data_type, k.polarisation))
            .collect()
    }

    fn set_data(
        &self,
        key: (f64, String, String),
        value: PyReadonlyArray1<f64>,
        wavelength: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<bool> {
        let value_vec = value.as_slice()?.to_vec();
        let wl_opt = wavelength.map(|arr| arr.as_slice().unwrap());
        let opt_key = OpticalKey::from(key);
        self.inner.set_data(opt_key, value_vec, wl_opt)
    }

    fn remove(&self, key: (f64, String, String)) -> PyResult<()> {
        self.inner.remove(&OpticalKey::from(key))
    }

    fn __repr__(&self) -> String {
        format!(
            "SpectralDataFrame(uid={}, keys={}, wl_points={})",
            self.inner.uid,
            self.inner.len(),
            self.inner.wavelength().len()
        )
    }
}

#[pyclass]
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
        match self.inner.display_spectral() {
            Unit::NM => "NM".to_string(),
            Unit::RAW => "RAW".to_string(),
        }
    }

    #[setter]
    fn set_display_spectral(&self, unit_str: String) -> PyResult<()> {
        let unit = match unit_str.as_str() {
            "NM" => Unit::NM,
            "RAW" => Unit::RAW,
            _ => return Err(PyValueError::new_err("Invalid spectral unit")),
        };
        self.inner.set_display_spectral(unit);
        Ok(())
    }

    #[getter]
    fn display_intensity(&self) -> String {
        match self.inner.display_intensity() {
            Unit::NM => "NM".to_string(),
            Unit::RAW => "RAW".to_string(),
        }
    }

    #[setter]
    fn set_display_intensity(&self, unit_str: String) -> PyResult<()> {
        let unit = match unit_str.as_str() {
            "RAW" => Unit::RAW,
            _ => return Err(PyValueError::new_err("Invalid intensity unit")),
        };
        self.inner.set_display_intensity(unit);
        Ok(())
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    fn __len__(&self) -> usize {
        self.inner.len_keys()
    }

    fn keys(&self) -> Vec<(f64, String, String)> {
        self.inner
            .keys()
            .into_iter()
            .map(|k| (k.wavelength, k.data_type, k.polarisation))
            .collect()
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
        let guard = self.inner.frames.read();
        guard
            .iter()
            .map(|f| PySpectralDataFrame { inner: f.clone() })
            .collect()
    }

    fn frames_for_key(&self, key: (f64, String, String)) -> PyResult<Vec<PySpectralDataFrame>> {
        let frames = self
            .inner
            .frames_for_key(&OpticalKey::from(key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok(frames
            .into_iter()
            .map(|f| PySpectralDataFrame { inner: f })
            .collect())
    }

    fn get_converted(
        &self,
        key: (f64, String, String),
        py: Python,
    ) -> PyResult<(Vec<Py<PyArray1<f64>>>, Vec<Py<PyArray1<f64>>>)> {
        let (data_list, wl_list) = self
            .inner
            .get_converted(&OpticalKey::from(key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        let py_data = data_list
            .into_iter()
            .map(|d| d.to_pyarray(py).to_owned())
            .collect();
        let py_wl = wl_list
            .into_iter()
            .map(|w| w.to_pyarray(py).to_owned())
            .collect();
        Ok((py_data, py_wl))
    }

    fn set_data(
        &self,
        key: (f64, String, String),
        value: PyReadonlyArray1<f64>,
        wavelength: PyReadonlyArray1<f64>,
        input_spectral: Option<String>,
        input_intensity: Option<String>,
    ) -> PyResult<()> {
        let value_vec = value.as_slice()?.to_vec();
        let wl_vec = wavelength.as_slice()?.to_vec();
        let spec = match input_spectral.as_deref() {
            Some("NM") | None => Unit::NM,
            _ => Unit::RAW,
        };
        let intens = match input_intensity.as_deref() {
            Some("RAW") | None => Unit::RAW,
            _ => Unit::RAW,
        };
        self.inner
            .set_data(OpticalKey::from(key), value_vec, wl_vec, spec, intens)
    }
}

#[pyclass]
struct PyOpticalWeaver {
    inner: Arc<OpticalWeaver>,
}

#[pymethods]
impl PyOpticalWeaver {
    #[new]
    fn new(cache_size: Option<usize>) -> Self {
        let size = cache_size.unwrap_or(128);
        PyOpticalWeaver {
            inner: Arc::new(OpticalWeaver::new(size)),
        }
    }

    #[getter]
    fn display_spectral(&self) -> String {
        match self.inner.display_spectral() {
            Unit::NM => "NM".to_string(),
            Unit::RAW => "RAW".to_string(),
        }
    }

    #[setter]
    fn set_display_spectral(&self, unit_str: String) -> PyResult<()> {
        let unit = match unit_str.as_str() {
            "NM" => Unit::NM,
            "RAW" => Unit::RAW,
            _ => return Err(PyValueError::new_err("Invalid spectral unit")),
        };
        self.inner.set_display_spectral(unit);
        Ok(())
    }

    #[getter]
    fn display_intensity(&self) -> String {
        match self.inner.display_intensity() {
            Unit::NM => "NM".to_string(),
            Unit::RAW => "RAW".to_string(),
        }
    }

    #[setter]
    fn set_display_intensity(&self, unit_str: String) -> PyResult<()> {
        let unit = match unit_str.as_str() {
            "RAW" => Unit::RAW,
            _ => return Err(PyValueError::new_err("Invalid intensity unit")),
        };
        self.inner.set_display_intensity(unit);
        Ok(())
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    fn __len__(&self) -> usize {
        self.inner.len_keys()
    }

    fn keys(&self) -> Vec<(f64, String, String)> {
        self.inner
            .keys()
            .into_iter()
            .map(|k| (k.wavelength, k.data_type, k.polarisation))
            .collect()
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
        let guard = self.inner.inner.frames.read();
        guard
            .iter()
            .map(|f| PySpectralDataFrame { inner: f.clone() })
            .collect()
    }

    fn frames_for_key(&self, key: (f64, String, String)) -> PyResult<Vec<PySpectralDataFrame>> {
        let frames = self
            .inner
            .frames_for_key(&OpticalKey::from(key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        Ok(frames
            .into_iter()
            .map(|f| PySpectralDataFrame { inner: f })
            .collect())
    }

    fn get_converted(
        &self,
        key: (f64, String, String),
        py: Python,
    ) -> PyResult<(Vec<Py<PyArray1<f64>>>, Vec<Py<PyArray1<f64>>>)> {
        let (data_list, wl_list) = self
            .inner
            .get_converted(&OpticalKey::from(key))
            .ok_or_else(|| PyKeyError::new_err("Key not found"))?;
        let py_data = data_list
            .into_iter()
            .map(|d| d.to_pyarray(py).to_owned())
            .collect();
        let py_wl = wl_list
            .into_iter()
            .map(|w| w.to_pyarray(py).to_owned())
            .collect();
        Ok((py_data, py_wl))
    }

    fn set_data(
        &self,
        key: (f64, String, String),
        value: PyReadonlyArray1<f64>,
        wavelength: PyReadonlyArray1<f64>,
        input_spectral: Option<String>,
        input_intensity: Option<String>,
    ) -> PyResult<()> {
        let value_vec = value.as_slice()?.to_vec();
        let wl_vec = wavelength.as_slice()?.to_vec();
        let spec = match input_spectral.as_deref() {
            Some("NM") | None => Unit::NM,
            _ => Unit::RAW,
        };
        let intens = match input_intensity.as_deref() {
            Some("RAW") | None => Unit::RAW,
            _ => Unit::RAW,
        };
        self.inner
            .set_data(OpticalKey::from(key), value_vec, wl_vec, spec, intens)
    }

    fn get_weaved(&self, key: (f64, String, String), py: Python) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let (wl, data) = self.inner.get_weaved(&OpticalKey::from(key))?;
        Ok((wl.to_pyarray(py).to_owned(), data.to_pyarray(py).to_owned()))
    }

    fn get_weaved_collections(&self, py: Python) -> Vec<(Py<PyArray1<f64>>, Py<PyDict>)> {
        let groups = self.inner.get_weaved_collections();
        groups
            .into_iter()
            .map(|(wl, data_map)| {
                let wl_arr = wl.to_pyarray(py).to_owned();
                let dict = PyDict::new(py);
                for (key, data) in data_map {
                    let key_tuple = (key.wavelength, key.data_type, key.polarisation);
                    let data_arr = data.to_pyarray(py).to_owned();
                    dict.set_item(key_tuple, data_arr).unwrap();
                }
                (wl_arr, dict)
            })
            .collect()
    }

    fn unweave(
        &self,
        key: (f64, String, String),
        full_wavelength: PyReadonlyArray1<f64>,
        full_data: PyReadonlyArray1<f64>,
    ) -> PyResult<usize> {
        let wl_slice = full_wavelength.as_slice()?;
        let data_slice = full_data.as_slice()?;
        self.inner
            .unweave(OpticalKey::from(key), wl_slice, data_slice)
    }

    fn unweave_collection(
        &self,
        common_wavelength: PyReadonlyArray1<f64>,
        data_batch: &PyDict,
    ) -> PyResult<usize> {
        let wl_slice = common_wavelength.as_slice()?;
        let mut batch = HashMap::new();
        for (key_obj, value_obj) in data_batch.iter() {
            let key_tuple: (f64, String, String) = key_obj.extract()?;
            let arr = value_obj.extract::<PyReadonlyArray1<f64>>()?;
            batch.insert(OpticalKey::from(key_tuple), arr.as_slice()?.to_vec());
        }
        self.inner.unweave_collection(wl_slice, batch)
    }

    fn invalidate_cache(&self) {
        self.inner.invalidate_cache();
    }
}

// ---------------------------------------------------------------------------
// Module initialisation
// ---------------------------------------------------------------------------
#[pymodule]
fn loom_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySpectralDataFrame>()?;
    m.add_class::<PyOpticalCollection>()?;
    m.add_class::<PyOpticalWeaver>()?;
    Ok(())
}