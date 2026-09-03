use std::sync::Arc;

use ahash::AHashMap;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use navette::spectralweave::opticalweaver::{
    OpticalCollection, OpticalKey, OpticalWeaver, SpectralData, SpectralDataFrame, Unit,
    parse_intensity, parse_spectral, unit_to_str,
};


// =============================================================================
// Python bindings (Optical Core)
// =============================================================================

#[pyclass(name = "SpectralDataFrame", frozen)]
pub struct PySpectralDataFrame {
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
        let value_data = SpectralData::from_arc(Arc::from(value_slice));
        match wavelength {
            Some(arr) => {
                let wl = arr.as_slice()?;
                self.inner
                    .set_data(key, value_data, Some(wl))
                    .map_err(PyValueError::new_err)
            }
            None => self.inner
                .set_data(key, value_data, None)
                .map_err(PyValueError::new_err),
        }
    }
    fn remove(&self, key: (f64, String, String)) -> PyResult<()> {
        self.inner
            .remove(&OpticalKey::from(key))
            .map_err(PyValueError::new_err)
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

#[pyclass(name = "OpticalCollection", frozen)]
pub struct PyOpticalCollection {
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
        let v_slice = value.as_slice()?;
        let v_ptr = v_slice.as_ptr() as usize;
        let v_len = v_slice.len();
        let w_slice = wavelength.as_slice()?;
        let w_ptr = w_slice.as_ptr() as usize;
        let w_len = w_slice.len();
        py.detach(move || {
            let value_data = unsafe { std::slice::from_raw_parts(v_ptr as *const f64, v_len) };
            let wl_data = unsafe { std::slice::from_raw_parts(w_ptr as *const f64, w_len) };
            self.inner
                .set_data(key, value_data, wl_data, s, i)
                .map(|_| ())
                .map_err(PyValueError::new_err)
        })
    }
}

#[pyclass(name = "OpticalWeaver", frozen)]
pub struct PyOpticalWeaver {
    pub(crate) inner: Arc<OpticalWeaver>,
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
        unit_to_str(self.inner.inner.display_spectral()).to_string()
    }
    #[setter]
    fn set_display_spectral(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() {
            "NM" => {
                self.inner.inner.set_display_spectral(Unit::NM);
                Ok(())
            }
            _ => Err(PyValueError::new_err("Invalid spectral unit")),
        }
    }
    #[getter]
    fn display_intensity(&self) -> String {
        unit_to_str(self.inner.inner.display_intensity()).to_string()
    }
    #[setter]
    fn set_display_intensity(&self, unit_str: String) -> PyResult<()> {
        match unit_str.as_str() {
            "RAW" => {
                self.inner.inner.set_display_intensity(Unit::RAW);
                Ok(())
            }
            _ => Err(PyValueError::new_err("Invalid intensity unit")),
        }
    }
    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.inner.frame_count()
    }
    #[getter]
    fn generation(&self) -> usize {
        self.inner.generation()
    }
    fn __len__(&self) -> usize {
        self.inner.inner.len_keys()
    }
    fn keys(&self) -> Vec<(f64, String, String)> {
        self.inner.inner.keys().iter().map(|k| k.as_tuple()).collect()
    }
    fn __contains__(&self, key: (f64, String, String)) -> bool {
        self.inner.inner.contains_key(&OpticalKey::from(key))
    }
    fn frame(&self, index: usize) -> PyResult<PySpectralDataFrame> {
        let frm = self
            .inner
            .inner
            .frame_at(index)
            .ok_or_else(|| PyValueError::new_err("Index out of range"))?;
        Ok(PySpectralDataFrame { inner: frm })
    }
    #[getter]
    fn frames(&self) -> Vec<PySpectralDataFrame> {
        self.inner
            .inner
            .frames_snapshot()
            .into_iter()
            .map(|inner| PySpectralDataFrame { inner })
            .collect()
    }
    fn frames_for_key(&self, key: (f64, String, String)) -> PyResult<Vec<PySpectralDataFrame>> {
        let frames = self
            .inner
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
            .detach(|| self.inner.inner.get_converted(&opt_key))
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
            self.inner
                .set_data(key, value_data, wl_data, s, i)
                .map_err(PyValueError::new_err)
        })
    }
    fn get_weaved<'py>(
        &self,
        py: Python<'py>,
        key: (f64, String, String),
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let opt_key = OpticalKey::from(key);
        let (wl, data) = py
            .detach(|| self.inner.get_weaved(&opt_key))
            .map_err(PyValueError::new_err)?;
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
            self.inner
                .unweave(key, wl, data)
                .map_err(PyValueError::new_err)
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
        let mut items = Vec::with_capacity(data_batch.len());
        let mut py_arrays = Vec::with_capacity(data_batch.len());
        for (key_obj, value_obj) in data_batch.iter() {
            let key_tuple: (f64, String, String) = key_obj.extract()?;
            let arr: PyReadonlyArray1<f64> = value_obj.extract()?;
            let slice = arr.as_slice()?;
            items.push((key_tuple, slice.as_ptr() as usize, slice.len()));
            py_arrays.push(arr);
        }
        py.detach(move || {
            let wl = unsafe { std::slice::from_raw_parts(w_ptr as *const f64, w_len) };
            let mut batch = AHashMap::with_capacity(items.len());
            for (key_tuple, ptr_addr, len) in items {
                let slice = unsafe { std::slice::from_raw_parts(ptr_addr as *const f64, len) };
                batch.insert(OpticalKey::from(key_tuple), slice);
            }
            self.inner
                .unweave_collection(wl, batch)
                .map_err(PyValueError::new_err)
        })
    }
    fn invalidate_cache(&self) {
        self.inner.invalidate_cache();
    }
}
