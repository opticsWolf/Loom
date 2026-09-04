use std::sync::Arc;

use numpy::{PyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use navette::spectralweave::opticalweaver::{OpticalKey, SpectralData, wl_bits_eq};
use navette::spectralweave::targetweaver::{ResolvedNormMode, TargetKind, TargetWeaver};

use super::spectralweave_optical::PyOpticalWeaver;


// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
#[pyclass(name = "TargetWeaver", frozen)]
/// Optimization target store plus merit evaluation.
pub struct PyTargetWeaver {
    pub(crate) inner: Arc<TargetWeaver>,
}

#[pymethods]
impl PyTargetWeaver {
    #[new]
    #[pyo3(signature = (cache_size=128, tolerance_floor=1e-12))]
/// Target store with plan cache and merit-denominator floor.
    fn new(cache_size: usize, tolerance_floor: f64) -> Self {
        PyTargetWeaver {
            inner: Arc::new(TargetWeaver::new(cache_size, tolerance_floor)),
        }
    }

    #[pyo3(signature = (wavelengths, values, tolerances, angle, polarization, spectral, kind, norm_mode, band=None))]
/// Ingest one target curve over wavelengths (kind e/a/b/r/c, norm mode).
/// `band` holds optional per-point half-widths for `r`/`c` (raw units).
    fn add_spectral_target(
        &self,
        py: Python<'_>,
        wavelengths: PyReadonlyArray1<'_, f64>,
        values: PyReadonlyArray1<'_, f64>,
        tolerances: PyReadonlyArray1<'_, f64>,
        angle: f64,
        polarization: String,
        spectral: String,
        kind: String,
        norm_mode: String,
        band: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let k = TargetKind::from_str(&kind)
            .ok_or_else(|| PyValueError::new_err("Invalid kind (use 'e', 'a', 'b', 'r', or 'c')"))?;
        let wl = wavelengths.as_slice()?;
        let val = values.as_slice()?;
        let tol = tolerances.as_slice()?;
        let band_sl = band.as_ref().map(|b| b.as_slice()).transpose()?;
        if let Some(b) = band_sl.as_ref() {
            if b.len() != val.len() {
                return Err(PyValueError::new_err("band length must match values length"));
            }
        }
        let key = OpticalKey::from((angle, polarization, spectral));

        let wl_ptr = wl.as_ptr() as usize; let wl_len = wl.len();
        let val_ptr = val.as_ptr() as usize; let val_len = val.len();
        let tol_ptr = tol.as_ptr() as usize; let tol_len = tol.len();
        let (band_ptr, band_len) = match band_sl.as_ref() {
            Some(b) => (b.as_ptr() as usize, b.len()),
            None => (0usize, 0usize),
        };

        py.detach(move || -> PyResult<()> {
            let wl_data = unsafe { std::slice::from_raw_parts(wl_ptr as *const f64, wl_len) };
            let val_data = unsafe { std::slice::from_raw_parts(val_ptr as *const f64, val_len) };
            let tol_data = unsafe { std::slice::from_raw_parts(tol_ptr as *const f64, tol_len) };
            let band_data: &[f64] = if band_len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(band_ptr as *const f64, band_len) }
            };

            let frame = self
                .inner
                .create_dedicated_frame(wl_data)
                .map_err(PyValueError::new_err)?;
            let data_arc = SpectralData::from_arc(Arc::from(val_data));
            frame
                .set_data(key.clone(), data_arc, Some(wl_data))
                .map_err(PyValueError::new_err)?;
            self.inner.inner.inner.map_frame_to_key(&key, &frame);

            self.inner.register_metadata(frame.uid, key, val_data, tol_data, k, &norm_mode, band_data);
            Ok(())
        })
    }

    /// Export every ingested entry for converters (insertion order): one dict
    /// per (frame, key) with the grid, normalized values, tolerances, band,
    /// kind/mode codes and norm factor — everything `MeritSpec` needs.
    fn export_entries(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let tw = &self.inner;
        let meta = tw.target_metadata.read();
        let mut out = Vec::new();
        for frame in tw.inner.inner.frames_snapshot() {
            let wl: Vec<f64> = frame.wavelength().to_vec();
            let entries = match meta.get(&frame.uid) {
                Some(m) => m,
                None => continue,
            };
            for key in frame.keys() {
                let entry = match entries.entries.get(&key) {
                    Some(e) => e,
                    None => continue,
                };
                let (angle, polarization, spectral) = key.as_tuple();
                let d = PyDict::new(py);
                d.set_item("uid", frame.uid)?;
                d.set_item("angle", angle)?;
                d.set_item("polarization", polarization)?;
                d.set_item("spectral", spectral)?;
                d.set_item("wavelengths", PyArray::from_vec(py, wl.clone()))?;
                d.set_item("targets", PyArray::from_vec(py, entry.normalized_targets.to_vec()))?;
                d.set_item("tolerances", PyArray::from_vec(py, entry.tolerances.to_vec()))?;
                d.set_item("band", PyArray::from_vec(py, entry.band.to_vec()))?;
                d.set_item("kind", entry.kind.as_str())?;
                d.set_item("mode", entry.resolved_mode.as_str())?;
                d.set_item("norm_factor", entry.norm_factor)?;
                out.push(d.unbind());
            }
        }
        Ok(out)
    }

    #[pyo3(signature = (wavelength, angles, values, tolerances, polarization, spectral, kind, norm_mode, band=None))]
/// Ingest one target curve over angles (kind e/a/b/r/c, norm mode).
/// `band` holds optional per-point half-widths for `r`/`c` (raw units).
    fn add_angular_target(
        &self,
        py: Python<'_>,
        wavelength: f64,
        angles: PyReadonlyArray1<'_, f64>,
        values: PyReadonlyArray1<'_, f64>,
        tolerances: PyReadonlyArray1<'_, f64>,
        polarization: String,
        spectral: String,
        kind: String,
        norm_mode: String,
        band: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let k = TargetKind::from_str(&kind)
            .ok_or_else(|| PyValueError::new_err("Invalid kind (use 'e', 'a', 'b', 'r', or 'c')"))?;
        let angs = angles.as_slice()?;
        let vals = values.as_slice()?;
        let tols = tolerances.as_slice()?;
        let band_sl = band.as_ref().map(|b| b.as_slice()).transpose()?;
        if let Some(b) = band_sl.as_ref() {
            if b.len() != vals.len() {
                return Err(PyValueError::new_err("band length must match values length"));
            }
        }

        let a_ptr = angs.as_ptr() as usize; let a_len = angs.len();
        let v_ptr = vals.as_ptr() as usize; let v_len = vals.len();
        let t_ptr = tols.as_ptr() as usize; let t_len = tols.len();
        let (band_ptr, band_len) = match band_sl.as_ref() {
            Some(b) => (b.as_ptr() as usize, b.len()),
            None => (0usize, 0usize),
        };
        let pol = polarization.clone();
        let spec = spectral.clone();

        py.detach(move || -> PyResult<()> {
            let a_data = unsafe { std::slice::from_raw_parts(a_ptr as *const f64, a_len) };
            let v_data = unsafe { std::slice::from_raw_parts(v_ptr as *const f64, v_len) };
            let t_data = unsafe { std::slice::from_raw_parts(t_ptr as *const f64, t_len) };
            let b_data: &[f64] = if band_len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(band_ptr as *const f64, band_len) }
            };

            let wl_point = vec![wavelength];
            let frame = self
                .inner
                .create_dedicated_frame(&wl_point)
                .map_err(PyValueError::new_err)?;

            for i in 0..a_len {
                let key = OpticalKey::from((a_data[i], pol.clone(), spec.clone()));
                let val_arr = vec![v_data[i]];
                let tol_arr = vec![t_data[i]];
                let band_arr = if b_data.is_empty() { vec![] } else { vec![b_data[i]] };

                frame
                    .set_data(
                        key.clone(),
                        SpectralData::from_arc(Arc::from(val_arr.clone())),
                        Some(&wl_point),
                    )
                    .map_err(PyValueError::new_err)?;
                self.inner.inner.inner.map_frame_to_key(&key, &frame);

                self.inner.register_metadata(frame.uid, key, &val_arr, &tol_arr, k, &norm_mode, &band_arr);
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Zero-Allocation Merit Function
// ---------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (sim_weaver, target_weaver, missing_penalty=1e6))]
/// Merit of simulated weaves vs targets (exact/above/below residuals).
pub fn calculate_merit(
    py: Python<'_>,
    sim_weaver: &PyOpticalWeaver,
    target_weaver: &PyTargetWeaver,
    missing_penalty: f64,
) -> f64 {
    let sim_core = sim_weaver.inner.clone();
    let tw_core = target_weaver.inner.clone();

    py.detach(move || {
        let mut total_merit = 0.0;
        let target_keys = tw_core.inner.inner.keys();
        let meta_guard = tw_core.target_metadata.read();

        for key in target_keys {
            let sim_res = sim_core.get_weaved(&key);
            let (sim_wl, sim_val) = match sim_res {
                Ok((w, v)) if !w.is_empty() => (w, v),
                _ => {
                    total_merit += missing_penalty;
                    continue;
                }
            };

            let target_frames = match tw_core.inner.inner.frames_for_key(&key) {
                Some(f) => f,
                None => continue,
            };

            for frm in target_frames {
                let t_wl = frm.wavelength();
                if t_wl.is_empty() { continue; }

                let entry = match meta_guard.get(&frm.uid).and_then(|m| m.entries.get(&key)) {
                    Some(e) => e,
                    None => continue,
                };

                // Skip frames whose grid does not overlap the simulated curve.
                if sim_wl.last().map_or(true, |&l| l < t_wl[0])
                    || sim_wl.first().zip(t_wl.last()).map_or(true, |(&f, &l)| f > l)
                {
                    continue;
                }

                // Fast path: when the target grid coincides bit-for-bit with a
                // contiguous block of the simulated grid, read simulated values
                // directly — no interpolation, no per-point division. Mirrors the
                // OpticalWeaver `candidate == frame_wl` alignment shortcut, and is
                // the common case when the solver is sampled on the target grid.
                let offset = sim_wl.partition_point(|&x| x < t_wl[0]);
                let aligned = offset + t_wl.len() <= sim_wl.len()
                    && wl_bits_eq(&sim_wl[offset..offset + t_wl.len()], t_wl);

                // Misaligned case: two-pointer interpolation. `sim_idx` advances
                // monotonically across the sorted target grid, giving O(n + m)
                // instead of an O(m log n) per-point binary search.
                let mut sim_idx = 0;
                for i in 0..t_wl.len() {
                    let sim_raw = if aligned {
                        sim_val[offset + i]
                    } else {
                        let target_w = t_wl[i];
                        while sim_idx + 1 < sim_wl.len() && sim_wl[sim_idx + 1] < target_w {
                            sim_idx += 1;
                        }
                        if sim_idx + 1 < sim_wl.len() && sim_wl[sim_idx] <= target_w {
                            let w0 = sim_wl[sim_idx];
                            let w1 = sim_wl[sim_idx + 1];
                            let v0 = sim_val[sim_idx];
                            let v1 = sim_val[sim_idx + 1];
                            if (w1 - w0).abs() < 1e-14 {
                                v0
                            } else {
                                v0 + (target_w - w0) * (v1 - v0) / (w1 - w0)
                            }
                        } else if sim_idx < sim_wl.len() {
                            sim_val[sim_idx]
                        } else {
                            sim_val[sim_val.len() - 1]
                        }
                    };

                    let target_scaled = entry.normalized_targets[i];

                    let scaled_diff = match entry.resolved_mode {
                        ResolvedNormMode::Phase => {
                            // Wrap the residual to [-pi, pi] without trig — cheaper
                            // than the equivalent sin/cos/atan2 formulation.
                            let diff = sim_raw - target_scaled;
                            diff - std::f64::consts::TAU * (diff / std::f64::consts::TAU).round()
                        }
                        ResolvedNormMode::Log => {
                            sim_raw.max(1e-12).log10() * entry.norm_factor - target_scaled
                        }
                        ResolvedNormMode::Linear | ResolvedNormMode::Complex => {
                            sim_raw * entry.norm_factor - target_scaled
                        }
                    };

                    let tol = entry.tolerances[i];

                    let residual = match entry.kind {
                        TargetKind::Exact => (scaled_diff / tol).powi(2),
                        TargetKind::Above if scaled_diff < 0.0 => (scaled_diff / tol).powi(2),
                        TargetKind::Below if scaled_diff > 0.0 => (scaled_diff / tol).powi(2),
                        TargetKind::Range => {
                            // Hard box: bare `r` without a band falls back to
                            // the tolerance as half-width (paired a/b).
                            let bw = entry.band[i];
                            let bw_eff = if bw <= 0.0 { tol } else { bw };
                            let ad = scaled_diff.abs();
                            if ad <= bw_eff { 0.0 } else { ((ad - bw_eff) / tol).powi(2) }
                        },
                        TargetKind::CenterBand => {
                            // Soft box: reduced `(d/bw)^2` inside (exact scaled
                            // by `(tol/bw)^2`), exceedance plus continuity outside.
                            let bw = entry.band[i];
                            if bw <= 0.0 {
                                (scaled_diff / tol).powi(2)
                            } else {
                                let ad = scaled_diff.abs();
                                if ad <= bw { (scaled_diff / bw).powi(2) }
                                else { ((ad - bw) / tol).powi(2) + 1.0 }
                            }
                        },
                        _ => 0.0,
                    };

                    total_merit += residual;
                }
            }
        }
        total_merit
    })
}
