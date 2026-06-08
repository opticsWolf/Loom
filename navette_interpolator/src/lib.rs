use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array1, Array2};
use rayon::prelude::*;

// Threshold above which a single-signal evaluation is split across threads.
const PAR_TARGET_THRESHOLD: usize = 8_192;
// Minimum work per thread chunk, to keep scheduling overhead negligible.
const MIN_PAR_CHUNK: usize = 1_024;

// -------------------------------------------------------------------------
// Auxiliary data
// -------------------------------------------------------------------------
#[derive(Clone)]
enum AuxData {
    None,
    Slopes(Array2<f64>),
    FHWeights(Array1<f64>),
}

// -------------------------------------------------------------------------
// Extrapolation modes
// -------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtrapMode {
    Linear,
    Clamp,
    Error,
}

impl ExtrapMode {
    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(ExtrapMode::Linear),
            "clamp" => Ok(ExtrapMode::Clamp),
            "error" => Ok(ExtrapMode::Error),
            _ => Err(PyValueError::new_err(
                "extrap must be 'linear', 'clamp', or 'error'",
            )),
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            ExtrapMode::Linear => "linear",
            ExtrapMode::Clamp => "clamp",
            ExtrapMode::Error => "error",
        }
    }
}

// -------------------------------------------------------------------------
// Main spline struct
// -------------------------------------------------------------------------
#[pyclass]
pub struct UniInterpolator {
    x: Array1<f64>,
    y: Array2<f64>,
    method: String,
    robust: bool,
    d: usize,
    is_batch: bool,
    aux_data: AuxData,
    extrap: ExtrapMode,
}

// -------------------------------------------------------------------------
// Python-visible methods
// -------------------------------------------------------------------------
#[pymethods]
impl UniInterpolator {
    #[new]
    #[pyo3(signature = (x, y, method="pchip", robust=false, d=3, extrap="linear"))]
    fn new<'py>(
        x: PyReadonlyArray1<'py, f64>,
        y: Bound<'py, PyAny>,
        method: &str,
        robust: bool,
        mut d: usize,
        extrap: &str,
    ) -> PyResult<Self> {
        let method = method.to_lowercase();
        let extrap_mode = ExtrapMode::from_str(extrap)?;
        let x_arr = x.as_array().to_owned();
        let n = x_arr.len();

        if n < 2 {
            return Err(PyValueError::new_err("x must have at least 2 points"));
        }
        for i in 1..n {
            if x_arr[i] <= x_arr[i - 1] {
                return Err(PyValueError::new_err("x must be strictly increasing"));
            }
        }

        let (y_arr, is_batch) = if let Ok(y_2d) = y.extract::<PyReadonlyArray2<f64>>() {
            let arr = y_2d.as_array().to_owned();
            if arr.ncols() != n {
                return Err(PyValueError::new_err(format!(
                    "y row length ({}) must match x length ({})",
                    arr.ncols(),
                    n
                )));
            }
            (arr, true)
        } else if let Ok(y_1d) = y.extract::<PyReadonlyArray1<f64>>() {
            let arr_1d = y_1d.as_array().to_owned();
            if arr_1d.len() != n {
                return Err(PyValueError::new_err(format!(
                    "y length ({}) must match x length ({})",
                    arr_1d.len(),
                    n
                )));
            }
            let arr_2d = arr_1d
                .into_shape_with_order((1, n))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (arr_2d, false)
        } else {
            return Err(PyValueError::new_err("y must be a 1D or 2D numpy array"));
        };

        // Clamp Floater-Hormann degree into a valid range.
        if d >= n {
            d = n.saturating_sub(1);
        }

        match method.as_str() {
            "sprague" => {
                if n < 6 {
                    return Err(PyValueError::new_err("Sprague requires at least 6 points"));
                }
            }
            "pchip" | "makima" | "floater_hormann" | "fh" | "linear" => {}
            _ => {
                return Err(PyValueError::new_err(format!("Unknown method: {}", method)));
            }
        }

        let aux_data = match method.as_str() {
            "pchip" => AuxData::Slopes(calc_pchip_slopes(&x_arr, &y_arr)),
            "makima" => AuxData::Slopes(calc_makima_slopes(&x_arr, &y_arr)),
            "floater_hormann" | "fh" => AuxData::FHWeights(calc_fh_weights(&x_arr, d)),
            _ => AuxData::None,
        };

        Ok(UniInterpolator {
            x: x_arr,
            y: y_arr,
            method,
            robust,
            d,
            is_batch,
            aux_data,
            extrap: extrap_mode,
        })
    }

    #[pyo3(name = "__call__")]
    #[pyo3(signature = (target_x, deriv=0, sorted_hint=None))]
    fn call<'py>(
        &self,
        py: Python<'py>,
        target_x: PyReadonlyArray1<'py, f64>,
        deriv: usize,
        sorted_hint: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tgt_x = target_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("target_x must be contiguous"))?;
        let n_tgt = tgt_x.len();

        if n_tgt == 0 {
            let empty = if self.is_batch {
                Array2::<f64>::zeros((self.y.nrows(), 0))
                    .into_pyarray(py)
                    .into_any()
            } else {
                Array1::<f64>::zeros(0).into_pyarray(py).into_any()
            };
            return Ok(empty);
        }

        let is_sorted = sorted_hint.unwrap_or_else(|| is_sorted_slice(tgt_x));
        let n_signals = self.y.nrows();
        let mut out = Array2::<f64>::zeros((n_signals, n_tgt));

        // Release the GIL: the heavy numeric work below touches only Rust data.
        py.detach(|| {
            let x_slice = self.x.as_slice().unwrap();
            let method_str = self.method.as_str();
            let robust = self.robust;
            let extrap = self.extrap;

            if n_signals == 1 {
                // Single signal: optionally parallelize across target chunks.
                let y_view = self.y.row(0);
                let y_slice = y_view.as_slice().unwrap();
                let slopes_row0 = match &self.aux_data {
                    AuxData::Slopes(s) => Some(s.row(0)),
                    _ => None,
                };
                let d_opt = slopes_row0.as_ref().map(|r| r.as_slice().unwrap());
                let w_opt = match &self.aux_data {
                    AuxData::FHWeights(w) => Some(w.as_slice().unwrap()),
                    _ => None,
                };
                let out_flat = out.as_slice_mut().unwrap();

                if n_tgt >= PAR_TARGET_THRESHOLD {
                    let nthreads = rayon::current_num_threads().max(1);
                    let chunk = n_tgt.div_ceil(nthreads).max(MIN_PAR_CHUNK);
                    out_flat
                        .par_chunks_mut(chunk)
                        .zip(tgt_x.par_chunks(chunk))
                        .for_each(|(o, t)| {
                            run_kernel(
                                method_str, robust, t, x_slice, y_slice, d_opt, w_opt, o,
                                deriv, is_sorted, extrap,
                            );
                        });
                } else {
                    run_kernel(
                        method_str, robust, tgt_x, x_slice, y_slice, d_opt, w_opt, out_flat,
                        deriv, is_sorted, extrap,
                    );
                }
            } else {
                // Batch: parallelize across signals (rows).
                let slopes_ref = match &self.aux_data {
                    AuxData::Slopes(s) => Some(s),
                    _ => None,
                };
                let w_opt = match &self.aux_data {
                    AuxData::FHWeights(w) => Some(w.as_slice().unwrap()),
                    _ => None,
                };
                out.as_slice_mut()
                    .unwrap()
                    .par_chunks_exact_mut(n_tgt)
                    .enumerate()
                    .for_each(|(k, out_slice)| {
                        let y_view = self.y.row(k);
                        let y_slice = y_view.as_slice().unwrap();
                        let d_view = slopes_ref.map(|s| s.row(k));
                        let d_opt = d_view.as_ref().map(|r| r.as_slice().unwrap());
                        run_kernel(
                            method_str, robust, tgt_x, x_slice, y_slice, d_opt, w_opt, out_slice,
                            deriv, is_sorted, extrap,
                        );
                    });
            }
        });

        if self.is_batch {
            Ok(out.into_pyarray(py).into_any())
        } else {
            let out_1d = out
                .into_shape_with_order(n_tgt)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(out_1d.into_pyarray(py).into_any())
        }
    }

    fn eval<'py>(
        &self,
        py: Python<'py>,
        target_x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyAny>> {
        Ok(self.call(py, target_x, 0, None)?.unbind())
    }

    fn derivative<'py>(
        &self,
        py: Python<'py>,
        target_x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyAny>> {
        Ok(self.call(py, target_x, 1, None)?.unbind())
    }

    fn get_x<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        Ok(self.x.clone().into_pyarray(py).unbind())
    }

    #[pyo3(signature = (signal=None))]
    fn get_y<'py>(&self, py: Python<'py>, signal: Option<usize>) -> PyResult<Py<PyAny>> {
        let n_sig = self.y.nrows();
        if let Some(idx) = signal {
            if idx >= n_sig {
                return Err(PyValueError::new_err("signal index out of range"));
            }
            let row = self.y.row(idx).to_owned();
            Ok(row.into_pyarray(py).into_any().unbind())
        } else if self.is_batch {
            Ok(self.y.clone().into_pyarray(py).into_any().unbind())
        } else {
            let flat = self.y.row(0).to_owned();
            Ok(flat.into_pyarray(py).into_any().unbind())
        }
    }

    #[pyo3(signature = (signal=None))]
    fn get_slopes<'py>(
        &self,
        py: Python<'py>,
        signal: Option<usize>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if let AuxData::Slopes(ref slopes) = self.aux_data {
            let n_sig = slopes.nrows();
            if let Some(idx) = signal {
                if idx >= n_sig {
                    return Err(PyValueError::new_err("signal index out of range"));
                }
                let row = slopes.row(idx).to_owned();
                Ok(Some(row.into_pyarray(py).into_any().unbind()))
            } else if n_sig == 1 && !self.is_batch {
                let flat = slopes.row(0).to_owned();
                Ok(Some(flat.into_pyarray(py).into_any().unbind()))
            } else {
                Ok(Some(slopes.clone().into_pyarray(py).into_any().unbind()))
            }
        } else {
            Ok(None)
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyTuple>> {
        // Restore original dimensionality for serialization.
        let y_out = if self.is_batch {
            self.y.clone().into_pyarray(py).into_any()
        } else {
            self.y.row(0).to_owned().into_pyarray(py).into_any()
        };

        let args = PyTuple::new(
            py,
            vec![
                self.x.clone().into_pyarray(py).into_any(),
                y_out,
                self.method.as_str().into_pyobject(py)?.into_any(),
                self.robust.into_pyobject(py)?.to_owned().into_any(),
                self.d.into_pyobject(py)?.into_any(),
                self.extrap.as_str().into_pyobject(py)?.into_any(),
            ],
        )?;

        let cls = py.get_type::<UniInterpolator>();
        let tuple = PyTuple::new(py, vec![cls.into_any(), args.into_any()])?;
        Ok(tuple.unbind())
    }
}

// -------------------------------------------------------------------------
// Single-signal dispatcher (operates purely on slices, no Python state)
// -------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    method: &str,
    robust: bool,
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d_opt: Option<&[f64]>,
    w_opt: Option<&[f64]>,
    out: &mut [f64],
    deriv: usize,
    sorted: bool,
    extrap: ExtrapMode,
) {
    match method {
        "linear" => {
            if deriv == 0 && sorted {
                eval_linear_sorted(tgt_x, x, y, out, extrap);
            } else {
                eval_linear_general(tgt_x, x, y, out, deriv, extrap);
            }
        }
        "pchip" | "makima" => {
            let d = d_opt.expect("hermite slopes missing");
            if deriv == 0 && sorted {
                eval_hermite_sorted(tgt_x, x, y, d, out, extrap);
            } else {
                eval_hermite_general(tgt_x, x, y, d, out, deriv, extrap);
            }
        }
        "sprague" => {
            if deriv == 0 {
                if sorted {
                    eval_sprague_sorted(tgt_x, x, y, out, extrap);
                } else {
                    eval_sprague_general(tgt_x, x, y, out, robust, extrap);
                }
            } else {
                finite_diff(method, robust, tgt_x, x, y, d_opt, w_opt, out, sorted, extrap);
            }
        }
        "floater_hormann" | "fh" => {
            let w = w_opt.expect("Floater-Hormann weights missing");
            if deriv == 0 {
                eval_fh(tgt_x, x, y, w, out, extrap);
            } else {
                finite_diff(method, robust, tgt_x, x, y, d_opt, w_opt, out, sorted, extrap);
            }
        }
        _ => {}
    }
}

/// Central-difference fallback for methods without an analytic derivative.
#[allow(clippy::too_many_arguments)]
fn finite_diff(
    method: &str,
    robust: bool,
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d_opt: Option<&[f64]>,
    w_opt: Option<&[f64]>,
    out: &mut [f64],
    sorted: bool,
    extrap: ExtrapMode,
) {
    let n = tgt_x.len();
    let span = (x[x.len() - 1] - x[0]).abs().max(1.0);
    let h = 1e-6 * span;
    let inv = 1.0 / (2.0 * h);

    let tp: Vec<f64> = tgt_x.iter().map(|&v| v + h).collect();
    let tm: Vec<f64> = tgt_x.iter().map(|&v| v - h).collect();
    let mut yp = vec![0.0; n];
    let mut ym = vec![0.0; n];

    run_kernel(method, robust, &tp, x, y, d_opt, w_opt, &mut yp, 0, sorted, extrap);
    run_kernel(method, robust, &tm, x, y, d_opt, w_opt, &mut ym, 0, sorted, extrap);

    for i in 0..n {
        out[i] = (yp[i] - ym[i]) * inv;
    }
}

fn is_sorted_slice(data: &[f64]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

// =============================================================================
// KERNELS
// =============================================================================

#[inline]
fn extrap_value(
    xi: f64,
    x: &[f64],
    y: &[f64],
    n: usize,
    left: bool,
    extrap: ExtrapMode,
) -> f64 {
    match extrap {
        ExtrapMode::Linear => {
            if left {
                let dx = x[1] - x[0];
                let dy = y[1] - y[0];
                if dx != 0.0 { y[0] + dy * (xi - x[0]) / dx } else { y[0] }
            } else {
                let dx = x[n - 1] - x[n - 2];
                let dy = y[n - 1] - y[n - 2];
                if dx != 0.0 { y[n - 1] + dy * (xi - x[n - 1]) / dx } else { y[n - 1] }
            }
        }
        ExtrapMode::Clamp => if left { y[0] } else { y[n - 1] },
        ExtrapMode::Error => f64::NAN,
    }
}

fn eval_linear_sorted(tgt_x: &[f64], x: &[f64], y: &[f64], out: &mut [f64], extrap: ExtrapMode) {
    let n = x.len();
    let mut j = 0;
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi < x[0] {
            out[i] = extrap_value(xi, x, y, n, true, extrap);
            continue;
        }
        if xi > x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, false, extrap);
            continue;
        }
        while j < n - 1 && xi > x[j + 1] {
            j += 1;
        }
        let dx = x[j + 1] - x[j];
        let t = if dx != 0.0 { (xi - x[j]) / dx } else { 0.0 };
        out[i] = y[j] * (1.0 - t) + y[j + 1] * t;
    }
}

fn eval_linear_general(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    out: &mut [f64],
    deriv: usize,
    extrap: ExtrapMode,
) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = if deriv == 0 {
                extrap_value(xi, x, y, n, true, extrap)
            } else {
                let dx = x[1] - x[0];
                if dx != 0.0 { (y[1] - y[0]) / dx } else { 0.0 }
            };
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = if deriv == 0 {
                extrap_value(xi, x, y, n, false, extrap)
            } else {
                let dx = x[n - 1] - x[n - 2];
                if dx != 0.0 { (y[n - 1] - y[n - 2]) / dx } else { 0.0 }
            };
            continue;
        }
        let lo = lower_bound(x, xi);
        let dx = x[lo + 1] - x[lo];
        out[i] = if deriv == 0 {
            let t = if dx != 0.0 { (xi - x[lo]) / dx } else { 0.0 };
            y[lo] * (1.0 - t) + y[lo + 1] * t
        } else if dx != 0.0 {
            (y[lo + 1] - y[lo]) / dx
        } else {
            0.0
        };
    }
}

fn eval_hermite_sorted(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d: &[f64],
    out: &mut [f64],
    extrap: ExtrapMode,
) {
    let n = x.len();
    let mut j = 0;
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi < x[0] {
            out[i] = match extrap {
                ExtrapMode::Linear => y[0] + d[0] * (xi - x[0]),
                ExtrapMode::Clamp => y[0],
                ExtrapMode::Error => f64::NAN,
            };
            continue;
        }
        if xi > x[n - 1] {
            out[i] = match extrap {
                ExtrapMode::Linear => y[n - 1] + d[n - 1] * (xi - x[n - 1]),
                ExtrapMode::Clamp => y[n - 1],
                ExtrapMode::Error => f64::NAN,
            };
            continue;
        }
        while j < n - 1 && xi > x[j + 1] {
            j += 1;
        }
        let h = x[j + 1] - x[j];
        if h == 0.0 {
            out[i] = y[j];
            continue;
        }
        let t = (xi - x[j]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        out[i] = h00 * y[j] + h10 * h * d[j] + h01 * y[j + 1] + h11 * h * d[j + 1];
    }
}

fn eval_hermite_general(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d: &[f64],
    out: &mut [f64],
    deriv: usize,
    extrap: ExtrapMode,
) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = if deriv == 0 {
                match extrap {
                    ExtrapMode::Linear => y[0] + d[0] * (xi - x[0]),
                    ExtrapMode::Clamp => y[0],
                    ExtrapMode::Error => f64::NAN,
                }
            } else {
                d[0]
            };
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = if deriv == 0 {
                match extrap {
                    ExtrapMode::Linear => y[n - 1] + d[n - 1] * (xi - x[n - 1]),
                    ExtrapMode::Clamp => y[n - 1],
                    ExtrapMode::Error => f64::NAN,
                }
            } else {
                d[n - 1]
            };
            continue;
        }
        let lo = lower_bound(x, xi);
        let h = x[lo + 1] - x[lo];
        if h == 0.0 {
            out[i] = if deriv == 0 { y[lo] } else { 0.0 };
            continue;
        }
        let t = (xi - x[lo]) / h;
        if deriv == 0 {
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            out[i] = h00 * y[lo] + h10 * h * d[lo] + h01 * y[lo + 1] + h11 * h * d[lo + 1];
        } else {
            let dt = 6.0 * t * t - 6.0 * t;
            let d_h00 = dt;
            let d_h10 = 3.0 * t * t - 4.0 * t + 1.0;
            let d_h01 = -dt;
            let d_h11 = 3.0 * t * t - 2.0 * t;
            out[i] =
                (d_h00 * y[lo] + d_h10 * h * d[lo] + d_h01 * y[lo + 1] + d_h11 * h * d[lo + 1]) / h;
        }
    }
}

/// Sprague (6-point local) interpolation for *sorted* queries.
/// Uses a marching window and caches the barycentric weights, recomputing them
/// only when the active 6-point window changes (huge win for dense queries).
fn eval_sprague_sorted(tgt_x: &[f64], x: &[f64], y: &[f64], out: &mut [f64], extrap: ExtrapMode) {
    let n = x.len();
    let mut node_ptr = 0usize; // count of source nodes <= xi (searchsorted 'right')
    let mut cur_start = usize::MAX;
    let mut w = [0.0f64; 6];
    let mut xloc = [0.0f64; 6];
    let mut yloc = [0.0f64; 6];

    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = extrap_value(xi, x, y, n, true, extrap);
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, false, extrap);
            continue;
        }

        while node_ptr < n && x[node_ptr] <= xi {
            node_ptr += 1;
        }
        let idx = node_ptr.max(1);
        let mut w_start = idx.saturating_sub(3);
        let max_start = n - 6;
        if w_start > max_start {
            w_start = max_start;
        }

        if w_start != cur_start {
            cur_start = w_start;
            xloc.copy_from_slice(&x[w_start..w_start + 6]);
            yloc.copy_from_slice(&y[w_start..w_start + 6]);
            for j in 0..6 {
                let mut wj = 1.0;
                for k in 0..6 {
                    if k != j {
                        wj /= xloc[j] - xloc[k];
                    }
                }
                w[j] = wj;
            }
        }

        let mut num = 0.0;
        let mut den = 0.0;
        let mut hit = false;
        for j in 0..6 {
            let diff = xi - xloc[j];
            if diff == 0.0 {
                out[i] = yloc[j];
                hit = true;
                break;
            }
            let term = w[j] / diff;
            num += term * yloc[j];
            den += term;
        }
        if !hit {
            out[i] = if den != 0.0 { num / den } else { 0.0 };
        }
    }
}

/// Sprague interpolation for arbitrary (possibly unsorted) queries.
/// Honors `robust` (barycentric vs naive Lagrange) so the two formulations
/// can be cross-validated against one another.
fn eval_sprague_general(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    out: &mut [f64],
    robust: bool,
    extrap: ExtrapMode,
) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = extrap_value(xi, x, y, n, true, extrap);
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, false, extrap);
            continue;
        }

        let idx = x.partition_point(|&v| v <= xi).max(1);
        let mut w_start = idx.saturating_sub(3);
        let max_start = n - 6;
        if w_start > max_start {
            w_start = max_start;
        }

        let x_loc = &x[w_start..w_start + 6];
        let y_loc = &y[w_start..w_start + 6];

        if robust {
            // Barycentric Lagrange formulation.
            let mut w = [1.0f64; 6];
            for j in 0..6 {
                for k in 0..6 {
                    if k != j {
                        w[j] /= x_loc[j] - x_loc[k];
                    }
                }
            }
            let mut num = 0.0;
            let mut den = 0.0;
            let mut hit = false;
            for j in 0..6 {
                let diff = xi - x_loc[j];
                if diff == 0.0 {
                    out[i] = y_loc[j];
                    hit = true;
                    break;
                }
                let term = w[j] / diff;
                num += term * y_loc[j];
                den += term;
            }
            if !hit {
                out[i] = if den != 0.0 { num / den } else { 0.0 };
            }
        } else {
            // Naive Lagrange expansion.
            let mut res = 0.0;
            for j in 0..6 {
                let mut basis = 1.0;
                let xj = x_loc[j];
                for k in 0..6 {
                    if k != j {
                        basis *= (xi - x_loc[k]) / (xj - x_loc[k]);
                    }
                }
                res += y_loc[j] * basis;
            }
            out[i] = res;
        }
    }
}

fn eval_fh(tgt_x: &[f64], x: &[f64], y: &[f64], w: &[f64], out: &mut [f64], extrap: ExtrapMode) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi < x[0] || xi > x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, xi < x[0], extrap);
            continue;
        }
        let mut num = 0.0;
        let mut den = 0.0;
        let mut hit = false;
        for k in 0..n {
            let diff = xi - x[k];
            if diff == 0.0 {
                out[i] = y[k];
                hit = true;
                break;
            }
            let term = w[k] / diff;
            num += term * y[k];
            den += term;
        }
        if !hit {
            out[i] = if den != 0.0 { num / den } else { 0.0 };
        }
    }
}

// -------------------------------------------------------------------------
// Small helpers
// -------------------------------------------------------------------------

/// Returns `lo` such that `x[lo] <= xi < x[lo+1]`, assuming `x[0] < xi < x[n-1]`.
#[inline]
fn lower_bound(x: &[f64], xi: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = x.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xi < x[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

// -------------------------------------------------------------------------
// Precomputation kernels
// -------------------------------------------------------------------------
fn calc_pchip_slopes(x: &Array1<f64>, y_batch: &Array2<f64>) -> Array2<f64> {
    let n = x.len();
    let mut slopes = Array2::<f64>::zeros(y_batch.raw_dim());
    let x_slice = x.as_slice().unwrap();
    let y_batch_slice = y_batch.as_slice().unwrap();

    if n == 2 {
        let dx = x_slice[1] - x_slice[0];
        if dx != 0.0 {
            slopes
                .as_slice_mut()
                .unwrap()
                .par_chunks_exact_mut(n)
                .zip(y_batch_slice.par_chunks_exact(n))
                .for_each(|(d, y)| {
                    let s0 = (y[1] - y[0]) / dx;
                    d[0] = s0;
                    d[1] = s0;
                });
        }
        return slopes;
    }

    slopes
        .as_slice_mut()
        .unwrap()
        .par_chunks_exact_mut(n)
        .zip(y_batch_slice.par_chunks_exact(n))
        .for_each(|(d, y)| {
            let mut h = vec![0.0; n - 1];
            let mut delta = vec![0.0; n - 1];
            for i in 0..n - 1 {
                h[i] = x_slice[i + 1] - x_slice[i];
                delta[i] = if h[i] != 0.0 {
                    (y[i + 1] - y[i]) / h[i]
                } else {
                    0.0
                };
            }
            for k in 1..n - 1 {
                if delta[k - 1] * delta[k] > 0.0 {
                    let w1 = 2.0 * h[k] + h[k - 1];
                    let w2 = h[k] + 2.0 * h[k - 1];
                    d[k] = (w1 + w2) * delta[k - 1] * delta[k]
                        / (w1 * delta[k] + w2 * delta[k - 1]);
                }
            }
            let end_deriv = |h0: f64, h1: f64, del0: f64, del1: f64| -> f64 {
                let d_val = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
                if d_val.signum() != del0.signum() {
                    return 0.0;
                }
                if (del0.signum() != del1.signum()) && (d_val.abs() > 3.0 * del0.abs()) {
                    return 3.0 * del0;
                }
                d_val
            };
            d[0] = end_deriv(h[0], h[1], delta[0], delta[1]);
            d[n - 1] = end_deriv(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        });
    slopes
}

fn calc_makima_slopes(x: &Array1<f64>, y_batch: &Array2<f64>) -> Array2<f64> {
    let n = x.len();
    let mut slopes = Array2::<f64>::zeros(y_batch.raw_dim());
    let x_slice = x.as_slice().unwrap();
    let y_batch_slice = y_batch.as_slice().unwrap();

    if n == 2 {
        let dx = x_slice[1] - x_slice[0];
        if dx != 0.0 {
            slopes
                .as_slice_mut()
                .unwrap()
                .par_chunks_exact_mut(n)
                .zip(y_batch_slice.par_chunks_exact(n))
                .for_each(|(d, y)| {
                    let s0 = (y[1] - y[0]) / dx;
                    d[0] = s0;
                    d[1] = s0;
                });
        }
        return slopes;
    }

    slopes
        .as_slice_mut()
        .unwrap()
        .par_chunks_exact_mut(n)
        .zip(y_batch_slice.par_chunks_exact(n))
        .for_each(|(s, y)| {
            let mut deltas = vec![0.0; n - 1];
            for i in 0..n - 1 {
                let dx = x_slice[i + 1] - x_slice[i];
                deltas[i] = if dx != 0.0 {
                    (y[i + 1] - y[i]) / dx
                } else {
                    0.0
                };
            }
            let mut d = vec![0.0; n + 3];
            d[2..n + 1].copy_from_slice(&deltas);
            d[1] = 2.0 * deltas[0] - deltas[1];
            d[0] = 2.0 * d[1] - deltas[0];
            d[n + 1] = 2.0 * deltas[n - 2] - deltas[n - 3];
            d[n + 2] = 2.0 * d[n + 1] - deltas[n - 2];

            for i in 0..n {
                let w1 = f64::abs(d[i + 3] - d[i + 2]) + f64::abs(d[i + 3] + d[i + 2]) * 0.5;
                let w2 = f64::abs(d[i + 1] - d[i]) + f64::abs(d[i + 1] + d[i]) * 0.5;
                let w_sum = w1 + w2;
                s[i] = if w_sum == 0.0 {
                    0.5 * (d[i + 1] + d[i + 2])
                } else {
                    (w1 * d[i + 1] + w2 * d[i + 2]) / w_sum
                };
            }
        });
    slopes
}

fn calc_fh_weights(x: &Array1<f64>, d: usize) -> Array1<f64> {
    let n = x.len();
    let mut w = Array1::<f64>::zeros(n);
    let x_slice = x.as_slice().unwrap();
    for k in 0..n {
        let mut s_val = 0.0;
        let i_min = k.saturating_sub(d);
        let i_max = k.min(n.saturating_sub(d + 1));
        for i in i_min..=i_max {
            let mut prod = 1.0;
            for j in i..=(i + d) {
                if j != k {
                    prod *= 1.0 / (x_slice[k] - x_slice[j]).abs();
                }
            }
            s_val += prod;
        }
        w[k] = if k % 2 == 1 { -s_val } else { s_val };
    }
    w
}

// -------------------------------------------------------------------------
// Module initialisation
// -------------------------------------------------------------------------
#[pymodule]
fn interpolate<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<UniInterpolator>()?;
    Ok(())
}