//! Thin PyO3 bindings for the Navette interpolation core.
//!
//! No math here: everything lives in the pure-Rust `navette-interpolate`
//! core. Each method owns its NumPy inputs, releases the GIL while the
//! (possibly rayon-parallel) kernel runs, and returns NumPy arrays.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use navette_interpolate::UniInterpolator as Core;

fn map_err(s: String) -> PyErr {
    PyValueError::new_err(s)
}

#[pyclass(name = "UniInterpolator")]
pub struct UniInterpolator {
    inner: Core,
}

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
        let x_arr = x.as_array().to_owned();
        let n = x_arr.len();
        // y: 1-D (single) or 2-D (batch)
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
        let _ = &mut d;
        let inner = Core::new(x_arr, y_arr, is_batch, method, robust, d, extrap).map_err(map_err)?;
        Ok(Self { inner })
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
        let tgt = target_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("target_x must be contiguous"))?;
        let out = py.detach(|| self.inner.evaluate(tgt, deriv, sorted_hint));
        if self.inner.is_batch() {
            Ok(out.into_pyarray(py).into_any())
        } else {
            let n_tgt = tgt.len();
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

    fn get_x<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        self.inner.x_clone().into_pyarray(py).unbind()
    }

    #[pyo3(signature = (signal=None))]
    fn get_y<'py>(&self, py: Python<'py>, signal: Option<usize>) -> PyResult<Py<PyAny>> {
        let y = self.inner.y_clone();
        let n_sig = y.nrows();
        if let Some(idx) = signal {
            if idx >= n_sig {
                return Err(PyValueError::new_err("signal index out of range"));
            }
            let row = y.row(idx).to_owned();
            Ok(row.into_pyarray(py).into_any().unbind())
        } else if self.inner.is_batch() {
            Ok(y.into_pyarray(py).into_any().unbind())
        } else {
            let flat = y.row(0).to_owned();
            Ok(flat.into_pyarray(py).into_any().unbind())
        }
    }

    #[pyo3(signature = (signal=None))]
    fn get_slopes<'py>(
        &self,
        py: Python<'py>,
        signal: Option<usize>,
    ) -> PyResult<Option<Py<PyAny>>> {
        match self.inner.slopes_clone() {
            Some(slopes) => {
                let n_sig = slopes.nrows();
                if let Some(idx) = signal {
                    if idx >= n_sig {
                        return Err(PyValueError::new_err("signal index out of range"));
                    }
                    let row = slopes.row(idx).to_owned();
                    Ok(Some(row.into_pyarray(py).into_any().unbind()))
                } else if n_sig == 1 && !self.inner.is_batch() {
                    let flat = slopes.row(0).to_owned();
                    Ok(Some(flat.into_pyarray(py).into_any().unbind()))
                } else {
                    Ok(Some(slopes.into_pyarray(py).into_any().unbind()))
                }
            }
            None => Ok(None),
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyTuple>> {
        let y_out = if self.inner.is_batch() {
            self.inner.y_clone().into_pyarray(py).into_any()
        } else {
            self.inner
                .y_clone()
                .row(0)
                .to_owned()
                .into_pyarray(py)
                .into_any()
        };
        let args = PyTuple::new(
            py,
            vec![
                self.inner.x_clone().into_pyarray(py).into_any(),
                y_out,
                self.inner.method().into_pyobject(py)?.into_any(),
                self.inner.robust().into_pyobject(py)?.to_owned().into_any(),
                self.inner.fh_d().into_pyobject(py)?.into_any(),
                self.inner.extrap_str().into_pyobject(py)?.into_any(),
            ],
        )?;
        let cls = py.get_type::<UniInterpolator>();
        let tuple = PyTuple::new(py, vec![cls.into_any(), args.into_any()])?;
        Ok(tuple.unbind())
    }
}

#[pymodule]
fn _interpolate<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<UniInterpolator>()?;
    Ok(())
}
