//! PyO3 bindings for the Navette materials core.
//!
//! This crate contains NO math. Each `#[pyfunction]`:
//!   1. takes NumPy arrays (wavelength in nm) + scalar params,
//!   2. owns the input data so the compute closure is `Send`,
//!   3. releases the GIL via `py.allow_threads` while the (possibly rayon-
//!      parallel) kernel runs,
//!   4. returns a NumPy complex128 array.
//!
//! Compiled module name: `navette.materials._native`.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette_materials as core;

/// Helper: own a 1-D f64 view as a `Send` ndarray for `allow_threads`.
fn owned1(a: PyReadonlyArray1<f64>) -> ndarray::Array1<f64> {
    a.as_array().to_owned()
}

/// Helper: own a 1-D complex view as a `Send` ndarray.
fn ownedc1(a: PyReadonlyArray1<Complex64>) -> Array1<Complex64> {
    a.as_array().to_owned()
}

/// Helper: own a 2-D f64 view (oscillator array) as a `Send` ndarray.
fn owned2(a: PyReadonlyArray2<f64>) -> Array2<f64> {
    a.as_array().to_owned()
}

#[pyfunction]
fn cauchy_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    a: f64,
    b: f64,
    c: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let out = py.allow_threads(move || core::cauchy::cauchy_nk(wl.view(), a, b, c));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cauchy_urbach_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    a: f64,
    b: f64,
    c: f64,
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let out = py.allow_threads(move || core::cauchy::cauchy_urbach_nk(wl.view(), a, b, c, alpha0, eu, lambda_g));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sellmeier_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    b3: f64,
    c3: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let out = py.allow_threads(move || core::sellmeier::sellmeier_nk(wl.view(), b1, c1, b2, c2, b3, c3));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sellmeier_urbach_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    b3: f64,
    c3: f64,
    alpha0: f64,
    eu: f64,
    lambda_g: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let out = py.allow_threads(move || {
        core::sellmeier::sellmeier_urbach_nk(wl.view(), b1, c1, b2, c2, b3, c3, alpha0, eu, lambda_g)
    });
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn lorentz_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    osc: PyReadonlyArray2<'py, f64>,
    eps_inf: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let o = owned2(osc);
    let out = py.allow_threads(move || core::lorentz::lorentz_nk(wl.view(), o.view(), eps_inf));
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn drude_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    omega_p: f64,
    gamma: f64,
    eps_inf: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let out = py.allow_threads(move || core::drude::drude_nk(wl.view(), omega_p, gamma, eps_inf));
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn drude_lorentz_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    omega_p: f64,
    gamma_d: f64,
    eps_inf: f64,
    osc: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let o = owned2(osc);
    let out = py.allow_threads(move || core::drude::drude_lorentz_nk(wl.view(), omega_p, gamma_d, eps_inf, o.view()));
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn cody_lorentz_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    eg: f64,
    et: f64,
    eu: f64,
    osc: PyReadonlyArray2<'py, f64>,
    eps_inf: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let wl = owned1(wavelength_nm);
    let o = owned2(osc);
    let res = py.allow_threads(move || core::cody_lorentz::cody_lorentz_nk(wl.view(), eg, et, eu, o.view(), eps_inf));
    match res {
        Ok(out) => Ok(out.into_pyarray_bound(py)),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
fn fb_interband_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    n_inf: f64,
    ib: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let t = owned2(ib);
    let out = py.allow_threads(move || core::forouhi_bloomer::fb_interband_nk(wl.view(), n_inf, t.view()));
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn fb_metal_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    n_inf: f64,
    fe: PyReadonlyArray1<'py, f64>,
    ib: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let fe_o = owned1(fe);
    let t = owned2(ib);
    let out = py.allow_threads(move || core::forouhi_bloomer::fb_metal_nk(wl.view(), n_inf, fe_o.view(), t.view()));
    out.into_pyarray_bound(py)
}

// --- EMA mixers (take inclusion/host refractive indices, return permittivity) ---

macro_rules! ema_simple {
    ($name:ident, $core:path) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            n_i: PyReadonlyArray1<'py, Complex64>,
            n_h: PyReadonlyArray1<'py, Complex64>,
            f: f64,
        ) -> Bound<'py, PyArray1<Complex64>> {
            let ni = ownedc1(n_i);
            let nh = ownedc1(n_h);
            let out = py.allow_threads(move || $core(ni.view(), nh.view(), f));
            out.into_pyarray_bound(py)
        }
    };
}

ema_simple!(ema_lichtenecker, core::ema::lichtenecker);
ema_simple!(ema_looyenga, core::ema::looyenga);
ema_simple!(ema_maxwell_garnett, core::ema::maxwell_garnett);

#[pyfunction]
#[pyo3(signature = (n_i, n_h, f, max_iter=100, tol=1e-9))]
fn ema_bruggeman<'py>(
    py: Python<'py>,
    n_i: PyReadonlyArray1<'py, Complex64>,
    n_h: PyReadonlyArray1<'py, Complex64>,
    f: f64,
    max_iter: usize,
    tol: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let ni = ownedc1(n_i);
    let nh = ownedc1(n_h);
    let out = py.allow_threads(move || core::ema::bruggeman(ni.view(), nh.view(), f, max_iter, tol));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (n_i, n_h, f, l=0.3333333333333333))]
fn ema_mori_tanaka<'py>(
    py: Python<'py>,
    n_i: PyReadonlyArray1<'py, Complex64>,
    n_h: PyReadonlyArray1<'py, Complex64>,
    f: f64,
    l: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let ni = ownedc1(n_i);
    let nh = ownedc1(n_h);
    let out = py.allow_threads(move || core::ema::mori_tanaka(ni.view(), nh.view(), f, l));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (n_i, n_h, f, alpha=0.5))]
fn ema_power_law<'py>(
    py: Python<'py>,
    n_i: PyReadonlyArray1<'py, Complex64>,
    n_h: PyReadonlyArray1<'py, Complex64>,
    f: f64,
    alpha: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let ni = ownedc1(n_i);
    let nh = ownedc1(n_h);
    let out = py.allow_threads(move || core::ema::general_power_law(ni.view(), nh.view(), f, alpha));
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn ema_roughness<'py>(
    py: Python<'py>,
    n_bottom: PyReadonlyArray1<'py, Complex64>,
    n_top: PyReadonlyArray1<'py, Complex64>,
) -> Bound<'py, PyArray1<Complex64>> {
    let nb = ownedc1(n_bottom);
    let nt = ownedc1(n_top);
    let out = py.allow_threads(move || core::ema::roughness_interface(nb.view(), nt.view()));
    out.into_pyarray_bound(py)
}

/// √ε for an array of permittivities (the EMA composition's final step).
#[pyfunction]
fn eps_to_nk<'py>(py: Python<'py>, eps: PyReadonlyArray1<'py, Complex64>) -> Bound<'py, PyArray1<Complex64>> {
    let e = ownedc1(eps);
    let out = py.allow_threads(move || core::ema::eps_to_nk(e.view()));
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn tauc_lorentz_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    eg: f64,
    osc: PyReadonlyArray2<'py, f64>,
    eps_inf: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let wl = owned1(wavelength_nm);
    let o = owned2(osc);
    let res = py.allow_threads(move || core::tauc_lorentz::tauc_lorentz_nk(wl.view(), eg, o.view(), eps_inf));
    match res {
        Ok(out) => Ok(out.into_pyarray_bound(py)),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
fn ubf_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    osc: PyReadonlyArray2<'py, f64>,
    eps_inf: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let wl = owned1(wavelength_nm);
    let o = owned2(osc);
    let res = py.allow_threads(move || core::ubf::ubf_nk(wl.view(), o.view(), eps_inf));
    match res {
        Ok(out) => Ok(out.into_pyarray_bound(py)),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
fn konstant_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    n: f64,
    k: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let out = py.allow_threads(move || core::table::konstant_nk(wl.view(), n, k));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(signature = (wavelength_nm, grid_wl, n_vals, k_vals=None, n_factor=1.0, k_factor=1.0))]
#[allow(clippy::too_many_arguments)]
fn table_nk<'py>(
    py: Python<'py>,
    wavelength_nm: PyReadonlyArray1<'py, f64>,
    grid_wl: PyReadonlyArray1<'py, f64>,
    n_vals: PyReadonlyArray1<'py, f64>,
    k_vals: Option<PyReadonlyArray1<'py, f64>>,
    n_factor: f64,
    k_factor: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    let wl = owned1(wavelength_nm);
    let g = owned1(grid_wl);
    let n = owned1(n_vals);
    let k = k_vals.map(owned1);
    let out = py.allow_threads(move || {
        core::table::table_nk(
            wl.view(),
            g.view(),
            n.view(),
            k.as_ref().map(|k| k.view()),
            n_factor,
            k_factor,
        )
    });
    out.into_pyarray_bound(py)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cauchy_nk, m)?)?;
    m.add_function(wrap_pyfunction!(cauchy_urbach_nk, m)?)?;
    m.add_function(wrap_pyfunction!(sellmeier_nk, m)?)?;
    m.add_function(wrap_pyfunction!(sellmeier_urbach_nk, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_nk, m)?)?;
    m.add_function(wrap_pyfunction!(drude_nk, m)?)?;
    m.add_function(wrap_pyfunction!(drude_lorentz_nk, m)?)?;
    m.add_function(wrap_pyfunction!(cody_lorentz_nk, m)?)?;
    m.add_function(wrap_pyfunction!(fb_interband_nk, m)?)?;
    m.add_function(wrap_pyfunction!(fb_metal_nk, m)?)?;
    m.add_function(wrap_pyfunction!(ema_lichtenecker, m)?)?;
    m.add_function(wrap_pyfunction!(ema_looyenga, m)?)?;
    m.add_function(wrap_pyfunction!(ema_maxwell_garnett, m)?)?;
    m.add_function(wrap_pyfunction!(ema_bruggeman, m)?)?;
    m.add_function(wrap_pyfunction!(ema_mori_tanaka, m)?)?;
    m.add_function(wrap_pyfunction!(ema_power_law, m)?)?;
    m.add_function(wrap_pyfunction!(ema_roughness, m)?)?;
    m.add_function(wrap_pyfunction!(eps_to_nk, m)?)?;
    m.add_function(wrap_pyfunction!(ubf_nk, m)?)?;
    m.add_function(wrap_pyfunction!(tauc_lorentz_nk, m)?)?;
    m.add_function(wrap_pyfunction!(konstant_nk, m)?)?;
    m.add_function(wrap_pyfunction!(table_nk, m)?)?;
    Ok(())
}
