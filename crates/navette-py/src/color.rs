//! Thin PyO3 bindings for the Navette color engine.
//!
//! No math here: every kernel lives in the pure-Rust `navette-color` core.
//! Each `#[pyfunction]` owns its NumPy inputs, releases the GIL while the
//! (possibly rayon-parallel) kernel runs, and returns a NumPy array.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette::color::common::{REF_WHITE_D50, REF_WHITE_D65};

// ---- Zero-Copy Memory Helpers ------------------------------------------

/// Zero-copy view of an (N, 3) C-contiguous numpy array as `&[[f64; 3]]`.
fn as_slice3<'a>(a: &'a PyReadonlyArray2<'_, f64>) -> PyResult<&'a [[f64; 3]]> {
    let view = a.as_array();
    if view.ncols() != 3 {
        return Err(PyValueError::new_err("Expected an (N, 3) float64 array"));
    }
    match view.as_slice() {
        Some(slice) => {
            // Safe: [f64; 3] has exact memory layout of 3 adjacent f64s.
            let ptr = slice.as_ptr() as *const [f64; 3];
            Ok(unsafe { std::slice::from_raw_parts(ptr, view.nrows()) })
        }
        None => Err(PyValueError::new_err("Input array must be C-contiguous. (Call np.ascontiguousarray() first)")),
    }
}

/// Zero-copy view of an (N, 2) C-contiguous numpy array as `&[[f64; 2]]`.
fn as_slice2<'a>(a: &'a PyReadonlyArray2<'_, f64>) -> PyResult<&'a [[f64; 2]]> {
    let view = a.as_array();
    if view.ncols() != 2 {
        return Err(PyValueError::new_err("Expected an (N, 2) float64 array"));
    }
    match view.as_slice() {
        Some(slice) => {
            let ptr = slice.as_ptr() as *const [f64; 2];
            Ok(unsafe { std::slice::from_raw_parts(ptr, view.nrows()) })
        }
        None => Err(PyValueError::new_err("Input array must be C-contiguous.")),
    }
}

/// Zero-copy view of a 1D C-contiguous numpy array.
fn as_slice1<'a>(a: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    let view = a.as_array();
    match view.as_slice() {
        Some(slice) => {
            let ptr = slice.as_ptr();
            Ok(unsafe { std::slice::from_raw_parts(ptr, view.len()) })
        }
        None => Err(PyValueError::new_err("Input array must be C-contiguous.")),
    }
}

/// Allocate an (N, 3) PyArray directly in Python, returning the bound array and a mutable Rust slice.
fn new_out3<'py>(py: Python<'py>, n: usize) -> (Bound<'py, PyArray2<f64>>, &'py mut [[f64; 3]]) {
    // Zeros allocates directly into Python memory space
    let arr = numpy::PyArray2::<f64>::zeros(py, [n, 3], false);
    // Safe: freshly allocated NumPy arrays are inherently C-contiguous
    let slice = unsafe { arr.as_slice_mut().unwrap() };
    let ptr = slice.as_mut_ptr() as *mut [f64; 3];
    let out_slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
    (arr, out_slice)
}

/// Allocate an (N, 2) PyArray directly in Python.
fn new_out2<'py>(py: Python<'py>, n: usize) -> (Bound<'py, PyArray2<f64>>, &'py mut [[f64; 2]]) {
    let arr = numpy::PyArray2::<f64>::zeros(py, [n, 2], false);
    let slice = unsafe { arr.as_slice_mut().unwrap() };
    let ptr = slice.as_mut_ptr() as *mut [f64; 2];
    let out_slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
    (arr, out_slice)
}

// ---- Core sRGB / Lab / XYZ conversions ---------------------------------

#[pyfunction(name = "sRGB_to_XYZ")]
#[pyo3(signature = (rgb, clip=true))]
fn srgb_to_xyz<'py>(py: Python<'py>, rgb: PyReadonlyArray2<'py, f64>, clip: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&rgb)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::common::srgb_to_xyz(inp, clip, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "XYZ_to_sRGB")]
#[pyo3(signature = (xyz, clip=true))]
fn xyz_to_srgb<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, clip: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::common::xyz_to_srgb(inp, clip, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "XYZ_to_Lab")]
#[pyo3(signature = (xyz, illuminant=None))]
fn xyz_to_lab<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, illuminant: Option<[f64; 3]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let illum = illuminant.unwrap_or(REF_WHITE_D65);
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::common::xyz_to_lab(inp, &illum, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "Lab_to_XYZ")]
#[pyo3(signature = (lab, illuminant=None))]
fn lab_to_xyz<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>, illuminant: Option<[f64; 3]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?;
    let illum = illuminant.unwrap_or(REF_WHITE_D65);
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::common::lab_to_xyz(inp, &illum, out_slice);
    Ok(out_arr)
}

// ---- Convenience Composites & Gamut ------------------------------------

#[pyfunction(name = "sRGB_to_Lab")]
fn srgb_to_lab<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_lab(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction(name = "Lab_to_sRGB")]
fn lab_to_srgb<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::lab_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction(name = "sRGB_to_LCHab")]
fn srgb_to_lch<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_lch(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction(name = "LCHab_to_sRGB")]
fn lch_to_srgb<'py>(py: Python<'py>, lch: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lch)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::lch_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction(name = "sRGB_to_Luv")]
fn srgb_to_luv<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_luv(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction(name = "Luv_to_sRGB")]
fn luv_to_srgb<'py>(py: Python<'py>, luv: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&luv)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::luv_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction(name = "sRGB_to_xyY")]
fn srgb_to_xy_y<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_xyy_bound(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction(name = "xyY_to_sRGB")]
fn xy_y_to_srgb<'py>(py: Python<'py>, xyy: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyy)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::xyy_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction]
fn clip_absolute<'py>(py: Python<'py>, rgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&rgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::clip_absolute(inp, out_slice); 
    Ok(out_arr)
}

#[pyfunction]
#[pyo3(signature = (enabled=true))]
#[allow(unused_variables)]
fn set_strict_ieee(enabled: bool) {
    // Rust lacks Numba's fastmath-reassociation toggle, so IEEE is inherently strict here.
    // Provided solely for drop-in API parity with Python.
}

// ---- func_01: XYZ <-> xyY ----------------------------------------------

#[pyfunction(name = "XYZ_to_xyY")]
fn xyz_to_xyy<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_01::xyz_to_xyy(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "xyY_to_XYZ")]
fn xyy_to_xyz<'py>(py: Python<'py>, xyy: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyy)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_01::xyy_to_xyz(inp, out_slice);
    Ok(out_arr)
}

// ---- func_02: Lab <-> LCh ----------------------------------------------

#[pyfunction(name = "Lab_to_LCHab")]
fn lab_to_lch<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_02::lab_to_lch(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "LCHab_to_Lab")]
fn lch_to_lab<'py>(py: Python<'py>, lch: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lch)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_02::lch_to_lab(inp, out_slice);
    Ok(out_arr)
}

// ---- func_03: XYZ <-> CIELUV (illuminant defaults to D65) ---------------

#[pyfunction(name = "XYZ_to_Luv")]
#[pyo3(signature = (xyz, illuminant=None))]
fn xyz_to_luv<'py>(
    py: Python<'py>,
    xyz: PyReadonlyArray2<'py, f64>,
    illuminant: Option<[f64; 3]>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let illum = illuminant.unwrap_or(REF_WHITE_D65);
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_03::xyz_to_luv(inp, &illum, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "Luv_to_XYZ")]
#[pyo3(signature = (luv, illuminant=None))]
fn luv_to_xyz<'py>(
    py: Python<'py>,
    luv: PyReadonlyArray2<'py, f64>,
    illuminant: Option<[f64; 3]>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&luv)?;
    let illum = illuminant.unwrap_or(REF_WHITE_D65);
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_03::luv_to_xyz(inp, &illum, out_slice);
    Ok(out_arr)
}

// ---- func_04: XYZ <-> Oklab --------------------------------------------

#[pyfunction(name = "XYZ_to_Oklab")]
fn xyz_to_oklab<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_04::xyz_to_oklab(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "Oklab_to_XYZ")]
fn oklab_to_xyz<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_04::oklab_to_xyz(inp, out_slice);
    Ok(out_arr)
}

// ---- func_05: sRGB <-> Oklab (legacy) ----------------------------------

#[pyfunction(name = "sRGB_to_Oklab")]
fn srgb_to_oklab<'py>(py: Python<'py>, rgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&rgb)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_05::srgb_to_oklab(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "Oklab_to_sRGB")]
fn oklab_to_srgb<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_05::oklab_to_srgb(inp, out_slice);
    Ok(out_arr)
}

// ---- func_06: CIE 1964 U*V*W* ------------------------------------------

#[pyfunction]
fn white_point_uv1960(illuminant: [f64; 3]) -> (f64, f64) {
    navette::color::func_06::white_point_uv1960(&illuminant)
}

#[pyfunction(name = "XYZ_to_UVW")]
#[pyo3(signature = (xyz, illuminant=None))]
fn xyz_to_uvw<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, illuminant: Option<[f64; 3]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let illum = illuminant.unwrap_or(REF_WHITE_D65);
    let (un, vn) = navette::color::func_06::white_point_uv1960(&illum);
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_06::xyz_to_uvw(inp, un, vn, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "UVW_to_XYZ")]
#[pyo3(signature = (uvw, illuminant=None))]
fn uvw_to_xyz<'py>(py: Python<'py>, uvw: PyReadonlyArray2<'py, f64>, illuminant: Option<[f64; 3]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&uvw)?;
    let illum = illuminant.unwrap_or(REF_WHITE_D65);
    let (un, vn) = navette::color::func_06::white_point_uv1960(&illum);
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_06::uvw_to_xyz(inp, un, vn, out_slice);
    Ok(out_arr)
}

// ---- func_07: CIE 1960 UCS & chromaticity ------------------------------

#[pyfunction(name = "XYZ_to_UCS")]
fn xyz_to_ucs<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_07::xyz_to_ucs(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "UCS_to_XYZ")]
fn ucs_to_xyz<'py>(py: Python<'py>, ucs: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&ucs)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_07::ucs_to_xyz(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "XYZ_to_UCS_uv")]
fn xyz_to_ucs_uv<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out2(py, inp.len());
    navette::color::func_07::xyz_to_ucs_uv(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "Luv_uv_to_xy")]
fn uv1976_to_xy<'py>(py: Python<'py>, uvp: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice2(&uvp)?;
    let (out_arr, out_slice) = new_out2(py, inp.len());
    navette::color::func_07::uv1976_to_xy(inp, out_slice);
    Ok(out_arr)
}

#[pyfunction(name = "UCS_uv_to_xy")]
fn uv1960_to_xy<'py>(py: Python<'py>, uv: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice2(&uv)?;
    let (out_arr, out_slice) = new_out2(py, inp.len());
    navette::color::func_07::uv1960_to_xy(inp, out_slice);
    Ok(out_arr)
}

// ---- func_08: Bradford chromatic adaptation ----------------------------

#[pyfunction(name = "chromatic_adaptation_VonKries")]
#[pyo3(signature = (xyz, src_white, dst_white, clip_negative=true))]
fn adapt<'py>(
    py: Python<'py>,
    xyz: PyReadonlyArray2<'py, f64>,
    src_white: [f64; 3],
    dst_white: [f64; 3],
    clip_negative: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_08::adapt(inp, &src_white, &dst_white, clip_negative, out_slice);
    Ok(out_arr)
}

/// Row-vector Bradford matrix: `adapted = white @ M`. Returns a (3, 3) array.
#[pyfunction]
fn calc_transform_matrix<'py>(py: Python<'py>, src_white: [f64; 3], dst_white: [f64; 3]) -> Bound<'py, PyArray2<f64>> {
    let m = navette::color::func_08::calc_transform_matrix(&src_white, &dst_white);
    let flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((3, 3), flat).expect("3x3").into_pyarray(py)
}

// ---- func_09..12 & 16: Delta-E metrics (return 1-D arrays) -------------

#[pyfunction(name = "delta_E_CIE2000")]
#[pyo3(signature = (lab1, lab2, k_L=1.0, k_C=1.0, k_H=1.0, textiles=false))]
#[allow(non_snake_case)]
fn delta_e_2000<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>, k_L: f64, k_C: f64, k_H: f64, textiles: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    let (kl, kc, kh) = if textiles { (2.0, 1.0, 1.0) } else { (k_L, k_C, k_H) };
    Ok(navette::color::func_16::delta_e_2000(a, b, kl, kc, kh).into_pyarray(py))
}

#[pyfunction(name = "delta_E_CIE1976")]
fn delta_e_76<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    Ok(navette::color::func_09::delta_e_76(a, b).into_pyarray(py))
}

#[pyfunction(name = "delta_E_CIE1994")]
#[pyo3(signature = (lab1, lab2, textiles=false))]
fn delta_e_94<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>, textiles: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    let p = if textiles {
        navette::color::func_10::De94Params::TEXTILES
    } else {
        navette::color::func_10::De94Params::GRAPHIC
    };
    Ok(navette::color::func_10::delta_e_94(a, b, p).into_pyarray(py))
}

#[pyfunction(name = "delta_E_CMC")]
#[pyo3(signature = (lab1, lab2, pl=2.0, pc=1.0))]
fn delta_e_cmc<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>, pl: f64, pc: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    Ok(navette::color::func_11::delta_e_cmc(a, b, pl, pc).into_pyarray(py))
}

#[pyfunction(name = "delta_E_DIN99")]
#[pyo3(signature = (lab1, lab2, textiles=false))]
fn delta_e_din99<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>, textiles: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    let (ke, kch) = if textiles { (2.0, 0.5) } else { (1.0, 1.0) };
    Ok(navette::color::func_12::delta_e_din99(a, b, ke, kch).into_pyarray(py))
}

// ---- func_13: spectral pipeline ----------------------------------------

#[pyfunction(name = "spectral_to_sRGB")]
#[pyo3(signature = (spd, cmfs, illum, interval, apply_adaptation=true))]
fn spectral_to_srgb<'py>(
    py: Python<'py>,
    spd: PyReadonlyArray1<'py, f64>,
    cmfs: PyReadonlyArray2<'py, f64>,
    illum: PyReadonlyArray1<'py, f64>,
    interval: f64,
    apply_adaptation: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spd = as_slice1(&spd)?;
    let cmfs = as_slice3(&cmfs)?;
    let illum = as_slice1(&illum)?;
    let rgb = navette::color::func_13::spectral_to_srgb(spd, cmfs, illum, interval, apply_adaptation);
    Ok(rgb.to_vec().into_pyarray(py))
}

// ---- func_14: photometry engine ----------------------------------------

#[pyclass(name = "PhotometryEngine")]
struct PyPhotometry {
    inner: navette::color::func_14::PhotometryEngine,
}

#[pymethods]
impl PyPhotometry {
    #[new]
    #[pyo3(signature = (v_photopic, v_scotopic, km_p=683.002, km_s=1700.05))]
    fn new(
        v_photopic: PyReadonlyArray1<'_, f64>,
        v_scotopic: PyReadonlyArray1<'_, f64>,
        km_p: f64,
        km_s: f64,
    ) -> PyResult<Self> {
        Ok(PyPhotometry {
            inner: navette::color::func_14::PhotometryEngine::with_constants(
                as_slice1(&v_photopic)?.to_vec(),
                as_slice1(&v_scotopic)?.to_vec(),
                km_p,
                km_s,
            ),
        })
    }

    /// `vision` is one of "photopic", "scotopic", "mesopic".
    /// `m` is the mesopic adaptation factor (1 = photopic, 0 = scotopic).
    #[pyo3(signature = (spd, vision="photopic", m=1.0, interval=1.0))]
    fn calculate_flux(&self, spd: PyReadonlyArray1<'_, f64>, vision: &str, m: f64, interval: f64) -> PyResult<f64> {
        let v = match vision.to_ascii_lowercase().as_str() {
            "photopic" => navette::color::func_14::Vision::Photopic,
            "scotopic" => navette::color::func_14::Vision::Scotopic,
            "mesopic" => navette::color::func_14::Vision::Mesopic,
            other => return Err(PyValueError::new_err(format!("unknown vision '{other}'"))),
        };
        Ok(self.inner.calculate_flux(as_slice1(&spd)?, v, m, interval))
    }

    fn calculate_sp_ratio(&self, spd: PyReadonlyArray1<'_, f64>, interval: f64) -> PyResult<f64> {
        Ok(self.inner.calculate_sp_ratio(as_slice1(&spd)?, interval))
    }
}

// ---- module registration -----------------------------------------------

#[pymodule]
pub fn _color(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("REF_WHITE_D65", REF_WHITE_D65.to_vec().into_pyarray(m.py()))?;
    m.add("REF_WHITE_D50", REF_WHITE_D50.to_vec().into_pyarray(m.py()))?;

    macro_rules! reg { ($($f:ident),* $(,)?) => { $( m.add_function(wrap_pyfunction!($f, m)?)?; )* }; }
    reg!(
        // Base Core (common)
        srgb_to_xyz, xyz_to_srgb, xyz_to_lab, lab_to_xyz,

        // Convenience Pipelines & Gamut (composites)
        srgb_to_lab, lab_to_srgb, srgb_to_lch, lch_to_srgb,
        srgb_to_luv, luv_to_srgb, srgb_to_xy_y, xy_y_to_srgb,
        clip_absolute, set_strict_ieee,

        // func_01 to func_07
        xyz_to_xyy, xyy_to_xyz,
        lab_to_lch, lch_to_lab,
        xyz_to_luv, luv_to_xyz,
        xyz_to_oklab, oklab_to_xyz,
        srgb_to_oklab, oklab_to_srgb,
        white_point_uv1960, xyz_to_uvw, uvw_to_xyz,
        xyz_to_ucs, ucs_to_xyz, xyz_to_ucs_uv, uv1976_to_xy, uv1960_to_xy,

        // func_08 (Bradford)
        adapt, calc_transform_matrix,

        // func_09 to func_12 & 16 (Metrics)
        delta_e_2000, delta_e_76, delta_e_94, delta_e_cmc, delta_e_din99,

        // func_13 (Spectral)
        spectral_to_srgb,
    );
    m.add_class::<PyPhotometry>()?;
    Ok(())
}
