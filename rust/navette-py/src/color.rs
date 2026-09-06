//! Thin PyO3 bindings for the Navette color engine.
//!
//! No math here: every kernel lives in the pure-Rust `navette-color` core.
//! Each `#[pyfunction]` owns its NumPy inputs, releases the GIL while the
//! (possibly rayon-parallel) kernel runs, and returns a NumPy array.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use navette::color::common::{REF_WHITE_D50, REF_WHITE_D65};

/// Broadcast guard for the ΔE batch entry points: the core's `map_pairs`
/// panics on incompatible shapes, and a Rust panic crossing the FFI
/// boundary surfaces as an uncatchable-looking `PanicException`. Reject
/// early with a clean `ValueError` instead (same rule as the core).
fn check_broadcast(a: &[[f64; 3]], b: &[[f64; 3]]) -> PyResult<()> {
    let (n1, n2) = (a.len(), b.len());
    if n1 == n2 || n1 == 1 || n2 == 1 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "shapes {n1} and {n2} are not broadcastable"
        )))
    }
}

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

/// Batch sRGB (gamma-encoded, N×3) to CIE XYZ. `clip` clamps inputs to [0, 1] first (default true).
#[pyfunction(name = "sRGB_to_XYZ")]
#[pyo3(signature = (rgb, clip=true))]
fn srgb_to_xyz<'py>(py: Python<'py>, rgb: PyReadonlyArray2<'py, f64>, clip: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&rgb)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::common::srgb_to_xyz(inp, clip, out_slice);
    Ok(out_arr)
}

/// Batch CIE XYZ to sRGB (gamma-encoded). `clip` clamps linear RGB to [0, 1] first (default true).
#[pyfunction(name = "XYZ_to_sRGB")]
#[pyo3(signature = (xyz, clip=true))]
fn xyz_to_srgb<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, clip: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::common::xyz_to_srgb(inp, clip, out_slice);
    Ok(out_arr)
}

/// Batch CIE XYZ to CIELAB. `illuminant` reference white, defaults to D65.
#[pyfunction(name = "XYZ_to_Lab")]
#[pyo3(signature = (xyz, illuminant=None))]
fn xyz_to_lab<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, illuminant: Option<[f64; 3]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let illum = illuminant.unwrap_or(REF_WHITE_D65);
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::common::xyz_to_lab(inp, &illum, out_slice);
    Ok(out_arr)
}

/// Batch CIELAB to CIE XYZ. `illuminant` reference white, defaults to D65.
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

/// Convenience pipeline sRGB to CIELAB (D65) via XYZ.
#[pyfunction(name = "sRGB_to_Lab")]
fn srgb_to_lab<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_lab(inp, out_slice); 
    Ok(out_arr)
}

/// Convenience pipeline CIELAB (D65) to sRGB via XYZ, clipped.
#[pyfunction(name = "Lab_to_sRGB")]
fn lab_to_srgb<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::lab_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

/// Convenience pipeline sRGB to cylindrical CIELCh (D65): [L, C, h-deg].
#[pyfunction(name = "sRGB_to_LCHab")]
fn srgb_to_lch<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_lch(inp, out_slice); 
    Ok(out_arr)
}

/// Convenience pipeline cylindrical CIELCh (D65) to sRGB, clipped.
#[pyfunction(name = "LCHab_to_sRGB")]
fn lch_to_srgb<'py>(py: Python<'py>, lch: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lch)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::lch_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

/// Convenience pipeline sRGB to CIELUV (D65) via XYZ.
#[pyfunction(name = "sRGB_to_Luv")]
fn srgb_to_luv<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_luv(inp, out_slice); 
    Ok(out_arr)
}

/// Convenience pipeline CIELUV (D65) to sRGB, clipped.
#[pyfunction(name = "Luv_to_sRGB")]
fn luv_to_srgb<'py>(py: Python<'py>, luv: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&luv)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::luv_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

/// Convenience pipeline sRGB to xyY chromaticity (D65): [x, y, Y].
#[pyfunction(name = "sRGB_to_xyY")]
fn srgb_to_xy_y<'py>(py: Python<'py>, srgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&srgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::srgb_to_xyy_bound(inp, out_slice); 
    Ok(out_arr)
}

/// Convenience pipeline xyY chromaticity (D65) to sRGB, clipped.
#[pyfunction(name = "xyY_to_sRGB")]
fn xy_y_to_srgb<'py>(py: Python<'py>, xyy: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyy)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::xyy_to_srgb(inp, out_slice); 
    Ok(out_arr)
}

/// Clamp every channel of an RGB batch into [0, 1] (gamut clip).
#[pyfunction]
fn clip_absolute<'py>(py: Python<'py>, rgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&rgb)?; 
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::composites::clip_absolute(inp, out_slice); 
    Ok(out_arr)
}

/// No-op kept for API parity: the Rust engine is inherently strict-IEEE.
#[pyfunction]
#[pyo3(signature = (enabled=true))]
#[allow(unused_variables)]
fn set_strict_ieee(enabled: bool) {
    // Rust lacks Numba's fastmath-reassociation toggle, so IEEE is inherently strict here.
    // Provided solely for drop-in API parity with Python.
}

// ---- func_01: XYZ <-> xyY ----------------------------------------------

/// CIE XYZ to xyY chromaticity: [x, y, Y]. Black maps to zeros.
#[pyfunction(name = "XYZ_to_xyY")]
fn xyz_to_xyy<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_01::xyz_to_xyy(inp, out_slice);
    Ok(out_arr)
}

/// xyY chromaticity back to CIE XYZ.
#[pyfunction(name = "xyY_to_XYZ")]
fn xyy_to_xyz<'py>(py: Python<'py>, xyy: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyy)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_01::xyy_to_xyz(inp, out_slice);
    Ok(out_arr)
}

// ---- func_02: Lab <-> LCh ----------------------------------------------

/// CIELAB to cylindrical CIELCh: [L, C, h] with hue in [0, 360) degrees.
#[pyfunction(name = "Lab_to_LCHab")]
fn lab_to_lch<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_02::lab_to_lch(inp, out_slice);
    Ok(out_arr)
}

/// Cylindrical CIELCh back to CIELAB.
#[pyfunction(name = "LCHab_to_Lab")]
fn lch_to_lab<'py>(py: Python<'py>, lch: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lch)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_02::lch_to_lab(inp, out_slice);
    Ok(out_arr)
}

// ---- func_03: XYZ <-> CIELUV (illuminant defaults to D65) ---------------

/// CIE XYZ to CIELUV. `illuminant` reference white, defaults to D65.
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

/// CIELUV back to CIE XYZ. `illuminant` reference white, defaults to D65.
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

/// CIE XYZ to Oklab (direct cone-response matrices).
#[pyfunction(name = "XYZ_to_Oklab")]
fn xyz_to_oklab<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_04::xyz_to_oklab(inp, out_slice);
    Ok(out_arr)
}

/// Oklab back to CIE XYZ.
#[pyfunction(name = "Oklab_to_XYZ")]
fn oklab_to_xyz<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_04::oklab_to_xyz(inp, out_slice);
    Ok(out_arr)
}

// ---- func_05: sRGB <-> Oklab (legacy) ----------------------------------

/// sRGB to Oklab via the legacy sRGB matrices.
#[pyfunction(name = "sRGB_to_Oklab")]
fn srgb_to_oklab<'py>(py: Python<'py>, rgb: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&rgb)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_05::srgb_to_oklab(inp, out_slice);
    Ok(out_arr)
}

/// Oklab back to sRGB via the legacy sRGB matrices, clipped.
#[pyfunction(name = "Oklab_to_sRGB")]
fn oklab_to_srgb<'py>(py: Python<'py>, lab: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&lab)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_05::oklab_to_srgb(inp, out_slice);
    Ok(out_arr)
}

// ---- func_06: CIE 1964 U*V*W* ------------------------------------------

/// CIE 1960 (u, v) chromaticity of an XYZ illuminant.
#[pyfunction]
fn white_point_uv1960(illuminant: [f64; 3]) -> (f64, f64) {
    navette::color::func_06::white_point_uv1960(&illuminant)
}

/// CIE XYZ to CIE 1964 U*V*W*. `illuminant` reference white, defaults to D65.
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

/// CIE 1964 U*V*W* back to CIE XYZ. `illuminant` defaults to D65.
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

/// CIE XYZ to CIE 1960 UCS.
#[pyfunction(name = "XYZ_to_UCS")]
fn xyz_to_ucs<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_07::xyz_to_ucs(inp, out_slice);
    Ok(out_arr)
}

/// CIE 1960 UCS back to CIE XYZ.
#[pyfunction(name = "UCS_to_XYZ")]
fn ucs_to_xyz<'py>(py: Python<'py>, ucs: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&ucs)?;
    let (out_arr, out_slice) = new_out3(py, inp.len());
    navette::color::func_07::ucs_to_xyz(inp, out_slice);
    Ok(out_arr)
}

/// CIE XYZ to CIE 1960 (u, v) chromaticity pairs.
#[pyfunction(name = "XYZ_to_UCS_uv")]
fn xyz_to_ucs_uv<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice3(&xyz)?;
    let (out_arr, out_slice) = new_out2(py, inp.len());
    navette::color::func_07::xyz_to_ucs_uv(inp, out_slice);
    Ok(out_arr)
}

/// CIE 1976 (u', v') chromaticity to xy.
#[pyfunction(name = "Luv_uv_to_xy")]
fn uv1976_to_xy<'py>(py: Python<'py>, uvp: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice2(&uvp)?;
    let (out_arr, out_slice) = new_out2(py, inp.len());
    navette::color::func_07::uv1976_to_xy(inp, out_slice);
    Ok(out_arr)
}

/// CIE 1960 (u, v) chromaticity to xy.
#[pyfunction(name = "UCS_uv_to_xy")]
fn uv1960_to_xy<'py>(py: Python<'py>, uv: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let inp = as_slice2(&uv)?;
    let (out_arr, out_slice) = new_out2(py, inp.len());
    navette::color::func_07::uv1960_to_xy(inp, out_slice);
    Ok(out_arr)
}

// ---- func_08: Bradford chromatic adaptation ----------------------------

/// Bradford chromatic adaptation between white points. Set `clip_negative` to clamp tiny negatives.
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
/// 3x3 Bradford adaptation matrix between two white points (row-vector convention).
#[pyfunction]
fn calc_transform_matrix<'py>(py: Python<'py>, src_white: [f64; 3], dst_white: [f64; 3]) -> Bound<'py, PyArray2<f64>> {
    let m = navette::color::func_08::calc_transform_matrix(&src_white, &dst_white);
    let flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((3, 3), flat).expect("3x3").into_pyarray(py)
}

// ---- func_09..12 & 16: Delta-E metrics (return 1-D arrays) -------------

/// CIEDE2000 colour difference with k_L/k_C/k_H weights (or `textiles` preset). Broadcasts 1-vs-N.
#[pyfunction(name = "delta_E_CIE2000")]
#[pyo3(signature = (lab1, lab2, k_L=1.0, k_C=1.0, k_H=1.0, textiles=false))]
#[allow(non_snake_case)]
fn delta_e_2000<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>, k_L: f64, k_C: f64, k_H: f64, textiles: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    let (kl, kc, kh) = if textiles { (2.0, 1.0, 1.0) } else { (k_L, k_C, k_H) };
    check_broadcast(a, b)?;
    Ok(navette::color::func_16::delta_e_2000(a, b, kl, kc, kh).into_pyarray(py))
}

/// CIE 1976 colour difference (Euclidean distance in CIELAB). Broadcasts 1-vs-N.
#[pyfunction(name = "delta_E_CIE1976")]
fn delta_e_76<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    check_broadcast(a, b)?;
    Ok(navette::color::func_09::delta_e_76(a, b).into_pyarray(py))
}

/// CIE 1994 colour difference (lab1 is the reference). `textiles` selects the textile weights. Broadcasts 1-vs-N.
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
    check_broadcast(a, b)?;
    Ok(navette::color::func_10::delta_e_94(a, b, p).into_pyarray(py))
}

/// CMC(l:c) colour difference with lightness/chroma weights `pl`/`pc` (acceptability 2.0/1.0). Broadcasts 1-vs-N.
#[pyfunction(name = "delta_E_CMC")]
#[pyo3(signature = (lab1, lab2, pl=2.0, pc=1.0))]
fn delta_e_cmc<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>, pl: f64, pc: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    check_broadcast(a, b)?;
    Ok(navette::color::func_11::delta_e_cmc(a, b, pl, pc).into_pyarray(py))
}

/// DIN99 colour difference. `textiles` selects the textile weights. Broadcasts 1-vs-N.
#[pyfunction(name = "delta_E_DIN99")]
#[pyo3(signature = (lab1, lab2, textiles=false))]
fn delta_e_din99<'py>(py: Python<'py>, lab1: PyReadonlyArray2<'py, f64>, lab2: PyReadonlyArray2<'py, f64>, textiles: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = as_slice3(&lab1)?;
    let b = as_slice3(&lab2)?;
    let (ke, kch) = if textiles { (2.0, 0.5) } else { (1.0, 1.0) };
    check_broadcast(a, b)?;
    Ok(navette::color::func_12::delta_e_din99(a, b, ke, kch).into_pyarray(py))
}

// ---- func_13: spectral pipeline ----------------------------------------

/// Integrate an SPD against CMFs and an illuminant to sRGB in [0, 1]; optionally Bradford-adapt to D65.
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

/// Photometry engine holding V-lambda and V-prime curves plus efficacy constants.
///
/// Construct with photopic/scotopic luminous-efficiency vectors (and optional
/// `km_p`/`km_s` efficacies in lm/W), then integrate SPDs to luminous flux.
#[pyclass(name = "PhotometryEngine")]
struct PyPhotometry {
    inner: navette::color::func_14::PhotometryEngine,
}

#[pymethods]
impl PyPhotometry {
    /// Build the engine from V(lambda) and V'(lambda) curves of equal length.
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

    /// Scotopic-to-photopic flux ratio of an SPD (0.0 when photopic flux vanishes).
    fn calculate_sp_ratio(&self, spd: PyReadonlyArray1<'_, f64>, interval: f64) -> PyResult<f64> {
        Ok(self.inner.calculate_sp_ratio(as_slice1(&spd)?, interval))
    }
}

// ---- CIE reference tables ----------------------------------------------

/// Parse CIE DataCite JSON text into `{column: float64 array}` (thin over
/// `color::tables::parse_cie_tables` — the single canonical parser).
#[pyfunction]
fn parse_cie_tables<'py>(py: Python<'py>, text: &str) -> PyResult<Py<PyDict>> {
    let table =
        navette::color::tables::parse_cie_tables(text).map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    for name in table.column_names() {
        let col = table.column(name).expect("name came from the table");
        out.set_item(name, PyArray1::from_vec(py, col.to_vec()))?;
    }
    Ok(out.into())
}

/// `(wavelengths, x, y, z)` triplet for CMF/chromaticity files (thin over
/// `xyz_column_names`; refuses non-triplet files naming their columns).
#[pyfunction]
fn cie_xyz_triplet(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyTuple;
    let table =
        navette::color::tables::parse_cie_tables(text).map_err(PyValueError::new_err)?;
    let wl = table.lambda().ok_or_else(|| {
        PyValueError::new_err("CIE table: no 'lambda' column for the XYZ triplet.")
    })?;
    let [x, y, z] = table.xyz_column_names().map_err(PyValueError::new_err)?;
    let get = |n: &str| table.column(n).expect("name came from the triplet").to_vec();
    let tup = PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, wl.to_vec()).into_any(),
            PyArray1::from_vec(py, get(&x)).into_any(),
            PyArray1::from_vec(py, get(&y)).into_any(),
            PyArray1::from_vec(py, get(&z)).into_any(),
        ],
    )?;
    Ok(tup.into_any().unbind())
}

// ---- module registration -----------------------------------------------

/// Register the colorimetry submodule (`navette._navette._color`).
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

        // tables (CIE reference data)
        parse_cie_tables, cie_xyz_triplet,
    );
    m.add_class::<PyPhotometry>()?;
    Ok(())
}
