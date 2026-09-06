//! Python bindings for the synthesis merit bridge (`navette-smatrix`).
//!
//! Exposes `MeritSpec` (flat optimization targets + residual kernel),
//! `SimCurves` (simulated/derived rows on the solver grid) and the
//! `build_needle_targets` fold. The intended producer is the Python
//! converter (`TargetCollection.build_merit_spec`), which copies finished
//! ingestion products out of `TargetWeaver.export_entries`.

use std::sync::Arc;

use num_complex::Complex64;
use numpy::{PyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use navette::smatrix::synthesis::merit::{
    ConstraintKind, CurveId, MeritKey, MeritSpec, MeritTarget, SimCurves, SimTransform,
};
use navette::smatrix::synthesis::needle_pass::build_needle_targets as core_fold;

fn parse_curve(s: &str) -> PyResult<CurveId> {
    CurveId::from_str(s).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Invalid curve id '{s}' (use Rs/Rp/Ru/Ts/Tp/Tu, As/Ap/Au, \
             RBs/RBp/RBu/TBs/TBp/TBu, ABs/ABp/ABu)"
        ))
    })
}

fn parse_kind(s: &str) -> PyResult<ConstraintKind> {
    ConstraintKind::from_str(s).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Invalid kind '{s}' (use 'e', 'a', 'b', 'r' or 'c')"
        ))
    })
}

fn parse_transform(s: &str) -> PyResult<SimTransform> {
    SimTransform::from_str(s).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Invalid transform '{s}' (use 'linear', 'log', 'phase' or 'complex')"
        ))
    })
}

#[pyclass(name = "SimCurves")]
/// Simulated rows on the solver grid (see `SimCurves` in the core).
pub struct PySimCurves {
    inner: SimCurves,
}

impl PySimCurves {
    pub(crate) fn wrap(inner: SimCurves) -> Self {
        PySimCurves { inner }
    }
}

#[pymethods]
impl PySimCurves {
    #[new]
    #[pyo3(signature = (angles, wavelengths, total_d=0.0, n_front=1.0, n_back=1.0))]
    /// Empty rows on the given axes; fill with `set_curve`/`set_complex`.
    /// `total_d`/`n_front`/`n_back` are the stack metadata for
    /// differential-phase (`PDts`/`PDtp`) demands (defaults zero the
    /// reference, i.e. differential ≡ absolute).
    fn new(
        angles: PyReadonlyArray1<'_, f64>,
        wavelengths: PyReadonlyArray1<'_, f64>,
        total_d: f64,
        n_front: f64,
        n_back: f64,
    ) -> PyResult<Self> {
        let a = angles.as_slice()?;
        let w = wavelengths.as_slice()?;
        Ok(PySimCurves {
            inner: SimCurves {
                angles: Arc::from(a),
                wavelengths: Arc::from(w),
                total_d,
                n_front_re: n_front,
                n_back_re: n_back,
                curves: Default::default(),
                back: Default::default(),
                cplx: Default::default(),
                cplx_back: Default::default(),
            },
        })
    }

    /// Store one intensity row (`Rs/…/ABu`; absorption ids rejected —
    /// absorptance derives from companions).
    fn set_curve(
        &mut self,
        curve_id: String,
        values: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        // Thin over the core setter (key rules + lengths validated there).
        let id = parse_curve(&curve_id)?;
        let arc: Arc<[f64]> = Arc::from(values.as_slice()?);
        self.inner
            .set_curve(id, arc)
            .map_err(PyValueError::new_err)
    }

    /// Store one complex-amplitude row for phase demands (`Rs/Rp/Ts/Tp`,
    /// `RBs/RBp/TBs/TBp`; absorption/unpolarized rejected).
    fn set_complex(
        &mut self,
        curve_id: String,
        values: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<()> {
        // Thin over the core setter (key rules + lengths validated there).
        let id = parse_curve(&curve_id)?;
        let arc: Arc<[Complex64]> = Arc::from(values.as_slice()?);
        self.inner
            .set_complex(id, arc)
            .map_err(PyValueError::new_err)
    }
}

#[pyclass(name = "MeritSpec")]
/// Flat optimization targets (see `MeritSpec` in the core).
pub struct PyMeritSpec {
    inner: MeritSpec,
}

impl PyMeritSpec {
    pub(crate) fn inner(&self) -> &MeritSpec {
        &self.inner
    }

    pub(crate) fn from_inner(inner: MeritSpec) -> Self {
        PyMeritSpec { inner }
    }
}

/// Compile a target set (JSON) into a native `MeritSpec` (thin over
/// `targets::compile_merit_spec`): the `build_merit_spec` path.
#[pyfunction]
pub(crate) fn compile_merit_spec(request_json: &str) -> PyResult<PyMeritSpec> {
    let set: navette::smatrix::synthesis::targets::TargetSet =
        serde_json::from_str(request_json)
            .map_err(|e| PyValueError::new_err(format!("compile_merit_spec: invalid request: {e}")))?;
    navette::smatrix::synthesis::targets::compile_merit_spec(&set)
        .map(PyMeritSpec::from_inner)
        .map_err(PyValueError::new_err)
}

#[pymethods]
impl PyMeritSpec {
    #[new]
    fn new() -> Self {
        PyMeritSpec { inner: MeritSpec::new() }
    }

    /// Register a `(angle, curve)` demand group; returns its key index.
    fn add_key(&mut self, angle: f64, curve_id: String) -> PyResult<usize> {
        let id = parse_curve(&curve_id)?;
        Ok(self.inner.add_key(MeritKey { angle, curve: id }))
    }

    #[pyo3(signature = (key_idx, wavelengths, normalized, tolerances, kind, transform,
                        norm_factor, band=None, phase=false, differential_passes=None,
                        weight=1.0, count_norm=None, integral=false))]
    /// Append one target frame. Arrays are per-point values on `wavelengths`.
    /// `differential_passes` (None = absolute phase; 1.0 = `PDts`/`PDtp`
    /// transmitted) subtracts the equivalent-medium reference (see
    /// `SimCurves` metadata); requires `phase=true`.
    /// `weight` scales the frame's merit sum; `count_norm` (target-level
    /// point count, resolved by the converter) divides it. `integral`
    /// constrains the mean of the scaled diffs (single residual).
    #[allow(clippy::too_many_arguments)]
    fn add_target(
        &mut self,
        key_idx: u32,
        wavelengths: PyReadonlyArray1<'_, f64>,
        normalized: PyReadonlyArray1<'_, f64>,
        tolerances: PyReadonlyArray1<'_, f64>,
        kind: String,
        transform: String,
        norm_factor: f64,
        band: Option<PyReadonlyArray1<'_, f64>>,
        phase: bool,
        differential_passes: Option<f64>,
        weight: f64,
        count_norm: Option<f64>,
        integral: bool,
    ) -> PyResult<()> {
        let wl = wavelengths.as_slice()?;
        let nt = normalized.as_slice()?;
        let tol = tolerances.as_slice()?;
        let b: Vec<f64> = match band {
            Some(a) => a.as_slice()?.to_vec(),
            None => Vec::new(),
        };
        self.inner
            .add_target(MeritTarget {
                key_idx,
                wavelengths: Arc::from(wl),
                kind: parse_kind(&kind)?,
                transform: parse_transform(&transform)?,
                norm_factor,
                normalized_targets: Arc::from(nt),
                tolerances: Arc::from(tol),
                band: Arc::from(b.as_slice()),
                phase,
                differential_passes,
                weight,
                count_norm,
                integral,
            })
            .map_err(PyValueError::new_err)
    }

    /// Scalar merit: Σ residual² + `missing_penalty` per missing key group.
    fn merit(
        &self,
        py: Python<'_>,
        sim: &PySimCurves,
        missing_penalty: f64,
    ) -> f64 {
        let inner = &self.inner;
        let sim_inner = &sim.inner;
        py.detach(move || inner.merit(sim_inner, missing_penalty))
    }

    /// Fixed-length residual vector (zeros where inactive).
    fn residuals(&self, py: Python<'_>, sim: &PySimCurves) -> PyResult<Py<PyArray<f64, numpy::Ix1>>> {
        let mut out = Vec::new();
        self.inner
            .residuals(&sim.inner, &mut out)
            .map_err(|id| PyValueError::new_err(format!("missing curve for demand {id:?}")))?;
        Ok(PyArray::from_vec(py, out).unbind())
    }

    /// Total residual components (active or not).
    fn n_residuals(&self) -> usize {
        self.inner.n_residuals()
    }
}

/// Reference-rotation factors for differential-phase demands (thin over
/// the core kernel): `exp(-i·ref)` per wavelength.
#[pyfunction]
#[pyo3(signature = (wavelengths, angle_deg, n_inc=1.0, total_d=0.0, passes=1.0))]
pub fn reference_rotation(
    py: Python<'_>,
    wavelengths: PyReadonlyArray1<'_, f64>,
    angle_deg: f64,
    n_inc: f64,
    total_d: f64,
    passes: f64,
) -> PyResult<Py<PyArray1<Complex64>>> {
    use navette::smatrix::synthesis::merit::reference_rotation as core_rot;
    let rot = core_rot(wavelengths.as_slice()?, angle_deg, n_inc, total_d, passes);
    Ok(PyArray::from_vec(py, rot).into())
}

/// Fold a spec into per-quantity `(targets, weights)` pairs (angle-major).
/// Returns a dict with `r/t/a/rb/tb/ab` pairs plus `phi0..phi3` (one pair
/// per S-matrix channel — emit one `P_PHI` call per used channel).
#[pyfunction]
#[pyo3(signature = (spec, angles, wavelengths, sim=None))]
pub fn build_needle_targets(
    py: Python<'_>,
    spec: &PyMeritSpec,
    angles: PyReadonlyArray1<'_, f64>,
    wavelengths: PyReadonlyArray1<'_, f64>,
    sim: Option<&PySimCurves>,
) -> PyResult<Py<PyDict>> {
    let a = angles.as_slice()?.to_vec();
    let w = wavelengths.as_slice()?.to_vec();
    let nt = py.detach({
        let spec_inner = &spec.inner;
        let sim_inner = sim.map(|s| &s.inner);
        move || core_fold(spec_inner, &a, &w, sim_inner)
    }).map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    let pair = |py: Python<'_>, name: &str, p: (Vec<f64>, Vec<f64>)| -> PyResult<()> {
        let inner = PyDict::new(py);
        inner.set_item("targets", PyArray::from_vec(py, p.0))?;
        inner.set_item("weights", PyArray::from_vec(py, p.1))?;
        d.set_item(name, inner)?;
        Ok(())
    };
    pair(py, "r", nt.r)?;
    pair(py, "t", nt.t)?;
    pair(py, "a", nt.a)?;
    pair(py, "rb", nt.rb)?;
    pair(py, "tb", nt.tb)?;
    pair(py, "ab", nt.ab)?;
    for (i, p) in nt.phi.into_iter().enumerate() {
        // Exact dM/dD correction for differential demands (0.0 otherwise):
        // subtract from the assembled P_PHI gradient (see `needle_gradient`
        // `gain_shift_phi`). Uniform in z — the needle site never moves.
        let inner = PyDict::new(py);
        inner.set_item("targets", PyArray::from_vec(py, p.0))?;
        inner.set_item("weights", PyArray::from_vec(py, p.1))?;
        inner.set_item("gain_shift", nt.phi_gain_shift[i])?;
        d.set_item(format!("phi{i}"), inner)?;
    }
    Ok(d.unbind())
}
