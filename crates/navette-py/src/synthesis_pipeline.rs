//! Python bindings for the needle synthesis pipeline (`navette-smatrix`).
//!
//! Exposes the optimizer loop as data-in/data-out classes; target definition
//! stays on the spectralweave surface (`TargetCollection` → `build_merit_spec`
//! → `MeritSpec`). The pipeline constructor takes that `MeritSpec` plus the
//! grid and folds it internally (`SpectralInputs::from_spec`, conservative
//! form), so no target-level option is re-declared here:
//!
//! ```python
//! spec = build_merit_spec(collection)          # full target options
//! stack = DesignStack(ambient, substrate, films)
//! pipe = NeedlePipeline(stack, spec, angles, wavls, contrast={...})
//! result = pipe.run()                          # dict + final "stack"
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use num_complex::Complex64;
use numpy::{PyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use navette::smatrix::synthesis::config::PipelineConfig;
use navette::smatrix::synthesis::context::DesignContext;
use navette::smatrix::synthesis::cycle::{ContrastMap, NeedleCycleConfig};
use navette::smatrix::synthesis::evaluator::SmatrixContext;
use navette::smatrix::synthesis::pipeline::{NeedlePipeline, PipelinePhaseResult, SpectralInputs};
use navette::smatrix::synthesis::structure::{DesignStack, LayerSpec};
use navette::smatrix::synthesis::thick_opt::LmConfig;

use crate::synthesis_merit::{PyMeritSpec, PySimCurves};

fn deg_to_sin(angles_deg: &[f64]) -> Vec<f64> {
    angles_deg
        .iter()
        .map(|a| (a * std::f64::consts::PI / 180.0).sin())
        .collect()
}

// ---------------------------------------------------------------------------
// LayerSpec / DesignStack
// ---------------------------------------------------------------------------

#[pyclass(name = "LayerSpec")]
/// One design-stack layer: material name + nk evaluated on the simulation
/// grid (see `LayerSpec` in the core). Evaluate materials in Python
/// (`navette.materials.evaluate`) and pass the array — this class is data.
#[derive(Clone)]
pub struct PyLayerSpec {
    inner: LayerSpec,
}

#[pymethods]
impl PyLayerSpec {
    #[new]
    #[pyo3(signature = (material, nk, thickness, coherent=true, rough_type=0, rough_val=0.0, optimize=true, needle=true))]
    /// `nk`: complex array, one entry per simulation wavelength.
    /// Ambient/substrate entries are fixed by construction (never
    /// optimized, removed, or hosting — pass `optimize=False,
    /// needle=False` for them).
    #[allow(clippy::too_many_arguments)]
    fn new(
        material: String,
        nk: PyReadonlyArray1<'_, Complex64>,
        thickness: f64,
        coherent: bool,
        rough_type: i32,
        rough_val: f64,
        optimize: bool,
        needle: bool,
    ) -> PyResult<Self> {
        let n = nk.as_slice()?;
        if n.is_empty() {
            return Err(PyValueError::new_err("nk must be non-empty"));
        }
        if !thickness.is_finite() || thickness < 0.0 {
            return Err(PyValueError::new_err(format!(
                "thickness must be finite and >= 0 (got {thickness})"
            )));
        }
        Ok(PyLayerSpec {
            inner: LayerSpec {
                material: Arc::from(material),
                nk: Arc::from(n),
                d_nm: thickness,
                coherent,
                rough_type,
                rough_val,
                optimize,
                needle,
            },
        })
    }

    #[getter]
    fn material(&self) -> String {
        self.inner.material.to_string()
    }

    #[getter]
    fn thickness(&self) -> f64 {
        self.inner.d_nm
    }

    #[getter]
    fn nk<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<Complex64, numpy::Ix1>> {
        PyArray::from_slice(py, &self.inner.nk)
    }

    #[getter]
    fn coherent(&self) -> bool {
        self.inner.coherent
    }

    #[getter]
    fn rough_type(&self) -> i32 {
        self.inner.rough_type
    }

    #[getter]
    fn rough_val(&self) -> f64 {
        self.inner.rough_val
    }

    #[getter]
    fn optimize(&self) -> bool {
        self.inner.optimize
    }

    #[getter]
    fn needle(&self) -> bool {
        self.inner.needle
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerSpec(material={:?}, d={:.3}nm, optimize={}, needle={})",
            self.inner.material.as_ref(),
            self.inner.d_nm,
            self.inner.optimize,
            self.inner.needle,
        )
    }
}

fn layer_dict<'a>(py: Python<'a>, l: &'a LayerSpec) -> PyResult<Bound<'a, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("material", l.material.as_ref())?;
    d.set_item("thickness", l.d_nm)?;
    d.set_item("nk", PyArray::from_slice(py, &l.nk))?;
    d.set_item("coherent", l.coherent)?;
    d.set_item("rough_type", l.rough_type)?;
    d.set_item("rough_val", l.rough_val)?;
    d.set_item("optimize", l.optimize)?;
    d.set_item("needle", l.needle)?;
    Ok(d)
}

#[pyclass(name = "DesignStack")]
/// Thin-film stack: fixed ambient + films + fixed substrate (see
/// `DesignStack` in the core). Mutated in place by the pipeline; read back
/// with `films()` / `to_dict()`.
pub struct PyDesignStack {
    inner: DesignStack,
}

impl PyDesignStack {
    pub(crate) fn from_inner(inner: DesignStack) -> Self {
        PyDesignStack { inner }
    }
}

#[pymethods]
impl PyDesignStack {
    #[new]
    /// `films` excludes ambient/substrate (films only — the mutable part).
    fn new(
        py: Python<'_>,
        ambient: Py<PyLayerSpec>,
        substrate: Py<PyLayerSpec>,
        films: Vec<Py<PyLayerSpec>>,
    ) -> PyResult<Self> {
        let a = ambient.bind(py).borrow().inner.clone();
        let s = substrate.bind(py).borrow().inner.clone();
        let f: Vec<LayerSpec> =
            films.iter().map(|l| l.bind(py).borrow().inner.clone()).collect();
        DesignStack::with_films(a, s, f)
            .map(PyDesignStack::from_inner)
            .map_err(PyValueError::new_err)
    }

    /// Expanded construction from design layers (first-class model path).
    ///
    /// `films` are design `Layer`s (material names); `nk` maps film name →
    /// evaluated nk on `wavelengths`; `groups` maps material → policy.
    /// Groups, nk scaling, roughness and interface slices expand here, so
    /// the silent-drop limitation is gone; graded profiles refuse loudly.
    #[staticmethod]
    #[pyo3(signature = (ambient, substrate, films, nk, groups, wavelengths))]
    fn from_design(
        py: Python<'_>,
        ambient: Py<PyLayerSpec>,
        substrate: Py<PyLayerSpec>,
        films: Vec<crate::structure::PyLayer>,
        nk: HashMap<String, PyReadonlyArray1<Complex64>>,
        groups: HashMap<String, crate::structure::PyGroup>,
        wavelengths: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let a = ambient.bind(py).borrow().inner.clone();
        let s = substrate.bind(py).borrow().inner.clone();
        let design: Vec<navette_structure::Layer> =
            films.iter().map(|l| l.inner_clone()).collect();
        let nk_map: HashMap<std::sync::Arc<str>, Vec<Complex64>> = nk
            .iter()
            .map(|(k, v)| {
                v.as_slice().map(|sl| (std::sync::Arc::from(k.as_str()), sl.to_vec()))
            })
            .collect::<Result<_, _>>()?;
        let gm: HashMap<String, navette_structure::Group> = groups
            .iter()
            .map(|(k, g)| (k.clone(), g.inner_clone()))
            .collect();
        DesignStack::from_design(a, s, &design, &nk_map, &gm, wavelengths.as_slice()?)
            .map(PyDesignStack::from_inner)
            .map_err(PyValueError::new_err)
    }

    /// Number of films (excludes ambient + substrate).
    fn film_count(&self) -> usize {
        self.inner.films().len()
    }

    /// Total film thickness (nm, excludes ambient + substrate).
    fn total_thickness(&self) -> f64 {
        self.inner.total_thickness_nm()
    }

    /// Films as a list of dicts (ambient/substrate excluded); round-trips
    /// through `LayerSpec(**{material, nk, thickness, ...})`.
    fn films(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner.films().iter().map(|l| layer_dict(py, l).map(|d| d.unbind())).collect()
    }

    /// Full stack incl. ambient/substrate (+ grid size).
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ambient", layer_dict(py, self.inner.ambient())?)?;
        d.set_item("substrate", layer_dict(py, self.inner.substrate())?)?;
        d.set_item("films", self.films(py)?)?;
        d.set_item("num_wavs", self.inner.num_wavs())?;
        Ok(d.unbind())
    }

    /// Set film thickness (films indexing; ambient/substrate excluded).
    fn set_thickness(&mut self, film_idx: usize, thickness: f64) -> PyResult<()> {
        self.inner.set_thickness(film_idx, thickness).map_err(PyValueError::new_err)
    }

    /// Split film `film_idx` and insert `seed` (host portions keep flags).
    fn insert_needle_seed(
        &mut self,
        py: Python<'_>,
        film_idx: usize,
        depth_into_layer_nm: f64,
        seed: Py<PyLayerSpec>,
    ) -> PyResult<()> {
        let s = seed.bind(py).borrow().inner.clone();
        self.inner
            .insert_needle_seed(film_idx, depth_into_layer_nm, s)
            .map_err(PyValueError::new_err)
    }

    /// Merge consecutive same-material films; returns merge count.
    fn merge_adjacent(&mut self) -> usize {
        self.inner.merge_adjacent()
    }

    /// Remove film `film_idx`; returns it as a `LayerSpec`.
    fn remove_film(&mut self, film_idx: usize) -> PyResult<PyLayerSpec> {
        self.inner
            .remove_film(film_idx)
            .map(wrap_layer)
            .map_err(PyValueError::new_err)
    }

    /// Enforce `[min_nm, max_nm]` (sub-min removed, above-max capped);
    /// returns `(n_removed, n_capped)`.
    fn clamp_all(&mut self, min_nm: f64, max_nm: f64) -> PyResult<(usize, usize)> {
        if !(min_nm >= 0.0) || !(max_nm > min_nm) {
            return Err(PyValueError::new_err(
                "need 0 <= min_nm < max_nm",
            ));
        }
        Ok(self.inner.clamp_all(min_nm, max_nm))
    }

    fn __repr__(&self) -> String {
        format!(
            "DesignStack(films={}, total={:.2}nm)",
            self.inner.films().len(),
            self.inner.total_thickness_nm(),
        )
    }
}

fn wrap_layer(l: LayerSpec) -> PyLayerSpec {
    PyLayerSpec { inner: l }
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

#[pyclass(name = "LmConfig", from_py_object)]
/// Bounded Levenberg-Marquardt knobs (see `LmConfig` in the core).
#[derive(Clone)]
pub struct PyLmConfig {
    inner: LmConfig,
}

#[pymethods]
impl PyLmConfig {
    #[new]
    #[pyo3(signature = (max_iterations=200, max_evals=100_000, ftol=1e-12, xtol=1e-12,
                        gtol=1e-10, lambda_init=1e-3, lambda_up=5.0, lambda_down=3.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iterations: usize,
        max_evals: usize,
        ftol: f64,
        xtol: f64,
        gtol: f64,
        lambda_init: f64,
        lambda_up: f64,
        lambda_down: f64,
    ) -> PyResult<Self> {
        for (name, v) in [("ftol", ftol), ("xtol", xtol), ("gtol", gtol),
                          ("lambda_init", lambda_init), ("lambda_up", lambda_up),
                          ("lambda_down", lambda_down)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(PyValueError::new_err(format!("{name} must be finite and > 0")));
            }
        }
        if max_iterations == 0 || max_evals == 0 {
            return Err(PyValueError::new_err("max_iterations/max_evals must be > 0"));
        }
        Ok(PyLmConfig {
            inner: LmConfig {
                max_iterations,
                max_evals,
                ftol,
                xtol,
                gtol,
                lambda_init,
                lambda_up,
                lambda_down,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!("LmConfig(max_iterations={}, ftol={:.1e})", self.inner.max_iterations, self.inner.ftol)
    }
}

#[pyclass(name = "PipelineConfig", from_py_object)]
/// Loop-control parameters (see `PipelineConfig` in the core; validated on
/// pipeline construction — `cleanup_min_nm=None` falls back to `clamp_min_nm`).
#[derive(Clone)]
pub struct PyPipelineConfig {
    inner: PipelineConfig,
}

#[pymethods]
impl PyPipelineConfig {
    #[new]
    #[pyo3(signature = (max_film_layers=40, max_total_thickness_nm=5000.0, max_macro_cycles=50,
                        merit_target=0.0, clamp_min_nm=2.0, clamp_max_nm=1000.0,
                        needles_per_cycle=3, enable_cleanup=true, cleanup_min_nm=None,
                        cleanup_max_removals=None, enable_inflate=false, inflate_addon_qwot=2.0,
                        inflate_reference_wl=550.0, inflate_max_layers=None,
                        stagnation_window=5, stagnation_gradient_tol=1e-4,
                        stagnation_oscillation_ratio=0.75, stagnation_divergence_count=3))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_film_layers: usize,
        max_total_thickness_nm: f64,
        max_macro_cycles: usize,
        merit_target: f64,
        clamp_min_nm: f64,
        clamp_max_nm: f64,
        needles_per_cycle: usize,
        enable_cleanup: bool,
        cleanup_min_nm: Option<f64>,
        cleanup_max_removals: Option<usize>,
        enable_inflate: bool,
        inflate_addon_qwot: f64,
        inflate_reference_wl: f64,
        inflate_max_layers: Option<usize>,
        stagnation_window: usize,
        stagnation_gradient_tol: f64,
        stagnation_oscillation_ratio: f64,
        stagnation_divergence_count: usize,
    ) -> Self {
        PyPipelineConfig {
            inner: PipelineConfig {
                max_film_layers,
                max_total_thickness_nm,
                max_macro_cycles,
                merit_target,
                clamp_min_nm,
                clamp_max_nm,
                needles_per_cycle,
                enable_cleanup,
                cleanup_min_nm,
                cleanup_max_removals,
                enable_inflate,
                inflate_addon_qwot,
                inflate_reference_wl,
                inflate_max_layers,
                stagnation_window,
                stagnation_gradient_tol,
                stagnation_oscillation_ratio,
                stagnation_divergence_count,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineConfig(cycles={}, needles_per_cycle={}, cleanup={}, inflate={})",
            self.inner.max_macro_cycles,
            self.inner.needles_per_cycle,
            self.inner.enable_cleanup,
            self.inner.enable_inflate,
        )
    }
}

#[pyclass(name = "NeedleCycleConfig", from_py_object)]
/// Inner insertion-loop knobs (see `NeedleCycleConfig` in the core).
/// NOTE: the pipeline overwrites `max_needles` with
/// `PipelineConfig.needles_per_cycle` every macro-cycle — it only takes
/// effect for direct (non-pipeline) cycle use.
#[derive(Clone)]
pub struct PyNeedleCycleConfig {
    inner: NeedleCycleConfig,
}

#[pymethods]
impl PyNeedleCycleConfig {
    #[new]
    #[pyo3(signature = (max_needles=10, convergence_threshold=1e-4,
                        needle_seed_thickness_nm=5.0, scan_step_nm=2.0,
                        refold_per_cycle=true))]
    fn new(
        max_needles: usize,
        convergence_threshold: f64,
        needle_seed_thickness_nm: f64,
        scan_step_nm: f64,
        refold_per_cycle: bool,
    ) -> PyResult<Self> {
        if scan_step_nm <= 0.0 || !scan_step_nm.is_finite() {
            return Err(PyValueError::new_err("scan_step_nm must be finite and > 0"));
        }
        if needle_seed_thickness_nm <= 0.0 || !needle_seed_thickness_nm.is_finite() {
            return Err(PyValueError::new_err(
                "needle_seed_thickness_nm must be finite and > 0",
            ));
        }
        Ok(PyNeedleCycleConfig {
            inner: NeedleCycleConfig {
                max_needles,
                convergence_threshold,
                needle_seed_thickness_nm,
                scan_step_nm,
                refold_per_cycle,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "NeedleCycleConfig(max_needles={}, seed={}nm, step={}nm, refold={})",
            self.inner.max_needles,
            self.inner.needle_seed_thickness_nm,
            self.inner.scan_step_nm,
            self.inner.refold_per_cycle,
        )
    }
}

// ---------------------------------------------------------------------------
// Solver context (real DesignContext)
// ---------------------------------------------------------------------------

#[pyclass(name = "SmatrixContext")]
/// Solver + merit context: the pipeline's numeric machinery (see
/// `SmatrixContext` in the core). Usable standalone — `simulate` /
/// `evaluate_merit` / `optimize_thicknesses` on any `DesignStack`.
pub struct PySmatrixContext {
    inner: SmatrixContext,
}

#[pymethods]
impl PySmatrixContext {
    #[new]
    #[pyo3(signature = (spec, angles_deg, wavelengths, clamp_min=2.0, clamp_max=1000.0, lm=None))]
    /// `angles_deg`: degrees (spec-key convention); converted to sines.
    fn new(
        spec: &PyMeritSpec,
        angles_deg: PyReadonlyArray1<'_, f64>,
        wavelengths: PyReadonlyArray1<'_, f64>,
        clamp_min: f64,
        clamp_max: f64,
        lm: Option<Py<PyLmConfig>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let a = angles_deg.as_slice()?;
        let w = wavelengths.as_slice()?;
        if a.is_empty() || w.is_empty() {
            return Err(PyValueError::new_err("angles/wavelengths must be non-empty"));
        }
        if !(clamp_min >= 0.0) || !(clamp_max > clamp_min) {
            return Err(PyValueError::new_err("need 0 <= clamp_min < clamp_max"));
        }
        Ok(PySmatrixContext {
            inner: SmatrixContext {
                wavls: w.to_vec(),
                sin_theta: deg_to_sin(a),
                spec: spec.inner().clone(),
                clamp_min_nm: clamp_min,
                clamp_max_nm: clamp_max,
                lm: lm
                    .map(|l| l.bind(py).borrow().inner.clone())
                    .unwrap_or_default(),
            },
        })
    }

    /// Simulate the stack on the fixed grid (see `SimCurves`).
    fn simulate(&self, py: Python<'_>, stack: &PyDesignStack) -> PyResult<Py<PySimCurves>> {
        let sim = py
            .detach({
                let inner = &self.inner;
                let st = &stack.inner;
                move || inner.simulate(st)
            })
            .map_err(PyValueError::new_err)?;
        Ok(Py::new(py, PySimCurves::wrap(sim))?)
    }

    /// Merit of the stack (missing-curve penalty 1e6).
    fn evaluate_merit(&self, py: Python<'_>, stack: &PyDesignStack) -> PyResult<f64> {
        py.detach({
            let inner = &self.inner;
            let st = &stack.inner;
            move || inner.evaluate_merit(st)
        })
        .map_err(PyValueError::new_err)
    }

    /// Bounded LM over optimize-flagged films, in place; sub-min films
    /// removed, above-max capped. Returns the post-optimization merit.
    fn optimize_thicknesses(
        &mut self,
        py: Python<'_>,
        stack: &mut PyDesignStack,
    ) -> PyResult<f64> {
        py.detach({
            let inner = &mut self.inner;
            let st = &mut stack.inner;
            move || inner.optimize_thicknesses(st)
        })
        .map_err(PyValueError::new_err)
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn insertion_dict(py: Python<'_>, film_idx: usize, depth_nm: f64, material: &str) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("film_idx", film_idx)?;
    d.set_item("depth_into_layer_nm", depth_nm)?;
    d.set_item("material", material)?;
    Ok(d.unbind())
}

fn phase_dict<'a>(py: Python<'a>, phase: &'a PipelinePhaseResult) -> PyResult<Bound<'a, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("macro_cycle", phase.macro_cycle)?;
    d.set_item("mf_after_needle", phase.mf_after_needle)?;
    d.set_item("mf_after_cleanup", phase.mf_after_cleanup)?;
    d.set_item("mf_after_inflate", phase.mf_after_inflate)?;
    d.set_item("mf_end", phase.mf_end)?;
    d.set_item("layer_count", phase.layer_count)?;
    d.set_item("total_thickness_nm", phase.total_thickness_nm)?;
    let mut cycles = Vec::with_capacity(phase.needle_results.len());
    for r in &phase.needle_results {
        let rd = PyDict::new(py);
        rd.set_item("cycle", r.cycle)?;
        rd.set_item("merit_before", r.merit_before)?;
        rd.set_item("merit_after", r.merit_after)?;
        rd.set_item("best_p", r.best_p)?;
        rd.set_item("predicted_improvement", r.predicted_improvement)?;
        rd.set_item("layer_count", r.layer_count)?;
        match &r.insertion {
            Some(ins) => rd.set_item(
                "insertion",
                insertion_dict(py, ins.film_idx, ins.depth_into_layer_nm, ins.material.as_ref())?,
            )?,
            None => rd.set_item("insertion", py.None())?,
        }
        cycles.push(rd.unbind());
    }
    d.set_item("needle_results", cycles)?;
    match &phase.cleanup_result {
        Some(c) => {
            let cd = PyDict::new(py);
            cd.set_item("merit_before", c.merit_before)?;
            cd.set_item("merit_after", c.merit_after)?;
            cd.set_item("layers_before", c.layers_before)?;
            cd.set_item("layers_after", c.layers_after)?;
            cd.set_item("layers_removed_thin", c.layers_removed_thin)?;
            cd.set_item("layers_merged", c.layers_merged)?;
            d.set_item("cleanup", cd)?;
        },
        None => d.set_item("cleanup", py.None())?,
    }
    match &phase.inflate_result {
        Some(r) => {
            let id = PyDict::new(py);
            id.set_item("merit_before", r.merit_before)?;
            id.set_item("merit_after", r.merit_after)?;
            id.set_item("total_thickness_before", r.total_thickness_before)?;
            id.set_item("total_thickness_after", r.total_thickness_after)?;
            id.set_item("layer_count", r.layer_count)?;
            id.set_item("addon_qwot", r.addon_qwot)?;
            id.set_item("reference_wavelength", r.reference_wavelength)?;
            d.set_item("inflate", id)?;
        },
        None => d.set_item("inflate", py.None())?,
    }
    Ok(d)
}

#[pyclass(name = "NeedlePipeline")]
/// Continuous iterative needle synthesis (see `NeedlePipeline` in the
/// core). Construct from a `DesignStack` + `MeritSpec` + grid; the needle
/// demands fold internally (`SpectralInputs::from_spec`, conservative
/// form — all quantities: R/T/A, back siblings, per-channel phase).
/// `contrast` maps host material → seed `LayerSpec` template (its nk on
/// the grid; thickness/flags ignored — the seed is built fresh).
pub struct PyNeedlePipeline {
    inner: NeedlePipeline,
    spec: navette::smatrix::synthesis::merit::MeritSpec,
    sin_theta: Vec<f64>,
    wavls: Vec<f64>,
    clamp_min: f64,
    clamp_max: f64,
    lm: LmConfig,
}

#[pymethods]
impl PyNeedlePipeline {
    #[new]
    #[pyo3(signature = (stack, spec, angles_deg, wavelengths, contrast,
                        pipeline_config=None, needle_config=None, lm=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        stack: &PyDesignStack,
        spec: &PyMeritSpec,
        angles_deg: PyReadonlyArray1<'_, f64>,
        wavelengths: PyReadonlyArray1<'_, f64>,
        contrast: HashMap<String, Py<PyLayerSpec>>,
        pipeline_config: Option<Py<PyPipelineConfig>>,
        needle_config: Option<Py<PyNeedleCycleConfig>>,
        lm: Option<Py<PyLmConfig>>,
    ) -> PyResult<Self> {
        let a = angles_deg.as_slice()?;
        let w = wavelengths.as_slice()?;
        if a.is_empty() || w.is_empty() {
            return Err(PyValueError::new_err("angles/wavelengths must be non-empty"));
        }
        let nw = w.len();
        if stack.inner.num_wavs() != nw {
            return Err(PyValueError::new_err(format!(
                "stack grid {} != {} wavelengths",
                stack.inner.num_wavs(),
                nw
            )));
        }
        let cfg = pipeline_config
            .map(|c| c.bind(py).borrow().inner.clone())
            .unwrap_or_default();
        let needle_cfg = needle_config
            .map(|c| c.bind(py).borrow().inner.clone())
            .unwrap_or_default();
        let lm_cfg = lm
            .map(|l| l.bind(py).borrow().inner.clone())
            .unwrap_or_default();
        let mut cmap = ContrastMap::new();
        for (mat, tmpl) in &contrast {
            let t = tmpl.bind(py).borrow().inner.clone();
            if t.nk.len() != nw {
                return Err(PyValueError::new_err(format!(
                    "contrast '{mat}': nk length {} != {} wavelengths",
                    t.nk.len(),
                    nw
                )));
            }
            cmap.insert(Arc::from(mat.as_str()), t);
        }
        let spectral = SpectralInputs::from_spec(spec.inner(), a, w)
            .map_err(PyValueError::new_err)?;
        let inner = NeedlePipeline::new(
            stack.inner.clone(),
            spectral,
            cfg.clone().validated().map_err(PyValueError::new_err)?,
            needle_cfg,
            cmap,
        )
        .map_err(PyValueError::new_err)?;
        Ok(PyNeedlePipeline {
            inner,
            spec: spec.inner().clone(),
            sin_theta: deg_to_sin(a),
            wavls: w.to_vec(),
            clamp_min: cfg.clamp_min_nm,
            clamp_max: cfg.clamp_max_nm,
            lm: lm_cfg,
        })
    }

    /// Execute the macro-loop. `callback(macro_cycle, phase_dict)` runs
    /// after each cycle; raising inside it aborts as `USER_ABORT`.
    /// Returns a dict: `termination`, `final_mf`, `final_layer_count`,
    /// `final_total_thickness_nm`, `stagnation_detail`, `phases`, and the
    /// final `stack` (`DesignStack`).
    #[pyo3(signature = (callback=None))]
    fn run(&mut self, py: Python<'_>, callback: Option<Py<PyAny>>) -> PyResult<Py<PyDict>> {
        let mut ctx = SmatrixContext {
            wavls: self.wavls.clone(),
            sin_theta: self.sin_theta.clone(),
            spec: self.spec.clone(),
            clamp_min_nm: self.clamp_min,
            clamp_max_nm: self.clamp_max,
            lm: self.lm.clone(),
        };
        let res = py
            .detach({
                let inner = &mut self.inner;
                move || {
                    inner.run(&mut ctx, |cycle, phase, _det| match &callback {
                        None => Ok(()),
                        Some(cb) => Python::attach(|py| {
                            let d = phase_dict(py, phase).map_err(|e| e.to_string())?;
                            cb.call1(py, (cycle, d))
                                .map(|_| ())
                                .map_err(|e| format!("needle callback failed: {e}"))
                        }),
                    })
                }
            })
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("termination", res.termination.name())?;
        d.set_item("final_mf", res.final_mf)?;
        d.set_item("final_layer_count", res.final_layer_count)?;
        d.set_item("final_total_thickness_nm", res.final_total_thickness_nm)?;
        d.set_item("stagnation_detail", res.stagnation_detail)?;
        let mut phases = Vec::with_capacity(res.phases.len());
        for p in &res.phases {
            phases.push(phase_dict(py, p)?.unbind());
        }
        d.set_item("phases", phases)?;
        d.set_item(
            "stack",
            Py::new(py, PyDesignStack::from_inner(self.inner.stack.clone()))?,
        )?;
        Ok(d.unbind())
    }

    fn __repr__(&self) -> String {
        format!("NeedlePipeline({})", PyDesignStack::from_inner(self.inner.stack.clone()).__repr__())
    }
}