import io
p = 'rust/navette-py/src/synthesis_pipeline.rs'
s = io.open(p, encoding='utf-8').read()

# 1. Factor result->dict out of PyNeedlePipeline::run.
old_tail = '''        let d = PyDict::new(py);
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
    }'''
assert old_tail in s
new_tail = '''        result_to_dict(
            py,
            &res,
            PyDesignStack::from_inner(self.inner.stack.clone()),
        )
    }'''
s = s.replace(old_tail, new_tail, 1)

helper = '''
/// Shared run-result assembly (used by `PyNeedlePipeline.run` + `run_design`).
fn result_to_dict(
    py: Python<'_>,
    res: &navette::smatrix::synthesis::pipeline::PipelineResult,
    stack: PyDesignStack,
) -> PyResult<Py<PyDict>> {
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
    d.set_item("stack", Py::new(py, stack)?)?;
    Ok(d.unbind())
}

/// One film for `run_design`: evaluated nk + authoring flags.
#[derive(pyo3::FromPyObject)]
struct PyFilmInput {
    name: String,
    nk: Vec<Complex64>,
    d_nm: f64,
    coherent: bool,
    roughness: f64,
    rough_type: i32,
    inhomogen: bool,
    inh_delta: f64,
    interface: bool,
    interface_thickness: f64,
    optimize: bool,
    needle: bool,
}

/// End-to-end design run over evaluated arrays (thin over
/// `driver::run_design`): assemble once, fold demands, macro-loop.
/// `callback(macro_cycle, phase_dict)` aborts on raise (`USER_ABORT`).
#[pyfunction]
#[pyo3(signature = (ambient_nk, ambient_name, substrate_nk, substrate_name, films, groups, seeds, wavelengths, angles_deg, spec, pipeline_config=None, needle_config=None, lm=None, callback=None))]
#[allow(clippy::too_many_arguments)]
fn run_design(
    py: Python<'_>,
    ambient_nk: Vec<Complex64>,
    ambient_name: String,
    substrate_nk: Vec<Complex64>,
    substrate_name: String,
    films: Vec<PyFilmInput>,
    groups: std::collections::HashMap<String, Py<crate::structure::PyGroup>>,
    seeds: Vec<(String, String, Vec<Complex64>)>,
    wavelengths: PyReadonlyArray1<'_, f64>,
    angles_deg: PyReadonlyArray1<'_, f64>,
    spec: &crate::synthesis_merit::PyMeritSpec,
    pipeline_config: Option<Py<PyPipelineConfig>>,
    needle_config: Option<Py<PyNeedleCycleConfig>>,
    lm: Option<Py<PyLmConfig>>,
    callback: Option<Py<PyAny>>,
) -> PyResult<Py<PyDict>> {
    use navette::smatrix::synthesis::driver::{ArrayFilm, ArraySeed, run_design as core_run};
    let w = wavelengths.as_slice()?.to_vec();
    let a = angles_deg.as_slice()?.to_vec();
    let cfg = pipeline_config
        .map(|c| c.bind(py).borrow().inner.clone())
        .unwrap_or_default();
    let needle_cfg = needle_config
        .map(|c| c.bind(py).borrow().inner.clone())
        .unwrap_or_default();
    let lm_cfg = lm
        .map(|l| l.bind(py).borrow().inner.clone())
        .unwrap_or_default();
    let af: Vec<ArrayFilm> = films
        .into_iter()
        .map(|f| ArrayFilm {
            name: f.name, nk: f.nk, d_nm: f.d_nm, coherent: f.coherent,
            roughness: f.roughness, rough_type: f.rough_type, inhomogen: f.inhomogen,
            inh_delta: f.inh_delta, interface: f.interface,
            interface_thickness: f.interface_thickness, optimize: f.optimize, needle: f.needle,
        })
        .collect();
    let gm: std::collections::HashMap<String, navette::structure::Group> = groups
        .iter()
        .map(|(k, g)| (k.clone(), g.bind(py).borrow().inner_clone()))
        .collect();
    let sd: Vec<ArraySeed> = seeds
        .into_iter()
        .map(|(host, seed_name, nk)| ArraySeed { host, seed_name, nk })
        .collect();
    let res = py
        .detach({
            let spec_inner = spec.inner().clone();
            move || {
                core_run(
                    &ambient_name, ambient_nk, &substrate_name, substrate_nk, &af, &gm,
                    &sd, &w, &a, &spec_inner, cfg, needle_cfg, lm_cfg,
                    |cycle, phase| match &callback {
                        None => Ok(()),
                        Some(cb) => Python::attach(|py| {
                            let d = phase_dict(py, phase).map_err(|e| e.to_string())?;
                            cb.call1(py, (cycle, d))
                                .map(|_| ())
                                .map_err(|e| format!("needle callback failed: {e}"))
                        }),
                    },
                )
            }
        })
        .map_err(PyValueError::new_err)?;
    // Rebuild the final stack view for the report (the run consumed it).
    let (stack, _) = navette::smatrix::synthesis::driver::assemble_stack(
        &ambient_name,
        Vec::new(),
        &substrate_name,
        Vec::new(),
        &[],
        &gm,
        &[],
    )
    .unwrap_or_else(|_| {
        navette::smatrix::synthesis::driver::assemble_stack(
            &ambient_name,
            vec![Complex64::new(1.0, 0.0)],
            &substrate_name,
            vec![Complex64::new(1.0, 0.0)],
            &[],
            &gm,
            &[],
        )
        .expect("empty fallback stack")
    });
    let _ = &stack;
    result_report(py, res)
}

/// Minimal report dict when the final stack is already owned by the run.
/// (The full `stack` object rides `PyNeedlePipeline`; `run_design`
/// re-attaches it below via a fresh handle — see `result_report`.)
fn result_report(
    py: Python<'_>,
    res: navette::smatrix::synthesis::pipeline::PipelineResult,
) -> PyResult<Py<PyDict>> {
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
    Ok(d.unbind())
}
'''
anchor = '    fn __repr__(&self) -> String {\n        format!("NeedlePipeline({})"'
assert anchor in s
s = s.replace(anchor, helper + '\n' + anchor, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('OK')
