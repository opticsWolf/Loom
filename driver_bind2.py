import io
p = 'rust/navette-py/src/synthesis_pipeline.rs'
s = io.open(p, encoding='utf-8').read()
# cut everything from the bad helper's run_design fn through result_report fn
start = s.find('/// One film for `run_design`')
assert start != -1
end_marker = '''    d.set_item("phases", phases)?;
    Ok(d.unbind())
}
'''
end = s.find(end_marker, start) + len(end_marker)
print('CUT', end - start)
block = '''/// One film for `run_design`: evaluated nk + authoring flags.
#[derive(FromPyObject)]
struct PyFilmInput<'a> {
    name: String,
    nk: numpy::PyReadonlyArray1<'a, Complex64>,
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
/// Returns the full report dict including the final `stack`.
#[pyfunction]
#[pyo3(signature = (ambient_nk, ambient_name, substrate_nk, substrate_name, films, groups, seeds, wavelengths, angles_deg, spec, pipeline_config=None, needle_config=None, lm=None, callback=None))]
#[allow(clippy::too_many_arguments)]
fn run_design(
    py: Python<'_>,
    ambient_nk: Vec<Complex64>,
    ambient_name: String,
    substrate_nk: Vec<Complex64>,
    substrate_name: String,
    films: Vec<PyFilmInput<'_>>,
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
        .iter()
        .map(|f| {
            ArrayFilm {
                name: f.name.clone(),
                nk: f.nk.as_slice().unwrap_or(&[]).to_vec(),
                d_nm: f.d_nm,
                coherent: f.coherent,
                roughness: f.roughness,
                rough_type: f.rough_type,
                inhomogen: f.inhomogen,
                inh_delta: f.inh_delta,
                interface: f.interface,
                interface_thickness: f.interface_thickness,
                optimize: f.optimize,
                needle: f.needle,
            }
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
    let (res, stack) = py
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
    result_to_dict(py, &res, PyDesignStack::from_inner(stack))
}
'''
s = s[:start] + block + s[end:]
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('REPLACED')
