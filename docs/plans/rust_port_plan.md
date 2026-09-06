# Rust-first completion plan — Navette drivable from Rust alone

**Target:** every feature runnable with zero Python. Rust owns all logic
**and all validation**; Python (PyO3 types + YAML→dict parsing) is a thin
addon. Conversely, **every** public Rust function is exposed to Python,
so Python can still drive the whole engine.

**Status: COMPLETE 2026-09-06.** All phases shipped, each verified
(bitwise twins), version-bumped, committed and pushed:
R1 solver/needle/eigen/orchestration (0.4.2–0.4.6), R2 providers
(0.4.7–0.4.10), R3 synthesis drivers (0.4.11–0.4.14), R4 config (0.4.15–
0.4.17), R5 audit (0.4.18–0.4.19). Pydantic deleted, no dual
maintenance remains. Exposure record: docs/plans/exposure_audit.md.

## 0. Binding principles (locked)

1. **Single source of truth is Rust, including validation.** Schemas
   are defined once as `serde` structs; pydantic is deleted, not
   mirrored. Python config classes are thin PyO3 types validated
   natively (`serde_path_to_error` for field-path errors). YAML→dict
   parsing may stay Python-side (format handling, no schema
   knowledge); dict→validated-object is always Rust. Kernels are
   defined once in the core; Python holds no math and no schema.
2. **Errors:** `Result<_, String>` → `PyValueError`. No `KeyError`/typed
   mapping — messages carry the failing key/value.
3. **Warnings:** returned as `Vec<String>`, re-emitted via the
   `emit_warnings` pattern (`stacklevel=3`). Nothing silent, nothing
   refused that the core can represent (homogenize-with-warning rule).
4. **Twin parity at every cutover:** old Python path vs new native path,
   bitwise (`array_equal`, `.hex()` floats — never numpy `repr`).
   Cutover deletes the Python logic the same commit.
5. **File I/O in Rust is JSON-only** (machine interchange, per format
   policy). YAML stays Python-side authoring; YAML→JSON conversion is
   external to the core.
6. **Verification per item:** standalone Rust tests (no interpreter) +
   wrapper parity tests + `test-pure` + full `validation` green.
7. **No pre-validation in Python.** Wrappers assemble plain dicts/kwargs
   (or parse YAML text into dicts) and hand them over unvalidated; the
   PyO3 types validate natively exactly once. Pydantic-before-serde on
   the same schema is double bookkeeping — refused.

## R1. Native solver (everything composes on this)

**Now:** `smatrix/smatrix.py::ScatterMatrix` (~590 lines) normalizes
inputs, lays out the flat Re/Im cache, dispatches `core_engine`, and
computes all derived views. Rust has primitives only.

**Build:** `navette::smatrix::solver::Solver`
- `Solver::new(wavelengths, angles_deg|rad, indices, thicknesses,
  incoherent_flags, roughness_types, roughness_values, coherence_mode)`
  with the Python validation rules (non-empty, shapes, ≥2 layers).
- `solve(request_mask) -> BTreeMap<String, Vec<f64>>` (same keys).
- Derived views as methods: `reflectance_transmittance(pol)`,
  `ellipsometry()`, `absorption()`, `complex_amplitudes()`, `stokes()`,
  `energy_conservation()`, `dispersion()` — ported formulas, same outputs.
- Eigen drivers: `landscape()`, `refine_mode()`, `find_eigenmodes()`,
  `field_profile()` over the existing native primitives.
- Single-angle squeeze: return 1-D — represent as
  `Solution { angles: usize, maps }` and let the binding squeeze.

**Bind:** `PySolver` (name `ScatterMatrix`? keep Python names stable —
bindings keep old class/function names; only internals move).
`needle_gradient` inputs (`NeedleRequest` flags, target/weight
normalization, z-grid, host mask) become
`solver::needle_inputs(...) -> NeedleGradient` or a
`Solver::needle_gradient(...)` method over native `needle_engine`.

**Then:** `solve_structure` becomes a 10-line wrapper (validate →
inputs → `Solver` → dict); later a native
`structure::solve(source, grid)` when R4 lands.

**Tests:** HEX oracles for R/T + ellipsometry on a 2-layer stack;
twin parity vs current `ScatterMatrix` on all views; `needle_gradient`
parity on targets/weights variants.

## R2. Providers (kill the live dual maintenance)

**Now:** `structure/materials.py` reimplements `get_nk`/grid logic in
numpy (`Dict`/`Object`/`Weaver` providers + `wrap_material_source`);
native `DictProvider`/`WeaverProvider` exist but Python never delegates.

**Build:**
- Audit native providers against the Python semantics (refresh
  atomicity, strict/lenient grids, table resampling path) — port any
  missing behavior into `navette::structure::providers`.
- Native `wrap_source`: enum over dict-of-arrays / specs / weaver-table
  inputs → `Box<dyn MaterialProvider>` for standalone consumers.
- Python classes become shells holding the native object (same import
  names, same methods); `wrap_material_source` builds native.

**Tests:** provider grid-refusal matrix in Rust; twin `get_nk`
bitwise on dict/object/weaver sources.

## R3. Synthesis drivers end-to-end

**Now:** `run_needle`, `stack_from_layers`, `layer_from_material`,
target→spec compilation, `sim_curves_from_arrays`,
`apply_reference_rotation` are Python.

**Build** (`smatrix::synthesis::driver`):
- `run_design(request: DesignRequest, targets: TargetSet, angles,
  grid, contrast, pipe_cfg) -> RunReport` composing `build_design` +
  `NeedlePipeline::run` — fully standalone runs.
- Native `TargetSet` (serde mirror of `TargetCollection`) →
  `MeritSpec`/`SimCurves` compilation (ports `build_merit_spec`,
  `sim_curves_from_arrays`, `_curve_id`).
- Port `apply_reference_rotation` as a kernel (it guards a 1e-12
  native/numpy agreement — becomes a Rust unit test).
- `stack_from_layers` thins onto `build_design` (or deletes in favor
  of `design_from_config`; decide at port time — one path survives).

**Bind:** `PyNeedlePipeline.run` stays; add `run_needle(request_json,
…)` module function; Python `run_needle` thins to validate → dump →
call.

**Tests:** standalone run on a 2-film AR design (termination +
merit improvement assertions); twin parity vs current `run_needle`.

## R4. Config persistence (file → engine, no Python)

**Now:** `config/io.py`, `loader.py`, `program.py`, `models.py`,
remaining `builders.py` — 100% Python.

**Build** (`navette::config`, new module):
- Serde mirrors: material library, groups, layers, named structures,
  architect blocks, program envelope (`PROGRAM_SCHEMA_VERSION` gate —
  **one** canonical gate; Python `_gate` delegates).
- `load_program_json(path) -> Program`; `program.structure(name) ->
  DesignRequest` bridging into R3's driver; `architect(name) ->
  native Architect`.
- `structure_from_config`/`architect_from_config` move here as native
  fns (Python thins to wrappers).
- Pydantic models are deleted. Each config class becomes a PyO3 type
  whose constructor validates via serde (kwargs/dict/JSON in,
  `ValueError` with field path out). YAML files parse to dicts
  Python-side, then validate natively — one schema, one validator.

**Tests:** `example_program.yaml` → JSON → native load → expanded
stack, compared against the Python path bitwise.

## R5. Cutover audit + surface completion

- Every `pub fn` in `navette` gets a PyO3 exposure (new lint step:
  `cargo doc` public API vs binding registry diffed in CI).
- Delete residual Python logic; `navette/*.py` may contain only:
  YAML→dict parsing, docstrings, re-exports, and warning plumbing.
  Pydantic dependency removed from `pyproject.toml`.
- `check_schema_version` deduplicated (native wins; Python delegates).
- Won't-ports recorded: `apply_to_all_layers(func)` (callable-crossing),
  cosmetic `__repr__`s.
- Release: minor bump, single-crate publish unchanged, README
  architecture section updated ("Rust core, Python addon").

## Order of work

R1 → R2 → R3 → R4 → R5. R1 first (all else composes on the solver);
R2 second (live dual maintenance is the only active correctness risk);
R3–R4 in dependency order; R5 closes.
