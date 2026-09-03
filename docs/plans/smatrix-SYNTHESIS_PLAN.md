# Navette Synthesis — Refactor & Implementation Plan

**Goal:** Port the Python `needle_pipeline.py` / `needle_synthesis.py`
machinery into Rust inside `navette_smatrix`, replacing the brute-force
P-function scan with the analytic `needle_operator` engine, and keeping
`navette_spectralweave` as the untouched source of truth for targets.

Status: **plan** — nothing implemented yet.
Companion session work already merged: `optics_core.rs` extraction,
module renames (`core_engine.rs`, `needle_operator.rs`, `needle_engine.rs`),
Python `navette.smatrix` + `navette.needle`.

---

## 1. Current state of the codebase

### 1.1 `navette_smatrix` (Rust, pyo3 ext `_smatrix`)

| Module | Role |
|---|---|
| `optics_core.rs` | Pure shared primitives: Redheffer stars (field/intensity/cross), roughness `w_function_inner`, fast complex kernels, `grad_nonuniform`, constants |
| `roughness.rs`, `redheffer_field.rs`, `redheffer_intensity.rs` | Thin pyo3 wrappers |
| `coherent_block.rs` | s/p/dual coherent block solvers |
| `core_engine.rs` | Request-driven unified engine (`REQ_*` bitmask) |
| `optimizer.rs` | Landscape scan, Nelder-Mead, field profile |
| `needle_operator.rs` | Analytic needle sensitivities (pure Rust): `p_function*`, `phase_dispersion_sensitivity`, cascade machinery |
| `needle_engine.rs` | Rayon/pyo3 request API over the operator (`NREQ_*`) |

Python package `navette/`: `smatrix.py` (ScatterMatrix), `needle.py`
(`NeedleRequest`, `needle_gradient`). Validated: kernel ≡ grid drivers,
P vs `core_engine` FD cross-check ≤ 0.013 %.

### 1.2 `navette_spectralweave` (Rust, pyo3 ext `spectralweave`)

- `OpticalWeaver` / `SpectralDataFrame` / `OpticalKey` — heterogeneous
  spectral data management (frames, unit conversion, fragment weaving,
  distribution cache).
- `TargetWeaver` — targets with **normalization applied at ingestion**
  (`register_metadata`): stores final `normalized_targets`, `norm_factor`,
  `resolved_mode` (Linear/Log/Phase/Complex), floored tolerances,
  `TargetKind` (Exact/Above/Below).
- `calculate_merit(sim_weaver, target_weaver)` — zero-alloc residual kernel
  with aligned-grid fast path and monotone interpolation.

### 1.3 Python legacy being ported

- `needle_synthesis.py`: `NeedleSynthesizer` — thickness optimization →
  P-function scan (**brute force**: insert 1 nm test needle → full solve →
  merit, per scan point) → insertion → re-optimize loop; QWOT helpers;
  cleanup (merge/prune/re-opt); impact-ranked QWOT inflation; QWOT rounding.
- `needle_pipeline.py`: `NeedlePipeline` macro-loop (needle pass → cleanup →
  inflate → stagnation check), `PipelineConfig` budgets, `TerminationReason`,
  `StagnationDetector` (plateau / oscillation / divergence),
  `ClampedNeedleSynthesizer` (hard thickness bounds).

---

## 2. Key design decisions

### D1 — Synthesis lives in `navette_smatrix` (Option A)
New modules under `src/synthesis/`, exposed through the existing `_smatrix`
extension. No cross-crate linkage issues; extraction into its own crate is
trivial later because the modules are self-contained behind clean traits.
*(Rejected Option B: separate `navette_synthesis` crate depending on
smatrix-as-rlib — pyo3 extension-linkage subtleties for no near-term gain.)*

### D2 — MeritSpec bridge instead of OpticalWeaver in the loop
`OpticalWeaver`'s frame/cache machinery is an impedance mismatch for the
synthesis inner loop:

1. **Key model**: `OpticalKey = (f64 wavelength-slot, Arc<str>, Arc<str>)` —
   point/hash oriented, not grid oriented; every access pays hashing +
   string clones.
2. **Frame lifecycle**: solver curves change every evaluation;
   `create_dedicated_frame` permanently appends to the frames vec, reuse
   means write-locking the same `RwLock<AHashMap>` and bumping the
   generation each eval; `get_weaved` allocates fresh `Vec`s per key.
3. **Parallelism**: LM Jacobian wants rayon across perturbed stacks; the
   weaver is lock-mediated mutable shared state. A flat spec is read-only
   `Send+Sync`.
4. **Unneeded generality**: distribution cache / fragment weaving / unit
   conversion exist for heterogeneously gridded interactive data; the
   synthesis sim grid is fixed for the whole run.

→ The loop consumes a flat **`MeritSpec`**, converted **once per pipeline
run** on the Python side from `TargetWeaver` (reads the *already normalized*
entries). Spectralweave stays untouched.

### D3 — Normalization is inherited, not re-implemented
Normalization happens at target ingestion in spectralweave. The converter
copies finished values. Algebraic folding for Linear mode:

```
residual = (nf·(sim − target)/tol)² ≡ ((sim − target)/(tol/nf))²
```

so linear-mode targets collapse to `(target_original, eff_tol = tol/nf,
kind)` triples — no mode field needed at eval time. Non-linear modes
(Log/Phase/Complex) carry a small transform enum applied to the *sim* side
only (~15 lines, lifted verbatim from `calculate_merit`).

### D4 — Analytic needle pass replaces the brute-force scan
One `needle_engine` sweep (`NREQ_P`) yields P(z) at all candidate depths in
a single rayon pass. Insertion selection flips from "argmin MF-after-insert"
to "most negative P(z)" (equivalent criterion; convergence thresholds must
be recalibrated from MF units to gradient units). No test-thickness bias.

### D5 — Thickness optimizer: bounded Levenberg–Marquardt in Rust
Replaces scipy `least_squares(method="trf")`. Jacobian by central finite
differences on layer thicknesses — `2·n_layers` solves per Jacobian,
rayon-parallelized across columns. Bounds handled by step projection +
post-step clamping (mirrors `ClampedNeedleSynthesizer` semantics:
below-min layers are *removed*, above-max hard-capped).
*Future upgrade:* analytic thickness Jacobian via the existing dual-number
cascade (∂/∂dⱼ is a propagation-phase derivative) — deferred.

---

## 3. Target module layout

```
navette_smatrix/src/synthesis/
├── mod.rs            module root, re-exports
├── config.rs         PipelineConfig, TerminationReason          (port)
├── stagnation.rs     StagnationDetector                       (port)
├── structure.rs      DesignStack + LayerSpec: split / insert /
│                     merge_adjacent / remove / clamp ops      (port)
├── merit.rs          MeritSpec + residual evaluator            (bridge)
├── thick_opt.rs      bounded LM over film thicknesses          (new)
├── needle_pass.rs    analytic P(z) sweep → insertion decision  (upgrade)
├── cleanup.rs        merge → prune → re-opt                    (port)
├── inflate.rs        QWOT helpers, impact-ranked inflate       (port)
└── pipeline.rs       NeedlePipeline::run + result types        (port)
```

Python side: `navette/synthesis.py` — `NeedlePipeline` facade,
`TargetWeaver → MeritSpec` converter, result dataclasses.

---

## 4. Core data structures (sketches)

```rust
// structure.rs
pub struct LayerSpec {
    pub material: Arc<str>,
    pub n: Arc<[Complex64]>,      // per simulation wavelength
    pub d_nm: f64,
    pub optimize: bool,
    pub needle: bool,             // admissible host
}
pub struct DesignStack {
    ambient: LayerSpec, substrate: LayerSpec,
    films: Vec<LayerSpec>,
    // cached solver arrays, invalidated on mutation
}

// merit.rs
pub enum ConstraintKind { Exact, Above, Below }   // TargetKind
pub enum SimTransform { Identity, Log10 { nf: f64 }, PhaseWrap }  // non-linear only
pub struct MeritPoint {
    pub wl_idx: u32,        // index into the fixed simulation λ-grid
    pub angle_idx: u32,
    pub pol: Pol,
    pub channel: Channel,   // R | T
    pub kind: ConstraintKind,
    pub target: f64,        // original units (linear-folded)
    pub eff_tol: f64,       // tol/norm_factor
    pub transform: Option<SimTransform>,
}
pub struct MeritSpec { points: Arc<[MeritPoint]> }   // Send+Sync

impl MeritSpec {
    /// residuals against core_engine output arrays ([n_angles, n_wavs] per key)
    pub fn residuals(&self, sim: &SimResult, out: &mut Vec<f64>);
    pub fn merit(&self, sim: &SimResult) -> f64;
}

// config.rs — direct port of PipelineConfig budgets/knobs
pub struct PipelineConfig { /* max_film_layers, clamp_min/max_nm,
    needles_per_cycle, enable_cleanup/inflate, inflate_addon_qwot,
    stagnation_*, merit_target, max_macro_cycles, … */ }
pub enum TerminationReason { LayerBudget, ThicknessBudget, StagnationPlateau,
    StagnationOscillation, StagnationDivergence, MaxIterations,
    MeritTarget, UserAbort }

// pipeline.rs
pub struct PhaseResult { macro_cycle, mf_needle, mf_cleanup, mf_inflate,
    mf_end, layer_count, total_thickness_nm, insertion: Option<Insertion> }
pub struct PipelineResult { phases, termination, final_mf, … }

// needle_pass.rs
pub struct Insertion { pub film_idx: usize, pub xi_nm: f64,
    pub material: Arc<str>, pub p_value: f64, pub predicted_df: f64 }
```

---

## 5. Implementation phases

Each phase ends green (build + tests) before the next starts.

### Phase 0 — plumbing ✅ DONE
- Created `src/synthesis/mod.rs`; registered `pub mod synthesis;` in lib.rs.
- Pure Rust, no pyo3 — scratch-testable like `needle_operator`.
- Fix conventions: element order, Im(β) ≥ 0, LOG_MIN regularization — all
  inherited from `optics_core`; document that `synthesis` may not re-derive
  optics math.

### Phase 1 — `structure.rs` (DesignStack) ✅ DONE (11/11 unit tests)
Ported with Python-parity semantics:
- `LayerSpec` (material Arc<str>, nk per-wav Complex64, d_nm, coherent,
  rough_type/val, optimize, needle), `DesignStack` = ambient + films +
  substrate with boundary invariants enforced by construction.
- `insert_needle_seed` (verbatim `_insert_needle`: [top, seed, bot], host
  flags preserved, degenerate zero-thickness portions permitted like Python).
- `merge_adjacent` (sum thicknesses, FIRST layer's properties win),
  `remove_film`, `clamp_all` (remove < min, cap > max → (removed, capped)),
  `set_thickness` for the optimizer.
- `solver_arrays()` materializes the exact smatrix layouts: n_stack_cache
  wav-major re/im interleave (base = w·n_layers·2 + slot·2), per-layer
  thicknesses/incoherent_flags/rough arrays — layout pinned by test against
  navette/smatrix.py conventions.
**Validated:** unit tests mirror Python behaviors incl. edge cases
(boundary-value clamps use strict < and >, merge chains, flag inheritance).

### Phase 2 — `merit.rs` (MeritSpec + evaluator) ✅ DONE (12/12 unit tests)
- `SimCurves`: row-major [n_angles, n_wav] Arc slices indexed by
  `CurveId` (Rs/Rp/Ru/Ts/Tp/Tu — mirrors `_RESULT_KEY_MAP`).
- `MeritSpec { keys: [MeritKey{angle, curve}], targets: [MeritTarget] }`:
  flat immutable Send+Sync; residual vector has FIXED length (inactive
  Above/Below points contribute zeros) so LM Jacobian dimensions are stable.
- Residual kernel lifted verbatim from `calculate_merit`: overlap skip,
  bit-exact aligned fast path, two-pointer monotone interpolation with edge
  clamping, Linear/Log/Phase transforms, kind activation. Missing curve →
  penalty ONCE per key group (`merit()`), or `Err(CurveId)` (`residuals()`).
- Normalization inherited (D3): converter copies finished normalized values.
**Validated:** hand-computed parity for every branch incl. phase wrap
(π+0.1 → −(π−0.1)), log-mode ×10 raw error, argmin angle-row selection,
aligned-vs-interp agreement.
**Deferred to Phase 7:** cross-check vs live `spectralweave.calculate_merit`
through Python (separate extension; needs built `_spectralweave` ext).

### Phase 3 — `thick_opt.rs` (bounded LM) ✅ DONE (10/10 unit tests + scipy parity)
- Classic Marquardt scaling `(JᵀJ + λ·diag(JᵀJ))δ = −Jᵀr`, diag floored;
  Cholesky solve w/ Gaussian-elimination partial-pivot fallback.
- Central-difference Jacobian, h_j = ∛ε·max(|x_j|, 1), **rayon across
  columns** (one TMM solve per eval in production — columns dominate).
- Bounds: component-wise veto of steps pushing past an ACTIVE bound,
  then trial clamp; sub-min removal stays the CALLER's job (Python
  contract). Out-of-bounds x0 rejected loudly.
- Terminations: Gradient/Step/Cost/MaxIterations/Stalled.
- ⚠ CONVENTION: `LmResult.cost = Σr²` but scipy `.cost = ½Σr²` —
  multiply by 2 when comparing trajectories with Python.
**Validated:** 10 unit tests (linear/exponential/boundary/corner/mixed
bounds/error propagation/max-iter); scipy `least_squares(trf)` parity on
the same four canonical problems (param agreement ≤ 6e-12, costs equal up
to the ½ factor). Full BBAR solver-level parity moves to Phase 4 gate.

### Phase 4 — `needle_pass.rs` (analytic insertion) ✅ DONE (10/10 unit tests)
- `build_scan_sites`: verbatim `compute_p_function` grid — interior
  k·step per admissible film, non-admissible films advance cumulative
  depth only.
- `needle_pass_scan`: rayon over angle-major spectral points; fields built
  once per polarization via `build_stack_fields_range`, P accumulated over
  points AND branches in fixed k order (bit-reproducible).
- `NeedlePassResult::best()`: most-negative-P site with predicted ΔF;
  None when no site is negative.
- `build_needle_targets`: MeritSpec → flat (raw target, folded weight)
  angle-major arrays. EXACT overlap fold via quadratic-form identity
  Σwₑ(R−tₑ) = W(R−t_eff); weights w = nf²/tol² so descent matches dF/dδ
  of the spectralweave merit. Above/Below activation masked against the
  current sim. R-channel + linear-normalization only (T/log rejected —
  documented limitation until a transmission kernel lands).
**Validated:** profile ≡ FD-validated `needle_operator::p_function`
oracle (< 1e-12 rel), dual-pol = s+p sum, site selection invariants,
fold exactness incl. overlapping entries, activation masking, rejection
paths.
**Deferred to Phase 7:** first-insertion parity vs live Python
`compute_p_function` on BBAR (needs the loom env).

### Phase 5 — `cleanup.rs` + `inflate.rs` ✅ DONE (13/13 unit tests)
- NEW `context.rs`: `DesignContext` trait (evaluate_merit /
  optimize_thicknesses) — the seam between synthesis algorithms and the
  numeric machinery. Algorithms are mock-testable now; Phase 6 wires the
  real evaluator (core_engine + MeritSpec + thick_opt LM).
- `cleanup.rs`: verbatim `remove_thin_layers` (trial-remove each thin
  candidate on a clone, lowest-MF removal wins, re-opt between iterations,
  budget-capped) + `cleanup_design` (merge → prune → merge → optional
  final re-opt). CleanupResult mirrors Python field-for-field.
- `inflate.rs`: QWOT helpers (λ₀/(4·n), real part at nearest grid point,
  non-positive n rejected); `inflate_design` with top-k trial-inflation
  ranking (stable ascending sort, ties keep film order — Python semantics);
  negative addon clamps at zero; `round_to_qwot` uses a **banker's-rounding
  helper** because Python's round() is half-to-even and Rust's f64::round is
  half-away-from-zero — exact-half QWOT ratios would otherwise diverge from
  the oracle.
**Validated:** 13 tests incl. least-impact ordering (hand-computed trial
MFs), budget cap with non-optimizable films, merge-count accounting,
top-k selection arithmetic, clamp-at-zero, banker's rounding at exact
halves (2.5→2, 3.5→4), min-one-step snapping.

### Phase 6 — config / stagnation / evaluator / cycle / pipeline ✅ DONE (23/23 tests)
- `config.rs`: PipelineConfig + TerminationReason, verbatim defaults;
  `validated()` applies __post_init__ semantics.
- `stagnation.rs`: StagnationDetector port. ⚠ TWO Python-semantics traps
  found and pinned by tests: (1) Rust `f64::signum(0.0) == 1.0` unlike
  numpy.sign → explicit three-way sign map for delta filtering;
  (2) consecutive-increases uses >= so FLAT/slowly-rising trajectories hit
  DIVERGENCE before plateau ever runs — faithful behavior, documented.
- `evaluator.rs`: SmatrixContext — the REAL DesignContext. simulate() =
  rayon dual-pol coherent solves over the angle×λ grid assembling SimCurves
  (Rs/Rp/Ts/Tp); optimize_thicknesses = LM over optimize-flagged films,
  bounds [0, clamp_max], post clamp_all sweep (removes sub-min).
  **Physics anchors green:** R+T=1 conservation; exact quarter-wave AR
  R=0 at d=λ/(4√n₀ns); LM recovers quarter-wave from bad start < 1 nm.
- `cycle.rs`: needle pass sub-loop (initial optimize → scan/select/insert/
  optimize ×N). Convergence metric recalibrated: predicted_improvement =
  −P_best·δ_seed vs threshold (Python measured −P·1 nm implicitly).
  Per-distinct-contrast-material sweeps, global best across materials.
- `pipeline.rs`: NeedlePipeline macro-loop, verbatim order: pre-flight
  budgets → needle pass → post-needle budget → cleanup(+clamp) → inflate
  (clamp semantics of the Clamped overrides) → stagnation record/check →
  post-cycle budget; final optimize+clamp+evaluate always runs; callback
  Err ≡ KeyboardInterrupt → UserAbort (overrides other reasons, final
  optimization still runs).
**Validated:** termination paths (MaxIterations/LayerBudget/
ThicknessBudget/MeritTarget pre-flight, UserAbort, Plateau), budget
ordering (pre-flight phases empty vs post-needle phase recorded), final
clamp enforcement.

### Phase 7 — Python integration & benchmarks
- `navette/synthesis.py`: `NeedlePipeline.from_targetweaver(tw, config)`,
  converter, docs; export from `navette/__init__`.
- Benchmark report: wall-clock per macro-cycle vs `needle_pipeline.py`;
  expected order-of-magnitude improvement on the P-scan-dominated path.
- Keep `needle_synthesis.py`/`needle_pipeline.py` frozen as numerical oracle.

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| LM behaves differently from scipy trf near bounds | Phase 3 parity gate before anything builds on it; fallback: keep projected-Gauss-Newton with line search |
| P-sign convention / weight mapping wrong → wrong insertion sites | weights derive from MeritSpec (`w ∝ 1/tol²` folded); Phase 4 parity test catches it |
| Angle-major layout mistakes (k = a·num_wavs + w) | reuse `needle_gradient` Python wrapper conventions; assert shapes in converter tests |
| Incoherent stacks: needles inside flagged spacers | DesignStack marks spacer films non-admissible hosts up front (engine rejects anyway) |
| Dispersion channels later need GDD targets | MeritSpec carries channel enum already; extend with GD/GDD target kinds when VLC-GA work begins |

## 7. Session lessons applied

- Gradient rows are Jacobian *columns* — validated by identity-first-block
  masking; keep the half-gradient convention documented at every surface.
- τ̂ formula `iβ′(1+r₁₂²)/(1−r₁₂²)` is exact — do not "fix".
- FD tolerances: 2e-3 rel for dr-type oracles; cascade 1e-5 @ h=1e-6;
  end-to-end GD/GDD 5–8 % vs O(δ)-linearized references.
