# Rust-first implementation plan — structure model in Rust, Python follows

Status: [P]roposed throughout. Workflow per phase: propose → review →
implement → verify. This file is the tracker (log at the bottom).

Direction: the canonical thin-film stack model (`Layer`, `Group`,
expansion, validation, `Structure`, `Architect`, providers) moves to Rust.
PyO3 binds it 1:1; `navette.structure` becomes a thin re-export layer;
the Python implementation is deleted. The solver pipeline
(`NeedlePipeline`, contexts) migrates onto the design model last.

## 0. Non-negotiables

1. **Behavioral oracle first.** The 73 committed probes in
   `validation/regression/structure/` are the acceptance suite: every one
   is ported to a Rust `#[test]` *before or with* its implementation, and
   must pass identically (mirror arrays, masks, bakes, gates, naming,
   versions, fingerprint).
2. **No silent reinterpretation, ever.** Units (nm), enum values, state
   keys, gate semantics — identical across the boundary, pinned by tests.
3. **Forward expansion bit-identity.** Rust forward output must equal
   Python forward output bit-for-bit on reference stacks (parity script,
   plus a randomized Python↔Rust differential test).
4. **Additive-safe states.** `SCHEMA_VERSION = 1` and the fingerprint
   discipline carry over: serde field names match the Python keys exactly
   (`material_name`, `thick_factor`, …); readers refuse foreign versions.
5. **No `unsafe` for the model.** Pure safe Rust; new deps limited to
   `serde`/`serde_json` (states) and `rand`/`rand_distr` (error draws).

## 1. Target architecture

New crate **`navette-structure`** (pure model, no solver dependency):

```
navette-structure  (Layer/Group/enums/providers/expansion/validation/
                    Structure/Architect/serialization)
      ▲                         ▲
      │                         │
navette-smatrix ──────────┘     │  (synthesis: DesignStack migrates here last)
      ▲                         │
      │                         │
navette (umbrella, re-export)   │
      ▲                         │
      │                         │
navette-py (cdylib _navette; new `structure` module binds it all)
      ▲
      │
navette.structure (thin re-export + bridge; Python impl deleted)
```

Dependency additions: `navette-structure` needs `num-complex`,
`serde{,_json}`, `rand{,_distr}` (workspace-pinned). No solver, no Python.

## 2. Canonical types (defined once, in `navette-structure`)

- `RoughnessType`: `#[repr(i32)]` enum `NONE=0, LINEAR=1, STEP=2,
  EXPONENTIAL=3, GAUSSIAN=4, NEVOT_CROCE=5` — discriminants pinned by test
  against the solver's interpretation (solver reads raw ints today).
- `LayerType`: `AMBIENT=0, FILM=1, SUBSTRATE=2`. `BlockKind`: `STACK=0,
  FILMS=1`. `OptMask`, `ErrorMask`, `LayerMask`, `ErrorType`: same values
  as Python (fingerprint their discriminants too).
- Units: nanometres everywhere, no unit type (decision already made;
  documented on the structs).
- `ValidationIssue { severity: Error|Warning, message: String }` —
  replaces the `warning:`-prefix string convention with a typed channel
  (the prefix stays on the Python side for compat, mapped at the boundary).

## 3. Phase A — `navette-structure` core (size: L)

**A1 — `Layer`.** All Python fields with identical names/semantics
(`material: String` = unresolved name — resolution stays provider-side,
as today). `sub_layer_count()` becomes a **derived method** (same formula),
not stored state: strictly purer than Python (no stale counts), bake and
expansion call it. Range checks (`rough_type`, `layer_type`) are
constructors returning `Result` (fail-closed; mirrors Python raises).

**A2 — `Group`.** Factors, summands, masks (`[u8; 7]` + `[u8; 6]`,
binary-checked), error laws + typed `ErrorParams` struct (the fixed
vocabulary, not a string map). `validate()` domains + mask shape.
`_apply_error` transliterated (scalar-draw = systematic offset, pinned by
test). RNG: `rand`/`rand_distr`; expansion takes `seed: Option<u64>`
(`StdRng::seed_from_u64` when given — reproducible; thread RNG otherwise).
HARD TRUTH (re-review): Rust `StdRng` (ChaCha) and NumPy `Generator`
(PCG64) produce DIFFERENT streams, so Monte-Carlo draws can never match
value-for-value across the boundary. Error paths are therefore verified
per-side (distribution shape + same-seed determinism), and
cross-boundary bit-identity differentials run error-FREE. The draw ORDER
(thick, nk, rough, iface, inhg) is still transliterated exactly, so
statistical behavior is comparable.
API change: Python passes `Generator` today; the bound API takes a seed.
Bridge NOW (pre-flip): a `rng_for(seed)` helper in the regression
`conftest` returning a `Generator` today and the raw seed post-flip —
all error tests use it, so the twin files stay byte-identical across
the flip (this repairs the §8 flip proof, which the seed change would
otherwise break).

**A3 — Providers.** `trait MaterialProvider { fn nk(&self, name, wavelengths: &[f64]) -> Result<Vec<Complex64>, String>; fn contains(&self, name: &str) -> bool; }`
— grid passed **explicitly** (kills the grid-implicit class of bugs;
Python keeps its implicit-grid providers behind the bridge assert).
Impls: `MapProvider` (name → evaluated arrays + grid-length check).
Spec evaluation (`MaterialSpec` dispatch) stays Python-side for now —
Python evaluates specs to arrays and registers them (existing machinery);
porting dispatch over the native kernels is Phase E. `WeaverProvider`:
not ported (Python-side weaver; revisit if needed).

**A4 — Expansion (the intricate one).** Transliterate the two-phase
algorithm exactly: phase-1 bulk resolve (independent n/k scaling,
identity (1,1)); phase-2 emission with draws in legacy order
(thick, nk, rough, iface, inhg); owner-group
summand+draws; donor-side buffer rescale; owner+carrier mix via
`navette-materials::ema::{looyenga, eps_to_nk}` (verified: these exact
paths exist — same kernels the Python
path calls — one implementation, no drift); roughness follows the plane
with/without slice; run-incident edges clean. Return
`(SolverArrays, Vec<Span{start,end,logical}})` natively (spans are
first-class, not a flag). All S8 probes ported (mirror/nk/rough/graded/
repeat/L0/repeat-edges/partial/MC-determinism —
MC determinism = same-seed reproducibility within Rust, NOT NumPy
value-equality, per A2).

**A5 — Validation.** Port every rule with its severity: errors (negative
geometry, unknown materials, bad factors, NaN/n<0/k<0, floored films,
chain violations, merge conflicts) vs warnings (overhang, orphans);
carve-exemption via spans. Solve-gate helper (`validate` then
raise-on-errors, warnings collected).

**A6 — `Structure` / `Architect`.** Layers, groups (material-keyed,
orphan rule), blocks (kind/inverted/repeat/label), order-only iteration,
chain validation, merge with conflict raise, global + solver-index
mapping (spans), split (preserving definition) / duplicate / insert /
remove / prune / replace, `bake_films`, `total_sub_layers` (BOTH tiers: exact-via-expansion and
the structural approximation fallback — the fallback is observable
behavior, port it),
`set_optimization_mask`, `get_optimization_parameters` (THICKNESS slot).
`bake_materials` waits for Phase E (needs spec construction).

**A7 — Serialization.** `serde` derives with exact Python key names;
`schema_version` checked on deserialize (refuse missing/stale/future);
fingerprint test ported (key sets + version).

Acceptance A: all ported oracle tests green; randomized differential
Python↔Rust forward expansion (1000+ random stacks, bit-identity);
`cargo test -p navette-structure` clean; no clippy warnings.

## 4. Phase B — Bindings (size: M)

New `crates/navette-py/src/structure.rs`: `PyLayer`, `PyGroup`,
`PyStructure`, `PyArchitect`, `PyMapProvider`, `PySolverArrays`
(attributes, not dicts), `PySpan`. Conventions (match existing modules):
`Result<_, String>` cores → `PyValueError`; `py.detach` around expansion;
numpy arrays via the `numpy` crate (same version discipline as `smatrix.rs`:
`ndarray 0.15` vs `0.17` boundary respected); warnings vector → Python
`warnings.warn` per item; `seed: Option<u64>` for error paths; state
dicts built as `PyDict` (compat — never JSON strings over the boundary).
Solver rough-type codes pass through as the enum (int-compatible).

Acceptance B: binding-level round-trip probes (construct in Python,
expand, compare against the Python implementation bit-for-bit);
`maturin develop` clean.

## 5. Phase C — Python flip (size: M)

`navette/structure/*.py` become thin: `from navette._navette import Layer
as Layer` (+ behavior-preserving shims ONLY where the API intentionally
changed: `rng_for` adapter, warnings mapping). `solve_structure`
and `wrap_material_source`-adjacent helpers stay Python (thin over bound
types). Compat checklist per class: constructor kwargs, method names,
state-dict shape (incl. `schema_version`), error/warning behavior,
`__repr__` text. Then DELETE the Python implementation (no dual
maintenance — Rust-first means one home). BLAST RADIUS (re-review):
`config/{builders,loader,models}.py` construct Python `Layer`/`Group`
today — they move to the bound classes in this phase (same names, so
mostly transparent; `from_state` dict paths re-verified). The 73
regression tests run unchanged against the bound classes (strongest
possible flip proof — kept honest by the `rng_for` helper from A2).

Acceptance C: full validation suite green with zero Python model code
left (grep for class/method bodies); probes + parity unaffected.

## 6. Phase D — Pipeline migration (size: L, last for a reason)

`DesignStack.films: Vec<LayerSpec>` (pre-evaluated, solver-grid-bound)
becomes design `Vec<Layer>` + provider + grid at expansion points:
`SmatrixContext::simulate` expands via `navette-structure`, then solves.
`LayerSpec` survives as the *expanded-slice* type for needle internals
(seed insertion keeps constructing it from grid data it already holds).
`SpectralInputs::from_spec` untouched (MeritSpec side). Python driver
(`stack_from_layers`) builds bound `Layer`s; expansion (grading,
interfaces, groups) finally reaches pipeline runs — the silent-drop
limitation dies here. BLAST RADIUS (re-review): `cycle.rs`,
`evaluator.rs`, `pipeline.rs`, the `synthesis_pipeline.rs` binding, and
`validation/parity/synthesis/test_pipeline.py` all change signature-
level; port them in one commit. Re-run ALL synthesis parity + benches (no
accuracy/speed regression gate, as before).

Acceptance D: `cargo test --workspace`, 73 regression, 11 parity,
bench deltas ~0; pipeline mixed-target run reproduces its merit
trajectory.

## 7. Phase E — Deferred (only if needed)

- `MaterialSpec` dispatch port (Rust spec type over the native kernels) →
  unlocks Rust `bake_materials` + `SpecProvider`. Do it when Python-side
  evaluation becomes the bottleneck or a second Rust consumer appears.
- `WeaverProvider` port. Contract tags if versions ever feel too coarse.

## 8. Verification: every behavior tested in Rust AND Python

Dual-verification is the rule, not the aspiration: no behavior ships
verified on only one side of the boundary. Three layers, with naming
parity (`mirror_exact` means the same assertion in `#[test]` and in
`pytest`) so twins are auditable against each other:

1. **Rust oracle twins** (`cargo test -p navette-structure`): each of
   the 73 committed Python probes gets a `#[test]` twin FIRST —
   array-level asserts on thicknesses/nk/roughness, masks, bakes, gates,
   naming chains, version refusals, fingerprint. New Rust-side behavior
   (typed `ValidationIssue`, seed RNG, span structs) is tested here.
2. **Python twins** (`pytest validation/regression/structure/`): the
   SAME assertions through the public Python API — pre-flip against the
   Python implementation, post-flip against the bound classes, FILE FOR
   FILE UNCHANGED. This is the flip proof: Phase C acceptance is these
   exact files green on both implementations (run once before, once
   after, diff the results — only the import line may differ). New
   boundary behavior (warning re-emission, seed adapter, dict states)
   gets Python-side tests here too.
3. **Cross-boundary differentials** (both runtimes in one test):
   randomized Python↔Rust forward-expansion bit-identity (seeded, 1000+
   stacks incl. groups/errors/inversion, dispersive materials so nk
   order bugs can't hide); lossless reciprocity through the bound
   bridge; gate refusal at every boundary (constructors, deserialization,
   solve gates) asserted from Python against Rust errors.

Coverage rule: a behavior is DONE only with all three layers green —
Rust twin, Python twin, differential. Phase acceptances (§3–§6) each
inherit this rule; counted explicitly in review (e.g. A4 closes with
~15 Rust twins + ~15 Python twins + the randomized differential).
Differentials with error models active assert distribution/
determinism per side, NEVER cross-value equality (A2).
Performance gate (re-review): an expansion bench (Python vs Rust,
flat + graded + grouped stacks) — Rust must be ≤ Python wall time;
tracked alongside, not gamed after.

## 9. Decisions needed (yours)

1. New `navette-structure` crate vs extending `navette-smatrix`?
   → DECIDED: new crate (pure model; smatrix depends on it; no cycles).

2. Seed-based RNG (`seed: Option<u64>`) replacing passed `Generator`s?
   Option A (recommend): bound API takes `seed: Option<u64>`; `None` =
   thread RNG. Reproducible across processes, trivially FFI-clean, no
   numpy dependency in the binding signature. Cost: the `rng_for` adapter
   in test/process code, and statistical (not value) parity with NumPy.
   Option B: accept a Python `Generator` and draw in Python. Preserves
   exact NumPy streams, but every error draw crosses the GIL, expansion
   can't `detach`, and Rust-side determinism tests become meaningless —
   it keeps the model half-Python. → Recommend A.

3. Spec dispatch stays Python-side until Phase E (Rust providers take
   evaluated arrays)?
   Option A (recommend): `MapProvider` only; Python evaluates the 23
   spec models with existing machinery and registers arrays. Critical
   path stays short; the boundary is one clean handoff (arrays +
   explicit grid). Cost: two homes for material knowledge until E, and
   Rust `bake_materials` waits.
   Option B: port spec dispatch now (Rust spec type over the native
   kernels). One home sooner, but puts a 23-model port + its own oracle
   suite on the critical path before expansion even starts. → Recommend
   A; trigger E when a second Rust consumer appears or spec evaluation
   profiles as the bottleneck.

4. Python flip = same class names rebound to Rust, then delete the
   Python implementation?
   Option A (recommend): `navette.structure` re-exports the bound
   classes under identical names; the 73 twin files run unchanged as the
   flip proof; then the Python model code is deleted. One home, no drift
   surface, no compat shims to maintain. Cost: a flag day (one commit,
   everything flips) — mitigated by the twin suite.
   Option B: dual maintenance (Python fallback + Rust fast path).
   Rejected in advance: two implementations of the trickiest algorithm
   we own is how silent divergence is born; the whole point of
   Rust-first is one home. → Recommend A.

5. `sub_layer_count` as a derived method, not stored state?
   Option A (recommend): `fn sub_layer_count(&self) -> u32` computed from
   the refinement rule on demand. No stale counts possible (the Python
   stored-attribute smell dies by construction); bake/expansion call it.
   Cost: states never carry it (they don't today either — confirmed
   serialized-absent by the fingerprint test), and anyone caching it
   across a thickness change gets a fresh value, which is the correct one.
   Option B: store + re-refine on set (Python behavior). Preserves the
   attribute idiom but also preserves the stale-state hazard. → Recommend A.

## 10. Order of work

A1 → A2 → A3 → A4 → A5 → A6 → A7 → B → C → D (→ E iff needed),
each with its oracle tests green before the next starts. Estimated
critical path: A4 (expansion) dominates; A1–A3 are straightforward;
D is integration labor, not research.

## Progress log

- 2026-09-05: plan written [P]. Awaiting decisions (§9) before A1.
- 2026-09-05: re-review pass — 6 findings, all folded in: (1) ChaCha≠PCG64
kills cross-value MC equality (differentials error-free; per-side
distribution/determinism); (2) `rng_for` helper saves the unchanged-
files flip proof; (3) ema kernel paths verified present; (4) config
package + pipeline files listed as blast radius; (5) approximation
fallback added to A6; (6) expansion perf gate added. §9.1 DECIDED (new
crate). Still [P]: awaiting §9.2–9.5 before A1.
