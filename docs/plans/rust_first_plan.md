# Rust-first implementation plan — structure model in Rust, Python follows

Status (2026-09-06): THE live tracker. Phases A–D1 DONE and verified
(86 validation incl. 58 structure regression + differentials, full
workspace green, 11/11 parity). Deferred: D2 (span-aware graded
optimization), Phase E (weaver port, contract tags). Prior Python-side
work STRUCT-1..10 (all [V]) is summarized in §0.1; `structure_plan.md`
is archived history (superseded). Workflow per phase: propose → review
→ implement → verify. Log at the bottom.

## 0.1 Prior work — STRUCT-1..10, all [V] (detail: git history +
`structure_plan.md`, archived)

- STRUCT-1: nm canonical everywhere (roughness Å→nm; roughness error
  defaults ×0.1; states unitless — old Å files read 10× small, by design).
- STRUCT-2: `material_name` state key (round-trips by construction).
- STRUCT-3: interface slice converts ε→n at insertion (`eps_to_nk`;
  native ε-contract untouched).
- STRUCT-4: independent n/k group scaling (`n*nf + i(k*kf)`, identity
  (1,1); old `k_factor=0.0` default was complex-rotation garbage).
- STRUCT-5: group summands wired (roughness/interface) + two-level
  validators (factor domains + dry-run result checks).
- STRUCT-6: architect validation (collect at authoring time, raise at
  the solve gate; recursion-safe split).
- STRUCT-7: `LayerType` (AMBIENT/FILM/SUBSTRATE) + `BlockKind`
  (STACK/FILMS) enums, chain rules, int-serialized strict rehydration.
- STRUCT-8: inversion transport (plane props move, bulk/policy stay;
  two-phase expander with owner carve + donor rescale; 10 committed
  mirror tests).
- STRUCT-9: roughness-type unification (solver's six forms canonical).
- STRUCT-10: `solve_structure` bridge (validate-gate + unit/ambient
  contract + bit-identity with hand-wired path).

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
Impls: `MapProvider` (name → evaluated arrays + grid-length check) and
`SpecProvider` (name → `MaterialSpec`, evaluated on the call grid).
DECIDED (§9.3B): spec dispatch is ported NOW, not Phase E — a Rust
`MaterialSpec` type (model + validated param map, serde-compatible)
with `evaluate(spec, grid)` dispatching over the existing
`navette-materials` kernels (same kernels Python calls — one
implementation). All 23 models get oracle twins against Python's
`evaluate` (seeded grids, bit-identity incl. Table linear replay and EMA
nesting). `bake_materials` therefore ports in A6, not E.
`WeaverProvider`: not ported (Python-side weaver; revisit if needed).

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
maintenance — Rust-first means one home). → DECIDED: A (sharp cutover).
BLAST RADIUS (re-review):
`config/{builders,loader,models}.py` construct Python `Layer`/`Group`
today — they move to the bound classes in this phase (same names, so
mostly transparent; `from_state` dict paths re-verified). The 73
regression tests run unchanged against the bound classes (strongest
possible flip proof — kept honest by the `rng_for` helper from A2).

Acceptance C: full validation suite green with zero Python model code
left (grep for class/method bodies); probes + parity unaffected.

## 6. Phase D — Pipeline migration (DONE as D1, bounded; D2 future)

AS BUILT (D1, 2026-09-05): the full live-design-layers refactor was cut
to a construction-time expansion — `DesignStack::from_design` (design
films + nk table + groups + grid → expansion → solver rows, once).
Physics from rows, identity/flags from carriers; interface-slice rows
are derived (optimize/needle false, never free, never hosts); graded
profiles refuse loudly. Rationale: per-`simulate()` re-expansion would
tax every merit eval for zero accuracy gain, and row-wise optimization
of sublayers would silently un-grade profiles (see D2). The silent-drop
limitation (groups/interfaces/nk-scaling never reached pipeline runs)
dies here; flat stacks reproduce the parity merit trajectory
bit-identically. `groups=` surfaces via `run_needle` kwargs; film names
must be unique (they key the nk table).

2026-09-06: refusal LIFTED → homogenize-with-warning. `from_design`
returns `(stack, warnings)`; graded films expand as uniform base-index
rows (groups/interfaces still apply) + one warning per film.
2026-09-06: pinned graded background implemented (`background_names`;
`per_film_flags` for per-film control): background graded expands WITH
the profile, whole span forced optimize/needle=false, no warning
(explicit opt-in). Operator guards (all parity-neutral on flat stacks,
proven by unchanged merit trajectory): merge keyed on (material, nk)
— protects slices, group-scaled rows and graded spans with zero span
bookkeeping; needle sites require the host flag (fixes D1 gap where
slices could host seeds); thin-removal candidates require optimize
(budget test re-aimed; derived rows never candidates). Architect prune
needs no guard (design-level, pre-expansion).
2026-09-06: proposal review — `background_names` DELETED before
shipping; background is IMPLIED (graded + optimize=False + needle=False
⇒ profile kept, silent; any other graded ⇒ homogenize + warning).
Driver derives the core background set from flags; `per_film_flags`
stays as the per-film vehicle (3-tuples NOT adopted). 97 validation
green, parity trajectory unchanged.

D2a (PLANNED): gradient-based refinement on graded carriers, three
wirings of ONE core — build order:
1. Standalone optimizer `refine_carriers(stack, ...)` (carrier params
   `(thickness, inh_delta)`, re-derive per eval, finite-difference
   gradients, adjoint out of scope). Usable alone (hand rugates,
   imports) and the testable unit. FIRST.
2. Optional final step (`final_carrier_refinement: bool|None = None`,
   DECIDED default auto → ON iff graded carriers present, else no-op):
   one refine run over the needle loop's final stack. Accept-on-
   improvement (provably never worsens mf) is what makes auto safe;
   explicit true/false overrides. SECOND.
3. Per-cycle phase (`carrier_refinement_per_cycle`, default OFF):
   Phase 4 in the macro-cycle; most consistent landscape, highest
   cost; enable for deep rugate co-design. THIRD.
Needle stays uniform-only throughout (carriers are background, never
hosts); merge keyed on `(material, span)` so profiles survive cleanup.
Requires pinned-background span machinery first.
D2b: UNSPECIFIED (request cut off) — awaiting definition.

Program persistence (PROPOSED 2026-09-06, review answers recorded):
envelope (`schema_version/kind/name/sections`) with per-section schemas
identical standalone or nested; name references across sections
(file-first, then injectable live context); results EMBED frozen states.
DECIDED: (a) per-load namespace with optional prefix on import for
multi-load collisions; (b) results as sidecar (`run.results.yaml`),
programs stay hand-authorable/diffable. Legacy flat files keep loading.
Phases: 1 envelope+registry+gate → 2 pipeline/run schemas →
3 merit/target schemas → 4 results writer.
2026-09-06: Phase 1 IMPLEMENTED (`config/program.py`): envelope +
`load_document` (fmt switch, legacy-flat detection), per-kind section
loaders (materials/groups/structure/named/architect — each standalone-
usable), `load_program` (dependency order, file-wins, context fills
absent, prefix namespaces with consistent ref rewrites), `BlockConfig`/
`NamedStructureConfig` + `architect_from_config`, `example_program.yaml`.
6 program tests; 103 validation green.
2026-09-06: branch policy DECIDED — history untouched, keep merging:
`main` promotions stay `--no-ff` (merge bubbles accepted); GitHub's
perpetual 'dev behind main' counter is ancestry cosmetics (trees verified
identical after every promotion), not divergence.
SUPERSEDED 2026-09-06: one reverse sync-merge (dev←main, 239ecab) made
`main` an ancestor of `dev` again with zero rewriting — promotions go
`--ff-only` from here; no new bubbles, counter stays 0.
2026-09-06: RELEASE 0.4.0 — version bumped everywhere (workspace +
crates + pyproject + __about__ + smoke assert; breaking vs 0.3.0:
Rust-first port), release wheel built + force-added (wheel verified:
141 entries, METADATA 0.4.0, single _navette.pyd), dev→main merged
(41 commits, main had no divergences) and pushed. Old 0.3.0 wheel
left in target/wheels (history, not referenced).
DECIDED (format): pure YAML or pure JSON per file, no hybrid;
machine-written YAML gets an emitted comment header (provenance:
timestamp, source program, config hash, warnings); JSON refuses
non-finite results with a clear error.

## 7. Phase E — Deferred (only if needed)

- `WeaverProvider` port. Trigger: a second Rust consumer of woven
  grids, or measurable snapshot-callback overhead. NOT triggered by
  principle — the weaver wraps live Python workflow state
  (re-woven grids, `target_wavelength`, strict mode), and porting it
  means porting spectralweave's weave machinery for one consumer.
- Contract tags: REJECTED, see §9.6 (schema_version only).
(Spec dispatch + `bake_materials` moved into A3/A6 per §9.3B.)

Phase E is therefore CLOSED (weaver ported 2026-09-06; tags rejected).
No open deferred items remain except D2 (span-aware graded optimization).

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
AS BUILT: 86 validation (58 structure regression incl. randomized
differentials: 300 stacks validity-agreed + bit-identical, 100 mixed
chains, error moments ±5%) + full workspace green + 11/11 parity.
Pipeline construction bench still open (D1 adds one expansion at
stack build; per-simulate cost unchanged by construction).

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
   it keeps the model half-Python. → DECIDED: A (different draws accepted;
   acceptance is statistical agreement + per-side determinism).

3. Spec dispatch stays Python-side until Phase E (Rust providers take
   evaluated arrays)?
   Option A (recommend): `MapProvider` only; Python evaluates the 23
   spec models with existing machinery and registers arrays. Critical
   path stays short; the boundary is one clean handoff (arrays +
   explicit grid). Cost: two homes for material knowledge until E, and
   Rust `bake_materials` waits.
   Option B: port spec dispatch now (Rust spec type over the native
   kernels). One home sooner, but puts a 23-model port + its own oracle
   suite on the critical path before expansion even starts. → DECIDED: B
   (Rust owns recipes end-to-end; A3 now includes the spec port — see
   revised A3 below).

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
   Rust-first is one home. → DECIDED: A (flip done 2026-09-05; ~15
   zero-caller helpers intentionally not re-bound — re-add on demand).

5. `sub_layer_count` as a derived method, not stored state?
   Option A (recommend): `fn sub_layer_count(&self) -> u32` computed from
   the refinement rule on demand. No stale counts possible (the Python
   stored-attribute smell dies by construction); bake/expansion call it.
   Cost: states never carry it (they don't today either — confirmed
   serialized-absent by the fingerprint test), and anyone caching it
   across a thickness change gets a fresh value, which is the correct one.
   Option B: store + re-refine on set (Python behavior). Preserves the
   attribute idiom but also preserves the stale-state hazard. → DECIDED: A.
   Perf audit: the rule is one pow + ceil (~10 ns), called O(layers) times
   per expansion — unmeasurable against µs–ms workloads, no memoization
   needed. Sole downside: hand-tuned counts deviating from the rule become
   unrepresentable — an undocumented escape hatch nothing uses (states
   don't carry it; only the bypass smell wrote it). Its removal is the
   point, not a cost.

6. Contract tags vs `SCHEMA_VERSION`? → DECIDED: schema_version only.
   `SCHEMA_VERSION` is the single versioning mechanism (bump = old
   states refuse; fingerprint test pins meaning per build). No
   per-subsystem tags in outputs — rejected as second versioning
   channel that would need its own bump discipline without a reader
   to enforce it.

- 2026-09-05: Phase B done: `_structure` submodule (Layer/Group/
DictProvider/SolverArrays/Structure/Architect + warnings re-emit, seed
RNG, GIL-detached expansion, `unsendable` shared handles). Binding
adaptations documented + tested (prefixed validate strings, snapshot
providers, fixed-length masks). Differential suite green: 300 stacks
(validity agreed + bit-identical), 100 mixed chains, error moments agree
within 5% (test caught a live rel-variance default). 86 validation +
41 Rust + full workspace (16 suites) green.

- 2026-09-05: Phase C done — THE FLIP. `models.py` re-exports bound
`Layer`/`Group`; thin `Navette_Structure`/`Navette_Architect` wrappers carry
providers (dict auto-wrap restored), own bake pour-back + shell tracking
(`is`-identity preserved); `expander.py` DELETED; `builders.py` moved to
`set_error_*`. Binding grew: materials slots with overwrite warnings,
provider snapshotter (dicts/_dict/get_nk/customs, full-shelf for collision
checks, unknown-tolerant for validation), grid resolution (stored/dummy/
refuse), error accessors, mappings with Python names + IndexError,
`__add__` surface, `apply_error` fn. Shared-group handles (`Rc`) mirror
Python reference semantics (bake visible via originals). 86 validation +
41 Rust + workspace + parity green with ZERO twin-file semantic changes
(only pre-approved migrations: native `apply_error`, `rng_for` fixture,
new-API differential).

- 2026-09-05: Phase D1 done (bounded): `DesignStack::from_design`
(design films + nk table + groups + grid → expansion → solver rows;
physics from rows, identity/flags from carriers, slice rows derived
with optimize/needle false; graded refused loudly). Driver builds bound
Layers + nk table (`groups=` surfaces via `run_needle` kwargs; names must
be unique). Flat parity trajectory bit-identical; groups/interfaces now
expand (were silently dropped). 86 validation + workspace green.
D2 (span-aware graded optimization) stays future.

- 2026-09-06: plans unified — `rust_first_plan.md` is THE tracker
(§0.1 folds STRUCT-1..10; `structure_plan.md` superseded, kept as
history). §6 records D1-as-built + D2; §7 triggers explicit; §8 counts
current; §9.4 → DECIDED A, §9.6 tags-vs-schema noted. Pure-Rust gate
`cargo test-pure` (.cargo alias; verified with Python scrubbed from
PATH). Publish path proven: `cargo publish --dry-run -p
navette-interpolate` packages+verifies; wheel metadata PyPI-ready;
version still triple-synced by hand (pyproject/Cargo/__about__).

- 2026-09-06: Phase E (weaver) TRIGGERED by request: Rust-only woven
runs. `WeaverProvider<B: WovenBackend>` in `navette-structure`
(`weaver.rs`) + impl for the native `OpticalWeaver`; same
`UniInterpolator` kernel → bit-identity by construction, pinned by HEX
frozen oracles (numpy repr truncates to 8 dp — oracles must be
`tobytes`, lesson learned the hard way: one-ulp false alarm). Faithful
quirks: n-required/k-zeros, strict refuse, memo + invalidate, target
setter clears, `is_exact`, caller-grid refusal, NaN/-0.0 key
normalization. 6 Rust tests (incl. real-weaver→expansion end to end)
+ 4-test Python twin. 47 structure / 90 validation / workspace green,
clippy clean.

- 2026-09-06: dropped-surface restoration (request): `Group.get_properties`
(== state, both classes have `set_properties`) + 5 per-channel draw
helpers (`value, seed=None`; legacy `thickness` arg was vestigial —
accepted, never used); structure `find/count/apply/insert/remove/
replace/active-alias/thickness/contains` (indexed mutation is new core
binding ops, list semantics); architect `copy/active-alias/
replace_material/total-thickness`. Still dropped: `generate_simple_
layer_list` (legacy row adapter + cache side effect), chain surgery
(insert/replace/move/remove/clear), `index`/label lookup (no callers),
architect block-`__contains__`. 94 validation green.

## 9.7 Open items (single consolidated list — update on change)

Awaiting user: D2b definition; dead `tests/` dir (delete/keep);
program Phases 2–4 go-ahead; D2a build go-ahead; release re-run
outcome (v0.4.0 retagged on current main, hardened workflow).
Ready on request: construction bench; config pipeline builder;
leftover dropped surface (chain surgery, index/label lookup,
simple_layer_list, architect block-contains).
Housekeeping: binaries untracked (registries host releases).
2026-09-06: SINGLE-CRATE MERGE done — engines are modules of one
`navette` crate (public paths unchanged); the six 0.4.0 subcrates
stay frozen orphans on crates.io; future publishes: `navette` only.
Branch policy: dev_rust →(no-ff) dev →(ff-only) main; verify
tree-identity after promotions.

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
crate). §9.2 DECIDED (seed RNG, full-Rust randomness).
- 2026-09-05: foundation laid [A0]: `navette-structure` crate (enums with
  pinned discriminants + fail-closed coercion, SCHEMA_VERSION gate, typed
  ValidationIssue) in workspace + umbrella; 4 Rust tests green,
  `cargo check --workspace` clean (2 pre-existing smatrix warnings,
  untouched). Still [P]: A1 unblocked — all §9 decided (§9.3B recipes in Rust,
§9.4A sharp cutover, §9.5A derived count). Order holds with A3 now
carrying the spec port.
- 2026-09-05: A1 done: `Layer` (fields/defaults/mask/Display/state
key-for-key + version-checked/unknown-ignoring deserialize,
`set_properties` warnings-as-return, derived `sub_layer_count`,
int-serde for all 7 enums). 9 Rust tests green incl. refinement counts
bit-matching Python (powf/ceil agreement holds on pinned values).
- 2026-09-05: A3a done: `MaterialProvider` trait (grid in the signature)
+ `MapProvider` (exact-grid serve, length-checked insert/construct,
atomic refresh) + `assert_provider_grid` bridge helper. Same-length-
other-values refused at serve (the Python silent case, closed). 21 Rust
tests green.
- 2026-09-05: A3b done: `MaterialSpec` + `evaluate` over the native
kernels (raw value-map params + per-model extractors; aliases, nesting,
error texts mirrored; table-size assert is a Result, not a panic). All 23
models bit-track Python on live-captured oracles + 9 dispatch-error twins.
23 Rust tests green. EMA cores confirmed in `navette-materials` (binding
only wraps) — no move needed.
- 2026-09-05: A4 done: two-phase `expand` (bulk resolve, mirror rule,
donor carve + buffered rescale, owner+carrier mix, plane roughness, draw
order, spans first-class) + `SolverArrays` + scalar-draw array perturb.
RNG widened to `&mut dyn RngCore` (seeded/thread). Flat/full/mirror
oracles bit-match Python (slice mix, 11x graded factors, roughness).
29 Rust tests green. Randomized differential deferred to Phase B (needs
bindings to drive both sides from pytest).
- 2026-09-05: A5/A6 done: `Structure` (validate incl. dry run +
carve-explained, gate, solver/error inputs, both total_sub_layers tiers,
bakes incl. Rust `bake_materials` with `_table[N]` naming, states) +
`Architect` (Rc-shared blocks, chain rules, merge conflicts, global/solver
mapping, split/duplicate/insert/remove/prune, masks, reference-preserving
states). Providers external (states never carried them). 41 Rust tests
green, clippy clean.
- 2026-09-05: A2 done: `Group` + `ErrorParams` (identity defaults incl.
roughness x0.1, validate domains, draw order + floors, state fingerprint
23 keys, set_properties warnings-as-return). `rand 0.9`/`rand_distr 0.5`;
`StdRng`-seeded determinism + Gaussian/Uniform statistics pinned. 15 Rust
tests green.
