# Structure work plan (Python side first, Rust-first port after)

Scope: `src/navette/structure/` (`models.py`, `expander.py`, `structure.py`,
`architect.py`, `materials.py`, `types.py`) plus its solver touchpoints.
Workflow per item: **propose → review → implement → verify**.
Deferred items (c, f, g) get a proposal only — no code until reviewed.

Status key: `[ ]` open · `[P]` proposed · `[R]` reviewed/approved ·
`[I]` implemented · `[V]` verified. Date log at the bottom.

## STRUCT-1 — Roughness unit → nm (a) — [V]

**Decision (user): nm is the canonical unit everywhere.**

- `Layer.roughness` docstring/`__repr__`/`validate()` message: Å → nm.
- `Group.sr_roughness_error` docstring: nm; signature unchanged.
- Error defaults: the shared `_DEFAULT_ERROR_PARAMS` serves nm channels
  (thickness, interface) AND the Å roughness channel today. After the
  switch everything is nm, so split the default to preserve physical
  behavior exactly:
  - `_DEFAULT_ERROR_PARAMS` unchanged (nm channels).
  - new `_DEFAULT_ROUGHNESS_ERROR_PARAMS`: copy with `abs_*` values
    ×0.1 (0.01 Å → 0.001 nm); `rel_*` untouched (dimensionless).
  - `Group.__init__` uses the roughness dict for `roughness_error_params`.
- Relative params (`rel_std_dev`, `rel_variance`, …) are unit-free: no change.
- Migration note (breaking): state files store unitless numbers. Old files
  authored in Å read 10× too small after the switch. No version field
  exists in state dicts — document in `structure.py` module docstring;
  do NOT add silent auto-detection heuristics.
- Verification: probe asserting `Layer(roughness=2.0)` → solver array `2.0`
  with nm wavelengths, i.e. unit contract test, not just passthrough.

## STRUCT-2 — `material_name` round-trip key (b) — [V]

**Decision (user): `material_name` key as default, no alias map.**

- `Layer.get_state()`: emit `"material_name"` instead of `"material"`.
- `from_state()` then round-trips by construction (key ∈ `__init__` params).
- During implementation: grep all readers of the `"material"` state key
  (`config/loader.py`, architect states, tools, examples) and update them
  in the same commit — no dual-key transition period.
- Verification: `Layer.from_state(L.get_state()).material == L.material`
  plus structure save→load probe via `config/loader.py`.

## STRUCT-3 — Interface ε vs n (c) — [V]

No code until the analysis is reviewed. Caller audit completed:

1. **Native contract is documented ε**: `ema.rs:4` ("returns the
   effective **permittivity**") and `materials.rs:214` ("EMA mixers
   (take inclusion/host refractive indices, return permittivity)").
   Changing the kernel's contract would break its documentation.
2. **The materials path already converts**: `materials/__init__.py:242`
   wraps the call as `eps_to_nk(ema_looyenga(...))` — the established
   correct-consumer pattern. The expander is the SOLE caller that skips
   the conversion (validation goldens pin the ε output; the mirror test
   only checks the binding, not the expander's use).
3. **Solver requirement**: `core_engine` consumes `indices` as complex
   refractive index n. No solver path wants ε — conversion is mandatory
   somewhere on the interface path regardless.
4. **Cost**: one complex `sqrt` per wavelength per interface slice, once
   per expansion (~ns against a µs–ms solve). Unit choice is NOT a
   performance driver; contract clarity is.
Recommendation (awaiting approval): option (i) — convert at insertion
in the expander (`eps_to_nk`, same helper the materials path uses),
keeping the documented native ε-contract untouched. Smallest blast
radius: one call site, zero binding changes, zero golden churn.

## STRUCT-4 — Group n/k independent scaling (d) — [V]

**Decision (user): independent scaling per recommendation.** Formula:

```python
layer_nk = complex(base_nk.real * group.n_factor,
                   base_nk.imag * group.k_factor)
```

applied when `(group.n_factor != 1.0 or group.k_factor != 1.0)`.

- `Group.__init__` default: `k_factor=1.0` (was 0.0).
- `GroupConfig.k_factor` default: `1.0` (was 0.0).
- Rationale: matches `table.rs` (`n*nf, k*kf`, identity (1,1)) and the
  `TableMaterialParams` (1.0/1.0) convention. Old default (1.0, 0.0) was
  identity only under the complex-multiplication reading, which no other
  code uses.
- Migration: old-default users are unaffected (identity → identity).
  Anyone who *set* `k_factor` explicitly got complex-rotation garbage, so
  no sane behavior is being preserved — no compat shim owed.
- Verification: probes `n_factor=1.1 → (1.65+0.01j)`,
  `k_factor=0.5 → (2.35+0.005j)` on `(2.35+0.01j)` base.

## STRUCT-5 — Wire group summands + factor validators (e) — [V]

In `_LayerExpander.expand`, matching the thickness pattern
(scale/summand first, error draws after):

```python
current_roughness = max(0.0, layer.roughness + group.roughness_summand)
...
t_interface = layer.interface_thickness + group.interface_summand
```

- `t_interface` then flows through the existing error draw + the
  `min(t_interface, layer_thickness)` clamp unchanged.
- `roughness_summand` applies to the nominal value, so graded sub-layers
  (roughness only on `ix == 0`) and error draws both see it.
- Validators, two levels (review feedback — factors must not silently
  produce unphysical stacks):
  - `Group.validate() -> List[str]` (new, factor *domains*):
    `thick_factor < 0` → error (negative is never meaningful;
    `== 0` allowed: with a summand it is a legitimate uniform-thickness
    override); `n_factor < 0` → error (no negative-index media in this
    engine); `k_factor < 0` → error (no gain media in the Fresnel path).
    Summands are unbounded floats — checked at the result level instead.
  - Result level in `structure.validate()` / architect `validate()`:
    nominal dry-run expansion, then flag what the factors *did*:
    films floored to 0 by `max(0, ...)` (silent deletion today),
    `n < 0` or `k < 0` in any emitted row, NaN/complex overflow.
    Factor domains are cheap authoring-time checks; the dry-run catches
    what domains cannot (e.g. a legal summand zeroing a thin film).
  - `structure.validate()` calls `Group.validate()` for every group in
    `group_dict`; the expander itself stays fail-fast (raise on NaN/empty).
- Verification: probes — roughness 3.0 + summand 5.0 → 8.0 in array;
  interface 5.0 + summand 2.0 → 7.0 nm slice with 43.0 nm remainder;
  `Group(thick_factor=-1).validate()` non-empty; dry-run flags a
  summand-zeroed film.

## STRUCT-6 — Architect validation (f) — [V]

Implemented:
- `has_material` → `contains` (protocol method), fixing the live
  `AttributeError`.
- `validate()` delegates: per-structure `structure.validate()` for every
  unique structure (thickness/roughness/interface/material-coverage),
  then architect-level checks: empty chain, empty structures (currently
  silently skipped by `_iter_layers`), group-merge conflicts (already
  raised — surface as issues, not exceptions), materials-unset warning.
- Collect vs raise (review question, answered here for approval):
  *Collect* means `validate()` returns a list of issue strings and never
  throws: the caller sees ALL problems at once ("layers 2, 5, 7 have
  …") and decides what to do — print, log, block, fix-and-retry. This
  suits authoring time (config files, node graphs, UIs) where fixing one
  error only to hit the next is miserable.
  *Raise* means fail-fast on the first problem: one exception, no partial
  results. This suits solve time, where proceeding with a known-bad
  stack can only produce garbage (or a native crash) — fail-closed.
  Implemented as recommended: collect everywhere in `validate()`
  (structures AND architect — authoring-time API); `get_solver_inputs()`
  on both classes runs full validation first and raises (solve-time
  gate). Recursion-safe split: `_expand_arrays()` is the unvalidated
  core used by `validate()`'s dry-run; the public getters validate then
  expand. Nothing that reaches the engine is known-bad; nothing
  authoring a config is interrupted.

## STRUCT-7 — Composition: stacks vs film blocks (g) — [V]

Implemented per the reviewed decisions (declared kinds, `layer_type`
enum, block-type enum):

- `Layer.layer_typ: int` → `Layer.layer_type: LayerType` (new `IntEnum`
  in `types.py`, re-exported; config schema `layer_type` maps onto it):
  `AMBIENT = 0`, `FILM = 1` (default — every existing state file already
  carries `1`, so old states map cleanly), `SUBSTRATE = 2`. Values
  outside the enum are rejected at set time. This finally gives the tag
  a vocabulary instead of an open integer.
- `StructureBlock.kind: BlockKind` (new enum, review-confirmed): `STACK`
  = half-space-to-half-space (must open with an `AMBIENT` row, close
  with a `SUBSTRATE` row — checked via the markers, not thickness
  sniffing); `FILMS` = thin-film-only (no `AMBIENT`/`SUBSTRATE` rows
  allowed, only `FILM`). Declared explicitly (`add_structure(..., kind
  =...)`, default `STACK` for backward compat). Both enums serialize
  as ints in state dicts, rehydrated via the enum constructors (reject
  unknown values at load, fail-closed).
- Chain rules: must start and end with a `STACK`; `FILMS` legal only
  between stacks (or films). `add_structure` checks eagerly;
  `get_solver_inputs` re-checks fail-closed; empty structure → error,
  never silent skip.
- `_iter_layers` contract unchanged (still yields `(layer, inv)`), so the
  expander is untouched by this change; STRUCT-8 flag transfer keys off
  the same stream.

## STRUCT-8 — Inversion transport (h) — [V]

**Decision (user): implement + rigorous hand-vs-expander tests.**

Rule: bulk properties stay with the layer; *plane* properties
(interface slice, roughness pair) describe the boundary with the
forward-predecessor and must travel under reversal.

- Side assessment (review question): confirmed — in the current form the
  interface slice is emitted BEFORE the carrying layer's bulk, i.e. it
  sits on the physical upper (incoming-light) side, and the solver meets
  it first. The slice itself is side-agnostic (symmetric 50/50 mix), so
  only its POSITION matters. Under the transfer rule the position
  mirrors exactly: forward, the plane between L{i-1}|L{i} is the upper
  side of L{i}; mirrored, the same physical plane is the upper side of
  L{i-1}-in-traversal — and the slice is emitted before L{i-1}'s bulk,
  i.e. again on the incoming-light side. So YES, the interface
  definition moves to the other side, and it must: "incoming side"
  flips with the light path. (Caveat for later: the mix is hardcoded
  f=0.5; if asymmetric mixes ever arrive, arg order starts to matter
  and the transfer must swap mix args too — noted, not implemented.)
- Superseded by the two-phase expander (round 2): `_iter_layers` is
  back to order-only (originals, no clones). Per entry, the plane takes
  its flags from the traversal predecessor when both entries are
  inverted, else from itself; k == 0 and inverted-run incident edges
  are clean. The slice width (owner-group summand + error draw) is
  computed once per plane and carved from the owner's bulk — own rows
  forward, donor rows (uniform buffer rescale = pre-split carve)
  inverted. Mix partners are the plane's two sides (owner + previous
  forward, owner + carrier inverted). Roughness follows the plane with
  or without a slice, always with owner-group draws. Forward output is
  bit-identical to the legacy traversal, including MC draw order.
- Non-inverted path yields originals, zero-cost, behavior-identical.
- Documented limitation: exactness holds for whole-chain reversal;
  single inverted blocks inside a larger chain mirror bulk order with
  in-block flag transfer; cross-boundary flags follow traversal order.
- Transfer set (review feedback — true mirror image): plane properties
  move, bulk/design properties stay with the layer:
  - MOVE (clone with successor's flags): `interface` /
    `interface_thickness`, `roughness` / `rough_type`.
  - STAY: `material`, `thickness`, `coherent`, `optimize`, `needle`
    (policy follows the film — needle candidacy must track the glass,
    not the plane).
  - STAY: `layer_type` (STRUCT-7 design role, not traversal position).
    Consequence: `STACK` boundary validation applies to the forward
    declaration only; inverted traversal is a mirror by construction
    and is exempt from boundary re-validation.
  - INHOMOGENEITY (review question — double-checked): direction reversal
    is already correct. `factors = linspace(1-d, 1+d)` emitted top-down
    forward; `factors[::-1]` under `inv` emits top-down mirrored, which
    is exactly the upside-down stack (first sublayer 1+d at the new
    top). Note the consistency argument: for graded layers roughness is
    positional (`ix == 0`, already incoming-side in both orientations),
    while for bulk layers it is layer-attached (hence the transfer) —
    both rules agree after the fix, verified by test 2/2b below.
- Verification (hand-computed mirrors, no solver involved):
  1. 2-film + interface: inverted must equal `[45, 5, 100]` + `n≈1.92`
     slice (currently `[50, 100]`, slice lost).
  2. 3-film with middle roughness: σ must move to the successor row,
     never onto the ambient row.
  2b. Graded 3-sublayer film: inverted sublayer nk sequence must equal
     the forward sequence reversed, element-wise (inhomogeneity mirror).
  3. Inverted `repeat_count=2`: transfer applied per repetition.
  4. Interface on first layer: dropped forward and dropped mirrored
     (consistent, documented).
  5. Double inversion == identity (flag-transfer involution check).
  6. Full existing suite (`cargo` + `pytest` + parity) stays green.

## STRUCT-9 — Roughness-type unification (i) — [V]

- `types.py` checked: `RoughnessType` has only `NONE/SCALAR` while the
  solver defines six forms — and the expander passes `int()` straight
  through, so `SCALAR(1)` silently means solver-`LINEAR(1)` today.
- Plan: single canonical enum shared by structure and smatrix layers
  (re-export, not a copy); `Layer.rough_type` setter validates range;
  docstrings state nm + solver semantics.
- Rust-first follow-up (not this file): canonical enum moves to Rust
  (`LayerSpec`), Python mirrors it — same treatment as `LayerType`.

## STRUCT-10 — Structure→solver bridge (j) — [V]

- New entry point (home: `structure/__init__.py`, re-exported):
  `solve_structure(structure_or_architect, wavelengths, angles, *,
  errors=False, rng=None, **solver_opts)` → expands nominal (or error
  draws), asserts dtypes/shapes, enforces the unit contract (nm, STRUCT-1)
  and ambient convention, builds `ScatterMatrix`, returns results.
- `get_solver_inputs()` stays as the array-level API; the bridge is the
  documented path. Failing validation raises before any solve.
- Verification: bridge output bit-identical to hand-wired
  expand→ScatterMatrix on a reference stack; error path reproducible
  under seeded `rng`.

## Progress log

- 2026-09-05: audit a–j completed (probes for b/c/d/e/f/h on file in chat).
  Plan written, all items [P]. Awaiting review before any implementation.
- 2026-09-05: review round 1 — STRUCT-5 gains two-level validators
  (Group.validate domains + dry-run result checks); STRUCT-6 collect-vs-
  raise explained, recommendation recorded; STRUCT-7 decided (declared
  BlockKind + LayerType enum AMBIENT/FILM/SUBSTRATE, FILM=1 default);
  STRUCT-8 side semantics assessed (interface is incoming-side, transfer
  preserves that). Still awaiting implementation approval.
- 2026-09-05: review round 2 — STRUCT-7 confirmed enums (int-serialized,
  strict rehydration); STRUCT-8 transfer set fixed (plane props move,
  bulk/policy/layer_type stay; STACK validation forward-only) and
  inhomogeneity reversal verified correct by trace (test 2b added).
- 2026-09-05: implementation round — STRUCT-1/2/4/5/8/9/10 [V] (probes:
  10 unit checks + 19 inversion checks ALL OK; `pytest tests/ validation`
  28 passed; smatrix/synthesis parity mirrors ALL OK). STRUCT-8 design
  change during implementation: carve follows the material (donor
  pre-shrink + carrier pre-compensate, `_thickness` direct to preserve
  sub-layer counts) after the thick-mirror probe exposed [50,5,95] vs
  true [45,5,100]; first-yielded clone carries L0 flags (dropped at the
  incident edge, exact at repeat boundaries via per-rep edge copies);
  `replace_material` rewritten off `_iter_layers` (clone mutation trap).
  STRUCT-3/6/7 remain [P] deferred (analysis for STRUCT-3 delivered).
- 2026-09-05: round 2 [V] — two-phase expander replaces the clone
  transfer (reverted `_iter_layers` to order-only): planes resolve by
  lookahead with owner-group summand+draws, donor-side buffer rescale;
  this closes the MC-exactness and group-summand approximations in one
  move. Two probe-caught bugs on the way: mix partners must be
  owner+carrier (not prev) in inverted mode; roughness follows the
  plane with or without a slice (decoupled from the interface flag).
  Vacuous ravel-based nk checks replaced by dispersive 2D row-flip
  checks; lossy-R asymmetry identified as correct physics (lossless
  reciprocity holds 1e-12). Added STRUCT-3 (`looyenga_n`, eps_to_nk at
  insertion), STRUCT-7 (LayerType/BlockKind, chain rules, state kinds),
  STRUCT-6 (collect + solve-gate raise, recursion-safe split),
  loader/mask wiring (OptMask, `structure_from_config`, write paths,
  live THICKNESS consumer, orphan-group flag, schema nm/enum fixes),
  smell fixes (exact total_sub_layers, enum coercion in set_properties).
  Full workspace: 249 cargo + 28 pytest + 11/11 parity ALL OK.
- 2026-09-05: bug_fix_plan sweep [V] — BUG-D (solver→logical resolver via
  expander return_spans + docstring contract), BUG-E (10 committed
  inversion tests), WART-1/2/3/5/6/7, NIT-1/2/3/5/6/7 (NIT-6 caught live
  by the new round-trip test). Two span-tracking bugs caught by the new
  tests during implementation: slices excluded from resolution spans
  (fixed: span starts before the slice) and donor carve rescaling the
  slice row (fixed: separate bulk_spans for carves). Commit layers:
  `validation/regression/structure/` (31 tests) now pins the whole
  plan. Full workspace: 249 cargo + 59 pytest + 11/11 parity ALL OK.
- 2026-09-05: WART-5 direction reversed on review — interface_thickness is a
  process parameter (preserved verbatim on split, like roughness), not a
  divisible budget; ratio-splitting would corrupt re-growth. Consequences
  implemented: overhang legalized (NIT-3 `>=` error dropped, clamp is THE
  policy), carve-explained zero rows exempted from the dry-run check via
  spans. 32 regression tests green.
- 2026-09-05: schema version tag — `SCHEMA_VERSION = 1` (clean slate, no
  past: untagged states are malformed, not legacy) in `types.py`; carried
  by all four state dicts + both pydantic states and refused on read
  (missing/stale/future all raise). Fingerprint test pins per-entity key
  sets with classify-first guidance (breaking→bump, additive→update).
  65 green.
- 2026-09-05: bake split — `bake_films()` (renamed, no compat) +
  `bake_materials(wavelengths)` (grid-carrying Table specs, `_table[N]`
  naming with provider+group collision skip, layers renamed, n/k reset).
  Groups follow their material through the rename (masks/errors survive).
  73 validation green.
- 2026-09-05: group factor baking — `bake_groups()` on structure +
  architect folds thickness/grading/roughness/interface adjustments into
  layers and resets factors (masks/policy untouched); refuses n/k-scaled
  groups atomically (pre-check, nothing half-baked). Graded films
  rediscretize at the new thickness (totals/endpoints identical, counts
  follow the rule) — flat stacks bit-identical. 41 regression green.
- 2026-09-05: no backward compatibility required (user) — migration shims
  dropped from scope; `LayerConfig`/`GroupConfig` set `extra="forbid"` so
  stale/typo keys raise instead of silently dropping (63 green).
- 2026-09-05: severity channel — `validate()` entries starting with
  `warning:` (overhang, orphan groups) are advisory: `gate_validation`
  re-emits them via `warnings.warn` and only raises on errors. Applies
  to structure/architect getters and the bridge. 35 regression tests,
  63 validation green.
