# Bug-fix plan — `navette.structure` (`structure.py`, `models.py`, `expander.py`, `architect.py`)

Source: code review of the four stack-model files (plus `types.py` / `materials.py`
where they meet), followed by a direction/inversion audit. Every 🔴 item was
reproduced empirically; 🟡/🟢 items are confirmed by reading. Checkbox per item so
this file doubles as the work tracker.

Conventions used below: "front" of a layer/slice = light-incident side in
traversal order; `rough_vals[k]` = interface at the front of slice `k`
(solver contract); an inverted block must be a **mirror** (order reversed AND
each layer flipped front↔back) — proven to be the intent by the (correct)
inhomogen gradient flip (`expander.py`, `factors[::-1]`).

---

## Phase 1 — Data loss / crashes (do first)

### [ ] BUG-1 🔴 `Layer.from_state` drops the material name (`models.py`, `Layer.from_state`)
- **Cause:** `get_state()` serializes `"material"`, but `__init__` takes
  `material_name`. The `co_varnames` allow-list filter discards it → `material_name=""`.
- **Repro:**
  ```python
  l = Layer(thickness=10.0, material_name='TiO2')
  Layer.from_state(l.get_state()).material  # '' — was 'TiO2'
  ```
- **Blast radius:** all persistence paths (`Navette_Structure.from_state`,
  `Navette_Architect.from_state` delegate here). Reloaded stacks fail
  `validate()` ("not found in material provider") or blow up in the expander.
- **Fix:** map the key explicitly (`material` → `material_name=`); stop relying
  on the `co_varnames` filter (it also silently admits `self` and will break
  again on any future param rename — replace with an explicit key map).
- **Tests:** round-trip `from_state(get_state()) == original` (all fields) for
  `Layer`, `Group`, `Navette_Structure`, `Navette_Architect` (incl. shared-block
  aliasing preserved, materials re-attached).
- **Accept:** repro returns `'TiO2'`; round-trip tests green.

### [ ] BUG-2 🔴 `Navette_Architect.validate()` crashes when materials are set (`architect.py`, `validate`)
- **Cause:** calls `self._materials.has_material(...)`; `MaterialProvider`
  defines `contains()`, not `has_material()` → `AttributeError` on every
  `validate()` with a provider configured. Hidden when no materials are set
  (the `if self._materials:` guard skips the loop). `Navette_Structure.validate()`
  correctly uses `contains()`.
- **Fix:** one-word fix (`contains`), plus extract a shared
  `_check_material_coverage()` helper used by both classes so they can't
  disagree again.
- **Tests:** `validate()` with materials set (missing-material error reported,
  no crash); with materials unset (skips coverage check).
- **Accept:** repro (architect + materials + `validate()`) returns issues list, no raise.

---

## Phase 2 — Inversion direction errors (mirror semantics)

### [ ] BUG-A 🔴 Inverted interface slice: wrong plane AND wrong mixing pair (`expander.py`, `_LayerExpander.expand`)
- **Evidence** (`A | B(interface 4 nm) | C`, interface at front of B = plane AB):

  | | slice order | interface nk |
  |---|---|---|
  | normal | `[A, IF, B, C]` | 2.165 = Looyenga(A,B) ✅ |
  | inverted | `[C, IF, B, A]` | 3.042 = Looyenga(**B,C**) ❌ |

  Correct mirror: `[C, B, IF, A]` with Looyenga(B,A) = 2.165. Current output
  fabricates a C–B interface from never-adjacent materials; the real AB
  transition is unmodelled.
- **Masking hazard:** thickness vector is `[0, 4, 46, 0]` in *both* cases — any
  test comparing only thicknesses passes. Assert nk rows + roughness.
- **Cause:** interface slice is emitted *before* the bulk mixing with
  `prev_eff_nk` (previously *yielded* layer) = wrong neighbor, wrong side
  under reversed traversal.
- **Fix:** under `inv`, emit bulk first, then the interface slice mixing
  `layer_nk` with the *next* yielded layer's nk (one layer of lookahead or
  deferred emission; Looyenga@0.5 is arg-symmetric so only the *pair* matters,
  not the order).
- **Tests:** asymmetric A/B/C probe asserting interface position, mixing pair
  (nk ≈ 2.165), and slice order; symmetric-stack full-mirror property
  (inverted arrays == reversed normal arrays).
- **Accept:** inverted interface nk == 2.165 at index 2 (`[C, B, IF, A]`).

### [ ] BUG-B 🔴 Inverted roughness stays on the wrong slice (`expander.py`, `_LayerExpander.expand`)
- **Evidence:** same stack, `rough=5` at index 2 (front of B) in *both* normal
  and inverted output. The physical AB plane is at the front of **A**
  (index 3) after mirroring, and the solver reads `rough_vals[k]` as front of
  slice `k` — so the loss is applied at the CB plane.
- **Cause:** roughness treated as glued to the `Layer` object; it is positional
  and must mirror like everything else. (Invisible for symmetric A/C pairs —
  likely why it survived.)
- **Fix:** under `inv`, carry `current_roughness`/`rough_type` to the
  *following* emitted slice's front. Open edge case to decide: an
  interface-bearing *last-yielded* layer's roughness falls off the end (it
  becomes the substrate-side exit plane) — define the convention explicitly
  (drop with documented rationale, or attach to the final slice).
- **Tests:** same A/B/C probe asserting roughness index (3 when inverted);
  sublayer-split case (roughness on traversal-front sublayer post-mirror).
- **Accept:** inverted `rough_vals == [0, 0, 0, 5]`.

### [ ] BUG-C 🟡 Phantom boundary interfaces for sandwiched inverted blocks (`expander.py` + `architect.py`)
- **Cause:** `prev_eff_nk` chains across block boundaries by design (correct for
  contiguous joins), but an inverted block's first-yielded layer brings
  interface/roughness flags whose mirrored plane faces *inward*, yet they mix
  across the boundary with the previous block's last layer.
- **Fix:** falls out of the BUG-A/BUG-B fix (mirror-aware emission); add a
  three-block test (normal–inverted–normal) asserting no interface slice at
  the entry boundary unless the mirrored geometry puts one there.
- **Accept:** new boundary test green.

### [ ] BUG-D 🟡 `map_global_index_to_layer` is in the wrong index space (`architect.py`)
- **Cause:** docstring says "global *simulation* layer index" but implementation
  counts *logical* layers. Any inhomogeneous/interface layer offsets the two,
  so needle sites in solver indices map to the wrong film.
- **Fix (choose one):** (a) thread expansion offsets through (solver-accurate;
  bigger change), or (b) rename/document as logical-layer space and add a
  separate solver→logical resolver. Minimum: docstring must stop saying the
  opposite.
- **Tests:** stack with one graded layer; assert mapping result in the
  documented space.
- **Accept:** docstring + behavior agree; test pins the contract.

### [ ] BUG-E 🟡 Zero coverage of inversion semantics (process fix)
- **Fact:** `grep inverted tests/ validation/ docs/ examples/` → nothing. The
  `inv` flag has exactly one consumer (`factors[::-1]`); the two places that
  needed it never got it.
- **Fix:** BUG-A/B/C tests above + document mirror semantics in
  `StructureBlock.inverted` and `_iter_layers` docstrings.
- **Accept:** `inverted` appears in tests; semantics written down once.

---

## Phase 3 — API coherence warts

### [ ] WART-1 🟡 `__len__` / `__getitem__` disagree (`architect.py`)
`len(a)` = total logical layers, but `a[0]` = first block
(`__getitem__`/`__iter__`/`__contains__`/`block_count` are all block-oriented).
`a[len(a)-1]` is nonsense/`IndexError`. **Fix:** `len` = block count (matches
everything else); global-layer count stays available via
`get_global_layer_count()`. **Tests:** `len(a) == block_count`; `a[i]` block
identity; layer-count accessor unchanged.

### [ ] WART-2 🟡 `get_group_for_material` leaks the mutable `_DEFAULT_GROUP` singleton (`structure.py`)
Any caller mutating the returned group poisons the global default for all
structures. Expander's internal use is read-only (safe); the public exposure
is the hazard. **Fix:** return a copy (cheap — small slots object) or document
"do not mutate" + consider `MappingProxy`-style freeze. **Tests:** mutate
returned group, assert fresh lookup unaffected.

### [ ] WART-3 🟡 `__add__` silently keeps `self`'s provider, ignores `other`'s (`structure.py`)
Same material name resolving differently in each stack merges into silently
wrong physics. **Fix:** if both providers set and not identical, warn (min)
or raise (preferred — matches the strict group-merge policy already in
`__add__`/`_merged_group_dict`). **Tests:** merge with conflicting providers
raises/warns; identical-provider merge stays silent.

### [ ] WART-4 🟡 `total_sub_layers` implemented twice, differently (`structure.py` vs `architect.py`)
The architect copy is admittedly approximate and mishandles inverted-block
interface chaining. **Fix:** single counting pass owned by/derived from the
expander (or exact `len(get_solver_inputs().thicknesses)` where exactness
matters, documented cost). **Tests:** both agree with expanded truth on a
stack with inhomogen + interface (+ inverted block for the architect one).

### [ ] WART-5 🟡 `split_layer_at_global` doubles the interface budget (`architect.py`)
Cloned halves each keep full `interface_thickness`. **Fix:** split the
interface thickness by ratio or assign to one half (document choice); same
review for `duplicate_layer_at_global` (duplication is arguably correct there —
state the distinction). **Tests:** total interface thickness conserved across
split.

### [ ] WART-6 🟡 `from_state` silently drops out-of-range block refs (`architect.py`)
`if 0 <= ref < len(structs)` with no `else` → corrupt configs load
"successfully" minus blocks. **Fix:** raise `ValueError` naming the bad ref.
**Tests:** corrupt ref raises.

### [ ] WART-7 🟡 `Group._apply_error` scalar-vs-array semantics implicit (`models.py`)
`rng.normal(scalar…)` → scalar perturbation across whole nk array = systematic
fabrication offset (physically defensible) vs per-λ noise. **Fix:** one comment
stating "systematic, not per-wavelength noise". (Drive-by: `rng or np.random`
swallows falsy custom rngs — use `if rng is None`.)
**Tests:** array input → uniform-across-λ offset (pins the contract).

---

## Phase 4 — Minor nits (batch into one cleanup commit)

### [ ] NIT-1 `apply_to_all_layers(self, func: callable)` — `callable` is the builtin, not a type (`structure.py`). Use `Callable[[Layer], None]`.
### [ ] NIT-2 Dead module docstrings in `models.py` / `expander.py` (placed after imports/code → no-op strings; `help()`/Sphinx see nothing). Move to top. (`structure.py`/`architect.py` are correct.)
### [ ] NIT-3 `validate()` vs expander policy clash: `interface_thickness >= thickness` is an *error* in `validate()` but silently *clamped* (`min(...)`) in the expander (`structure.py` / `expander.py`). Pick one (raise in expander, or downgrade validate to warning/clamp). Also: no check for negative `interface_thickness`.
### [ ] NIT-4 Two `RoughnessType` enums (`structure.types`: NONE/SCALAR vs `smatrix`: NONE/LINEAR/STEP/…). The int flows into engine arrays read under the smatrix enum (`SCALAR=1` ≡ `LINEAR=1` by luck). Document the mapping or unify.
### [ ] NIT-5 `Structure.clone()` shares the provider by reference — presumably intended, undocumented (contrast architect's loud aliasing warnings). One line.
### [ ] NIT-6 `Group.from_state` aliases the state's `error_params` dicts (no copy), unlike `__init__` which `.copy()`s the default. Copy on ingest.
### [ ] NIT-7 `Architect.materials` setter can `None`-clobber deliberately distinct per-structure providers. At least document; consider warning when overwriting non-identical providers (pairs with WART-3).

---

## Suggested order of work

1. Phase 1 (BUG-1, BUG-2) + round-trip/coverage tests — data loss and crash.
2. Phase 2 (BUG-A → BUG-E) — inversion mirror fix + tests + documented semantics.
3. Phase 3 (WART-1 → WART-7) — API coherence before it calcifies.
4. Phase 4 (NIT-1 → NIT-7) — single cleanup commit.
