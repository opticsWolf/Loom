# Color targets + needle drivers — Option B implementation plan

**Locked decisions (2026-09-06):** quantities `Lab | XyY` × distances
`DeltaE2000 | DeltaE76 | Channels`; defaults D65 + 1931 2°, all fields
overridable; reflectance + transmittance in v1 (front; back refused with
a clear error); **Option B** (dedicated gradient bucket); embed D65 +
1931-2° tables in the crate with CI sync vs `src/navette/data/CIE/`.

**Why B over A (recap):** numerically identical at the op point; B keeps
`NeedleTargets` honest (no fake per-λ "targets"), keeps predicted-gain
bookkeeping first-order-exact by construction, and gives later work
(second-order terms, phase-color) a real home. Cost is a small,
well-guarded addition to the hot pass, free when unused.

Conventions: nm everywhere (solver grid = CMF grid units, no conversion);
`Exact` demands only in v1 (other kinds refused at compile, §3).

---

## D1. Canonical data (embed + sync)

1. Extract minimal tables (script, checked in as `tools/extract_cie_defaults.py`):
   - `rust/navette/data/cmf_1931_2deg.json` —
     `{"wavelengths":[…471…],"x":[…],"y":[…],"z":[…]}` from
     `src/navette/data/CIE/cmf/CIE_xyz_1931_2deg.json`
     (`/data/lambda/values`, `/data/{x,y,z}_bar(lambda)/values`).
   - `rust/navette/data/illum_d65.json` —
     `{"wavelengths":[…],"values":[…]}` from
     `src/navette/data/CIE/sds/CIE_std_illum_D65_S_D65.json`.
2. `rust/navette/data/NOTICE` — attribution: CIE data © CIE,
   CC-BY-SA 4.0 (as labeled upstream); tables unmodified, only
   reformatted. (Share-alike binds the *data files*, not the crate.)
3. `tools/check_cie_sync.py` — re-extracts from `src/navette/data/CIE/`
   and byte-compares against `rust/navette/data/`; nonzero exit on drift.
   Wire into CI (new `lint` job or the existing check step — wherever
   `check_exposure.py` runs; if exposure runs in CI, co-locate).
4. Rust access: `include_str!("../data/cmf_1931_2deg.json")` parsed once
   (`std::sync::OnceLock`) into the canonical default tables. No runtime
   I/O — the core's no-I/O rule survives (embedding is compile-time).

## D2. Core kernel — `smatrix/synthesis/color_merit.rs` (new, ~250 lines)

```rust
// P1: Lab | XyY. P2 adds LCh | Oklab | Y (scalar) — §D9 progression.
// Reserved (on demand, no work planned): Srgb | Luv | Xyz | Din99.
pub enum ColorQuantity { Lab, XyY, LCh, Oklab, Y }
pub enum ColorDistance { DeltaE2000, DeltaE76, Channels }
/// Triple reference, or scalar for `Y` (luminance-only demands).
/// JSON: `[62, 18, -34]` vs `12.5` (untagged; validated against quantity).
pub enum ColorReference { Triple([f64; 3]), Scalar(f64) }
pub struct ColorDemand {
  pub key_idx: u32,            // normal MeritKey (angle+curve) — §3
  pub cmf: Vec<[f64; 3]>,      // NATIVE table grid (resampled at eval, §D2.4)
  pub cmf_wl: Vec<f64>,
  pub illuminant: Vec<f64>,
  pub illum_wl: Vec<f64>,
  pub quantity: ColorQuantity,
  pub reference: ColorReference,
  pub distance: ColorDistance,
  pub weight: f64,
}
```

Quantity × distance compatibility (enforced at compile, §3):

| quantity | DeltaE2000/76 | Channels | notes |
|---|---|---|---|
| Lab | yes | yes (L/a/b tol) | white = illuminant white (§D2.2) |
| XyY | REFUSED (ΔE is Lab-space) | yes (x/y/Y tol) | fully analytic → bitwise twins |
| LCh (P2) | yes (ref converted LCh→Lab, ΔE in Lab) | yes, **hue diff wrapped** to [−180,180] | singularity at C=0 → FD-twin |
| Oklab (P2) | REFUSED — use equal-tol Channels, mathematically identical to unweighted Euclidean | yes | smooth incl. neutrals (cbrt guarded by FD) |
| Y (P2) | n/a (scalar) | scalar tol | the AR classic: dark residual, hue free |

Cross-illuminant rules: Lab/LCh white point = the demand illuminant's
own white (computed at compile from native tables — never adapt
D65-numbers to F2-numbers). Oklab is D65-defined: non-D65 demands
Bradford-adapt XYZ → D65 first (existing `func_08`, illuminant white
known at compile, D65 tristimulus constant in-tree), then Oklab.
Documented, tested (adapted-white identity twin).

1. `xyz_of_spectrum(sim_row, sim_wl, cmf, cmf_wl, illum, illum_wl)` —
   generalize `color::func_13` integration: resample tables onto `sim_wl`
   (linear; CMFs smooth), `k = 1/ΣE·ȳ·Δλ`, `XYZ = Σ R·E·cmf·k·Δλ`.
   Returns `Err` on empty overlap (callers map to skip/refuse per §D4).
2. `color_of_xyz(xyz, quantity)` — Lab via `common::xyz_to_lab` (D65
   white default = illuminant white? **rule:** Lab white point = the
   demand illuminant's own white — computed once at compile from the
   native tables; avoids adapting D65-numbers to F2-numbers nonsense);
   xyY via `func_01::xyz_to_xyy`.
3. `eval_color(demand, sim_row, sim_wl) -> (residual, grad)`:
   - residual: `√w·ΔE` (ΔE76/2000 via `func_09/16`, k*=1) or per-channel
     `(c−c_t)/tol` folded as `Σ` with `tol = (1,1,1)` default for
     `Channels` (documented: unit tolerances, weight scales).
   - gradient `g(λ) = ∂F/∂R(λ)`, `F = w·ΔE²` (resp. channel sum):
     analytic `dXYZ/dR(λ) = E(λ)·cmf(λ)·k·Δλ` (exact, linear) × 3-pt FD
     of the XYZ→quantity→distance map (`h = 1e-6·(1+|XYZ|)`, central).
     Cost per demand: 1 XYZ + 6 tiny map evals — negligible vs an EM solve.
4. Resample policy: tables resampled **per eval into a scratch buffer**
   (no `RefCell` memo — keeps `MeritSpec: Send+Sync`, thread-safe under
   rayon; cost O(nw) lerp vs O(nl·nw·na) solve noise).
5. Back/absorption/phase curves refused at *compile* (§3), so the kernel
   never sees them; kernel still defensively asserts finite inputs.

## D3. Schema + compile (`targets.rs` arm)

1. `TargetSet` gains `#[serde(default)] pub color: Vec<ColorTargetJson>`:
   `{curve ("Ru"|"Rs"|"Rp"|"Tu"|"Ts"|"Tp"), angle, illuminant (name|table),
   observer (name|cmf-table), quantity, reference ([3] or scalar for `Y`),
   distance, weight}`.
   Names resolved: `"D65"` → embedded default; `"1931_2deg"` → embedded
   default; anything else must be an explicit `{wavelengths, values}`
   table (mirrors the explicit-grid provider rule — no registry to drift).
2. `compile_merit_spec` color arm (no weaver involvement — color carries
   its own tables, not `TargetEntry` curves):
   - curve ∉ {Rs,Rp,Ru,Ts,Tp,Tu} → refuse (`"...: color needs a front
     R/T intensity curve, got {curve}."`); back curves refused in v1
     (`"...: back-incidence color is not supported yet."`).
   - quantity/distance strings unknown → refuse; reference shape must
     match quantity (scalar ⟺ `Y`, triple otherwise — refuse with
     `"color: scalar reference needs quantity 'Y'"` / converse);
     quantity×distance pairs outside the §D2 matrix → refuse naming both;
     weight via `check_weight`; kind ≠ Exact → refuse;
     transform ≠ linear → refuse (mirrors the needle linear-only rule);
     `integral`/`count_norm`/`phase`/`band` set → refuse (color *is*
     integral; no double counting).
   - registers a **normal `MeritKey{angle, curve}`** (no `MeritTarget`
     entries) → missing-curve penalties, `angle_row`, and polarization
     branch enabling reuse existing machinery untouched. Dedupe keys
     against spectral/angular ones (same join-by-content rule).
   - appends `ColorDemand{cmf/illum NATIVE grids, white point precomputed,
     …}` to new `MeritSpec.color: Vec<ColorDemand>`. Existing structs
     (`MeritKey/MeritTarget/MeritSpec` fields) otherwise untouched —
     no migration risk.

## D4. Scalar merit (`merit.rs` arm — 4 touch points, enumerated)

1. `n_residuals()` — `+ self.color.len()` (1 per demand, not `nw`;
   fixed-length vector intact for LM/thick-opt).
2. `residuals()` — after existing per-key loop, push color residuals in
   **demand insertion order** (deterministic; documented).
3. `merit()` — same inclusion (sums `r²`, identical to pointwise path).
4. `residuals_into` — new `color_residuals_into(sim, out)`: per demand,
   `angle_row`, `irow(key.curve)` (missing → `Err(key.curve)` = the
   standard missing-penalty path, parity with pointwise); empty
   table/sim overlap → `Err` (refuse-loud: a color demand that sees
   nothing is a spec bug, unlike pointwise grid-miss skips — documented
   divergence, tested). LCh-hue residual wrapped to [−180,180] before
   scaling (wrap-twin: h=179° vs h=−179° ≡ 2°, not 358°). `Y` demands
   push one scalar residual.
   - Thick-opt/LM work with color with **zero further changes**
     (they consume `residuals()`).

## D5. Option B needle machinery (the heart)

### D5.1 `NeedleTargets` — two new fields
```rust
pub struct NeedleTargets {
  pub r: …, pub t: …, pub a: …, pub rb: …, pub tb: …, pub ab: …,
  pub phi: …,
  /// v1: front R/T only (back refused at compile). Angle-major na*nw,
  /// zeros default. Carries g(λ)=∂F/∂curve(λ) (chain-rule factor incl.
  /// weight, residual, and the U-curve ½ — §D5.3), NOT (target,weight).
  pub grad_r: Vec<f64>,
  pub grad_t: Vec<f64>,
  pub phi_gain_shift: [f64; 4],
}
```
- `Clone` preserved (plain `Vec`s). Exactly **4 construction sites** to
  update (grep-verified 2026-09-06): `needle_pass.rs:419` (production),
  `cycle.rs:322`, `needle_pass.rs:884`, `pipeline.rs:323` (all test
  helpers) — zero them there.

### D5.2 Gradient P-kernels (`needle_operator.rs`, ~20 lines)
Mirror bodies, replace the residual line — visual-diff reviewable:
```rust
pub fn p_coherent_grad_r_from_fields(fields, nsin_fi, lam, pol, needle_n,
    grad: f64, thicknesses, start_idx, end_idx, z_grid) -> Vec<f64> {
  let r_k = fields.s_left[end_idx].0; let rc = r_k.conj(); …same loop…
  out[zi] = grad * (rc * dr).re;      // was: resid * (rc*dr).re
}
pub fn p_coherent_grad_t_from_fields(…same…, grad, …) {
  … out[zi] = grad * f * (tc * dt).re; // flux factor KEPT (boundary-invariant)
}
```
No new `NREQ_*` flag: gradient buckets ride whichever R/T channels the
fold computed — the engine needs no new request kind.

### D5.3 Fold arm (`build_needle_targets`, after the integral block)
Per color demand (Exact only — guaranteed by compile):
1. `op` spectrum row at demand angle (same `irow` helper; `None` before
   first sim → skip demand this pass, mirrors conservative fold).
2. `(R_resid, g) = eval_color(demand, op_row, sim_wl)` with the CURRENT
   residual (activation-correct by construction).
3. **U-curve ½ (load-bearing):** `Ru = (Rs+Rp)/2`, so each polarization
   branch receives `g/2`; s/p demands deposit full `g` (their branch is
   the only one enabled — existing `pol_on` machinery). The fold applies
   the factor at **deposit time** (branches sum linearly, so per-demand
   splitting stays exact under mixed Rs+Ru sets).
4. Deposit into `grad_r`/`grad_t` at solver-point indices (demand angle
   → nearest angle row, same `argmin` rule; λ onto the solver grid —
   color integrates on the SIM grid, so indices are direct, no
   interpolation).
5. No-sim / missing-curve → skip (fold-conservative, mirrors `op: None`
   handling); never invent gradients.

### D5.4 Hot-pass accumulation (`needle_pass.rs:790-796` region)
```rust
if fold.grad_r[k] != 0.0 {
    add(p_coherent_grad_r_from_fields(&fields, nsin_fi, lam, pol, np_c,
        fold.grad_r[k], th, si, ei, &z_grid));
}
if fold.grad_t[k] != 0.0 { …same for T… }
```
Nonzero-skip guards = existing pattern (`fold.r.1[k] != 0.0`): color-free
sets pay exactly one float compare per point. Back-bucket symmetry
preserved for later (fields exist when back color lands).

## D6. Python surface (thin, per R5 rules)
- `spectralweave/target.py`: `ColorTarget` dataclass
  (`curve="Ru"`, `angle=0.0`, `illuminant="D65"`, `observer="1931_2deg"`,
  `quantity="Lab"`, `reference=(…)` triple **or scalar for `Y`**,
  `distance="DeltaE2000"`, `weight=1.0`) with `_dump()` (names resolved
  to arrays from `data/CIE/`, or explicit arrays passed through);
  shape/compat pre-checks mirror §D3 messages (native re-validates —
  single validator, no duplication); `TargetCollection.color_targets`;
  `validate_targets` arm natively (fail-fast messages mirror §D3).
- `synthesis/__init__.py::build_merit_spec`: include the `"color"`
  section (same 5-line shape as spectral/angular).
- `NeedleRequest`: **unchanged** (no new flag needed).
- Docs: target-kinds doc gains the color section (kinds/refusals).

## D7. Bindings + lint
- Exposure lint (confirmed mechanism 2026-09-06: `p_coherent_*` are
  explicitly allowlisted in `tools/check_exposure.py:28-30`): append
  `"p_coherent_grad_r_from_fields", "p_coherent_grad_t_from_fields"`
  to that block (entry point `run_needle_pass`). `ColorDemand` /
  `ColorQuantity` / `ColorDistance` flow through the already-bound
  `compile_merit_spec` — no new binding required. Embedded-table
  accessor (if added for debugging) gets bound or allowlisted
  deliberately, never silently. Verify lint green.
- Embedded-table accessors (if exposed for debugging, e.g.
  `default_tables()`) get bound or allowlisted deliberately.

## D8. Regression fortress (extensive, layered)

**R0 — existing suite must not move (gates, every step):**
`cargo test-pure` (292+22+15), exposure lint, `pytest validation` (130),
parity mirrors (synthesis 4+9+6 + physics/backside/needle scripts),
zero new warnings (`-W error::UserWarning` probe on program load +
merit paths). `NeedleTargets` stays `Clone`; `MeritSpec` public API
unchanged (only additive); all current `targets_builder_*` /
`integral_fold_*` / `fold_applies_weight_and_count` tests untouched and
green — the B-accumulation guards guarantee bit-identical output for
grad-free sets (prove with a dedicated test: fold with empty grad vecs
≡ old output bitwise).

**R1 — Rust unit (`color_merit.rs`, standalone, no Python):**
- XYZ of a flat R=1 spectrum = illuminant white (k-normalization self-check).
- XYZ vs generalized `func_13` path on a ramp spectrum (bitwise).
- Lab/xyY vs `common::xyz_to_lab` / `func_01` direct calls (bitwise).
- ΔE wrappers vs `func_09/16` singles (bitwise).
- Gradient FD cross-check: analytic-Jacobian+3pt-FD `g(λ)` vs brute-force
  per-λ bumped full eval (tol 1e-9 relative) on a 3-λ toy.
- Empty-overlap → `Err`; non-finite table → `Err`.

**R2 — compile/refusal tests (Rust):** each §D3 refusal fires with the
specified message (bad curve, back curve, bad quantity/distance,
non-3 reference, non-Exact kind, non-linear transform, phase/integral/
count_norm/band set, unknown illuminant name); key-dedupe with a spectral
demand on the same (angle, curve) shares one key; white point =
illuminant white (assert vs direct integration).

**R3 — merit twins (Python, vs `navette._color` oracles):**
- xyY+Channels set: merit value HEX-equal vs hand-rolled NumPy oracle
  (fully analytic path — strictest test in the batch).
- Lab+ΔE₀₀ set: merit vs `_color` pipeline (spectral→Lab→ΔE2000) at
  1e-12 relative.
- Missing-curve penalty parity with pointwise demands; `n_residuals`
  counting test; LM smoke: 2-film stack improves a color merit over
  3 iterations (no NaN, monotone-ish).

**R4 — needle twins (the critical batch):**
- Chain-rule vs brute force: 2-layer stack, color demand (R, then T):
  B-needle `P(z)` vs finite-differenced merit over inserted-needle
  thickness sweep (tol 1e-6 relative, interior sites).
- U-curve ½: Ru-demand `P(z)` ≡ (Rs-demand + Rp-demand)/2 profiles
  (bitwise — same slopes, linearity check).
- s-only demand enables s-branch only (profile ≡ s-branch reference).
- Grad-free regression: full needle pipeline on the existing graded
  demo ≡ pre-change profile bitwise.
- Skip-guard perf: color-free pass timing unchanged (existing bench).

**R5 — Python surface:** `ColorTarget` validation errors mirror native
messages; `_dump` name→array resolution (D65 + explicit-table paths);
`build_merit_spec` color section compiles; program-document roundtrip
with a color demand (schema v1 carries it — assert restore → same merit).

## D9. Progression (vertical slices — each phase works end to end)

Rationale: slice by *quantity set*, not by layer. P1 proves the whole
chain (kernel → schema → merit → needle-B → Python) on Lab|xyY; P2 runs
the same arms with LCh|Oklab|Y, which by then is mostly new tests — the
machinery already exists. Horizontal layering (all quantities in the
kernel first) would touch every arm twice.

- **P1 — core color, Lab|xyY (0.4.23–0.4.26):**
  - 0.4.23: D1 remainder (embedded defaults + sync CI) + D2 kernel
    (Lab|xyY) + R1 tests. DONE 2026-09-06 (dev): `tools/cie_defaults.py` +
    extract + sync scripts, `rust/navette/data/{cmf_1931_2deg,illum_d65}.json`
    + NOTICE, CI `check-cie-sync` job gating wheels/crates, `tables.rs`
    `default_tables()` (OnceLock, `pub(crate)`), `synthesis/color_merit.rs`
    (all fns `pub(crate)` — exposure unchanged at 204/90), 7 R1 tests
    (306 lib green). Notes: white-Y identity is 1-ulp not bitwise (per-term
    k rounding — asserted 1e-15); Δλ is forward-difference (uniform grid ⇒
    op-for-op `func_13` summation); kernel dead_code warnings live until
    0.4.24 wires D3/D4 consumers.
  - 0.4.24: D3+D4 schema/compile/merit (Lab|xyY) + R2/R3 tests. DONE
    2026-09-06 (dev): `ColorTargetJson` (+`IllumJson`/`CmfJson` name|table,
    `ReferenceJson` catch-all for named non-3 refusals), compile arm
    (front-R/T only, back refused, Exact+linear only, integral/count_norm/
    phase/band refused, shared key registry), `MeritSpec.color` +
    `add_color_demand` + key-grouped `color_residual_into` (missing/overlap
    → `Err(key.curve)` = standard penalty path; pointwise grid-miss still
    skips — documented divergence). R2: 4 tests (compile+dedupe, 16
    refusals, JSON-shape loudness, embedded-name white bitwise). R3: 5
    twins incl. xyY-Channels HEX vs pure-Python-loop oracle (1-ulp-strict).
    LM smoke lives in Rust (`lm_drives_color_residual_to_zero`, cost →4e-16
    = 1-ulp chromaticity floor). 312+22+15 Rust, exposure 204/90, 334 green.
  - 0.4.25: D5 needle-B (R/T buckets, U-½) + R4 tests. DONE 2026-09-06
    (dev): `NeedleTargets.grad_r/grad_t` + 4 construction sites,
    `p_coherent_grad_r/t_from_fields` (mirror bodies, flux kept) allowlisted
    (D7-lint-half early), fold arm (op-sim eval, U-½ at deposit, no-sim /
    missing-curve skip, overlap-Err propagates), hot-pass nonzero-skip arms
    + shape validation. R4: 6 Rust tests (deposit-half bitwise, fold-level
    central-FD chain rule 1e-6, U-curve BITWISE, branch additivity bitwise,
    grad-free inertness bitwise, kernel-mirror ratio constancy). CORRECTION
    vs draft: the U-identity needs single-branch Rs/Rp references —
    `P_Ru(both) == (P_Rs(s-only)+P_Rp(p-only))/2` (the draft's unqualified
    form is off by 2x; s/p deposit FULL g mirroring pointwise, so both-
    branch single-pol runs inherit the engine's existing cross-term
    convention — no new error class). The planned s-only-branch test was
    replaced by branch-additivity (branch gating is out of scope).
    318+22+15 Rust, exposure 206/92, 334 green, refold bench OK.
  - 0.4.26: D6+D7 Python surface + docs + R5 tests. DONE 2026-09-06
    (dev): `ColorTarget` (frozen, `_dump` + native `__post_init__` check) +
    `TargetCollection.color_targets` + `build_merit_spec` color section +
    `validate_targets` color arm via shared `check_color_demand` (cross-
    crate `pub` + allowlisted); target-kinds doc color section. R5: 5
    Python tests (refusal mirroring, passthrough dump, collection+compile,
    JSON roundtrip HEX, named-vs-explicit HEX) + 1 Rust roundtrip test.
    DEVIATIONS: (1) `_dump` passes names through (no 471-pt bake into docs;
    embedded resolution is sync-guarded identical — proven HEX-equal);
    (2) no program-document roundtrip — program schema carries no targets
    section (TargetSet JSON roundtrip instead); (3) `check_color_demand`
    is `pub` (navette-py is a separate crate) + allowlisted.
    319+22+15 Rust, exposure 207/93, 339 green.
- **P2 — extended quantities (0.4.27):** LCh + Oklab + Y-scalar through
  all arms in one patch: scalar-`reference` JSON shape, §D2 compat
  matrix, hue-wrap, Oklab D65-adapt rule. New twins: LCh-Channels vs
  `_color` LCh pipeline; Oklab-Channels vs `func_04` direct (bitwise —
  smooth map, analytic Jacobian + FD only at cbrt-zero guard);
  Y-scalar vs hand-rolled ΣR·E·ȳ oracle (bitwise); hue-wrap unit test;
  adapted-white identity (Oklab under F2 ≡ Oklab of adapted XYZ).
  Needle R4 batch re-run per new quantity (chain-rule vs FD sweep).
- **P3 — on demand (unversioned, each its own patch+twin):** sRGB / Luv /
  XYZ-raw / DIN99-coords (all kernels in-tree); whiteness/yellowness
  (~20-line kernels first); dominant-wavelength+purity (2-vector ref +
  purple-line branch rule); opacity (two-spectrum demand — architecture
  decision first, §D10).

Each patch: implement → gates (R0) → twin batch → commit/push dev →
ff to main. Parser prerequisite shipped 0.4.22 (all 97 files bitwise).

## D10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| ΔE₀₀ singular at ΔE=0 / neutral LCh angle | fixed relative FD step; twin asserts finite grad; merit unaffected (value exact) |
| CMF linear-resample error | CMFs smooth; resample error ≪ solver grid error; xyY-HEX twin bounds it |
| Table/Python drift | CI byte-compare fails loud, not silent |
| CC-BY-SA data in crate | NOTICE attribution; data files isolated in `data/` |
| Hot-pass regression | nonzero-skip guards + bitwise grad-free test + bench |
| Hue wrap (LCh Channels) | wrap before scaling; dedicated wrap-twin |
| Oklab under non-D65 | Bradford-adapt to D65 (in-tree `func_08`), identity-tested |
| Scalar-vs-triple ref confusion | untagged enum + quantity-gated validation, both directions tested |
| Scope creep (sRGB/Luv/whiteness/opacity/boxes) | refused-with-message at compile; P3 per-item with own twin |
