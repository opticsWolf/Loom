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
pub enum ColorQuantity { Lab, XyY }                    // LCh/sRGB/Oklab later
pub enum ColorDistance { DeltaE2000, DeltaE76, Channels }
pub struct ColorDemand {
  pub key_idx: u32,            // normal MeritKey (angle+curve) — §3
  pub cmf: Vec<[f64; 3]>,      // NATIVE table grid (resampled at eval, §D2.4)
  pub cmf_wl: Vec<f64>,
  pub illuminant: Vec<f64>,
  pub illum_wl: Vec<f64>,
  pub quantity: ColorQuantity,
  pub reference: [f64; 3],
  pub distance: ColorDistance,
  pub weight: f64,
}
```

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
   observer (name|cmf-table), quantity, reference [3], distance, weight}`.
   Names resolved: `"D65"` → embedded default; `"1931_2deg"` → embedded
   default; anything else must be an explicit `{wavelengths, values}`
   table (mirrors the explicit-grid provider rule — no registry to drift).
2. `compile_merit_spec` color arm (no weaver involvement — color carries
   its own tables, not `TargetEntry` curves):
   - curve ∉ {Rs,Rp,Ru,Ts,Tp,Tu} → refuse (`"...: color needs a front
     R/T intensity curve, got {curve}."`); back curves refused in v1
     (`"...: back-incidence color is not supported yet."`).
   - quantity/distance strings unknown → refuse; `reference.len() != 3`
     → refuse; weight via `check_weight`; kind ≠ Exact → refuse;
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
   divergence, tested).
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
  `quantity="Lab"`, `reference=(…)`, `distance="DeltaE2000"`, `weight=1.0`)
  with `_dump()` (names resolved to arrays from `data/CIE/`, or explicit
  arrays passed through); `TargetCollection.color_targets`;
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

## D9. Sequencing (one feature per patch)
- **0.4.22 (D1+D2)** — data + kernel + R1 tests.
- **0.4.23 (D3+D4)** — schema/compile/merit + R2/R3 tests.
- **0.4.24 (D5)** — Option B machinery + R4 tests.
- **0.4.25 (D6+D7)** — Python surface + docs + R5 tests.
Each: implement → gates (R0) → twin batch → commit/push dev → ff to main.

## D10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| ΔE₀₀ singular at ΔE=0 / neutral LCh angle | fixed relative FD step; twin asserts finite grad; merit unaffected (value exact) |
| CMF linear-resample error | CMFs smooth; resample error ≪ solver grid error; xyY-HEX twin bounds it |
| Table/Python drift | CI byte-compare fails loud, not silent |
| CC-BY-SA data in crate | NOTICE attribution; data files isolated in `data/` |
| Hot-pass regression | nonzero-skip guards + bitwise grad-free test + bench |
| Scope creep (LCh/sRGB/back/boxes) | refused-with-message at compile; each a future patch with its own twin |
