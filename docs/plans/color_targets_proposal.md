# Proposal: color values as targets + needle drivers

**Status:** proposal (not implemented). Goal: optimize thin-film designs
toward colorimetric goals — e.g. "match Lab (62, 18, −34) under D65" or
"minimize ΔE₀₀ to reference white" — with both the scalar merit and the
needle algorithm driven by the same color residual.

## 1. Key insight: no new EM kernel is needed

A color merit `F(R(λ))` is a scalar function of the simulated spectrum.
Its needle gradient follows the **chain rule, exact at the operating
point**:

```
P_color(z) = Σ_λ [ ∂F/∂R(λ) ] · P_R(λ, z)
```

`P_R(λ,z)` is the existing per-wavelength reflectance needle channel.
So color folds into the *existing* R/T bucket machinery as per-wavelength
weights — the same loss class as the current integral-demand fold, but
with non-uniform weights. The needle *engine* (`needle_operator.rs`,
`NREQ_*` flags, the hot pass) is **untouched**; only the fold
(`build_needle_targets`) and the merit residuals gain a color arm.

## 2. New pieces (all Rust-first, Python thins over)

### C-a. Color evaluation kernel (new file, ~250 lines)
`smatrix/synthesis/color_merit.rs` reusing the existing color kernels
(`func_01/02/05/09/16`, XYZ integration generalized from `func_13`):

```rust
pub struct ColorDemand {
  pub curve: CurveId,        // spectrum source: Ru default; Rs/Rp/Tu/… allowed
  pub angle: f64,            // demand angle (nearest-row semantics, as today)
  pub cmf: Vec<[f64; 3]>,    // resampled to sim grid at compile time
  pub illuminant: Vec<f64>,  // resampled to sim grid at compile time
  pub quantity: ColorQuantity,   // Lab (default) | XyY | LCh | Srgb | Oklab
  pub reference: [f64; 3],       // target color in `quantity` space
  pub distance: ColorDistance,   // DeltaE2000 (default) | DeltaE76 | Channels
  pub weight: f64,
}
pub fn eval_color(demand, sim_row, sim_wl)
  -> Result<(residual, Vec<f64> /* dF/dR per λ */), String>
```

Gradient strategy (exact where cheap, FD where not):
- `dXYZ/dR(λ) = E(λ)·cmf(λ)·k·Δλ` — **analytic, exact** (integration is linear).
- color-space map + ΔE: **3-point finite difference over the XYZ
  3-vector** (3 extra Lab/ΔE evals — trivial cost), composed with the
  analytic Jacobian. No per-λ bumping (`nw` evals saved); bitwise-exact
  for XYZ/linear quantities, FD-tolerance only through Lab/ΔE.
- Resampling of CMF/illuminant onto the sim grid happens **once at
  compile** (`compile_merit_spec`), linear interp (CMFs are smooth;
  refuse non-overlap like other demands).

### C-b. Target schema (native-owned, as per R3/R4)
- `TargetSet` JSON gains a `"color": [...]` section next to
  `"spectral"`/`"angular"`; `compile_merit_spec` validates (unknown
  quantity/distance refused, reference arity, grid overlap, `Exact`
  only in v1 — see §5).
- Python: `ColorTarget` dataclass (`spectralweave/target.py`) with
  `_dump()` + `TargetCollection.color_targets`; `validate_targets`
  arm natively. Illuminant/observer given as **names** (`"D65"`,
  `"F2"`, `"1931_2deg"`…) resolved by Python from the bundled
  `src/navette/data/CIE/{cmf,sds}` JSONs into arrays, **or** as
  explicit arrays (mirrors the explicit-grid provider rule).

### C-c. Scalar merit (`merit.rs` arm)
`residuals_into` gains a color branch: one residual per demand
(`√w · ΔE`, or per-channel Lab residuals under `Channels`). Key-group
accounting (`n_residuals`) counts 1 per demand, not `nw` — keeps the
thickness optimizer's fixed-length residual vector intact, so LM and
thick-opt work with color for free.

### C-d. Needle fold (`needle_pass.rs` arm)
At the operating point, with `g(λ) = 2·w·residual·∂F/∂R(λ)`:
- **Option A (recommended, fold-only):** deposit pseudo-demands into
  the existing R (or T) bucket: `w'(λ) = max(|g(λ)|/2, floor)`,
  `target'(λ) = sim(λ) − g(λ)/(2·w'(λ))`. The pass then forms exactly
  `Σ g(λ)·P(λ,z)` with **zero hot-pass changes**.
- Option B: a dedicated gradient bucket carried into the pass.
  Honest but touches the hot loop for no v1 gain. Defer unless
  profiling or the LM path needs it (it doesn't — it uses residuals).

## 3. Data flow (Rust core does no I/O)

| Table | Source | Handover |
|---|---|---|
| CMF 1931-2° / 1964-10° | `data/CIE/cmf/*.json` | Python reads → arrays in target JSON |
| Illuminants D65/D55/D75/C/FL… | `data/CIE/sds/*.json` | same |
| Standalone-Rust default | — | **decision:** embed D65+1931-2° via `include_str!` (copy one JSON into `rust/navette`) or require explicit tables |

## 4. What falls out for free
- **Metamerism design**: N color demands, same spectrum, different
  illuminants — no extra machinery.
- **Gonio-apparent color**: one demand per angle (existing multi-angle
  semantics).
- **Transmissive color**: same arm with `curve = Tu`, T bucket.
- **Thickness optimizer + LM**: work via `residuals()` unchanged.
- **Self-twinning**: reference values computed with the *existing*
  `navette._color` bindings (Python) vs the synthesis path (Rust) —
  HEX parity for XYZ, FD-tolerance for ΔE paths.

## 5. Decided (2026-09-06)
- Quantities/distances: **both** `Lab | XyY` × `DeltaE2000 | DeltaE76 |
  Channels` (xyY+Channels is fully analytic/bitwise-twinable;
  Lab+ΔE paths twin at FD tolerance).
- Defaults **D65 + 1931 2°**, every field independently overridable.
- **Reflectance + transmittance in v1** (fold deposits into the
  r- or t-bucket by curve; back-side RB/TB allowed, not default).

## 5b. Deliberate v1 limits
- `Exact` (ΔE) demands only. `Above`/`Below`/boxes on ΔE are
  geometrically odd (ΔE is a distance — one-sided activation needs an
  operating-point convention we should design, not improvise).
- No new `NREQ_*` flag, no engine change, no Python math.

## 6. Work plan (one feature per patch, per convention)
- **C1** — `color_merit.rs` + tests (XYZ bitwise vs `spectral_to_srgb`
  generalized; Lab/ΔE vs `_color` twins; gradient FD cross-check).
- **C2** — `TargetSet` color section + `compile_merit_spec` arm + merit
  residuals arm + twin tests.
- **C3** — needle-fold arm (Option A) + needle twin (chain-rule vs
  brute-force FD needle on a 2-layer stack).
- **C4** — Python `ColorTarget` + validators + docs + exposure lint
  stays green.
- Estimated total: ~600 Rust lines + ~150 Python lines + tests.

## 7. Decisions needed
4. Option A (pseudo-demand fold, recommended) vs B (gradient bucket)?
   — see analysis 2026-09-06 in chat; numerically identical at the op
   point, differ in honesty/extensibility only.
5. Embed default CIE tables (D65 + 1931-2°, ~50 KB) in the Rust crate
   with a CI byte-compare against `src/navette/data/CIE/` (recommended)
   vs explicit-tables-only for standalone Rust?
