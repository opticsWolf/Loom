# SpectralWeave target kinds and the needle fold

Target semantics live in two Rust mirrors that must stay in lockstep:

- `rust/navette/src/spectralweave/targetweaver.rs` — ingestion
  (`TargetKind`, `TargetEntry`, `register_metadata`) and
- `rust/navette-py/src/spectralweave_target.rs` — `calculate_merit`.
- `rust/navette/src/smatrix/synthesis/merit.rs` — `ConstraintKind`,
  `MeritTarget`, `MeritSpec`: a verbatim lift of the merit kernel into
  residual space for the thickness optimizer and the needle pass.
- `rust/navette/src/smatrix/synthesis/needle_pass.rs` —
  `build_needle_targets`: folds a `MeritSpec` into flat
  `(targets_r, weights_r)` quadratics for the analytic needle operator.

Python surface: `src/navette/spectralweave/target.py`
(`SpectralTarget`, `AngularTarget`, `TargetCollection`).

## Normalization strategy (verified)

Each target curve is self-normalized at ingestion so heterogeneous
quantities (R ~ 1, weak T ~ 0.01, phases in radians) contribute comparably:
values are scaled to O(1), the sim is scaled identically, and the residual
is divided by the **raw** tolerance. Consequences:

| mode | norm_factor | scaled values | tolerance means |
|---|---|---|---|
| `linear` | $1/\|mean\|$ | raw·nf | **fraction of the curve mean** |
| `log` | $1/mean(\|log_{10}\|)$ | log₁₀·nf | fraction of the mean log magnitude |
| `phase` | $1$ | raw (radians) | absolute radians |
| `complex` | $1$ | raw (no normalization) | raw units |
| `auto` | log iff all-positive and max/min ≥ 100×, else linear | — | — |

So `tolerances=[0.05]` on an R ~ 0.5 curve is an effective absolute window
of 0.025 (5% of the mean) — tolerances are **relative to the curve level**,
not absolute, except in phase/complex modes. The same triple
(normalized, nf, floored tolerances) feeds `calculate_merit`, `MeritSpec`
and the fold, so all three agree by construction.

Resolution scope and guards:

- Normalization is **per curve**: one `(mode, nf)` for a whole spectral
  curve, and — since the angular fix — one shared pair for a whole
  angular curve (previously each angle normalized itself, weighting equal
  absolute deviations differently per point).
- **Zero-mean guard** (linear): mean-normalization would explode on
  zero-mean/cancelling data (old nf hit 1e12), so when
  $|mean| \le 10^{-9}\cdot spread$ the curve scales by the half-range
  instead (exact O(1) for symmetric data); all-constant data keeps raw
  scale (nf = 1). Non-degenerate curves are bit-for-bit legacy.
- **~= 1 guard** (log): all-$|log| \approx 0$ data (values ~= 1) falls
  back to raw log scale for the same reason.
- Tolerance floor (`tolerance_floor`, default 1e-12) clamps near-zero
  tolerances so the merit can never divide by zero.
- `complex` is currently **raw-scale linear** (real f64 data, nf = 1) —
  an opt-out of normalization, not complex-number support.

## Weights, count normalization, integral targets

Merit is a sum over points — a 200-point target outweighs a 10-point one
20:1 at equal residuals (measured), and a single angular point drowns next
to dense spectral curves. Two per-target knobs control this (all engines —
weaver, spec/LM, fold/needle — apply them identically):

- `weight` (default 1): multiplies the frame's merit sum (residuals scale
  by √weight, so LM Jacobians stay consistent). Relative importance across
  targets; finite and ≥ 0 (rejected otherwise, at every trust boundary).
- `normalize_count` (default off): divides the frame's sum by the
  target-level point count (spectral grid size, angular angle count —
  resolved at ingestion, since angular targets span many single-point
  entries). Turns the sum into a mean: equal say regardless of sampling.
  `tol·√N` per frame is exactly equivalent (pinned by test).
- `integral` (default off): constrains the MEAN of the scaled diffs —
  single residual `R = mean(d)/mean(tol)` with kinds applied once to the
  mean (integral-`a` = lower bound on the average). Rejects
  `normalize_count` (the mean already is one — the combo would
double-dilute). Regular and integral targets mix freely in one run.

Frame contribution = `weight × (Σr² / count)` pointwise,
`weight × kind(R)` integral. Missing-data penalties are unaffected by
weights (drop a target to silence those — weight 0 only mutes present
data). The CenterBand `+1` accounting becomes
$M_{true} = M_{folded} + \Sigma\, weight/count$ over violated `c` points.

The integral fold matches the mean-form merit's UNIFORM gradient exactly
at the operating point (`w_i = W/N²`, `t_i = s_i − N·G` per point) —
values differ by dropped constants (same loss class as the `+1`/overlap
terms), gradients superpose exactly across overlapping integral frames,
and the PD gain-shift formula needs no changes (it lands on
$−2W(\bar m−T)\overline{k_z}$ automatically).

## Kinds (per point, merit space)

Scaled residual `d = sim_scaled − target_scaled`, floored tolerance `tol`,
scaled band half-width `bw`. `d` is phase-wrapped to $[-\pi,\pi]$ in phase
mode; in log mode both `d` and `bw` live in log-normalized space.

| kind | code | merit contribution |
|---|---|---|
| Exact | `e` | $(d/tol)^2$ |
| Above (lower bound) | `a` | $d<0 \Rightarrow (d/tol)^2$, else $0$ |
| Below (upper bound) | `b` | $d>0 \Rightarrow (d/tol)^2$, else $0$ |
| Range (hard box) | `r` | $\|d\|\le bw \Rightarrow 0$, else $((\|d\|-bw)/tol)^2$ |
| CenterBand (soft box) | `c` | $\|d\|\le bw \Rightarrow (d/bw)^2$, else $((\|d\|-bw)/tol)^2 + 1$ |

`r` is exactly the combination of paired `a`/`b` targets at centre∓band.
`c` keeps an `e`-style centre with proportionally reduced weight inside:
$(d/bw)^2 = (d/tol)^2\cdot(tol/bw)^2$. A band of $1\%$ with tolerance
$0.1\%$ pulls 100× softer inside than `e`, reaches $1.0$ at the band edge
from both sides (continuous by construction), then grows with the outer
$0.1\%$ scale outside.

## The `band` parameter

Per-point half-width in **raw units** (same units as `values`); a Python
scalar broadcasts. Scaled at ingestion by the same `norm_factor` as the
targets — exact for linear/phase/complex, per-point exact on the upward
side for log (symmetric approximation otherwise; avoid huge relative
bands on log targets). Negative bands are rejected in Python and floored
in Rust. Omitting `band`:

- `r` falls back to `tol` as the half-width (dead-band of ±tolerance);
- `c` degrades gracefully to `e`.

`MeritSpec` accepts an empty `band` array meaning all-zero (unused).

## The needle fold

The analytic needle operator understands one merit shape per spectral
point — a homogeneous quadratic $f_k = w_k\cdot(R_k - t_k)^2$ — fed via
`targets_r`/`weights_r`. `build_needle_targets` converts each `MeritSpec`
entry at the current operating point (active-set linearization,
recomputed every iteration):

| kind | condition | folded $(t_k, w_k)$ |
|---|---|---|
| `e` | always | centre, $nf^2/tol^2$ |
| `a` / `b` | violated (at current sim) | centre, $nf^2/tol^2$; satisfied → skipped |
| `r` | violated | nearest band edge, $nf^2/tol^2$; in-band → skipped |
| `c` | inside | centre, **reduced** $nf^2/bw^2$ |
| `c` | outside | nearest band edge, $nf^2/tol^2$ |
| `r` / `c` | no sim yet (first iteration) | centre, $nf^2/tol^2$ (conservative) |

Folding is per quantity: front R/T/A demands → `(targets_r, weights_r)` /
`(targets_t, weights_t)` / `(targets_a, weights_a)`; back demands → the
`rb`/`tb`/`ab` siblings; absorption (`As`/`Ap`/`Au`, `ABs`/`ABp`/`ABu`)
derives $A = 1 − R − T$ from the companion pairs — the kind table above
applies identically in every bucket. Missing companions/rows fold
conservatively (exact at centre).

Phase demands fold to one `(targets, weights)` pair per S-matrix channel
(`phi[0..=3]` → one `P_PHI` call each). Intensity/absorption demands
require linear normalization; phase demands accept linear or phase
(wrapped residuals mirror the evaluator); anything else is an `Err`.
Phase demands must carry raw values with `norm_factor == 1` (the phase
arm scales nothing — the converter unscales the resolved triple).

Spectral-label mapping for the converter (`TargetCollection` → `MeritSpec`):
`R/T/A` × s/p/u → front demands, `RB/TB/AB` × s/p/u → back demands;
`phase=True` targets become phase demands on the mapped curve's element
(R → r_front, T → t_fwd, RB → r_back, TB → t_back). Anything else raises
`ValueError`. Angular targets expand to one single-point demand per angle.

## Differential phase (`PDts`/`PDtp`)

`PDts`/`PDtp` demand the coating-induced transmitted phase: the design's
`arg(t)` minus the equivalent incidence-medium layer,

$$\Delta\varphi(\lambda) = \arg t(\lambda) - passes\cdot\frac{2\pi\,n_{inc}\,D\cos\theta_{inc}}{\lambda},$$

with `passes = 1` (single traversal; `passes = 2` covers a reflection
round trip if reflection labels are ever added), `D` the total coating
thickness and `n_inc` the real incidence index. Group delay / GDD over
$\Delta\varphi$ come for free (finite differences kill the reference
anyway); the differential form matters for absolute-phase targets and
for correct needle-gain bookkeeping.

Evaluation points (all in solver convention — see the sign note below):

- Ingestion forces phase normalization (raw radians, `nf = 1`); the
  converter maps `PDts` → (`Ts`, phase, passes 1), `PDtp` → (`Tp`, …).
  `phase=False` or a polarization mismatch raises `ValueError`.
- `SimCurves` carries `total_d`/`n_front_re`/`n_back_re` (defaults 0/1/1
  zero the reference); the thickness-optimizer evaluator fills them from
  the stack (ambient index at centre λ — dispersive ambients are
  pathological, documented approximation) — but only when the spec asks
  (`uses_phase()` gates complex-row assembly, `uses_differential()` the
  metadata; intensity-only LM loops pay zero extra allocations).
  `total_d = 0` reproduces absolute phase bit-for-bit.
- The merit Phase arm subtracts the reference before wrapping; the fold
  passes PD demands to the `phi` buckets unchanged (same channel as the
  absolute element) and additionally accumulates the exact `dM/dD`
  correction `phi_gain_shift[ch] = Σ −2·kz·w·(s−rt)` (`kz =
  passes·reference_wavenumber`). Subtract it from the assembled `P_PHI`
  (`needle_gradient(…, gain_shift_phi=…)`); it is uniform in z, so the
  needle site (`argmax`) never moves — only predicted-gain bookkeeping.
- Manual-sim recipe (zero core involvement, doubles as test oracle):
  `apply_reference_rotation` multiplies complex rows by `e^{−i·ref}`;
  an absolute-phase demand on rotated rows is exactly a differential
  demand on raw rows.

Sign convention: the reference is `+kD`, matching this crate's
forward-propagation phase (an all-matched slab simulated by the solver
has `arg(tf) = +kD` — pinned by test). That is the conjugate of
Macleod/`e^{+iωt}` textbooks; the crate is self-consistent (absolute
phase demands, `P_PHI`, GD/GDD share it), so only textbook-imported
target numbers need conjugating — never solver-produced ones.

### The dropped `+1` level

Outside the band, `calculate_merit` contributes
$((|d|-bw)/tol)^2 + 1$ while the fold carries only
$w\cdot(R-t_{edge})^2 = ((|d|-bw)/tol)^2$. The $+1$ is a pure number:
independent of $R$, thicknesses, and everything the needle can perturb.
Hence needle gradients, the $P(z)$ profile, site ranking and the best-site
pick are **exact**; only the scalar merit value reads lower by exactly
$N_{outside}$ (the count of currently-violated `c` points):

$$M_{true} = M_{folded} + N_{outside}$$

Carrying the constant would need a third per-point array through the
`needle_gradient` FFI and every call site, for a term that cannot change
any decision — so the fold drops it by design. If a future consumer needs
true values from the folded path (e.g. a trust-region acceptance test),
return the outside-count alongside `(targets, weights)`; the fold already
knows it.

### Overlap constant (same class of loss)

When several demands fold onto one solver point, completing the square
$\sum w_i(s-t_i)^2 = W(s-\bar t)^2 + [\sum w_i t_i^2 - W\bar t^2]$ keeps
$(\bar t, W)$ and drops the bracket. Gradients stay exact; folded merit
*values* under-read by that constant. Exact value identity holds only for
non-overlapping demands (at most one demand per solver point per bucket).
Like the $+1$, this never affects needle gradients, site ranking, or
insertion decisions — only human-readable merit comparisons.

### Multiblock (incoherent) needle

`Pmb` differentiates the cascade totals through the adjoint weights
($g[p][e] = \partial v[e]/\partial param$): R via $v[0]$, T via $v[2]$,
A via $1 − v[0] − v[2]$ with negated summed weights. Request with
`P_MB`/`P_MB_T`/`P_MB_A` (reusing the R/T/A target arrays); phase has no
multiblock path (phases only live inside coherent blocks).

### Kinks

Value-continuity holds at every band edge (both sides read $1.0$ for `c`,
$0.0$ for `r`), but the *slope* has a kink there. A needle step that
crosses an edge is evaluated with the pre-step linearization and
self-corrects on the next re-fold; persistent oscillation around an edge
means the step size is too large — damp it.

## Color demands (`ColorTarget`, v1)

One demand = front R/T intensity curve × angle × illuminant × observer ×
quantity × distance × weight. Defaults D65 + 1931 2°, all overridable;
names (`"D65"`, `"1931_2deg"`) resolve to the embedded defaults, anything
else must be an explicit table (no registry to drift).

- **Quantities/distances:** Lab|XyY × DeltaE2000|DeltaE76|Channels, with
  the compat matrix enforced at compile (XyY takes Channels only — DeltaE
  is Lab-space). Lab white is always the demand illuminant's own white
  (computed at compile from native tables, never adapted).
- **Kinds/transforms:** `Exact` + `linear` only. No `integral` /
  `count_norm` / `phase` / `band` — a color demand already is integral
  (one residual per demand, not per wavelength).
- **Merit path:** missing curve fails the whole key group (standard
  missing-penalty path, parity with pointwise); empty table/sim overlap
  errors instead of skipping (a demand that sees nothing is a spec bug).
- **Needle fold (Option B):** each demand deposits its analytic gradient
  g(point) = dF/dcurve (weight, residual, and the U-curve half included)
  into the `grad_r`/`grad_t` buckets — not a (target, weight) pair. `Ru`
  / `Tu` split half per polarization branch (Ru is the Rs/Rp mean); s/p
  demands ride the shared bucket under the same convention as pointwise
  targets. Zero points are skipped in the hot pass, so color-free sets
  are bitwise unaffected.

### Extended quantities (P2: LCh | Oklab | Y)

- **LCh:** references convert LCh→Lab once per eval; DeltaE is measured
  in Lab. Channels compare (L, C, h) with the hue difference wrapped to
  [−180, 180] *before* scaling (179 vs −179 is 2 deg, not 358).
  Near-neutral colors (C≈0) have ill-defined hue — the kernel gradient
  stays finite (FD class) but directionally unstable; weight hue
  tolerance generously near the achromatic axis.
- **Oklab:** D65-defined. Non-D65 demands Bradford-adapt XYZ→D65 first
  (same `adapt` the bindings use, unclipped so the FD gradient stays
  smooth). DeltaE on Oklab is refused — equal-tol Channels *is*
  unweighted Euclidean. Note: the in-tree D65 white constant and the
  Oklab matrix disagree ~1e-4 in b (pre-existing); systematic, negligible.
- **Y:** scalar reference, single residual off `tol[0]` (the AR classic:
  dark residual, hue free).
