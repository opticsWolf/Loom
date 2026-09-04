# SpectralWeave target kinds and the needle fold

Target semantics live in two Rust mirrors that must stay in lockstep:

- `crates/navette-spectralweave/src/targetweaver.rs` — ingestion
  (`TargetKind`, `TargetEntry`, `register_metadata`) and
- `crates/navette-py/src/spectralweave_target.rs` — `calculate_merit`.
- `crates/navette-smatrix/src/synthesis/merit.rs` — `ConstraintKind`,
  `MeritTarget`, `MeritSpec`: a verbatim lift of the merit kernel into
  residual space for the thickness optimizer and the needle pass.
- `crates/navette-smatrix/src/synthesis/needle_pass.rs` —
  `build_needle_targets`: folds a `MeritSpec` into flat
  `(targets_r, weights_r)` quadratics for the analytic needle operator.

Python surface: `src/navette/spectralweave/target.py`
(`SpectralTarget`, `AngularTarget`, `TargetCollection`).

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
