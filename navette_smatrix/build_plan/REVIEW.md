# Loom Matrix — Rust Review, Optimization & Verification

## 1. Summary

The Rust port is numerically faithful to the Numba reference once three issues are
fixed (one of them a real correctness bug). After restructuring the hot path it now
matches Python to **~1e-13** across every scenario and runs **~1.27× faster
single-core** on the dual-polarization paths, with true multi-core scaling now
available (it wasn't before — see §4).

---

## 2. Issues found in the original Rust

### 2.1 Correctness bug — transmission cross-term used the wrong field *(now fixed)*

In `core_engine_rigorous_ellipsometry`, the transmission Stokes cross-term was
accumulated from the **reflection-front** field instead of the **forward-transmission**
field:

```rust
// before — rs_f / rp_f are the REFLECTION-front amplitudes (tuple element 0)
cross_t_acc = cross_t_acc * (rp_f * rs_f.conj());
```

`BlockResult` is ordered `(r_front, t_back, t_fwd, r_back, R_front, T_back, T_fwd, R_back)`,
so the transmission cross-term must use element **2** (`t_fwd`), not element 0:

```rust
// after
let (rs_f, _, ts_f, _, ..) = s_res;
let (rp_f, _, tp_f, _, ..) = p_res;
cross_t_acc *= tp_f * ts_f.conj();
```

Symptom before the fix: `Delta_T` was off by exactly π and `DOP_T` pinned near 1.0.
Reflection ellipsometry was correct because it (correctly) used element 0.

### 2.2 Latent slice/index mismatch

The engines sliced the n-stack starting at `current_idx` but then passed **absolute**
indices into a solver that indexed `n_slice[start_idx]`. This only worked because the
test stack kept `current_idx == 0` (no incoherent layers before a coherent block). Any
stack with a leading incoherent layer would have indexed out of the slice. Fixed by
passing the **full per-wavelength slice** plus absolute indices into the inner solver —
which also matches Numba's indexing semantics exactly.

### 2.3 Performance bug — per-block NumPy allocation in the inner loop

This was the main reason the Rust was *slower* than Numba. For every block, every
polarization, at every (angle, wavelength) point, the engine built four NumPy arrays and
called the `#[pyfunction]` solver across the Python boundary:

```rust
let n_arr  = PyArray::from_vec(py, n_block);   // 4 heap+Python allocs
let d_arr  = PyArray::from_vec(py, d_block);   // per block, per pol,
let rv_arr = PyArray::from_vec(py, rv_block);  // per point …
let rt_arr = PyArray::from_vec(py, rt_block);
let (...) = solve_coherent_block_fields(/* PyReadonlyArray args */)?;
```

The GIL was held the entire time, so rayon (already a dependency) could never help.

---

## 3. Optimizations applied

**Inner / wrapper split.** Every function is now a pure-Rust `*_inner` (marked
`#[inline]`, no PyO3, no NumPy, operates on plain slices) plus a thin `#[pyfunction]`
wrapper that preserves the exact original Python API. The engines call the inner
functions directly — zero allocation and zero Python-boundary crossing on the hot path.

**Dual-polarization solver.** `solve_coherent_block_fields_dual` solves s and p in a
single interface sweep, computing the polarization-independent quantities once instead
of twice: branch-safe cos θ, the propagation phase φ, and the roughness form factors
(`w_function` calls). It is arithmetically identical to two single-pol calls. The
ellipsometry engine always uses it; the photometry engine uses it for unpolarized mode
and the single-pol solver for explicit `s`/`p` requests.

**Single-pol kernel specialization.** The explicit `s`/`p` path can't share work across
polarizations (there's only one), so the gap to Numba's `fastmath` there was closed
with portable kernel work instead:

- *Monomorphization.* `solve_pol_specialized<const IS_S: bool>` resolves the s/p
  admittance choice at compile time; each instantiation is a branch-free loop with the
  dead polarization's guard code eliminated, matching the dual solver's per-pol codegen.
- *Fast transcendentals.* `csqrt_fast` replaces `num_complex`'s polar-method `sqrt`
  (an `atan2` plus a `sin`/`cos` for every cos θ) with a direct algebraic principal
  root from `|z|` and two real `sqrt`s; `cexp_fast` computes the phase via a single
  `sin_cos`. Same principal branch, ~1 ULP difference, and both help every path.
- *Reciprocal precompute.* `1/n_l` is computed once per wavelength and reused across all
  angles, turning the per-interface complex division `nsin_fi / n_l` into a multiply.
  This is the dual solver's amortize-shared-work idea applied across angles rather than
  polarizations, and it speeds up the ellipsometry and unpolarized paths too.

**FMA was tried and rejected.** Contracting the complex products with `f64::mul_add`
*regressed* the single-pol path to ~0.71× on a portable build: without a hardware-FMA
target feature, `mul_add` falls back to the correctly-rounded *software* `fma()`, which
is far slower than a plain multiply-add. Numba's `fastmath` uses FMA *contraction*
(hardware if present, plain mul+add otherwise) and never pays that cost, so this is not
portably matchable in stable Rust.

**Real parallelism.** The per-point loop is now
`(0..total_points).into_par_iter()` inside `py.detach(|| …)`, collecting into a
struct-of-arrays that is scattered into the output buffers afterward. This is the
parallel-over-points analogue of Numba's `prange`. (`Python::detach` is the pyo3 0.28
name for what was `allow_threads` in earlier versions — same GIL-releasing semantics.)

**Build profile.** Added `codegen-units = 1` so LTO can fully inline the inner
functions. `target-cpu=native` was tested and **rejected** — it was slower on the noisy
VM and hurts binary portability.

---

## 4. Results

### Parity (vs. Numba reference)

All four scenarios pass. Worst observed max-abs error is `Delta_R ≈ 4.6e-13`; everything
else is `< 1e-13`. The residual is consistent with Numba's `fastmath` reorderings vs.
Rust's strict IEEE arithmetic — not an algorithmic difference.

| scenario          | what it exercises                                    |
|-------------------|------------------------------------------------------|
| `bragg_coherent`  | 10-pair quarter-wave Bragg mirror (coherent stack)   |
| `incoherent_slab` | thick incoherent glass → intensity-Redheffer path    |
| `roughness_sweep` | all roughness form-factor types 0–5                  |
| `bare_brewster`   | bare interface near Brewster → Δ-convention check    |

### Benchmarks

> **Important caveat:** the build/verification sandbox has **only 1 CPU**, so neither
> rayon nor Numba's `prange` can show any parallel benefit here. The numbers below are
> single-core. On your multi-core machine the rayon engines should scale roughly with
> core count, which is where the bigger win is.

Single-core, 21-layer × 2-pol stacks (stable min-of-windows):

| path                   | speedup vs Numba |
|------------------------|------------------|
| ellipsometry           | ~1.26–1.29×      |
| photometry, unpolarized| ~1.28–1.30×      |
| photometry, s only     | ~0.93–0.94×      |

The single-pol `s`/`p` path was brought from ~0.88× to ~0.94× by the kernel
specialization above (monomorphization, fast transcendentals, reciprocal precompute) —
all without touching the already-verified dual-pol numerics. The residual ~6% is
Numba's `fastmath` (approximate/SVML transcendentals and reassociation that shortens
the Redheffer dependency chain) vs. Rust's strict IEEE, which is not closable portably
in stable Rust. If you build for a fixed target and want that last bit, enabling
`target-cpu=native` (and/or `+fma`) on your own hardware lets LLVM use hardware FMA and
shorter math — opt-in, since it sacrifices binary portability and was therefore left out
of the default profile. The unpolarized and ellipsometry paths — the expensive ones —
remain the bigger win even single-core, and scale further with rayon across cores.

---

## 5. Build & test

```bash
# build the extension module
maturin develop --release          # or: maturin build --release

# full parity + benchmark report
python3 test_loom.py

# or as a pytest suite
pytest -q test_loom.py
```

**Note on toolchain.** The delivered `Cargo.toml` targets your original
edition 2024 / pyo3 0.28 / numpy 0.28. Local verification here used pyo3 0.25 +
numpy 0.25 + edition 2021 only because the sandbox shipped an old `rustc` (1.75, which
predates edition 2024). Almost all API idioms are identical across both versions
(`PyTuple::new(py, vec)?`, `PyArray::from_vec(py, vec).reshape(shape)?`,
`&Bound<PyModule>`). The one 0.28-only spelling is GIL release: pyo3 0.28 renamed
`Python::allow_threads` to `Python::detach` (same semantics), which the engines in
`func_4.rs` / `func_5.rs` use. If you ever build against pyo3 ≤ 0.27, change those two
`py.detach(|| …)` calls back to `py.allow_threads(|| …)`.
