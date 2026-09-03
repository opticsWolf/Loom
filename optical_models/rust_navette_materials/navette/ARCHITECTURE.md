# Loom — Rust core + thin Python wrapper: architecture

## TL;DR recommendation

Split into **three layers**:

1. **`loom-core`** — a pure Rust crate. No Python, no I/O. Holds every numerical kernel you currently write as `@njit` functions, exposed as plain functions over array views, plus a `Dispersion` trait + `Model` enum so a whole material can be evaluated in one call. Independently unit-testable and benchmarkable.
2. **`loom-py`** — a PyO3 binding crate (`loom._core` at runtime). Receives NumPy arrays zero-/low-copy, releases the GIL during compute, calls `loom-core`, returns NumPy arrays. Contains *no math*.
3. **`loom/`** — the thin Python package. Keeps every class you already have (`Lorentz`, `Cauchy`, `EffectiveMaterial`, …) and *all* the ergonomic machinery (param dicts, flat-key `__getattr__`/`__setattr__`, `_sync()`, validation, config loading, `self.nk` caching). The only change inside each class is that `complex_refractive_index()` calls `loom._core.*` instead of the local `@njit` kernel.

Why this shape: your numerics are already pure functions; the classes are pure orchestration. Porting the kernels is mechanical and high-value (they run inside fitting loops); porting the param/dict/config sugar would be a lot of work in Rust's worst ergonomic area for zero performance gain. Keep that in Python.

Build with **maturin**; ship **abi3** wheels.

---

## The boundary: what moves, what stays

| Concern | Today | Target | Rationale |
|---|---|---|---|
| Dispersion kernels (Cauchy, Sellmeier, Lorentz, Drude(-Lorentz), Forouhi-Bloomer) | `@njit` free functions | **Rust** (`loom-core`) | Hot, pure, called per fit-iteration |
| Cody-Lorentz / UBF ε₂ + FFT-KK + interp | `@njit` + `np.fft` | **Rust** (`rustfft`/`realfft`) | Hottest path, O(N log N), worth it |
| EMA mixing kernels (Bruggeman, MG, Looyenga, …) | `@njit` (+`prange`) | **Rust** (`rayon`) | Hot, pure |
| `wavl_generator` | `@njit` | **Rust** (or leave; it's cheap) | Trivial; port for completeness |
| Param dict / flat keys / `_sync()` | Python | **Python** (unchanged) | Orchestration, not hot |
| `__getattr__`/`__setattr__`, validation, config load | Python | **Python** (unchanged) | Python's strength, good error messages |
| `self.nk` caching, `set_wavelength_range` | Python | **Python** (unchanged) | Cache key logic is policy, not math |
| Composition tree (`EffectiveMaterial`) | Python | **Python in v1**, optional Rust tree in v2 | See "EMA composition" below |

The contract across the boundary is deliberately narrow: **arrays of `f64` / `Complex64` in, array of `Complex64` out, scalars passed by value.** No Python objects cross into Rust, no Rust state is held across calls (except internal FFT planners).

---

## Repository layout

```
loom/                          # git root
├── Cargo.toml                 # [workspace] members = ["loom-core", "loom-py"]
├── pyproject.toml             # maturin backend, points at loom-py
├── rust/
│   ├── loom-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs         # re-exports, Dispersion trait, Model enum
│   │       ├── units.rs       # eV<->nm<->µm conversions, HC constant
│   │       ├── cauchy.rs
│   │       ├── sellmeier.rs
│   │       ├── lorentz.rs
│   │       ├── drude.rs
│   │       ├── forouhi_bloomer.rs
│   │       ├── cody_lorentz.rs
│   │       ├── kk.rs          # FFT Kramers-Kronig (shared by Cody/UBF)
│   │       ├── ema.rs         # mixing kernels + Bruggeman solver
│   │       └── grid.rs        # wavelength generator
│   └── loom-py/
│       ├── Cargo.toml         # cdylib; depends on loom-core + pyo3 + numpy
│       └── src/lib.rs         # #[pymodule] loom._core, all #[pyfunction]s
└── python/
    └── loom/
        ├── __init__.py
        ├── material.py        # base class (unchanged except kernel call sites)
        ├── lorentz.py         # class kept; kernel call -> loom._core.lorentz_nk
        ├── cauchy_sellmeier.py
        ├── drudelorentz.py
        ├── forouhibloomer.py
        ├── codylorentz.py
        ├── ema_material.py
        └── _core.pyi          # type stubs for the compiled extension
```

Keep `loom-core` Python-free on purpose: it stays a normal Rust library you can `cargo test`, `cargo bench`, and even reuse from other Rust tools, with the binding crate as the only thing that knows about Python.

---

## The core contract (`loom-core`)

Two complementary surfaces. Use whichever the call site wants.

**(a) Free functions** — a 1:1 replacement for each `@njit` kernel. This is what the v1 Python wrapper calls.

```rust
// lorentz.rs
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_complex::Complex64;

/// Lorentz oscillator complex refractive index n + ik.
/// `osc` is row-major (N_osc, 3): columns (E0, Gamma, f0). Inputs assumed validated.
pub fn lorentz_nk(
    energy: ArrayView1<f64>,
    osc: ArrayView2<f64>,
    eps_inf: f64,
) -> Array1<Complex64> {
    energy.mapv(|e| {
        let e_sq = e * e;
        let mut eps = Complex64::new(eps_inf, 0.0);
        for row in osc.rows() {
            let (e0, gamma, f0) = (row[0], row[1], row[2]);
            let e0_sq = e0 * e0;
            // (e0^2 - e^2) - i*(e*gamma)
            let denom = Complex64::new(e0_sq - e_sq, -e * gamma);
            eps += f0 * e0_sq / denom;
        }
        eps.sqrt()
    })
}
```

**(b) A trait + enum** — lets a whole material (including composites) evaluate in one call, which is what you want for a Rust-side fitting loop later.

```rust
// lib.rs
pub trait Dispersion {
    /// n + ik at photon energies (eV).
    fn nk(&self, energy: ArrayView1<f64>) -> Array1<Complex64>;
}

pub enum Model {
    Cauchy { a: f64, b: f64, c: f64 },
    Sellmeier { b: [f64; 3], c: [f64; 3] },
    Lorentz { osc: Array2<f64>, eps_inf: f64 },
    DrudeLorentz { omega_p: f64, gamma: f64, eps_inf: f64, osc: Array2<f64> },
    CodyLorentz { eg: f64, et: f64, eu: f64, eps_inf: f64, osc: Array2<f64> },
    ForouhiBloomer { /* ... */ },
    Effective {                       // <-- composition, v2
        host: Box<Model>,
        inclusion: Box<Model>,
        fraction: f64,
        rule: MixRule,
    },
}

impl Dispersion for Model { /* match self { ... } */ }
```

In v1 the Python wrapper only touches surface (a). Surface (b) is the seam you grow into for v2 (see migration plan); building it now costs almost nothing because each arm just calls the corresponding free function.

---

## Worked example: Lorentz end-to-end

This is the template you replicate for every model. Three files change.

### 1. Rust core — `rust/loom-core/src/lorentz.rs`
(the function shown above)

### 2. PyO3 binding — `rust/loom-py/src/lib.rs`

```rust
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use num_complex::Complex64;
use pyo3::prelude::*;

#[pyfunction]
fn lorentz_nk<'py>(
    py: Python<'py>,
    energy: PyReadonlyArray1<'py, f64>,
    osc: PyReadonlyArray2<'py, f64>,
    eps_inf: f64,
) -> Bound<'py, PyArray1<Complex64>> {
    // Own the inputs so the compute closure is Send and we can drop the GIL.
    let e = energy.as_array().to_owned();
    let o = osc.as_array().to_owned();
    let out = py.allow_threads(move || loom_core::lorentz_nk(e.view(), o.view(), eps_inf));
    out.into_pyarray(py)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lorentz_nk, m)?)?;
    // ... one line per kernel
    Ok(())
}
```

> **GIL/`Send` gotcha (read this once, then it's muscle memory):** an `ArrayView` borrowed from a NumPy array is *not* `Send`, because Python owns the buffer. To call `py.allow_threads` (needed so `rayon` actually parallelizes and so other Python threads aren't blocked) the closure must be `Send`, so own the data first with `.to_owned()`. For these array sizes the copy is far cheaper than the compute you're unblocking. If you later profile the copy as significant for tiny arrays, you can branch: compute inline (no `allow_threads`) below some length, release the GIL above it.

### 3. Python wrapper — `python/loom/lorentz.py`

Only the call site inside `complex_refractive_index()` changes. Everything else — `_osc_params`, `_sync()`, flat keys, `__getattr__`, validation — stays exactly as it is.

```python
# before:
# from .material import Material, compute_energy
# self.nk = compute_lorentz_complex_nk(self.E, self._lorentz_params, eps_inf)

from . import _core  # the compiled extension

def complex_refractive_index(self, wavelength=None):
    if wavelength is not None:
        self.set_wavelength_range(wavelength)
    if self.nk is None:
        self.nk = _core.lorentz_nk(
            self.E,                      # float64 (N,)  eV
            self._lorentz_params,        # float64 (N_osc, 3) — already built by _sync()
            float(self.params["epsilon_inf"]),
        )
    return self.nk
```

`_sync()` already produces the contiguous `(N_osc, 3)` float64 array you need — that array *is* the boundary payload, so the oscillator-management complexity never leaves Python.

---

## The hard parts (and how each is handled)

### Oscillator parameter sync — stays in Python, unchanged
The three-representation dance (`_osc_params` list ↔ flat keys `E0_0,…` ↔ Numba array) is orchestration, runs once per mutation, and is not on the hot path. Leave it. Rust only ever receives the final `(N_osc, k)` array. This applies identically to Lorentz, Drude-Lorentz, Cody-Lorentz, and UBF.

### FFT Kramers-Kronig (Cody-Lorentz / UBF) — the one fiddly port
Today: build odd-extended buffer, `np.fft.rfft`, multiply positive freqs by `-i`, `np.fft.irfft`, slice. Port with **`realfft`** (built on `rustfft`). Two things must match NumPy exactly or golden tests fail:

- **Normalization.** NumPy `rfft` is unnormalized forward; `irfft` divides by `n`. `realfft`'s forward is also unnormalized, but its inverse is *also* unnormalized — so you must divide the inverse output by `_KK_M` yourself.
- **The Hilbert multiplier.** `_KK_H[0] = 0`, `_KK_H[1:] = -i` maps directly: zero the DC bin of the half-spectrum, multiply every other bin by `Complex64::new(0.0, -1.0)`.

Cache the planners — build `_E_FULL`, `_KK_M`, forward/inverse plans once in a struct stored behind `OnceLock<KkPlan>` (the Rust equivalent of your module-global precompute). Sketch:

```rust
// kk.rs
pub struct KkPlan { fwd: Arc<dyn RealToComplex<f64>>, inv: Arc<dyn ComplexToReal<f64>>, m: usize }

pub fn kk_fft(plan: &KkPlan, eps2: &[f64], eps_inf: f64, n: usize) -> Vec<f64> {
    let mut buf = vec![0.0f64; plan.m];
    buf[n + 1..n + 1 + n].copy_from_slice(eps2);
    for i in 0..n { buf[1 + i] = -eps2[n - 1 - i]; }   // -eps2[::-1]
    let mut spec = plan.fwd.make_output_vec();
    plan.fwd.process(&mut buf, &mut spec).unwrap();
    spec[0] = Complex64::new(0.0, 0.0);
    for b in spec[1..].iter_mut() { *b *= Complex64::new(0.0, -1.0); }
    let mut hilb = plan.inv.make_output_vec();
    plan.inv.process(&mut spec, &mut hilb).unwrap();
    let scale = 1.0 / plan.m as f64;                    // numpy irfft 1/n normalization
    (n..n + n).map(|i| eps_inf - hilb[i] * scale).collect()
}
```

Then `interp` to the target grid (a small monotonic `np.interp` equivalent — write a 20-line linear interpolator, or use the `interp1d`-style helper in `ndarray`-adjacent crates). Keep the "target energies exceed KK grid" guard and surface it as `PyValueError` from the binding.

### Bruggeman — per-element Newton-Raphson → `rayon`
Your `prange` over elements, each running a complex Newton iteration, maps cleanly:

```rust
use rayon::prelude::*;
eps_eff.par_iter_mut().zip(eps_i.par_iter()).zip(eps_h.par_iter())
    .for_each(|((out, &ei), &eh)| {
        let mut e = (ei + eh) * 0.5;
        for _ in 0..max_iter {
            // F, F' exactly as in the @njit body
            let delta = -f_total / (df + Complex64::new(1e-15, 0.0));
            e += delta;
            if delta.norm_sqr() < tol * tol { break; }
        }
        *out = e;
    });
```

The analytic mixers (Maxwell-Garnett, Looyenga, Lichtenecker, Mori-Tanaka, power law, Wiener/HS bounds) are element-wise `mapv` / `Zip` — trivial.

### EMA composition — Python in v1, optional Rust tree in v2
`EffectiveMaterial.complex_refractive_index()` warms the two children's caches, grabs `host.nk` / `inclusion.nk`, calls the mixer, takes `sqrt`. **In v1, keep this orchestration in Python**: it's two kernel calls + one mixer call, each crossing the boundary once for a *whole array*, so the per-crossing overhead is amortized across hundreds/thousands of wavelengths and is negligible. Concretely:

```python
def complex_refractive_index(self, wavelength=None):
    if wavelength is not None:
        self.set_wavelength_range(wavelength)
    n_h = self.host.nk          # warmed by set_wavelength_range
    n_i = self.inclusion.nk
    eps_eff = _core.ema_mix(self.model_name, n_i, n_h, self.fraction, self.model_args)
    self.nk = _core.csqrt(eps_eff)   # or fold sqrt into ema_mix to save a crossing
    return self.nk
```

Fold the final `sqrt` into the mixer (return n̂ instead of ε) to drop one crossing — that mirrors what `_parallel_sqrt` does today.

**v2 (only if a profiler says the per-evaluation crossings matter in deep fits):** add the `Model::Effective` arm and a constructor that builds the tree once in Rust from a serialized spec, so an entire stack evaluates in a single call with no Python round-trips per fit-iteration. Don't build this until measurements justify it.

### Units / energy conversion
Models split between energy-native (Lorentz, Drude, Cody — eV) and wavelength-native (Cauchy/Sellmeier — µm²; Urbach also wants nm and m). Two options:

- **Keep the wrapper preparing arrays** (`self.E`, `wavelength_µm_2`, `wavelength_m`) exactly as today and pass them in. Minimal change, maximum parity. Recommended for v1.
- **Pass raw wavelength (nm) and convert in `units.rs`.** Narrower boundary, but changes call sites. Migrate to this in v2 if you want the wrapper thinner.

Put `_HC_EV_NM = 1239.8419843320028` in `units.rs` as the single source of truth so the constant can't drift between modules.

### `fastmath`
Your kernels use `fastmath=True`. Rust has no global fastmath switch and stable Rust won't reorder FP ops for you. **Port without it first** and get bit-for-tolerance parity, then, only if a kernel is FP-bound, hand-optimize (FMA via `f64::mul_add`, precomputed reciprocals). Don't reach for `-ffast-math`-style nightly intrinsics — they'll silently break your golden tests for marginal gains.

### Parallelism mapping
`prange` → `rayon` `par_iter`/`par_iter_mut`. Always wrap the compute in `py.allow_threads` in the binding so the GIL is released while rayon runs. Note that for small arrays rayon's overhead can lose to serial; gate parallelism on length if you see it (`if n > THRESHOLD { par } else { serial }`).

---

## Numerical parity strategy (do this before porting anything)

A numerical rewrite lives or dies on regression tests. Lock the current behavior *first*:

1. **Snapshot the current Python outputs.** Write a script that, for every model, evaluates a battery of `(params, wavelength_grid)` cases — including edge cases: single wavelength, energies near `E=0`, `B3=0` Sellmeier branch, Cody energies straddling `Et`/`Eg`, metal/dielectric pairs for every EMA rule, fractions `{0.1,…,0.9}` — and saves inputs+outputs to `tests/golden/*.npz`.
2. **Assert in Rust and in Python.** `loom-core` gets `#[test]`s that load the `.npz` (via `ndarray-npy`) and check `assert_abs_diff_eq!(rust, golden, epsilon = ...)`. The Python test suite checks `_core.*` against the same goldens with `np.testing.assert_allclose(rtol=1e-10, atol=1e-12)` (loosen only where `fastmath`/FFT genuinely demands).
3. **Watch the two known-sensitive kernels:** FFT-KK (normalization, odd-extension indexing) and Bruggeman (Newton convergence/branch). Give them their own dense golden sets.
4. **Keep the old `@njit` path importable during migration** behind a flag (`LOOM_BACKEND=numba|rust`) so you can A/B and bisect any discrepancy on real workloads.

---

## Build & packaging

`pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "loom"
requires-python = ">=3.9"

[tool.maturin]
manifest-path = "rust/loom-py/Cargo.toml"
python-source = "python"          # ships the python/loom package
module-name = "loom._core"
features = ["pyo3/abi3-py39"]     # one wheel for 3.9+
```

`rust/loom-py/Cargo.toml` deps: `pyo3` (abi3), `numpy`, `num-complex`, `loom-core` (path). `loom-core` deps: `ndarray`, `num-complex`, `rayon`, `realfft`/`rustfft`; dev-deps `ndarray-npy`, `approx`.

Dev loop: `maturin develop --release` builds and installs into the active venv so `import loom` picks up the compiled `_core`. Release: `maturin build --release` (add `cibuildwheel` later for manylinux/macos/windows matrices). Ship `_core.pyi` stubs so editors keep autocompletion across the boundary.

---

## Phased migration plan

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 | Stand up the workspace, maturin, CI; port `compute_energy` + `wavl_generator` as a trivial first kernel to prove the toolchain end-to-end | `import loom._core` works; one kernel matches golden |
| 1 | Golden-snapshot harness for *all* models from the current Numba code | `.npz` goldens committed; numba path passes its own goldens |
| 2 | Port the analytic kernels: Cauchy, Sellmeier (+Urbach), Lorentz, Drude(-Lorentz), Forouhi-Bloomer | each matches golden within tol; wrapper classes switched |
| 3 | Port EMA mixers + Bruggeman solver; keep composition in Python | EMA goldens pass; `EffectiveMaterial` unchanged externally |
| 4 | Port FFT-KK + Cody-Lorentz/UBF | KK goldens pass at agreed tol |
| 5 | Delete the Numba kernels; drop the `LOOM_BACKEND` flag | numba removed from deps; benchmarks recorded |
| 6 (optional) | `Model::Effective` Rust tree + Rust-side fit loop | only if profiling justifies it |

Each phase ships independently — the `LOOM_BACKEND` flag means you're never in a broken half-ported state.

---

## Decisions to confirm before phase 0

These change a few details above; worth nailing down up front:

- **Do fits run in Python (scipy.optimize, as in your usage doc) or do you want the fit loop itself in Rust?** If Python-only, v1 (free functions) is the whole story and v2 is unnecessary. If you envision Rust-side fitting, build the `Model` enum from phase 2 so the seam exists.
- **Acceptable tolerance for the FFT-KK path** — bit-exact is unrealistic across FFT libraries; pick an `rtol` (1e-9 is usually achievable) so phase 4 has a clear pass/fail.
- **Target platforms / Python versions** for wheels (drives the abi3 floor and the cibuildwheel matrix).
- **Whether to keep wavelength-array prep in Python (v1, recommended) or move unit conversion into Rust** — affects how thin "thin" actually is.
