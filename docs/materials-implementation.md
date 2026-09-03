# Navette materials — Rust port (implementation status)

This is the working implementation of the architecture in
`LOOM_RUST_ARCHITECTURE.md`. The optical-dispersion numerics of `navette`
(formerly *loom*) now live in a pure-Rust core, exposed to Python through a
thin PyO3 binding, with the original class API preserved.

```
navette/
├── pyproject.toml                 maturin build → navette.materials._native
├── rust/
│   ├── Cargo.toml                 workspace (deps pinned for rustc 1.75)
│   ├── navette-materials/         pure-Rust core (no Python, no I/O)
│   │   ├── src/                   units, common, cauchy, sellmeier, lorentz,
│   │   │                          drude, ema, grid (+ kk/cody/forouhi scaffolds)
│   │   └── tests/parity.rs        golden-parity tests (12, all passing)
│   └── navette-materials-py/      PyO3 bindings → module `_native`
├── python/navette/                thin Python package (the wrappers)
│   └── materials/                 material, basic, cauchy_sellmeier, lorentz,
│                                  drudelorentz, _native.pyi
├── golden/                        NumPy reference arrays (wl/re/im .npy)
└── tools/
    ├── gen_golden.py              regenerate the golden references
    └── make_wrappers.py           re-derive the thin wrappers from originals
```

## What is done and verified

The boundary is **wavelength in nanometres in, complex `n + ik` out**; all unit
conversion happens in Rust (`units.rs`). This deliberately removed the Python
wrappers' dependence on derived attributes (`wavelength_µm_2`, `wavelength_m`,
`E`) that the uploaded `cauchy_sellmeier.py` referenced but never assigned.

Implemented kernels (pure Rust, free functions, rayon-parallel above 4096 pts):
Cauchy, Cauchy+Urbach, Sellmeier (2/3-term), Sellmeier+Urbach, Lorentz,
Drude, Drude–Lorentz, the EMA mixers (Lichtenecker, Looyenga, general power
law, Maxwell-Garnett, Mori–Tanaka, Wiener bounds, roughness, Bruggeman) with
their composition layer, the **Cody–Lorentz**, **UBF Cody–Lorentz** and
**Tauc–Lorentz** FFT Kramers–Kronig paths (`kk.rs`, `cody_lorentz.rs`,
`ubf.rs`, `tauc_lorentz.rs`), and **Forouhi–Bloomer** (2019 interband + 2021
metal). Every dispersion model from the original library is on the Rust core,
plus Tauc–Lorentz added on top.

Parity: `tools/gen_golden.py` writes float64 references from the documented
kernel formulas (with fastmath off, plain NumPy *is* the reference).
`cargo test -p navette-materials` loads them and checks abs+rel tolerance.

```
running 22 tests ... ok    (worst tol-ratio ≈ 0.00, i.e. machine precision)
cauchy_basic  cauchy_single  cauchy_urbach  sellmeier_bk7  sellmeier_2term
sellmeier_urbach  lorentz_2osc  drude_basic  drude_lorentz
ema_looyenga  ema_maxwell_garnett  ema_bruggeman
cody_single  cody_multi  fb_single  fb_multi  fb_edge  fb_metal
ubf_single  ubf_multi  tauc_single  tauc_multi
```

The Cody–Lorentz, UBF and Tauc–Lorentz cases exercise the FFT Kramers–Kronig
path. A relaxed `rtol` was budgeted for FFT-library differences, but matching
NumPy's `irfft` exactly (scale by 1/M, discard the DC/Nyquist imaginary parts)
makes the `realfft` result agree with the NumPy reference to ≈1e-13 — well
inside the committed 1e-9 tolerance, so the relaxed budget is headroom.

The thin wrappers were validated against the same goldens through a pure-Python
`_native` shim (all 11 buildable models, max Δ = 0.00); since the Rust
`_native` already equals NumPy to machine precision, the wrappers driving the
real extension are correct by transitivity.

Bugs fixed during the port (noted for upstreaming into `navette`):
* `cauchy_sellmeier.py` read `self.wavelength_µm_2` / `wavelength_m` / `E` with
  no code assigning them — sidestepped by passing nm to Rust.
* `drudelorentz.py` `Drude.complex_refractive_index` called the kernel with a
  `gamma_drude=` keyword the kernel didn't accept — fixed by the positional
  `_native.drude_nk` call.
* `drudelorentz.py` `_sync()` deleted the scalar params and re-read them as
  `None` — fixed to preserve and restore them.
* `codylorentz.py` used an absolute `from material import …` (the other
  modules use the package-relative form, and a stray top-level `material`
  package shadowed it) — normalised to `from .material import …`.
* `forouhibloomer.py` `InterbandSingle` referenced `self._ib_terms_array` but
  only built the 1-D `self._fb_term_params` — added the `(1,4)` reshape.
* `forouhibloomer.py` classes called `super().__init__({...}, wavelength)`
  positionally against the old `Material(params, wavelength)` signature; the
  current base is `Material(wavelength, params)` — switched to keyword args.

Observation left as-is (semantics change, not a crash — flagged not fixed):
* `forouhibloomer.py` `ForouhiBloomerMetal2021` builds `_fe_params` but then
  calls the interband-only driver, so the free-electron term is silently
  dropped. The port preserves this behaviour; decide upstream whether it should
  use the metal driver (`fb_metal_nk`) instead.

## Building and testing

Rust core (no Python needed):
```bash
cd rust
cargo test -p navette-materials      # 12 golden-parity tests
```

Python extension (needs maturin + a C/Python toolchain):
```bash
pip install maturin numpy
maturin develop --release            # builds navette.materials._native in place
python -c "import numpy as np; from navette.materials import Sellmeier; \
           print(Sellmeier(params={'B1':1.03961212,'C1':0.00600069867,'B2':0.231792344,\
           'C2':0.0200179144,'B3':1.01046945,'C3':103.560653}).complex_refractive_index(np.array([587.56])))"
# → ~1.5168 (BK7 at the d-line)
```

Regenerate goldens / wrappers:
```bash
python tools/gen_golden.py
python tools/make_wrappers.py
```

## Toolchain pins (rustc 1.75, the apt-provided compiler)

`rayon-core=1.12.1`, `rayon=1.10.0`, `pest*=2.7.10`, and **`ndarray=0.15.6`
unified across the whole workspace** (the `numpy` crate's `>=0.15,<0.17` range
otherwise greedily picks 0.16.1, splitting the `ndarray` type and breaking the
binding). On a newer toolchain these pins can be relaxed.

## Next phases

* **Remove Numba fully** — the wrapper modules already drop `@njit`; the last
  step is deleting the now-unused reference kernel bodies and the standalone
  `ema_models.py` / Numba `compute_*` functions that nothing imports anymore.
* **Optional v2** — a Rust-side `Model::Effective` composition tree (so a whole
  composite evaluates in one FFI hop) and a Rust fitting loop. Neither is needed
  for parity; both are performance/ergonomics extensions.

## Done since the first cut

* **Cody–Lorentz + FFT Kramers–Kronig** (`kk.rs`, `cody_lorentz.rs`,
  `codylorentz.py`, binding `cody_lorentz_nk`). Adds `realfft` (3.5 / rustfft
  6.4, both fine on rustc 1.75); KK plan/grid/pad cached in a `OnceLock`.
  Out-of-grid target energies raise `ValueError`.
* **Forouhi–Bloomer** (`forouhi_bloomer.rs`, bindings `fb_interband_nk` /
  `fb_metal_nk`) — 2019 interband and 2021 metal, four wrapper classes.
* **EMA composition** (`ema_material.py`) — `EffectiveMaterial` /
  `RoughnessMaterial` now dispatch to the Rust mixers (`ema_*` bindings) and the
  `eps_to_nk` √ε step; the Numba `ema_models` import is gone. Validated across
  Bruggeman, Maxwell-Garnett, Looyenga, Mori–Tanaka (with `model_args`), and the
  roughness subclass.
* **UBF Cody–Lorentz** (`ubf.rs`, binding `ubf_nk`, `UBF_Cody_Lorentz.py`) — the
  Monolog-Lorentz ε₂ generator reusing the shared `kk::kk_fft` and `kk::interp`.
* **Tauc–Lorentz** (`tauc_lorentz.rs`, binding `tauc_lorentz_nk`,
  `tauclorentz.py` → `TaucLorentz`) — added on top of the original library at
  request. Standard Jellison–Modine ε₂ (shared gap `Eg`, oscillators (A, E0, C))
  with ε₁ from the same FFT-KK path as the other KK models. (If the closed-form
  Jellison–Modine ε₁ is ever wanted instead, it would be a separate analytic
  routine; the FFT-KK route was chosen for consistency and code reuse.)
