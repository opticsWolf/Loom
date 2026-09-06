# Navette - Weaving thin-film systems that perform

**Navette** is a high-performance, physically rigorous 1D optical engine designed for the simulation of light propagation in stratified media. Built on a modern **Scattering Matrix (S-matrix)** architecture, it offers a numerically stable and vectorized alternative to traditional Transfer Matrix Methods (TMM).

### 1. Unconditional Numerical Stability

Traditional TMM suffers from numerical divergence (exponentially growing evanescent waves) when dealing with thick layers or highly absorbing materials. Navette utilizes the **Redheffer Star Product** to propagate scattering matrices, ensuring that all matrix elements remain bounded and physically meaningful, regardless of layer thickness.

### 2. High-Concurrency Performance

As a Principal Performance Engineer, you need tools that scale. Navette is built for speed:

- **Parallel Execution**: Utilizes Rust + rayon data-parallelism across wavelengths/angles to saturate all available CPU cores.
    
- **Vectorized Engine**: Operations are performed across the entire (wavelength × angle) coordinate space in a single pass, eliminating Python's loop overhead.
    
- **Memory Efficiency**: Collapses multi-layer stacks into a compact global S-matrix to minimize cache misses.
    

### 3. Partial Coherence Support

Real-world systems often involve thick substrates (like a 1mm glass slide) where phase information is lost. Navette features a **Hybrid Coherence Engine**:

- **Coherent Blocks**: Preserves phase for thin-film interference.
    
- **Incoherent Interfaces**: Switches to intensity-based propagation for thick layers, preventing the "unphysical ringing" caused by assuming perfect coherence across a macroscopic substrate.
    

### 4. Advanced Physics Modeling

Navette goes beyond simple Fresnel equations to provide research-grade accuracy:

- **Interface Roughness**: Implements the **Névot-Croce** model, providing superior accuracy for high-frequency or X-ray reflectometry compared to standard Gaussian approximations.
    
- **Ellipsometric Rigor**: Outputs (Ψ,Δ) parameters that strictly follow the **Azzam & Bashara** convention, ensuring direct compatibility with commercial ellipsometers (e.g., Woollam, Horiba).

### 5. Automated Coating Design

Navette doesn't just simulate — it synthesizes, with the classic **needle method** running natively on the same engine:

- **Needle Insertion**: Probes every candidate position with an infinitesimal test layer and inserts real material where the merit function improves most — the Tikhonravov needle algorithm, merit-driven and target-aware.

- **Thickness Optimization**: Levenberg-Marquardt refinement over free layers with bounds and clamping, interleaved with insertion passes and impact-ranked cleanup (merge, thin-layer removal, re-optimization).

- **Graded Media**: Gradient-index profiles expand natively for simulation and serve as pinned background (substrate diffusion gradients, rugate foundations) while the needle designs around them.
### Technical Specifications

|**Feature**|**Implementation & Engineering Benefit**|
|---|---|
|**Core Algorithm**|**1D Scattering Matrix ($S$-matrix)**: Utilizes the Redheffer Star Product to eliminate numerical divergence and precision loss in thick or highly absorbing layers.|
|**Propagation Logic**|**Hybrid Mixed Coherence**: Sophisticated dual-stage engine supporting phase-accurate (coherent) and intensity-only (incoherent) layers within a single pass.|
|**Coherent Blocks**|**$2 \times 2$ Complex Field Matrices**: Maintains full phase and amplitude information, ensuring rigorous calculation of thin-film interference and ellipsometric parameters.|
|**Incoherent Blocks**|**Stokes-Mueller / Intensity Redheffer**: Prevents unphysical interference artifacts in macroscopic substrates by utilizing intensity-based propagation.|
|**Roughness Model**|**Névot-Croce (Exact Wavevector)**: Achieves research-grade accuracy for X-ray and UV interfaces by modeling exact wavevector correlations across boundaries.|
|**Optimization**|**Rust / rayon + PyO3**: Native multi-threaded kernels (GIL released) with a thin Python API, optimized for high-concurrency simulation and real-time GUI responsiveness.|
|**Polarization**|**Full $s$ and $p$ Support**: Comprehensive Jones and Stokes calculus integration, following standard commercial ellipsometry conventions (Azzam & Bashara).|
|**Complexity**|**$O(N)$ Scaling**: Optimized linear time complexity relative to the number of layers, ensuring stable performance for complex multi-stack architectures.|

### Project layout

```
Navette/
├── Cargo.toml                # Rust workspace (cargo check/test --workspace)
├── pyproject.toml            # maturin project: builds the `navette` wheel (src layout)
├── src/navette/              # unified Python package
│   ├── __init__.py           # version + public surface
│   ├── color/                # wrapper over native `navette._color`
│   ├── interpolate/          # wrapper over native `navette._interpolate`
│   ├── smatrix/              # ScatterMatrix + needle (native `navette._smatrix`)
│   ├── spectralweave/        # weavers + merit (native `navette._spectralweave`)
│   ├── materials/            # dispersion models (native `navette._materials`)
│   ├── _*.py                 # shims re-exporting the `navette._navette` submodules
│   ├── structure/            # stacks, architect (native model + thin wrappers)
│   ├── synthesis/            # needle pipeline driver (native DesignStack)
│   ├── config/               # YAML/JSON libraries, stacks, program documents
│   └── data/CIE/             # bundled reference spectra
├── rust/                     # Rust sources: one engine crate + bindings
│   ├── navette/              # pure-Rust engine (color/interpolate/materials/
│   │                         # smatrix/spectralweave/structure modules;
│   │                         # published as `navette` on crates.io)
│   └── navette-py/           # PyO3 aggregator -> navette._navette (one wheel)
├── validation/               # tests, parity, benches, goldens + references (see validation/README.md)
├── examples/  tools/  docs/plans/  attic/
```

### Install & build

```powershell
# Single aggregated native extension (navette._navette, all engines):
maturin develop
# checks
cargo check --workspace
cargo test --workspace     # everything (needs Python for binding crates)
cargo test-pure            # pure-Rust gate (no Python needed)
pytest validation
```

### Layout notes

- `rust/` holds the Cargo workspace (the single `navette` engine crate
  plus the `navette-py` PyO3 aggregator) — the idiomatic Rust layout,
  publishable to crates.io.
- `src/navette/` is the Python package in src-layout — the idiomatic
  Python layout, which maturin detects automatically for mixed projects.

### Release & publish

Release automation: tag `vX.Y.Z` (must match `pyproject.toml`, workspace
`Cargo.toml`, `__about__.py` — enforced by CI) → `.github/workflows/release.yml`
builds wheels (Linux/Windows/macOS) and publishes to PyPI (trusted
publisher) + crates.io (token), leaf crates first.

```powershell
maturin build --release   # -> target/wheels/navette-0.4.0-*.whl (single wheel, all engines)
```

Manual fallback: `cargo publish -p navette`;
`maturin upload target/wheels/navette-0.4.0-*.whl`.
