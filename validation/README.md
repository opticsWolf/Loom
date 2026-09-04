# validation/

All test, parity, benchmark, golden and reference material in one place.
`pytest validation` runs the collected suites; parity/bench scripts are
standalone (they execute workloads on import) and must be run explicitly.

```
validation/
├── conftest.py               # collects smoke/ + goldens/, ignores parity/ + benches/
├── smoke/
│   └── test_navette_imports.py        # pytest: import surface (6 tests)
├── goldens/
│   └── spectralweave/
│       └── test_golden.py             # pytest: pinned numeric goldens (22 tests)
├── parity/                   # numba/NumPy-vs-Rust parity scripts (run directly)
│   ├── smatrix/
│   │   ├── test_w_function.py                    # PASS, ~0.4x (µs kernel)
│   │   ├── test_redheffer_product_real.py        # PASS, ~1.7x
│   │   ├── test_redheffer_product_complex_field.py # PASS, ~1.7x
│   │   ├── test_solve_coherent_block_fields.py   # PASS, ~1.1x
│   │   ├── test_core_engine_photometry_only.py   # NEEDS PORT (old entry point)
│   │   ├── test_core_engine_rigorous_ellipsometry.py # NEEDS PORT
│   │   └── refs/
│   │       └── loom_matrix.py         # numba reference (all kernels)
│   └── materials/
│       ├── gen_goldens.py             # regenerates goldens/ via NumPy
│       └── goldens/*.npy              # read by crates/navette-materials/tests/parity.rs (22 tests)
└── benches/                  # timing scripts (run directly)
    ├── spectralweave/
    │   ├── navette_spectral_bench.py  # [--quick] Python-vs-Rust weave/unweave
    │   ├── navette_target_bench.py    # [--quick] ingest + merit timings
    │   └── refs/
    │       └── loom_spectraldata.py   # pure-Python reference weaver
    ├── interpolate/
    │   ├── 1dinterpol_test_bench.py   # SciPy/loom/Rust interpolation shootout
    │   └── refs/
    │       └── loom_unispline.py      # numba reference interpolator
    └── color/
        ├── bench_validate_color.py    # vs colour-science + loom (needs `colour` pkg)
        ├── refs/
        │   └── loom_colorengine.py    # numba reference color engine
        └── gen/
            ├── gen_golden.py / gen_matrices.py  # regenerate golden.rs
            └── golden.rs                  # copy of crates/navette-color/src/golden.rs
```

## Commands

```powershell
# pytest suites (smoke + spectral goldens)
pytest validation

# Rust suites (incl. materials parity against parity/materials/goldens/)
cargo test --workspace

# parity scripts (from the repo root, .venv active)
python validation/parity/smatrix/test_w_function.py
# ... etc.

# benches
python validation/benches/spectralweave/navette_spectral_bench.py --quick
python validation/benches/spectralweave/navette_target_bench.py --quick
python validation/benches/interpolate/1dinterpol_test_bench.py
python validation/benches/color/bench_validate_color.py

# regenerate materials goldens (then: cargo test -p navette-materials)
python validation/parity/materials/gen_goldens.py
```

## Notes

- `parity/smatrix/test_core_engine_*.py` predate the request-driven
  `core_engine` API (`navette._smatrix.core_engine`) and need porting;
  everything else in `parity/` passes against the release build.
- Legacy pre-unification scripts that tested removed modules (`smatrix`,
  `navette_interpolator`, `request_flags`) were deleted as unportable;
  git history retains them.
- Speedups quoted above are illustrative (Windows, release LTO build);
  re-run on your hardware.
