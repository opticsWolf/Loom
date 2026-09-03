# rust-codegen-worker Task — Unit func_5

## Unit Info
- ID: func_5
- Name: core_engine_photometry_only
- Kind: function (calls solve_coherent_block_fields internally)
- Jit Type: @njit(parallel=True, fastmath=True, cache=True)
- **Uses prange — needs rayon parallel runtime**

## Source Code (full function body from loom_matrix.py lines 718-904)

The function signature:
```python
def core_engine_photometry_only(
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, calc_s, calc_p
):
```

Returns 4-element tuple of f64[;] arrays: (Rs_out, Rp_out, Ts_out, Tp_out) — shape [num_angles, num_wavs]

Key features to port:
- Flattened prange loop over total_points = num_wavs * num_angles
- Conditional s/p polarization computation via calc_s and calc_p flags
- Per-polarization: coherent block identification, solve_coherent_block_fields calls, real Redheffer product accumulation
- Incoherent layer propagation with absorption factor

IMPORTANT:
- The parallel prange is a Numba feature. For the Rust port, implement as sequential loop (rayon parallelism can be added later). Focus on correctness first.
- All numpy ops (np.sqrt, np.abs, np.exp) need equivalents from num_complex or std lib
- Return type: 4 separate Vec<f64> arrays packed into a Python tuple

Write output to C:/Users/Frank/rust-rewrite/ (src/func_5.rs, benchmarks/test_core_engine_photometry_only.py).
For the test, use a small setup (~50 wavelengths x ~10 angles) for speed.
