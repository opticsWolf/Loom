# rust-codegen-worker Task — Unit func_4

## Unit Info
- ID: func_4
- Name: core_engine_rigorous_ellipsometry
- Kind: function (calls solve_coherent_block_fields internally)
- Jit Type: @njit(parallel=True, fastmath=True, cache=True)
- **Uses prange — needs rayon parallel runtime**

## Source Code (full function body from loom_matrix.py lines 410-716)

The function signature:
```python
def core_engine_rigorous_ellipsometry(
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, debug_flag
):
```

Returns 13-element tuple of f64[;] arrays: (Psi_R, Delta_R, DOP_R, Rs, Rp, R_avg, Psi_T, Delta_T, DOP_T, Ts, Tp, T_avg, conservation_err)

Key features to port:
- Flattened prange loop over total_points = num_wavs * num_angles
- Per-point: Snell's law, coherent block identification via while loops, solve_coherent_block_fields calls for both polarizations (s and p)
- Mueller cross-term accumulation for transmission ellipsometry
- Reflection ellipsometry from first coherent block phase capture
- Stokes parameters S0-S3 computation for R and T
- arctan/arctan2 for Psi/Delta extraction
- DOP computation with clamping

IMPORTANT:
- The parallel prange is a Numba feature. For the Rust port, implement as sequential loop (rayon parallelism can be added later). Focus on correctness first.
- All numpy ops (np.sqrt, np.abs, np.exp, np.sin, np.cos) need equivalents from num_complex or std lib
- Return type: 13 separate Vec<f64> arrays packed into a Python tuple

Write output to C:/Users/Frank/rust-rewrite/ (src/func_4.rs, benchmarks/test_core_engine_rigorous_ellipsometry.py).
For the test, use a small setup (~50 wavelengths x ~10 angles) for speed.
