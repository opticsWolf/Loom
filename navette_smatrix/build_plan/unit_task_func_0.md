# rust-codegen-worker Task — Unit func_0

## Unit Info
- **ID:** func_0
- **Name:** w_function
- **Kind:** function (standalone, no dependencies)
- **Jit Type:** @njit(cache=True, inline='always')

## Source Code
```python
@njit(cache=True, inline='always')
def w_function(q, rough_type):
    """
    Calculates the roughness factor W(q) for optical thin films.
    rough_type 0: W = 1 (sharp interface)
    rough_type 1: W = sin(sqrt(3)*q) / (sqrt(3)*q) (triangle form factor)
    rough_type 2: W = cos(q) (cosine form factor)
    rough_type 3: W = 1 / (1 + q^2/2) (exponential/Lorentzian)
    rough_type 4: W = exp(-q^2/2) (Gaussian/Debye-Waller)
    """
    if rough_type == 0:
        return 1.0 + 0j

    if rough_type == 1:
        val = q * SQRT3   # SQRT3 = 1.73205080757
        if np.abs(val) < 1e-9:
            return 1.0 + 0j
        return np.sin(val) / val

    elif rough_type == 2:
        return np.cos(q)

    elif rough_type == 3:
        return 1.0 / (1.0 + (q * q) * 0.5)

    elif rough_type == 4:
        return np.exp(-(q * q) * 0.5)

    return 1.0 + 0j
```

## Signature Analysis
- **Input types:** `q: complex128`, `rough_type: int32` (from @njit inference)
- **Return type:** `complex128`
- **Dependencies:** none (standalone — uses module-level SQRT3 constant and numpy functions on complex)

## Target Output Files (all into C:/Users/Frank/rust-rewrite/)

### 1. src/func_0.rs (or append to lib.rs)
Rust implementation using PyO3 >= 0.28 with edition 2024:
```rust
use pyo3::prelude::*;
use num_complex::Complex64;

const SQRT3: f64 = 1.73205080757;

#[pyfunction]
pub fn w_function(q: Complex64, rough_type: i32) -> PyResult<Complex64> {
    // ... implementation
}
```

### 2. benchmarks/test_w_function.py
Comparison test that:
a) Imports the original numba function from loom_matrix module
b) Runs both versions with identical inputs for all roughness types (0-4)
c) Compares outputs with np.allclose() tolerance (rtol=1e-12, atol=1e-15)
d) Benchmarks both with time.perf_counter (>= 50 iterations each)
e) Prints results in parseable format:
   ```
   CORRECTNESS w_function_type<N> PASS | diff_max=<value>
   BENCH_RESULT w_function numba_time=<ms> rust_time=<ms> speedup=<x>
   ```

## Constraints
- Rust >= 1.85, PyO3 >= 0.28 compatible only
- Use edition = "2024" in Cargo.toml
- All output must go into C:/Users/Frank/rust-rewrite/ — do not modify source files
- The SQRT3 constant must match the Python value exactly (1.73205080757)
