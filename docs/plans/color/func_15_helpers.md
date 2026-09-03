# rust-codegen-worker Task — Unit func_15

## Unit Info
- **ID:** func_15
- **Name:** Shape Handling & Broadcasting
- **Kind:** utility / idiomatic Rust
- **Jit Type:** N/A

## Critical detail — Rust idioms replace Python decorators
Python used `@handle_shapes` to coerce `(3,)` → `(1,3)` and `_prepare_inputs` to
broadcast. In Rust those runtime checks are mostly eliminated by strong typing:

- Color batches: `&[[f64; 3]]` — the stride-of-3 guarantee lets the compiler
  auto-vectorize aggressively (preferred over `ndarray`/flat `&[f64]` for N×3).
- Spectra: `&[f64]`.
- A single color is a one-element slice (`&[color]`).

The **one** genuine runtime broadcast — 1 reference vs N samples for the Delta-E
metrics — lives in `metrics::map_pairs`, re-exported here. It implements
NumPy-style broadcasting (`equal | 1-vs-N | N-vs-1`, else panic) and is
`#[cfg(feature = "parallel")]`-gated to rayon (sequential by default).

## Target Output Files

### src/func_15.rs
```rust
pub use crate::metrics::map_pairs;
```

### src/metrics.rs (the runtime home)
```rust
pub fn map_pairs<F>(lab1: &[[f64; 3]], lab2: &[[f64; 3]], f: F) -> Vec<f64>
where F: Fn(&[f64; 3], &[f64; 3]) -> f64 + Sync + Send {
    let (n1, n2) = (lab1.len(), lab2.len());
    let n = match (n1, n2) {
        (a, b) if a == b => a,
        (1, b) => b,
        (a, 1) => a,
        (a, b) => panic!("shapes {a} and {b} are not broadcastable"),
    };
    let get = |arr: &[[f64; 3]], i: usize| if arr.len() == 1 { arr[0] } else { arr[i] };
    // rayon when feature="parallel", else sequential
    (0..n).map(|i| f(&get(lab1, i), &get(lab2, i))).collect()
}
```

## Tests
Equal-length, 1-vs-N, and N-vs-1 broadcasts; non-broadcastable lengths panic.
