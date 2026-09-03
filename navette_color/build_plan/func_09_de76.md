# rust-codegen-worker Task — Unit func_09

## Unit Info
- **ID:** func_09
- **Name:** Delta E 76
- **Kind:** batch metric
- **Jit Type:** `rayon` (optional `parallel` feature)

> Simple Euclidean distance in CIELAB. Symmetric.

## Critical detail — broadcasting & optional parallelism
The per-pair kernel is trivial; the important part is that all four Delta-E units
(func_14–17) share one driver, `metrics::map_pairs`, which provides NumPy-style
**1-vs-N broadcasting** and is `#[cfg(feature = "parallel")]`-gated to rayon
(sequential otherwise — zero deps by default).

> NOTE: rayon-core 1.13 needs rustc ≥ 1.80; the default (sequential) build works
> on rustc 1.75. Pin an older rayon or use ≥1.80 to compile `--features parallel`.

## Target Output Files

### src/func_09.rs
```rust
#[inline(always)]
pub fn delta_e_76_single(lab1: &[f64; 3], lab2: &[f64; 3]) -> f64 {
    let dl = lab1[0] - lab2[0];
    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    (dl * dl + da * da + db * db).sqrt()
}

pub fn delta_e_76(lab1: &[[f64; 3]], lab2: &[[f64; 3]]) -> Vec<f64> {
    crate::metrics::map_pairs(lab1, lab2, delta_e_76_single)
}
```

## Tests
Golden parity on batches; 1-vs-N broadcast; identity pair → 0.0.
