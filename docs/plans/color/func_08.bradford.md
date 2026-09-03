# rust-codegen-worker Task — Unit func_08

## Unit Info
- **ID:** func_08
- **Name:** Chromatic Adaptation (Bradford)
- **Kind:** matrix math
- **Jit Type:** N/A

## Critical detail — gains, clamps, identity short-circuit, clip
The Python `lru_cache` is dropped: a 3×3 adapt matrix is ~100 FLOPs, computed on
the fly. The parity-critical pieces the earlier draft omitted:

1. **Von Kries gains** in Bradford cone space:
   `src_lms = M_BRADFORD · src_white`, `dst_lms = M_BRADFORD · dst_white`,
   then **clamp** each `|src_lms[i]| < 1e-12 → 1e-12` before
   `gains = dst_lms / src_lms`.
2. **Identity short-circuit:** if `np.allclose(src_white, dst_white)` (NumPy
   defaults `rtol=1e-5, atol=1e-8`), skip adaptation entirely (apply only the
   negative-clip if requested).
3. **clip_negative (default true):** after the inverse transform, components
   `< −1e-6` are clamped to `0.0`.
4. **Matrix convention:** the reference composes for **row-vector** multiply
   (`out = xyz @ M`) as `M_BRADFORD_T @ diag(gains) @ M_BRADFORD_INV_T`. The
   crate stores natural matrices and either (a) applies gains element-wise in
   cone space per pixel, or (b) builds the equivalent composite via
   `transpose(M_BRADFORD_INV · diag(gains) · M_BRADFORD)` — both are identical.

## Target Output Files

### src/func_08.rs
```rust
use crate::common::{mat3_mul, mat3_mul_vec};
use crate::matrices::{M_BRADFORD, M_BRADFORD_INV};

fn bradford_gains(src_white: &[f64; 3], dst_white: &[f64; 3]) -> [f64; 3] {
    let mut src_lms = mat3_mul_vec(&M_BRADFORD, src_white);
    let dst_lms = mat3_mul_vec(&M_BRADFORD, dst_white);
    for c in src_lms.iter_mut() { if c.abs() < 1e-12 { *c = 1e-12; } }
    [dst_lms[0]/src_lms[0], dst_lms[1]/src_lms[1], dst_lms[2]/src_lms[2]]
}

pub fn adapt(
    xyz: &[[f64; 3]], src_white: &[f64; 3], dst_white: &[f64; 3],
    clip_negative: bool, out: &mut [[f64; 3]],
) {
    let clip = |v: f64| if v < -1e-6 { 0.0 } else { v };
    if allclose3(src_white, dst_white) { /* identity (+clip) */ return; }
    let gains = bradford_gains(src_white, dst_white);
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let mut lms = mat3_mul_vec(&M_BRADFORD, xyz);
        for k in 0..3 { lms[k] *= gains[k]; }
        let res = mat3_mul_vec(&M_BRADFORD_INV, &lms);
        *o = if clip_negative { [clip(res[0]), clip(res[1]), clip(res[2])] } else { res };
    }
}
```
Also exposes `calc_transform_matrix(src, dst)` returning the row-vector composite.

## Tests
Golden parity D65→D50 and D50→D65; identity (D65→D65); negative-clip behaviour.
