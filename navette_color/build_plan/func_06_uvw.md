# rust-codegen-worker Task — Unit func_06

## Unit Info
- **ID:** func_06
- **Name:** CIE 1964 U*V*W*
- **Kind:** batch conversion
- **Jit Type:** N/A

## Critical detail — Y-range, the v formula, and the (un, vn) anchor
- **Y-range scaling:** CIE 1964 expects `Y ∈ [0,100]`; the engine works in
  `[0,1]`, so multiply `X, Y, Z` by `100.0` inside the kernel.
- **W\*** = `25·Y₁₀₀^(1/3) − 17`.
- The chromatic `(u, v)` here use the *scaled* XYZ with denominator
  `X + 15Y + 3Z`, and crucially `v = 6Y/denom` (a **CIE-1960 v**, i.e. `(2/3)v'`),
  **not** `9Y/denom`. Guard `denom < 1e-12 → (0,0)`.
- The white-point anchor is in CIE-1960 coordinates: `un = u'n`,
  `vn = (2/3) v'n`. The helper `white_point_uv1960` builds this from an XYZ
  illuminant.
- `u* = 13·W*·(u − un)`, `v* = 13·W*·(v − vn)`.

> CORRECTION: the earlier stub stopped at "calculate u, v (1960) and scale by
> 13·W*". The decisive parity points are (a) `6Y` in the v-numerator and (b) the
> 1960 anchor `vn = (2/3)v'n`. The inverse reverses each guard
> (`|W*| < 1e-12`, `|den| < 1e-12`, `|y| < 1e-12`) and divides the result back
> by 100.

## Target Output Files

### src/func_06.rs
```rust
use crate::common::xyz_to_uv_prime;

#[inline]
pub fn white_point_uv1960(illuminant: &[f64; 3]) -> (f64, f64) {
    let (up, vp) = xyz_to_uv_prime(illuminant);
    (up, (2.0 / 3.0) * vp)
}

pub fn xyz_to_uvw(xyz: &[[f64; 3]], un: f64, vn: f64, out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let x = xyz[0] * 100.0; let y = xyz[1] * 100.0; let z = xyz[2] * 100.0;
        let denom = x + 15.0 * y + 3.0 * z;
        let (u, v) = if denom < 1e-12 { (0.0, 0.0) }
                     else { (4.0 * x / denom, 6.0 * y / denom) };
        let w_star = 25.0 * y.powf(1.0 / 3.0) - 17.0;
        o[0] = 13.0 * w_star * (u - un);
        o[1] = 13.0 * w_star * (v - vn);
        o[2] = w_star;
    }
}
// uvw_to_xyz reverses the above; see crate source for the full inverse.
```

## Tests
Golden parity forward/inverse vs Python with an explicit D65 `(un, vn)` anchor.
