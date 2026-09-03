# rust-codegen-worker Task — Unit func_07

## Unit Info
- **ID:** func_07
- **Name:** CIE 1960 UCS & Chromaticity
- **Kind:** batch conversion
- **Jit Type:** N/A

> Linear UCS transforms plus 1960/1976 chromaticity conversions.

## Critical detail — four distinct kernels
1. **XYZ ↔ UCS (linear UVW tristimulus):**
   `U = (2/3)X`, `V = Y`, `W = ½(−X + 3Y + Z)`.
   Inverse: `X = 1.5U`, `Y = V`, `Z = 1.5U − 3V + 2W`.
2. **XYZ → 1960 (u,v) chromaticity:** `u = u'`, `v = (2/3)v'` (via `xyz_to_uv_prime`).
3. **uv1976 → xy:** denom `6u' − 16v' + 12`, `x = 9u'/denom`, `y = 4v'/denom`.
4. **uv1960 → xy:** denom `2u − 8v + 4`, `x = 3u/denom`, `y = 2v/denom`.

All chromaticity inverses guard `|denom| < 1e-12 → (0,0)`.

> CORRECTION: the earlier draft only described the 1960-vs-1976 denominator
> distinction. The unit also contains the linear UCS tristimulus pair (kernels 1)
> with the exact coefficients above, which the parity tests cover.

## Target Output Files

### src/func_07.rs
```rust
use crate::common::xyz_to_uv_prime;

pub fn xyz_to_ucs(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
        o[0] = (2.0 / 3.0) * x;
        o[1] = y;
        o[2] = 0.5 * (-x + 3.0 * y + z);
    }
}
pub fn ucs_to_xyz(ucs: &[[f64; 3]], out: &mut [[f64; 3]]) { /* inverse */ }
pub fn xyz_to_ucs_uv(xyz: &[[f64; 3]], out: &mut [[f64; 2]]) { /* u', (2/3)v' */ }
pub fn uv1976_to_xy(uvp: &[[f64; 2]], out: &mut [[f64; 2]]) { /* 6u'-16v'+12 */ }
pub fn uv1960_to_xy(uv: &[[f64; 2]], out: &mut [[f64; 2]])  { /* 2u-8v+4 */ }
```

## Tests
Golden parity on all four kernels; UCS round-trip; cross-check 1960↔1976 on a
common XYZ.
