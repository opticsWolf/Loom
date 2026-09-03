# rust-codegen-worker Task — Unit func_03

## Unit Info
- **ID:** func_03
- **Name:** XYZ ↔ CIELUV conversions
- **Kind:** batch conversion
- **Jit Type:** N/A

> Useful for emitted light and white-point estimation (u'v').

## Critical detail — u'v' Calculation and inverse masks
`u' = 4X / (X + 15Y + 3Z)`, `v' = 9Y / (X + 15Y + 3Z)`; for black pixels the
shared helper `xyz_to_uv_prime` returns `(0, 0)` when the denominator `< 1e-12`.

`L* = 116·f(Y/Yn) − 16` uses the **same** `f()` as CIELAB. The illuminant's
`(u'n, v'n)` anchor the chromatic offsets: `u* = 13·L·(u'−u'n)`, `v* = 13·L·(v'−v'n)`.

Inverse has **two** independent guards: it needs `L > 1e-12` to recover `(u',v')`
(else falls back to the white point), and it needs both `v' > 1e-12` **and**
`L > 1e-12` to recover `X, Z` (else `X = Z = 0`). `Y` is always
`f⁻¹((L+16)/116)·Yn`.

> NOTE: doc earlier showed only the `xyz_to_uv_prime` helper. The full kernel,
> the `lab_f`/`lab_f_inv` coupling, and the dual inverse masks are all required
> for parity — this is the unit with the largest observed FP delta (~2.8e-14).

## Target Output Files

### src/func_03.rs
```rust
use crate::common::{lab_f, lab_f_inv, xyz_to_uv_prime, REF_WHITE_D65};

pub fn xyz_to_luv(xyz: &[[f64; 3]], illuminant: &[f64; 3], out: &mut [[f64; 3]]) {
    let (un, vn) = xyz_to_uv_prime(illuminant);
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let (up, vp) = xyz_to_uv_prime(xyz);
        let l = 116.0 * lab_f(xyz[1] / illuminant[1]) - 16.0;
        o[0] = l;
        o[1] = 13.0 * l * (up - un);
        o[2] = 13.0 * l * (vp - vn);
    }
}

pub fn luv_to_xyz(luv: &[[f64; 3]], illuminant: &[f64; 3], out: &mut [[f64; 3]]) {
    let (un, vn) = xyz_to_uv_prime(illuminant);
    for (luv, o) in luv.iter().zip(out.iter_mut()) {
        let (l, u, v) = (luv[0], luv[1], luv[2]);
        let (up, vp) = if l > 1e-12 {
            let inv = 1.0 / (13.0 * l);
            (u * inv + un, v * inv + vn)
        } else { (un, vn) };
        let fy = (l + 16.0) / 116.0;
        let big_y = lab_f_inv(fy) * illuminant[1];
        let (mut x, mut z) = (0.0, 0.0);
        if vp > 1e-12 && l > 1e-12 {
            let inv4v = 1.0 / (4.0 * vp);
            x = big_y * 9.0 * up * inv4v;
            z = big_y * (12.0 - 3.0 * up - 20.0 * vp) * inv4v;
        }
        *o = [x, big_y, z];
    }
}

pub fn xyz_to_luv_d65(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    xyz_to_luv(xyz, &REF_WHITE_D65, out)
}
```

## Tests
Golden parity (forward + inverse) under D65; black-pixel → (L,0,0); round-trip.
