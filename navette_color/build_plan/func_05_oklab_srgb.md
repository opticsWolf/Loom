# rust-codegen-worker Task — Unit func_05

## Unit Info
- **ID:** func_05
- **Name:** Oklab (sRGB Pipeline, legacy)
- **Kind:** batch conversion
- **Jit Type:** N/A

> Legacy Oklab matrices that bake the sRGB linearization into M1 (Ottosson's
> original blog post coefficients).

## Critical detail — Clip points, and signed_pow again
- `srgb_to_oklab`: clip input to `[0,1]` → inverse-gamma (linearize) → M1 →
  `signed_pow(·, 1/3)` → M2.
- `oklab_to_srgb`: M2⁻¹ → `signed_pow(·, 3)` → M1⁻¹ → **clip linear RGB to
  `[0,1]`** → gamma encode. The clip before gamma is what makes the result
  display-referred.

> CORRECTION: earlier draft left the body as `// ...`. Both directions clip to
> `[0,1]`, and both use `signed_pow` (not `cbrt`/`powf`) for parity, matching
> func_09.

## Target Output Files

### src/func_05.rs
```rust
use crate::common::{clip01, gamma_srgb, inverse_gamma_srgb, mat3_mul_vec, signed_pow};
use crate::matrices::{M1_OKLAB_SRGB, M1_OKLAB_SRGB_INV, M2_OKLAB_SRGB, M2_OKLAB_SRGB_INV};

pub fn srgb_to_oklab(rgb: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (rgb, o) in rgb.iter().zip(out.iter_mut()) {
        let lin = [
            inverse_gamma_srgb(clip01(rgb[0])),
            inverse_gamma_srgb(clip01(rgb[1])),
            inverse_gamma_srgb(clip01(rgb[2])),
        ];
        let lms = mat3_mul_vec(&M1_OKLAB_SRGB, &lin);
        let lms_c = [
            signed_pow(lms[0], 1.0 / 3.0),
            signed_pow(lms[1], 1.0 / 3.0),
            signed_pow(lms[2], 1.0 / 3.0),
        ];
        *o = mat3_mul_vec(&M2_OKLAB_SRGB, &lms_c);
    }
}

pub fn oklab_to_srgb(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lab, o) in lab.iter().zip(out.iter_mut()) {
        let lms_p = mat3_mul_vec(&M2_OKLAB_SRGB_INV, lab);
        let lms_lin = [
            signed_pow(lms_p[0], 3.0),
            signed_pow(lms_p[1], 3.0),
            signed_pow(lms_p[2], 3.0),
        ];
        let lin = mat3_mul_vec(&M1_OKLAB_SRGB_INV, &lms_lin);
        *o = [
            gamma_srgb(clip01(lin[0])),
            gamma_srgb(clip01(lin[1])),
            gamma_srgb(clip01(lin[2])),
        ];
    }
}
```

## Tests
Golden parity both directions; sRGB→Oklab→sRGB round-trip within in-gamut range.
