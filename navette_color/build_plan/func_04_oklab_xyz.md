# rust-codegen-worker Task — Unit func_04

## Unit Info
- **ID:** func_04
- **Name:** Oklab (Direct XYZ Pipeline)
- **Kind:** batch conversion
- **Jit Type:** N/A

> Standard Oklab matrices operating directly on XYZ.

## Critical detail — Sign-preserving cube root (NOT plain cbrt)
The reference computes `lms_prime = sign(lms) * abs(lms)**(1/3)`.

> CORRECTION to the earlier draft: do **not** use `f64::cbrt()`. While `cbrt`
> also preserves sign, `sign(x)*abs(x).powf(1/3)` and `cbrt(x)` differ by a few
> ULPs, which breaks bit-level parity with NumPy. Use a shared `signed_pow`
> helper that matches NumPy semantics, including `sign(0.0) == 0.0`. The same
> helper raises to power `3.0` on the inverse path.

## Target Output Files

### src/func_04.rs
```rust
use crate::common::{mat3_mul_vec, signed_pow};
use crate::matrices::{
    M1_LMS_TO_XYZ_OKLAB, M1_XYZ_TO_LMS_OKLAB, M2_LAB_TO_LMS_OKLAB, M2_LMS_TO_LAB_OKLAB,
};

pub fn xyz_to_oklab(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let lms = mat3_mul_vec(&M1_XYZ_TO_LMS_OKLAB, xyz);
        let lms_p = [
            signed_pow(lms[0], 1.0 / 3.0),
            signed_pow(lms[1], 1.0 / 3.0),
            signed_pow(lms[2], 1.0 / 3.0),
        ];
        *o = mat3_mul_vec(&M2_LMS_TO_LAB_OKLAB, &lms_p);
    }
}

pub fn oklab_to_xyz(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lab, o) in lab.iter().zip(out.iter_mut()) {
        let lms_p = mat3_mul_vec(&M2_LAB_TO_LMS_OKLAB, lab);
        let lms = [
            signed_pow(lms_p[0], 3.0),
            signed_pow(lms_p[1], 3.0),
            signed_pow(lms_p[2], 3.0),
        ];
        *o = mat3_mul_vec(&M1_LMS_TO_XYZ_OKLAB, &lms);
    }
}
```

## Matrix provenance
`M1_*`/`M2_*` and their inverses are emitted by `refgen/gen_matrices.py` straight
from the Python engine constants (17-significant-digit literals; inverses via
`numpy.linalg.inv`) so the Rust matrices are bit-identical to the reference.

## Tests
Golden parity forward/inverse over a spread of XYZ including negatives; round-trip.
