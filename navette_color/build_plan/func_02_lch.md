# rust-codegen-worker Task — Unit func_02

## Unit Info
- **ID:** func_02
- **Name:** Lab ↔ LCh conversions
- **Kind:** batch conversion
- **Jit Type:** N/A

> Cylindrical representation of CIELAB.

## Critical detail — Hue Angle
Hue `h = atan2(b, a)` converted to degrees and wrapped to `[0, 360)`: negative
results get `+ 360.0`. Chroma `C = hypot(a, b)`. `L` passes through unchanged.
Inverse: `a = C·cos(h), b = C·sin(h)` with `h` back in radians.

## Target Output Files

### src/func_02.rs
```rust
use crate::common::{DEG2RAD, RAD2DEG};

pub fn lab_to_lch(lab: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lab, o) in lab.iter().zip(out.iter_mut()) {
        let (l, a, b) = (lab[0], lab[1], lab[2]);
        let c = a.hypot(b);
        let mut h = b.atan2(a) * RAD2DEG;
        if h < 0.0 { h += 360.0; }
        *o = [l, c, h];
    }
}

pub fn lch_to_lab(lch: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (lch, o) in lch.iter().zip(out.iter_mut()) {
        let (l, c, h_deg) = (lch[0], lch[1], lch[2]);
        let h = h_deg * DEG2RAD;
        *o = [l, c * h.cos(), c * h.sin()];
    }
}
```

## Tests
Golden parity vs Python for assorted Lab triples, plus a Lab→LCh→Lab round-trip.
