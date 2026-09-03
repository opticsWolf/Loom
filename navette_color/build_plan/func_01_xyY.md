# rust-codegen-worker Task — Unit func_01

## Unit Info
- **ID:** func_01
- **Name:** XYZ ↔ xyY conversions
- **Kind:** batch conversion
- **Jit Type:** N/A

> Converts between CIE XYZ and xyY chromaticity coordinates.

## Source Code (Python reference — `loom_colorengine.py`)
```python
# XYZ -> xyY
sum_xyz = X + Y + Z
if sum_xyz < 1e-12:        # black-pixel guard
    x, y, Yout = 0.3127, 0.3290, 0.0   # D65 chromaticity, Y = 0
else:
    x, y, Yout = X / sum_xyz, Y / sum_xyz, Y
# xyY -> XYZ
if y < 1e-12:
    X, Y, Z = 0.0, 0.0, 0.0
else:
    X, Y, Z = x * Yout / y, Yout, (1 - x - y) * Yout / y
```

## Signature Analysis
- **Types:** `&[[f64; 3]]` → `&mut [[f64; 3]]`
- **Dependencies:** None.

## Critical detail — Black-Pixel Convention
For zero-luminance (black) pixels (`sum_xyz < 1e-12`), the engine substitutes the
D65 white-point chromaticity (`x=0.3127, y=0.3290, Y=0.0`) to guarantee NaN-free
output. This is a Lindbloom design decision, not strictly CIE-mandated. The
inverse guards on `y < 1e-12` and returns all-zero.

## Target Output Files

### src/func_01.rs
```rust
pub fn xyz_to_xyy(xyz: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyz, o) in xyz.iter().zip(out.iter_mut()) {
        let sum = xyz[0] + xyz[1] + xyz[2];
        if sum > 1e-12 {
            let inv = 1.0 / sum;
            o[0] = xyz[0] * inv;
            o[1] = xyz[1] * inv;
            o[2] = xyz[1];
        } else {
            o[0] = 0.3127; o[1] = 0.3290; o[2] = 0.0;
        }
    }
}

pub fn xyy_to_xyz(xyy: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (xyy, o) in xyy.iter().zip(out.iter_mut()) {
        let (x, y, big_y) = (xyy[0], xyy[1], xyy[2]);
        if y > 1e-12 {
            let factor = big_y / y;
            o[0] = x * factor;
            o[1] = big_y;
            o[2] = (1.0 - x - y) * factor;
        } else {
            *o = [0.0, 0.0, 0.0];
        }
    }
}
```

## Tests
```rust
#[test]
fn black_pixel_convention() {
    let mut out = [[0.0; 3]];
    xyz_to_xyy(&[[0.0, 0.0, 0.0]], &mut out);
    assert!((out[0][0] - 0.3127).abs() < 1e-12);
    assert!((out[0][1] - 0.3290).abs() < 1e-12);
    assert_eq!(out[0][2], 0.0);
}
```
Plus golden-vector parity vs the Python reference (round-trip + non-trivial XYZ).
