# rust-codegen-worker Task — Unit func_13

## Unit Info
- **ID:** func_13
- **Name:** Spectral Pipeline
- **Kind:** integration & adaptation
- **Jit Type:** N/A

## Critical detail — k-normalization, the source white, adapt, then sRGB
Pipeline: SPD → XYZ → (optional Bradford adapt to D65) → sRGB.

1. **Normalization constant** `k = 1 / Σ(E(λ)·ȳ(λ)·Δλ)` over the illuminant `E`
   and the **ȳ** CMF column. Guard `|denom| < 1e-12 → k = 1`.
2. **Integrate** `XYZ = Σ spd·cmfs·E·k·Δλ` (per channel `x̄, ȳ, z̄`). This is
   relative XYZ where a perfect reflecting diffuser maps to `Y = 1`.
3. **Adaptation (if enabled):** the source white is the illuminant integrated
   against the CMFs, **renormalized so its Y = 1**
   (`raw_white / raw_white[1]`), then Bradford-adapted to D65 with
   `clip_negative = true`.
4. **sRGB** via `xyz_to_srgb(..., clip = true)`.

> CORRECTION: earlier draft listed the four steps but not (a) the `ȳ`-only
> denominator for `k`, nor (b) that the source white is the **renormalized**
> CMF-integrated illuminant (`/raw_white[1]`) rather than the raw integral. Both
> matter for parity.

## Target Output Files

### src/func_13.rs
```rust
use crate::common::{xyz_to_srgb, REF_WHITE_D65};
use crate::func_13::adapt;

pub fn spectral_to_srgb(
    spd: &[f64], cmfs: &[[f64; 3]], illum: &[f64],
    interval: f64, apply_adaptation: bool,
) -> [f64; 3] {
    let w = cmfs.len();
    assert!(spd.len() == w && illum.len() == w, "wavelength count mismatch");

    let denom: f64 = (0..w).map(|i| illum[i] * cmfs[i][1] * interval).sum();
    let k = if denom.abs() > 1e-12 { 1.0 / denom } else { 1.0 };

    let mut xyz = [0.0f64; 3];
    for i in 0..w {
        let s = spd[i] * illum[i] * k * interval;
        xyz[0] += s * cmfs[i][0];
        xyz[1] += s * cmfs[i][1];
        xyz[2] += s * cmfs[i][2];
    }

    if apply_adaptation {
        let mut raw = [0.0f64; 3];
        for i in 0..w {
            let e = illum[i] * interval;
            raw[0] += cmfs[i][0]*e; raw[1] += cmfs[i][1]*e; raw[2] += cmfs[i][2]*e;
        }
        let source_white = if raw[1] > 1e-12 { [raw[0]/raw[1], 1.0, raw[2]/raw[1]] } else { raw };
        let mut adapted = [[0.0; 3]; 1];
        adapt(&[xyz], &source_white, &REF_WHITE_D65, true, &mut adapted);
        xyz = adapted[0];
    }
    let mut srgb = [[0.0; 3]; 1];
    xyz_to_srgb(&[xyz], true, &mut srgb);
    srgb[0]
}
```

## Tests
Golden parity with/without adaptation on a synthetic SPD + CMF + illuminant grid.
