# rust-codegen-worker Task — Unit func_11

## Unit Info
- **ID:** func_11
- **Name:** Delta E CMC(l:c)
- **Kind:** batch metric (asymmetric)
- **Jit Type:** `rayon`

## Critical detail — T-factor branch, F, S_L branch, defaults
All weighting factors depend on the **reference** sample `lab1` (chroma `C1`, hue
`h1 = atan2(b1,a1)` in degrees, wrapped `[0,360)`):

- `S_L = 0.511` if `L1 < 16`, else `0.040975·L1 / (1 + 0.01765·L1)`.
- `S_C = 0.0638·C1 / (1 + 0.0131·C1) + 0.638`.
- **T branch:** if `164 ≤ h1 ≤ 345`: `T = 0.56 + |0.2·cos(h1+168°)|`;
  otherwise `T = 0.36 + |0.4·cos(h1+35°)|`.
- `F = √(C1⁴ / (C1⁴ + 1900))`.
- `S_H = S_C·(F·T + 1 − F)`.
- `dH²` clamped `≥ 0` as in func_15.
- Distance `= √[(dL/(pl·S_L))² + (dC/(pc·S_C))² + dH²/S_H²]`.
- **Defaults `pl=2, pc=1`** (acceptability); `pl=1` for imperceptibility.

> CORRECTION: earlier draft mentioned only "T and F depend on reference hue" and
> the pl/pc defaults. The exact branch boundaries (`164..=345`), the offset
> angles (`+168°`, `+35°`), the `S_L` 16-threshold, and the `1900` constant are
> all parity-critical.

## Target Output Files

### src/func_11.rs
```rust
use crate::common::DEG2RAD;

#[inline(always)]
pub fn delta_e_cmc_single(lab1: &[f64; 3], lab2: &[f64; 3], pl: f64, pc: f64) -> f64 {
    let dl = lab1[0] - lab2[0];
    let c1 = (lab1[1]*lab1[1] + lab1[2]*lab1[2]).sqrt();
    let c2 = (lab2[1]*lab2[1] + lab2[2]*lab2[2]).sqrt();
    let dc = c1 - c2;
    let da = lab1[1]-lab2[1]; let db = lab1[2]-lab2[2];
    let dh_sq = (da*da + db*db - dc*dc).max(0.0);
    let h1 = lab1[2].atan2(lab1[1]).to_degrees().rem_euclid(360.0);
    let sl = if lab1[0] < 16.0 { 0.511 }
             else { 0.040975*lab1[0] / (1.0 + 0.01765*lab1[0]) };
    let sc = 0.0638*c1 / (1.0 + 0.0131*c1) + 0.638;
    let t = if (164.0..=345.0).contains(&h1) {
        0.56 + (0.2*((h1+168.0)*DEG2RAD).cos()).abs()
    } else {
        0.36 + (0.4*((h1+35.0)*DEG2RAD).cos()).abs()
    };
    let c1_4 = c1.powi(4);
    let f = (c1_4 / (c1_4 + 1900.0)).sqrt();
    let sh = sc * (f*t + 1.0 - f);
    let tl = dl/(pl*sl); let tc = dc/(pc*sc);
    (tl*tl + tc*tc + dh_sq/(sh*sh)).sqrt()
}
```

## Tests
Golden parity for default (2:1) and (1:1); hue near both branch edges.
