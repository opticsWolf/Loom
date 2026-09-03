# rust-codegen-worker Task — Unit func_10

## Unit Info
- **ID:** func_10
- **Name:** Delta E 94
- **Kind:** batch metric (asymmetric)
- **Jit Type:** `rayon`

## Critical detail — asymmetry, weights from the reference, dH² clamp
- **Asymmetric:** `lab1` is the *reference*, `lab2` the *sample*. The weighting
  terms `S_C = 1 + K1·C1` and `S_H = 1 + K2·C1` use the **reference** chroma
  `C1` (from `lab1`), so swapping the arguments changes the result.
- `dH² = da² + db² − dC²` can go slightly negative from FP noise → clamp `≥ 0`.
- `S_L = 1`. Distance `= √[(dL/(kL·S_L))² + (dC/S_C)² + dH²/S_H²]`.
- **Parameter presets:** graphic arts `kL=1, K1=0.045, K2=0.015`;
  textiles `kL=2, K1=0.048, K2=0.014`.

> CORRECTION: the earlier draft noted the asymmetry/clamp but not that the S_C/S_H
> weights are computed from `C1` (reference), nor the two named parameter presets.
> Both are required for parity.

## Target Output Files

### src/func_10.rs
```rust
#[derive(Clone, Copy)]
pub struct De94Params { pub k_l: f64, pub k1: f64, pub k2: f64 }
impl De94Params {
    pub const GRAPHIC:  De94Params = De94Params { k_l: 1.0, k1: 0.045, k2: 0.015 };
    pub const TEXTILES: De94Params = De94Params { k_l: 2.0, k1: 0.048, k2: 0.014 };
}

#[inline(always)]
pub fn delta_e_94_single(lab1: &[f64; 3], lab2: &[f64; 3], p: De94Params) -> f64 {
    let dl = lab1[0] - lab2[0];
    let c1 = (lab1[1]*lab1[1] + lab1[2]*lab1[2]).sqrt();
    let c2 = (lab2[1]*lab2[1] + lab2[2]*lab2[2]).sqrt();
    let dc = c1 - c2;
    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    let dh_sq = (da*da + db*db - dc*dc).max(0.0);
    let sc = 1.0 + p.k1 * c1;      // reference chroma
    let sh = 1.0 + p.k2 * c1;
    let tl = dl / p.k_l; let tc = dc / sc;
    (tl*tl + tc*tc + dh_sq/(sh*sh)).sqrt()
}

pub fn delta_e_94(lab1: &[[f64; 3]], lab2: &[[f64; 3]], p: De94Params) -> Vec<f64> {
    crate::metrics::map_pairs(lab1, lab2, |a, b| delta_e_94_single(a, b, p))
}
```

## Tests
Golden parity for both presets; explicit asymmetry check (swap args → differ).
