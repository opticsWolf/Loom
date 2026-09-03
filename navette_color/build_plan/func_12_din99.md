# rust-codegen-worker Task — Unit func_12

## Unit Info
- **ID:** func_12
- **Name:** DIN99 Color Difference (DIN 6176)
- **Kind:** batch metric
- **Jit Type:** `rayon`

## Critical detail — the full DIN99 coordinate transform
Both colors are mapped to DIN99 space, then a plain Euclidean distance is taken.
The per-color transform (16° rotation + log compression):

- `L99 = 105.51 · ln(1 + 0.0158·L) · kE`
- `e =  a·cos16 + b·sin16`
- `f = 0.7·(−a·sin16 + b·cos16)`   (the 0.7 flattening is essential)
- `G = hypot(e, f)`
- if `G < 1e-12`: `C99 = 0, h99 = 0`; else
  `C99 = ln(1 + 0.045·G)·kCH`, `h99 = atan2(f, e)`
- `a99 = C99·cos(h99)`, `b99 = C99·sin(h99)`
- `ΔE99 = √(ΔL99² + Δa99² + Δb99²)`

Presets: graphic `kE=1, kCH=1`; textiles `kE=2, kCH=0.5`. `cos16/sin16` are
`cos/sin(16°)`.

> CORRECTION: earlier draft only said "rotate a,b by 16° and log-scale L,C". The
> 0.7 factor on `f`, the `105.51`/`0.0158` and `0.045` constants, the `G<1e-12`
> guard, and the kE/kCH presets are all required for parity.

## Target Output Files

### src/func_12.rs
```rust
fn din99_coords(lab: &[f64; 3], ke: f64, kch: f64, cos16: f64, sin16: f64)
    -> (f64, f64, f64)
{
    let (l, a, b) = (lab[0], lab[1], lab[2]);
    let l99 = 105.51 * (1.0 + 0.0158 * l).ln() * ke;
    let e = a * cos16 + b * sin16;
    let f = 0.7 * (-a * sin16 + b * cos16);
    let g = (e*e + f*f).sqrt();
    let (c99, h99) = if g < 1e-12 { (0.0, 0.0) }
                     else { ((1.0 + 0.045 * g).ln() * kch, f.atan2(e)) };
    (l99, c99 * h99.cos(), c99 * h99.sin())
}

#[inline(always)]
pub fn delta_e_din99_single(lab1: &[f64;3], lab2: &[f64;3], ke: f64, kch: f64) -> f64 {
    let cos16 = 16f64.to_radians().cos();
    let sin16 = 16f64.to_radians().sin();
    let (l1,a1,b1) = din99_coords(lab1, ke, kch, cos16, sin16);
    let (l2,a2,b2) = din99_coords(lab2, ke, kch, cos16, sin16);
    let (dl,da,db) = (l1-l2, a1-a2, b1-b2);
    (dl*dl + da*da + db*db).sqrt()
}
```

## Tests
Golden parity for both presets; achromatic pair (a=b=0) hits the G<1e-12 branch.
