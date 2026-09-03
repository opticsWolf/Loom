# rust-codegen-worker Task — Unit func_14

## Unit Info
- **ID:** func_14
- **Name:** Photometry Engine
- **Kind:** integration
- **Jit Type:** N/A

> Photopic, scotopic, and mesopic luminous flux.

## Critical detail — constants, weights, and S/P ratio
Fused kernel: `total += spd[i] · (vp[i]·w_p + vs[i]·w_s)`, then `× interval`.

- Luminous-efficacy constants: photopic `Km_p = 683.002`, scotopic
  `Km_s = 1700.05`.
- Weight selection by vision mode:
  - **Photopic:** `w_p = Km_p, w_s = 0`.
  - **Scotopic:** `w_p = 0, w_s = Km_s`.
  - **Mesopic(m):** `w_p = m·Km_p`, `w_s = (1−m)·Km_s`, with `m ∈ [0,1]`
    (1 = photopic, 0 = scotopic).
- **S/P ratio:** `scotopic_flux / photopic_flux`, returning `0.0` when
  `photopic_flux < 1e-12`.

> CORRECTION: earlier draft described only the fused-loop shape. The exact
> constants (`683.002`, `1700.05`), the mesopic blend, and the S/P guard are
> parity-critical.

## Target Output Files

### src/func_14.rs
```rust
pub struct PhotometryEngine { vp: Vec<f64>, vs: Vec<f64>, pub km_p: f64, pub km_s: f64 }

pub enum Vision { Photopic, Scotopic, Mesopic } // Mesopic carries m via the call

impl PhotometryEngine {
    pub fn new(v_photopic: Vec<f64>, v_scotopic: Vec<f64>) -> Self {
        Self::with_constants(v_photopic, v_scotopic, 683.002, 1700.05)
    }
    fn flux_kernel(&self, spd: &[f64], w_p: f64, w_s: f64, interval: f64) -> f64 {
        let mut total = 0.0;
        for i in 0..spd.len() { total += spd[i]*(self.vp[i]*w_p + self.vs[i]*w_s); }
        total * interval
    }
    pub fn calculate_flux(&self, spd: &[f64], vision: Vision, m: f64, interval: f64) -> f64 {
        let (w_p, w_s) = match vision {
            Vision::Photopic => (self.km_p, 0.0),
            Vision::Scotopic => (0.0, self.km_s),
            Vision::Mesopic  => (m*self.km_p, (1.0-m)*self.km_s),
        };
        self.flux_kernel(spd, w_p, w_s, interval)
    }
    pub fn calculate_sp_ratio(&self, spd: &[f64], interval: f64) -> f64 {
        let p = self.flux_kernel(spd, self.km_p, 0.0, interval);
        if p < 1e-12 { return 0.0; }
        self.flux_kernel(spd, 0.0, self.km_s, interval) / p
    }
}
```

## Tests
Golden parity for photopic/scotopic/mesopic flux + S/P ratio on a sample SPD.
