//! FFT Kramers–Kronig (Hilbert) transform — shared by Cody-Lorentz and UBF.
//!
//! ε₁ via an FFT Hilbert transform with odd extension. Faithful port of
//! `_kk_fft`: build a padded buffer with the ε₂ spectrum and its negated
//! reflection, forward rfft, multiply by the Hilbert kernel (H[0]=0, H[k]=−i),
//! inverse rfft, and read back the central window.
//!
//! Two details make this match NumPy's `irfft` exactly rather than only
//! approximately:
//!   * `realfft`'s inverse (c2r) is UNNORMALISED whereas `np.fft.irfft`
//!     divides by M — so we scale the inverse output by 1/M.
//!   * NumPy's real inverse discards the imaginary parts of the DC and Nyquist
//!     bins; we zero them explicitly (the Hilbert kernel leaves the Nyquist bin
//!     purely imaginary, which `realfft` would otherwise reject).
//!
//! The plan, grid size and pad length are computed once behind a `OnceLock`,
//! mirroring the Python module-level precompute.

use std::sync::{Arc, OnceLock};

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Number of points on the KK integration grid.
pub const GRID_N: usize = 8192;
/// Lower energy bound of the KK grid [eV].
pub const GRID_MIN: f64 = 0.01;
/// Upper energy bound of the KK grid [eV].
pub const GRID_MAX: f64 = 80.0;

/// Zero-padded transform length: next power of two ≥ 2·N+1 (= 32768).
fn pad_len() -> usize {
    let mut m = 1usize;
    while m < 2 * GRID_N + 1 {
        m <<= 1;
    }
    m
}

struct KkPlan {
    m: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
}

static KK: OnceLock<KkPlan> = OnceLock::new();

fn plan() -> &'static KkPlan {
    KK.get_or_init(|| {
        let m = pad_len();
        let mut planner = RealFftPlanner::<f64>::new();
        KkPlan {
            m,
            r2c: planner.plan_fft_forward(m),
            c2r: planner.plan_fft_inverse(m),
        }
    })
}

/// ε₁(grid) = eps_inf − Hilbert{ε₂}. `eps2` must have length [`GRID_N`].
pub fn kk_fft(eps2: &[f64], eps_inf: f64) -> Vec<f64> {
    assert_eq!(eps2.len(), GRID_N, "kk_fft expects GRID_N samples");
    let pl = plan();
    let n = GRID_N;
    let m = pl.m;

    // Odd extension: buf[1..=N] = -eps2[::-1], buf[N+1..2N+1] = eps2, rest 0.
    let mut buf = vec![0.0f64; m];
    for i in 0..n {
        buf[1 + i] = -eps2[n - 1 - i];
        buf[n + 1 + i] = eps2[i];
    }

    let mut spec = pl.r2c.make_output_vec(); // length m/2 + 1
    pl.r2c.process(&mut buf, &mut spec).expect("rfft");

    // Multiply by Hilbert kernel: H[0]=0, H[k≥1] = -i  →  (a+bi)·(-i) = b - a·i.
    spec[0].re = 0.0;
    spec[0].im = 0.0;
    for c in spec.iter_mut().skip(1) {
        let (a, b) = (c.re, c.im);
        c.re = b;
        c.im = -a;
    }
    // NumPy's irfft discards DC & Nyquist imaginary parts; match that (the
    // Nyquist bin is purely imaginary after ·(-i), so it must be zeroed).
    spec[0].im = 0.0;
    let last = spec.len() - 1;
    spec[last].im = 0.0;

    let mut hilb = pl.c2r.make_output_vec(); // length m
    pl.c2r.process(&mut spec, &mut hilb).expect("irfft");

    let inv_m = 1.0 / (m as f64);
    let mut eps1 = vec![0.0f64; n];
    for i in 0..n {
        eps1[i] = eps_inf - hilb[n + i] * inv_m;
    }
    eps1
}

/// NumPy-faithful `linspace(GRID_MIN, GRID_MAX, GRID_N)` (endpoint forced).
pub fn energy_grid() -> Vec<f64> {
    let n = GRID_N;
    let step = (GRID_MAX - GRID_MIN) / ((n - 1) as f64);
    let mut g: Vec<f64> = (0..n).map(|i| GRID_MIN + (i as f64) * step).collect();
    g[n - 1] = GRID_MAX;
    g
}

/// Linear interpolation matching `np.interp` (clamped at the endpoints).
/// `xp` ascending. Used to lift ε₁ from the KK grid onto target energies.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let i = xp.partition_point(|&v| v <= x) - 1; // last index with v ≤ x
    let (x0, x1, f0, f1) = (xp[i], xp[i + 1], fp[i], fp[i + 1]);
    f0 + (f1 - f0) * (x - x0) / (x1 - x0)
}
