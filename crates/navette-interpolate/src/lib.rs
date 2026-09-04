//! Pure-Rust univariate interpolation core (no Python, no I/O).
//!
//! Batch-aware kernels with rayon-parallel evaluation. The Python bindings
//! live in the `navette-py` aggregator crate and expose [`UniInterpolator`]
//! as `navette._interpolate.UniInterpolator`.
//!
//! # Methods
//!
//! `"pchip"` (shape-preserving cubic Hermite), `"makima"` (modified
//! Akima), `"sprague"` (5th-order, needs ≥ 6 points), `"floater_hormann"`
//!/`"fh"` (barycentric rational, degree `d`), and `"linear"`.
//! `deriv` selects value (0) or first-derivative output where the method
//! supports it. Out-of-range queries follow [`ExtrapMode`].

use ndarray::{Array1, Array2};
use rayon::prelude::*;

// Threshold above which a single-signal evaluation is split across threads.
const PAR_TARGET_THRESHOLD: usize = 8_192;
// Minimum work per thread chunk, to keep scheduling overhead negligible.
const MIN_PAR_CHUNK: usize = 1_024;

// -------------------------------------------------------------------------
// Auxiliary data
// -------------------------------------------------------------------------
#[derive(Clone)]
enum AuxData {
    None,
    Slopes(Array2<f64>),
    FHWeights(Array1<f64>),
}

// -------------------------------------------------------------------------
// Extrapolation modes
// -------------------------------------------------------------------------
/// Behaviour for query points outside the knot range.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtrapMode {
    Linear,
    Clamp,
    Error,
}

impl ExtrapMode {
    /// Parse `"linear"`, `"clamp"` or `"error"` (case-insensitive).
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(ExtrapMode::Linear),
            "clamp" => Ok(ExtrapMode::Clamp),
            "error" => Ok(ExtrapMode::Error),
            _ => Err("extrap must be 'linear', 'clamp', or 'error'".to_string()),
        }
    }
    /// Canonical lowercase name, used by the bindings for round-tripping.
    pub fn as_str(self) -> &'static str {
        match self {
            ExtrapMode::Linear => "linear",
            ExtrapMode::Clamp => "clamp",
            ExtrapMode::Error => "error",
        }
    }
}

// -------------------------------------------------------------------------
// Main spline struct
// -------------------------------------------------------------------------

/// Core interpolator (plain Rust; no Python types).
///
/// Owns strictly-increasing knots `x` and one row of values per signal in
/// `y`. Method-specific setup (Hermite slopes, Floater–Hormann weights) is
/// precomputed once in [`UniInterpolator::new`]; [`UniInterpolator::evaluate`]
/// then answers any number of query batches, rayon-parallelised.
#[derive(Clone)]
pub struct UniInterpolator {
    x: Array1<f64>,
    y: Array2<f64>,
    method: String,
    robust: bool,
    d: usize,
    is_batch: bool,
    aux_data: AuxData,
    extrap: ExtrapMode,
}

impl UniInterpolator {
    /// Build and validate an interpolator.
    ///
    /// `x` must be strictly increasing with ≥ 2 points; every row of `y`
    /// must match `x` in length. `method` is one of `"pchip"`, `"makima"`,
    /// `"sprague"` (needs ≥ 6 points), `"floater_hormann"`/`"fh"`, or
    /// `"linear"`. `d` is the Floater–Hormann degree (clamped to `n − 1` and
    /// ignored by other methods); `robust` selects guarded evaluation where
    /// available; `extrap` is `"linear"`, `"clamp"` or `"error"`.
    ///
    /// Returns `Err` describing the first violated contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: Array1<f64>,
        y: Array2<f64>,
        is_batch: bool,
        method: &str,
        robust: bool,
        mut d: usize,
        extrap: &str,
    ) -> Result<Self, String> {
        let method = method.to_lowercase();
        let extrap_mode = ExtrapMode::from_str(extrap)?;
        let n = x.len();
        if n < 2 {
            return Err("x must have at least 2 points".to_string());
        }
        for i in 1..n {
            if x[i] <= x[i - 1] {
                return Err("x must be strictly increasing".to_string());
            }
        }
        if y.ncols() != n {
            return Err(format!("y row length ({}) must match x length ({})", y.ncols(), n));
        }
        if d >= n {
            d = n.saturating_sub(1);
        }
        match method.as_str() {
            "sprague" => {
                if n < 6 {
                    return Err("Sprague requires at least 6 points".to_string());
                }
            }
            "pchip" | "makima" | "floater_hormann" | "fh" | "linear" => {}
            _ => return Err(format!("Unknown method: {}", method)),
        }
        let aux_data = match method.as_str() {
            "pchip" => AuxData::Slopes(calc_pchip_slopes(&x, &y)),
            "makima" => AuxData::Slopes(calc_makima_slopes(&x, &y)),
            "floater_hormann" | "fh" => AuxData::FHWeights(calc_fh_weights(&x, d)),
            _ => AuxData::None,
        };
        Ok(Self { x, y, method, robust, d, is_batch, aux_data, extrap: extrap_mode })
    }

    /// Evaluate all signals at `tgt_x`, returning `(n_signals, n_tgt)` values.
    ///
    /// `deriv` selects values (0) or first derivatives (1) where supported.
    /// `sorted_hint` skips the sortedness check (`None` auto-detects, letting
    /// the kernel take the faster sorted path). Single-signal grids above
    /// `PAR_TARGET_THRESHOLD` points split across threads; multi-signal
    /// batches parallelise over rows.
    pub fn evaluate(&self, tgt_x: &[f64], deriv: usize, sorted_hint: Option<bool>) -> Array2<f64> {
        let n_tgt = tgt_x.len();
        let n_signals = self.y.nrows();
        let mut out = Array2::<f64>::zeros((n_signals, n_tgt));
        if n_tgt == 0 { return out; }
        let is_sorted = sorted_hint.unwrap_or_else(|| is_sorted_slice(tgt_x));
        let x_slice = self.x.as_slice().unwrap();
        let method_str = self.method.as_str();
        let robust = self.robust;
        let extrap = self.extrap;
        if n_signals == 1 {
            let y_view = self.y.row(0);
            let y_slice = y_view.as_slice().unwrap();
            let slopes_row0 = match &self.aux_data { AuxData::Slopes(s) => Some(s.row(0)), _ => None };
            let d_opt = slopes_row0.as_ref().map(|r| r.as_slice().unwrap());
            let w_opt = match &self.aux_data { AuxData::FHWeights(w) => Some(w.as_slice().unwrap()), _ => None };
            let out_flat = out.as_slice_mut().unwrap();
            if n_tgt >= PAR_TARGET_THRESHOLD {
                let nthreads = rayon::current_num_threads().max(1);
                let chunk = n_tgt.div_ceil(nthreads).max(MIN_PAR_CHUNK);
                out_flat.par_chunks_mut(chunk).zip(tgt_x.par_chunks(chunk)).for_each(|(o, t)| {
                    run_kernel(method_str, robust, t, x_slice, y_slice, d_opt, w_opt, o, deriv, is_sorted, extrap);
                });
            } else {
                run_kernel(method_str, robust, tgt_x, x_slice, y_slice, d_opt, w_opt, out_flat, deriv, is_sorted, extrap);
            }
        } else {
            let slopes_ref = match &self.aux_data { AuxData::Slopes(s) => Some(s), _ => None };
            let w_opt = match &self.aux_data { AuxData::FHWeights(w) => Some(w.as_slice().unwrap()), _ => None };
            out.as_slice_mut().unwrap().par_chunks_exact_mut(n_tgt).enumerate().for_each(|(k, out_slice)| {
                let y_view = self.y.row(k);
                let y_slice = y_view.as_slice().unwrap();
                let d_view = slopes_ref.map(|s| s.row(k));
                let d_opt = d_view.as_ref().map(|r| r.as_slice().unwrap());
                run_kernel(method_str, robust, tgt_x, x_slice, y_slice, d_opt, w_opt, out_slice, deriv, is_sorted, extrap);
            });
        }
        out
    }

    /// Cloned knot vector (binding accessor).
    pub fn x_clone(&self) -> Array1<f64> { self.x.clone() }
    /// Cloned value rows (binding accessor).
    pub fn y_clone(&self) -> Array2<f64> { self.y.clone() }
    pub fn slopes_clone(&self) -> Option<Array2<f64>> {
        match &self.aux_data { AuxData::Slopes(s) => Some(s.clone()), _ => None }
    }
    /// Canonical method name chosen at construction.
    pub fn method(&self) -> &str { &self.method }
    pub fn robust(&self) -> bool { self.robust }
    pub fn fh_d(&self) -> usize { self.d }
    pub fn is_batch(&self) -> bool { self.is_batch }
    pub fn extrap_str(&self) -> &'static str { self.extrap.as_str() }
}

// Single-signal dispatcher (operates purely on slices, no Python state)
// -------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
/// Dispatch one signal to the active method kernel (values or 1st derivatives).
fn run_kernel(
    method: &str,
    robust: bool,
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d_opt: Option<&[f64]>,
    w_opt: Option<&[f64]>,
    out: &mut [f64],
    deriv: usize,
    sorted: bool,
    extrap: ExtrapMode,
) {
    match method {
        "linear" => {
            if deriv == 0 && sorted {
                eval_linear_sorted(tgt_x, x, y, out, extrap);
            } else {
                eval_linear_general(tgt_x, x, y, out, deriv, extrap);
            }
        }
        "pchip" | "makima" => {
            let d = d_opt.expect("hermite slopes missing");
            if deriv == 0 && sorted {
                eval_hermite_sorted(tgt_x, x, y, d, out, extrap);
            } else {
                eval_hermite_general(tgt_x, x, y, d, out, deriv, extrap);
            }
        }
        "sprague" => {
            if deriv == 0 {
                if sorted {
                    eval_sprague_sorted(tgt_x, x, y, out, extrap);
                } else {
                    eval_sprague_general(tgt_x, x, y, out, robust, extrap);
                }
            } else {
                finite_diff(method, robust, tgt_x, x, y, d_opt, w_opt, out, sorted, extrap);
            }
        }
        "floater_hormann" | "fh" => {
            let w = w_opt.expect("Floater-Hormann weights missing");
            if deriv == 0 {
                eval_fh(tgt_x, x, y, w, out, extrap);
            } else {
                finite_diff(method, robust, tgt_x, x, y, d_opt, w_opt, out, sorted, extrap);
            }
        }
        _ => {}
    }
}

/// Central-difference fallback for methods without an analytic derivative.
#[allow(clippy::too_many_arguments)]
/// Central/one-sided finite difference used for endpoint Hermite slopes.
fn finite_diff(
    method: &str,
    robust: bool,
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d_opt: Option<&[f64]>,
    w_opt: Option<&[f64]>,
    out: &mut [f64],
    sorted: bool,
    extrap: ExtrapMode,
) {
    let n = tgt_x.len();
    let span = (x[x.len() - 1] - x[0]).abs().max(1.0);
    let h = 1e-6 * span;
    let inv = 1.0 / (2.0 * h);

    let tp: Vec<f64> = tgt_x.iter().map(|&v| v + h).collect();
    let tm: Vec<f64> = tgt_x.iter().map(|&v| v - h).collect();
    let mut yp = vec![0.0; n];
    let mut ym = vec![0.0; n];

    run_kernel(method, robust, &tp, x, y, d_opt, w_opt, &mut yp, 0, sorted, extrap);
    run_kernel(method, robust, &tm, x, y, d_opt, w_opt, &mut ym, 0, sorted, extrap);

    for i in 0..n {
        out[i] = (yp[i] - ym[i]) * inv;
    }
}

/// True when `data` is non-decreasing (selects the fast sorted kernels).
fn is_sorted_slice(data: &[f64]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

// =============================================================================
// KERNELS
// =============================================================================

#[inline]
/// Out-of-range value under `extrap`: linear extension, clamp, or NaN+flag for `Error`.
fn extrap_value(
    xi: f64,
    x: &[f64],
    y: &[f64],
    n: usize,
    left: bool,
    extrap: ExtrapMode,
) -> f64 {
    match extrap {
        ExtrapMode::Linear => {
            if left {
                let dx = x[1] - x[0];
                let dy = y[1] - y[0];
                if dx != 0.0 { y[0] + dy * (xi - x[0]) / dx } else { y[0] }
            } else {
                let dx = x[n - 1] - x[n - 2];
                let dy = y[n - 1] - y[n - 2];
                if dx != 0.0 { y[n - 1] + dy * (xi - x[n - 1]) / dx } else { y[n - 1] }
            }
        }
        ExtrapMode::Clamp => if left { y[0] } else { y[n - 1] },
        ExtrapMode::Error => f64::NAN,
    }
}

/// Piecewise-linear interpolation; `tgt_x` ascending.
fn eval_linear_sorted(tgt_x: &[f64], x: &[f64], y: &[f64], out: &mut [f64], extrap: ExtrapMode) {
    let n = x.len();
    let mut j = 0;
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi < x[0] {
            out[i] = extrap_value(xi, x, y, n, true, extrap);
            continue;
        }
        if xi > x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, false, extrap);
            continue;
        }
        while j < n - 1 && xi > x[j + 1] {
            j += 1;
        }
        let dx = x[j + 1] - x[j];
        let t = if dx != 0.0 { (xi - x[j]) / dx } else { 0.0 };
        out[i] = y[j] * (1.0 - t) + y[j + 1] * t;
    }
}

/// Piecewise-linear interpolation; `tgt_x` in any order.
fn eval_linear_general(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    out: &mut [f64],
    deriv: usize,
    extrap: ExtrapMode,
) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = if deriv == 0 {
                extrap_value(xi, x, y, n, true, extrap)
            } else {
                let dx = x[1] - x[0];
                if dx != 0.0 { (y[1] - y[0]) / dx } else { 0.0 }
            };
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = if deriv == 0 {
                extrap_value(xi, x, y, n, false, extrap)
            } else {
                let dx = x[n - 1] - x[n - 2];
                if dx != 0.0 { (y[n - 1] - y[n - 2]) / dx } else { 0.0 }
            };
            continue;
        }
        let lo = lower_bound(x, xi);
        let dx = x[lo + 1] - x[lo];
        out[i] = if deriv == 0 {
            let t = if dx != 0.0 { (xi - x[lo]) / dx } else { 0.0 };
            y[lo] * (1.0 - t) + y[lo + 1] * t
        } else if dx != 0.0 {
            (y[lo + 1] - y[lo]) / dx
        } else {
            0.0
        };
    }
}

/// Cubic Hermite interpolation with precomputed slopes; `tgt_x` ascending.
fn eval_hermite_sorted(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d: &[f64],
    out: &mut [f64],
    extrap: ExtrapMode,
) {
    let n = x.len();
    let mut j = 0;
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi < x[0] {
            out[i] = match extrap {
                ExtrapMode::Linear => y[0] + d[0] * (xi - x[0]),
                ExtrapMode::Clamp => y[0],
                ExtrapMode::Error => f64::NAN,
            };
            continue;
        }
        if xi > x[n - 1] {
            out[i] = match extrap {
                ExtrapMode::Linear => y[n - 1] + d[n - 1] * (xi - x[n - 1]),
                ExtrapMode::Clamp => y[n - 1],
                ExtrapMode::Error => f64::NAN,
            };
            continue;
        }
        while j < n - 1 && xi > x[j + 1] {
            j += 1;
        }
        let h = x[j + 1] - x[j];
        if h == 0.0 {
            out[i] = y[j];
            continue;
        }
        let t = (xi - x[j]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        out[i] = h00 * y[j] + h10 * h * d[j] + h01 * y[j + 1] + h11 * h * d[j + 1];
    }
}

/// Cubic Hermite interpolation with precomputed slopes; `tgt_x` in any order.
fn eval_hermite_general(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    d: &[f64],
    out: &mut [f64],
    deriv: usize,
    extrap: ExtrapMode,
) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = if deriv == 0 {
                match extrap {
                    ExtrapMode::Linear => y[0] + d[0] * (xi - x[0]),
                    ExtrapMode::Clamp => y[0],
                    ExtrapMode::Error => f64::NAN,
                }
            } else {
                d[0]
            };
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = if deriv == 0 {
                match extrap {
                    ExtrapMode::Linear => y[n - 1] + d[n - 1] * (xi - x[n - 1]),
                    ExtrapMode::Clamp => y[n - 1],
                    ExtrapMode::Error => f64::NAN,
                }
            } else {
                d[n - 1]
            };
            continue;
        }
        let lo = lower_bound(x, xi);
        let h = x[lo + 1] - x[lo];
        if h == 0.0 {
            out[i] = if deriv == 0 { y[lo] } else { 0.0 };
            continue;
        }
        let t = (xi - x[lo]) / h;
        if deriv == 0 {
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            out[i] = h00 * y[lo] + h10 * h * d[lo] + h01 * y[lo + 1] + h11 * h * d[lo + 1];
        } else {
            let dt = 6.0 * t * t - 6.0 * t;
            let d_h00 = dt;
            let d_h10 = 3.0 * t * t - 4.0 * t + 1.0;
            let d_h01 = -dt;
            let d_h11 = 3.0 * t * t - 2.0 * t;
            out[i] =
                (d_h00 * y[lo] + d_h10 * h * d[lo] + d_h01 * y[lo + 1] + d_h11 * h * d[lo + 1]) / h;
        }
    }
}

/// Sprague (6-point local) interpolation for *sorted* queries.
/// Uses a marching window and caches the barycentric weights, recomputing them
/// only when the active 6-point window changes (huge win for dense queries).
fn eval_sprague_sorted(tgt_x: &[f64], x: &[f64], y: &[f64], out: &mut [f64], extrap: ExtrapMode) {
    let n = x.len();
    let mut node_ptr = 0usize; // count of source nodes <= xi (searchsorted 'right')
    let mut cur_start = usize::MAX;
    let mut w = [0.0f64; 6];
    let mut xloc = [0.0f64; 6];
    let mut yloc = [0.0f64; 6];

    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = extrap_value(xi, x, y, n, true, extrap);
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, false, extrap);
            continue;
        }

        while node_ptr < n && x[node_ptr] <= xi {
            node_ptr += 1;
        }
        let idx = node_ptr.max(1);
        let mut w_start = idx.saturating_sub(3);
        let max_start = n - 6;
        if w_start > max_start {
            w_start = max_start;
        }

        if w_start != cur_start {
            cur_start = w_start;
            xloc.copy_from_slice(&x[w_start..w_start + 6]);
            yloc.copy_from_slice(&y[w_start..w_start + 6]);
            for j in 0..6 {
                let mut wj = 1.0;
                for k in 0..6 {
                    if k != j {
                        wj /= xloc[j] - xloc[k];
                    }
                }
                w[j] = wj;
            }
        }

        let mut num = 0.0;
        let mut den = 0.0;
        let mut hit = false;
        for j in 0..6 {
            let diff = xi - xloc[j];
            if diff == 0.0 {
                out[i] = yloc[j];
                hit = true;
                break;
            }
            let term = w[j] / diff;
            num += term * yloc[j];
            den += term;
        }
        if !hit {
            out[i] = if den != 0.0 { num / den } else { 0.0 };
        }
    }
}

/// Sprague interpolation for arbitrary (possibly unsorted) queries.
/// Honors `robust` (barycentric vs naive Lagrange) so the two formulations
/// can be cross-validated against one another.
fn eval_sprague_general(
    tgt_x: &[f64],
    x: &[f64],
    y: &[f64],
    out: &mut [f64],
    robust: bool,
    extrap: ExtrapMode,
) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi <= x[0] {
            out[i] = extrap_value(xi, x, y, n, true, extrap);
            continue;
        }
        if xi >= x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, false, extrap);
            continue;
        }

        let idx = x.partition_point(|&v| v <= xi).max(1);
        let mut w_start = idx.saturating_sub(3);
        let max_start = n - 6;
        if w_start > max_start {
            w_start = max_start;
        }

        let x_loc = &x[w_start..w_start + 6];
        let y_loc = &y[w_start..w_start + 6];

        if robust {
            // Barycentric Lagrange formulation.
            let mut w = [1.0f64; 6];
            for j in 0..6 {
                for k in 0..6 {
                    if k != j {
                        w[j] /= x_loc[j] - x_loc[k];
                    }
                }
            }
            let mut num = 0.0;
            let mut den = 0.0;
            let mut hit = false;
            for j in 0..6 {
                let diff = xi - x_loc[j];
                if diff == 0.0 {
                    out[i] = y_loc[j];
                    hit = true;
                    break;
                }
                let term = w[j] / diff;
                num += term * y_loc[j];
                den += term;
            }
            if !hit {
                out[i] = if den != 0.0 { num / den } else { 0.0 };
            }
        } else {
            // Naive Lagrange expansion.
            let mut res = 0.0;
            for j in 0..6 {
                let mut basis = 1.0;
                let xj = x_loc[j];
                for k in 0..6 {
                    if k != j {
                        basis *= (xi - x_loc[k]) / (xj - x_loc[k]);
                    }
                }
                res += y_loc[j] * basis;
            }
            out[i] = res;
        }
    }
}

/// Floater–Hormann barycentric rational interpolation with weights `w`.
fn eval_fh(tgt_x: &[f64], x: &[f64], y: &[f64], w: &[f64], out: &mut [f64], extrap: ExtrapMode) {
    let n = x.len();
    for (i, &xi) in tgt_x.iter().enumerate() {
        if xi < x[0] || xi > x[n - 1] {
            out[i] = extrap_value(xi, x, y, n, xi < x[0], extrap);
            continue;
        }
        let mut num = 0.0;
        let mut den = 0.0;
        let mut hit = false;
        for k in 0..n {
            let diff = xi - x[k];
            if diff == 0.0 {
                out[i] = y[k];
                hit = true;
                break;
            }
            let term = w[k] / diff;
            num += term * y[k];
            den += term;
        }
        if !hit {
            out[i] = if den != 0.0 { num / den } else { 0.0 };
        }
    }
}

// -------------------------------------------------------------------------
// Small helpers
// -------------------------------------------------------------------------

/// Returns `lo` such that `x[lo] <= xi < x[lo+1]`, assuming `x[0] < xi < x[n-1]`.
#[inline]
/// First index `i` with `x[i] >= xi` (binary search over ascending knots).
fn lower_bound(x: &[f64], xi: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = x.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xi < x[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

// -------------------------------------------------------------------------
// Precomputation kernels
// -------------------------------------------------------------------------
/// Fritsch–Carlson shape-preserving slopes, one row per signal.
fn calc_pchip_slopes(x: &Array1<f64>, y_batch: &Array2<f64>) -> Array2<f64> {
    let n = x.len();
    let mut slopes = Array2::<f64>::zeros(y_batch.raw_dim());
    let x_slice = x.as_slice().unwrap();
    let y_batch_slice = y_batch.as_slice().unwrap();

    if n == 2 {
        let dx = x_slice[1] - x_slice[0];
        if dx != 0.0 {
            slopes
                .as_slice_mut()
                .unwrap()
                .par_chunks_exact_mut(n)
                .zip(y_batch_slice.par_chunks_exact(n))
                .for_each(|(d, y)| {
                    let s0 = (y[1] - y[0]) / dx;
                    d[0] = s0;
                    d[1] = s0;
                });
        }
        return slopes;
    }

    slopes
        .as_slice_mut()
        .unwrap()
        .par_chunks_exact_mut(n)
        .zip(y_batch_slice.par_chunks_exact(n))
        .for_each(|(d, y)| {
            let mut h = vec![0.0; n - 1];
            let mut delta = vec![0.0; n - 1];
            for i in 0..n - 1 {
                h[i] = x_slice[i + 1] - x_slice[i];
                delta[i] = if h[i] != 0.0 {
                    (y[i + 1] - y[i]) / h[i]
                } else {
                    0.0
                };
            }
            for k in 1..n - 1 {
                if delta[k - 1] * delta[k] > 0.0 {
                    let w1 = 2.0 * h[k] + h[k - 1];
                    let w2 = h[k] + 2.0 * h[k - 1];
                    d[k] = (w1 + w2) * delta[k - 1] * delta[k]
                        / (w1 * delta[k] + w2 * delta[k - 1]);
                }
            }
            let end_deriv = |h0: f64, h1: f64, del0: f64, del1: f64| -> f64 {
                let d_val = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
                if d_val.signum() != del0.signum() {
                    return 0.0;
                }
                if (del0.signum() != del1.signum()) && (d_val.abs() > 3.0 * del0.abs()) {
                    return 3.0 * del0;
                }
                d_val
            };
            d[0] = end_deriv(h[0], h[1], delta[0], delta[1]);
            d[n - 1] = end_deriv(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        });
    slopes
}

/// Modified Akima slopes (less overshoot than classic Akima), one row per signal.
fn calc_makima_slopes(x: &Array1<f64>, y_batch: &Array2<f64>) -> Array2<f64> {
    let n = x.len();
    let mut slopes = Array2::<f64>::zeros(y_batch.raw_dim());
    let x_slice = x.as_slice().unwrap();
    let y_batch_slice = y_batch.as_slice().unwrap();

    if n == 2 {
        let dx = x_slice[1] - x_slice[0];
        if dx != 0.0 {
            slopes
                .as_slice_mut()
                .unwrap()
                .par_chunks_exact_mut(n)
                .zip(y_batch_slice.par_chunks_exact(n))
                .for_each(|(d, y)| {
                    let s0 = (y[1] - y[0]) / dx;
                    d[0] = s0;
                    d[1] = s0;
                });
        }
        return slopes;
    }

    slopes
        .as_slice_mut()
        .unwrap()
        .par_chunks_exact_mut(n)
        .zip(y_batch_slice.par_chunks_exact(n))
        .for_each(|(s, y)| {
            let mut deltas = vec![0.0; n - 1];
            for i in 0..n - 1 {
                let dx = x_slice[i + 1] - x_slice[i];
                deltas[i] = if dx != 0.0 {
                    (y[i + 1] - y[i]) / dx
                } else {
                    0.0
                };
            }
            let mut d = vec![0.0; n + 3];
            d[2..n + 1].copy_from_slice(&deltas);
            d[1] = 2.0 * deltas[0] - deltas[1];
            d[0] = 2.0 * d[1] - deltas[0];
            d[n + 1] = 2.0 * deltas[n - 2] - deltas[n - 3];
            d[n + 2] = 2.0 * d[n + 1] - deltas[n - 2];

            for i in 0..n {
                let w1 = f64::abs(d[i + 3] - d[i + 2]) + f64::abs(d[i + 3] + d[i + 2]) * 0.5;
                let w2 = f64::abs(d[i + 1] - d[i]) + f64::abs(d[i + 1] + d[i]) * 0.5;
                let w_sum = w1 + w2;
                s[i] = if w_sum == 0.0 {
                    0.5 * (d[i + 1] + d[i + 2])
                } else {
                    (w1 * d[i + 1] + w2 * d[i + 2]) / w_sum
                };
            }
        });
    slopes
}

/// Floater–Hormann barycentric weights of degree `d` on knots `x`.
fn calc_fh_weights(x: &Array1<f64>, d: usize) -> Array1<f64> {
    let n = x.len();
    let mut w = Array1::<f64>::zeros(n);
    let x_slice = x.as_slice().unwrap();
    for k in 0..n {
        let mut s_val = 0.0;
        let i_min = k.saturating_sub(d);
        let i_max = k.min(n.saturating_sub(d + 1));
        for i in i_min..=i_max {
            let mut prod = 1.0;
            for j in i..=(i + d) {
                if j != k {
                    prod *= 1.0 / (x_slice[k] - x_slice[j]).abs();
                }
            }
            s_val += prod;
        }
        w[k] = if k % 2 == 1 { -s_val } else { s_val };
    }
    w
}

// -------------------------------------------------------------------------
// Module initialisation
// -------------------------------------------------------------------------
