# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import numpy as np
from numba import njit, prange
from typing import Literal, Optional

__all__ = ["UniSpline"]

# ==============================================================================
# 0. LINEAR KERNELS (NEW ADDITION)
# ==============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def _eval_linear_1d(
    tgt_x: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    out: np.ndarray
):
    """Numba-optimized 1D Linear Interpolation with linear extrapolation.

    Args:
        tgt_x (np.ndarray): Target x-coordinates for evaluation.
        src_x (np.ndarray): Sorted source x-coordinates.
        src_y (np.ndarray): Source y-values corresponding to src_x.
        out (np.ndarray): Output buffer for interpolated values.
    """
    n_src = len(src_x)
    n_tgt = len(tgt_x)

    for i in prange(n_tgt):
        tx = tgt_x[i]
        
        # Binary search
        # side='right' returns i such that a[i-1] <= v < a[i]
        idx = np.searchsorted(src_x, tx, side='right') - 1
        
        # Handle boundaries for Linear Extrapolation
        if idx < 0:
            idx = 0
        elif idx >= n_src - 1:
            idx = n_src - 2
            
        x0 = src_x[idx]
        x1 = src_x[idx+1]
        y0 = src_y[idx]
        y1 = src_y[idx+1]
        
        # Linear interp formula: y = y0 + slope * (x - x0)
        # Division by zero protection not strictly needed if x is strictly increasing
        slope = (y1 - y0) / (x1 - x0)
        out[i] = y0 + slope * (tx - x0)

@njit(fastmath=True, cache=True, parallel=True)
def _eval_linear_batch(
    tgt_x: np.ndarray,
    src_x: np.ndarray,
    src_y_batch: np.ndarray,
    out_batch: np.ndarray
):
    """Batch Linear Interpolation parallelized over signals.

    Args:
        tgt_x (np.ndarray): Target x-coordinates for evaluation.
        src_x (np.ndarray): Sorted source x-coordinates.
        src_y_batch (np.ndarray): 2D array of source y-values [K signals, N points].
        out_batch (np.ndarray): 2D output buffer [K signals, M target points].
    """
    n_signals = src_y_batch.shape[0]
    n_src = len(src_x)
    n_tgt = len(tgt_x)

    # Parallelize over signals
    for k in prange(n_signals):
        # Serial loop over target points for this signal
        for i in range(n_tgt):
            tx = tgt_x[i]
            
            idx = np.searchsorted(src_x, tx, side='right') - 1
            
            if idx < 0:
                idx = 0
            elif idx >= n_src - 1:
                idx = n_src - 2
                
            x0 = src_x[idx]
            x1 = src_x[idx+1]
            
            # Access the specific signal row
            y0 = src_y_batch[k, idx]
            y1 = src_y_batch[k, idx+1]
            
            slope = (y1 - y0) / (x1 - x0)
            out_batch[k, i] = y0 + slope * (tx - x0)

# ==============================================================================
# 1. SHARED KERNELS (HERMITE EVALUATION)
# ==============================================================================

@njit(fastmath=True, cache=True)
def _hermite_basis(t: float) -> tuple[float, float, float, float]:
    """Calculates cubic Hermite basis functions for normalized t in [0, 1].

    Args:
        t (float): Normalized distance between 0 and 1.

    Returns:
        tuple[float, float, float, float]: The four Hermite basis coefficients.
    """
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0*t3 - 3.0*t2 + 1.0
    h10 = t3 - 2.0*t2 + t
    h01 = -2.0*t3 + 3.0*t2
    h11 = t3 - t2
    return h00, h10, h01, h11

@njit(fastmath=True, cache=True, parallel=True)
def _eval_cubic_hermite_1d(
    tgt_x: np.ndarray, 
    src_x: np.ndarray, 
    src_y: np.ndarray, 
    src_d: np.ndarray, 
    out: np.ndarray
):
    """Standard 1D Hermite evaluation parallelized over target points.

    Args:
        tgt_x (np.ndarray): Target x-coordinates.
        src_x (np.ndarray): Source x-coordinates.
        src_y (np.ndarray): Source y-values.
        src_d (np.ndarray): Pre-calculated slopes (derivatives) at source points.
        out (np.ndarray): Output buffer for interpolated values.
    """
    n_src = len(src_x)
    n_tgt = len(tgt_x)
    
    for i in prange(n_tgt):
        tx = tgt_x[i]
        
        # Linear Extrapolation (Left)
        if tx <= src_x[0]:
            out[i] = src_y[0] + src_d[0] * (tx - src_x[0])
            continue
        # Linear Extrapolation (Right)
        if tx >= src_x[n_src - 1]:
            out[i] = src_y[n_src - 1] + src_d[n_src - 1] * (tx - src_x[n_src - 1])
            continue
            
        # Binary Search
        idx = np.searchsorted(src_x, tx, side='right') - 1
        if idx < 0: idx = 0
        if idx >= n_src - 1: idx = n_src - 2
            
        # Interpolation
        x_k = src_x[idx]
        dx = src_x[idx+1] - x_k
        t = (tx - x_k) / dx
        
        h00, h10, h01, h11 = _hermite_basis(t)
        
        out[i] = (h00 * src_y[idx] + 
                  h10 * dx * src_d[idx] + 
                  h01 * src_y[idx+1] + 
                  h11 * dx * src_d[idx+1])

@njit(fastmath=True, cache=True, parallel=True)
def _eval_cubic_hermite_batch(
    tgt_x: np.ndarray, 
    src_x: np.ndarray, 
    src_y_batch: np.ndarray, 
    src_d_batch: np.ndarray, 
    out_batch: np.ndarray
):
    """Batch Hermite evaluation parallelized over signals (rows).

    Args:
        tgt_x (np.ndarray): Target x-coordinates.
        src_x (np.ndarray): Source x-coordinates.
        src_y_batch (np.ndarray): 2D array of source y-values.
        src_d_batch (np.ndarray): 2D array of source derivatives.
        out_batch (np.ndarray): 2D output buffer.
    """
    n_signals = src_y_batch.shape[0]
    n_src = len(src_x)
    n_tgt = len(tgt_x)
    
    # Loop over signals in parallel
    for k in prange(n_signals):
        # For each signal, loop over all target points (serial is fine here)
        for i in range(n_tgt):
            tx = tgt_x[i]
            
            # --- Linear Extrapolation ---
            if tx <= src_x[0]:
                out_batch[k, i] = src_y_batch[k, 0] + src_d_batch[k, 0] * (tx - src_x[0])
                continue
            if tx >= src_x[n_src - 1]:
                out_batch[k, i] = (src_y_batch[k, n_src - 1] + 
                                   src_d_batch[k, n_src - 1] * (tx - src_x[n_src - 1]))
                continue
                
            # --- Interpolation ---
            idx = np.searchsorted(src_x, tx, side='right') - 1
            if idx < 0: idx = 0
            if idx >= n_src - 1: idx = n_src - 2
            
            x_k = src_x[idx]
            dx = src_x[idx+1] - x_k
            t = (tx - x_k) / dx
            
            h00, h10, h01, h11 = _hermite_basis(t)
            
            y_k = src_y_batch[k, idx]
            y_k1 = src_y_batch[k, idx+1]
            d_k = src_d_batch[k, idx]
            d_k1 = src_d_batch[k, idx+1]
            
            out_batch[k, i] = (h00 * y_k + h10 * dx * d_k + h01 * y_k1 + h11 * dx * d_k1)

# ==============================================================================
# 2. PCHIP SLOPE CALCULATION
# ==============================================================================

@njit(fastmath=True, cache=True)
def _calc_pchip_derivs_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) derivatives.

    Args:
        x (np.ndarray): Sorted x-coordinates.
        y (np.ndarray): Corresponding y-values.

    Returns:
        np.ndarray: Derivatives at each source point.
    """
    n = len(x)
    d = np.zeros(n, dtype=np.float64)
    
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        d[:] = slope
        return d

    h = np.empty(n - 1)
    delta = np.empty(n - 1)
    for i in range(n - 1):
        h[i] = x[i+1] - x[i]
        delta[i] = (y[i+1] - y[i]) / h[i]

    # Interior
    for k in range(1, n - 1):
        if delta[k-1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k-1]
            w2 = h[k] + 2 * h[k-1]
            d[k] = (w1 + w2) * delta[k-1] * delta[k] / (w1 * delta[k] + w2 * delta[k-1])
        else:
            d[k] = 0.0

    # Endpoints
    def end_deriv(h0, h1, del0, del1):
        d_val = ((2*h0 + h1)*del0 - h0*del1) / (h0 + h1)
        if np.sign(d_val) != np.sign(del0): return 0.0
        if (np.sign(del0) != np.sign(del1)) and (abs(d_val) > 3*abs(del0)): return 3*del0
        return d_val

    d[0] = end_deriv(h[0], h[1], delta[0], delta[1])
    d[-1] = end_deriv(h[-1], h[-2], delta[-1], delta[-2])
    return d

@njit(fastmath=True, cache=True, parallel=True)
def _calc_pchip_derivs_batch(x: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """Computes PCHIP derivatives for multiple signals in parallel.

    Args:
        x (np.ndarray): Sorted x-coordinates.
        y_batch (np.ndarray): 2D array of source y-values.

    Returns:
        np.ndarray: 2D array of derivatives.
    """
    n_signals = y_batch.shape[0]
    n_pts = y_batch.shape[1]
    d_out = np.empty_like(y_batch)
    
    for k in prange(n_signals):
        d_out[k, :] = _calc_pchip_derivs_1d(x, y_batch[k, :])
    return d_out

# ==============================================================================
# 3. MAKIMA SLOPE CALCULATION
# ==============================================================================

@njit(fastmath=True, cache=True)
def _calc_makima_slopes_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Modified Akima (MAKIMA) derivative calculation.

    Args:
        x (np.ndarray): Sorted x-coordinates.
        y (np.ndarray): Corresponding y-values.

    Returns:
        np.ndarray: Derivatives at each source point.
    """
    n = len(x)
    # Secants
    deltas = np.empty(n - 1)
    for i in range(n - 1):
        dx = x[i+1] - x[i]
        deltas[i] = (y[i+1] - y[i]) / dx if dx != 0 else 0.0

    # Pad for extrapolation
    d = np.empty(n + 3)
    d[2:n+1] = deltas
    # Quadratic Extrap
    d[1] = 2.0*deltas[0] - deltas[1]
    d[0] = 2.0*d[1] - deltas[0]
    d[n+1] = 2.0*deltas[-1] - deltas[-2]
    d[n+2] = 2.0*d[n+1] - deltas[-1]

    slopes = np.empty(n)
    for i in range(n):
        w1 = abs(d[i+3] - d[i+2]) + abs(d[i+3] + d[i+2]) * 0.5
        w2 = abs(d[i+1] - d[i])   + abs(d[i+1] + d[i])   * 0.5
        w_sum = w1 + w2
        
        if w_sum == 0.0:
            slopes[i] = 0.5 * (d[i+1] + d[i+2])
        else:
            slopes[i] = (w1 * d[i+1] + w2 * d[i+2]) / w_sum
            
    return slopes

@njit(fastmath=True, cache=True, parallel=True)
def _calc_makima_slopes_batch(x: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """Computes MAKIMA derivatives for multiple signals in parallel.

    Args:
        x (np.ndarray): Sorted x-coordinates.
        y_batch (np.ndarray): 2D array of source y-values.

    Returns:
        np.ndarray: 2D array of derivatives.
    """
    n_signals = y_batch.shape[0]
    s_out = np.empty_like(y_batch)
    for k in prange(n_signals):
        s_out[k, :] = _calc_makima_slopes_1d(x, y_batch[k, :])
    return s_out

# ==============================================================================
# 4. SPRAGUE (LAGRANGE) EVALUATION
# ==============================================================================

@njit(fastmath=True, cache=True)
def _lagrange6_naive(x: float, xn: np.ndarray, yn: np.ndarray) -> float:
    """Standard Lagrange Polynomial expansion for 6 points.

    Args:
        x (float): Target point.
        xn (np.ndarray): Local neighborhood of 6 x-coordinates.
        yn (np.ndarray): Local neighborhood of 6 y-values.

    Returns:
        float: Interpolated value.
    """
    res = 0.0
    for j in range(6):
        basis = 1.0
        xj = xn[j]
        for k in range(6):
            if k != j:
                basis *= (x - xn[k]) / (xj - xn[k])
        res += yn[j] * basis
    return res

@njit(fastmath=True, cache=True)
def _lagrange6_barycentric(x: float, xn: np.ndarray, yn: np.ndarray) -> float:
    """Barycentric Lagrange Interpolation for 6 points.

    Args:
        x (float): Target point.
        xn (np.ndarray): Local neighborhood of 6 x-coordinates.
        yn (np.ndarray): Local neighborhood of 6 y-values.

    Returns:
        float: Numerically stable interpolated value.
    """
    # 1. Compute Barycentric Weights (O(k^2))
    w = np.ones(6, dtype=np.float64)
    for j in range(6):
        xj = xn[j]
        for k in range(6):
            if k != j:
                w[j] /= (xj - xn[k])

    # 2. Compute Barycentric Form
    numerator = 0.0
    denominator = 0.0
    
    for j in range(6):
        diff = x - xn[j]
        
        # Exact node hit check (Required for Barycentric form)
        if diff == 0.0:
            return yn[j]
        
        term = w[j] / diff
        numerator += term * yn[j]
        denominator += term
    
    if denominator == 0.0:
        return 0.0
        
    return numerator / denominator

@njit(fastmath=True, cache=True, parallel=True)
def _eval_sprague_1d_dispatch(
    tgt_x: np.ndarray, 
    src_x: np.ndarray, 
    src_y: np.ndarray, 
    out: np.ndarray,
    use_robust: bool
):
    """1D Sprague (Fifth-order) evaluation using a 6-point local window.

    Args:
        tgt_x (np.ndarray): Target x-coordinates.
        src_x (np.ndarray): Source x-coordinates.
        src_y (np.ndarray): Source y-values.
        out (np.ndarray): Output buffer.
        use_robust (bool): Whether to use the Barycentric form.
    """
    n_src = len(src_x)
    n_tgt = len(tgt_x)
    
    for i in prange(n_tgt):
        tx = tgt_x[i]
        idx = np.searchsorted(src_x, tx, side='right')
        w_start = idx - 3
        if w_start < 0: w_start = 0
        if w_start > n_src - 6: w_start = n_src - 6
        
        x_loc = src_x[w_start:w_start+6]
        y_loc = src_y[w_start:w_start+6]
        
        if use_robust:
            out[i] = _lagrange6_barycentric(tx, x_loc, y_loc)
        else:
            out[i] = _lagrange6_naive(tx, x_loc, y_loc)

@njit(fastmath=True, cache=True, parallel=True)
def _eval_sprague_batch_dispatch(
    tgt_x: np.ndarray, 
    src_x: np.ndarray, 
    src_y_batch: np.ndarray, 
    out_batch: np.ndarray,
    use_robust: bool
):
    """Batch Sprague evaluation parallelized over signals.

    Args:
        tgt_x (np.ndarray): Target x-coordinates.
        src_x (np.ndarray): Source x-coordinates.
        src_y_batch (np.ndarray): 2D source y-values.
        out_batch (np.ndarray): 2D output buffer.
        use_robust (bool): Whether to use the Barycentric form.
    """
    n_signals = src_y_batch.shape[0]
    n_src = len(src_x)
    n_tgt = len(tgt_x)
    
    for k in prange(n_signals):
        for i in range(n_tgt):
            tx = tgt_x[i]
            idx = np.searchsorted(src_x, tx, side='right')
            w_start = idx - 3
            if w_start < 0: w_start = 0
            if w_start > n_src - 6: w_start = n_src - 6
            
            x_loc = src_x[w_start:w_start+6]
            y_loc = src_y_batch[k, w_start:w_start+6]
            
            if use_robust:
                out_batch[k, i] = _lagrange6_barycentric(tx, x_loc, y_loc)
            else:
                out_batch[k, i] = _lagrange6_naive(tx, x_loc, y_loc)

# ==============================================================================
# 5. FLOATER-HORMANN (RATIONAL) KERNELS - NEW!
# ==============================================================================

@njit(fastmath=True, cache=True)
def _calc_fh_weights(x: np.ndarray, d: int) -> np.ndarray:
    """Computes weights for Floater-Hormann Rational Interpolation.

    Formula: w_k = (-1)^k * sum_{i=...} (1 / prod_{j!=i} |x_i - x_j|)

    Args:
        x (np.ndarray): Source x-coordinates.
        d (int): Approximation degree (0 <= d < n).

    Returns:
        np.ndarray: Barycentric rational weights.
    """
    n = len(x)
    w = np.zeros(n, dtype=np.float64)
    
    # Precompute inverse distance products
    # This is O(N*d) which is acceptable for setup
    for k in range(n):
        s_val = 0.0
        
        # Determine range of i: max(0, k-d) <= i <= min(k, n-1-d)
        i_min = 0 if (k - d) < 0 else (k - d)
        i_max = k
        if i_max > (n - 1 - d):
            i_max = n - 1 - d
            
        # Summation
        for i in range(i_min, i_max + 1):
            prod = 1.0
            for j in range(i, i + d + 1):
                if j != k:
                    prod *= (1.0 / abs(x[k] - x[j]))
            s_val += prod
            
        w[k] = ((-1.0)**k) * s_val
        
    return w

@njit(fastmath=True, cache=True, parallel=True)
def _eval_fh_1d(
    tgt_x: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    w: np.ndarray,
    out: np.ndarray
):
    """1D Barycentric Rational Interpolation (Floater-Hormann).

    Args:
        tgt_x (np.ndarray): Target x-coordinates.
        src_x (np.ndarray): Source x-coordinates.
        src_y (np.ndarray): Source y-values.
        w (np.ndarray): Pre-calculated FH weights.
        out (np.ndarray): Output buffer.
    """
    n_tgt = len(tgt_x)
    n_src = len(src_x)
    
    for i in prange(n_tgt):
        tx = tgt_x[i]
        
        # Check for exact node match to avoid division by zero
        exact_match = False
        exact_idx = -1
        
        # Quick check if near boundaries or search (optional, but robust)
        # Using a small epsilon or exact match
        for k in range(n_src):
            if tx == src_x[k]:
                out[i] = src_y[k]
                exact_match = True
                break
        
        if exact_match:
            continue
            
        numerator = 0.0
        denominator = 0.0
        
        # Sum over all nodes (O(N) per target point)
        for k in range(n_src):
            diff = tx - src_x[k]
            # Safety double-check (though exact_match loop should catch it)
            if diff == 0.0:
                numerator = src_y[k]
                denominator = 1.0
                break
                
            term = w[k] / diff
            numerator += term * src_y[k]
            denominator += term
            
        if denominator != 0.0:
            out[i] = numerator / denominator
        else:
            out[i] = 0.0

@njit(fastmath=True, cache=True, parallel=True)
def _eval_fh_batch(
    tgt_x: np.ndarray,
    src_x: np.ndarray,
    src_y_batch: np.ndarray,
    w: np.ndarray,
    out_batch: np.ndarray
):
    """Batch Floater-Hormann evaluation parallelized over signals.

    Args:
        tgt_x (np.ndarray): Target x-coordinates.
        src_x (np.ndarray): Source x-coordinates.
        src_y_batch (np.ndarray): 2D source y-values.
        w (np.ndarray): Pre-calculated FH weights.
        out_batch (np.ndarray): 2D output buffer.
    """
    n_signals = src_y_batch.shape[0]
    n_src = len(src_x)
    n_tgt = len(tgt_x)
    
    for k in prange(n_signals):
        for i in range(n_tgt):
            tx = tgt_x[i]
            
            # Exact node match check
            exact_match = False
            match_idx = -1
            
            # Note: For very large N, a binary search here is better than linear scan,
            # but for FH interpolation N is usually moderate (<100).
            for j in range(n_src):
                if tx == src_x[j]:
                    match_idx = j
                    exact_match = True
                    break
            
            if exact_match:
                out_batch[k, i] = src_y_batch[k, match_idx]
                continue
            
            numerator = 0.0
            denominator = 0.0
            
            for j in range(n_src):
                diff = tx - src_x[j]
                term = w[j] / diff
                numerator += term * src_y_batch[k, j]
                denominator += term
                
            if denominator != 0.0:
                out_batch[k, i] = numerator / denominator
            else:
                out_batch[k, i] = 0.0

# ==============================================================================
# 6. MAIN CLASS
# ==============================================================================

class UniSpline:
    """Unified High-Performance Interpolation Backend.

    Provides a consistent interface for multiple interpolation methods including
    Linear, PCHIP, MAKIMA, Sprague, and Floater-Hormann. Supports both single
    signal (1D) and batch processing (2D) with Numba-accelerated kernels.
    """
    
    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        method: Literal["linear", "pchip", "makima", "sprague", "floater_hormann", "fh"] = "pchip",
        robust: bool = False,
        d: int = 3
    ):
        """Initializes the UniSpline engine and pre-calculates necessary auxiliary data.

        Args:
            x (np.ndarray): Source x array (must be strictly increasing).
            y (np.ndarray): Source y array (1D or 2D [signals, points]).
            method (Literal): Interpolation method to use.
                - 'linear': Basic linear interpolation.
                - 'pchip': Piecewise Cubic Hermite (Monotone preserving).
                - 'makima': Modified Akima (Reduced oscillations).
                - 'sprague': Fifth-order Lagrange local interpolation.
                - 'floater_hormann' (or 'fh'): Barycentric Rational interpolation.
            robust (bool): If True, uses numerically stable algorithms (e.g., Barycentric Sprague).
            d (int): Degree for Floater-Hormann (default=3). 0 <= d < len(x).

        Raises:
            ValueError: If y is not 1D or 2D.
        """
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.method = method.lower()
        self.robust = robust
        self.d = d
        
        # Detect Batch
        if y.ndim == 1:
            self.is_batch = False
            self.y = np.ascontiguousarray(y, dtype=np.float64)
        elif y.ndim == 2:
            self.is_batch = True
            self.y = np.ascontiguousarray(y, dtype=np.float64)
        else:
            raise ValueError("y must be 1D or 2D")

        # Pre-compute Data (Slopes or Weights)
        self.aux_data = None 
        
        if self.method == "pchip":
            if self.is_batch: self.aux_data = _calc_pchip_derivs_batch(self.x, self.y)
            else: self.aux_data = _calc_pchip_derivs_1d(self.x, self.y)
            
        elif self.method == "makima":
            if self.is_batch: self.aux_data = _calc_makima_slopes_batch(self.x, self.y)
            else: self.aux_data = _calc_makima_slopes_1d(self.x, self.y)
            
        elif self.method in ("floater_hormann", "fh"):
            # FH Weights depend only on X and d, so shared across batch
            # Ensure d is valid
            n = len(self.x)
            if self.d < 0: self.d = 0
            if self.d >= n: self.d = n - 1
            self.aux_data = _calc_fh_weights(self.x, self.d)

    def __call__(self, target_x: np.ndarray) -> np.ndarray:
        """Evaluates the spline at the target x-coordinates.

        Args:
            target_x (np.ndarray): Coordinates at which to evaluate the spline.

        Returns:
            np.ndarray: Interpolated values (1D or 2D depending on input y).
        """
        t_x = np.ascontiguousarray(target_x, dtype=np.float64)
        
        # Allocate
        if self.is_batch:
            out = np.empty((self.y.shape[0], len(t_x)), dtype=np.float64)
        else:
            out = np.empty_like(t_x)

        # Dispatch
        if self.method == "linear":
            if self.is_batch: _eval_linear_batch(t_x, self.x, self.y, out)
            else: _eval_linear_1d(t_x, self.x, self.y, out)

        elif self.method in ("pchip", "makima"):
            # aux_data holds slopes
            if self.is_batch: _eval_cubic_hermite_batch(t_x, self.x, self.y, self.aux_data, out)
            else: _eval_cubic_hermite_1d(t_x, self.x, self.y, self.aux_data, out)
                
        elif self.method == "sprague":
            if self.is_batch:
                _eval_sprague_batch_dispatch(t_x, self.x, self.y, out, self.robust)
            else:
                _eval_sprague_1d_dispatch(t_x, self.x, self.y, out, self.robust)
        
        elif self.method in ("floater_hormann", "fh"):
            # aux_data holds weights
            if self.is_batch:
                _eval_fh_batch(t_x, self.x, self.y, self.aux_data, out)
            else:
                _eval_fh_1d(t_x, self.x, self.y, self.aux_data, out)
                
        return out

    def _fallback_linear(self, t_x: np.ndarray) -> np.ndarray:
        """Internal NumPy-based linear interpolation fallback.

        Args:
            t_x (np.ndarray): Target x-coordinates.

        Returns:
            np.ndarray: Linearly interpolated results.
        """
        if not self.is_batch:
            return np.interp(t_x, self.x, self.y)
        else:
            out = np.empty((self.y.shape[0], len(t_x)), dtype=np.float64)
            for k in range(self.y.shape[0]):
                out[k, :] = np.interp(t_x, self.x, self.y[k, :])
            return out

# --- Verification ---

if __name__ == "__main__":
    
    import time
    import timeit
    import gc
    import numpy as np
    from typing import List, Callable, Dict, Any
    from dataclasses import dataclass
    
    # Ensure UniSpline is available (assuming it's defined above in the same file)
    # from your_module import UniSpline 

    @dataclass
    class BenchResult:
        name: str
        min_time_ms: float
        mean_time_ms: float
        std_dev_ms: float
        factor: float = 1.0
    
    class PerformanceSuite:
        """
        A robust benchmarking suite for measuring Python execution time
        minimizing OS noise and GC interference.
        """
    
        def __init__(self, repeats: int = 5, loops: int = 1):
            """
            Args:
                repeats: How many times to repeat the full test (to find the best run).
                loops: How many times to execute the function *inside* the timer (to avg out overhead).
            """
            self.repeats = repeats
            self.loops = loops
    
        def benchmark(self, name: str, func: Callable[[], Any], baseline_ms: float = None) -> BenchResult:
            """
            Runs the benchmark with GC disabled during timing.
            """
            # Warmup
            func()
    
            times = []
            for _ in range(self.repeats):
                # Disable GC to prevent spikes during critical path execution
                gc_old = gc.isenabled()
                gc.disable()
                
                try:
                    t0 = time.perf_counter()
                    for _ in range(self.loops):
                        func()
                    t1 = time.perf_counter()
                finally:
                    if gc_old:
                        gc.enable()
                
                # Normalize time per loop
                times.append((t1 - t0) / self.loops)
    
            times_ms = np.array(times) * 1000.0
            min_ms = np.min(times_ms)
            mean_ms = np.mean(times_ms)
            std_ms = np.std(times_ms)
    
            factor = 1.0
            if baseline_ms:
                factor = min_ms / baseline_ms
    
            return BenchResult(name, min_ms, mean_ms, std_ms, factor)
    
        @staticmethod
        def print_table(title: str, results: List[BenchResult]):
            print(f"\n=== {title} ===")
            # Header
            print(f"{'Method':<15} | {'Min Time':<12} | {'Mean ± Std':<20} | {'Factor':<10}")
            print("-" * 65)
            
            for r in results:
                mean_str = f"{r.mean_time_ms:.3f} ± {r.std_dev_ms:.3f}"
                print(f"{r.name:<15} | {r.min_time_ms:<8.3f} ms  | {mean_str:<20} | {r.factor:.1f}x")

    print("=== UniSpline Advanced Performance Verification ===")

    # Configuration: Added 'linear' to the list
    methods = ["linear", "pchip", "makima", "sprague", "floater_hormann"]
    
    # Initialize Suite
    suite = PerformanceSuite(repeats=7, loops=1)

    # ---------------------------------------------------------
    # Test 1: Accuracy & Overshoot (Validation)
    # ---------------------------------------------------------
    print("\n[Validation] Monotonicity Check")
    src_x = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10], dtype=float)
    src_y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=float)
    dst_x = np.linspace(0, 10, 100)

    print(f"{'Method':<10} | {'Min':<10} | {'Max':<10} | {'Status'}")
    print("-" * 45)
    
    # Baseline
    lin = np.interp(dst_x, src_x, src_y)
    print(f"{'np.interp':<10} | {lin.min():<10.4f} | {lin.max():<10.4f} | OK")

    for m in methods:
        spline = UniSpline(src_x, src_y, method=m)
        res = spline(dst_x)
        
        # Simple heuristic for status
        overshoot = res.max() > 1.0001 or res.min() < -0.0001
        status = "OVERSHOOT" if overshoot else "OK"
        if m == "sprague": status = "OK (Expected)" # Sprague allows overshoot
        
        print(f"{m:<10} | {res.min():<10.4f} | {res.max():<10.4f} | {status}")


    # ---------------------------------------------------------
    # Test 2: Single Signal (Construction + Eval)
    # ---------------------------------------------------------
    N_SRC = 100
    N_DST = 10_000
    x_src = np.linspace(0, 100, N_SRC)
    y_src = np.random.rand(N_SRC)
    x_dst = np.linspace(0, 100, N_DST)

    results_single: List[BenchResult] = []

    # 2a. Baseline
    def run_numpy():
        np.interp(x_dst, x_src, y_src)
    
    # We increase loops for numpy because it is very fast
    suite.loops = 10
    res_np = suite.benchmark("np.interp", run_numpy)
    results_single.append(res_np)

    # 2b. Splines
    suite.loops = 5 
    for m in methods:
        def run_spline():
            # Measures BOTH init and call
            UniSpline(x_src, y_src, method=m)(x_dst)
        
        res = suite.benchmark(m, run_spline, baseline_ms=res_np.min_time_ms)
        results_single.append(res)

    suite.print_table(f"1D Performance (Init + Eval) [{N_SRC}->{N_DST} pts]", results_single)


    # ---------------------------------------------------------
    # Test 3: Batch Performance (Eval Only vs Total)
    # ---------------------------------------------------------
    # This test separates caching the spline vs recreating it
    N_SIG = 1000
    N_S = 100
    N_D = 1000
    
    x_b_src = np.linspace(0, 100, N_S)
    x_b_dst = np.linspace(0, 100, N_D)
    y_b_src = np.random.rand(N_SIG, N_S)

    results_batch: List[BenchResult] = []
    
    # 3a. Baseline (Looped)
    def run_numpy_batch():
        # List comprehension is the standard pythonic way for 2D interp
        [np.interp(x_b_dst, x_b_src, row) for row in y_b_src]

    suite.loops = 1
    res_np_batch = suite.benchmark("np.interp (loop)", run_numpy_batch)
    results_batch.append(res_np_batch)

    # 3b. Splines (Cached Eval Only)
    # Scenario: We build the spline ONCE, and evaluate it many times
    for m in methods:
        # Pre-compute (Setup phase - not timed)
        spline_obj = UniSpline(x_b_src, y_b_src, method=m)
        
        def run_eval_only():
            spline_obj(x_b_dst)
            
        res = suite.benchmark(f"{m} (eval)", run_eval_only, baseline_ms=res_np_batch.min_time_ms)
        results_batch.append(res)

    suite.print_table(f"2D Batch Throughput (Eval Only) [{N_SIG} signals]", results_batch)
    print("\n=== Verification Complete ===")