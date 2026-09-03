#!/usr/bin/env python3
"""
Test and benchmark suite for navette_interpolator (Rust extension)
against scipy, numba, and pure‑Python references.

Usage:
    python test_bench_unispline.py [--bench] [--plot]
"""

import sys
import time
import argparse
from functools import wraps

import numpy as np
import scipy.interpolate as si
from scipy.special import binom

# Import the Rust module (must be built and in PYTHONPATH)
try:
    import navette_interpolator
except ImportError:
    print("Error: Could not import 'navette_interpolator'. Make sure the Rust extension is built.")
    sys.exit(1)

# Optional numba
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not installed. Numba references will be skipped.")


# =============================================================================
# Reference implementations for methods not in SciPy
# =============================================================================

def floater_hormann_python(x, y, d, xi):
    """
    Floater–Hormann rational interpolation (pure Python).
    """
    n = len(x)
    yi = np.empty_like(xi)
    for i, xv in enumerate(xi):
        # exact match?
        if xv in x:
            yi[i] = y[np.where(x == xv)[0][0]]
            continue
        num = 0.0
        den = 0.0
        for k in range(n):
            diff = xv - x[k]
            if diff == 0:
                yi[i] = y[k]
                break
            # compute weight w_k
            w = 0.0
            i_min = max(0, k - d)
            i_max = min(k, n - d - 1)
            for i_idx in range(i_min, i_max + 1):
                prod = 1.0
                for j in range(i_idx, i_idx + d + 1):
                    if j != k:
                        prod *= 1.0 / abs(x[k] - x[j])
                w += (-1)**(i_idx) * prod   # sign convention
            term = w / diff
            num += term * y[k]
            den += term
        yi[i] = num / den if den != 0 else 0.0
    return yi

def sprague_python(x, y, xi, robust=False):
    """
    Sprague interpolation: local 6‑point rational or polynomial.
    """
    n = len(x)
    yi = np.empty_like(xi)
    for i, xv in enumerate(xi):
        idx = np.searchsorted(x, xv)
        start = max(0, min(idx - 3, n - 6))
        if robust:
            # rational version (same as in Rust)
            w = np.ones(6)
            for j in range(6):
                xj = x[start + j]
                for k in range(6):
                    if k != j:
                        w[j] /= (xj - x[start + k])
            num = 0.0
            den = 0.0
            exact = False
            for j in range(6):
                diff = xv - x[start + j]
                if diff == 0:
                    yi[i] = y[start + j]
                    exact = True
                    break
                term = w[j] / diff
                num += term * y[start + j]
                den += term
            if not exact and den != 0:
                yi[i] = num / den
        else:
            # polynomial interpolation over 6 points
            res = 0.0
            for j in range(6):
                basis = 1.0
                xj = x[start + j]
                for k in range(6):
                    if k != j:
                        basis *= (xv - x[start + k]) / (xj - x[start + k])
                res += y[start + j] * basis
            yi[i] = res
    return yi


# =============================================================================
# Decorator for timing
# =============================================================================
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


# =============================================================================
# Correctness tests
# =============================================================================

def test_correctness():
    """Compare Rust outputs with reference implementations."""
    np.random.seed(42)

    # Test parameters
    n_src = 20
    n_tgt = 100
    methods = ["linear", "pchip", "makima", "floater_hormann", "sprague"]
    # For 1D and 2D y
    shapes = [("1D", (n_tgt,)), ("2D", (3, n_tgt))]   # 3 signals

    # Generate strictly increasing x
    x = np.cumsum(np.random.rand(n_src)) + 0.5
    x[0] = 0.0
    x = np.sort(x)

    target_x = np.linspace(x[0] - 0.2, x[-1] + 0.2, n_tgt)

    results = {}

    for method in methods:
        # Prepare reference evaluator
        if method == "linear":
            ref_interp = si.interp1d(x, None, kind='linear', fill_value='extrapolate')
        elif method == "pchip":
            ref_interp = si.PchipInterpolator(x, None, extrapolate=True)
        elif method == "makima":
            ref_interp = si.Akima1DInterpolator(x, None, extrapolate=True)
        elif method == "floater_hormann":
            # No scipy, we use our Python version
            ref_interp = None
        elif method == "sprague":
            ref_interp = None   # use Python reference
        else:
            continue

        for shape_desc, y_shape in shapes:
            if shape_desc == "1D":
                y = np.random.randn(n_src)
                # Create rust spline
                rust_spline = navette_interpolator.UniSpline(x, y, method=method, robust=False)
                rust_out = rust_spline(target_x)

                # Reference
                if ref_interp is not None:
                    ref_interp.y = y  # update values
                    ref_out = ref_interp(target_x)
                else:
                    if method == "floater_hormann":
                        d = 3
                        ref_out = floater_hormann_python(x, y, d, target_x)
                    else:  # sprague
                        ref_out = sprague_python(x, y, target_x, robust=False)

                # Compare
                tol = 1e-10 if method == "linear" else 1e-8
                max_diff = np.max(np.abs(rust_out - ref_out))
                rel_diff = np.max(np.abs((rust_out - ref_out) / (ref_out + 1e-12)))
                assert max_diff < tol or rel_diff < 1e-7, \
                    f"Method {method}, shape {shape_desc}: max diff = {max_diff}, rel diff = {rel_diff}"
                results[(method, shape_desc)] = max_diff

            else:  # 2D, multiple signals
                y = np.random.randn(3, n_src)
                rust_spline = navette_interpolator.UniSpline(x, y, method=method, robust=False)
                rust_out = rust_spline(target_x)   # returns (3, n_tgt)

                # Compare each row against separate 1D references
                for row in range(3):
                    y_row = y[row, :]
                    if ref_interp is not None:
                        ref_interp.y = y_row
                        ref_out = ref_interp(target_x)
                    else:
                        if method == "floater_hormann":
                            ref_out = floater_hormann_python(x, y_row, 3, target_x)
                        else:  # sprague
                            ref_out = sprague_python(x, y_row, target_x, robust=False)
                    max_diff = np.max(np.abs(rust_out[row] - ref_out))
                    assert max_diff < 1e-8, f"Method {method}, row {row}: max diff = {max_diff}"

    print("✅ All correctness tests passed.")


# =============================================================================
# Benchmarks
# =============================================================================

def run_benchmarks():
    """Benchmark Rust, SciPy, Numba, and Python references."""
    np.random.seed(123)

    configs = [
        # (n_signals, n_src, n_tgt, description)
        (1, 50, 1000, "1 signal, 50 knots, 1000 targets"),
        (1, 200, 50000, "1 signal, 200 knots, 50000 targets"),
        (10, 100, 2000, "10 signals, 100 knots, 2000 targets"),
        (100, 500, 1000, "100 signals, 500 knots, 1000 targets"),
    ]

    methods = ["linear", "pchip", "makima", "floater_hormann", "sprague"]

    # Prepare references
    references = {}
    if HAS_NUMBA:
        @njit
        def linear_numba(x, y, xi):
            yi = np.empty_like(xi)
            for i, xv in enumerate(xi):
                idx = np.searchsorted(x, xv)
                if idx == 0:
                    slope = (y[1] - y[0]) / (x[1] - x[0])
                    yi[i] = y[0] + slope * (xv - x[0])
                elif idx == len(x):
                    slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
                    yi[i] = y[-1] + slope * (xv - x[-1])
                else:
                    if xv == x[idx-1]:
                        yi[i] = y[idx-1]
                    else:
                        t = (xv - x[idx-1]) / (x[idx] - x[idx-1])
                        yi[i] = y[idx-1] * (1 - t) + y[idx] * t
            return yi
        references["linear"] = linear_numba
        # For Hermite we can't easily do PCHIP in numba; skip for now.
    else:
        references["linear"] = lambda x, y, xi: si.interp1d(x, y, kind='linear', fill_value='extrapolate')(xi)

    # SciPy objects
    def scipy_linear(x, y, xi):
        return si.interp1d(x, y, kind='linear', fill_value='extrapolate')(xi)
    def scipy_pchip(x, y, xi):
        return si.PchipInterpolator(x, y, extrapolate=True)(xi)
    def scipy_makima(x, y, xi):
        return si.Akima1DInterpolator(x, y, extrapolate=True)(xi)

    references["scipy_linear"] = scipy_linear
    references["scipy_pchip"] = scipy_pchip
    references["scipy_makima"] = scipy_makima

    # Python implementations for missing methods
    references["py_fh"] = lambda x, y, xi: floater_hormann_python(x, y, 3, xi)
    references["py_sprague"] = lambda x, y, xi: sprague_python(x, y, xi, robust=False)

    # Store timing results
    table = []

    for n_sig, n_src, n_tgt, desc in configs:
        # Generate data
        x = np.cumsum(np.random.rand(n_src)) + 0.5
        x = np.sort(x)
        if n_sig == 1:
            y = np.random.randn(n_src)
        else:
            y = np.random.randn(n_sig, n_src)
        target_x = np.linspace(x[0] - 0.1, x[-1] + 0.1, n_tgt)

        for method in methods:
            # --- Rust ---
            rust_spline = navette_interpolator.UniSpline(x, y, method=method, robust=False)
            _, t_rust = timed(rust_spline)(target_x)

            # --- Reference (SciPy or Python) ---
            if method == "linear":
                ref_func = scipy_linear
            elif method == "pchip":
                ref_func = scipy_pchip
            elif method == "makima":
                ref_func = scipy_makima
            elif method == "floater_hormann":
                ref_func = references["py_fh"]
            else:  # sprague
                ref_func = references["py_sprague"]

            # For multi‑signal, reference must be called per signal (they are 1D).
            if n_sig == 1:
                _, t_ref = timed(ref_func)(x, y, target_x)
            else:
                start = time.perf_counter()
                out_ref = np.empty((n_sig, n_tgt))
                for i in range(n_sig):
                    out_ref[i] = ref_func(x, y[i], target_x)
                t_ref = time.perf_counter() - start

            # --- Numba (only linear) ---
            if method == "linear" and HAS_NUMBA and n_sig == 1:
                _, t_numba = timed(linear_numba)(x, y, target_x)
            else:
                t_numba = np.nan

            table.append({
                "config": desc,
                "method": method,
                "n_signals": n_sig,
                "n_src": n_src,
                "n_tgt": n_tgt,
                "Rust (s)": t_rust,
                "Ref (s)": t_ref,
                "Numba (s)": t_numba if not np.isnan(t_numba) else "-",
                "Speedup vs Ref": t_ref / t_rust if t_rust > 0 else np.inf,
            })

    # Print table
    print("\n" + "=" * 100)
    print(f"{'Config':<40} {'Method':<15} {'Rust (s)':>10} {'Ref (s)':>10} {'Numba (s)':>10} {'Speedup':>10}")
    print("=" * 100)
    for row in table:
        print(f"{row['config']:<40} {row['method']:<15} {row['Rust (s)']:10.4f} {row['Ref (s)']:10.4f} "
              f"{row['Numba (s)'] if isinstance(row['Numba (s)'], str) else f'{row['Numba (s)']:10.4f}':>10} "
              f"{row['Speedup vs Ref']:10.2f}")
    print("=" * 100)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--plot", action="store_true", help="Plot results (requires matplotlib)")
    args = parser.parse_args()

    # Always run correctness
    test_correctness()

    if args.bench:
        run_benchmarks()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            # Example plot: speedup vs method (you can extend)
            print("Plotting not implemented in this example, but you can add custom plots.")
        except ImportError:
            print("matplotlib not available, skipping plot.")