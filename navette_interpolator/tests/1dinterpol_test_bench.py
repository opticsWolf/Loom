#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniSpline Comprehensive Test & Benchmark Suite
Combines functional tests, cross-implementation deviation checks, and performance 
benchmarks against SciPy/NumPy references.

Usage:
    python 1dinterpol_test_bench.py [--test] [--bench] [--scale]
"""

import sys
import time
import argparse
import pickle
import gc
import numpy as np
from typing import Dict, Tuple, Callable, Any, List

# --- Backend Discovery ---

# 1. NumPy & SciPy (References)
try:
    import scipy.interpolate as si
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  SciPy not found. SciPy references (PCHIP, Makima) will be skipped.")

# 2. Python Backend (Loom)
try:
    import loom_unispline as unispline_py
    HAS_PY = True
except ImportError:
    HAS_PY = False
    print("⚠️  loom_unispline not found. Skipping Python backend.")

# 3. Rust Backend (Navette)
try:
    import navette_interpolator as unispline_rs
    HAS_RS = True
except ImportError:
    HAS_RS = False
    print("⚠️  navette_interpolator not found. Skipping Rust backend.")


# =============================================================================
# Python Reference Implementations (For methods missing in SciPy)
# =============================================================================

def floater_hormann_python(x, y, d, xi):
    """Floater–Hormann rational interpolation (pure Python fallback)."""
    n = len(x)
    yi = np.empty_like(xi)
    for i, xv in enumerate(xi):
        if xv in x:
            yi[i] = y[np.where(x == xv)[0][0]]
            continue
        num, den = 0.0, 0.0
        for k in range(n):
            diff = xv - x[k]
            w = 0.0
            i_min = max(0, k - d)
            i_max = min(k, n - d - 1)
            for i_idx in range(i_min, i_max + 1):
                prod = 1.0
                for j in range(i_idx, i_idx + d + 1):
                    if j != k:
                        prod *= 1.0 / abs(x[k] - x[j])
                w += prod
            # The sign is independent of the inner product summation
            w *= (-1.0)**k
            term = w / diff
            num += term * y[k]
            den += term
        yi[i] = num / den if den != 0 else 0.0
    return yi

def sprague_python(x, y, xi, robust=False):
    """Sprague local 6-point interpolation (pure Python fallback)."""
    n = len(x)
    yi = np.empty_like(xi)
    for i, xv in enumerate(xi):
        idx = np.searchsorted(x, xv)
        start = max(0, min(idx - 3, n - 6))
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
# Reference Dispatcher
# =============================================================================

def get_reference_evaluator(method: str) -> Callable:
    """Returns a function: f(x, y, tx) -> ty matching the requested method."""
    method = method.lower()
    
    def eval_linear(x, y, tx):
        return np.interp(tx, x, y)
        
    def eval_pchip(x, y, tx):
        if HAS_SCIPY:
            return si.PchipInterpolator(x, y, extrapolate=True)(tx)
        return np.interp(tx, x, y) # Ultimate fallback
        
    def eval_makima(x, y, tx):
        if HAS_SCIPY:
            try:
                # SciPy >= 1.13.0 (and some 1.7+) explicit makima
                return si.Akima1DInterpolator(x, y, method="makima", extrapolate=True)(tx)
            except TypeError:
                return si.Akima1DInterpolator(x, y, extrapolate=True)(tx)
        return np.interp(tx, x, y)
        
    def eval_sprague(x, y, tx):
        return sprague_python(x, y, tx, robust=False)
        
    def eval_fh(x, y, tx):
        if HAS_SCIPY and hasattr(si, 'FloaterHormannInterpolator'):
            return si.FloaterHormannInterpolator(x, y, d=3)(tx)
        return floater_hormann_python(x, y, 3, tx)

    dispatch = {
        "linear": eval_linear,
        "pchip": eval_pchip,
        "makima": eval_makima,
        "sprague": eval_sprague,
        "floater_hormann": eval_fh,
        "fh": eval_fh
    }
    return dispatch.get(method, eval_linear)


# =============================================================================
# 1. Functional Tests (Extended Features & Integrity)
# =============================================================================

def run_functional_tests():
    print("\n" + "="*80)
    print("🧪 RUNNING FUNCTIONAL & EXTENDED TESTS")
    print("="*80)

    if not HAS_RS:
        print("⚠️ Skipping Rust-specific extended tests (navette_interpolator missing).")
        return

    # 1. Unsorted Queries
    try:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])
        spline = unispline_rs.UniSpline(x, y, method="pchip")
        t_unsorted = np.array([2.5, 0.5, 1.5, 3.5])
        result = spline.eval(t_unsorted)
        expected = np.array([6.21875, 0.3125, 2.21875, 12.0])   
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        print("✅ [Rust] Unsorted queries handle smoothly.")
    except Exception as e:
        print(f"❌ [Rust] Unsorted queries failed: {e}")

    # 2. Exact SciPy Derivative Matching
    try:
        x = np.linspace(0, 10, 15)
        y = np.sin(x) * x
        t = np.linspace(0, 10, 100)
        
        spline = unispline_rs.UniSpline(x, y, method="pchip")
        deriv_rs = spline.derivative(t)
        
        if HAS_SCIPY:
            sp_pchip = si.PchipInterpolator(x, y, extrapolate=True)
            deriv_sp = sp_pchip.derivative()(t)
            np.testing.assert_allclose(deriv_rs, deriv_sp, atol=1e-12, rtol=1e-12)
            print("✅ [Rust] 1st Derivatives match SciPy PCHIP analytically exactly.")
        else:
            print("⚠️ [Rust] Derivative test passed basic execution (SciPy needed for tight validation).")
    except Exception as e:
        print(f"❌ [Rust] Derivatives failed: {e}")

    # 3. Extrapolation Modes
    try:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        t_out = np.array([-1.0, 3.0])
        
        spline_lin = unispline_rs.UniSpline(x, y, method="linear", extrap="linear")
        np.testing.assert_allclose(spline_lin.eval(t_out), [-1.0, 7.0])
        
        spline_clamp = unispline_rs.UniSpline(x, y, method="linear", extrap="clamp")
        np.testing.assert_allclose(spline_clamp.eval(t_out), [0.0, 4.0])
        
        spline_err = unispline_rs.UniSpline(x, y, extrap="error")
        res_err = spline_err.eval(t_out)
        if not np.isnan(res_err).all():
            print("❌ [Rust] Extrapolation error mode did not yield NaN.")
        else:
            print("✅ [Rust] Extrapolation modes (linear, clamp, error) conform strictly.")
    except Exception as e:
        print(f"❌ [Rust] Extrapolation modes failed: {e}")

    # 4. Pickling (Dimensionality check)
    try:
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        spline = unispline_rs.UniSpline(x, y, method="pchip")
        unispline_rs.UniSpline.__module__ = "navette_interpolator"
        
        data = pickle.dumps(spline)
        spline2 = pickle.loads(data)
        t = np.linspace(0, 10, 50)
        
        res1 = spline.eval(t)
        res2 = spline2.eval(t)
        np.testing.assert_allclose(res1, res2)
        # Verify 1D state is maintained
        assert res2.ndim == 1, f"Pickled array changed dimensions to {res2.ndim}D"
        print("✅ [Rust] Serialization (Pickling) preserves state & dimensionality perfectly.")
    except Exception as e:
        print(f"❌ [Rust] Pickling failed: {e}")

    # 5. Knot Access
    try:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([[0.0, 1.0, 4.0], [1.0, 2.0, 5.0]]) 
        spline = unispline_rs.UniSpline(x, y, method="pchip")
        np.testing.assert_equal(spline.get_x(), x)
        np.testing.assert_equal(spline.get_y(None), y)
        np.testing.assert_equal(spline.get_y(0), y[0])
        assert spline.get_slopes(None) is not None
        print("✅ [Rust] Knot and Slopes internal access verified.")
    except Exception as e:
        print(f"❌ [Rust] Knot access failed: {e}")

    # 6. Robust vs Naive Sprague Alignment
    try:
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        t = np.linspace(0, 10, 100)
        
        s_naive = unispline_rs.UniSpline(x, y, method="sprague", robust=False)
        s_robust = unispline_rs.UniSpline(x, y, method="sprague", robust=True)
        
        np.testing.assert_allclose(s_naive.eval(t), s_robust.eval(t), rtol=1e-10)
        print("✅ [Rust] Sprague robust (Barycentric) & naive (Lagrange) align mathematically.")
    except Exception as e:
        print(f"❌ [Rust] Sprague robust alignment failed: {e}")


# =============================================================================
# 2. Benchmarks & Deviation Checking
# =============================================================================

def generate_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate 1D and 2D batch datasets for testing."""
    datasets = {}
    
    # 1. Standard Smooth Curve
    sx = np.linspace(380, 780, 21)
    datasets["Smooth Peak"] = (sx, 100 * np.exp(-0.5 * ((sx - 550) / 40)**2), np.linspace(380, 780, 401))
    
    # 2. Discontinuous/Sharp Steps
    sx = np.linspace(0, 10, 11)
    datasets["Hard Step"] = (sx, np.array([0,0,0,1,1,1,1,0,0,0,0], dtype=float), np.linspace(0, 10, 500))
    
    # 3. High Polynomial Tension
    sx = np.linspace(-1, 1, 15)
    datasets["Runge Function"] = (sx, 1.0 / (1.0 + 25.0 * sx**2), np.linspace(-1, 1, 500))
    
    # 4. Chaotic Noise
    rng = np.random.default_rng(42)
    sx = np.linspace(0, 100, 20)
    datasets["Spiky Random"] = (sx, rng.random(20) * 100, np.linspace(0, 100, 1000))

    # 5. Non-Uniform Grids (New!)
    sx = np.geomspace(1, 1000, 20)
    datasets["Non-Uniform (Log)"] = (sx, np.log10(sx) + np.sin(sx/10), np.linspace(1, 1000, 500))

    # 6. High-Load Batch Dataset (2D)
    N_SIG = 100
    N_S = 100
    N_D = 1000
    sx = np.linspace(0, 100, N_S)
    tx = np.linspace(0, 100, N_D)
    sy_batch = rng.standard_normal((N_SIG, N_S))
    datasets["Batch (100 sig)"] = (sx, sy_batch, tx)

    return datasets

def time_execution(func: Callable, loops: int = 10) -> float:
    """Benchmark execution time (GC disabled) in milliseconds."""
    try: func() # Warmup
    except Exception: return -1.0
    
    gc_old = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter()
        for _ in range(loops):
            func()
        t1 = time.perf_counter()
    finally:
        if gc_old: gc.enable()
        
    return ((t1 - t0) * 1000.0) / loops

def run_benchmarks_and_deviations():
    print("\n" + "="*95)
    print("📊 BENCHMARKS & DEVIATIONS (vs NumPy/SciPy References)")
    print("="*95)

    datasets = generate_datasets()
    methods = ["linear", "pchip", "makima", "sprague", "floater_hormann"]
    
    backends = []
    if HAS_PY: backends.append(("Py/Loom", unispline_py.UniSpline))
    if HAS_RS: backends.append(("Rs/Navette", unispline_rs.UniSpline))

    if not backends:
        print("⚠️ No valid backends found.")
        return

    header = f"{'Dataset':<17} | {'Method':<15} | {'Backend':<12} | {'Time (ms)':<9} | {'Speedup':<7} | {'Max Err':<9} | {'Status'}"
    print(header)
    print("-" * 95)

    for ds_name, (sx, sy, tx) in datasets.items():
        is_batch = sy.ndim == 2

        for method in methods:
            if method == "sprague" and "Spiky" in ds_name:
                continue # Skip sprague on spiky to avoid Runge explosions

            ref_func = get_reference_evaluator(method)

            # Calculate Reference Time & Output
            if is_batch:
                def run_ref(): return np.array([ref_func(sx, row, tx) for row in sy])
            else:
                def run_ref(): return ref_func(sx, sy, tx)
            
            ref_ms = time_execution(run_ref, loops=5 if is_batch else 20)
            try: ref_out = run_ref()
            except Exception: ref_out = None

            for be_name, be_class in backends:
                try:
                    spline = be_class(sx, sy, method=method)
                    
                    def run_target():
                        if hasattr(spline, 'eval'): return spline.eval(tx)
                        return spline(tx)
                    
                    tgt_ms = time_execution(run_target, loops=100 if not is_batch else 10)
                    
                    if ref_out is not None:
                        tgt_out = run_target()
                        max_err = np.max(np.abs(tgt_out - ref_out))
                        err_str = f"{max_err:.2e}"
                        
                        if max_err < 1e-6: status = "✅ OK"
                        elif max_err < 1e-2: status = "⚠️ Dev"
                        else: status = "❌ Fail"
                    else:
                        err_str = "N/A"
                        status = "⚠️ No Ref"
                    
                    speedup = f"{ref_ms / tgt_ms:.1f}x" if tgt_ms > 0 else "N/A"
                    print(f"{ds_name:<17} | {method:<15} | {be_name:<12} | {tgt_ms:>6.3f} ms | {speedup:>7} | {err_str:>9} | {status}")
                except Exception as e:
                    print(f"{ds_name:<17} | {method:<15} | {be_name:<12} | {'-':>9} | {'-':>7} | {'-':>9} | ❌ Error ({type(e).__name__})")
        print("-" * 95)


# =============================================================================
# 3. Throughput / Scaling Benchmarks
# =============================================================================

def run_scaling_benchmarks():
    print("\n" + "="*80)
    print("📈 SCALING BENCHMARKS (Throughput in Millions of Evals / Sec)")
    print("="*80)
    
    backends = []
    if HAS_PY: backends.append(("Py/Loom", unispline_py.UniSpline))
    if HAS_RS: backends.append(("Rs/Navette", unispline_rs.UniSpline))

    if not backends:
        print("⚠️ No valid backends found.")
        return

    n_source = 100
    x_src = np.linspace(0, 100, n_source)
    y_src = np.sin(x_src)
    
    target_sizes = [10_000, 100_000, 1_000_000]
    methods = ["pchip", "sprague", "floater_hormann"]

    print(f"{'Method':<16} | {'Target Pts':<12} | {'Backend':<12} | {'Time (ms)':<9} | {'Throughput (MEval/s)'}")
    print("-" * 80)

    for method in methods:
        for size in target_sizes:
            x_tgt = np.linspace(0, 100, size)
            
            for be_name, be_class in backends:
                try:
                    spline = be_class(x_src, y_src, method=method)
                    
                    def run_eval():
                        if hasattr(spline, 'eval'): spline.eval(x_tgt)
                        else: spline(x_tgt)
                        
                    loops = 50 if size == 10_000 else (10 if size == 100_000 else 3)
                    exec_ms = time_execution(run_eval, loops=loops)
                    
                    # Calculate MEval/s
                    # (Points per second) = (size / (exec_ms / 1000))
                    # MEval/s = Points per second / 1_000_000
                    meval_s = (size / (exec_ms / 1000.0)) / 1_000_000.0
                    
                    print(f"{method:<16} | {size:<12,d} | {be_name:<12} | {exec_ms:>6.2f} ms | {meval_s:>6.2f} M/s")
                except Exception as e:
                    print(f"{method:<16} | {size:<12,d} | {be_name:<12} | {'Error':>9} | {'-':>12}")
        print("-" * 80)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniSpline Combined Test & Bench Suite")
    parser.add_argument("--test", action="store_true", help="Run extended functional tests only")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks and deviations only")
    parser.add_argument("--scale", action="store_true", help="Run scalability and throughput tests only")
    
    args = parser.parse_args()

    run_all = not args.test and not args.bench and not args.scale

    if run_all or args.test:
        run_functional_tests()

    if run_all or args.bench:
        run_benchmarks_and_deviations()
        
    if run_all or args.scale:
        run_scaling_benchmarks()

    print("\n🎉 Suite execution complete!")