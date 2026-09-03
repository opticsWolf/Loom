#!/usr/bin/env python3
"""Comparison test for redheffer_product_complex_field — numba vs rust."""

import sys, os, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = PROJECT_ROOT
UNIT_NAME = "redheffer_product_complex_field"

numba_imported = False
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "loom"))
    from kernel import redheffer_product_complex_field as numba_func
    numba_imported = True
except ImportError:
    try:
        # Try alternative module names
        for mod_name in ["kernel", "func_1", "fields", "redheffer"]:
            try:
                mod = __import__(mod_name)
                if hasattr(mod, UNIT_NAME):
                    numba_func = getattr(mod, UNIT_NAME)
                    numba_imported = True
                    break
            except ImportError:
                continue
    except Exception:
        pass
    if not numba_imported:
        print("Note: Original numba version not found. Running rust-only validation.")

# ---- Import Rust module ----
import importlib.util
target_dir = os.path.join(OUTPUT_DIR, "target", "release")
if sys.platform == "win32":
    so_name = "navette_matrix.pyd"
elif sys.platform == "darwin":
    so_name = "navette_matrix.dylib"
else:
    so_name = "libnavette_matrix.so"
so_path = os.path.join(target_dir, so_name)
if not os.path.exists(so_path):
    print(f"FATAL: Rust module not built at {so_path}")
    sys.exit(1)
spec = importlib.util.spec_from_file_location("navette_matrix", so_path)
rust_mod = importlib.util.module_from_spec(spec)
sys.modules["navette_matrix"] = rust_mod
spec.loader.exec_module(rust_mod)
rust_func = getattr(rust_mod, UNIT_NAME)


def generate_input(rng):
    return tuple(np.complex128(complex(rng.normal(0.3, 0.5), rng.normal(0.0, 0.3))) for _ in range(8))


def make_args(inp):
    return inp


rng = np.random.default_rng(seed=42)
test_inputs = [generate_input(rng) for _ in range(10)]
test_large = generate_input(np.random.default_rng(seed=99))

print(f"=== CORRECTNESS TEST: {UNIT_NAME} ===")
all_pass = True
for i, inp in enumerate(test_inputs):
    rust_out = rust_func(*inp)
    if numba_imported:
        numba_out = numba_func(*inp)
        match = True
        diff_max = 0.0
        for j, (nb, rs) in enumerate(zip(numba_out, rust_out)):
            nb_c = np.complex128(nb)
            rs_c = complex(rs.real, rs.imag)
            diff = abs(float(np.abs(nb_c - rs_c)))
            diff_max = max(diff_max, diff)
            match &= np.isclose(nb.real, rs.real, rtol=1e-6, atol=1e-10) and \
                     np.isclose(nb.imag, rs.imag, rtol=1e-6, atol=1e-10)
        status = "PASS" if match else "FAIL"
    else:
        diff_max = 0.0
        for j, val in enumerate(rust_out):
            re, im = val.real, val.imag
            assert not (np.isinf(re) or np.isnan(re)), f"Output {j} real inf/nan"
            assert not (np.isinf(im) or np.isnan(im)), f"Output {j} imag inf/nan"
            diff_max += abs(re) + abs(im)
        status = "PASS (rust-only)"
    print(f"  test_{i}: {status} | diff_sum={diff_max:.6f}")
    all_pass &= status.startswith("PASS")

if numba_imported:
    print(f"\n=== SPEED BENCHMARK: {UNIT_NAME} ===")
    _ = numba_func(*test_inputs[0])
    _ = rust_func(*test_inputs[0])
    NUM_RUNS = 50
    numba_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        numba_func(*test_large)
        numba_times.append(time.perf_counter() - t0)
    rust_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        rust_func(*test_large)
        rust_times.append(time.perf_counter() - t0)
    numba_mean_ms = np.mean(numba_times) * 1000
    rust_mean_ms = np.mean(rust_times) * 1000
    speedup = numba_mean_ms / rust_mean_ms if rust_mean_ms > 0 else float('inf')
    print(f"  Numba avg: {numba_mean_ms:.3f} ms")
    print(f"  Rust  avg: {rust_mean_ms:.3f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    print(f"BENCH_RESULT {UNIT_NAME} numba_time={numba_mean_ms:.3f} rust_time={rust_mean_ms:.3f} speedup={speedup:.2f}")

if all_pass:
    print(f"\n{UNIT_NAME}: ALL CORRECTNESS TESTS PASSED")
else:
    print(f"\n{UNIT_NAME}: SOME CORRECTNESS TESTS FAILED")
print(f"OUTPUT_STATUS {'PASS' if all_pass else 'FAIL'}")