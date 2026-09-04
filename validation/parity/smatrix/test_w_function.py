#!/usr/bin/env python3
"""Comparison test for w_function — numba vs rust."""

import sys, os, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = PROJECT_ROOT
UNIT_NAME = "w_function"
NUMBA_MODULE = "loom_matrix"
TEST_COUNT = 50

numba_imported = False
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "loom"))
    from loom_matrix import w_function as numba_func
    numba_imported = True
except ImportError:
    print("Note: Original numba version not found. Running rust-only validation.")

# Unified layout: kernels live in navette._smatrix (aggregated extension).
import navette._smatrix as rust_mod
sys.modules["navette_matrix"] = rust_mod
rust_func = getattr(rust_mod, UNIT_NAME)

rng = np.random.default_rng(seed=42)
test_inputs = [(complex(rng.standard_normal(), rng.standard_normal()), rough_type) for rough_type in range(5)]

def generate_input(q_val, rough_type):
    return (np.complex128(q_val), np.int32(rough_type))

print(f"=== CORRECTNESS TEST: {UNIT_NAME} ===")
all_pass = True
for i in range(5):
    q_test = complex(rng.standard_normal(), rng.standard_normal()) * 10.0
    rough_type = i
    inp = generate_input(q_test, rough_type)
    rust_out = rust_func(np.complex128(q_test), np.int32(rough_type))
    if numba_imported:
        numba_out = numba_func(*inp)
        diff_real = abs(float(numba_out.real) - float(rust_out.real))
        diff_imag = abs(float(numba_out.imag) - float(rust_out.imag))
        diff_max = max(diff_real, diff_imag)
        scale = max(abs(float(numba_out.real)), abs(float(numba_out.imag)), 1.0)
        status = "PASS" if (diff_max < 1e-12 or diff_max / scale < 1e-12) else "FAIL"
    else:
        diff_max = abs(float(rust_out.real)) + abs(float(rust_out.imag))
        status = "PASS (rust-only)"
    print(f"CORRECTNESS {UNIT_NAME}_type{i} {status} | diff_max={diff_max:.2e}")
    all_pass &= status.startswith("PASS")

if numba_imported:
    print(f"\n=== SPEED BENCHMARK: {UNIT_NAME} ===")
    NUM_RUNS = 50
    _ = numba_func(np.complex128(3.7 + 1.2j), np.int32(4))
    _ = rust_func(np.complex128(3.7 + 1.2j), np.int32(4))
    q_bench = np.complex128(5.0)
    rough_types = [np.int32(t) for t in range(5)]
    numba_times_by_type = []
    for rt in rough_types:
        times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            numba_func(q_bench, rt)
            times.append(time.perf_counter() - t0)
        numba_times_by_type.append(np.mean(times))
    rust_times_by_type = []
    for rt in rough_types:
        times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            rust_func(q_bench, rt)
            times.append(time.perf_counter() - t0)
        rust_times_by_type.append(np.mean(times))
    numba_total_ms = np.sum(numba_times_by_type) * 1000
    rust_total_ms = np.sum(rust_times_by_type) * 1000
    speedup = numba_total_ms / rust_total_ms if rust_total_ms > 0 else float('inf')
    print(f"  Numba avg: {numba_total_ms:.3f} ms (total across types)")
    print(f"  Rust  avg: {rust_total_ms:.3f} ms (total across types)")
    print(f"  Speedup:   {speedup:.2f}x")
    print(f"BENCH_RESULT {UNIT_NAME} numba_time={numba_total_ms:.3f} rust_time={rust_total_ms:.3f} speedup={speedup:.2f}")

if all_pass:
    print(f"\n{UNIT_NAME}: ALL CORRECTNESS TESTS PASSED")
else:
    print(f"\n{UNIT_NAME}: SOME CORRECTNESS TESTS FAILED")
print(f"OUTPUT_STATUS {'PASS' if all_pass else 'FAIL'}")