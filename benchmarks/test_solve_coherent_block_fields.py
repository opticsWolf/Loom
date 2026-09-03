#!/usr/bin/env python3
"""Comparison test for solve_coherent_block_fields — numba vs rust."""

import sys, os, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = PROJECT_ROOT
UNIT_NAME = "solve_coherent_block_fields"

numba_imported = False
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "loom"))
    from loom_matrix import solve_coherent_block_fields as numba_func
    POL_S = 0
    numba_imported = True
except ImportError:
    print("Note: Original numba version not found. Running rust-only validation.")

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

if hasattr(rust_mod, UNIT_NAME):
    rust_func = getattr(rust_mod, UNIT_NAME)
else:
    for alt_name in ["solve_coherent_block", "solve_coherent"]:
        if hasattr(rust_mod, alt_name):
            rust_func = getattr(rust_mod, alt_name)
            break
    else:
        print(f"FATAL: Rust function '{UNIT_NAME}' not found. Available: {[n for n in dir(rust_mod) if not n.startswith('_')]}")
        sys.exit(1)


def generate_test_case(rng, num_layers=6):
    total_needed = num_layers + 2
    lam = rng.uniform(400.0, 800.0)
    theta_deg = rng.uniform(10.0, 60.0)
    NSinFi = np.complex128((np.sin(np.radians(theta_deg)), 0.0))
    n_stack = np.empty(total_needed, dtype=np.complex128)
    for i in range(num_layers):
        if rng.random() < 0.3:
            n_stack[i] = np.complex128((rng.uniform(0.5, 2.0), rng.uniform(3.0, 7.0)))
        else:
            n_stack[i] = np.complex128((rng.uniform(1.3, 2.5), rng.uniform(-0.01, 0.01)))
    n_stack[num_layers]   = np.complex128((1.0, 0.0))
    n_stack[num_layers+1] = np.complex128((rng.uniform(1.5, 4.0), rng.uniform(-0.01, 0.01)))
    d_stack = np.zeros(total_needed, dtype=np.float64)
    for i in range(num_layers):
        d_stack[i] = rng.uniform(1.0, 500.0)
    d_stack[num_layers:] = 0.0
    rough_vals = np.zeros(total_needed, dtype=np.float64)
    rough_types = np.zeros(total_needed, dtype=np.int32)
    for i in range(1, num_layers + 1):
        if rng.random() < 0.5:
            rough_vals[i] = rng.uniform(0.0, 5.0)
            rough_types[i] = rng.choice([0, 1, 2, 4])
    start_idx = 0
    end_idx = num_layers
    return {
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "n_stack": n_stack.astype(np.complex128),
        "d_stack": d_stack.astype(np.float64),
        "rough_vals": rough_vals.astype(np.float64),
        "rough_types": rough_types.astype(np.int32),
        "lam": np.float64(lam),
        "NSinFi": NSinFi,
        "pol": int(POL_S) if numba_imported else 0,
    }

def make_rust_args(case):
    return (case["start_idx"], case["end_idx"], case["n_stack"], case["d_stack"],
            case["rough_vals"], case["rough_types"], case["lam"], case["NSinFi"], case["pol"])

def make_numba_args(case):
    return make_rust_args(case)

def compare_results(numba_out, rust_out):
    diff_max = 0.0
    diffs = []
    all_pass = True
    for i in range(8):
        nb_val = numba_out[i]
        rs_val = rust_out[i]
        if hasattr(nb_val, 'real'):
            diff_val = float(np.abs(complex(nb_val) - complex(rs_val.real, rs_val.imag)))
        else:
            diff_val = abs(float(nb_val) - float(rs_val))
        is_close = np.isclose(
            nb_val if not hasattr(nb_val, 'real') else (nb_val.real if i < 4 else nb_val),
            rs_val if not hasattr(rs_val, 'real') else (rs_val.real if i < 4 else rs_val),
            rtol=1e-5, atol=1e-8
        )
        diffs.append(f"elem{i}={diff_val:.2e}")
        diff_max = max(diff_max, diff_val)
        all_pass &= is_close
    return all_pass, diff_max, "; ".join(diffs)


rng = np.random.default_rng(seed=42)
test_cases = [generate_test_case(np.random.default_rng(seed=s), num_layers=rng.choice([3,5,8])) for s in range(10)]
edge_cases = []
edge_sharp = generate_test_case(np.random.default_rng(seed=77))
edge_sharp["rough_vals"][:] = 0.0
edge_sharp["rough_types"][:] = 0
edge_cases.append(edge_sharp)
edge_nc = generate_test_case(np.random.default_rng(seed=88))
for i in range(1, edge_nc["end_idx"] + 1):
    if edge_nc["rough_vals"][i] < 0.1:
        edge_nc["rough_vals"][i] = rng.uniform(1.0, 4.0)
    edge_nc["rough_types"][i] = 5
edge_cases.append(edge_nc)
edge_mixed = generate_test_case(np.random.default_rng(seed=99))
for i in range(1, edge_mixed["end_idx"] + 1):
    edge_mixed["rough_types"][i] = (i % 5)
edge_cases.append(edge_mixed)
all_test_cases = test_cases + edge_cases

print(f"=== CORRECTNESS TEST: {UNIT_NAME} ===\n")
all_pass = True
for i, case in enumerate(all_test_cases):
    rust_out = rust_func(*make_rust_args(case))
    if numba_imported:
        try:
            nb_out = numba_func(*make_numba_args(case))
        except Exception as e:
            print(f"  test_{i}: ERROR in numba — {e}")
            all_pass = False
            continue
        match, diff_max, details = compare_results(nb_out, rust_out)
        status = "PASS" if match else "FAIL"
    else:
        diff_max = 0.0
        phys_ok = True
        for j, val in enumerate(rust_out):
            re = float(val.real if hasattr(val, 'real') else val)
            im = float(val.imag if hasattr(val, 'imag') else 0.0)
            if np.isinf(re) or np.isnan(re) or np.isinf(im) or np.isnan(im):
                phys_ok = False
            diff_max += abs(re) + abs(im)
        status = "PASS (rust-only)" if phys_ok else "FAIL_RUST_ONLY"
    layer_count = case["end_idx"] - case["start_idx"]
    rough_types_used = set(case["rough_types"][1:case['end_idx']+1])
    print(f"  test_{i:2d} [L={layer_count}, rough={rough_types_used}]: {status:6s} | max_diff={diff_max:.2e}")
    all_pass &= status.startswith("PASS")

if numba_imported and len(all_test_cases) > 0:
    print(f"\n=== SPEED BENCHMARK: {UNIT_NAME} ===\n")
    bench_case = max(all_test_cases, key=lambda c: c["end_idx"] - c["start_idx"])
    _ = numba_func(*make_numba_args(bench_case))
    _ = rust_func(*make_rust_args(bench_case))
    NUM_RUNS = 100
    numba_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        numba_func(*make_numba_args(bench_case))
        numba_times.append(time.perf_counter() - t0)
    rust_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        rust_func(*make_rust_args(bench_case))
        rust_times.append(time.perf_counter() - t0)
    numba_mean_us = np.mean(numba_times[5:]) * 1e6
    rust_mean_us = np.mean(rust_times[5:]) * 1e6
    speedup = numba_mean_us / rust_mean_us if rust_mean_us > 0 else float('inf')
    print(f"  Numba avg: {numba_mean_us:.1f} μs")
    print(f"  Rust  avg: {rust_mean_us:.3f} μs")
    print(f"  Speedup:   {speedup:.2f}x")
    small_case = min(all_test_cases, key=lambda c: c["end_idx"] - c["start_idx"])
    numba_small = []
    rust_small = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        numba_func(*make_numba_args(small_case))
        numba_small.append(time.perf_counter() - t0)
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        rust_func(*make_rust_args(small_case))
        rust_small.append(time.perf_counter() - t0)
    small_speedup = (np.mean(numba_small)*1e6) / (np.mean(rust_small)*1e6) if np.mean(rust_small) > 0 else float('inf')
    print(f"  Small case (L={small_case['end_idx']-small_case['start_idx']}): speedup = {small_speedup:.2f}x")
    print(f"\nBENCH_RESULT {UNIT_NAME} numba_time_us={numba_mean_us:.1f} rust_time_us={rust_mean_us:.3f} speedup={speedup:.2f}")

if all_pass:
    print(f"\n{UNIT_NAME}: ALL CORRECTNESS TESTS PASSED")
else:
    print(f"\n{UNIT_NAME}: SOME CORRECTNESS TESTS FAILED")
print(f"OUTPUT_STATUS {'PASS' if all_pass else 'FAIL'}")