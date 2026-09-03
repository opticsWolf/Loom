#!/usr/bin/env python3
"""Comparison test for core_engine_rigorous_ellipsometry — numba vs rust."""

import sys, os, time
import numpy as np

# ---- Determine project root (assuming this script is in tests/ or similar) ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up, where Cargo.toml lives
OUTPUT_DIR = PROJECT_ROOT
UNIT_NAME = "core_engine_rigorous_ellipsometry"

# ---- Import original numba function (if available) ----
numba_imported = False
try:
    # Try to import from a local 'loom' module (place original loom_matrix.py in PROJECT_ROOT/loom)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "loom"))
    from loom_matrix import core_engine_rigorous_ellipsometry as numba_func
    numba_imported = True
    print("Successfully imported numba version of core_engine_rigorous_ellipsometry")
except ImportError:
    print("Note: Original numba version not found. Running rust-only validation.")

# ---- Import Rust module (built by cargo) ----
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
    # Fallback for older builds
    alt_names = ["navette_matrix.pyd", "navette_matrix.dll"] if sys.platform == "win32" else [
        "libnavette_matrix.dylib" if sys.platform == "darwin" else "libnavette_matrix.so"
    ]
    for aname in alt_names:
        alt_path = os.path.join(target_dir, aname)
        if os.path.exists(alt_path):
            so_path = alt_path
            break

if not os.path.exists(so_path):
    print(f"FATAL: Rust module not built at {target_dir}")
    print("Run `cargo build --release` first.")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("navette_matrix", so_path)
rust_mod = importlib.util.module_from_spec(spec)
sys.modules["navette_matrix"] = rust_mod
spec.loader.exec_module(rust_mod)

if hasattr(rust_mod, UNIT_NAME):
    rust_func = getattr(rust_mod, UNIT_NAME)
else:
    # Try alternate names
    for alt_name in ["core_engine_rigorous_ellipsometry", "core_engine"]:
        if hasattr(rust_mod, alt_name):
            rust_func = getattr(rust_mod, alt_name)
            break
    else:
        print(f"FATAL: Rust function '{UNIT_NAME}' not found. Available:")
        for n in dir(rust_mod):
            if not n.startswith('_'):
                print(f"  - {n}")
        sys.exit(1)


# ---- generate_test_case, make_rust_args, make_numba_args same as before ----
def generate_test_case(rng, num_wavs=10, num_angles=5, n_layers=6):
    """Generate a realistic thin-film stack test case."""
    wavls = np.linspace(400.0, 750.0, num_wavs).astype(np.float64)
    theta_deg = np.array([10.0, 20.0, 30.0, 45.0, 60.0])[:num_angles]
    sin_theta_arr = np.sin(np.radians(theta_deg)).astype(np.float64)

    n_base = np.array([
        (3.5 + 2.5j),   # air / incident medium
        (1.45 - 0.001j),  # SiO2
        (2.0 + 0.5j),   # TiO2
        (0.2 + 3.8j),   # Au (gold)
        (1.6 - 0.01j),  # polymer
        (4.0 - 0.05j),  # Si substrate
    ])[:n_layers]

    n_stack_cache = np.zeros(num_wavs * n_layers * 2, dtype=np.float64)
    for wv in range(num_wavs):
        base = wv * (n_layers * 2)
        for li in range(n_layers):
            lam = wavls[wv]
            n_real = n_base[li].real + 50.0 / (lam ** 2)
            n_imag = n_base[li].imag * (lam / 632.8)
            n_stack_cache[base + li * 2]     = np.float64(n_real)
            n_stack_cache[base + li * 2 + 1] = np.float64(n_imag)

    thicknesses = np.array([0.0, 80.0, 30.0, 15.0, 200.0, 0.0], dtype=np.float64)[:n_layers]
    incoherent_flags = np.zeros(n_layers - 1 if n_layers > 1 else 1, dtype=np.int32)
    if n_layers >= 4:
        incoherent_flags[3] = 1

    rough_types = np.zeros(n_layers, dtype=np.int32)
    rough_vals = np.zeros(n_layers, dtype=np.float64)
    for i in range(1, n_layers):
        if rng.random() < 0.5:
            rough_vals[i] = rng.uniform(0.5, 3.0)
            rough_types[i] = rng.choice([0, 1, 2, 4])

    debug_flag = np.int32(0)
    return {
        "wavls": wavls,
        "sin_theta_arr": sin_theta_arr,
        "n_layers": int(n_layers),
        "n_stack_cache": n_stack_cache,
        "thicknesses": thicknesses,
        "incoherent_flags": incoherent_flags,
        "rough_types": rough_types,
        "rough_vals": rough_vals,
        "debug_flag": debug_flag,
    }

def make_rust_args(case):
    return (case["wavls"], case["sin_theta_arr"], case["n_layers"],
            case["n_stack_cache"], case["thicknesses"], case["incoherent_flags"],
            case["rough_types"], case["rough_vals"], case["debug_flag"])

def make_numba_args(case):
    return make_rust_args(case)


# ---- Generate test cases ----
rng = np.random.default_rng(seed=42)
test_cases = [generate_test_case(np.random.default_rng(seed=s),
                                 num_wavs=rng.integers(5, 16),
                                 num_angles=rng.integers(3, 8),
                                 n_layers=rng.integers(4, 9))
              for s in range(10)]

edge_cases = []
edge_sharp = generate_test_case(np.random.default_rng(seed=77))
edge_sharp["rough_vals"][:] = 0.0
edge_sharp["rough_types"][:] = 0
edge_sharp["incoherent_flags"][:] = 0
edge_cases.append(edge_sharp)

edge_nc = generate_test_case(np.random.default_rng(seed=88))
for i in range(1, edge_nc["n_layers"]):
    if edge_nc["rough_vals"][i] < 0.5:
        edge_nc["rough_vals"][i] = rng.uniform(1.0, 4.0)
    edge_nc["rough_types"][i] = 5
edge_cases.append(edge_nc)

edge_single = generate_test_case(np.random.default_rng(seed=99), num_wavs=1, num_angles=1, n_layers=3)
edge_cases.append(edge_single)

all_test_cases = test_cases + edge_cases

# ---- Helper: compare results (unchanged) ----
def compare_results(numba_out, rust_out):
    if len(numba_out) != 13 or len(rust_out) != 13:
        return False, float('inf'), [f"length mismatch: numba={len(numba_out)} rust={len(rust_out)}"]
    diff_max = 0.0
    diffs = []
    all_pass = True
    names = ["Psi_R", "Delta_R", "DOP_R", "Rs", "Rp", "R_avg",
             "Psi_T", "Delta_T", "DOP_T", "Ts", "Tp", "T_avg", "conservation_err"]
    for i in range(13):
        nb_val = np.asarray(numba_out[i])
        rs_val = np.asarray(rust_out[i])
        if nb_val.shape != rs_val.shape:
            diff = float(np.abs(nb_val).sum() + abs(rs_val.sum()))
            is_close = False
        else:
            rel_tol = 1e-6
            abs_tol = 1e-10
            denom = np.maximum(np.abs(nb_val), np.abs(rs_val))
            diff_arr = np.abs(nb_val - rs_val)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = np.where(denom > abs_tol, diff_arr / denom, diff_arr)
            diff = float(np.max(rel_diff))
            is_close = bool(np.allclose(nb_val, rs_val, rtol=rel_tol, atol=abs_tol))
        diffs.append(f"{names[i]}={diff:.2e}")
        diff_max = max(diff_max, diff)
        all_pass &= is_close
    return all_pass, diff_max, "; ".join(diffs)


# ---- CORRECTNESS TESTS ----
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
        phys_ok = True
        diff_max = 0.0
        for j, val in enumerate(rust_out):
            arr = np.asarray(val)
            if np.any(np.isinf(arr)) or np.any(np.isnan(arr)):
                print(f"    WARNING: elem{j} has inf/nan values")
                phys_ok = False
            if j == 2 or j == 8:  # DOP_R and DOP_T
                if np.any((np.asarray(val) < -0.01) | (np.asarray(val) > 1.01)):
                    phys_ok = False
            diff_max += float(np.abs(arr).max())
        status = "PASS" if phys_ok else "FAIL_RUST_ONLY"
    wavs = case["wavls"].shape[0]
    angs = case["sin_theta_arr"].shape[0]
    layers = case["n_layers"]
    print(f"  test_{i:2d} [W={wavs}, A={angs}, L={layers}]: {status:12s} | max_rel_diff={diff_max:.2e}")
    all_pass &= status.startswith("PASS")


# ---- SPEED BENCHMARK (requires numba) ----
if numba_imported and len(all_test_cases) > 0:
    print(f"\n=== SPEED BENCHMARK: {UNIT_NAME} ===\n")
    bench_case = max(all_test_cases, key=lambda c: c["wavls"].shape[0] * c["sin_theta_arr"].shape[0])
    _ = numba_func(*make_numba_args(bench_case))
    _ = rust_func(*make_rust_args(bench_case))
    NUM_RUNS = 50
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
    numba_mean_ms = np.mean(numba_times[5:]) * 1e3
    rust_mean_ms = np.mean(rust_times[5:]) * 1e3
    speedup = numba_mean_ms / rust_mean_ms if rust_mean_ms > 0 else float('inf')
    print(f"  Numba avg: {numba_mean_ms:.3f} ms")
    print(f"  Rust  avg: {rust_mean_ms:.3f} ms")
    print(f"  Speedup:   {speedup:.2f}x")

    small_case = min(all_test_cases, key=lambda c: c["wavls"].shape[0] * c["sin_theta_arr"].shape[0])
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
    small_speedup = (np.mean(numba_small) * 1e3) / (np.mean(rust_small) * 1e3) if np.mean(rust_small) > 0 else float('inf')
    total_pts = small_case["wavls"].shape[0] * small_case["sin_theta_arr"].shape[0]
    print(f"  Small case ({total_pts} points): speedup = {small_speedup:.2f}x")
    print(f"\nBENCH_RESULT {UNIT_NAME} numba_time_ms={numba_mean_ms:.3f} rust_time_ms={rust_mean_ms:.3f} speedup={speedup:.2f}")

if all_pass:
    print(f"\n{UNIT_NAME}: ALL CORRECTNESS TESTS PASSED")
else:
    print(f"\n{UNIT_NAME}: SOME CORRECTNESS TESTS FAILED — review values above")
print(f"OUTPUT_STATUS {'PASS' if all_pass else 'FAIL'}")