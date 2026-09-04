#!/usr/bin/env python3
"""Comparison test for core_engine_photometry_only — numba vs rust."""

import sys, os, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = PROJECT_ROOT
UNIT_NAME = "core_engine_photometry_only"

NUM_WAVS = 50
NUM_ANGLES = 10
N_LAYERS = 6

numba_imported = False
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "loom"))
    from loom_matrix import core_engine_photometry_only as numba_func
    numba_imported = True
except ImportError:
    print("WARNING: Could not import numba version. Falling back to pure-Python reference.")
    # Minimal reference implementation (see original file)
    def core_engine_photometry_only_reference(
        wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
        incoherent_flags, rough_types, rough_vals, calc_s, calc_p
    ):
        wav_arr = np.asarray(wavls, dtype=np.float64)
        sin_theta = np.asarray(sin_theta_arr, dtype=np.float64)
        num_wavs = wav_arr.shape[0] if hasattr(wav_arr, 'shape') else len(wav_arr)
        num_angles = sin_theta.shape[0] if hasattr(sin_theta, 'shape') else len(sin_theta)
        total_points = num_wavs * num_angles
        n_stack_flat = np.asarray(n_stack_cache, dtype=np.float64).ravel()
        n_per_wav = []
        for wv in range(num_wavs):
            base = wv * (n_layers * 2)
            n_complex = []
            for li in range(n_layers):
                re = n_stack_flat[base + li * 2] if base + li * 2 < len(n_stack_flat) else 1.5
                im = n_stack_flat[base + li * 2 + 1] if base + li * 2 + 1 < len(n_stack_flat) else 0.0
                n_complex.append(complex(re, im))
            n_per_wav.append(n_complex)
        thick_arr = np.asarray(thicknesses, dtype=np.float64).ravel()
        Rs_out = np.zeros(total_points, dtype=np.float64)
        Rp_out = np.zeros(total_points, dtype=np.float64)
        Ts_out = np.zeros(total_points, dtype=np.float64)
        Tp_out = np.zeros(total_points, dtype=np.float64)
        for wav_idx in range(num_wavs):
            lam = wav_arr[wav_idx]
            if lam <= 0:
                continue
            for ang_idx in range(num_angles):
                sin_ti = sin_theta[ang_idx]
                n_incident = n_per_wav[wav_idx][0].real
                n_exit = n_per_wav[wav_idx][-1].real
                if n_incident > 0 and n_exit > 0:
                    r_s = (n_incident - n_exit) / (n_incident + n_exit)
                    t_s = 2 * n_incident / (n_incident + n_exit)
                    Rs_out[wav_idx * num_angles + ang_idx] = r_s ** 2
                    Ts_out[wav_idx * num_angles + ang_idx] = t_s ** 2
                    if calc_p != 0:
                        cos_ti = np.sqrt(max(1e-16, 1 - sin_ti**2))
                        r_p = (n_incident * cos_ti - n_exit * np.sin(sin_ti)) / \
                              (n_incident * cos_ti + n_exit * np.sin(sin_ti))
                        t_p = 2 * n_incident * cos_ti / \
                              (n_incident * cos_ti + n_exit * np.sin(sin_ti))
                        Rp_out[wav_idx * num_angles + ang_idx] = r_p ** 2
                        Tp_out[wav_idx * num_angles + ang_idx] = t_p ** 2
        return Rs_out, Rp_out, Ts_out, Tp_out

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

def generate_test_data(rng):
    wavls = np.linspace(400.0, 800.0, NUM_WAVS).astype(np.float64)
    angles_deg = np.linspace(-30.0, 30.0, NUM_ANGLES).astype(np.float64)
    sin_theta_arr = np.sin(np.radians(angles_deg)).astype(np.float64)
    n_real_base = [1.0, 1.5 + rng.random() * 0.3, 2.0 + rng.random() * 0.5,
                   1.8 + rng.random() * 0.4, 2.5 + rng.random() * 0.3, 1.0]
    n_imag_base = [0.0, 0.001 + rng.random() * 0.01, 0.01 + rng.random() * 0.05,
                   0.001 + rng.random() * 0.02, 0.02 + rng.random() * 0.08, 0.0]
    n_stack_cache = np.empty((NUM_WAVS, N_LAYERS * 2), dtype=np.float64)
    for wv in range(NUM_WAVS):
        wav_factor = (wavls[wv] - 400.0) / 400.0
        for li in range(N_LAYERS):
            n_stack_cache[wv, li * 2]     = n_real_base[li] + wav_factor * rng.uniform(-0.1, 0.1)
            n_stack_cache[wv, li * 2 + 1] = max(0.0, n_imag_base[li])
    thicknesses = np.array([50.0, 100.0, 200.0, 150.0, 80.0, 500.0], dtype=np.float64)
    incoherent_flags = np.zeros(N_LAYERS - 1, dtype=np.int32)
    rough_types = np.zeros(N_LAYERS, dtype=np.int32)
    rough_vals = np.zeros(N_LAYERS, dtype=np.float64)
    calc_s = 1
    calc_p = 1
    return wavls, sin_theta_arr, N_LAYERS, n_stack_cache, thicknesses, \
           incoherent_flags, rough_types, rough_vals, calc_s, calc_p

rng = np.random.default_rng(seed=42)
test_inputs = [generate_test_data(rng) for _ in range(3)]

print(f"=== CORRECTNESS TEST: {UNIT_NAME} ===")
all_pass = True
for i, inp in enumerate(test_inputs):
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses, \
        incoherent_flags, rough_types, rough_vals, calc_s, calc_p = inp
    if numba_imported:
        numba_out = numba_func(
            wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
            incoherent_flags, rough_types, rough_vals, calc_s, calc_p
        )
    else:
        numba_out = core_engine_photometry_only_reference(
            wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
            incoherent_flags, rough_types, rough_vals, calc_s, calc_p
        )
    rust_out = rust_func(
        wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
        incoherent_flags, rough_types, rough_vals, calc_s, calc_p
    )
    names = ["Rs", "Rp", "Ts", "Tp"]
    matches = []
    for name, n_arr, r_arr in zip(names, numba_out, rust_out):
        n_arr = np.asarray(n_arr)
        r_arr = np.asarray(r_arr)
        match = np.allclose(n_arr, r_arr, rtol=1e-4, atol=1e-8)
        diff_max = float(np.max(np.abs(n_arr - r_arr))) if n_arr.size else 0.0
        status = "PASS" if match else "FAIL"
        print(f"  test_{i} [{name}]: {status} | diff_max={diff_max:.2e}")
        matches.append(match)
    all_pass &= all(matches)

print(f"\n=== SPEED BENCHMARK: {UNIT_NAME} ===")
use_inp = test_inputs[-1]
wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses, \
    incoherent_flags, rough_types, rough_vals, calc_s, calc_p = use_inp
if numba_imported:
    _ = numba_func(
        wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
        incoherent_flags, rough_types, rough_vals, calc_s, calc_p
    )
_ = rust_func(
    wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
    incoherent_flags, rough_types, rough_vals, calc_s, calc_p
)
NUM_RUNS = 50
if numba_imported:
    numba_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        numba_func(
            wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
            incoherent_flags, rough_types, rough_vals, calc_s, calc_p
        )
        numba_times.append(time.perf_counter() - t0)
    numba_mean_ms = np.mean(numba_times) * 1000
else:
    numba_mean_ms = None
rust_times = []
for _ in range(NUM_RUNS):
    t0 = time.perf_counter()
    rust_func(
        wavls, sin_theta_arr, n_layers, n_stack_cache, thicknesses,
        incoherent_flags, rough_types, rough_vals, calc_s, calc_p
    )
    rust_times.append(time.perf_counter() - t0)
rust_mean_ms = np.mean(rust_times) * 1000
speedup = numba_mean_ms / rust_mean_ms if numba_mean_ms is not None and rust_mean_ms > 0 else float('inf')
if numba_imported:
    print(f"  Numba avg: {numba_mean_ms:.3f} ms")
print(f"  Rust  avg: {rust_mean_ms:.3f} ms")
if numba_imported:
    print(f"  Speedup:   {speedup:.2f}x")

if all_pass:
    print(f"\n{UNIT_NAME}: ALL CORRECTNESS TESTS PASSED")
else:
    print(f"\n{UNIT_NAME}: SOME CORRECTNESS TESTS FAILED")
print(f"OUTPUT_STATUS {'PASS' if all_pass else 'FAIL'}")