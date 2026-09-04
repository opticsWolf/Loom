#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loom / Navette / Colour-Science — Extensive Benchmark & Parity Suite

This script validates all major functions from the `navette_color` Rust extension 
and the `loom_colorengine` Python module against the standard `colour-science` library.
It measures mathematical parity between Navette and Colour-Science, while benchmarking
the computational throughput speedup of Navette over Loom.
"""

import os
import sys
import time
import warnings
import numpy as np
import colour
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs"))
import loom_colorengine as ce
from navette import color as nc

# Suppress colour-science domain warnings (e.g., values slightly outside [0, 1])
warnings.filterwarnings('ignore', category=colour.utilities.ColourUsageWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_ACCURACY = 10_000       # Size of dataset for accuracy/parity assertions
N_PERF = 1_000_000        # Size of dataset for speed benchmarking

# White points
ILLUM_xy_D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
ILLUM_XYZ_D65 = ce.REF_WHITE_D65
ILLUM_XYZ_D50 = ce.REF_WHITE_D50

print(f"===================================================================================================================")
print(f" Loom / Navette / Colour-Science Benchmark Suite")
print(f" Accuracy Batch Size: {N_ACCURACY:,} pixels")
print(f" Performance Batch Size: {N_PERF:,} pixels")
print(f"===================================================================================================================")

# -----------------------------------------------------------------------------
# Data Generation
# -----------------------------------------------------------------------------
print("Generating test datasets...", end="", flush=True)
np.random.seed(42)

# Edge cases prepended to the ACCURACY set so every task's parity check also
# exercises them. Black is the key one — zero denominators in xyY/Luv/UVW/UCS
# and an undefined hue — and is never produced by the random in-gamut data.
EDGE_RGB = np.array([
    [0.0,  0.0,  0.0 ],   # black
    [1.0,  1.0,  1.0 ],   # white
    [1e-4, 1e-4, 1e-4],   # near-black (tiny but non-zero denominators)
], dtype=np.float64)
N_EDGE = EDGE_RGB.shape[0]

# Random draw order is preserved, so the random rows are bit-identical to before
# (the edge rows are simply stacked on top).
rgb_rand = np.random.rand(N_ACCURACY, 3) * 0.99 + 0.01
rgb_acc = np.vstack([EDGE_RGB, rgb_rand])
rgb_perf = np.random.rand(N_PERF, 3) * 0.99 + 0.01

xyz_acc = ce.ColorSpaceEngine.srgb_to_xyz(rgb_acc)
xyz_perf = ce.ColorSpaceEngine.srgb_to_xyz(rgb_perf)

lab_acc = ce.ColorSpaceEngine.xyz_to_lab(xyz_acc)
lab_perf = ce.ColorSpaceEngine.xyz_to_lab(xyz_perf)

lch_acc = ce.ColorSpaceEngine.lab_to_lch(lab_acc)
lch_perf = ce.ColorSpaceEngine.lab_to_lch(lab_perf)

luv_acc = ce.ColorSpaceEngine.xyz_to_luv(xyz_acc)
luv_perf = ce.ColorSpaceEngine.xyz_to_luv(xyz_perf)

xyy_acc = ce.ColorSpaceEngine.xyz_to_xyY(xyz_acc)
xyy_perf = ce.ColorSpaceEngine.xyz_to_xyY(xyz_perf)

oklab_acc = ce.ColorSpaceEngine.xyz_to_oklab(xyz_acc)
oklab_perf = ce.ColorSpaceEngine.xyz_to_oklab(xyz_perf)

uvw_acc = ce.ColorSpaceEngine.xyz_to_uvw(xyz_acc)
uvw_perf = ce.ColorSpaceEngine.xyz_to_uvw(xyz_perf)

ucs_acc = ce.ColorSpaceEngine.xyz_to_ucs(xyz_acc)
ucs_perf = ce.ColorSpaceEngine.xyz_to_ucs(xyz_perf)

lab2_acc = lab_acc + np.random.randn(lab_acc.shape[0], 3) * 5.0
lab2_perf = lab_perf + np.random.randn(N_PERF, 3) * 5.0
print(" Done.\n")

# -----------------------------------------------------------------------------
# Property & Edge Case Tests (Mirroring func_05 & func_08)
# -----------------------------------------------------------------------------
def run_property_tests():
    print("===================================================================================================================")
    print(" Unit Property & Edge Case Tests (func_05 & func_08 Equivalents)")
    print("===================================================================================================================")

    def test_assert(name, condition, details=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name:<35} {details}")

    # --- func_05: Round Trip In-Gamut ---
    try:
        rgb_in = np.array([
            [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 
            [0.2, 0.6, 0.9], [0.9, 0.1, 0.3]
        ], dtype=np.float64)
        
        oklab_out_fn = getattr(nc, 'sRGB_to_Oklab', None)
        srgb_out_fn = getattr(nc, 'Oklab_to_sRGB', None)
        if oklab_out_fn is not None and srgb_out_fn is not None:
            nav_oklab = oklab_out_fn(rgb_in)
            nav_round = srgb_out_fn(nav_oklab)
            diff = np.max(np.abs(rgb_in - nav_round))
            test_assert("func_05: round_trip_in_gamut", diff < 1e-12, f"(diff={diff:.2e})")
        else:
            print("[SKIP] func_05: round_trip_in_gamut (Functions not found)")
    except Exception as e:
        print(f"[FAIL] func_05: round_trip_in_gamut (Exception: {e})")

    # --- func_05: Oklab Neutral Grey ---
    try:
        grey_in = np.array([[0.4, 0.4, 0.4], [0.7, 0.7, 0.7]], dtype=np.float64)
        oklab_out_fn = getattr(nc, 'sRGB_to_Oklab', None)
        if oklab_out_fn is not None:
            res = oklab_out_fn(grey_in)
            a_max = np.max(np.abs(res[:, 1]))
            b_max = np.max(np.abs(res[:, 2]))
            test_assert("func_05: neutral_grey_property", a_max < 1e-12 and b_max < 1e-12, f"(max_a={a_max:.2e}, max_b={b_max:.2e})")
        else:
            print("[SKIP] func_05: neutral_grey_property (nc.sRGB_to_Oklab not found)")
    except Exception as e:
        print(f"[FAIL] func_05: neutral_grey_property (Exception: {e})")

    # --- func_05: Oklab Clip Behaviour ---
    try:
        rgb_over = np.array([[1.5, -0.2, 0.8]], dtype=np.float64)
        rgb_clam = np.array([[1.0,  0.0, 0.8]], dtype=np.float64)
        oklab_out_fn = getattr(nc, 'sRGB_to_Oklab', None)
        if oklab_out_fn is not None:
            nav_over = oklab_out_fn(rgb_over)
            nav_clam = oklab_out_fn(rgb_clam)
            diff = np.max(np.abs(nav_over - nav_clam))
            test_assert("func_05: clip_behaviour", diff < 1e-12, f"(diff={diff:.2e})")
        else:
            print("[SKIP] func_05: clip_behaviour (nc.sRGB_to_Oklab not found)")
    except Exception as e:
        print(f"[FAIL] func_05: clip_behaviour (Exception: {e})")

    # --- func_08: Identity Short-Circuit ---
    try:
        xyz_in = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]], dtype=np.float64)
        adapt_fn = getattr(nc, 'chromatic_adaptation_VonKries', None)
        if adapt_fn is not None:
            out_nav = adapt_fn(xyz_in, ILLUM_XYZ_D65, ILLUM_XYZ_D65)
            diff = np.max(np.abs(out_nav - xyz_in))
            test_assert("func_08: identity_short_circuit", diff < 1e-12, f"(diff={diff:.2e})")
        else:
            print("[SKIP] func_08: identity_short_circuit (nc.chromatic_adaptation_VonKries not found)")
    except Exception as e:
        print(f"[FAIL] func_08: identity_short_circuit (Exception: {e})")

    # --- func_08: Negative Clipping ---
    try:
        xyz_neg = np.array([[-0.1, 0.5, 0.5]], dtype=np.float64)
        adapt_fn = getattr(nc, 'chromatic_adaptation_VonKries', None)
        if adapt_fn is not None:
            out_nav = adapt_fn(xyz_neg, ILLUM_XYZ_D65, ILLUM_XYZ_D50)
            test_assert("func_08: negative_clipping", out_nav[0, 0] == 0.0 and out_nav[0, 1] > 0.0 and out_nav[0, 2] > 0.0, f"(X={out_nav[0,0]:.4f})")
        else:
            print("[SKIP] func_08: negative_clipping")
    except Exception as e:
        print(f"[FAIL] func_08: negative_clipping (Exception: {e})")

    # --- func_08: D65 to D50 Consistency ---
    try:
        xyz_d65 = np.array([ILLUM_XYZ_D65], dtype=np.float64)
        adapt_fn = getattr(nc, 'chromatic_adaptation_VonKries', None)
        if adapt_fn is not None:
            out_nav = adapt_fn(xyz_d65, ILLUM_XYZ_D65, ILLUM_XYZ_D50)
            diff = np.max(np.abs(out_nav[0] - ILLUM_XYZ_D50))
            test_assert("func_08: d65_to_d50_consistency", diff < 1e-6, f"(diff={diff:.2e})")
        else:
            print("[SKIP] func_08: d65_to_d50_consistency")
    except Exception as e:
        print(f"[FAIL] func_08: d65_to_d50_consistency (Exception: {e})")

    # --- func_08: Calc Matrix Round Trip ---
    try:
        # Tries to find the exact matrix function exposed in navette_color
        matrix_fn = getattr(nc, 'calc_transform_matrix', getattr(nc, 'matrix_chromatic_adaptation_VonKries', None))
        if matrix_fn is not None:
            m = np.array(matrix_fn(ILLUM_XYZ_D65, ILLUM_XYZ_D50))
            d65_vec = np.array(ILLUM_XYZ_D65)
            d50_vec = np.array(ILLUM_XYZ_D50)
            
            adapted_row = d65_vec @ m
            adapted_col = m @ d65_vec
            
            diff_row = np.max(np.abs(adapted_row - d50_vec))
            diff_col = np.max(np.abs(adapted_col - d50_vec))
            
            # This helps identify if the matrix transposition in Rust is flipped (col vs row major)
            if diff_row < 1e-12:
                test_assert("func_08: calc_matrix_round_trip", True, f"(Row-major matched, diff={diff_row:.2e})")
            elif diff_col < 1e-12:
                test_assert("func_08: calc_matrix_round_trip", True, f"(Col-major matched, diff={diff_col:.2e})")
            else:
                test_assert("func_08: calc_matrix_round_trip", False, f"(FAIL: diff_row={diff_row:.2e}, diff_col={diff_col:.2e})")
        else:
            print("[SKIP] func_08: calc_matrix_round_trip (Matrix function not found in nc)")
    except Exception as e:
        print(f"[FAIL] func_08: calc_matrix_round_trip (Exception: {e})")
        
    print("===================================================================================================================\n")

run_property_tests()


# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------
def run_fn(fn, args):
    if fn is None: return None
    try:
        return fn(*args)
    except Exception as e:
        return None

def bench_fn(fn, args):
    if fn is None: return None
    # Warmup (critical for Numba JIT compilation)
    try:
        fn(*args)
    except Exception:
        return None
    
    t0 = time.perf_counter()
    fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms

def format_err(err):
    if err is None: return "N/A"
    if err < 1e-12: return "Exact"
    return f"{err:.2e}"

def format_time(t):
    return f"{t:.1f}" if t is not None else "N/A"

def format_speedup(t_base, t_fast):
    if t_base is None or t_fast is None or t_fast <= 0: return "N/A"
    return f"{t_base / t_fast:.1f}x"

def print_table_header():
    fmt = "| {:<19} | {:<12} | {:<12} | {:<10} | {:>9} | {:>8} | {:>8} | {:>9} |"
    header = fmt.format("Operation", "Nav vs Col", "Loom vs Col", "Edge Err",
                        "Colour(ms)", "Loom(ms)", "Nav(ms)", "Nav vs Lm")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    return fmt, sep

def _max_err(nav, col, sl):
    if nav is None or col is None:
        return None
    d = np.abs(np.asarray(nav)[sl] - np.asarray(col)[sl])
    return float(np.max(d)) if d.size else None

def evaluate_task(name, args_acc, args_perf, fn_col, fn_loom, fn_nav, fmt):
    # 1. Accuracy Pass — BOTH engines vs the colour-science reference.
    out_col = run_fn(fn_col, args_acc)
    out_nav = run_fn(fn_nav, args_acc)
    out_loom = run_fn(fn_loom, args_acc)

    # Navette parity split into random vs edge rows; Loom parity over random rows
    # (so Loom's correctness is actually checked, not just its speed).
    err_nav = _max_err(out_nav, out_col, slice(N_EDGE, None))
    err_loom = _max_err(out_loom, out_col, slice(N_EDGE, None))
    err_edge = _max_err(out_nav, out_col, slice(0, N_EDGE))

    # 2. Performance Pass
    t_col = bench_fn(fn_col, args_perf)
    t_loom = bench_fn(fn_loom, args_perf)
    t_nav = bench_fn(fn_nav, args_perf)

    # 3. Print
    print(fmt.format(
        name,
        format_err(err_nav),
        format_err(err_loom),
        format_err(err_edge),
        format_time(t_col),
        format_time(t_loom),
        format_time(t_nav),
        format_speedup(t_loom, t_nav)
    ))

# -----------------------------------------------------------------------------
# Tasks Definition
# -----------------------------------------------------------------------------
tasks = [
    # --- Base Color Spaces ---
    ("sRGB -> XYZ", (rgb_acc,), (rgb_perf,), 
     colour.sRGB_to_XYZ, ce.ColorSpaceEngine.srgb_to_xyz, nc.sRGB_to_XYZ),
     
    ("XYZ -> sRGB", (xyz_acc,), (xyz_perf,), 
     colour.XYZ_to_sRGB, ce.ColorSpaceEngine.xyz_to_srgb, nc.XYZ_to_sRGB),
     
    ("XYZ -> Lab", (xyz_acc,), (xyz_perf,), 
     lambda x: colour.XYZ_to_Lab(x, ILLUM_xy_D65), ce.ColorSpaceEngine.xyz_to_lab, nc.XYZ_to_Lab),
     
    ("Lab -> XYZ", (lab_acc,), (lab_perf,), 
     lambda x: colour.Lab_to_XYZ(x, ILLUM_xy_D65), ce.ColorSpaceEngine.lab_to_xyz, nc.Lab_to_XYZ),
     
    ("XYZ -> xyY", (xyz_acc,), (xyz_perf,), 
     colour.XYZ_to_xyY, ce.ColorSpaceEngine.xyz_to_xyY, nc.XYZ_to_xyY),
     
    ("xyY -> XYZ", (xyy_acc,), (xyy_perf,), 
     colour.xyY_to_XYZ, ce.ColorSpaceEngine.xyY_to_xyz, nc.xyY_to_XYZ),
     
    ("Lab -> LCh", (lab_acc,), (lab_perf,), 
     colour.Lab_to_LCHab, ce.ColorSpaceEngine.lab_to_lch, nc.Lab_to_LCHab),
     
    ("LCh -> Lab", (lch_acc,), (lch_perf,), 
     colour.LCHab_to_Lab, ce.ColorSpaceEngine.lch_to_lab, nc.LCHab_to_Lab),
     
    ("XYZ -> Luv", (xyz_acc,), (xyz_perf,), 
     lambda x: colour.XYZ_to_Luv(x, ILLUM_xy_D65), ce.ColorSpaceEngine.xyz_to_luv, nc.XYZ_to_Luv),
     
    ("Luv -> XYZ", (luv_acc,), (luv_perf,), 
     lambda x: colour.Luv_to_XYZ(x, ILLUM_xy_D65), ce.ColorSpaceEngine.luv_to_xyz, nc.Luv_to_XYZ),
     
    ("XYZ -> Oklab", (xyz_acc,), (xyz_perf,), 
     colour.XYZ_to_Oklab, ce.ColorSpaceEngine.xyz_to_oklab, nc.XYZ_to_Oklab),
     
    ("Oklab -> XYZ", (oklab_acc,), (oklab_perf,), 
     colour.Oklab_to_XYZ, ce.ColorSpaceEngine.oklab_to_xyz, nc.Oklab_to_XYZ),

    # Legacy direct sRGB to Oklab pipelines (using getattr since it's an extension of the original script)
    ("sRGB -> Oklab (dir)", (rgb_acc,), (rgb_perf,), 
     lambda x: colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(x)), 
     getattr(ce.ColorSpaceEngine, 'srgb_to_oklab', None), getattr(nc, 'sRGB_to_Oklab', None)),
     
    ("Oklab -> sRGB (dir)", (oklab_acc,), (oklab_perf,), 
     lambda x: colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(x)), 
     getattr(ce.ColorSpaceEngine, 'oklab_to_srgb', None), getattr(nc, 'Oklab_to_sRGB', None)),
     
    ("XYZ -> UCS (1960)", (xyz_acc,), (xyz_perf,), 
     colour.XYZ_to_UCS, ce.ColorSpaceEngine.xyz_to_ucs, nc.XYZ_to_UCS),
     
    ("UCS -> XYZ", (ucs_acc,), (ucs_perf,), 
     colour.UCS_to_XYZ, ce.ColorSpaceEngine.ucs_to_xyz, nc.UCS_to_XYZ),
     
    ("XYZ -> UVW (1964)", (xyz_acc,), (xyz_perf,), 
    lambda x: colour.XYZ_to_UVW(x * 100.0, ILLUM_xy_D65), 
    ce.ColorSpaceEngine.xyz_to_uvw, nc.XYZ_to_UVW),
     
    ("UVW -> XYZ", (uvw_acc,), (uvw_perf,), 
    lambda x: colour.UVW_to_XYZ(x, ILLUM_xy_D65) / 100.0, 
    ce.ColorSpaceEngine.uvw_to_xyz, nc.UVW_to_XYZ),

    # --- Convenience Composites ---
    ("sRGB -> Lab", (rgb_acc,), (rgb_perf,),
     lambda x: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(x), ILLUM_xy_D65), ce.ColorSpaceEngine.srgb_to_lab, nc.sRGB_to_Lab),
     
    ("Lab -> sRGB", (lab_acc,), (lab_perf,),
     lambda x: colour.XYZ_to_sRGB(colour.Lab_to_XYZ(x, ILLUM_xy_D65)), ce.ColorSpaceEngine.lab_to_srgb, nc.Lab_to_sRGB),

    # --- Chromatic Adaptation ---
    ("Bradford Adapt", (xyz_acc,), (xyz_perf,),
     lambda x: colour.adaptation.chromatic_adaptation_VonKries(x, ILLUM_XYZ_D65, ILLUM_XYZ_D50, transform='Bradford'),
     lambda x: ce.ChromaticAdaptation.adapt(x, ILLUM_XYZ_D65, ILLUM_XYZ_D50),
     lambda x: nc.chromatic_adaptation_VonKries(x, ILLUM_XYZ_D65, ILLUM_XYZ_D50)),

    # --- Color Metrics ---
    ("Delta E 76", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     colour.difference.delta_E_CIE1976, ce.ColorMetrics.delta_E_76, nc.delta_E_CIE1976),
     
    ("Delta E 94", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     lambda a, b: colour.difference.delta_E_CIE1994(a, b, textiles=False), 
     ce.ColorMetrics.delta_E_94, nc.delta_E_CIE1994),
     
    ("Delta E CMC", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     lambda a, b: colour.difference.delta_E_CMC(a, b, l=2, c=1), 
     ce.ColorMetrics.delta_E_CMC, nc.delta_E_CMC),
     
    ("Delta E DIN99", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     colour.difference.delta_E_DIN99, ce.ColorMetrics.delta_E_DIN99, nc.delta_E_DIN99),

    # --- Non-default presets (the paths the default-only rows never exercise) ---
    ("Delta E 94 tex", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     lambda a, b: colour.difference.delta_E_CIE1994(a, b, textiles=True),
     lambda a, b: ce.ColorMetrics.delta_E_94(a, b, textiles=True),
     lambda a, b: nc.delta_E_CIE1994(a, b, True)),

    ("Delta E CMC 1:1", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     lambda a, b: colour.difference.delta_E_CMC(a, b, l=1, c=1),
     lambda a, b: ce.ColorMetrics.delta_E_CMC(a, b, pl=1, pc=1),
     lambda a, b: nc.delta_E_CMC(a, b, 1.0, 1.0)),

    ("Delta E DIN99 tex", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     lambda a, b: colour.difference.delta_E_DIN99(a, b, textiles=True),
     lambda a, b: ce.ColorMetrics.delta_E_DIN99(a, b, textiles=True),
     lambda a, b: nc.delta_E_DIN99(a, b, True)),

    ("Delta E 2000", (lab_acc, lab2_acc), (lab_perf, lab2_perf),
     colour.difference.delta_E_CIE2000, ce.ColorMetrics.delta_E_2000, nc.delta_E_CIE2000),
]

fmt, sep = print_table_header()

for t in tasks:
    evaluate_task(*t, fmt)

print(sep)
print("\nNav vs Col / Loom vs Col = max |engine - Colour| over random in-gamut rows.")
print("Edge Err = max |Nav - Colour| over edge rows: black, white, near-black.")
print("Benchmark Suite Completed successfully.")