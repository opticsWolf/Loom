#!/usr/bin/env python3
"""
navette_target_bench.py — stress-test + benchmark for the Rust TargetWeaver
(constraint ingestion + Merit Function) in `navette.spectralweave`.

Unlike navette_spectral_bench.py, there is **no pure-Python reference** for the
TargetWeaver (the shipped wrappers just delegate to Rust). So correctness here
is anchored two ways instead of a Python-vs-Rust diff:

  1. A faithful NumPy reference of the *linear / aligned-grid* merit. When the
     simulation grid equals the target grid, interpolation collapses to identity,
     so the residual reduces to  ((sim - raw) * norm_factor / tol_floored)**2
     summed over points and targets — computable exactly in NumPy and compared
     against Rust `calculate_merit`.
  2. Analytic invariants that need no reimplementation of the interpolator:
       * exact-match sim  -> merit == 0
       * a missing key    -> exactly missing_penalty
       * above/below kind -> correct residual sign
       * doubling tolerances -> merit / 4   (residual ∝ 1/tol²)
       * duplicated target -> 2× merit      (additivity)
       * determinism      -> identical bits across repeated calls

The two hot paths measured are:
  * ingest_spectral   — building a TargetWeaver of N curves × M points
  * merit             — calculate_merit with sim grid == target grid (identity
                        interp; measures lookup + normalization + arithmetic)
  * merit_interp      — calculate_merit with a *denser* sim grid (exercises the
                        interpolation search — this is where an O(n+m) two-pointer
                        beats a per-point binary search)

Usage:
    python navette_target_bench.py                 # correctness + benchmark
    python navette_target_bench.py --corr-only
    python navette_target_bench.py --bench-only
    python navette_target_bench.py --quick
    python navette_target_bench.py --tag inc0-baseline   # label the run
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Engine discovery  (Rust-only — no Python fallback exists for TargetWeaver)
# --------------------------------------------------------------------------- #
try:
    from navette import spectralweave as sw
except Exception as exc:  # pragma: no cover
    print(f"FATAL: could not import navette.spectralweave: {exc}", file=sys.stderr)
    print("Build/install it first:  maturin develop --release", file=sys.stderr)
    raise SystemExit(2)

TargetWeaver = sw.TargetWeaver
OpticalWeaver = sw.OpticalWeaver
calculate_merit = sw.calculate_merit


def arr(x) -> np.ndarray:
    """Contiguous float64 — required by the Rust `as_slice()` path."""
    return np.ascontiguousarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# NumPy reference (linear mode, aligned grid => identity interpolation)
# --------------------------------------------------------------------------- #
def ref_norm_linear(raw: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mirror Rust `register_metadata` for mode == 'linear'."""
    t_avg = abs(raw.sum() / len(raw))
    nf = 1.0 / max(t_avg, 1e-12)
    return nf, raw * nf


def ref_merit_linear_aligned(
    curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    tol_floor: float,
) -> float:
    """
    Reference merit for a set of (raw_target, sim_value, tol, kind) curves that
    all share the sim/target wavelength grid (so sim_raw == sim_value exactly).
    """
    total = 0.0
    for raw, sim, tol, kind in curves:
        nf, scaled = ref_norm_linear(raw)
        tol_f = np.maximum(tol, tol_floor)
        scaled_diff = sim * nf - scaled           # == (sim - raw) * nf
        if kind == "a":
            scaled_diff = np.where(scaled_diff < 0.0, scaled_diff, 0.0)
        elif kind == "b":
            scaled_diff = np.where(scaled_diff > 0.0, scaled_diff, 0.0)
        total += float(np.sum((scaled_diff / tol_f) ** 2))
    return total


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #
def make_pair(
    m_points: int,
    n_targets: int,
    *,
    kind: str = "e",
    norm_mode: str = "linear",
    delta: float = 0.01,
    sim_density: int = 1,
    tol: float = 0.05,
    cache_size: int | None = None,
):
    """
    Build a matched (sim OpticalWeaver, TargetWeaver) pair with `n_targets`
    curves of `m_points` each. Returns (sim, tw, ref_curves) where ref_curves is
    the aligned-grid reference input (only meaningful when sim_density == 1).

    Each target uses key (angle=i, pol="s", spec="R"); the sim stores a matching
    curve under the same key. sim = target + `delta`.
    """
    cache = cache_size if cache_size is not None else max(8, n_targets * 2)
    tw = TargetWeaver(cache_size=cache)
    sim = OpticalWeaver(cache_size=cache)
    wl = arr(np.linspace(400.0, 800.0, m_points))

    ref_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for i in range(n_targets):
        raw = arr(np.sin(wl / (i + 50.0)) + 2.0)   # strictly positive -> stable norm
        tols = arr(np.full(m_points, tol))
        angle = float(i)
        tw.add_spectral_target(wl, raw, tols, angle, "s", "R", kind, norm_mode)

        if sim_density == 1:
            sim_wl = wl
            sim_vals = arr(raw + delta)
        else:
            sim_wl = arr(np.linspace(400.0, 800.0, m_points * sim_density))
            sim_vals = arr(np.sin(sim_wl / (i + 50.0)) + 2.0 + delta)
        sim.set_data((angle, "s", "R"), sim_vals, sim_wl)
        ref_curves.append((raw, arr(raw + delta), tols, kind))

    return sim, tw, ref_curves


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
RTOL, ATOL = 1e-9, 1e-9
MISSING = 1e6


def _approx(a: float, b: float, rtol=RTOL, atol=ATOL) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def run_correctness() -> bool:
    print("=" * 72)
    print("CORRECTNESS  (Rust merit vs NumPy reference / analytic invariants)")
    print("=" * 72)
    ok = True

    def check(label: str, cond: bool, detail: str = ""):
        nonlocal ok
        if cond:
            print(f"  [ok    ] {label}")
        else:
            ok = False
            print(f"  [FAIL  ] {label}  {detail}")

    # 1. exact match -> merit ~ 0
    sim, tw, _ = make_pair(200, 8, delta=0.0)
    m = calculate_merit(sim, tw, MISSING)
    check("exact_match_zero", _approx(m, 0.0, atol=1e-6), f"merit={m}")

    # 2. known linear residual vs NumPy reference (aligned grid, identity interp)
    sim, tw, refc = make_pair(500, 20, delta=0.013, tol=0.07)
    m = calculate_merit(sim, tw, MISSING)
    ref = ref_merit_linear_aligned(refc, tol_floor=1e-12)
    check("known_linear_vs_ref", _approx(m, ref), f"rust={m:.6e} ref={ref:.6e}")

    # 3. missing key -> exactly missing_penalty (one target, empty sim)
    tw2 = TargetWeaver(cache_size=8)
    wl = arr(np.linspace(400, 800, 100))
    tw2.add_spectral_target(wl, arr(np.ones(100) * 2), arr(np.full(100, 0.1)),
                            5.0, "s", "R", "e", "linear")
    empty_sim = OpticalWeaver(cache_size=8)
    m = calculate_merit(empty_sim, tw2, MISSING)
    check("missing_penalty", _approx(m, MISSING), f"merit={m}")

    # 4a. above kind, sim above target -> residual 0
    sim, tw, _ = make_pair(200, 5, kind="a", delta=+0.05)   # sim > target
    m = calculate_merit(sim, tw, MISSING)
    check("above_satisfied_zero", _approx(m, 0.0, atol=1e-6), f"merit={m}")

    # 4b. above kind, sim below target -> positive, matches reference
    sim, tw, refc = make_pair(200, 5, kind="a", delta=-0.05)  # sim < target
    m = calculate_merit(sim, tw, MISSING)
    ref = ref_merit_linear_aligned(refc, tol_floor=1e-12)
    check("above_violated_vs_ref", m > 0 and _approx(m, ref),
          f"rust={m:.6e} ref={ref:.6e}")

    # 5. tolerance scaling: doubling tol -> merit / 4
    sim, tw, _ = make_pair(300, 10, delta=0.02, tol=0.05)
    m1 = calculate_merit(sim, tw, MISSING)
    sim2, tw2, _ = make_pair(300, 10, delta=0.02, tol=0.10)
    m2 = calculate_merit(sim2, tw2, MISSING)
    check("tolerance_inverse_square", _approx(m1 / m2, 4.0, rtol=1e-6),
          f"ratio={m1 / m2:.6f}")

    # 6. additivity: two identical targets -> 2x one
    def one_and_two(n):
        tw = TargetWeaver(cache_size=8)
        sim = OpticalWeaver(cache_size=8)
        wl = arr(np.linspace(400, 800, 128))
        raw = arr(np.cos(wl / 60) + 2.0)
        tols = arr(np.full(128, 0.05))
        for i in range(n):
            tw.add_spectral_target(wl, raw, tols, float(i), "s", "R", "e", "linear")
            sim.set_data((float(i), "s", "R"), arr(raw + 0.02), wl)
        return calculate_merit(sim, tw, MISSING)
    check("additivity", _approx(one_and_two(2), 2.0 * one_and_two(1)),
          f"m1={one_and_two(1):.4e} m2={one_and_two(2):.4e}")

    # 7. determinism: identical bits across calls
    sim, tw, _ = make_pair(400, 15, delta=0.017, sim_density=3)  # interp path too
    vals = {calculate_merit(sim, tw, MISSING) for _ in range(5)}
    check("determinism", len(vals) == 1, f"distinct={len(vals)}")

    # 8. smoke: log / phase / auto modes must run without error
    try:
        for mode in ("auto", "log", "phase", "complex"):
            s, t, _ = make_pair(200, 4, norm_mode=mode, delta=0.05)
            _ = calculate_merit(s, t, MISSING)
        check("norm_modes_smoke", True)
    except Exception as e:  # pragma: no cover
        check("norm_modes_smoke", False, f"{type(e).__name__}: {e}")

    print(f"\nCorrectness: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def timeit(fn: Callable, repeats: int = 7, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_config(m_points: int, n_targets: int) -> Dict[str, float]:
    res: Dict[str, float] = {}
    cache = max(8, n_targets * 2)

    # --- ingest_spectral: build a fresh TargetWeaver of N curves × M points ---
    wl = arr(np.linspace(400.0, 800.0, m_points))
    raws = [arr(np.sin(wl / (i + 50.0)) + 2.0) for i in range(n_targets)]
    tols = arr(np.full(m_points, 0.05))

    def do_ingest():
        tw = TargetWeaver(cache_size=cache)
        for i in range(n_targets):
            tw.add_spectral_target(wl, raws[i], tols, float(i), "s", "R", "e", "linear")
    res["ingest_spectral"] = timeit(do_ingest)

    # --- merit: aligned grid (identity interp) ---
    sim, tw, _ = make_pair(m_points, n_targets, delta=0.01, sim_density=1)
    res["merit"] = timeit(lambda: calculate_merit(sim, tw, MISSING))

    # --- merit_interp: denser sim grid (exercises interpolation search) ---
    sim_i, tw_i, _ = make_pair(m_points, n_targets, delta=0.01, sim_density=3)
    res["merit_interp"] = timeit(lambda: calculate_merit(sim_i, tw_i, MISSING))

    return res


def run_benchmark(quick: bool, tag: str | None) -> None:
    print("=" * 72)
    hdr = "BENCHMARK   (median of repeated runs; lower is better)"
    if tag:
        hdr += f"   [tag: {tag}]"
    print(hdr)
    print("=" * 72)

    if quick:
        configs = [(1_000, 10), (1_000, 100)]
    else:
        configs = [
            (100, 10),
            (1_000, 50),
            (1_000, 500),
            (10_000, 20),
            (5_000, 200),
        ]

    ops = ["ingest_spectral", "merit", "merit_interp"]
    for m_points, n_targets in configs:
        print(f"\n  grid={m_points:,} pts | targets={n_targets}")
        r = bench_config(m_points, n_targets)
        print(f"    {'operation':18s} {'rust (ms)':>12s}")
        print("    " + "-" * 31)
        for op in ops:
            print(f"    {op:18s} {r[op] * 1e3:>12.3f}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-only", action="store_true", help="skip correctness phase")
    ap.add_argument("--corr-only", action="store_true", help="skip benchmark phase")
    ap.add_argument("--quick", action="store_true", help="smaller benchmark sizes")
    ap.add_argument("--tag", default=None, help="label printed in the benchmark header")
    args = ap.parse_args()

    print(f"navette.spectralweave @ {sw.__file__}")
    print(f"numpy {np.__version__}, python {sys.version.split()[0]}\n")

    ok = True
    if not args.bench_only:
        ok = run_correctness()
    if not args.corr_only:
        run_benchmark(args.quick, args.tag)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
