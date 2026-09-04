#!/usr/bin/env python3
"""
loom_spectral_bench.py — stress-test + benchmark for the Loom spectral engine.

Runs an identical workload against:
  * the pure-Python reference   (loom_spectraldata)
  * the Rust extension          (navette_spectralweave, built via `maturin develop --release`)

Two phases:
  1. CORRECTNESS  — many scenarios, cross-checked Python-vs-Rust for bit/numeric
                    equality, plus regression tests for the realigned cache
                    generation semantics.
  2. BENCHMARK    — timed across grid sizes / frame counts, reports speedup.

If `navette_spectralweave` is not importable, the script still runs (Python-only): correctness
becomes a self-consistency check and the benchmark reports Python timings alone.

Usage:
    python loom_spectral_bench.py                # both phases
    python loom_spectral_bench.py --bench-only
    python loom_spectral_bench.py --quick        # smaller benchmark sizes
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Engine discovery
# --------------------------------------------------------------------------- #
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "refs"))
import loom_spectraldata as loom_py  # pure-Python reference (required)

try:
    import navette.spectralweave as navette_spectralweave # Rust extension (optional)
    HAVE_RUST = True
except Exception as exc:  # pragma: no cover
    navette_spectralweave = None
    HAVE_RUST = False
    _RUST_IMPORT_ERR = exc

ENGINES = ["python"] + (["rust"] if HAVE_RUST else [])


def make_weaver(engine: str, cache_size: int = 128):
    if engine == "python":
        return loom_py.OpticalWeaver(cache_size=cache_size)
    # The rust module exports the class purely as OpticalWeaver now
    return navette_spectralweave.OpticalWeaver(cache_size=cache_size)


def get_gen(w) -> int:
    """Read the structural generation counter regardless of engine."""
    g = getattr(w, "generation", None)
    return int(g) if g is not None else int(getattr(w, "_gen"))


def arr(x) -> np.ndarray:
    """Contiguous float64 — required by the Rust `as_slice()` path."""
    return np.ascontiguousarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
RTOL, ATOL = 1e-9, 1e-12


def eq_arrays(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.allclose(a, b, rtol=RTOL, atol=ATOL)


def norm_collections(groups) -> Dict[tuple, Dict[tuple, np.ndarray]]:
    """Normalise get_weaved_collections output into an order-independent map."""
    out: Dict[tuple, Dict[tuple, np.ndarray]] = {}
    for wl, data_map in groups:
        wl = np.asarray(wl)
        wl_key = tuple(np.round(wl, 9).tolist())
        inner = {tuple(k): np.asarray(v) for k, v in dict(data_map).items()}
        out[wl_key] = inner
    return out


def compare_results(name: str, py_res, rs_res) -> None:
    """Assert two engine results are equivalent; raise AssertionError on diff."""
    if isinstance(py_res, tuple) and len(py_res) == 2 and isinstance(py_res[0], np.ndarray):
        # (wl, data) pair
        assert eq_arrays(py_res[0], rs_res[0]), f"{name}: wavelength mismatch"
        assert eq_arrays(py_res[1], rs_res[1]), f"{name}: data mismatch"
    elif isinstance(py_res, dict):
        assert py_res.keys() == rs_res.keys(), f"{name}: collection geometry sets differ"
        for wl_key in py_res:
            pi, ri = py_res[wl_key], rs_res[wl_key]
            assert pi.keys() == ri.keys(), f"{name}: keys differ for geometry"
            for k in pi:
                assert eq_arrays(pi[k], ri[k]), f"{name}: data differs for {k}"
    else:
        assert py_res == rs_res, f"{name}: {py_res!r} != {rs_res!r}"


# --------------------------------------------------------------------------- #
# Scenario builders — each takes a weaver factory, returns comparable output
# --------------------------------------------------------------------------- #
def s_basic(make) -> Tuple[np.ndarray, np.ndarray]:
    w = make()
    wl = arr(np.linspace(400, 800, 200))
    w.set_data((550.0, "R", "s"), arr(np.sin(wl / 100)), wl)
    return w.get_weaved((550.0, "R", "s"))


def s_multiframe_weave(make) -> Tuple[np.ndarray, np.ndarray]:
    """Same key spread over three contiguous, non-overlapping sub-bands."""
    w = make()
    full = arr(np.arange(400.0, 800.0, 1.0))  # 400 points
    bands = [full[0:120], full[120:300], full[300:400]]
    key = (632.8, "T", "p")
    for b in bands:
        w.set_data(key, arr(np.cos(b / 50)), arr(b))
    return w.get_weaved(key)


def s_dedup(make) -> int:
    w = make()
    wl = arr(np.linspace(400, 700, 64))
    w.set_data((1.0, "R", "s"), arr(np.ones(64)), wl)
    w.set_data((2.0, "R", "p"), arr(np.zeros(64)), wl)  # same grid -> same frame
    return w.frame_count


def s_unweave_roundtrip(make) -> Tuple[np.ndarray, np.ndarray]:
    """Tile frames over a master grid, unweave a full curve, weave it back."""
    w = make()
    full = arr(np.arange(0.0, 400.0, 1.0))
    seed = (0.0, "seed", "x")
    for lo, hi in [(0, 100), (100, 250), (250, 400)]:
        b = full[lo:hi]
        w.set_data(seed, arr(np.zeros(hi - lo)), arr(b))
    tgt = (550.0, "R", "s")
    curve = arr(np.sin(full / 30) + 0.1 * full)
    n = w.unweave(tgt, full, curve)
    assert n == 3, f"unweave should touch 3 frames, got {n}"
    return w.get_weaved(tgt)  # must reconstruct `curve` over `full`


def s_unweave_collection(make) -> Tuple[np.ndarray, np.ndarray]:
    w = make()
    full = arr(np.arange(0.0, 300.0, 1.0))
    seed = (0.0, "seed", "x")
    for lo, hi in [(0, 150), (150, 300)]:
        b = full[lo:hi]
        w.set_data(seed, arr(np.zeros(hi - lo)), arr(b))
    batch = {
        (1.0, "R", "s"): arr(np.sin(full / 20)),
        (2.0, "T", "p"): arr(np.cos(full / 25)),
        (3.0, "A", "s"): arr(np.sqrt(full + 1)),
    }
    n = w.unweave_collection(full, batch)
    assert n == len(batch) * 2, f"expected {len(batch)*2} updates, got {n}"
    return w.get_weaved((2.0, "T", "p"))


def s_weaved_collections(make) -> Dict[tuple, Dict[tuple, np.ndarray]]:
    """Two distinct geometries, several keys; group by geometry."""
    w = make()
    g1 = arr(np.linspace(400, 600, 50))
    g2 = arr(np.linspace(600, 900, 80))
    w.set_data((1.0, "R", "s"), arr(np.sin(g1 / 10)), g1)
    w.set_data((2.0, "R", "p"), arr(np.cos(g1 / 10)), g1)
    w.set_data((3.0, "T", "s"), arr(np.tan(g2 / 500)), g2)
    return norm_collections(w.get_weaved_collections())


def s_slowpath_unweave(make) -> Tuple[np.ndarray, np.ndarray]:
    """Frame grid is a strided (non-contiguous) subset -> index-array plan path."""
    w = make()
    full = arr(np.arange(0.0, 200.0, 1.0))
    sub = arr(full[::2])  # 100 pts, exact values, non-contiguous positions
    seed = (0.0, "seed", "x")
    w.set_data(seed, arr(np.zeros(len(sub))), sub)
    tgt = (770.0, "R", "s")
    curve = arr(np.exp(-full / 100))
    n = w.unweave(tgt, full, curve)
    assert n == 1, f"slow-path unweave should touch 1 frame, got {n}"
    wl, data = w.get_weaved(tgt)
    # The reconstructed fragment must equal curve sampled at the even indices.
    assert eq_arrays(wl, full[::2]), "slow-path: wavelength subset wrong"
    assert eq_arrays(data, curve[::2]), "slow-path: data subset wrong"
    return wl, data


def s_generation_invalidation(make) -> int:
    """
    REGRESSION TEST for the realigned cache generation.

    After a full-grid unweave caches a plan, creating a NEW frame whose grid is a
    subset of that same full grid must invalidate the cache so the next unweave
    populates the new frame too. With the original 'never-bump-on-unweave' bug
    this still worked (frame creation bumped), but with a 'cache-by-signature
    without generation' design it would silently miss the new frame.
    """
    w = make()
    full = arr(np.arange(0.0, 100.0, 1.0))
    seedA = (0.0, "A", "x")
    w.set_data(seedA, arr(np.zeros(50)), arr(full[0:50]))      # frame A on [0,50)
    n1 = w.unweave((1.0, "R", "s"), full, arr(np.sin(full)))   # plan = {A} -> 1
    assert n1 == 1, f"first unweave should touch 1 frame, got {n1}"

    gen_before = get_gen(w)
    w.set_data((9.0, "B", "x"), arr(np.zeros(50)), arr(full[50:100]))  # NEW frame B
    gen_after = get_gen(w)
    assert gen_after != gen_before, "creating a frame must change generation"

    n2 = w.unweave((2.0, "T", "p"), full, arr(np.cos(full)))   # plan must now be {A,B}
    assert n2 == 2, f"after adding a frame, unweave must touch 2 frames, got {n2}"
    return n2


def s_generation_no_bump_on_overwrite(make) -> int:
    """Plain data overwrite (existing key, existing frame) must NOT bump gen."""
    w = make()
    wl = arr(np.linspace(0, 10, 16))
    key = (5.0, "R", "s")
    w.set_data(key, arr(np.zeros(16)), wl)
    g0 = get_gen(w)
    w.set_data(key, arr(np.ones(16)), wl)   # overwrite same key/frame
    g1 = get_gen(w)
    assert g1 == g0, f"overwrite must not bump generation ({g0} -> {g1})"
    # Sanity: the value really changed.
    _, data = w.get_weaved(key)
    assert eq_arrays(data, np.ones(16)), "overwrite did not take effect"
    return g1 - g0


# Scenarios that produce a comparable value for Python-vs-Rust cross-checks.
CROSS_SCENARIOS: List[Tuple[str, Callable]] = [
    ("basic_weave", s_basic),
    ("multiframe_weave", s_multiframe_weave),
    ("frame_dedup", s_dedup),
    ("unweave_roundtrip", s_unweave_roundtrip),
    ("unweave_collection", s_unweave_collection),
    ("weaved_collections", s_weaved_collections),
    ("slowpath_unweave", s_slowpath_unweave),
]

# Scenarios that assert intrinsic invariants per engine.
INVARIANT_SCENARIOS: List[Tuple[str, Callable]] = [
    ("generation_invalidation", s_generation_invalidation),
    ("generation_no_bump_on_overwrite", s_generation_no_bump_on_overwrite),
]


def s_errors(engine: str) -> List[str]:
    """Operations that must raise on a given engine."""
    failures = []

    def expect_raise(label, fn):
        try:
            fn()
        except Exception:
            return
        failures.append(label)

    w = make_weaver(engine)
    wl = arr(np.linspace(0, 10, 8))
    w.set_data((1.0, "R", "s"), arr(np.zeros(8)), wl)

    # value/wavelength length mismatch
    expect_raise("shape_mismatch", lambda: w.set_data((2.0, "R", "s"), arr(np.zeros(7)), wl))
    # unsorted grid
    bad = arr([1.0, 3.0, 2.0, 4.0])
    expect_raise("unsorted_grid", lambda: w.set_data((3.0, "R", "s"), arr(np.zeros(4)), bad))
    # missing key on get_weaved
    expect_raise("missing_key", lambda: w.get_weaved((999.0, "Z", "z")))
    # unweave length mismatch
    full = arr(np.arange(0.0, 8.0, 1.0))
    expect_raise("unweave_len_mismatch", lambda: w.unweave((4.0, "R", "s"), full, arr(np.zeros(7))))
    return failures


# --------------------------------------------------------------------------- #
# Correctness driver
# --------------------------------------------------------------------------- #
def run_correctness() -> bool:
    print("=" * 72)
    print("CORRECTNESS")
    print("=" * 72)
    ok = True

    # Cross-engine scenarios
    for name, fn in CROSS_SCENARIOS:
        try:
            py_res = fn(lambda: make_weaver("python"))
            if HAVE_RUST:
                rs_res = fn(lambda: make_weaver("rust"))
                compare_results(name, py_res, rs_res)
                print(f"  [PY==RS] {name:32s} ok")
            else:
                print(f"  [PY    ] {name:32s} ok (rust unavailable)")
        except AssertionError as e:
            ok = False
            print(f"  [FAIL  ] {name:32s} {e}")
        except Exception as e:
            ok = False
            print(f"  [ERROR ] {name:32s} {type(e).__name__}: {e}")

    # Per-engine invariants
    for name, fn in INVARIANT_SCENARIOS:
        for engine in ENGINES:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fn(lambda: make_weaver(engine))
                print(f"  [{engine:6s}] {name:32s} ok")
            except AssertionError as e:
                ok = False
                print(f"  [FAIL  ] {engine}:{name:32s} {e}")
            except Exception as e:
                ok = False
                print(f"  [ERROR ] {engine}:{name:32s} {type(e).__name__}: {e}")

    # Error handling
    for engine in ENGINES:
        miss = s_errors(engine)
        if miss:
            ok = False
            print(f"  [FAIL  ] {engine}: did NOT raise for: {miss}")
        else:
            print(f"  [{engine:6s}] error_handling                  ok")

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


def build_tiled_weaver(engine: str, n_points: int, n_frames: int, cache_size: int):
    """A weaver with `n_frames` contiguous bands tiling a master grid."""
    w = make_weaver(engine, cache_size=cache_size)
    full = arr(np.linspace(0.0, 1000.0, n_points))
    edges = np.linspace(0, n_points, n_frames + 1).astype(int)
    seed = (0.0, "seed", "x")
    for i in range(n_frames):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        b = arr(full[lo:hi])
        w.set_data(seed, arr(np.zeros(hi - lo)), b)
    return w, full


def bench_engine(engine: str, n_points: int, n_frames: int, n_keys: int) -> Dict[str, float]:
    res: Dict[str, float] = {}
    cache = max(8, n_frames * 2)

    # --- set_data: fill n_keys onto a single shared grid -------------------- #
    wl = arr(np.linspace(0.0, 1000.0, n_points))
    payloads = [arr(np.sin(wl / (k + 3))) for k in range(n_keys)]

    def do_set():
        w = make_weaver(engine, cache_size=cache)
        for k in range(n_keys):
            w.set_data((float(k), "R", "s"), payloads[k], wl)
    res["set_data"] = timeit(do_set)

    # --- get_weaved: many keys, each tiled across frames -------------------- #
    wge, full = build_tiled_weaver(engine, n_points, n_frames, cache)
    keys = [(float(k), "R", "s") for k in range(n_keys)]
    for key in keys:
        wge.unweave(key, full, arr(np.cos(full / (hash(key) % 7 + 2))))

    def do_weave():
        for key in keys:
            wge.get_weaved(key)
    res["get_weaved"] = timeit(do_weave)

    # --- unweave (repeated, same grid -> exercises plan cache) -------------- #
    wun, full2 = build_tiled_weaver(engine, n_points, n_frames, cache)
    curve = arr(np.sin(full2 / 13) + 0.01 * full2)

    def do_unweave():
        for k in range(n_keys):
            wun.unweave((float(k), "U", "s"), full2, curve)
    res["unweave_cached"] = timeit(do_unweave)

    # --- unweave_collection (batch, single plan resolve) -------------------- #
    wuc, full3 = build_tiled_weaver(engine, n_points, n_frames, cache)
    batch = {(float(k), "C", "s"): arr(np.cos(full3 / (k + 2))) for k in range(n_keys)}

    def do_unweave_coll():
        wuc.unweave_collection(full3, dict(batch))
    res["unweave_collection"] = timeit(do_unweave_coll, repeats=7)

    return res


def run_benchmark(quick: bool) -> None:
    print("=" * 72)
    print("BENCHMARK   (median of repeated runs; lower is better)")
    print("=" * 72)
    if not HAVE_RUST:
        print(f"  (navette_spectralweave unavailable: {_RUST_IMPORT_ERR}); Python-only timings.\n")

    if quick:
        configs = [(1_000, 4, 8), (10_000, 8, 16)]
    else:
        configs = [
            (1_000, 4, 16),
            (1_000, 32, 128),
            (10_000, 8, 32),
            (10_000, 64, 256),
            (10_000, 256, 64),
            (100_000, 16, 32),
            (100_000, 512, 32),
            (100_000, 32, 512),
            (500_000, 32, 16),
        ]

    ops = ["set_data", "get_weaved", "unweave_cached", "unweave_collection"]

    for n_points, n_frames, n_keys in configs:
        print(f"\n  grid={n_points:,} pts | frames={n_frames} | keys={n_keys}")
        py = bench_engine("python", n_points, n_frames, n_keys)
        rs = bench_engine("rust", n_points, n_frames, n_keys) if HAVE_RUST else None

        hdr = f"    {'operation':22s} {'python (ms)':>14s}"
        if HAVE_RUST:
            hdr += f" {'rust (ms)':>12s} {'speedup':>9s}"
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for op in ops:
            line = f"    {op:22s} {py[op]*1e3:>14.3f}"
            if HAVE_RUST:
                sp = py[op] / rs[op] if rs[op] > 0 else float("inf")
                line += f" {rs[op]*1e3:>12.3f} {sp:>8.2f}x"
            print(line)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-only", action="store_true", help="skip correctness phase")
    ap.add_argument("--corr-only", action="store_true", help="skip benchmark phase")
    ap.add_argument("--quick", action="store_true", help="smaller benchmark sizes")
    args = ap.parse_args()

    print(f"engines available: {', '.join(ENGINES)}")
    print(f"numpy {np.__version__}, python {sys.version.split()[0]}\n")

    ok = True
    if not args.bench_only:
        ok = run_correctness()
    if not args.corr_only:
        run_benchmark(args.quick)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())