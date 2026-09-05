#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark: per-cycle needle-target re-fold cost vs static fold.

Compares the same design problem with `refold_per_cycle=True/False`
(wall time, merit trajectory, insertion counts), plus a cost breakdown
of one re-fold (simulate + fold) against one LM optimization, so the
overhead can be read both relatively and absolutely.

Run explicitly:  python validation/benches/synthesis/bench_refold.py
"""

import sys
import time

import numpy as np

from navette.materials import MaterialSpec
from navette.spectralweave.target import SpectralTarget, TargetCollection
from navette.synthesis import build_merit_spec, build_needle_targets
from navette.synthesis.pipeline import run_needle, stack_from_layers
from navette._smatrix import NeedleCycleConfig, PipelineConfig, SmatrixContext

WL = np.array([900., 1000., 1100.])
ANGS = np.array([0.0])

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


def problem():
    # Same satisfiable problem as test_pipeline.py (insertions occur).
    layers = [(MaterialSpec("Konstant", dict(n=np.sqrt(1.52))), 120.0)]
    contrast = {"film0": MaterialSpec("Konstant", dict(n=2.35))}
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.zeros(3), np.full(3, 0.01), 0.0, "s", "R",
                          kind="e", weight=2.0))
    tc.add(SpectralTarget(WL, np.zeros(3), np.full(3, 0.05), 0.0, "s", "R",
                          kind="a", integral=True, weight=3.0))
    return layers, contrast, tc


def timed_run(refold, repeats=3):
    layers, contrast, tc = problem()
    cfg = PipelineConfig(max_macro_cycles=2, needles_per_cycle=2,
                         enable_cleanup=False, enable_inflate=False,
                         stagnation_window=100)
    nc = NeedleCycleConfig(scan_step_nm=5.0, refold_per_cycle=refold)
    best = None
    dt_best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = run_needle(layers, tc, ANGS, WL, contrast, pipeline_config=cfg,
                         needle_config=nc, names=["L"])
        dt = time.perf_counter() - t0
        if dt < dt_best:
            dt_best, best = dt, res
    return dt_best, best


def main():
    print("--- refold on vs off (best of 3, 2 macro-cycles) ---")
    dt_on, res_on = timed_run(True)
    dt_off, res_off = timed_run(False)
    n_ins = lambda r: sum(1 for p in r["phases"] for c in p["needle_results"]
                          if c["insertion"] is not None)
    print(f"  refold ON : {dt_on * 1e3:8.1f} ms  mf={res_on['final_mf']:.4f}  "
          f"insertions={n_ins(res_on)}  layers={res_on['final_layer_count']}")
    print(f"  refold OFF: {dt_off * 1e3:8.1f} ms  mf={res_off['final_mf']:.4f}  "
          f"insertions={n_ins(res_off)}  layers={res_off['final_layer_count']}")
    overhead = (dt_on - dt_off) / dt_off * 100.0 if dt_off > 0 else float("nan")
    print(f"  overhead: {overhead:+.1f}% of total run time")
    # Both modes must actually exercise needle cycles and terminate sanely;
    # merit trajectories may differ (different insertion decisions).
    check("both_terminate",
          res_on["termination"] == res_off["termination"] == "MAX_ITERATIONS_REACHED",
          f'{res_on["termination"]} / {res_off["termination"]}')
    check("cycles_ran",
          n_ins(res_on) + n_ins(res_off) > 0,
          "no insertions in either mode — bench is degenerate")

    print("--- cost breakdown (one re-fold vs one LM optimize) ---")
    layers, contrast, tc = problem()
    spec = build_merit_spec(tc)
    stack, _ = stack_from_layers(layers, WL, {}, names=["L"])
    ctx = SmatrixContext(spec, ANGS, WL)
    t0 = time.perf_counter()
    sim = ctx.simulate(stack)
    dt_sim = time.perf_counter() - t0
    t0 = time.perf_counter()
    build_needle_targets(spec, ANGS, WL, sim)
    dt_fold = time.perf_counter() - t0
    # fresh stack for the optimize timing (LM cost depends on start point).
    stack2, _ = stack_from_layers(layers, WL, {}, names=["L"])
    t0 = time.perf_counter()
    ctx.optimize_thicknesses(stack2)
    dt_opt = time.perf_counter() - t0
    print(f"  simulate:      {dt_sim * 1e3:8.3f} ms")
    print(f"  fold(w/ sim):  {dt_fold * 1e3:8.3f} ms")
    print(f"  one re-fold:   {(dt_sim + dt_fold) * 1e3:8.3f} ms")
    print(f"  one LM opt:    {dt_opt * 1e3:8.3f} ms")
    print(f"  re-fold / LM:  {(dt_sim + dt_fold) / dt_opt * 100:.1f}%")
    check("refold_cheaper_than_opt", dt_sim + dt_fold < dt_opt)

    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
