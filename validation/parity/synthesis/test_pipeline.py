#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end validation for the bound needle pipeline.

Covers, through the public Python surface only:
1. Stack/context primitives (DesignStack round-trip, SmatrixContext
   simulate/merit/optimize vs the ScatterMatrix oracle).
2. A real 2-cycle run on an AR problem with a MIXED target set
   (R exact + weighted + angular + integral-T) — merit must drop,
   film count must grow, termination sane.
3. Callback invocation + abort path (raising → USER_ABORT).
4. Config validation (bad clamps rejected at construction).

Run explicitly:  python validation/parity/synthesis/test_pipeline.py
"""

import sys

import numpy as np

from navette._smatrix import (
    DesignStack,
    LayerSpec,
    NeedleCycleConfig,
    NeedlePipeline,
    PipelineConfig,
)
from navette.materials import MaterialSpec, evaluate
from navette.smatrix.smatrix import Request, ScatterMatrix
from navette.spectralweave.target import (
    AngularTarget,
    SpectralTarget,
    TargetCollection,
)
from navette.synthesis import build_merit_spec, sim_curves_from_arrays
from navette.synthesis.pipeline import run_needle, stack_from_layers

FAILURES = []
WL = np.array([900., 1000., 1100.])
ANGS = np.array([0.0])


def check(name, ok, detail=""):
    print(f"  {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


def nk_of(model, params):
    return np.ascontiguousarray(evaluate(MaterialSpec(model, params), WL))


N_L = nk_of("Konstant", dict(n=np.sqrt(1.52)))
N_H = nk_of("Konstant", dict(n=2.35))
N_AIR = np.full(3, 1.0 + 0j)
N_SUB = np.full(3, 1.52 + 0j)


def air():
    return LayerSpec("air", N_AIR, 0.0, optimize=False, needle=False)


def sub():
    return LayerSpec("sub", N_SUB, 0.0, optimize=False, needle=False)


def test_primitives():
    print("--- stack/context primitives ---")
    films = [LayerSpec("L", N_L, 200.0), LayerSpec("H", N_H, 50.0)]
    st = DesignStack(air(), sub(), films)
    check("film_count", st.film_count() == 2, f"{st}")
    check("total_thickness", abs(st.total_thickness() - 250.0) < 1e-12)
    back = st.films()
    check("films_roundtrip",
          [f["material"] for f in back] == ["L", "H"]
          and abs(back[0]["thickness"] - 200.0) < 1e-12
          and bool(np.allclose(np.asarray(back[1]["nk"]), N_H)))
    d = st.to_dict()
    check("to_dict", d["num_wavs"] == 3 and d["ambient"]["material"] == "air")
    st.set_thickness(0, 210.0)
    check("set_thickness", abs(st.films()[0]["thickness"] - 210.0) < 1e-12)
    st.insert_needle_seed(0, 100.0, LayerSpec("H", N_H, 5.0))
    check("insert_splits", st.film_count() == 4
          and [f["material"] for f in st.films()] == ["L", "H", "L", "H"])
    n = st.merge_adjacent()
    check("merge", n == 0 and st.film_count() == 4)  # alternating: nothing merges
    removed = st.remove_film(1)
    check("remove", removed.material == "H" and st.film_count() == 3)
    r, c = st.clamp_all(2.0, 1000.0)
    check("clamp", (r, c) == (0, 0))
    try:
        st.clamp_all(5.0, 5.0)
        check("clamp_rejects", False)
    except ValueError:
        check("clamp_rejects", True)
    try:
        LayerSpec("x", np.array([], dtype=np.complex128), 1.0)
        check("empty_nk_rejects", False)
    except ValueError:
        check("empty_nk_rejects", True)

    # context vs ScatterMatrix oracle.
    from navette._smatrix import MeritSpec, SmatrixContext
    spec = MeritSpec()
    k = spec.add_key(0.0, "Rs")
    spec.add_target(k, WL.copy(), np.zeros(3), np.full(3, 0.01),
                    "e", "linear", 1.0)
    ctx = SmatrixContext(spec, ANGS, WL)
    films = [LayerSpec("L", N_L, 150.0)]
    st = DesignStack(air(), sub(), films)
    m0 = ctx.evaluate_merit(st)
    sim = ctx.simulate(st)
    check("simulate_merit_agree",
          abs(m0 - spec.merit(sim, 1e6)) < 1e-12, f"mf={m0:.4f}")
    sm = ScatterMatrix(np.array([N_AIR, N_L, N_SUB]), np.array([0., 150., 0.]),
                       wavelengths=WL, angles=[0.0])
    oracle = sim_curves_from_arrays(ANGS, WL, {"Rs": np.atleast_2d(sm.compute(Request.RS)["Rs"])})
    check("context_matches_solver",
          abs(m0 - spec.merit(oracle, 1e6)) < 1e-9 * max(1.0, m0), f"mf={m0:.4f}")
    m1 = ctx.optimize_thicknesses(st)
    check("optimize_improves", m1 <= m0 + 1e-9, f"{m0:.4f} -> {m1:.4f}")


def mixed_collection():
    # Satisfiable mix exercising every knob: weighted exact + angular with
    # count-norm + integral bound (mean >= 0 always holds → silent masking
    # path, nonzero weight still applies).
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.array([0.0, 0.0, 0.0]), np.full(3, 0.01),
                          0.0, "s", "R", kind="e", weight=2.0))
    tc.add(SpectralTarget(WL, np.array([0.0, 0.0, 0.0]), np.full(3, 0.05),
                          0.0, "s", "R", kind="a", integral=True,
                          weight=3.0))
    tc.add(AngularTarget(1000.0, np.array([0.0]), np.array([0.0]),
                         np.array([0.02]), "p", "R", kind="e",
                         normalize_count=True))
    return tc


def test_run():
    print("--- full pipeline run (mixed targets) ---")
    layers = [(MaterialSpec("Konstant", dict(n=np.sqrt(1.52))), 120.0)]
    contrast = {"film0": MaterialSpec("Konstant", dict(n=2.35))}
    tc = mixed_collection()
    seen = []

    def cb(cycle, phase):
        seen.append((cycle, phase["mf_end"]))

    cfg = PipelineConfig(max_macro_cycles=2, needles_per_cycle=2,
                         enable_cleanup=True, enable_inflate=False,
                         stagnation_window=100)
    # initial merit of the unoptimized start stack (for the improvement check).
    from navette._smatrix import SmatrixContext
    stack0, _ = stack_from_layers(layers, WL, {}, names=["L"])
    m_init = SmatrixContext(build_merit_spec(tc), ANGS, WL).evaluate_merit(stack0)
    res = run_needle(layers, tc, ANGS, WL, contrast, pipeline_config=cfg,
                     callback=cb, names=["L"])
    check("termination", res["termination"] in
          ("MAX_ITERATIONS_REACHED", "STAGNATION_PLATEAU", "MERIT_TARGET_REACHED"),
          res["termination"])
    check("mf_dropped", res["final_mf"] < m_init * 0.5,
          f'{m_init:.4f} -> {res["final_mf"]:.4f}')
    check("grew", res["final_layer_count"] > 1,
          f'layers={res["final_layer_count"]}')
    check("callback", len(seen) == len(res["phases"]) and seen[0][0] == 1,
          f"{len(seen)} calls")
    check("needle_history",
          all(len(p["needle_results"]) >= 1 for p in res["phases"]))
    films = res["stack"].films()
    check("final_stack", len(films) == res["final_layer_count"]
          and all(f["thickness"] >= 0 for f in films))
    # integral + weighted + angular demands all present in the merit the
    # pipeline optimized: re-evaluate final stack, must equal final_mf.
    spec = build_merit_spec(tc)
    angs = np.array([0.0])
    idx_rows = {"air": N_AIR}
    mats = [N_AIR] + [np.asarray(f["nk"]) for f in films] + [N_SUB]
    th = [0.0] + [float(f["thickness"]) for f in films] + [0.0]
    sm = ScatterMatrix(np.array(mats), np.array(th),
                       wavelengths=WL, angles=[0.0])
    o = sm.compute(Request.RS | Request.RP)
    sim = sim_curves_from_arrays(angs, WL, {"Rs": np.atleast_2d(o["Rs"]),
                                            "Rp": np.atleast_2d(o["Rp"])})
    check("final_mf_reproducible",
          abs(spec.merit(sim, 1e6) - res["final_mf"]) < 1e-6 * max(1.0, res["final_mf"]),
          f'{spec.merit(sim, 1e6):.4f} vs {res["final_mf"]:.4f}')


def test_abort_and_validation():
    print("--- callback abort + config validation ---")
    layers = [(MaterialSpec("Konstant", dict(n=np.sqrt(1.52))), 120.0)]
    contrast = {"film0": MaterialSpec("Konstant", dict(n=2.35))}
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.zeros(3), np.full(3, 0.01), 0.0, "s", "R"))
    cfg = PipelineConfig(max_macro_cycles=3, needles_per_cycle=1,
                         stagnation_window=100)

    def stop_at_2(cycle, phase):
        if cycle >= 2:
            raise RuntimeError("stop now")

    res = run_needle(layers, tc, ANGS, WL, contrast, pipeline_config=cfg,
                     callback=stop_at_2, names=["L"])
    check("user_abort", res["termination"] == "USER_ABORT"
          and len(res["phases"]) == 2, res["termination"])
    try:
        PipelineConfig(clamp_min_nm=5.0, clamp_max_nm=5.0)
        # constructed fine (validation happens at pipeline construction);
        # the error surfaces on run_needle:
        run_needle(layers, tc, ANGS, WL, contrast,
                   pipeline_config=PipelineConfig(clamp_min_nm=5.0, clamp_max_nm=5.0),
                   names=["L"])
        check("bad_clamps_rejected", False)
    except ValueError:
        check("bad_clamps_rejected", True)
    # grid mismatch rejected.
    try:
        stack, cmap = stack_from_layers(
            [(MaterialSpec("Konstant", dict(n=1.5)), 10.0)], WL, {}, names=["X"])
        NeedlePipeline(stack, build_merit_spec(tc), ANGS,
                       np.array([500., 600.]), cmap)
        check("grid_mismatch_rejected", False)
    except ValueError:
        check("grid_mismatch_rejected", True)


def test_cycle_config_passthrough():
    print("--- needle cycle config ---")
    # needles_per_cycle is the per-cycle insertion budget (it overrides
    # NeedleCycleConfig.max_needles inside the pipeline).
    nc = NeedleCycleConfig(scan_step_nm=5.0)
    layers = [(MaterialSpec("Konstant", dict(n=np.sqrt(1.52))), 120.0)]
    contrast = {"film0": MaterialSpec("Konstant", dict(n=2.35))}
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.zeros(3), np.full(3, 0.01), 0.0, "s", "R"))
    cfg = PipelineConfig(max_macro_cycles=1, needles_per_cycle=1,
                         stagnation_window=100)
    res = run_needle(layers, tc, ANGS, WL, contrast, pipeline_config=cfg,
                     needle_config=nc, names=["L"])
    n = len(res["phases"][0]["needle_results"])
    check("needles_per_cycle_1", n <= 1, f"{n} cycles")


if __name__ == "__main__":
    test_primitives()
    test_run()
    test_abort_and_validation()
    test_cycle_config_passthrough()
    print("ALL OK" if not FAILURES else f"MISMATCH {FAILURES}")
    sys.exit(1 if FAILURES else 0)
