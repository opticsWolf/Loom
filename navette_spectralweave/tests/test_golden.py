#!/usr/bin/env python3
"""
Golden / regression tests for the `navette.spectralweave` Rust extension.

These pin *exact numeric outputs* of the OpticalWeaver weave/unweave paths and
the TargetWeaver merit function on small, fully-deterministic scenarios, so any
behavioural drift (algorithm change, refactor, dependency bump) is caught.

Every merit golden is **cross-checked against an independent NumPy computation**
(`test_goldens_are_self_consistent`) so the pinned literals are provably correct,
not merely whatever the code happened to emit when they were captured.

Regenerate the literal GOLDEN_MERIT table after an *intended* behaviour change:

    python tests/test_golden.py --emit-golden

Run the suite:

    python -m pytest tests/test_golden.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

sw = pytest.importorskip("navette.spectralweave")

TAU = 2.0 * np.pi
MISSING = 1e6


def arr(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Deterministic merit scenarios: name -> (build_pair, numpy_reference)
# --------------------------------------------------------------------------- #
# Each builder returns a matched (sim OpticalWeaver, TargetWeaver). The numpy
# reference recomputes the expected merit independently for the same inputs.

def _linear_norm(raw: np.ndarray):
    nf = 1.0 / max(abs(raw.sum() / len(raw)), 1e-12)
    return nf, raw * nf


def _scenario_linear_aligned():
    wl = arr([400, 500, 600, 700, 800])
    raw = arr([1.0, 2.0, 3.0, 4.0, 5.0])
    tol = arr([0.1] * 5)
    sim_vals = raw + 0.05
    tw = sw.TargetWeaver(cache_size=8)
    tw.add_spectral_target(wl, raw, tol, 0.0, "s", "R", "e", "linear")
    sim = sw.OpticalWeaver(cache_size=8)
    sim.set_data((0.0, "s", "R"), sim_vals, wl)

    def ref():
        nf, scaled = _linear_norm(raw)
        d = sim_vals * nf - scaled
        return float(np.sum((d / np.maximum(tol, 1e-12)) ** 2))
    return sim, tw, ref


def _scenario_interp():
    # sim is a straight ramp value == (wl-300)/100 on a coarse grid; target grid
    # sits at the midpoints, so every point is genuinely interpolated.
    sim_wl = arr([400, 500, 600, 700, 800])
    sim_vals = arr([1.0, 2.0, 3.0, 4.0, 5.0])
    t_wl = arr([450, 550, 650])
    raw = arr([2.0, 3.0, 4.0])
    tol = arr([0.1] * 3)
    tw = sw.TargetWeaver(cache_size=8)
    tw.add_spectral_target(t_wl, raw, tol, 1.0, "s", "R", "e", "linear")
    sim = sw.OpticalWeaver(cache_size=8)
    sim.set_data((1.0, "s", "R"), sim_vals, sim_wl)

    def ref():
        interp = np.interp(t_wl, sim_wl, sim_vals)  # [1.5, 2.5, 3.5]
        nf, scaled = _linear_norm(raw)
        d = interp * nf - scaled
        return float(np.sum((d / np.maximum(tol, 1e-12)) ** 2))
    return sim, tw, ref


def _scenario_phase():
    wl = arr([400, 500])
    raw = arr([0.0, 3.0])
    tol = arr([0.5, 0.5])
    sim_vals = arr([7.0, 3.0])
    tw = sw.TargetWeaver(cache_size=8)
    tw.add_spectral_target(wl, raw, tol, 2.0, "s", "R", "e", "phase")
    sim = sw.OpticalWeaver(cache_size=8)
    sim.set_data((2.0, "s", "R"), sim_vals, wl)

    def ref():
        # phase: norm_factor == 1, residual wrapped to (-pi, pi] via round.
        diff = sim_vals - raw
        wrapped = diff - TAU * np.round(diff / TAU)
        return float(np.sum((wrapped / tol) ** 2))
    return sim, tw, ref


def _scenario_above_violated():
    wl = arr([400, 500, 600, 700, 800])
    raw = arr([2.0] * 5)
    tol = arr([0.1] * 5)
    sim_vals = arr([1.9] * 5)  # below an "above" target -> penalised
    tw = sw.TargetWeaver(cache_size=8)
    tw.add_spectral_target(wl, raw, tol, 3.0, "s", "R", "a", "linear")
    sim = sw.OpticalWeaver(cache_size=8)
    sim.set_data((3.0, "s", "R"), sim_vals, wl)

    def ref():
        nf, scaled = _linear_norm(raw)
        d = sim_vals * nf - scaled
        d = np.where(d < 0.0, d, 0.0)
        return float(np.sum((d / np.maximum(tol, 1e-12)) ** 2))
    return sim, tw, ref


def _scenario_below_satisfied():
    wl = arr([400, 500, 600, 700, 800])
    raw = arr([2.0] * 5)
    tol = arr([0.1] * 5)
    sim_vals = arr([1.9] * 5)  # below a "below" target -> satisfied, zero
    tw = sw.TargetWeaver(cache_size=8)
    tw.add_spectral_target(wl, raw, tol, 4.0, "s", "R", "b", "linear")
    sim = sw.OpticalWeaver(cache_size=8)
    sim.set_data((4.0, "s", "R"), sim_vals, wl)
    return sim, tw, lambda: 0.0


def _scenario_missing():
    wl = arr([400, 500, 600])
    tw = sw.TargetWeaver(cache_size=8)
    tw.add_spectral_target(wl, arr([2.0, 2.0, 2.0]), arr([0.1] * 3),
                           5.0, "s", "R", "e", "linear")
    sim = sw.OpticalWeaver(cache_size=8)  # no data for the target key
    return sim, tw, lambda: MISSING


SCENARIOS = {
    "linear_aligned": _scenario_linear_aligned,
    "interp": _scenario_interp,
    "phase": _scenario_phase,
    "above_violated": _scenario_above_violated,
    "below_satisfied": _scenario_below_satisfied,
    "missing": _scenario_missing,
}

# Pinned golden merit values (captured from a correctness-verified build and
# validated against the NumPy reference — see test_goldens_are_self_consistent).
GOLDEN_MERIT = {
    "linear_aligned": 0.1388888888888881,
    "interp": 8.333333333333337,
    "phase": 2.0552932153728967,
    "above_violated": 1.2500000000000022,
    "below_satisfied": 0.0,
    "missing": 1000000.0,
}


# --------------------------------------------------------------------------- #
# Merit golden tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_merit_golden(name):
    sim, tw, _ref = SCENARIOS[name]()
    got = sw.calculate_merit(sim, tw, MISSING)
    expected = GOLDEN_MERIT[name]
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12), \
        f"{name}: merit drifted {got!r} != golden {expected!r}"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_goldens_are_self_consistent(name):
    """The pinned literals must equal an independent NumPy computation."""
    _sim, _tw, ref = SCENARIOS[name]()
    assert ref() == pytest.approx(GOLDEN_MERIT[name], rel=1e-9, abs=1e-12)


def test_merit_deterministic():
    sim, tw, _ = _scenario_interp()
    vals = {sw.calculate_merit(sim, tw, MISSING) for _ in range(8)}
    assert len(vals) == 1, f"merit not bit-deterministic: {vals}"


def test_merit_tolerance_inverse_square():
    """Doubling every tolerance must quarter the merit (residual ∝ 1/tol²)."""
    def merit_with_tol(t):
        wl = arr([400, 500, 600, 700, 800])
        raw = arr([1.0, 2.0, 3.0, 4.0, 5.0])
        tw = sw.TargetWeaver(cache_size=8)
        tw.add_spectral_target(wl, raw, arr([t] * 5), 0.0, "s", "R", "e", "linear")
        sim = sw.OpticalWeaver(cache_size=8)
        sim.set_data((0.0, "s", "R"), raw + 0.05, wl)
        return sw.calculate_merit(sim, tw, MISSING)
    assert merit_with_tol(0.05) / merit_with_tol(0.10) == pytest.approx(4.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# OpticalWeaver golden tests (structural / exact)
# --------------------------------------------------------------------------- #
def test_weave_roundtrip_exact():
    w = sw.OpticalWeaver(cache_size=8)
    wl = arr([400, 500, 600, 700])
    data = arr([1.0, 2.0, 3.0, 4.0])
    w.set_data((550.0, "R", "s"), data, wl)
    out_wl, out_data = w.get_weaved((550.0, "R", "s"))
    np.testing.assert_array_equal(out_wl, wl)
    np.testing.assert_array_equal(out_data, data)


def test_multiframe_weave_exact():
    """Three contiguous sub-bands stitch back to the full curve, in order."""
    w = sw.OpticalWeaver(cache_size=8)
    full = arr(np.arange(400.0, 800.0, 1.0))
    key = (632.8, "T", "p")
    for lo, hi in [(0, 120), (120, 300), (300, 400)]:
        w.set_data(key, arr(np.cos(full[lo:hi] / 50)), arr(full[lo:hi]))
    out_wl, out_data = w.get_weaved(key)
    np.testing.assert_array_equal(out_wl, full)
    np.testing.assert_array_equal(out_data, np.cos(full / 50))


def test_frame_dedup_count():
    w = sw.OpticalWeaver(cache_size=8)
    wl = arr(np.linspace(400, 700, 64))
    w.set_data((1.0, "R", "s"), arr(np.ones(64)), wl)
    w.set_data((2.0, "R", "p"), arr(np.zeros(64)), wl)  # same grid -> one frame
    assert w.frame_count == 1


def test_unweave_roundtrip_exact():
    w = sw.OpticalWeaver(cache_size=8)
    full = arr(np.arange(0.0, 300.0, 1.0))
    seed = (0.0, "seed", "x")
    for lo, hi in [(0, 100), (100, 250), (250, 300)]:
        w.set_data(seed, arr(np.zeros(hi - lo)), arr(full[lo:hi]))
    tgt = (550.0, "R", "s")
    curve = arr(np.sin(full / 30) + 0.1 * full)
    n = w.unweave(tgt, full, curve)
    assert n == 3
    out_wl, out_data = w.get_weaved(tgt)
    np.testing.assert_array_equal(out_wl, full)
    np.testing.assert_array_equal(out_data, curve)


def test_generation_counter_semantics():
    w = sw.OpticalWeaver(cache_size=8)
    assert w.generation == 0
    wl = arr(np.linspace(0, 10, 16))
    key = (5.0, "R", "s")
    w.set_data(key, arr(np.zeros(16)), wl)
    g1 = w.generation
    assert g1 == 1                      # new frame bumps generation
    w.set_data(key, arr(np.ones(16)), wl)
    assert w.generation == g1           # overwrite same key/frame: no bump
    w.set_data((6.0, "R", "s"), arr(np.zeros(8)), arr(np.linspace(0, 5, 8)))
    assert w.generation == g1 + 1       # new grid -> new frame -> bump


# --------------------------------------------------------------------------- #
# Error-path goldens
# --------------------------------------------------------------------------- #
def test_shape_mismatch_raises():
    w = sw.OpticalWeaver(cache_size=8)
    wl = arr(np.linspace(0, 10, 8))
    with pytest.raises(Exception):
        w.set_data((1.0, "R", "s"), arr(np.zeros(7)), wl)


def test_unsorted_grid_raises():
    w = sw.OpticalWeaver(cache_size=8)
    with pytest.raises(Exception):
        w.set_data((1.0, "R", "s"), arr(np.zeros(4)), arr([1.0, 3.0, 2.0, 4.0]))


def test_missing_key_raises():
    w = sw.OpticalWeaver(cache_size=8)
    with pytest.raises(Exception):
        w.get_weaved((999.0, "Z", "z"))


# --------------------------------------------------------------------------- #
# Regeneration helper
# --------------------------------------------------------------------------- #
def _emit_golden():
    print("GOLDEN_MERIT = {")
    for name, build in SCENARIOS.items():
        sim, tw, ref = build()
        rust = sw.calculate_merit(sim, tw, MISSING)
        ref_v = ref()
        tag = "OK" if abs(rust - ref_v) <= 1e-9 + 1e-9 * abs(ref_v) else "MISMATCH!"
        print(f"    {name!r}: {rust!r},   # numpy ref={ref_v!r} [{tag}]")
    print("}")


if __name__ == "__main__":
    import sys
    if "--emit-golden" in sys.argv:
        _emit_golden()
    else:
        print("run:  python -m pytest tests/test_golden.py -q")
        print("or :  python tests/test_golden.py --emit-golden")
