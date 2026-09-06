#!/usr/bin/env python3
"""Synthesis bridge: TargetCollection -> MeritSpec converter validation.

  1. Merit agreement: MeritSpec.merit == calculate_merit on identical data
     (shared ingestion — must agree for R/T/A/range/angular/back demands).
  2. Fold identity: folded quadratics reproduce MeritSpec residuals
     (plus the documented +1 per violated `c` point).
  3. Phase demands: hand-checked wrapped residuals + phi-bucket fold.
  4. Error paths: unknown labels, phase-on-absorption.

Run explicitly:  python validation/parity/synthesis/test_merit_bridge.py
"""

import numpy as np

from navette.spectralweave.optical import SimulationWeaver, OpticalFragment
from navette.spectralweave.target import (
    TargetCollection, SpectralTarget, AngularTarget, calculate_merit,
)
from navette.synthesis import build_merit_spec, sim_curves_from_arrays, build_needle_targets

WL = np.array([400.0, 500.0, 600.0])
ANGLES = np.array([0.0, 5.0, 10.0])
OK = True


def check(name, cond, detail=""):
    global OK
    print(f"  {name}: {'OK' if cond else 'FAIL'} {detail}")
    OK = bool(cond) and OK


R_VALS = np.array([0.52, 0.61, 0.69])
B_VALS = np.array([0.55, 0.62, 0.70])
T_VALS = np.array([0.28, 0.31, 0.33])
# A must equal 1 - R - T where derived (angle 5 rows mirror angle 0):
# the weaver path reads supplied A fragments, the spec path derives them.
A_VALS = 1.0 - R_VALS - T_VALS
RB_VALS = np.array([0.41, 0.39, 0.42])


def r_target_collection():
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.array([0.5, 0.6, 0.7]), np.full(3, 0.05),
                          0.0, "s", "R", kind="e"))
    tc.add(SpectralTarget(WL, np.array([0.3, 0.3, 0.3]), np.full(3, 0.05),
                          0.0, "s", "T", kind="a"))
    tc.add(SpectralTarget(WL, np.array([0.5, 0.5, 0.5]), np.full(3, 0.05),
                          0.0, "s", "R", kind="r", band=0.02))
    tc.add(SpectralTarget(WL, np.array([0.2, 0.2, 0.2]), np.full(3, 0.05),
                          5.0, "s", "A", kind="e"))
    tc.add(SpectralTarget(WL, np.array([0.4, 0.4, 0.4]), np.full(3, 0.05),
                          0.0, "s", "RB", kind="e"))
    # Angular demand shares the (0.0, R) key with the spectral targets, so
    # its sim fragment must carry the same curve (one value per key).
    tc.add(AngularTarget(500.0, np.array([0.0, 10.0]), np.array([0.5, 0.6]),
                         np.array([0.05, 0.05]), "s", "R", kind="e"))
    return tc


def matched_sim():
    """One sim seen both ways: weaver fragments + solver-grid rows."""
    sw = SimulationWeaver()
    sw.add_fragment(OpticalFragment(WL, R_VALS, 0.0, "s", "R"))
    sw.add_fragment(OpticalFragment(WL, B_VALS, 10.0, "s", "R"))
    sw.add_fragment(OpticalFragment(WL, T_VALS, 0.0, "s", "T"))
    sw.add_fragment(OpticalFragment(WL, A_VALS, 5.0, "s", "A"))
    sw.add_fragment(OpticalFragment(WL, RB_VALS, 0.0, "s", "RB"))
    # Note: no "As" rows — absorptance derives from the Rs/Ts companions.
    rows = {
        "Rs": np.array([R_VALS, R_VALS, B_VALS]),
        "Ts": np.array([T_VALS, T_VALS, T_VALS]),
        "RBs": np.array([RB_VALS, RB_VALS, RB_VALS]),
    }
    sim = sim_curves_from_arrays(ANGLES, WL, rows)
    return sw, sim, rows


def folded_total(folded, rows, angles=ANGLES):
    """Sum w*(t-s)^2 over buckets against the known sim rows."""
    # absorptance derives from companions, like the core does
    full = dict(rows)
    if "Rs" in rows and "Ts" in rows:
        full["As"] = 1.0 - rows["Rs"] - rows["Ts"]
    if "RBs" in rows and "TBs" in rows:
        full["ABs"] = 1.0 - rows["RBs"] - rows["TBs"]
    total = 0.0
    na = len(np.atleast_1d(angles))
    for bucket, code in [("r", "Rs"), ("t", "Ts"), ("a", "As"),
                         ("rb", "RBs"), ("tb", "TBs"), ("ab", "ABs")]:
        w = np.asarray(folded[bucket]["weights"]).reshape(na, len(WL))
        t = np.asarray(folded[bucket]["targets"]).reshape(na, len(WL))
        if code in full:
            total += float(np.sum(w * (t - full[code]) ** 2))
        else:
            assert float(np.sum(w)) == 0.0, f"unexpected {bucket} weight"
    return total


def test_agreement():
    print("--- MeritSpec.merit == calculate_merit ---")
    tc = r_target_collection()
    sw, sim, rows = matched_sim()
    tw = tc.build_weaver()
    spec = build_merit_spec(tc)
    m1 = calculate_merit(sw.backend, tw)
    m2 = spec.merit(sim, 1e6)
    check("merit agreement", abs(m1 - m2) < 1e-9 * max(1.0, abs(m1)),
          f"weaver={m1:.6f} spec={m2:.6f}")


def test_fold_identity():
    print("--- fold identity (non-overlapping demands: exact) ---")
    # NOTE: overlapping demands on one solver point fold exactly for
    # GRADIENTS but drop the completing-the-square constant from VALUES
    # (same class of deliberate loss as the c-level +1). Exact value
    # identity only holds without overlap, as constructed here.
    fa = np.array([0.0, 5.0, 10.0, 15.0])
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.array([0.5, 0.6, 0.7]), np.full(3, 0.05),
                          0.0, "s", "R", kind="e"))
    tc.add(SpectralTarget(WL, np.array([0.5, 0.5, 0.5]), np.full(3, 0.05),
                          5.0, "s", "R", kind="r", band=0.02))
    tc.add(AngularTarget(500.0, np.array([10.0]), np.array([0.55]),
                         np.array([0.05]), "s", "R", kind="e"))
    tc.add(SpectralTarget(WL, np.array([0.3, 0.3, 0.3]), np.full(3, 0.05),
                          15.0, "s", "T", kind="a"))
    spec = build_merit_spec(tc)
    r0 = np.array([0.52, 0.61, 0.69])
    r1 = np.array([0.505, 0.60, 0.40])  # pt0 in-band, pt2 violated-below
    rows = {"Rs": np.array([r0, r0, r0, r1]),
            "Ts": np.array([T_VALS, T_VALS, T_VALS, T_VALS])}
    sim = sim_curves_from_arrays(fa, WL, rows)
    folded = build_needle_targets(spec, fa, WL, sim)
    fsum = folded_total(folded, rows, angles=fa)
    m = spec.merit(sim, 1e6)
    check("folded == merit (no c-outside)", abs(fsum - m) < 1e-9 * max(1.0, m),
          f"folded={fsum:.6f} merit={m:.6f}")
    # All-violated soft-box: folded + n_points == merit.
    tc2 = TargetCollection()
    tc2.add(SpectralTarget(WL, np.full(3, 0.5), np.full(3, 0.05),
                           0.0, "s", "R", kind="c", band=0.01))
    spec2 = build_merit_spec(tc2)
    rows2 = {"Rs": np.array([np.full(3, 0.8)])}
    sim2 = sim_curves_from_arrays(np.array([0.0]), WL, rows2)
    f2 = build_needle_targets(spec2, np.array([0.0]), WL, sim2)
    fsum2 = folded_total(f2, rows2, angles=np.array([0.0]))
    m2 = spec2.merit(sim2, 1e6)
    check("c-outside +1 accounting", abs(fsum2 + 3 - m2) < 1e-9 * m2,
          f"folded={fsum2:.6f} merit={m2:.6f}")


def test_phase():
    print("--- phase demands ---")
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.array([0.5, 0.5, 1.0]), np.full(3, 0.1),
                          0.0, "s", "R", kind="e", phase=True))
    spec = build_merit_spec(tc)
    cplx = np.exp(1j * np.array([[0.52, 0.48, 1.05]]))
    sim = sim_curves_from_arrays(np.array([0.0]), WL, {}, {"Rs": cplx})
    m = spec.merit(sim, 1e6)
    # converter unscales phase demands to raw radians (nf == 1 invariant)
    d = np.array([0.52, 0.48, 1.05]) - np.array([0.5, 0.5, 1.0])
    d = d - 2 * np.pi * np.round(d / (2 * np.pi))
    expect = float(np.sum((d / 0.1) ** 2))
    check("wrapped phase merit", abs(m - expect) < 1e-9 * max(1.0, expect),
          f"got={m:.6f} expect={expect:.6f}")
    folded = build_needle_targets(spec, np.array([0.0]), WL, sim)
    check("phi0 bucket active, others empty",
          abs(float(folded["phi0"]["weights"][0]) - (1.0 / 0.01)) < 1e-9
          and all(float(np.sum(folded[f"phi{i}"]["weights"])) == 0.0 for i in [1, 2, 3]))


def test_errors():
    print("--- error paths ---")
    tc = TargetCollection()
    tc.add(SpectralTarget(WL, np.full(3, 0.5), np.full(3, 0.05),
                          0.0, "s", "MYCHAN", kind="e"))
    try:
        build_merit_spec(tc)
        check("unknown label rejected", False)
    except ValueError:
        check("unknown label rejected", True)
    tc2 = TargetCollection()
    tc2.add(SpectralTarget(WL, np.full(3, 0.5), np.full(3, 0.05),
                           0.0, "s", "A", kind="e", phase=True))
    try:
        build_merit_spec(tc2)
        check("phase-on-absorption rejected", False)
    except ValueError:
        check("phase-on-absorption rejected", True)


def main():
    test_agreement()
    test_fold_identity()
    test_phase()
    test_errors()
    print("ALL OK" if OK else "MISMATCH")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
