#!/usr/bin/env python3
"""Solver speed: old-request workloads must not regress with backside outputs.

Measures core_engine throughput on a fixed workload (40 wav x 3 angles,
5-layer stack) for three masks:
  A: broad legacy mask (no backside bits) -> the regression case
  B: BACKSIDE only                            -> marginal cost of new outputs
  C: A + BACKSIDE                             -> full load

Usage:  python validation/benches/smatrix/bench_backside_speed.py [--out FILE]
Writes JSON {mask: {median_ms, min_ms}} to stdout (and FILE if given).
"""

import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from navette.smatrix.smatrix import ScatterMatrix, Request

WAVLS = np.linspace(400.0, 800.0, 40)
ANGLES = [0.0, 30.0, 60.0]
rng = np.random.default_rng(7)
NL = 5
IDX = (rng.uniform(1.3, 2.6, (NL, len(WAVLS)))
       + 1j * rng.uniform(0.0, 0.2, (NL, len(WAVLS))))
IDX[0] = 1.0 + 0j
IDX[-1] = 1.52 + 0j
THICK = np.array([0.0, 40.0, 80.0, 30.0, 0.0])

MASK_A = (Request.PHOTOMETRY | Request.ELLIPSOMETRY | Request.ABSORPTION |
          Request.RS_C | Request.RP_C | Request.TS_C | Request.TP_C |
          Request.DISP_R_S | Request.DISP_T_S)

WARMUP = 5
REPS = 60


def bench(mask):
    st = ScatterMatrix(IDX, THICK, wavelengths=WAVLS, angles=ANGLES)
    for _ in range(WARMUP):
        st.compute(mask)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        st.compute(mask)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "min_ms": float(np.min(ts)),
            "reps": REPS}


def main():
    backside = getattr(Request, "BACKSIDE", 0)
    masks = {"A_legacy": MASK_A}
    if backside:
        masks["B_backside_only"] = Request(backside)
        masks["C_full"] = MASK_A | Request(backside)
    res = {name: bench(m) for name, m in masks.items()}
    text = json.dumps(res, indent=1)
    print(text)
    if len(sys.argv) == 3 and sys.argv[1] == "--out":
        with open(sys.argv[2], "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
