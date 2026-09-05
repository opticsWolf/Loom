#!/usr/bin/env python3
"""Grid-assertion cost: solve_structure with vs without provider grids.

Compares full-solve wall time on a fixed workload (100 wav, 7-layer
dispersive stack) for:
  A: grid attached  -> length assert + byte-compare (new behavior)
  B: gridless       -> length assert + getattr miss (old code path)
  C: assert helper alone (isolated overhead of _assert_provider_grids)

Usage:  python validation/benches/structure/bench_grid_assert.py [--out FILE]
Writes JSON {case: {median_ms, min_ms}} to stdout (and FILE if given).
"""

import json
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "src")
from navette.structure import Navette_Structure, solve_structure
from navette.structure.models import Layer
from navette.structure.materials import DictMaterialProvider

NWL = int(sys.argv[sys.argv.index("--nwl") + 1]) if "--nwl" in sys.argv else 100
WAVLS = np.linspace(400.0, 800.0, NWL)
rng = np.random.default_rng(3)
ARRAYS = {
  "glass": np.full(NWL, 1.52 + 0j),
  "TiO2": rng.uniform(2.3, 2.6, NWL) + 1j * rng.uniform(0.0, 0.01, NWL),
  "SiO2": np.full(NWL, 1.46 + 0j),
  "Ag": rng.uniform(0.1, 0.3, NWL) + 1j * rng.uniform(2.0, 4.0, NWL),
}
LAYERS = [Layer(0.0, "glass"), Layer(45.0, "TiO2"), Layer(90.0, "SiO2"),
          Layer(12.0, "Ag"), Layer(60.0, "TiO2"), Layer(90.0, "SiO2"),
          Layer(0.0, "glass")]


def make(with_grid):
  # Identical arrays and stack; ONLY the grid attachment differs, so A-B
  # isolates the assertion cost (serve-time length checks + bridge compare).
  prov = DictMaterialProvider(dict(ARRAYS),
                              wavelength=WAVLS.copy() if with_grid else None)
  return Navette_Structure([Layer(l.thickness, l.material) for l in LAYERS], {}, prov)


def bench(fn, n=25):
  fn()  # warmup
  ts = []
  for _ in range(n):
    t0 = time.perf_counter()
    fn()
    ts.append((time.perf_counter() - t0) * 1e3)
  return {"median_ms": float(np.median(ts)), "min_ms": float(np.min(ts))}


with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  a = bench(lambda: solve_structure(make(True), WAVLS, 0.0))
  b = bench(lambda: solve_structure(make(False), WAVLS, 0.0))

from navette.structure import _assert_provider_grids as chk
st = make(True)
wl = np.ascontiguousarray(WAVLS)
chk(st, wl)
c_ts = []
for _ in range(2000):
  t0 = time.perf_counter()
  chk(st, wl)
  c_ts.append((time.perf_counter() - t0) * 1e6)
c = {"median_us": float(np.median(c_ts)), "min_us": float(np.min(c_ts))}

out = {"A_grid_attached_ms": a, "B_gridless_ms": b, "C_assert_only_us": c}
print(json.dumps(out, indent=2))
if "--out" in sys.argv:
  with open(sys.argv[sys.argv.index("--out") + 1], "w") as fh:
    json.dump(out, fh, indent=2)
