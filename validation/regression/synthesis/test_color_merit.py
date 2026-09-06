# -*- coding: utf-8 -*-
"""Color demands end to end: bound ``compile_merit_spec`` -> merit/residuals
vs independent oracles (Rust ``color_merit`` + ``merit`` twins, plan R3)."""
import json

import numpy as np
import pytest

from navette._smatrix import compile_merit_spec
from navette.synthesis import sim_curves_from_arrays
import navette._color as C
from navette.data import load_cie_table

CMF = load_cie_table("CIE", "cmf", "CIE_xyz_1931_2deg.json")
D65 = load_cie_table("CIE", "sds", "CIE_std_illum_D65_S_D65.json")
LAM = np.asarray(CMF["lambda"], dtype=np.float64)
CX = np.asarray(CMF["x_bar(lambda)"], dtype=np.float64)
CY = np.asarray(CMF["y_bar(lambda)"], dtype=np.float64)
CZ = np.asarray(CMF["z_bar(lambda)"], dtype=np.float64)
DLAM = np.asarray(D65["lambda"], dtype=np.float64)
E65 = np.asarray(D65["S_D65(lambda)"], dtype=np.float64)


def window(a, b):
  """Interior slice [a, b] of the 1 nm tables (node-exact resample)."""
  m = (LAM >= a) & (LAM <= b)
  me = (DLAM >= a) & (DLAM <= b)
  return (LAM[m], CX[m], CY[m], CZ[m], DLAM[me], E65[me])


def demand(curve="Rs", angle=0.0, quantity="Lab", reference=(60.0, 10.0, -20.0),
           distance="DeltaE2000", weight=1.0, tables=None, names=False):
  if names:
    illum, cmf = "D65", "1931_2deg"
  else:
    wl, x, y, z, el, ev = tables
    illum = {"wavelengths": el.tolist(), "values": ev.tolist()}
    cmf = {"wavelengths": wl.tolist(),
           "xyz": np.stack([x, y, z], axis=1).tolist()}
  doc = {"spectral": [], "angular": [],
         "color": [{"curve": curve, "angle": angle,
                    "illuminant": illum, "observer": cmf,
                    "quantity": quantity, "reference": list(reference),
                    "distance": distance, "weight": weight}],
         "cache_size": 128, "tolerance_floor": 1e-12}
  return compile_merit_spec(json.dumps(doc))


def xyz_oracle(row, wl, x, y, z, ev):
  """Pure-Python-loop XYZ (mirrors the Rust summation op-for-op).
  Aligned grids only (no resample) — callers slice interior nodes."""
  n = len(wl)
  dw = [wl[i + 1] - wl[i] for i in range(n - 1)] + [wl[-1] - wl[-2]]
  den = 0.0
  for i in range(n):
    den += ev[i] * y[i] * dw[i]
  k = 1.0 / den
  out = [0.0, 0.0, 0.0]
  for c, cc in enumerate((x, y, z)):
    s = 0.0
    for i in range(n):
      w = row[i] * ev[i] * k * dw[i]
      s += w * cc[i]
    out[c] = s
  return out


def xyy_of(xyz):
  s = xyz[0] + xyz[1] + xyz[2]
  inv = 1.0 / s
  return [xyz[0] * inv, xyz[1] * inv, xyz[1]]


def test_xyy_channels_hex_against_oracle():
  # Strictest test in the batch: fully analytic path, bitwise.
  wl, x, y, z, el, ev = window(500.0, 519.0)
  row = 0.2 + 0.6 * (wl - wl[0]) / (wl[-1] - wl[0])
  ref = (0.35, 0.36, 0.55)
  spec = demand(quantity="XyY", reference=ref, distance="Channels",
                weight=2.0, tables=(wl, x, y, z, el, ev))
  sim = sim_curves_from_arrays(np.array([0.0]), wl, {"Rs": row.reshape(1, -1)})
  got = spec.residuals(sim)
  assert spec.n_residuals() == 1 and len(got) == 1
  xyz = xyz_oracle(row.tolist(), wl.tolist(), x.tolist(), y.tolist(),
                   z.tolist(), ev.tolist())
  c = xyy_of(xyz)
  f = 2.0 * sum(((a - b) / 1.0) ** 2 for a, b in zip(c, ref))
  assert got[0].hex() == (f ** 0.5).hex()
  assert spec.merit(sim, 1e6).hex() == (got[0] * got[0]).hex()


def test_lab_de00_against_color_pipeline():
  wl, x, y, z, el, ev = window(400.0, 700.0)
  assert len(wl) == 301
  row = 0.5 + 0.3 * np.sin((wl - 400.0) / 300.0 * np.pi)
  ref = (62.0, 18.0, -34.0)
  spec = demand(quantity="Lab", reference=ref, distance="DeltaE2000",
                tables=(wl, x, y, z, el, ev))
  sim = sim_curves_from_arrays(np.array([0.0]), wl, {"Rs": row.reshape(1, -1)})
  got = spec.residuals(sim)
  xyz = xyz_oracle(row.tolist(), wl.tolist(), x.tolist(), y.tolist(),
                   z.tolist(), ev.tolist())
  white = xyz_oracle([1.0] * len(wl), wl.tolist(), x.tolist(), y.tolist(),
                     z.tolist(), ev.tolist())
  lab = C.XYZ_to_Lab(np.array([xyz]), illuminant=white)[0]
  d = C.delta_E_CIE2000(np.array([lab]), np.array([ref]), 1.0, 1.0, 1.0, False)[0]
  assert got[0] == pytest.approx(float(d), rel=1e-12)


def test_missing_curve_penalty_parity():
  wl, x, y, z, el, ev = window(500.0, 519.0)
  spec = demand(quantity="XyY", reference=(0.3, 0.3, 0.5),
                distance="Channels",
                tables=(wl, x, y, z, el, ev))
  row = np.full((1, len(wl)), 0.5)
  full = sim_curves_from_arrays(np.array([0.0]), wl, {"Rs": row})
  assert np.isfinite(spec.merit(full, 1e6))
  empty = sim_curves_from_arrays(np.array([0.0]), wl, {})
  assert spec.merit(empty, 1e6) == 1e6  # one penalty (shared key group)
  with pytest.raises(ValueError, match="Rs"):
    spec.residuals(empty)


def test_n_residuals_counts_color_as_one():
  wl, x, y, z, el, ev = window(500.0, 519.0)
  spec = demand(quantity="XyY", reference=(0.3, 0.3, 0.5),
                distance="Channels", tables=(wl, x, y, z, el, ev))
  assert spec.n_residuals() == 1
  row = np.full((1, len(wl)), 0.5)
  sim = sim_curves_from_arrays(np.array([0.0]), wl, {"Rs": row})
  assert len(spec.residuals(sim)) == 1


def test_named_tables_resolve():
  wl, x, y, z, el, ev = window(500.0, 519.0)
  spec = demand(quantity="Lab", reference=(60.0, 10.0, -20.0),
                distance="DeltaE76", names=True)
  row = np.full((1, len(wl)), 0.5)
  sim = sim_curves_from_arrays(np.array([0.0]), wl, {"Rs": row})
  assert np.isfinite(spec.merit(sim, 1e6))
