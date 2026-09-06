# -*- coding: utf-8 -*-
"""Color Python surface: ``ColorTarget`` validation mirrors native messages,
_dump resolution, ``build_merit_spec`` color section, TargetSet roundtrip
(plan R5). Program documents carry no targets section (materials/groups/
structure/architect/program only), so the roundtrip is TargetSet JSON —
dump -> JSON -> compile -> same merit."""
import json

import numpy as np
import pytest

from navette.spectralweave import ColorTarget, TargetCollection
from navette.synthesis import build_merit_spec, sim_curves_from_arrays
from navette.data import load_cie_table

WL = np.linspace(500., 519., 20)


def test_construction_refusals_mirror_native():
  with pytest.raises(ValueError, match="front R/T"):
    ColorTarget(curve="Nope")
  with pytest.raises(ValueError, match="back-incidence"):
    ColorTarget(curve="RBs")
  with pytest.raises(ValueError, match="unknown quantity"):
    ColorTarget(quantity="LCh")
  with pytest.raises(ValueError, match="unknown distance"):
    ColorTarget(distance="DIN99")
  with pytest.raises(ValueError, match="scalar reference"):
    ColorTarget(reference=1.5)
  with pytest.raises(ValueError, match="\\[3\\] triple"):
    ColorTarget(reference=[1.0, 2.0])
  with pytest.raises(ValueError, match="unknown illuminant"):
    ColorTarget(illuminant="F2")
  # Defaults construct clean.
  ColorTarget()
  ColorTarget(quantity="XyY", reference=(0.3, 0.3, 0.5), distance="Channels")


def test_dump_passes_names_through():
  # Deviation from the D6 draft (recorded in the plan): _dump does NOT
  # bake 471-point tables into every document — names ride through and the
  # native compiler resolves them from the sync-guarded embedded defaults.
  # Smaller docs, single canonical table source at compile.
  d = ColorTarget()._dump()
  assert d["illuminant"] == "D65" and d["observer"] == "1931_2deg"
  assert d["kind"] == "Exact" and d["transform"] == "linear"
  assert d["reference"] == [62.0, 18.0, -34.0]
  # Explicit dicts pass through (arrays normalized to lists).
  e = ColorTarget(illuminant={"wavelengths": np.array([500., 600.]),
                              "values": np.array([1., 1.])})._dump()
  assert e["illuminant"] == {"wavelengths": [500., 600.], "values": [1., 1.]}


def test_collection_and_build_merit_spec():
  col = TargetCollection()
  col.add(ColorTarget())
  assert col.count == 1 and len(col.color_targets) == 1
  spec = build_merit_spec(col)
  assert spec.n_residuals() == 1
  row = np.full((1, len(WL)), 0.5)
  sim = sim_curves_from_arrays(np.array([0.0]), WL, {"Ru": row})
  assert np.isfinite(spec.merit(sim, 1e6))
  col.clear()
  assert col.count == 0


def test_target_set_json_roundtrip_same_merit():
  col = TargetCollection()
  col.add(ColorTarget(quantity="XyY", reference=(0.3, 0.3, 0.5),
                      distance="Channels", weight=2.0))
  doc = {"spectral": [], "angular": [],
         "color": [t._dump() for t in col.color_targets]}
  from navette._smatrix import compile_merit_spec
  a = compile_merit_spec(json.dumps(doc))
  b = compile_merit_spec(json.dumps(json.loads(json.dumps(doc))))
  row = np.full((1, len(WL)), 0.5)
  sim = sim_curves_from_arrays(np.array([0.0]), WL, {"Ru": row})
  assert a.n_residuals() == b.n_residuals() == 1
  assert a.merit(sim, 1e6).hex() == b.merit(sim, 1e6).hex()
  assert a.residuals(sim).tobytes().hex() == b.residuals(sim).tobytes().hex()


def test_named_and_explicit_tables_agree():
  # Embedded-name resolution == source-file tables, numerically.
  cmf = load_cie_table("CIE", "cmf", "CIE_xyz_1931_2deg.json")
  d65 = load_cie_table("CIE", "sds", "CIE_std_illum_D65_S_D65.json")
  from navette._smatrix import compile_merit_spec
  ref = [60.0, 10.0, -20.0]
  named = compile_merit_spec(json.dumps(
    {"spectral": [], "angular": [],
     "color": [{"curve": "Ru", "angle": 0.0, "illuminant": "D65",
                 "observer": "1931_2deg", "quantity": "Lab",
                 "reference": ref, "distance": "DeltaE76"}]}))
  explicit = compile_merit_spec(json.dumps(
    {"spectral": [], "angular": [],
     "color": [{"curve": "Ru", "angle": 0.0,
                 "illuminant": {"wavelengths": d65["lambda"].tolist(),
                                 "values": d65["S_D65(lambda)"].tolist()},
                 "observer": {"wavelengths": cmf["lambda"].tolist(),
                              "xyz": np.stack([cmf["x_bar(lambda)"], cmf["y_bar(lambda)"],
                                                 cmf["z_bar(lambda)"]], axis=1).tolist()},
                 "quantity": "Lab", "reference": ref, "distance": "DeltaE76"}]}))
  row = np.full((1, len(WL)), 0.5)
  sim = sim_curves_from_arrays(np.array([0.0]), WL, {"Ru": row})
  assert named.merit(sim, 1e6).hex() == explicit.merit(sim, 1e6).hex()
