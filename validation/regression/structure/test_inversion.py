# -*- coding: utf-8 -*-
"""Regression: inversion mirror semantics (BUG-A/B/C/E accept).

An inverted block must be a MIRROR: order reversed AND plane quantities
(interface slice, roughness) transported to the same physical plane, with
the carve following the material. Asymmetric A/B/C stacks pin position,
mixing pair and roughness index; dispersive materials make the nk checks
order-sensitive (2D row-flip, wavelength axis intact).
"""

import numpy as np

from navette.structure import (
  DictMaterialProvider,
  Layer,
  Navette_Architect,
  Navette_Structure,
  solve_structure,
)
from navette.structure.types import BlockKind

WL = np.array([900.0, 1000.0, 1100.0])


def _disp(n, k, s):
  w = (WL - 1000.0) / 1000.0
  return (n + s * w) + 1j * np.maximum(k + 0.3 * s * w, 0.0)


MATS = DictMaterialProvider({
  "glass": _disp(1.52, 0.0, 0.02),
  "TiO2": _disp(2.35, 0.01, 0.15),
  "SiO2": _disp(1.46, 0.0, 0.03),
})


def _arch(st, inv=False, rep=1):
  arch = Navette_Architect(materials=MATS)
  arch.add_structure(st, inverted=inv, repeat=rep)
  return arch


def _mirror_eq(a, b):
  assert a.thicknesses.shape == b.thicknesses.shape
  np.testing.assert_allclose(a.thicknesses, b.thicknesses[::-1], rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(a.indices, b.indices[::-1], rtol=1e-12, atol=1e-12)


def _row_of(sa, nk_ref, thick_ref):
  for i in range(sa.thicknesses.shape[0]):
    if abs(sa.thicknesses[i] - thick_ref) < 1e-9 and np.allclose(sa.indices[i], nk_ref, rtol=1e-9, atol=1e-12):
      return i
  raise AssertionError(f"row nk~{nk_ref} t={thick_ref} not found")


def test_asymmetric_interface_position_and_pair():
  # BUG-A accept (STRUCT-3 corrected value: sqrt(eps), not eps).
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(50.0, "TiO2", interface=True, interface_thickness=5.0),
  ], {}, None)
  fwd = _arch(st).get_solver_inputs()
  inv = _arch(st, inv=True).get_solver_inputs()
  assert inv.thicknesses.tolist() == [45.0, 5.0, 100.0]
  sl = _row_of(inv, inv.indices[1], 5.0)
  assert sl == 1
  n_slice = inv.indices[1]
  assert 1.5 < n_slice.real.min() < 2.35  # true mix, between glass and TiO2
  _mirror_eq(fwd, inv)


def test_inverted_roughness_follows_plane():
  # BUG-B accept: sigma at the front of B's bulk row after mirroring.
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(50.0, "TiO2", roughness=4.0, rough_type=5),
  ], {}, None)
  inv = _arch(st, inv=True).get_solver_inputs()
  # Mirrored order is [TiO2, glass]; the physical (glass|TiO2) plane sits
  # at the front of the glass row — sigma must be there, not on TiO2.
  g_row = _row_of(inv, MATS.get_nk("glass"), 100.0)
  assert inv.rough_vals[g_row] == 4.0
  assert inv.rough_types[g_row] == 5
  assert inv.rough_vals[0] == 0.0  # incident edge is clean


def test_double_interface_mirror():
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(50.0, "TiO2", interface=True, interface_thickness=3.0),
    Layer(60.0, "SiO2", interface=True, interface_thickness=4.0),
  ], {}, None)
  fwd = _arch(st).get_solver_inputs()
  inv = _arch(st, inv=True).get_solver_inputs()
  assert fwd.thicknesses.tolist() == [100.0, 3.0, 47.0, 4.0, 56.0]
  assert inv.thicknesses.tolist() == [56.0, 4.0, 47.0, 3.0, 100.0]
  _mirror_eq(fwd, inv)


def test_graded_sublayer_reversal():
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(60.0, "TiO2", inhomogen=True, inh_delta=0.2),
  ], {}, MATS)
  fwd = _arch(st).get_solver_inputs()
  inv = _arch(st, inv=True).get_solver_inputs()
  n = st[1].sub_layer_count
  assert fwd.thicknesses.shape == inv.thicknesses.shape == (1 + n,)
  _mirror_eq(fwd, inv)


def test_graded_interface_combined():
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(60.0, "TiO2", inhomogen=True, inh_delta=0.2,
          interface=True, interface_thickness=4.0),
  ], {}, MATS)
  _mirror_eq(_arch(st).get_solver_inputs(), _arch(st, inv=True).get_solver_inputs())


def test_inverted_repeat():
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(50.0, "TiO2", interface=True, interface_thickness=5.0),
  ], {}, None)
  inv2 = _arch(st, inv=True, rep=2).get_solver_inputs()
  assert inv2.thicknesses.tolist() == [45.0, 5.0, 100.0] * 2


def test_first_layer_interface_repeat_mirror():
  st = Navette_Structure([
    Layer(100.0, "glass", interface=True, interface_thickness=5.0),
    Layer(50.0, "TiO2"),
  ], {}, None)
  _mirror_eq(_arch(st, inv=False, rep=2).get_solver_inputs(),
             _arch(st, inv=True, rep=2).get_solver_inputs())


def test_partial_inversion_run_edge_clean():
  # BUG-C accept: the inverted run's incident edge hosts no slice and a
  # neighbor block's edge flags never teleport into the run.
  chain = Navette_Architect(materials=MATS)
  chain.add_structure(Navette_Structure(
    [Layer(100.0, "glass", interface=True, interface_thickness=9.0)], {}, None))
  chain.add_structure(Navette_Structure(
    [Layer(10.0, "TiO2", interface=True, interface_thickness=2.0),
     Layer(30.0, "SiO2")], {}, None), inverted=True)
  chain.add_structure(Navette_Structure([Layer(40.0, "TiO2")], {}, None))
  sa = chain.get_solver_inputs()
  # B's interface described the (glass|B) plane, which the partial mirror
  # removes: interior transfer only, dangling treatments die.
  assert sa.thicknesses.tolist() == [100.0, 30.0, 10.0, 40.0]
  assert sa.indices.shape[0] == 4


def test_lossless_reciprocity():
  # Mirror-image stacks give identical R (lossless, same media both sides).
  wl0 = np.array([1000.0])
  mats = DictMaterialProvider({"glass": np.array([1.52 + 0j]), "TiO2": np.array([2.35 + 0j])})
  st = Navette_Structure([
    Layer(0.0, "glass"),
    Layer(50.0, "TiO2", interface=True, interface_thickness=5.0),
    Layer(0.0, "glass"),
  ], {}, mats)
  fwd = Navette_Architect(materials=mats)
  fwd.add_structure(st)
  bwd = Navette_Architect(materials=mats)
  bwd.add_structure(st, inverted=True)
  rf = np.asarray(solve_structure(fwd, wl0, [0.0])["R_avg"])
  rb = np.asarray(solve_structure(bwd, wl0, [0.0])["R_avg"])
  np.testing.assert_allclose(rf, rb, rtol=1e-12, atol=1e-12)


def test_shared_structure_not_mutated_by_inversion():
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(50.0, "TiO2", interface=True, interface_thickness=5.0, roughness=1.0),
  ], {}, None)
  _arch(st, inv=True).get_solver_inputs()
  assert st[0].thickness == 100.0 and not st[0].interface
  assert st[1].thickness == 50.0 and st[1].interface
