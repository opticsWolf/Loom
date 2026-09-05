# -*- coding: utf-8 -*-
"""Regression: index spaces, coherence warts and nits (BUG-D, WART-1/2/3/5/6/7,
NIT-1/2/3/5/6/7 accept)."""

import warnings

import numpy as np
import pytest

import navette.structure.expander as expander_mod
import navette.structure.models as models_mod
from navette.structure import (
  BlockKind,
  DictMaterialProvider,
  Group,
  Layer,
  Navette_Architect,
  Navette_Structure,
)

MATS = DictMaterialProvider({
  "glass": np.full(1, 1.52 + 0j),
  "TiO2": np.full(1, 2.35 + 0.01j),
})


def _arch(st, **kw):
  arch = Navette_Architect(materials=MATS)
  arch.add_structure(st, **kw)
  return arch


# BUG-D: logical vs solver index spaces -------------------------------------
def test_global_index_is_logical_space():
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(60.0, "TiO2", inhomogen=True, inh_delta=0.2),
  ], {}, MATS)
  arch = _arch(st)
  n_sub = st[1].sub_layer_count
  assert n_sub > 1
  struct, local = arch.map_global_index_to_layer(1)
  assert struct is st and local == 1  # one slot per Layer, not per slice


def test_solver_index_resolves_expanded_rows():
  st = Navette_Structure([
    Layer(100.0, "glass"),
    Layer(60.0, "TiO2", inhomogen=True, inh_delta=0.2,
          interface=True, interface_thickness=4.0),
  ], {}, MATS)
  arch = _arch(st)
  sa = arch.get_solver_inputs()
  n = sa.thicknesses.shape[0]
  assert n > 2
  # Interface slice (row 1) belongs to its carrier's logical layer (1).
  _, loc_slice = arch.map_solver_index_to_layer(1)
  assert loc_slice == 1
  # Last sublayer row still resolves to logical layer 1.
  _, loc_last = arch.map_solver_index_to_layer(n - 1)
  assert loc_last == 1
  # First row resolves to logical layer 0.
  _, loc_first = arch.map_solver_index_to_layer(0)
  assert loc_first == 0
  with pytest.raises(IndexError):
    arch.map_solver_index_to_layer(n)


# WART-1: len/block coherence -------------------------------------------------
def test_len_is_block_count():
  arch = _arch(Navette_Structure([Layer(10.0, "glass")], {}, None))
  arch.add_structure(Navette_Structure([Layer(20.0, "TiO2")], {}, None))
  assert len(arch) == arch.block_count == 2
  assert arch[0] is arch.blocks[0]
  assert arch.get_global_layer_count() == 2


# WART-2: no singleton leak ---------------------------------------------------
def test_group_lookup_returns_copy():
  st = Navette_Structure([Layer(10.0, "unlisted")], {}, MATS)
  g = st.get_group_for_material("unlisted")
  g.thick_factor = -5.0
  assert st.get_group_for_material("unlisted").thick_factor == 1.0


# WART-3: provider conflicts raise --------------------------------------------
def test_add_conflicting_providers_raise():
  other_mats = DictMaterialProvider({"glass": np.full(1, 9.0 + 0j)})
  a = Navette_Structure([Layer(10.0, "glass")], {}, MATS)
  b = Navette_Structure([Layer(10.0, "glass")], {}, other_mats)
  with pytest.raises(ValueError):
    a + b
  c = Navette_Structure([Layer(10.0, "glass")], {}, MATS)
  assert (a + c).validate() == []


# WART-5: split preserves the interface definition --------------------------------
def test_split_preserves_interface_definition():
  # interface_thickness is a process parameter, not a divisible budget:
  # both halves keep it so re-growing restores the intended boundary.
  st = Navette_Structure(
    [Layer(100.0, "TiO2", interface=True, interface_thickness=6.0)], {}, MATS)
  arch = _arch(st)
  arch.split_layer_at_global(0, split_ratio=0.25)
  assert len(st) == 2
  assert st[0].interface_thickness == pytest.approx(6.0)
  assert st[1].interface_thickness == pytest.approx(6.0)
  assert st[0].thickness + st[1].thickness == pytest.approx(100.0)


def test_split_transient_expands_and_validates():
  # 10 nm film, 5 nm interface, needle at 3 nm: thin half keeps the full
  # definition, expands to slice + 0 nm bulk via the clamp, and validates
  # (carve-explained zeros are legitimate transients).
  st = Navette_Structure(
    [Layer(100.0, "glass"),
     Layer(10.0, "TiO2", interface=True, interface_thickness=5.0)], {}, MATS)
  arch = _arch(st)
  arch.split_layer_at_global(1, split_ratio=0.3)
  from navette.structure.types import is_warning
  assert all(is_warning(i) for i in st.validate())  # overhang warns, never blocks
  with pytest.warns(UserWarning):
    sa = arch.get_solver_inputs()
  assert sa.thicknesses.tolist() == pytest.approx([100.0, 3.0, 0.0, 5.0, 2.0])


def test_duplicate_keeps_full_budget():
  st = Navette_Structure(
    [Layer(100.0, "TiO2", interface=True, interface_thickness=6.0)], {}, MATS)
  arch = _arch(st)
  arch.duplicate_layer_at_global(0)
  assert [l.interface_thickness for l in st] == [6.0, 6.0]


# WART-6: corrupt refs raise ----------------------------------------------------
def test_from_state_bad_ref_raises():
  arch = Navette_Architect(materials=MATS)
  arch.add_structure(Navette_Structure([Layer(10.0, "glass")], {}, None))
  state = arch.get_state()
  state["blocks"].append({"structure_ref": 7, "inverted": False,
                          "repeat_count": 1, "label": "x"})
  with pytest.raises(ValueError):
    Navette_Architect.from_state(state, materials=MATS)


# WART-7: systematic error semantics --------------------------------------------
def test_apply_error_is_systematic_across_wl():
  g = Group("x")
  arr = np.full(4, 1.5)
  out1 = Group._apply_error(arr, 0, g.thickness_error_params,
                            rng=np.random.default_rng(3))
  out2 = Group._apply_error(arr, 0, g.thickness_error_params,
                            rng=np.random.default_rng(3))
  np.testing.assert_allclose(out1, out2)  # seeded reproducible
  assert np.ptp(out1) == 0.0  # one offset across lambda, not per-lambda noise


# bake_films -------------------------------------------------------------------------
def test_bake_is_expansion_identical():
  g = Group("TiO2", thick_factor=1.1, thick_summand=2.0, inh_delta_summand=0.1,
              roughness_summand=1.0, interface_summand=2.0)
  st = Navette_Structure(
    [Layer(100.0, "glass"),
     Layer(50.0, "TiO2", roughness=3.0, inh_delta=0.2, inhomogen=True,
           interface=True, interface_thickness=5.0)],
    {"TiO2": g}, MATS)
  before = st.get_solver_inputs()
  assert st.bake_films() == 1
  assert (g.thick_factor, g.thick_summand, g.inh_delta_summand,
          g.roughness_summand, g.interface_summand) == (1.0, 0.0, 0.0, 0.0, 0.0)
  assert st[1].thickness == pytest.approx(57.0)
  assert st[1].inh_delta == pytest.approx(0.3)
  assert st[1].roughness == pytest.approx(4.0)
  assert st[1].interface_thickness == pytest.approx(7.0)
  after = st.get_solver_inputs()
  # Same totals, same slice, same grading endpoints; graded row count
  # re-follows the refinement rule at the new thickness (rediscretized).
  assert after.thicknesses.sum() == pytest.approx(before.thicknesses.sum())
  assert after.thicknesses[1] == pytest.approx(before.thicknesses[1])
  n_b, n_a = before.thicknesses.shape[0], after.thicknesses.shape[0]
  np.testing.assert_allclose(after.indices[2], before.indices[2])
  np.testing.assert_allclose(after.indices[n_a - 1], before.indices[n_b - 1])
  fresh = Layer(57.0, "TiO2", inhomogen=True, inh_delta=0.3)
  assert st[1].sub_layer_count == fresh.sub_layer_count


def test_bake_flat_stack_is_bit_identical():
  g = Group("TiO2", thick_factor=1.1, thick_summand=2.0,
              roughness_summand=1.0, interface_summand=2.0)
  st = Navette_Structure(
    [Layer(100.0, "glass"),
     Layer(50.0, "TiO2", roughness=3.0, interface=True, interface_thickness=5.0)],
    {"TiO2": g}, MATS)
  before = st.get_solver_inputs()
  st.bake_films()
  after = st.get_solver_inputs()
  np.testing.assert_allclose(after.thicknesses, before.thicknesses)
  np.testing.assert_allclose(after.indices, before.indices)


def test_bake_refuses_nk_scaling_atomically():
  g = Group("TiO2", thick_factor=2.0, n_factor=1.1)
  st = Navette_Structure([Layer(50.0, "TiO2"), Layer(60.0, "TiO2")], {"TiO2": g}, MATS)
  with pytest.raises(ValueError):
    st.bake_films()
  assert st[0].thickness == 50.0 and st[1].thickness == 60.0  # untouched
  assert g.thick_factor == 2.0  # not reset either


def test_bake_leaves_orphans_alone():
  orphan = Group("Nobody", thick_factor=3.0)
  st = Navette_Structure([Layer(50.0, "TiO2")],
                         {"TiO2": Group("TiO2", thick_factor=2.0), "Nobody": orphan}, MATS)
  assert st.bake_films() == 1
  assert st[0].thickness == 100.0
  assert orphan.thick_factor == 3.0


# bake_materials --------------------------------------------------------------------
def test_bake_materials_creates_table_and_renames():
  wl = np.array([900., 1000., 1100.])
  mats = {"TiO2": np.full(3, 2.35 + 0.01j), "glass": np.full(3, 1.52 + 0j)}
  st = Navette_Structure(
    [Layer(0.0, "glass"), Layer(50.0, "TiO2"), Layer(0.0, "glass")],
    {"TiO2": Group("TiO2", n_factor=1.1, k_factor=0.5)}, dict(mats))
  before = st.get_solver_inputs()
  mapping = st.bake_materials(wl)
  assert mapping == {"TiO2": "TiO2_table"}
  assert [l.material for l in st] == ["glass", "TiO2_table", "glass"]
  assert (st.group_dict["TiO2_table"].n_factor, st.group_dict["TiO2_table"].k_factor) == (1.0, 1.0)
  spec = st.materials._dict["TiO2_table"]
  assert spec.model == "Table"
  after = st.get_solver_inputs()
  np.testing.assert_allclose(after.thicknesses, before.thicknesses)
  np.testing.assert_allclose(after.indices, before.indices)


def test_bake_materials_group_collision_skipped():
  wl = np.array([1000.])
  mats = {"TiO2": np.full(1, 2.35 + 0.01j)}
  st = Navette_Structure([Layer(50.0, "TiO2")],
                         {"TiO2": Group("TiO2", n_factor=2.0),
                          "TiO2_table": Group("TiO2_table")},
                         dict(mats))
  assert st.bake_materials(wl) == {"TiO2": "TiO2_table2"}
  assert "TiO2_table" in st.group_dict  # orphan preserved
  assert st.group_dict["TiO2_table2"].group_name == "TiO2_table2"


def test_bake_materials_version_chain():
  wl = np.array([1000.])
  mats = {"TiO2": np.full(1, 2.35 + 0.01j), "TiO2_table": np.full(1, 9.0 + 0j)}
  st = Navette_Structure([Layer(50.0, "TiO2")], {"TiO2": Group("TiO2", n_factor=2.0)}, dict(mats))
  assert st.bake_materials(wl) == {"TiO2": "TiO2_table2"}
  st2 = Navette_Structure([Layer(50.0, "Xtable")],
                          {"Xtable": Group("Xtable", k_factor=3.0)},
                          dict({"Xtable": np.full(1, 1.5 + 0.02j)}))
  assert st2.bake_materials(wl) == {"Xtable": "Xtable2"}


def test_bake_materials_grid_mismatch_and_film_refusal():
  st = Navette_Structure([Layer(50.0, "TiO2")], {"TiO2": Group("TiO2", n_factor=2.0)},
                         dict({"TiO2": np.full(5, 2.35 + 0j)}))
  with pytest.raises(ValueError):
    st.bake_materials(np.array([900., 1000.]))
  st2 = Navette_Structure([Layer(50.0, "TiO2")], {"TiO2": Group("TiO2", n_factor=2.0)},
                          dict({"TiO2": np.full(1, 2.35 + 0j)}))
  with pytest.raises(ValueError):
    st2.bake_films()  # n/k side must go through bake_materials


# bridge grid assurance ---------------------------------------------------------------
WL3 = np.array([900., 1000., 1100.])
MATS3 = {"glass": np.full(3, 1.52 + 0j), "TiO2": np.full(3, 2.35 + 0.01j)}


def _three_layer(mats):
  from navette.structure import Navette_Structure
  return Navette_Structure(
    [Layer(0.0, "glass"), Layer(50.0, "TiO2"), Layer(0.0, "glass")], {}, mats)


def test_bridge_grid_match_solves_silently():
  import warnings
  from navette.structure import solve_structure
  from navette.structure.materials import DictMaterialProvider
  with warnings.catch_warnings():
    warnings.simplefilter("error")  # no warning on exact grid match
    out = solve_structure(
      _three_layer(DictMaterialProvider(dict(MATS3), wavelength=WL3)), WL3, 0.0)
  assert out["Rs"].shape == (3,)


def test_bridge_grid_value_mismatch_refused():
  from navette.structure import solve_structure
  from navette.structure.materials import DictMaterialProvider
  bad = DictMaterialProvider(dict(MATS3), wavelength=np.array([800., 900., 1000.]))
  with pytest.raises(ValueError):
    solve_structure(_three_layer(bad), WL3, 0.0)  # same length, other values


def test_bridge_gridless_warns_and_solves():
  from navette.structure import solve_structure
  with pytest.warns(UserWarning, match="grid unknown"):
    out = solve_structure(_three_layer(dict(MATS3)), WL3, 0.0)
  assert out["Rs"].shape == (3,)


def test_bridge_spec_grid_mismatch_refused():
  from navette.structure import solve_structure
  from navette.structure.materials import MaterialObjectProvider
  from navette.materials import MaterialSpec
  specs = {"glass": MaterialSpec("Konstant", {"n": 1.52}),
           "TiO2": MaterialSpec("Konstant", {"n": 2.35, "k": 0.01})}
  prov = MaterialObjectProvider(dict(specs), np.array([700., 750., 800.]))
  with pytest.raises(ValueError):
    solve_structure(_three_layer(prov), WL3, 0.0)


def test_dict_provider_refresh_swaps_atomically():
  from navette.structure.materials import DictMaterialProvider
  prov = DictMaterialProvider({"TiO2": np.full(3, 2.0 + 0j)}, wavelength=WL3)
  assert prov.grid.tolist() == WL3.tolist()
  prov.refresh({"TiO2": np.full(2, 2.0 + 0j)}, np.array([500., 600.]))
  assert prov.grid.tolist() == [500., 600.]
  np.testing.assert_allclose(prov.get_nk("TiO2"), np.full(2, 2.0 + 0j))


def test_dict_provider_off_grid_array_refused_at_serve():
  from navette.structure.materials import DictMaterialProvider
  prov = DictMaterialProvider({"TiO2": np.full(5, 2.0 + 0j)}, wavelength=WL3)
  with pytest.raises(ValueError):
    prov.get_nk("TiO2")


# weaver provider strictness ----------------------------------------------------------
class _StubWeaver:
  """Dict-backed weaver stub: {(prefix, label, pol): (wl, data)}."""
  def __init__(self, frags):
    self._frags = dict(frags)
  def __contains__(self, key):
    return key in self._frags
  def get_weaved(self, key):
    return self._frags[key]


def _stub_weaver_provider(wl, strict=False):
  from navette.structure.materials import WeaverMaterialProvider
  frags = {(0.0, "n", "Si"): (np.array([400., 500., 600.]), np.array([3.5, 3.6, 3.7])),\
           (0.0, "k", "Si"): (np.array([400., 500., 600.]), np.array([0.01, 0.02, 0.03]))}
  return WeaverMaterialProvider(_StubWeaver(frags), np.asarray(wl, dtype=np.float64),
                                strict=strict)


def test_weaver_strict_exact_serves_without_fallback():
  import navette.structure.materials as matmod
  prov = _stub_weaver_provider([400., 500., 600.], strict=True)
  assert prov.is_exact("Si")
  old, matmod.UniSpline = matmod.UniSpline, None  # fallback unavailable…
  try:
    nk = prov.get_nk("Si")  # …yet exact-grid serving works
  finally:
    matmod.UniSpline = old
  np.testing.assert_allclose(nk, [3.5 + 0.01j, 3.6 + 0.02j, 3.7 + 0.03j])


def test_weaver_strict_off_grid_refuses():
  prov = _stub_weaver_provider([450., 550., 650.], strict=True)
  assert not prov.is_exact("Si")
  with pytest.raises(ValueError, match="strict"):
    prov.get_nk("Si")


def test_weaver_non_strict_interpolates():
  prov = _stub_weaver_provider([450., 550., 650.], strict=False)
  nk = prov.get_nk("Si")  # UniSpline fallback path
  assert nk.shape == (3,)
  assert 3.5 < nk[0].real < 3.7


def test_weaver_target_reset_clears_cache():
  prov = _stub_weaver_provider([400., 500., 600.], strict=False)
  before = prov.get_nk("Si")
  assert "Si" in prov._cache
  prov.target_wavelength = np.array([400., 500., 600.])  # identical: no-op
  assert "Si" in prov._cache
  prov.target_wavelength = np.array([450., 550., 650.])  # fresh grid: cleared
  assert "Si" not in prov._cache
  assert prov.grid.tolist() == [450., 550., 650.]
  assert not prov.is_exact("Si")


# NIT-2: module docstrings live --------------------------------------------------
def test_module_docstrings_present():
  assert models_mod.__doc__ and "Layer and Group" in models_mod.__doc__
  assert expander_mod.__doc__ and "SolverArrays" in expander_mod.__doc__


# NIT-3: interface policy ---------------------------------------------------------
def test_negative_interface_thickness_flagged():
  st = Navette_Structure(
    [Layer(100.0, "glass"), Layer(50.0, "TiO2", interface=True, interface_thickness=-2.0)],
    {}, MATS)
  assert any("Negative interface thickness" in i for i in st.validate())


# Severity channel: warnings never block -----------------------------------------
def test_overhang_is_warning_and_solves():
  from navette.structure.types import is_warning
  st = Navette_Structure(
    [Layer(100.0, "glass"), Layer(3.0, "TiO2", interface=True, interface_thickness=5.0)],
    {}, MATS)
  issues = st.validate()
  assert len(issues) == 1 and is_warning(issues[0])
  with pytest.warns(UserWarning):
    sa = st.get_solver_inputs()  # flow continues despite the warning
  assert sa.thicknesses.tolist() == [100.0, 3.0, 0.0]


def test_orphan_group_is_warning_and_solves():
  from navette.structure.types import is_warning
  st = Navette_Structure([Layer(50.0, "TiO2")], {"Nobody": Group("Nobody")}, MATS)
  issues = st.validate()
  assert len(issues) == 1 and is_warning(issues[0])
  with pytest.warns(UserWarning):
    assert st.get_solver_inputs().thicknesses.tolist() == [50.0]


def test_errors_still_block():
  st = Navette_Structure([Layer(-5.0, "TiO2")], {}, MATS)
  with pytest.raises(ValueError):
    st.get_solver_inputs()


# NIT-6: from_state deep-copies (also pinned in test_roundtrip) --------------------
def test_group_from_state_independent_params():
  g = Group("x")
  back = Group.from_state(g.get_state())
  back.thickness_error_params["abs_std_dev"] = 99.0
  assert g.thickness_error_params["abs_std_dev"] == 0.01


# NIT-7: provider-overwrite warning --------------------------------------------------
def test_materials_setter_warns_on_overwrite():
  arch = Navette_Architect()
  arch.add_structure(Navette_Structure([Layer(10.0, "glass")], {}, MATS))
  other = DictMaterialProvider({"glass": np.full(1, 1.5 + 0j)})
  with pytest.warns(UserWarning):
    arch.materials = other
  arch2 = Navette_Architect()
  arch2.add_structure(Navette_Structure([Layer(10.0, "glass")], {}, None))
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    arch2.materials = MATS  # None -> provider: silent
