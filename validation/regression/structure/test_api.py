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
