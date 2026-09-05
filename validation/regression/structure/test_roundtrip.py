# -*- coding: utf-8 -*-
"""Regression: state round-trips and validation (BUG-1, BUG-2 accept).

BUG-1: `from_state(get_state())` preserves every field — material names
included — for Layer, Group, Navette_Structure and Navette_Architect
(shared-block aliasing and material re-attachment included).
BUG-2: `validate()` never crashes; missing materials are reported.
"""

import numpy as np

from navette.structure import (
  DictMaterialProvider,
  Group,
  Layer,
  Navette_Architect,
  Navette_Structure,
)

WL = np.array([1000.0])
MATS = DictMaterialProvider({
  "glass": np.full(1, 1.52 + 0j),
  "TiO2": np.full(1, 2.35 + 0.01j),
})


def _full_layer() -> Layer:
  return Layer(
    thickness=50.0, material_name="TiO2", coherent=False,
    roughness=2.0, rough_type=5, inhomogen=True, inh_delta=0.2,
    interface=True, interface_thickness=3.0, optimize=False,
    needle=False, layer_type=1,
  )


def test_layer_roundtrip_all_fields():
  original = _full_layer()
  back = Layer.from_state(original.get_state())
  assert back.get_state() == original.get_state()
  assert back.material == "TiO2"  # BUG-1: was '' before STRUCT-2


def test_layer_roundtrip_minimal():
  original = Layer(thickness=10.0, material_name="glass")
  back = Layer.from_state(original.get_state())
  assert (back.material, back.thickness) == ("glass", 10.0)


def test_group_roundtrip():
  original = Group("TiO2", thick_factor=1.1, n_factor=0.9, k_factor=1.2)
  original.error_mask = [1, 0, 1, 0, 0, 0]
  original.optimization_mask = [0, 1, 1, 1, 1, 1, 1]
  back = Group.from_state(original.get_state())
  assert back.get_state() == original.get_state()
  # Masks are independent copies, not aliased dicts/lists.
  back.error_mask[0] = 9
  assert original.error_mask[0] == 1


def test_structure_roundtrip():
  original = Navette_Structure(
    [_full_layer(), Layer(100.0, "glass")],
    {"TiO2": Group("TiO2", n_factor=1.1)},
    MATS,
  )
  back = Navette_Structure.from_state(original.get_state(), materials=MATS)
  assert [l.get_state() for l in back] == [l.get_state() for l in original]
  assert set(back.group_dict) == {"TiO2"}
  assert back.group_dict["TiO2"].n_factor == 1.1
  assert back.validate() == []


def test_architect_roundtrip_shared_aliasing():
  shared = Navette_Structure([Layer(50.0, "TiO2")], {}, MATS)
  original = Navette_Architect(materials=MATS)
  original.add_structure(shared, label="a")
  original.add_structure(shared, inverted=True, label="b")
  back = Navette_Architect.from_state(original.get_state(), materials=MATS)
  assert back.block_count == 2
  # Shared-block aliasing survives the trip (one state, two refs).
  assert back.blocks[0].structure is back.blocks[1].structure
  assert back.blocks[1].inverted is True
  assert back.validate() == []


def test_validate_missing_material_no_crash():
  arch = Navette_Architect(materials=MATS)
  arch.add_structure(Navette_Structure([Layer(50.0, "Nope")], {}, None))
  issues = arch.validate()  # BUG-2: AttributeError before STRUCT-6
  assert issues == ["Layer 0: Material 'Nope' not found in material provider."]


def test_validate_without_materials_skips_coverage():
  arch = Navette_Architect()
  arch.add_structure(Navette_Structure([Layer(50.0, "Nope")], {}, None))
  assert arch.validate() == []


def test_validate_catches_solver_blockers():
  bad = Navette_Structure([Layer(-5.0, "TiO2")], {}, MATS)
  assert any("Negative thickness" in i for i in bad.validate())
  arch = Navette_Architect(materials=MATS)
  arch.add_structure(bad)
  try:
    arch.get_solver_inputs()
  except ValueError:
    pass
  else:
    raise AssertionError("solve gate did not raise on invalid structure")
