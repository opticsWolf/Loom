# -*- coding: utf-8 -*-
"""Regression: state round-trips and validation (BUG-1, BUG-2 accept).

BUG-1: `from_state(get_state())` preserves every field — material names
included — for Layer, Group, Navette_Structure and Navette_Architect
(shared-block aliasing and material re-attachment included).
BUG-2: `validate()` never crashes; missing materials are reported.
"""

import numpy as np
import pytest

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


def test_stale_schema_versions_refused():
  from navette.structure.types import SCHEMA_VERSION
  layer_state = _full_layer().get_state()
  assert layer_state["schema_version"] == SCHEMA_VERSION
  untagged = dict(layer_state)
  del untagged["schema_version"]  # no past: untagged is malformed
  with pytest.raises(ValueError):
    Layer.from_state(untagged)
  arch = Navette_Architect(materials=MATS)
  arch.add_structure(Navette_Structure([Layer(10.0, "glass")], {}, None))
  stale = arch.get_state()
  stale["schema_version"] = SCHEMA_VERSION - 1
  with pytest.raises(ValueError):
    Navette_Architect.from_state(stale, materials=MATS)
  future = arch.get_state()
  future["schema_version"] = SCHEMA_VERSION + 999
  with pytest.raises(ValueError):
    Navette_Architect.from_state(future, materials=MATS)


# Fingerprint: the exact serialized key set per entity, at the version it
# was recorded at. If this fails, the key set changed — classify FIRST:
# removed/renamed key or changed meaning  -> breaking: bump SCHEMA_VERSION,
#   then update the fingerprint below;
# purely additive key (unknown keys are ignored by every from_state, so old
#   readers stay safe)                    -> update the fingerprint only.
FINGERPRINT = {
  "version": 1,
  "Layer": ["coherent", "inh_delta", "inhomogen", "interface",
              "interface_thickness", "layer_type", "material_name",
              "needle", "optimize", "rough_type", "roughness",
              "schema_version", "thickness"],
  "Group": ["group_name", "inh_delta_summand", "inh_delta_error_params",
              "inh_delta_error_type", "interface_error_params",
              "interface_error_type", "interface_summand", "k_error_params",
              "k_error_type", "k_factor", "n_error_params", "n_error_type",
              "n_factor", "roughness_error_params", "roughness_error_type",
              "roughness_summand", "schema_version", "thick_factor",
              "thick_summand", "thickness_error_params",
              "thickness_error_type", "error_mask", "optimization_mask"],
  "Navette_Structure": ["groups", "layers", "schema_version"],
  "Navette_Architect": ["blocks", "schema_version", "structures"],
}


def test_state_fingerprint():
  from navette.structure.types import SCHEMA_VERSION
  assert FINGERPRINT["version"] == SCHEMA_VERSION, \
    "fingerprint recorded at a different version — re-classify (see comment)"
  assert sorted(_full_layer().get_state()) == sorted(FINGERPRINT["Layer"])
  assert sorted(Group("g").get_state()) == sorted(FINGERPRINT["Group"])
  st = Navette_Structure([Layer(10.0, "glass")], {"glass": Group("glass")}, MATS)
  assert sorted(st.get_state()) == sorted(FINGERPRINT["Navette_Structure"])
  arch = Navette_Architect(materials=MATS)
  arch.add_structure(st)
  assert sorted(arch.get_state()) == sorted(FINGERPRINT["Navette_Architect"])


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
