# -*- coding: utf-8 -*-
"""Restored GUI/legacy surface: Group props+draws, structure/architect
conveniences (re-added on request; twins of the documented behavior)."""
import pytest

from navette._structure import Group, Layer
from navette.structure import Navette_Architect, Navette_Structure


def test_group_properties_round_trip():
  g = Group("TiO2", thick_factor=1.1)
  props = g.get_properties()
  assert props["group_name"] == "TiO2"
  assert props["thick_factor"] == 1.1
  assert set(props) == set(g.get_state())
  g.set_properties({"thick_factor": 2.0})
  assert g.get_properties()["thick_factor"] == 2.0


def test_group_draws_seeded_and_shaped():
  g = Group("TiO2")  # identity defaults: gaussian thickness law
  assert g.thickness_error(50.0, seed=7) == g.thickness_error(50.0, seed=7)
  assert g.inh_delta_error(0.2, seed=7) == g.inh_delta_error(0.2, seed=7)
  assert g.sr_roughness_error(1.5, seed=7) == g.sr_roughness_error(1.5, seed=7)
  assert g.interface_error(2.0, seed=7) == g.interface_error(2.0, seed=7)
  z1 = g.nk_error(2.35 + 0.01j, seed=7)
  z2 = g.nk_error(complex(2.35, 0.01), seed=7)
  assert isinstance(z1, complex) and z1 == z2
  # Floors hold under an adversarial law (huge negative shift).
  g.set_error_params("thickness", {"abs_mean_delta_g": -1e6, "abs_std_dev": 0.0,
                                   "rel_mean_delta_g": 0.0, "rel_std_dev": 0.0,
                                   "abs_mean_delta_h": 0.0, "abs_variance": 0.0,
                                   "rel_mean_delta_h": 0.0, "rel_variance": 0.0})
  assert g.thickness_error(50.0, seed=1) == 0.0


def test_structure_conveniences():
  s = Navette_Structure([Layer(10.0, "glass"), Layer(20.0, "TiO2"),
                         Layer(30.0, "TiO2")], {}, None)
  assert s.find_layers_by_material("TiO2") == [1, 2]
  assert s.count_material("glass") == 1
  assert ("TiO2" in s) and ("Au" not in s)
  assert s.total_physical_thickness() == 60.0
  s.apply_to_all_layers(lambda layer: setattr(layer, "thickness", layer.thickness * 2))
  assert s.total_physical_thickness() == 120.0
  s.insert_layer(1, Layer(5.0, "Au"))
  assert [layer.material for layer in s.layer_list] == ["glass", "Au", "TiO2", "TiO2"]
  out = s.remove_layer(1)
  assert out.material == "Au" and len(s) == 3
  s.replace_layer(0, Layer(11.0, "glass"))
  assert s[0].thickness == 11.0
  with pytest.raises(IndexError):
    s.remove_layer(99)
  with pytest.raises(IndexError):
    s.replace_layer(-99, Layer(1.0, "x"))
  assert s.active_material_dict is s.materials
  s.active_material_dict = {"glass": [1.5 + 0j]}
  assert s.active_material_dict.contains("glass")


def test_architect_conveniences():
  a = Navette_Architect()
  a.add_structure(Navette_Structure([Layer(10.0, "glass")], {}, None))
  a.add_structure(Navette_Structure([Layer(20.0, "TiO2")], {}, None),
                  repeat=2, label="abs")
  assert a.replace_material("TiO2", "Ta2O5") == 1
  assert a.get_total_physical_thickness() == 50.0  # 10 + 20×2
  assert a.active_material_dict is a.materials
  c = a.copy()
  assert c.get_total_physical_thickness() == 50.0
  assert [b.label for b in c.blocks] == ["", "abs"]
  c.replace_material("Ta2O5", "Nb2O5")
  assert "Ta2O5" in a._shells[1] and "Nb2O5" in c._shells[1]
