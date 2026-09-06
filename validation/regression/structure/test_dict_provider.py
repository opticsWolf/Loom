"""DictMaterialProvider thinned onto the native DictProvider: twins."""
import numpy as np
import pytest

from navette.structure.materials import DictMaterialProvider
from navette.materials import MaterialSpec, evaluate

WL = np.array([500.0, 600.0, 700.0])

def test_arrays_serve_bitwise_and_refresh_atomic():
    d = {"TiO2": np.full(3, 2.0 + 0j)}
    prov = DictMaterialProvider(d, wavelength=WL)
    assert np.array_equal(prov.get_nk("TiO2"), np.full(3, 2.0 + 0j))
    prov.refresh({"TiO2": np.full(2, 2.0 + 0j)}, np.array([500., 600.]))
    assert prov.grid.tolist() == [500., 600.]
    assert np.array_equal(prov.get_nk("TiO2"), np.full(2, 2.0 + 0j))

def test_specs_evaluate_on_grid():
    spec = MaterialSpec("Konstant", {"n": 1.52})
    prov = DictMaterialProvider({"glass": spec}, wavelength=WL)
    assert np.array_equal(prov.get_nk("glass"), np.ascontiguousarray(evaluate(spec, WL)))

def test_off_grid_array_refused_at_serve():
    prov = DictMaterialProvider({"TiO2": np.full(5, 2.0 + 0j)}, wavelength=WL)
    with pytest.raises(ValueError):
        prov.get_nk("TiO2")

def test_wrong_length_insert_refused():
    prov = DictMaterialProvider({"TiO2": np.full(3, 2.0 + 0j)}, wavelength=WL)
    prov._dict["Si"] = np.full(5, 3.5 + 0j)
    with pytest.raises(ValueError):
        prov.get_nk("Si")

def test_gridless_spec_refused_and_keyerror():
    prov = DictMaterialProvider({"s": MaterialSpec("Konstant", {"n": 1.5})})
    with pytest.raises(AttributeError):
        prov.get_nk("s")
    with pytest.raises(KeyError):
        prov.get_nk("nope")
    assert not prov.contains("nope")

def test_pour_back_write_through():
    prov = DictMaterialProvider({"TiO2": np.full(3, 2.0 + 0j)}, wavelength=WL)
    prov._dict["Si"] = np.full(3, 3.5 + 0j)  # external add, then serve syncs
    assert np.array_equal(prov.get_nk("Si"), np.full(3, 3.5 + 0j))
