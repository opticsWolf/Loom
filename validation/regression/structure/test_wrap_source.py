"""wrap_material_source dispatch over the native-backed providers."""
import numpy as np
import pytest

from navette.structure.materials import (
    DictMaterialProvider, MaterialObjectProvider, WeaverMaterialProvider,
    wrap_material_source,
)
from navette.materials import MaterialSpec

WL = np.array([500.0, 600.0])

def test_dict_routes_by_grid():
    assert isinstance(wrap_material_source({"a": np.ones(2)}),
                      DictMaterialProvider)
    assert isinstance(wrap_material_source({"a": np.ones(2)}, wavelength=WL),
                      MaterialObjectProvider)
    assert isinstance(wrap_material_source({"a": np.ones(2)}, target_wavelength=WL),
                      MaterialObjectProvider)

def test_passthrough_and_errors():
    p = DictMaterialProvider({"a": np.ones(2)})
    assert wrap_material_source(p) is p
    with pytest.raises(TypeError):
        wrap_material_source(42)
    class FakeWeaver:
        def get_weaved(self, key):
            raise KeyError(key)
    with pytest.raises(ValueError, match="target_wavelength"):
        wrap_material_source(FakeWeaver())
    w = wrap_material_source(FakeWeaver(), target_wavelength=WL)
    assert isinstance(w, WeaverMaterialProvider) and w._native is None

def test_native_backed_serve():
    p = wrap_material_source({"L": MaterialSpec("Konstant", {"n": 1.45})},
                             wavelength=WL)
    assert p._native is not None
    assert np.array_equal(p.get_nk("L"), np.full(2, 1.45 + 0j))
    d = wrap_material_source({"L": np.full(2, 1.45 + 0j)})
    assert d._native is not None
    assert np.array_equal(d.get_nk("L"), np.full(2, 1.45 + 0j))
