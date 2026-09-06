"""MaterialObjectProvider thinned onto the native SpecProvider: twins."""
import numpy as np
import pytest

from navette.structure.materials import MaterialObjectProvider
from navette.materials import MaterialSpec, evaluate

WL = np.array([500.0, 600.0, 700.0])

def _lib():
    return {
        "L": MaterialSpec("Konstant", {"n": 1.45}),
        "H": MaterialSpec("Table", {
            "n_data": (np.array([400., 500., 600., 700., 800.]),
                       np.array([2.2, 2.15, 2.1, 2.08, 2.05]))}),
    }

def test_serve_matches_direct_evaluate_bitwise():
    prov = MaterialObjectProvider(dict(_lib()), WL)
    for name, spec in _lib().items():
        want = np.ascontiguousarray(evaluate(spec, WL))
        got = prov.get_nk(name)
        assert got.shape == want.shape
        assert np.array_equal(got, want)

def test_memoized_identity_and_invalidate():
    lib = _lib()
    prov = MaterialObjectProvider(lib, WL)
    a = prov.get_nk("L")
    assert prov.get_nk("L") is a
    lib["L"] = MaterialSpec("Konstant", {"n": 1.50})
    prov.invalidate("L")
    b = prov.get_nk("L")
    assert b is not a
    assert np.array_equal(b, np.ascontiguousarray(evaluate(lib["L"], WL)))

def test_wavelength_setter_resets_cache():
    prov = MaterialObjectProvider(dict(_lib()), WL)
    prov.get_nk("L")
    wl2 = np.array([500.0, 800.0])
    prov.wavelength = wl2
    got = prov.get_nk("L")
    assert got.shape == (2,)
    prov.wavelength = WL
    assert prov.get_nk("L").shape == (3,)

def test_unknown_and_keyerror():
    prov = MaterialObjectProvider(dict(_lib()), WL)
    assert not prov.contains("X")
    with pytest.raises(KeyError):
        prov.get_nk("X")
