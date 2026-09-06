"""Native-backed weaver path twins the duck-backend adapter, bitwise."""
import numpy as np
import pytest

from navette.structure.materials import WeaverMaterialProvider
from navette.structure.types import InterpolationSettings
from navette._spectralweave import OpticalWeaver

T = [400., 500., 600., 700., 800.]
C = [400., 600., 800.]

FRAGS = {
    (0.0, "n", "A"): (C, [2.0, 2.3, 2.4]),
    (0.0, "k", "A"): (T, [0.01, 0.02, 0.03, 0.04, 0.05]),
    (0.0, "n", "B"): (T, [1.5] * 5),
}

class FakeWeaver:
    def __init__(self, d):
        self.d = dict(d)

    def __contains__(self, k):
        return tuple(k) in self.d

    def get_weaved(self, k):
        w, v = self.d[tuple(k)]
        return np.asarray(w, float), np.asarray(v, float)


def native_backend():
    wv = OpticalWeaver(128)
    for (p, lab, mat), (w, v) in FRAGS.items():
        wv.set_data((p, lab, mat), np.asarray(v, float), np.asarray(w, float))
    return wv


def target():
    return np.asarray(T, float)


def hex_re(arr):
    return np.ascontiguousarray(arr.real, dtype=np.float64).tobytes().hex()


def test_native_matches_duck_bitwise():
    duck = WeaverMaterialProvider(FakeWeaver(FRAGS), target())
    nat = WeaverMaterialProvider(native_backend(), target())
    assert nat._native is not None
    for name in ("A", "B"):
        assert hex_re(nat.get_nk(name)) == hex_re(duck.get_nk(name))
    assert nat.contains("A") and not nat.contains("C")
    assert nat.is_exact("B") and not nat.is_exact("A")


def test_native_strict_and_cache():
    nat = WeaverMaterialProvider(native_backend(), target(), strict=True)
    with pytest.raises(ValueError, match="strict"):
        nat.get_nk("A")
    assert nat.is_exact("B")
    a = nat.get_nk("B")
    assert nat.get_nk("B") is a
    nat.invalidate_cache("B")
    assert nat.get_nk("B") is not a
    nat.strict = False
    assert nat._native.strict is False
    nat.target_wavelength = np.asarray(C, float)
    assert nat.is_exact("A")
