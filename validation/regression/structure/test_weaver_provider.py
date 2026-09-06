# -*- coding: utf-8 -*-
"""Twin of the Rust `weaver::tests`: native woven-grid provider behavior.

Same fragments, same frozen oracles (HEX — numpy array printing truncates
to 8 decimals, so only ``tobytes`` is oracle-grade). The Rust side asserts
``f64::from_bits`` of these same strings: bit-identity, both directions.
"""
import numpy as np
import pytest

from navette.structure.materials import WeaverMaterialProvider
from navette.structure.types import InterpolationSettings


class FakeWeaver:
  def __init__(self, d):
    self.d = dict(d)

  def __contains__(self, k):
    return tuple(k) in self.d

  def get_weaved(self, k):
    w, v = self.d[tuple(k)]
    return np.asarray(w, float), np.asarray(v, float)


T = [400., 500., 600., 700., 800.]
C = [400., 600., 800.]


def fake():
  return FakeWeaver({
    (0.0, "n", "A"): (C, [2.0, 2.3, 2.4]),
    (0.0, "k", "A"): (T, [0.01, 0.02, 0.03, 0.04, 0.05]),
    (0.0, "n", "B"): (T, [1.5] * 5),
  })


def target():
  return np.asarray(T, float)


def hex_re(arr):
  return np.ascontiguousarray(arr.real, dtype=np.float64).tobytes().hex()


def test_frozen_oracles():
  assert hex_re(WeaverMaterialProvider(
    fake(), target(), interp=InterpolationSettings(method="linear")).get_nk("A")) == \
    "000000000000004033333333333301406666666666660240cccccccccccc02403333333333330340"
  assert hex_re(WeaverMaterialProvider(
    fake(), target(), interp=InterpolationSettings(method="pchip")).get_nk("A")) == \
    "0000000000000040343333333373014066666666666602403333333333f302403333333333330340"
  assert hex_re(WeaverMaterialProvider(
    fake(), target(), interp=InterpolationSettings(method="makima")).get_nk("A")) == \
    "0000000000000040abaaaaaaaa6a014066666666666602403333333333f302403333333333330340"


def test_exact_and_missing_k_zeros():
  p = WeaverMaterialProvider(fake(), target())
  nk = p.get_nk("B")
  assert (nk.real == 1.5).all() and (nk.imag == 0.0).all()
  assert p.contains("A") and p.contains("B") and not p.contains("C")
  assert p.grid is not None


def test_missing_n_and_strict():
  p = WeaverMaterialProvider(fake(), target())
  with pytest.raises(KeyError):
    p.get_nk("C")
  s = WeaverMaterialProvider(fake(), target(), strict=True)
  with pytest.raises(ValueError, match="strict"):
    s.get_nk("A")
  # Absent k still defaults under strict (absence ≠ staleness).
  assert (s.get_nk("B").imag == 0.0).all()


def test_cache_target_and_invalidate():
  fw = fake()
  p = WeaverMaterialProvider(fw, target())
  assert p.is_exact("B") and not p.is_exact("A") and not p.is_exact("C")
  assert (p.get_nk("B").real == 1.5).all()
  fw.d[(0.0, "n", "B")] = (T, [9.0] * 5)
  assert (p.get_nk("B").real == 1.5).all()  # stale until invalidated
  p.invalidate_cache("B")
  assert (p.get_nk("B").real == 9.0).all()
  p.target_wavelength = target()  # same grid: no-op, cache survives
  assert (p.get_nk("B").real == 9.0).all()
  p.target_wavelength = np.array([400., 800.])  # new grid clears
  assert (p.get_nk("B").real == 9.0).all()
