# -*- coding: utf-8 -*-
"""Smoke tests for the pure-Python surface (no Rust build required)."""

from __future__ import annotations


def test_package_version() -> None:
  import navette

  assert navette.__version__ == "0.4.19"


def test_structure_imports() -> None:
  from navette.structure import (
    Navette_Structure,
    Navette_Architect,
    Layer,
    Group,
    SolverArrays,
  )

  s = Navette_Structure()
  assert s.layer_list == []


def test_config_imports() -> None:
  import navette.config as cfg

  for name in ("load_yaml", "save_yaml", "MaterialDefinition", "LayerConfig"):
    assert hasattr(cfg, name), name


def test_materials_evaluate_with_native() -> None:
  import numpy as np

  import navette.materials as m

  for name in ("MaterialSpec", "evaluate", "MODELS"):
    assert hasattr(m, name), name
  assert "Cauchy" in m.MODELS and "Lorentz" in m.MODELS

  spec = m.MaterialSpec(model="Cauchy", params={"A": 1.5, "B": 0.004, "C": 0.0})
  assert spec.model == "Cauchy"
  nk = m.evaluate(spec, np.array([550.0]))
  assert nk.shape == (1,)
  assert abs(nk[0].real - (1.5 + 0.004 / 0.55**2)) < 1e-9
  assert nk[0].imag == 0.0


def test_data_bundled() -> None:
  from navette.data import data_path

  cmf = data_path("CIE", "cmf", "CIE_xyz_1931_2deg.json")
  assert cmf.is_file()


def test_native_wrappers_import() -> None:
  import navette._color  # noqa: F401
  import navette._interpolate  # noqa: F401
  import navette._smatrix  # noqa: F401
  import navette._spectralweave  # noqa: F401
  import navette.smatrix  # noqa: F401

  from navette.interpolate import UniInterpolator
  from navette.spectralweave import OpticalWeaver, TargetWeaver

  assert UniInterpolator is not None
  assert OpticalWeaver is not None and TargetWeaver is not None
