# -*- coding: utf-8 -*-
"""Smoke tests for the pure-Python surface (no Rust build required)."""

from __future__ import annotations


def test_package_version() -> None:
  import navette

  assert navette.__version__ == "0.3.0"


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


def test_materials_imports_without_native() -> None:
  import navette.materials as m

  for name in ("Material", "Cauchy", "LorentzOscillator", "TaucLorentz"):
    assert hasattr(m, name), name


def test_data_bundled() -> None:
  from navette.data import data_path

  cmf = data_path("CIE", "cmf", "CIE_xyz_1931_2deg.json")
  assert cmf.is_file()


def test_native_wrappers_fail_helpfully() -> None:
  import pytest

  with pytest.raises(ImportError, match="maturin develop"):
    import navette.smatrix  # noqa: F401
