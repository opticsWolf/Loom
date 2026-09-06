# -*- coding: utf-8 -*-
"""Graded films in the synthesis pipeline: homogenize-with-warning by
default, pinned background on request (Rust `from_design` twins)."""
import numpy as np
import pytest

from navette.synthesis.pipeline import stack_from_layers

WL = np.linspace(400., 800., 8)
TIO2 = {"model": "Konstant", "params": {"n": 2.35, "k": 0.01}}
SIO2 = {"model": "Konstant", "params": {"n": 1.46}}


def films(stack):
  return stack.to_dict()["films"]


def test_graded_homogenizes_with_warning():
  with pytest.warns(UserWarning, match="homogeneous"):
    st, _ = stack_from_layers(
      [(TIO2, 50.0)], WL, {}, names=["TiO2"],
      per_film_flags={"TiO2": {"inhomogen": True, "inh_delta": 0.2}})
  fs = films(st)
  assert len(fs) == 1  # single base-index row, profile dropped loudly
  assert fs[0]["thickness"] == pytest.approx(50.0)
  assert fs[0]["optimize"] and fs[0]["needle"]


def test_background_pins_profile_silently():
  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("error")  # background must NOT warn
    st, cmap = stack_from_layers(
      [(SIO2, 100.0), (TIO2, 50.0)], WL, {"TiO2": SIO2},
      names=["sub", "TiO2"],
      per_film_flags={"sub": {"inhomogen": True, "inh_delta": 0.2,
                                "optimize": False, "needle": False}})
  fs = films(st)
  subs = [f for f in fs if f["material"] == "sub"]
  assert len(subs) > 1  # profile expanded, not flattened
  assert sum(f["thickness"] for f in subs) == pytest.approx(100.0)
  assert all(not f["optimize"] and not f["needle"] for f in subs)
  design = [f for f in fs if f["material"] == "TiO2"]
  assert len(design) == 1 and design[0]["optimize"] and design[0]["needle"]
  # Pinned span survives merge (nk-keyed, not name-keyed).
  assert st.merge_adjacent() == 0
  assert len(films(st)) == len(fs)


def test_flat_stacks_stay_silent():
  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    st, _ = stack_from_layers([(TIO2, 50.0)], WL, {}, names=["TiO2"])
  assert len(films(st)) == 1
