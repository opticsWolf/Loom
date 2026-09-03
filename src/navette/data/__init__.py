# -*- coding: utf-8 -*-
"""``navette.data`` — bundled reference spectra (CIE illuminants, CMFs, …)."""

from __future__ import annotations

from importlib.resources import files


def data_path(*parts: str):
  """Return a traversable path to a bundled data file."""
  return files(__name__).joinpath(*parts)
