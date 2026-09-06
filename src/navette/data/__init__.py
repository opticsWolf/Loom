# -*- coding: utf-8 -*-
"""``navette.data`` — bundled reference spectra (CIE illuminants, CMFs, …)."""

from __future__ import annotations

from importlib.resources import files

from typing import Dict

import numpy as np


def data_path(*parts: str):
  """Return a traversable path to a bundled data file."""
  return files(__name__).joinpath(*parts)


def load_cie_table(*parts: str) -> Dict[str, np.ndarray]:
  """Read a bundled CIE JSON file and parse it natively.

  File I/O stays here (text in); the DataCite envelope
  (``data: {column: {values: [...]}}``) is parsed by the Rust core
  (``navette._color.parse_cie_tables``) — single canonical parser,
  bitwise twins in validation. Returns ``{column: float64 array}``.
  """
  from navette._color import parse_cie_tables as _parse
  text = files(__name__).joinpath(*parts).read_text(encoding="utf-8")
  try:
    return dict(_parse(text))
  except ValueError as exc:
    raise ValueError(f"{'/'.join(parts)}: {exc}") from exc
