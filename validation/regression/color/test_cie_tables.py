# -*- coding: utf-8 -*-
"""Native CIE DataCite parser: all 97 bundled files, bitwise vs json.load.

The Rust core owns the envelope (``data: {column: {values: [...]}}``);
Python only reads the file text. Every value must survive the round trip
bit-identically (both sides use correctly-rounded float parsing).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from navette._color import cie_xyz_triplet
from navette.data import load_cie_table

CIE = Path("src/navette/data/CIE")
FILES = sorted(CIE.rglob("*.json"))
assert len(FILES) == 97, f"expected 97 CIE files, found {len(FILES)}"

# CMF + chromaticity files whose columns form an XYZ triplet.
TRIPLET = {
    "cmf/CIE_cfb_stv_10deg.json", "cmf/CIE_cfb_stv_2deg.json",
    "cmf/CIE_xyz_1931_2deg.json", "cmf/CIE_xyz_1964_10deg.json",
    "cc/CIE_cc_1931_2deg.json", "cc/CIE_cc_1964_10deg.json",
    "cc/CIE_smb_cc_2deg.json",
}


def _oracle(path):
  raw = json.loads(path.read_text(encoding="utf-8"))["data"]
  return {k: np.asarray(v["values"], dtype=np.float64) for k, v in raw.items()}


@pytest.mark.parametrize("path", FILES, ids=[str(p) for p in FILES])
def test_file_bitwise_vs_json_load(path):
  rel = path.relative_to(CIE).as_posix()
  got = load_cie_table("CIE", *path.relative_to(CIE).parts)
  want = _oracle(path)
  assert set(got) == set(want), rel
  for name, arr in want.items():
    assert got[name].tobytes() == arr.tobytes(), f"{rel}:{name}"


@pytest.mark.parametrize("path", FILES, ids=[str(p) for p in FILES])
def test_lambda_monotonic_where_present(path):
  got = load_cie_table("CIE", *path.relative_to(CIE).parts)
  if "lambda" in got:
    wl = got["lambda"]
    assert np.all(np.diff(wl) > 0), str(path)


def test_xyz_triplet_resolves_on_xyz_files():
  for rel in sorted(TRIPLET - {"cc/CIE_smb_cc_2deg.json"}):
    text = (CIE / rel).read_text(encoding="utf-8")
    wl, x, y, z = cie_xyz_triplet(text)
    want = _oracle(CIE / rel)
    xname = [k for k in want if k != "lambda" and k[0].lower() == "x"][0]
    yname = [k for k in want if k != "lambda" and k[0].lower() == "y"][0]
    zname = [k for k in want if k != "lambda" and k[0].lower() == "z"][0]
    assert wl.tobytes() == want["lambda"].tobytes()
    assert x.tobytes() == want[xname].tobytes()
    assert y.tobytes() == want[yname].tobytes()
    assert z.tobytes() == want[zname].tobytes()


def test_macloed_boynton_is_no_triplet():
  text = (CIE / "cc/CIE_smb_cc_2deg.json").read_text(encoding="utf-8")
  with pytest.raises(ValueError, match="no XYZ triplet"):
    cie_xyz_triplet(text)


def test_non_triplet_file_refused_with_columns():
  text = (CIE / "sds/CIE_illum_FLs_1nm_FL1.json").read_text(encoding="utf-8")
  with pytest.raises(ValueError, match="FL1"):
    cie_xyz_triplet(text)


def test_lambda_free_file_has_no_triplet():
  text = (CIE / "lef/CIE_max_sle_mesopic.json").read_text(encoding="utf-8")
  got = load_cie_table("CIE", "lef", "CIE_max_sle_mesopic.json")
  assert set(got) == {"m", "K_m,mes;m"}
  with pytest.raises(ValueError, match="no 'lambda'"):
    cie_xyz_triplet(text)


def test_refusals_name_the_culprit():
  from navette._color import parse_cie_tables
  with pytest.raises(ValueError, match="invalid JSON"):
    parse_cie_tables("not json")
  with pytest.raises(ValueError, match="missing 'data'"):
    parse_cie_tables("{}")
  with pytest.raises(ValueError, match="'b'.*row 1"):
    parse_cie_tables('{"data": {"a": {"values": [1, 2]}, "b": {"values": [1, "x"]}}}')
  with pytest.raises(ValueError, match="expected 2"):
    parse_cie_tables('{"data": {"a": {"values": [1, 2]}, "b": {"values": [1, 2, 3]}}}')
