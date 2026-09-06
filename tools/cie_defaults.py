"""Shared extraction of the embedded color defaults (stdlib only).

Single source of column knowledge for `extract_cie_defaults.py` (writes
`rust/navette/data/*.json`) and `check_cie_sync.py` (re-extracts and
byte-compares). No `navette` import — both tools must run without a
built extension (CI sync job has no Rust toolchain output).
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CIE = ROOT / "src" / "navette" / "data" / "CIE"
DATA = ROOT / "rust" / "navette" / "data"


def _values(path: pathlib.Path, column: str) -> list:
    doc = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = doc["data"][column]["values"]
    except KeyError:
        raise KeyError(f"{path.name}: envelope has no data[{column!r}].values")
    if not values:
        raise ValueError(f"{path.name}: data[{column!r}].values is empty")
    return list(values)


def extract_cmf() -> dict:
    """1931 2° CMF: 471 points, 360-830 nm, rectangular minimal format."""
    src = CIE / "cmf" / "CIE_xyz_1931_2deg.json"
    out = {
        "wavelengths": _values(src, "lambda"),
        "x": _values(src, "x_bar(lambda)"),
        "y": _values(src, "y_bar(lambda)"),
        "z": _values(src, "z_bar(lambda)"),
    }
    n = len(out["wavelengths"])
    assert n == 471, f"CMF grid changed: {n} points (expected 471)"
    for key in ("x", "y", "z"):
        assert len(out[key]) == n, f"CMF {key}: length drift"
    return out


def extract_d65() -> dict:
    """CIE standard illuminant D65 relative SPD, native source grid."""
    src = CIE / "sds" / "CIE_std_illum_D65_S_D65.json"
    out = {
        "wavelengths": _values(src, "lambda"),
        "values": _values(src, "S_D65(lambda)"),
    }
    assert len(out["wavelengths"]) == len(out["values"]), "D65 length drift"
    return out


def dump_canonical(payload: dict) -> str:
    """Deterministic serialization: sorted keys, shortest round-trip
    floats, single trailing newline. Both tools use this so the sync
    check is a pure byte-compare."""
    return json.dumps(payload, sort_keys=True) + "\n"


TARGETS = (
    ("cmf_1931_2deg.json", extract_cmf),
    ("illum_d65.json", extract_d65),
)
