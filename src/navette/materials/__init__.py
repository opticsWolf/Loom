# -*- coding: utf-8 -*-
"""
``navette.materials`` — optical dispersion models over the native core.

Thin layer: :class:`MaterialSpec` values (``model`` + plain ``params`` dict)
are evaluated by :func:`evaluate`, which dispatches straight to the compiled
``navette.materials._native`` extension
(``crates/navette-materials-py``)::

    maturin develop -m crates/navette-materials-py/Cargo.toml

No Python kernels remain; every model (including Konstant/Table) is native.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
  from . import _native
except ImportError as exc:  # pragma: no cover - native not built yet
  raise ImportError(
    "Could not import the compiled `navette.materials._native` extension. "
    "Build the Rust crate so it is importable, then retry:\n"
    "    maturin develop -m crates/navette-materials-py/Cargo.toml"
  ) from exc

__all__ = [
  "MaterialSpec",
  "evaluate",
  "MODELS",
]

_BUILD_HINT = "maturin develop -m crates/navette-materials-py/Cargo.toml"

#: All models handled by :func:`evaluate`.
MODELS: Tuple[str, ...] = (
  "Konstant",
  "Table",
  "Cauchy",
  "CauchyUrbach",
  "Sellmeier",
  "SellmeierUrbach",
  "Lorentz",
  "Drude",
  "DrudeLorentz",
  "CodyLorentz",
  "ForouhiBloomerSingle",
  "ForouhiBloomerMulti",
  "ForouhiBloomerMetal",
  "ForouhiBloomerMetal2021",
  "TaucLorentz",
  "UBF",
  "Bruggeman",
  "MaxwellGarnett",
  "Looyenga",
  "Lichtenecker",
  "MoriTanaka",
  "PowerLaw",
  "Roughness",
)


@dataclass(frozen=True)
class MaterialSpec:
  """A material as data: ``model`` name plus plain ``params``.

  Oscillator conventions (lists of tuples, matching the native layouts):

  - Lorentz / DrudeLorentz ``osc``: ``(E0, Gamma, f0)``
  - CodyLorentz ``osc``: ``(E0, A, Gamma, Ep)``
  - TaucLorentz ``osc``: ``(A, E0, C)``
  - UBF ``osc``: dicts ``{Eg, Ec, Eu, A, Gamma, gamma}``
  - Forouhi-Bloomer ``ib``: ``(Eg, A, B, C)``; metal ``fe``: ``(A, B, C)``
  - EMA composites nest specs: ``host`` / ``inclusion`` are :class:`MaterialSpec`
  - Table: ``n_data`` / ``k_data`` as ``(wavelengths, values)`` pairs
    (linear interpolation; resample exotic tables offline)
  """

  model: str
  params: Dict[str, Any] = field(default_factory=dict)


def _as_osc_array(
  osc: Sequence[Sequence[float]] | np.ndarray, width: int, what: str
) -> np.ndarray:
  arr = np.asarray(osc, dtype=np.float64)
  if arr.ndim != 2 or arr.shape[1] != width:
    raise ValueError(f"{what} needs an (N, {width}) array, got shape {arr.shape}")
  if arr.shape[0] == 0:
    raise ValueError(f"{what} needs at least one oscillator")
  return np.ascontiguousarray(arr)


def _ubf_array(oscs: Sequence[Mapping[str, float]]) -> np.ndarray:
  rows = []
  for i, o in enumerate(oscs):
    try:
      eu = float(o["Eu"])
      rows.append([
        float(o["Eg"]),
        float(o["Ec"]),
        1.0 / eu,  # β = 1/Eu kernel convention
        float(o["A"]),
        float(o["Gamma"]),
        float(o["gamma"]),
      ])
    except KeyError as exc:
      raise ValueError(f"UBF oscillator {i} missing key {exc}") from exc
    if eu <= 0:
      raise ValueError(f"UBF oscillator {i}: Eu must be > 0")
  return np.ascontiguousarray(rows, dtype=np.float64)


def evaluate(
  spec: MaterialSpec | Mapping[str, Any],
  wavelength_nm: np.ndarray,
) -> np.ndarray:
  """Evaluate a material spec to complex ``n + ik`` on ``wavelength_nm``."""
  if isinstance(spec, Mapping):
    spec = MaterialSpec(model=spec["model"], params=dict(spec.get("params", {})))
  wl = np.ascontiguousarray(np.asarray(wavelength_nm, dtype=np.float64))
  if wl.ndim != 1 or wl.size == 0:
    raise ValueError("wavelength_nm must be a non-empty 1-D array")
  p = spec.params
  get = p.get

  def req(*names: str) -> List[float]:
    missing = [n for n in names if n not in p]
    if missing:
      raise ValueError(f"{spec.model} missing params: {missing}")
    return [float(p[n]) for n in names]

  model = spec.model
  if model == "Konstant":
    n, k = req("n"), [float(p.get("k", 0.0))]
    return _native.konstant_nk(wl, n[0], k[0])
  if model == "Table":
    n_data = p.get("n_data")
    if n_data is None:
      raise ValueError("Table missing 'n_data' ((wavelengths, values) pair)")
    gw, nv = (np.asarray(a, dtype=np.float64) for a in n_data)
    k_data = p.get("k_data")
    kv = None
    if k_data is not None:
      _, kv = (np.asarray(a, dtype=np.float64) for a in k_data)
      kv = np.ascontiguousarray(kv)
    for key in ("interpolation_type_n", "interpolation_type_k"):
      if key in p and p[key] != "linear":
        raise ValueError(
          f"Table {key}={p[key]!r} unsupported: native core is linear-only, "
          "resample the table offline"
        )
    return _native.table_nk(
      wl, np.ascontiguousarray(gw), np.ascontiguousarray(nv), kv,
      float(p.get("n_factor", 1.0)), float(p.get("k_factor", 1.0)),
    )
  if model == "Cauchy":
    a, b, c = req("A", "B", "C")
    return _native.cauchy_nk(wl, a, b, c)
  if model == "CauchyUrbach":
    a, b, c, alpha0, eu, lg = req("A", "B", "C", "alpha0", "Eu", "lambda_g")
    return _native.cauchy_urbach_nk(wl, a, b, c, alpha0, eu, lg)
  if model == "Sellmeier":
    b1, c1, b2, c2, b3, c3 = req("B1", "C1", "B2", "C2", "B3", "C3")
    return _native.sellmeier_nk(wl, b1, c1, b2, c2, b3, c3)
  if model == "SellmeierUrbach":
    vals = req("B1", "C1", "B2", "C2", "B3", "C3", "alpha0", "Eu", "lambda_g")
    return _native.sellmeier_urbach_nk(wl, *vals)
  if model == "Lorentz":
    osc = _as_osc_array(p.get("osc", []), 3, "Lorentz osc")
    return _native.lorentz_nk(wl, osc, float(get("epsilon_inf", 1.0)))
  if model == "Drude":
    wp, gamma, eps = req("omega_p", "gamma", "epsilon_inf")
    return _native.drude_nk(wl, wp, gamma, eps)
  if model == "DrudeLorentz":
    wp, gamma, eps = (
      float(p.get("omega_p", p.get("wp", 0.0))),
      float(p.get("gamma_drude", p.get("gamma", 0.0))),
      float(get("epsilon_inf", 1.0)),
    )
    osc = _as_osc_array(p.get("osc", []), 3, "DrudeLorentz osc")
    return _native.drude_lorentz_nk(wl, wp, gamma, eps, osc)
  if model == "CodyLorentz":
    eg, et, eu = req("Eg", "Et", "Eu")
    osc = _as_osc_array(p.get("osc", []), 4, "CodyLorentz osc")
    return _native.cody_lorentz_nk(wl, eg, et, eu, osc, float(get("epsilon_inf", 1.0)))
  if model in ("ForouhiBloomerSingle", "ForouhiBloomerMulti"):
    ib = _as_osc_array(p.get("ib", []), 4, "ForouhiBloomer ib")
    return _native.fb_interband_nk(wl, float(get("n_inf", 1.0)), ib)
  if model in ("ForouhiBloomerMetal", "ForouhiBloomerMetal2021"):
    ib = _as_osc_array(p.get("ib", []), 4, "ForouhiBloomer ib")
    fe = np.ascontiguousarray(
      [float(p["A_fe"]), float(p["B_fe"]), float(p["C_fe"])], dtype=np.float64
    ) if "A_fe" in p else np.ascontiguousarray(
      p.get("fe", []), dtype=np.float64
    )
    if fe.shape != (3,):
      raise ValueError("ForouhiBloomerMetal needs fe (A_fe, B_fe, C_fe)")
    return _native.fb_metal_nk(wl, float(get("n_inf", 1.0)), fe, ib)
  if model == "TaucLorentz":
    eg = float(req("Eg")[0])
    osc = _as_osc_array(p.get("osc", []), 3, "TaucLorentz osc")
    return _native.tauc_lorentz_nk(wl, eg, osc, float(get("epsilon_inf", 1.0)))
  if model == "UBF":
    osc = _ubf_array(p.get("osc", []))
    return _native.ubf_nk(wl, osc, float(get("epsilon_inf", 1.0)))
  if model in (
    "Bruggeman", "MaxwellGarnett", "Looyenga", "Lichtenecker",
    "MoriTanaka", "PowerLaw",
  ):
    host = _eval_nested(p.get("host"), wl, "host")
    incl = _eval_nested(p.get("inclusion"), wl, "inclusion")
    f = float(req("fraction")[0])
    if model == "Bruggeman":
      return _native.eps_to_nk(_native.ema_bruggeman(incl, host, f))
    if model == "MaxwellGarnett":
      return _native.eps_to_nk(_native.ema_maxwell_garnett(incl, host, f))
    if model == "Looyenga":
      return _native.eps_to_nk(_native.ema_looyenga(incl, host, f))
    if model == "Lichtenecker":
      return _native.eps_to_nk(_native.ema_lichtenecker(incl, host, f))
    if model == "MoriTanaka":
      return _native.eps_to_nk(
        _native.ema_mori_tanaka(incl, host, f, float(p.get("L", 1.0 / 3.0)))
      )
    return _native.eps_to_nk(
      _native.ema_power_law(incl, host, f, float(p.get("alpha", 0.5)))
    )
  if model == "Roughness":
    bottom = _eval_nested(p.get("bottom"), wl, "bottom")
    top = _eval_nested(p.get("top"), wl, "top")
    return _native.eps_to_nk(_native.ema_roughness(bottom, top))
  raise ValueError(f"Unknown material model {model!r}. Available: {MODELS}")


def _eval_nested(
  spec: MaterialSpec | Mapping[str, Any] | None, wl: np.ndarray, what: str
) -> np.ndarray:
  if spec is None:
    raise ValueError(f"Composite material missing '{what}' spec")
  return evaluate(spec, wl)
