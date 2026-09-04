# -*- coding: utf-8 -*-
"""Material providers: name → nk arrays for stacks and architects."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

from navette.materials import MaterialSpec, evaluate

from .types import InterpolationSettings

try:
  from navette.interpolate import UniInterpolator as UniSpline
except ImportError:  # pragma: no cover - native not built; only needed for table resampling
  UniSpline = None  # type: ignore[assignment,misc]


@runtime_checkable
class MaterialProvider(Protocol):
  """Material index source: ``get_nk`` returns n+ik arrays, ``contains`` tests names."""
  def get_nk(self, material_name: str) -> np.ndarray: ...
  def contains(self, material_name: str) -> bool: ...


class DictMaterialProvider:
  """Wraps a dict of precomputed nk arrays (or specs).

  Values may be arrays, :class:`MaterialSpec` (evaluated when a wavelength
  grid was given), or objects exposing ``.nk``.
  """

  __slots__ = ("_dict", "_wavelength")

  def __init__(
    self,
    mat_dict: Dict[str, Any],
    wavelength: Optional[np.ndarray] = None,
  ) -> None:
    self._dict = mat_dict
    self._wavelength = (
      np.ascontiguousarray(np.asarray(wavelength, dtype=np.float64))
      if wavelength is not None
      else None
    )

  def get_nk(self, material_name: str) -> np.ndarray:
    """Resolve one material to an n+ik array (KeyError when unknown)."""
    mat = self._dict[material_name]
    if isinstance(mat, np.ndarray):
      return mat
    if isinstance(mat, (MaterialSpec, dict)) and self._wavelength is None:
      raise AttributeError(
        f"DictMaterialProvider: material '{material_name}' is a spec but "
        "no wavelength was given; pass wavelength= to evaluate specs."
      )
    if isinstance(mat, (MaterialSpec, dict)):
      return evaluate(mat, self._wavelength)
    nk = getattr(mat, "nk", None)
    if nk is not None:
      return nk
    raise AttributeError(
      f"DictMaterialProvider: material '{material_name}' is not an array, "
      "a MaterialSpec, or an object with .nk."
    )

  def contains(self, material_name: str) -> bool:
    """True when the dict holds ``material_name``."""
    return material_name in self._dict


class MaterialObjectProvider:
  """Wraps ``{name: MaterialSpec}`` evaluated on a shared wavelength grid."""

  __slots__ = ("_dict", "_wavelength", "_wl_sig", "_cache")

  def __init__(
    self, mat_dict: Dict[str, Any], wavelength: np.ndarray
  ) -> None:
    self._dict = mat_dict
    self._wavelength = np.asarray(wavelength, dtype=np.float64)
    self._wl_sig: bytes = self._wavelength.tobytes()
    self._cache: Dict[str, np.ndarray] = {}

  @property
  def wavelength(self) -> np.ndarray:
    """Shared evaluation grid; reassigning it clears the nk cache."""
    return self._wavelength

  @wavelength.setter
  def wavelength(self, wl: np.ndarray) -> None:
    wl = np.asarray(wl, dtype=np.float64)
    sig = wl.tobytes()
    if sig != self._wl_sig:
      self._wavelength = wl
      self._wl_sig = sig
      self._cache.clear()

  def get_nk(self, material_name: str) -> np.ndarray:
    """Cached spec evaluation on the shared grid."""
    cached = self._cache.get(material_name)
    if cached is not None:
      return cached
    nk = evaluate(self._dict[material_name], self._wavelength)
    self._cache[material_name] = nk
    return nk

  def contains(self, material_name: str) -> bool:
    """True when the dict holds ``material_name``."""
    return material_name in self._dict

  def invalidate(self, material_name: Optional[str] = None) -> None:
    """Drop cached evaluations (one material, or all when omitted)."""
    if material_name is None:
      self._cache.clear()
    else:
      self._cache.pop(material_name, None)


class WeaverMaterialProvider:
  """Serves n/k curves woven from an :class:`OpticalWeaver` backend.

  Looks up ``(angle, n/k-label, polarisation)`` fragments on a target grid
  and interpolates them to the solver wavelengths.
  """
  __slots__ = (
    "_weaver", "_target_wl", "_cache",
    "_key_prefix", "_n_label", "_k_label",
    "_interp_settings",
  )

  def __init__(
    self,
    weaver: Any,
    target_wavelength: np.ndarray,
    key_prefix: float = 0.0,
    n_label: str = "n",
    k_label: str = "k",
    interp: InterpolationSettings = InterpolationSettings(),
  ) -> None:
    self._weaver = weaver
    self._target_wl = np.asarray(target_wavelength, dtype=np.float64)
    self._cache: Dict[str, np.ndarray] = {}
    self._key_prefix = key_prefix
    self._n_label = n_label
    self._k_label = k_label
    self._interp_settings = interp

  def get_nk(self, material_name: str) -> np.ndarray:
    """Cached n/k weave interpolated to the target grid (k defaults to 0)."""
    cached = self._cache.get(material_name)
    if cached is not None:
      return cached
    n_key = (self._key_prefix, self._n_label, material_name)
    k_key = (self._key_prefix, self._k_label, material_name)
    n_arr = self._fetch_and_interpolate(n_key)
    if n_arr is None:
      raise KeyError(f"WeaverMaterialProvider: material '{material_name}' not found.")
    k_arr = self._fetch_and_interpolate(k_key)
    if k_arr is None:
      k_arr = np.zeros_like(n_arr)
    nk = n_arr + 1j * k_arr
    self._cache[material_name] = nk
    return nk

  def contains(self, material_name: str) -> bool:
    """True when an n-fragment exists for ``material_name``."""
    n_key = (self._key_prefix, self._n_label, material_name)
    return n_key in self._weaver

  def invalidate_cache(self, material_name: Optional[str] = None) -> None:
    """Drop interpolated caches (one material, or all when omitted)."""
    if material_name is None:
      self._cache.clear()
    else:
      self._cache.pop(material_name, None)

  def _fetch_and_interpolate(self, key: tuple) -> Optional[np.ndarray]:
    """Woven fragment resampled to the target grid (None when key unknown)."""
    if key not in self._weaver:
      return None
    src_wl, src_data = self._weaver.get_weaved(key)
    if src_wl.size == 0:
      return None
    if (src_wl.shape == self._target_wl.shape and np.array_equal(src_wl, self._target_wl)):
      return src_data.astype(np.float64)
    if UniSpline is None:  # pragma: no cover
      raise ImportError(
        "Table resampling needs the compiled `navette._interpolate` extension. "
        "Build it with: maturin develop  # from the repo root"
      )
    spline = UniSpline(
      src_wl, src_data,
      method=self._interp_settings.method,
      robust=self._interp_settings.robust,
      d=self._interp_settings.floater_hormann_d,
    )
    return spline(self._target_wl).astype(np.float64)


def wrap_material_source(source: Any, **kwargs: Any) -> MaterialProvider:
  """Coerce dicts, spec maps and weavers into a :class:`MaterialProvider`.

  Pass ``wavelength=`` for spec dicts, ``target_wavelength=`` for weavers.
  """
  if isinstance(source, MaterialProvider):
    return source
  if isinstance(source, dict):
    wl = kwargs.get("wavelength") or kwargs.get("target_wavelength")
    if wl is not None:
      return MaterialObjectProvider(source, wl)
    return DictMaterialProvider(source)
  if hasattr(source, "get_weaved"):
    target_wl = kwargs.get("target_wavelength")
    if target_wl is None:
      raise ValueError("wrap_material_source: OpticalWeaver detected but no 'target_wavelength' kwarg provided.")
    return WeaverMaterialProvider(source, target_wl, **{
      k: v for k, v in kwargs.items() if k != "target_wavelength"
    })
  raise TypeError(f"wrap_material_source: unsupported type {type(source).__name__}")
