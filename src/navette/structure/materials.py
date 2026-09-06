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
  """Structural contract for material index sources.

  Purpose: decouple the stack model (which thinks in material *names*) from
  material storage (arrays, specs, weavers). Expansion and validation only
  ever touch these two methods, so ANY object implementing them — including
  user-defined classes that never import this module — works as a provider
  (the protocol is runtime-checkable; ``wrap_material_source`` accepts
  structural matches via ``isinstance``).

  Contract: ``get_nk`` returns a 1-D complex ``n + ik`` array for the
  material; ``contains`` reports whether the name is servable. The array's
  wavelength grid is implicit — the provider does NOT know or check what
  grid it lives on. Grid agreement with the solver is the CALLER's
  responsibility, enforced by convention: providers expose their grid via
  a ``.grid`` attribute (``None`` when unknown — user duck-typed providers
  need not implement it); the ``solve_structure`` bridge value-compares it
  against the solver grid and refuses mismatches. Length-only agreement
  (equal length, different wavelengths) is rejected, not trusted.
  """
  def get_nk(self, material_name: str) -> np.ndarray: ...
  def contains(self, material_name: str) -> bool: ...


class DictMaterialProvider:
  """Passive shelf for precomputed nk arrays (and friends).

  Purpose: the zero-ceremony provider — hand it a ``{name: value}`` dict
  and layers resolve by name. Ideal when nk data already exists (measured
  tables, offline evaluations, baked ``_table`` specs from
  ``bake_materials``). Holds the caller's dict BY REFERENCE (no copy), so
  later insertions (e.g. baked materials) are visible immediately.

  Values may be:
  - ``np.ndarray`` — served as-is (no copy, no check). The array's grid
    is implicit: the provider cannot know or verify its wavelengths.
  - :class:`MaterialSpec` (or spec dict) — evaluated on ``wavelength=``
    at construction; raises ``AttributeError`` when the value is a spec
    but no grid was given.
  - objects exposing ``.nk`` — duck-typed escape hatch, served directly.

  Behavior: a passive shelf, not a registry. It never audits which entries
  are used (a library superset is the normal case; leftovers are inert)
  and never checks completeness — unknown names raise ``KeyError`` at
  expansion/validation time, downstream. Proper use: ensure every array
  value lives on the SAME grid you will solve on; the bridge only checks
  lengths, never wavelength values.
  """

  # Thin over the native DictProvider: serving, length checks and spec
  # evaluation live in Rust. `_dict` stays as the live shelf (snapshotter
  # + pour-back write it); writes go through to the native mirror, so
  # external raw-dict mutation is no longer visible — use refresh().
  __slots__ = ("_dict", "_wavelength", "_native")

  def __init__(
    self,
    mat_dict: Dict[str, Any],
    wavelength: Optional[np.ndarray] = None,
  ) -> None:
    from navette._structure import DictProvider as _NativeDict
    self._dict = mat_dict
    self._wavelength = (
      np.ascontiguousarray(np.asarray(wavelength, dtype=np.float64))
      if wavelength is not None
      else None
    )
    self._native = _NativeDict(
      {k: _native_value(v) for k, v in dict(mat_dict).items()}, self._wavelength)

  def refresh(self, mat_dict: Dict[str, Any],
              wavelength: Optional[np.ndarray] = None) -> None:
    """Atomically replace contents AND grid (the safe update path).

    Prefer this over in-place dict edits: both halves swap in one call, so
    "new data + new grid" can never be half-applied between statements.
    """
    self._dict = mat_dict
    self._wavelength = (
      np.ascontiguousarray(np.asarray(wavelength, dtype=np.float64))
      if wavelength is not None
      else None
    )
    self._native.refresh(
      {k: _native_value(v) for k, v in dict(mat_dict).items()}, self._wavelength)

  @property
  def grid(self) -> Optional[np.ndarray]:
    """Construction grid (``None`` when the provider was built gridless).

    When set, array values are length-checked against it at serve time and
    the ``solve_structure`` bridge value-compares it against the solver
    grid. When ``None``, only the solver-grid length assert applies (plus
    a warning at the bridge).

    Treat provider + grid as an immutable unit: when the arrays change,
    rebuild (or ``refresh``) the provider with the new grid — in-place dict
    edits of same-length/different-wavelength data are undetectable from
    inside (arrays carry no grid identity) and will pass every check while
    solving unphysically.
    """
    return self._wavelength

  def get_nk(self, material_name: str) -> np.ndarray:
    """Resolve one material to a complex n+ik array.

    Arrays pass through (length-checked when a grid is known); specs
    evaluate on the construction grid; ``.nk`` objects serve their
    attribute. Raises ``KeyError`` for unknown names, ``ValueError`` for
    arrays off the known grid, ``AttributeError`` for spec values without
    a grid or for unrecognized value types.
    """
    if material_name not in self._dict:
      raise KeyError(material_name)
    mat = self._dict[material_name]
    if isinstance(mat, (MaterialSpec, dict)) and self._wavelength is None:
      raise AttributeError(
        f"DictMaterialProvider: material '{material_name}' is a spec but "
        "no wavelength was given; pass wavelength= to evaluate specs."
      )
    if self._wavelength is None and not isinstance(mat, np.ndarray):
      nk = getattr(mat, "nk", None)
      if nk is not None:
        return nk
      raise AttributeError(
        f"DictMaterialProvider: material '{material_name}' is not an array, "
        "a MaterialSpec, or an object with .nk."
      )
    if not self._native.contains(material_name):
      self._native.insert(material_name, _native_value(self._dict[material_name]))
    grid = self._wavelength if self._wavelength is not None else np.zeros(0)
    return self._native.get_nk(material_name, grid)

  def contains(self, material_name: str) -> bool:
    """True when the dict holds ``material_name`` (no evaluation)."""
    return material_name in self._dict


class MaterialObjectProvider:
  """Spec library evaluated on one shared grid, with memoization.

  Purpose: the provider for parametric materials — hand it ``{name:
  MaterialSpec}`` plus the wavelength grid and every spec evaluates on
  that grid, cached per material. Ideal when materials are recipes
  (Cauchy/Lorentz/Tauc-Lorentz/…) rather than tables, and when the grid
  is fixed for a whole study. Proper use: construct once per grid;
  reassigning ``.wavelength`` clears the cache automatically, and
  ``invalidate(name?)`` drops entries after editing a spec in place.

  Behavior: grid is MANDATORY (constructor argument, not optional) and
  explicit — but still untethered: nothing compares it to the solver's
  grid, so point this provider at the grid you will solve on. Raw arrays
  are NOT valid values here (everything funnels through ``evaluate``);
  mix arrays in via :class:`DictMaterialProvider` instead. Like its
  sibling, a passive shelf: supersets are fine, unknown names raise at
  expansion, leftovers inert.
  """

  # Thin over the native SpecProvider: evaluation, memoization and the
  # grid live in Rust. `_dict` stays as the live shelf (snapshotter +
  # pour-back read it); serving syncs native on demand.
  __slots__ = ("_dict", "_wavelength", "_native", "_memo")

  def __init__(
    self, mat_dict: Dict[str, Any], wavelength: np.ndarray
  ) -> None:
    from navette._structure import SpecProvider as _NativeSpec
    self._dict = mat_dict
    self._wavelength = np.asarray(wavelength, dtype=np.float64)
    self._native = _NativeSpec(dict(mat_dict), self._wavelength)
    self._memo: Dict[str, np.ndarray] = {}

  @property
  def grid(self) -> np.ndarray:
    """Shared evaluation grid (always known for this provider)."""
    return self._wavelength

  @property
  def wavelength(self) -> np.ndarray:
    """Shared evaluation grid.

    Reassigning it clears the nk cache, but ONLY when the new grid differs
    (byte-compare on the raw values) — reassigning an identical grid is a
    no-op and keeps the cache.
    """
    return self._wavelength

  @wavelength.setter
  def wavelength(self, wl: np.ndarray) -> None:
    wl = np.asarray(wl, dtype=np.float64)
    self._wavelength = wl
    self._memo.clear()
    self._native.set_wavelength(wl)

  def get_nk(self, material_name: str) -> np.ndarray:
    """Spec evaluated on the shared grid, memoized per material.

    First call evaluates and caches; later calls serve the cache (same
    object). Raises ``KeyError`` for unknown names. After editing a spec
    dict in place, call ``invalidate`` — the cache cannot see the edit.
    """
    if material_name not in self._dict:
      raise KeyError(material_name)
    hit = self._memo.get(material_name)
    if hit is not None:
      return hit
    if not self._native.has(material_name):
      self._native.insert(material_name, self._dict[material_name])
    nk = self._native.get_nk(material_name, self._wavelength)
    self._memo[material_name] = nk
    return nk

  def contains(self, material_name: str) -> bool:
    """True when the dict holds ``material_name`` (no evaluation)."""
    return material_name in self._dict

  def invalidate(self, material_name: Optional[str] = None) -> None:
    """Drop cached evaluations (one material, or all when omitted).

    Required after mutating a spec dict in place — the cache keys on the
    material name only and cannot detect param edits.
    """
    if material_name is None:
      self._memo.clear()
      self._native.invalidate()
    elif material_name in self._dict:
      self._memo.pop(material_name, None)
      self._native.insert(material_name, self._dict[material_name])
    else:
      self._memo.pop(material_name, None)
      self._native.invalidate(material_name)
  


class WeaverMaterialProvider:
  """Live n/k curves woven from an :class:`OpticalWeaver` backend.

  Purpose: the provider for measured data — n/k fragments live in the
  weaver (keyed ``(prefix, label, polarisation)``, typically measured at
  one angle), and this provider resamples them onto ``target_wavelength``
  on demand. Ideal when ellipsometry spectra, not models, are the source
  of truth. Proper use: set ``target_wavelength`` to the grid you will
  solve on (output is always on that grid); match ``key_prefix`` to the
  measurement angle stored in the weaver; call ``invalidate_cache`` after
  re-weaving, since woven results are memoized and the cache cannot see
  backend changes.

  Behavior: per material, the ``n`` fragment is REQUIRED (absent →
  ``KeyError``), while a missing ``k`` fragment silently defaults to
  zeros — a pure-n entry becomes lossless rather than erroring. An
  exact-grid fast path skips interpolation when the woven fragment
  already sits on the target grid (shape AND values equal); otherwise
  ``UniInterpolator`` resamples per ``interp`` settings. The only
  provider whose contents can change under it — hence the explicit cache
  controls. Grid caveat as everywhere: nothing checks ``target_wavelength``
  against the solver's grid (the ``solve_structure`` bridge does).

  Strict vs lenient: ``strict=True`` refuses off-grid fragments instead of
  interpolating — use it when you OWN the weave (re-woven per study, so an
  off-grid fragment means a stale process, and interpolation would mask a
  workflow bug). Stay lenient when you BORROW the weave (archival/literature
  data on foreign grids, exploratory grids, coarser k-grids), where
  resampling is the legitimate bridge. Freshness workflow with a re-woven
  backend: assign ``target_wavelength`` (cache auto-clears), assert
  ``is_exact`` over your materials, then solve — any staleness raises
  instead of interpolating.
  """
  __slots__ = (
    "_weaver", "_target_wl", "_cache",
    "_key_prefix", "_n_label", "_k_label",
    "_interp_settings", "_strict", "_native",
  )

  @property
  def grid(self) -> np.ndarray:
    """Target grid the weaves are resampled onto (always known)."""
    return self._target_wl

  # Native-backed path delegates everything to the Rust WeaverProvider
  # (same semantics, same errors); duck-typed backends keep the Python
  # adapter below (foreign-object probing is presentation, not physics).
  def __init__(
    self,
    weaver: Any,
    target_wavelength: np.ndarray,
    key_prefix: float = 0.0,
    n_label: str = "n",
    k_label: str = "k",
    interp: InterpolationSettings = InterpolationSettings(),
    strict: bool = False,
  ) -> None:
    self._weaver = weaver
    self._target_wl = np.asarray(target_wavelength, dtype=np.float64)
    self._cache: Dict[str, np.ndarray] = {}
    self._key_prefix = key_prefix
    self._n_label = n_label
    self._k_label = k_label
    self._interp_settings = interp
    self._strict = bool(strict)
    self._native = None
    try:
      from navette._spectralweave import OpticalWeaver as _NativeWeaver
      from navette._structure import WeaverProvider as _NativeProvider
    except ImportError:
      _NativeWeaver = None  # type: ignore[assignment]
    if _NativeWeaver is not None and isinstance(weaver, _NativeWeaver):
      self._native = _NativeProvider(
        weaver, self._target_wl, key_prefix, n_label, k_label,
        interp.method, interp.robust, interp.floater_hormann_d, bool(strict))

  @property
  def strict(self) -> bool:
    """When True, present-but-off-grid fragments raise instead of
    interpolating (absent k still defaults to zeros — absence is not
    staleness). Toggle freely; cached exact-grid values are unaffected."""
    return self._strict

  @strict.setter
  def strict(self, value: bool) -> None:
    self._strict = bool(value)
    if self._native is not None:
      self._native.strict = bool(value)

  @property
  def target_wavelength(self) -> np.ndarray:
    """Grid the weaves are served on.

    Reassigning a different grid clears the cache (identical grids are a
    no-op). After re-weaving the backend onto a new grid, assign it here
    and verify with ``is_exact`` — in ``strict`` mode the next ``get_nk``
    then proves exactness instead of assuming it.
    """
    return self._target_wl

  @target_wavelength.setter
  def target_wavelength(self, wl: np.ndarray) -> None:
    wl = np.asarray(wl, dtype=np.float64)
    if wl.tobytes() != self._target_wl.tobytes():
      self._target_wl = wl
      self._cache.clear()
      if self._native is not None:
        self._native.set_target(wl)

  def is_exact(self, material_name: str) -> bool:
    """True when the weave sits exactly on the target grid (no fallback).

    Probes without serving: checks the ``n`` fragment's grid (shape AND
    values) against the target. False for unknown materials. Use it to
    verify freshness after re-weaving — in ``strict`` mode this is what
    ``get_nk`` enforces per fragment (n AND k).
    """
    if self._native is not None:
      return bool(self._native.is_exact(material_name))
    n_key = (self._key_prefix, self._n_label, material_name)
    if n_key not in self._weaver:
      return False
    src_wl, src_data = self._weaver.get_weaved(n_key)
    return (src_wl.shape == self._target_wl.shape
            and np.array_equal(src_wl, self._target_wl))

  def get_nk(self, material_name: str) -> np.ndarray:
    """Woven n/k interpolated onto the target grid, memoized per material.

    Raises ``KeyError`` when no ``n`` fragment exists for the name; a
    missing ``k`` fragment becomes zeros (lossless by default, not an
    error — check the weaver when loss is expected but absent). In
    ``strict`` mode any present-but-off-grid fragment raises ``ValueError``
    instead of interpolating (absent k still defaults to zeros).
    """
    cached = self._cache.get(material_name)
    if cached is not None:
      return cached
    if self._native is not None:
      nk = self._native.get_nk(material_name, self._target_wl)
      self._cache[material_name] = nk
      return nk
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
    """True when an ``n`` fragment exists for ``material_name``.

    Note: only the ``n`` key is probed — a material with n-but-no-k
    reports True and serves zeros for k (see ``get_nk``).
    """
    if self._native is not None:
      return bool(self._native.contains(material_name))
    n_key = (self._key_prefix, self._n_label, material_name)
    return n_key in self._weaver

  def invalidate_cache(self, material_name: Optional[str] = None) -> None:
    """Drop interpolated caches (one material, or all when omitted).

    Required after re-weaving the backend — the cache cannot see weaver
    edits, and stale n/k would otherwise persist silently.
    """
    if material_name is None:
      self._cache.clear()
    else:
      self._cache.pop(material_name, None)
    if self._native is not None:
      self._native.invalidate(material_name)
  

  def _fetch_and_interpolate(self, key: tuple) -> Optional[np.ndarray]:
    """Woven fragment resampled to the target grid.

    Returns None when the key is unknown or the fragment is empty;
    serves directly on exact-grid match, else interpolates per settings
    — unless ``strict``, which raises on any grid mismatch instead.
    """
    if key not in self._weaver:
      return None
    src_wl, src_data = self._weaver.get_weaved(key)
    if src_wl.size == 0:
      return None
    if (src_wl.shape == self._target_wl.shape and np.array_equal(src_wl, self._target_wl)):
      return src_data.astype(np.float64)
    if self._strict:
      raise ValueError(
        f"WeaverMaterialProvider(strict): weave {key} is not on the target "
        f"grid ({src_wl.shape[0]} vs {self._target_wl.shape[0]} points); "
        f"re-weave onto the target grid or disable strict mode.")
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


def _native_value(value: Any) -> Any:
  """Coerce list/tuple shelves to arrays at the native sync boundary."""
  if isinstance(value, (list, tuple)):
    return np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
  return value


def wrap_material_source(source: Any, **kwargs: Any) -> MaterialProvider:
  """Coerce dicts, spec maps and weavers into a :class:`MaterialProvider`.

  Dispatch: providers (structural match) pass through untouched; dicts
  become :class:`MaterialObjectProvider` when a grid is given
  (``wavelength=`` or ``target_wavelength=``) else
  :class:`DictMaterialProvider`; weaver-likes (``get_weaved`` attribute)
  become :class:`WeaverMaterialProvider` and REQUIRE ``target_wavelength=``.
  Anything else raises ``TypeError``.

  The grid kwargs choose the provider's evaluation grid — point them at
  the grid you will solve on (see the provider docstrings: grid VALUE
  agreement is never checked downstream, only array length).
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
