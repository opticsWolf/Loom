# -*- coding: utf-8 -*-
"""
Navette: Weaving the mathematics of light in thin film systems
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: navette_structure.py — Thin-film layer stack definition and solver-array generation.

Rework notes (v2):
  1.  MaterialProvider protocol decouples material lookup from a specific
      container.  Concrete adapters exist for the legacy dict format
      (DictMaterialProvider) and for OpticalWeaver (WeaverMaterialProvider).
  2.  Shared expansion logic lives in _LayerExpander — a stateless helper
      consumed by both Navette_Structure and Navette_Architect, eliminating the ~200
      lines of duplicated flatten-logic.
  3.  get_solver_inputs / get_error_solver_inputs are thin wrappers around
      _LayerExpander.expand(), differing only in whether stochastic errors
      are requested.
  4.  SolverArrays NamedTuple replaces the ad-hoc 5-tuple return, giving
      field names (.indices, .thicknesses, …) that are self-documenting and
      unpackable at call-sites that still want positional access.

Phase 4 improvements:
  - Magic numbers replaced with IntEnums (ErrorType, RoughnessType, ErrorMask).
  - Module hygiene: __all__ defined, unused imports cleaned.

Phase 5 improvements:
  - Layer.mask reinstated as a *computed property* (see LayerMask) so it can
    never go stale — it is always derived from coherent / inhomogen /
    rough_type at access time.  Layer.layer_typ reinstated as plain metadata.
  - RoughnessType wired into Layer defaults and mask/rough-type generation.
  - set_properties now WARNS on unknown / read-only keys instead of raising,
    keeping it tolerant of superset state dicts.
  - get_error_solver_inputs accepts an optional np.random.Generator (rng),
    forwarded through _LayerExpander into the Group error model for
    reproducible Monte-Carlo runs.
  - _LayerExpander.expand takes a single uniform contract: an iterator of
    (Layer, invert_gradient) tuples.  The vestigial invert_inhomogen flag and
    the per-layer isinstance() branch are gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    NamedTuple,
    Union,
    runtime_checkable,
)

import numpy as np

from navette_interpolator import UniSpline
from optical_models.ema_models import looyenga_eps


# ═══════════════════════════════════════════════════════════════════════════════
# Standardised numeric types (Numba-friendly)
# ═══════════════════════════════════════════════════════════════════════════════
FLOAT_TYPE = np.float64
COMPLEX_TYPE = np.complex128
INT_TYPE = np.int32

__all__ = [
    "FLOAT_TYPE",
    "COMPLEX_TYPE",
    "INT_TYPE",
    "InterpolationSettings",
    "SolverArrays",
    "MaterialProvider",
    "DictMaterialProvider",
    "MaterialObjectProvider",
    "WeaverMaterialProvider",
    "wrap_material_source",
    "Layer",
    "Group",
    "Navette_Structure",
    "ErrorType",
    "RoughnessType",
    "ErrorMask",
    "LayerMask",
]


# -------------------------------------------------------------------------
# Enums for self-documenting code
# -------------------------------------------------------------------------
class ErrorType(IntEnum):
    """Stochastic error distribution types."""
    GAUSSIAN = 0
    UNIFORM = 1
    COMBINED = 2


class RoughnessType(IntEnum):
    """
    Roughness model types (the solver consumes these integers).

    NONE means "no roughness contribution".  Additional model types should
    be appended here as the solver defines them; ``Layer.rough_type`` also
    accepts raw integers for solver model IDs that are not yet enumerated.
    """
    NONE = 0
    SCALAR = 1          # e.g., scalar scattering
    # Add more as defined by the solver


class ErrorMask(IntEnum):
    """Index positions in Group.error_mask list."""
    THICKNESS = 0
    N_REAL = 1
    N_IMAG = 2
    ROUGHNESS = 3
    INH_DELTA = 4
    INTERFACE = 5


class LayerMask(IntEnum):
    """
    Index positions in the solver-facing Layer.mask vector.

    Building Layer.mask through these names (rather than by raw position)
    keeps construction and consumption in lock-step: reordering the enum
    can never silently scramble the array.
    """
    ACTIVE = 0       # always 1 for a real layer
    COHERENT = 1
    INHOMOGEN = 2
    ROUGHNESS = 3


# -------------------------------------------------------------------------
# InterpolationSettings – groups all UniSpline configuration
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class InterpolationSettings:
    """
    Settings for UniSpline interpolation inside WeaverMaterialProvider.

    Attributes
    ----------
    method : str
        Interpolation method: 'linear', 'pchip', 'makima', 'sprague',
        'floater_hormann' (or 'fh'). Default 'linear'.
    floater_hormann_d : int
        Degree for Floater‑Hormann interpolation (ignored for other methods).
        Must satisfy 0 <= d < len(source_wavelength). Default 3.
    robust : bool
        If True, use the numerically stable barycentric form for Sprague.
        Default False (uses naive Lagrange).
    """
    method: str = "linear"
    floater_hormann_d: int = 3
    robust: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SolverArrays — typed return container
# ═══════════════════════════════════════════════════════════════════════════════
class SolverArrays(NamedTuple):
    """
    Flat, solver-ready representation of a thin-film stack.

    All arrays use strict dtypes compatible with the Numba-JIT
    FastScatterMatrix engine.  The tuple is both subscriptable
    (positional unpacking) and attribute-accessible.
    """
    indices: np.ndarray          # complex128, shape (n_total, n_wavs)
    thicknesses: np.ndarray      # float64,    shape (n_total,)
    incoherent_flags: np.ndarray # bool,       shape (n_total,)
    rough_types: np.ndarray      # int32,      shape (n_total,)
    rough_vals: np.ndarray       # float64,    shape (n_total,)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MaterialProvider — pluggable material-data source
# ═══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class MaterialProvider(Protocol):
    """
    Minimal interface a material data source must satisfy.

    get_nk(name) → complex128 array of shape (n_wavs,)
    contains(name) → bool

    Concrete implementations:
      - DictMaterialProvider  (wraps the legacy {name: obj_with_nk} dict)
      - WeaverMaterialProvider (wraps OpticalWeaver)
    """
    def get_nk(self, material_name: str) -> np.ndarray: ...
    def contains(self, material_name: str) -> bool: ...


class DictMaterialProvider:
    """
    Backward-compatible adapter for the legacy active_material_dict format
    where each value has a ``.nk`` attribute (complex128 ndarray).

    Safety: if ``.nk`` has not been computed yet (the Material's
    ``complex_refractive_index()`` was never called), this adapter will
    call it automatically — provided a wavelength grid has been set on
    the Material.  If neither ``.nk`` nor ``.wavelength`` exist, a clear
    error is raised instead of a cryptic AttributeError.
    """
    __slots__ = ("_dict",)

    def __init__(self, mat_dict: Dict[str, Any]) -> None:
        self._dict = mat_dict

    def get_nk(self, material_name: str) -> np.ndarray:
        mat = self._dict[material_name]

        # Fast path: .nk already computed
        nk = getattr(mat, "nk", None)
        if nk is not None:
            return nk

        # Slow path: trigger computation
        if hasattr(mat, "complex_refractive_index"):
            mat.complex_refractive_index()
            nk = getattr(mat, "nk", None)
            if nk is not None:
                return nk

        raise AttributeError(
            f"DictMaterialProvider: material '{material_name}' has no .nk "
            f"attribute and complex_refractive_index() did not produce one. "
            f"Ensure set_wavelength_range() was called first."
        )

    def contains(self, material_name: str) -> bool:
        return material_name in self._dict


class MaterialObjectProvider:
    """
    Provider that wraps a dict of Material *objects* and manages wavelength
    state properly.

    Unlike DictMaterialProvider (which trusts that .nk is pre-computed),
    this provider owns a target wavelength grid and ensures every Material's
    ``complex_refractive_index()`` is called with that grid before returning
    the result.  Results are cached per-material and invalidated when the
    wavelength grid changes.

    This is the recommended provider when working directly with Material /
    Konstant / TableMaterial / EffectiveMaterial objects.

    Parameters
    ----------
    mat_dict : dict[str, Material]
        Mapping of material names to Material instances.
    wavelength : np.ndarray
        Target wavelength grid (nm).  All materials will be evaluated on
        this grid.
    """
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
        cached = self._cache.get(material_name)
        if cached is not None:
            return cached

        mat = self._dict[material_name]
        nk = mat.complex_refractive_index(self._wavelength)
        self._cache[material_name] = nk
        return nk

    def contains(self, material_name: str) -> bool:
        return material_name in self._dict

    def invalidate(self, material_name: Optional[str] = None) -> None:
        """Drop cached nk values (call after material parameters change)."""
        if material_name is None:
            self._cache.clear()
        else:
            self._cache.pop(material_name, None)


class WeaverMaterialProvider:
    """
    Adapter that pulls n+ik data from an OpticalWeaver (or OpticalCollection)
    and interpolates onto a common wavelength grid.

    The weaver stores data keyed by OpticalKeyAlias = (float, str, str).
    This adapter expects keys of the form:
        n data:  (key_prefix, n_label, material_name)
        k data:  (key_prefix, k_label, material_name)

    If a material only has an 'n' key and no 'k' key, k defaults to 0.

    Parameters
    ----------
    weaver : OpticalWeaver | OpticalCollection
        The data source.
    target_wavelength : np.ndarray
        The common wavelength grid all materials will be interpolated onto.
    key_prefix : float
        First element of the OpticalKeyAlias tuple (default 0.0, typically AOI).
    n_label : str
        Second element used for the real-part key (default 'n').
    k_label : str
        Second element used for the imaginary-part key (default 'k').
    interp : InterpolationSettings
        Interpolation method and its parameters.
    """
    __slots__ = (
        "_weaver", "_target_wl", "_cache",
        "_key_prefix", "_n_label", "_k_label",
        "_interp_settings",
    )

    def __init__(
        self,
        weaver: Any,  # OpticalWeaver | OpticalCollection
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
        cached = self._cache.get(material_name)
        if cached is not None:
            return cached

        n_key = (self._key_prefix, self._n_label, material_name)
        k_key = (self._key_prefix, self._k_label, material_name)

        n_arr = self._fetch_and_interpolate(n_key)
        if n_arr is None:
            raise KeyError(
                f"WeaverMaterialProvider: material '{material_name}' "
                f"not found (key {n_key})."
            )

        k_arr = self._fetch_and_interpolate(k_key)
        if k_arr is None:
            k_arr = np.zeros_like(n_arr)

        nk = n_arr + 1j * k_arr
        self._cache[material_name] = nk
        return nk

    def contains(self, material_name: str) -> bool:
        n_key = (self._key_prefix, self._n_label, material_name)
        return n_key in self._weaver

    def invalidate_cache(self, material_name: Optional[str] = None) -> None:
        """Drop cached interpolations (call after weaver data changes)."""
        if material_name is None:
            self._cache.clear()
        else:
            self._cache.pop(material_name, None)

    # -- internal --
    def _fetch_and_interpolate(self, key: tuple) -> Optional[np.ndarray]:
        if key not in self._weaver:
            return None

        # get_weaved returns (wl, data) sorted and concatenated
        src_wl, src_data = self._weaver.get_weaved(key)
        if src_wl.size == 0:
            return None

        # Fast path: grids match exactly (common in single-frame setups)
        if (src_wl.shape == self._target_wl.shape
                and np.array_equal(src_wl, self._target_wl)):
            return src_data.astype(np.float64)

        # Interpolate onto target grid
        spline = UniSpline(
            src_wl, src_data,
            method=self._interp_settings.method,
            robust=self._interp_settings.robust,
            d=self._interp_settings.floater_hormann_d,
        )
        return spline(self._target_wl).astype(np.float64)


def wrap_material_source(source: Any, **kwargs: Any) -> MaterialProvider:
    """
    Convenience factory: auto-detect and wrap a material data source.

    Accepts:
      - An existing MaterialProvider (returned as-is)
      - A dict (wrapped in DictMaterialProvider, or MaterialObjectProvider
        if ``wavelength`` kwarg is provided)
      - An object with get_weaved (wrapped in WeaverMaterialProvider; requires
        target_wavelength kwarg)
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
            raise ValueError(
                "wrap_material_source: OpticalWeaver detected but no "
                "'target_wavelength' kwarg provided."
            )
        return WeaverMaterialProvider(source, target_wl, **{
            k: v for k, v in kwargs.items() if k != "target_wavelength"
        })
    raise TypeError(
        f"wrap_material_source: unsupported type {type(source).__name__}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Layer
# ═══════════════════════════════════════════════════════════════════════════════
class Layer:
    """
    Represents a single layer in a thin-film structure.

    Uses __slots__ for faster attribute access and reduced memory footprint.
    Designed to be serialisable (get_state / from_state) for node-graph
    persistence and cloneable (clone) for optimisation algorithms.

    ``mask`` is exposed as a computed, read-only property (a solver-facing
    flag vector indexed by LayerMask) so it is always consistent with the
    current coherent / inhomogen / rough_type values.  Set the underlying
    attributes, not ``mask`` itself.
    """
    __slots__ = (
        "material", "coherent", "_inhomogen", "rough_type", "_inh_delta",
        "roughness", "interface", "interface_thickness", "_thickness",
        "optimize", "needle", "layer_typ", "sub_layer_count",
    )

    def __init__(
        self,
        thickness: float = 1.0,
        material_name: str = "",
        coherent: bool = True,
        roughness: float = 0.0,
        rough_type: Union[int, RoughnessType] = RoughnessType.NONE,
        inhomogen: bool = False,
        inh_delta: float = 0.1,
        interface: bool = False,
        interface_thickness: float = 0.0,
        optimize: bool = True,
        needle: bool = True,
        layer_typ: int = 1,
    ) -> None:
        self.material: str = material_name
        self.coherent: bool = coherent
        self._inhomogen: bool = inhomogen
        self.rough_type: Union[int, RoughnessType] = rough_type
        self._inh_delta: float = inh_delta
        self.roughness: float = roughness
        self.interface: bool = interface
        self.interface_thickness: float = interface_thickness
        self._thickness: float = float(thickness)
        self.optimize: bool = optimize
        self.needle: bool = needle
        self.layer_typ: int = layer_typ

        self._refine_layer_count()

    # -- callable shorthand ------------------------------------------------
    def __call__(self) -> Tuple[str, float]:
        return (self.material, self._thickness)

    # -- properties --------------------------------------------------------
    @property
    def thickness(self) -> float:
        return self._thickness

    @thickness.setter
    def thickness(self, value: float) -> None:
        self._thickness = float(value)
        if self._inhomogen:
            self._refine_layer_count()

    @property
    def inhomogen(self) -> bool:
        return self._inhomogen

    @inhomogen.setter
    def inhomogen(self, value: bool) -> None:
        self._inhomogen = bool(value)
        if self._inhomogen:
            self._refine_layer_count()

    @property
    def inh_delta(self) -> float:
        return self._inh_delta

    @inh_delta.setter
    def inh_delta(self, value: float) -> None:
        self._inh_delta = float(value)
        if self._inhomogen:
            self._refine_layer_count()

    @property
    def mask(self) -> np.ndarray:
        """
        Solver-facing flag vector, derived fresh from current layer state.

        Indexed by LayerMask: [ACTIVE, COHERENT, INHOMOGEN, ROUGHNESS].
        Computed on access, so it can never drift out of sync with the
        attributes it summarises.  Read-only by design.
        """
        m = np.zeros(len(LayerMask), dtype=INT_TYPE)
        m[LayerMask.ACTIVE] = 1
        m[LayerMask.COHERENT] = int(self.coherent)
        m[LayerMask.INHOMOGEN] = int(self._inhomogen)
        m[LayerMask.ROUGHNESS] = int(self.rough_type != RoughnessType.NONE)
        return m

    # -- internal helpers --------------------------------------------------
    def _refine_layer_count(self) -> None:
        if self._inhomogen and self._thickness > 0:
            factor = 1.0 + (self._inh_delta / 0.1) * 0.5
            self.sub_layer_count = int(np.ceil(self._thickness ** 0.4) * factor) + 1
        else:
            self.sub_layer_count = 1

    # -- serialisation (node-graph friendly) -------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Full serialisable snapshot (replaces get_properties).

        ``mask`` is intentionally omitted — it is derived, not stored.
        """
        return {
            "thickness": self._thickness,
            "material": self.material,
            "coherent": self.coherent,
            "inhomogen": self._inhomogen,
            "inh_delta": self._inh_delta,
            "rough_type": int(self.rough_type),
            "roughness": self.roughness,
            "interface": self.interface,
            "interface_thickness": self.interface_thickness,
            "optimize": self.optimize,
            "needle": self.needle,
            "layer_typ": self.layer_typ,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Layer":
        """Reconstruct a Layer from a serialised dict."""
        return cls(
            thickness=state.get("thickness", 1.0),
            material_name=state.get("material", ""),
            coherent=state.get("coherent", True),
            roughness=state.get("roughness", 0.0),
            rough_type=state.get("rough_type", RoughnessType.NONE),
            inhomogen=state.get("inhomogen", False),
            inh_delta=state.get("inh_delta", 0.1),
            interface=state.get("interface", False),
            interface_thickness=state.get("interface_thickness", 0.0),
            optimize=state.get("optimize", True),
            needle=state.get("needle", True),
            layer_typ=state.get("layer_typ", 1),
        )

    # Backward compat aliases
    get_properties = get_state

    def set_properties(self, properties: Dict[str, Any]) -> None:
        """
        Update layer attributes from a dict.

        Unknown or read-only keys (e.g. the derived ``mask``) are skipped
        with a warning rather than raising, so callers may safely pass a
        superset of state / widget values.
        """
        for key, value in properties.items():
            if not hasattr(self, key):
                warnings.warn(
                    f"Layer.set_properties: ignoring unknown attribute "
                    f"'{key}'.",
                    stacklevel=2,
                )
                continue
            try:
                setattr(self, key, value)
            except AttributeError:
                warnings.warn(
                    f"Layer.set_properties: '{key}' is read-only; ignoring.",
                    stacklevel=2,
                )

        # Ensure derived state is refreshed
        if self.interface or self._inhomogen:
            self._refine_layer_count()

    # -- clone -------------------------------------------------------------
    def clone(self) -> "Layer":
        obj = Layer.__new__(Layer)
        obj.material = self.material
        obj.coherent = self.coherent
        obj._inhomogen = self._inhomogen
        obj.rough_type = self.rough_type
        obj._inh_delta = self._inh_delta
        obj.roughness = self.roughness
        obj.interface = self.interface
        obj.interface_thickness = self.interface_thickness
        obj._thickness = self._thickness
        obj.optimize = self.optimize
        obj.needle = self.needle
        obj.layer_typ = self.layer_typ
        obj.sub_layer_count = self.sub_layer_count
        # mask is a computed property — nothing to copy.
        return obj

    def __repr__(self) -> str:
        return (
            f"Layer(mat='{self.material}', d={self._thickness:.2f}nm, "
            f"rough={self.roughness:.2f}A, opt={self.optimize})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Group
# ═══════════════════════════════════════════════════════════════════════════════
class Group:
    """
    Represents a group of materials sharing common optical and manufacturing
    properties (scaling factors, error models).
    """
    __slots__ = (
        "group_name", "thick_factor", "thick_summand", "n_factor", "k_factor",
        "inh_delta_summand", "roughness_summand", "interface_summand",
        "error_mask", "optimization_mask",
        "thickness_error_type", "n_error_type", "k_error_type",
        "inh_delta_error_type", "roughness_error_type", "interface_error_type",
        "thickness_error_params", "inh_delta_error_params",
        "roughness_error_params", "interface_error_params",
        "n_error_params", "k_error_params",
    )

    _DEFAULT_ERROR_PARAMS: Dict[str, float] = {
        "abs_mean_delta_g": 0.0, "abs_std_dev": 0.01,
        "rel_mean_delta_g": 0.0, "rel_std_dev": 1.0,
        "abs_mean_delta_h": 0.0, "abs_variance": 0.01,
        "rel_mean_delta_h": 0.0, "rel_variance": 1.0,
    }

    def __init__(
        self,
        group_name: str,
        thick_factor: float = 1.0,
        thick_summand: float = 0.0,
        n_factor: float = 1.0,
        k_factor: float = 0.0,
        inh_delta_summand: float = 0.0,
        roughness_summand: float = 0.0,
        interface_summand: float = 0.0,
    ) -> None:
        self.group_name = group_name
        self.thick_factor = thick_factor
        self.thick_summand = thick_summand
        self.n_factor = n_factor
        self.k_factor = k_factor
        self.inh_delta_summand = inh_delta_summand
        self.roughness_summand = roughness_summand
        self.interface_summand = interface_summand

        self.error_mask: List[int] = [0] * len(ErrorMask)
        self.optimization_mask: List[int] = [0] * 7

        self.thickness_error_type = ErrorType.GAUSSIAN
        self.n_error_type = ErrorType.GAUSSIAN
        self.k_error_type = ErrorType.GAUSSIAN
        self.inh_delta_error_type = ErrorType.GAUSSIAN
        self.roughness_error_type = ErrorType.GAUSSIAN
        self.interface_error_type = ErrorType.GAUSSIAN

        dp = self._DEFAULT_ERROR_PARAMS
        self.thickness_error_params = dp.copy()
        self.inh_delta_error_params = dp.copy()
        self.roughness_error_params = dp.copy()
        self.interface_error_params = dp.copy()
        self.n_error_params = dp.copy()
        self.k_error_params = dp.copy()

    @property
    def nk_factor(self) -> complex:
        return complex(self.n_factor, self.k_factor)

    # -- stochastic error application (with optional rng) -----------------
    @staticmethod
    def _apply_error(
        value: Any,
        error_type: int,
        error_params: Dict[str, float],
        rng: Optional[np.random.Generator] = None,
    ) -> Any:
        """Apply a stochastic manufacturing error to *value*.

        Parameters
        ----------
        value : numeric
            Base value to perturb.
        error_type : int (ErrorType)
            0=Gaussian, 1=Uniform, 2=Combined.
        error_params : dict
            Distribution parameters.
        rng : np.random.Generator, optional
            Random generator. If None, uses the legacy global np.random.
        """
        if rng is None:
            rng = np.random

        if error_type == ErrorType.GAUSSIAN:
            abs_err = rng.normal(error_params["abs_mean_delta_g"],
                                 error_params["abs_std_dev"])
            rel_err = rng.normal(error_params["rel_mean_delta_g"],
                                 error_params["rel_std_dev"]) * value
            return value + abs_err + rel_err

        if error_type == ErrorType.UNIFORM:
            abs_err = rng.uniform(-error_params["abs_variance"],
                                   error_params["abs_variance"])
            rel_err = rng.uniform(-error_params["rel_variance"],
                                   error_params["rel_variance"]) * value
            return value + abs_err + rel_err

        if error_type == ErrorType.COMBINED:
            g_abs = rng.normal(error_params["abs_mean_delta_g"],
                               error_params["abs_std_dev"])
            g_rel = rng.normal(error_params["rel_mean_delta_g"],
                               error_params["rel_std_dev"]) * value
            u_abs = rng.uniform(-error_params["abs_variance"],
                                 error_params["abs_variance"])
            u_rel = rng.uniform(-error_params["rel_variance"],
                                 error_params["rel_variance"]) * value
            return value + g_abs + g_rel + u_abs + u_rel

        return value

    def thickness_error(self, value: float, rng: Optional[np.random.Generator] = None) -> float:
        return max(0.0, self._apply_error(
            value, self.thickness_error_type, self.thickness_error_params, rng=rng))

    def inh_delta_error(self, value: float, rng: Optional[np.random.Generator] = None) -> float:
        return self._apply_error(
            value, self.inh_delta_error_type, self.inh_delta_error_params, rng=rng)

    def sr_roughness_error(self, value: float, thickness: float, rng: Optional[np.random.Generator] = None) -> float:
        return max(0.0, self._apply_error(
            value, self.roughness_error_type, self.roughness_error_params, rng=rng))

    def interface_error(self, value: float, thickness: float, rng: Optional[np.random.Generator] = None) -> float:
        return max(0.0, self._apply_error(
            value, self.interface_error_type, self.interface_error_params, rng=rng))

    def nk_error(self, nk_value: complex, rng: Optional[np.random.Generator] = None) -> complex:
        n_val = self._apply_error(nk_value.real, self.n_error_type, self.n_error_params, rng=rng)
        k_val = self._apply_error(nk_value.imag, self.k_error_type, self.k_error_params, rng=rng)
        return complex(max(0.0, n_val), k_val)

    # -- serialisation -----------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self.__slots__}

    get_properties = get_state

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Group":
        obj = cls(state.get("group_name", "default"))
        for key, value in state.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj

    def set_properties(self, properties: Dict[str, Any]) -> None:
        """
        Update group attributes from a dict.

        Unknown or read-only keys (e.g. the derived ``nk_factor``) are
        skipped with a warning rather than raising.
        """
        for key, value in properties.items():
            if not hasattr(self, key):
                warnings.warn(
                    f"Group.set_properties: ignoring unknown attribute "
                    f"'{key}'.",
                    stacklevel=2,
                )
                continue
            try:
                setattr(self, key, value)
            except AttributeError:
                warnings.warn(
                    f"Group.set_properties: '{key}' is read-only; ignoring.",
                    stacklevel=2,
                )

    # -- clone -------------------------------------------------------------
    def clone(self) -> "Group":
        obj = Group.__new__(Group)
        for attr in self.__slots__:
            val = getattr(self, attr)
            if isinstance(val, (list, dict)):
                setattr(obj, attr, val.copy())
            else:
                setattr(obj, attr, val)
        return obj

    def __repr__(self) -> str:
        return f"Group(name='{self.group_name}', thick_factor={self.thick_factor:.3f})"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  _LayerExpander — shared expansion engine (DRY)
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULT_GROUP = Group("_default_")
_NO_ROUGHNESS = int(RoughnessType.NONE)


class _LayerExpander:
    """
    Stateless helper that expands a logical layer sequence into flat columnar
    arrays for the solver.

    This is the *single* implementation of the expansion logic —
    Navette_Structure and Navette_Architect both delegate here.

    The caller provides an iterator of (Layer, invert_gradient) tuples.  The
    bool controls only whether an inhomogeneous layer's gradient direction is
    flipped (set True when the owning block is traversed in reverse).  A flat
    homogeneous stack passes ``(layer, False)`` for every entry.
    """

    @staticmethod
    def expand(
        layers: Iterator[Tuple[Layer, bool]],
        materials: MaterialProvider,
        group_dict: Dict[str, Group],
        *,
        apply_errors: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> SolverArrays:
        """
        Expand a sequence of layers into solver-ready columnar arrays.

        Parameters
        ----------
        layers : iterable of (Layer, bool)
            Ordered (layer, invert_gradient) pairs, ambient first / substrate
            last.  The bool flips the inhomogeneous gradient direction for
            that specific layer.
        materials : MaterialProvider
            Source for complex refractive indices.
        group_dict : dict[str, Group]
            Material-name → Group mapping for factors/errors.
        apply_errors : bool
            If True, apply stochastic manufacturing errors from groups.
        rng : np.random.Generator, optional
            Random number generator for reproducible errors. If None and
            `apply_errors` is True, uses the legacy global np.random state.

        Returns
        -------
        SolverArrays

        Raises
        ------
        ValueError
            If no layers are provided (empty sequence).
        """
        col_thick: List[float] = []
        col_nk: List[Union[complex, np.ndarray]] = []
        col_coh: List[bool] = []
        col_r_val: List[float] = []
        col_r_type: List[int] = []

        get_group = group_dict.get
        prev_eff_nk: Optional[np.ndarray] = None

        for layer, inv in layers:
            mat_name = layer.material
            group = get_group(mat_name, _DEFAULT_GROUP)

            # --- Base nk with systematic group factors ------
            base_nk = materials.get_nk(mat_name)

            # Identity for the complex gain (n_factor + i*k_factor) is
            # (1 + 0j); skip the multiply only when truly trivial.
            if group.n_factor != 1.0 or group.k_factor != 0.0:
                layer_nk = base_nk * group.nk_factor
            else:
                layer_nk = base_nk

            layer_thickness = (
                layer.thickness * group.thick_factor + group.thick_summand
            )

            # --- Stochastic errors (when requested) ---------
            current_roughness = layer.roughness

            if apply_errors:
                # Thickness
                if group.error_mask[ErrorMask.THICKNESS]:
                    layer_thickness = group.thickness_error(layer_thickness, rng=rng)

                # n and k
                if group.error_mask[ErrorMask.N_REAL] or group.error_mask[ErrorMask.N_IMAG]:
                    n_part = layer_nk.real
                    k_part = layer_nk.imag
                    if group.error_mask[ErrorMask.N_REAL]:
                        n_part = Group._apply_error(
                            n_part, group.n_error_type,
                            group.n_error_params, rng=rng
                        )
                        n_part = np.maximum(0.0, n_part)
                    if group.error_mask[ErrorMask.N_IMAG]:
                        k_part = Group._apply_error(
                            k_part, group.k_error_type,
                            group.k_error_params, rng=rng
                        )
                    layer_nk = n_part + 1j * k_part

                # Roughness
                if group.error_mask[ErrorMask.ROUGHNESS]:
                    current_roughness = group.sr_roughness_error(
                        current_roughness, layer_thickness, rng=rng
                    )

            if layer_thickness < 0.0:
                layer_thickness = 0.0

            # --- A. Interface Generation --------------------
            if layer.interface and prev_eff_nk is not None:
                t_interface = layer.interface_thickness

                if apply_errors and group.error_mask[ErrorMask.INTERFACE]:
                    t_interface = group.interface_error(
                        t_interface, layer.thickness, rng=rng
                    )

                if t_interface > layer_thickness:
                    t_interface = layer_thickness
                layer_thickness -= t_interface

                interface_nk = looyenga_eps(layer_nk, prev_eff_nk, 0.5)

                col_thick.append(t_interface)
                col_nk.append(interface_nk)
                col_coh.append(True)
                col_r_val.append(0.0)
                col_r_type.append(_NO_ROUGHNESS)

            # --- B. Inhomogeneity Generation ---------------
            if layer.inhomogen and layer.sub_layer_count > 1:
                sub_div = layer.sub_layer_count

                current_delta = (
                    (layer.inh_delta + group.inh_delta_summand) * 0.5
                )

                if apply_errors and group.error_mask[ErrorMask.INH_DELTA]:
                    current_delta = group.inh_delta_error(current_delta, rng=rng)

                factors = np.linspace(
                    1.0 - current_delta, 1.0 + current_delta, sub_div
                )
                if inv:   # per-layer inversion flag
                    factors = factors[::-1]

                step_t = layer_thickness / sub_div

                for ix, f in enumerate(factors):
                    col_thick.append(step_t)
                    col_nk.append(layer_nk * f)
                    col_coh.append(layer.coherent)

                    if ix == 0:
                        col_r_val.append(current_roughness)
                        col_r_type.append(int(layer.rough_type))
                    else:
                        col_r_val.append(0.0)
                        col_r_type.append(_NO_ROUGHNESS)

            # --- C. Standard Layer -------------------------
            else:
                col_thick.append(layer_thickness)
                col_nk.append(layer_nk)
                col_coh.append(layer.coherent)
                col_r_val.append(current_roughness)
                col_r_type.append(int(layer.rough_type))

            prev_eff_nk = layer_nk

        # --- Guard against empty stacks ---------------------
        if not col_nk:
            raise ValueError(
                "_LayerExpander.expand: No layers to expand. "
                "Empty layer sequence provided."
            )

        # --- Final conversion to strict-typed arrays ------
        return SolverArrays(
            indices=np.vstack(col_nk).astype(COMPLEX_TYPE),
            thicknesses=np.array(col_thick, dtype=FLOAT_TYPE),
            incoherent_flags=np.array(
                [not c for c in col_coh], dtype=np.bool_
            ),
            rough_types=np.array(col_r_type, dtype=INT_TYPE),
            rough_vals=np.array(col_r_val, dtype=FLOAT_TYPE),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Navette_Structure
# ═══════════════════════════════════════════════════════════════════════════════
class Navette_Structure:
    """
    Manages the translation of high-level Layer definitions into numerical
    arrays compatible with the FastScatterMatrix solver.

    Now accepts a MaterialProvider instead of a raw dict, and delegates all
    expansion logic to _LayerExpander.
    """

    def __init__(
        self,
        layer_list: Optional[List[Layer]] = None,
        group_dict: Optional[Dict[str, Group]] = None,
        materials: Optional[Union[MaterialProvider, Dict[str, Any]]] = None,
    ):
        """
        Parameters
        ----------
        layer_list : list[Layer], optional
            Ordered layer stack.  Defaults to empty.
        group_dict : dict[str, Group], optional
            Material-name → Group mapping.
        materials : MaterialProvider | dict, optional
            Source for optical constants.  A plain dict is auto-wrapped in
            DictMaterialProvider for backward compatibility.
        """
        self.layer_list: List[Layer] = layer_list or []
        self.group_dict: Dict[str, Group] = group_dict or {}

        # Accept either a provider or a legacy dict
        if materials is None:
            self._materials: Optional[MaterialProvider] = None
        elif isinstance(materials, dict):
            self._materials = DictMaterialProvider(materials)
        else:
            self._materials = materials

        # Legacy attribute for backward compatibility
        self.simple_layer_list: List[List[Any]] = []

    # -- material provider access (settable for Architect injection) ----
    @property
    def materials(self) -> Optional[MaterialProvider]:
        return self._materials

    @materials.setter
    def materials(self, value: Any) -> None:
        if isinstance(value, dict):
            self._materials = DictMaterialProvider(value)
        else:
            self._materials = value

    # Backward compat: setting active_material_dict still works
    @property
    def active_material_dict(self) -> Optional[MaterialProvider]:
        return self._materials

    @active_material_dict.setter
    def active_material_dict(self, value: Any) -> None:
        self.materials = value

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Validate physical constraints on all layers."""
        errors: List[str] = []
        if not self.layer_list:
            errors.append("Structure contains no layers.")
            return errors

        for i, layer in enumerate(self.layer_list):
            if layer.thickness < 0:
                errors.append(
                    f"Layer {i} ({layer.material}): Negative thickness "
                    f"{layer.thickness} nm."
                )
            if layer.roughness < 0:
                errors.append(
                    f"Layer {i} ({layer.material}): Negative roughness "
                    f"{layer.roughness} A."
                )
            if layer.interface and layer.interface_thickness >= layer.thickness:
                errors.append(
                    f"Layer {i} ({layer.material}): Interface thickness "
                    f"({layer.interface_thickness}) >= layer thickness "
                    f"({layer.thickness})."
                )
            if self._materials and not self._materials.contains(layer.material):
                errors.append(
                    f"Layer {i}: Material '{layer.material}' not found "
                    f"in material provider."
                )
        return errors

    # -- solver array generation -------------------------------------------
    def get_solver_inputs(self) -> SolverArrays:
        """
        Generate flat solver arrays (no stochastic errors).

        Returns
        -------
        SolverArrays
            Named tuple of (indices, thicknesses, incoherent_flags,
            rough_types, rough_vals).
        """
        if not self.layer_list:
            raise ValueError("Structure is empty.")
        if self._materials is None:
            raise ValueError("No material provider set.")

        # A standalone structure is never traversed in reverse → invert=False.
        return _LayerExpander.expand(
            ((layer, False) for layer in self.layer_list),
            self._materials,
            self.group_dict,
            apply_errors=False,
        )

    def get_error_solver_inputs(
        self, rng: Optional[np.random.Generator] = None
    ) -> SolverArrays:
        """
        Generate flat solver arrays WITH stochastic manufacturing errors.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Generator for reproducible Monte-Carlo runs.  If None, the legacy
            global np.random state is used.

        Returns
        -------
        SolverArrays
        """
        if not self.layer_list:
            raise ValueError("Structure is empty.")
        if self._materials is None:
            raise ValueError("No material provider set.")

        return _LayerExpander.expand(
            ((layer, False) for layer in self.layer_list),
            self._materials,
            self.group_dict,
            apply_errors=True,
            rng=rng,
        )

    # -- legacy compat -----------------------------------------------------
    def generate_simple_layer_list(self) -> List[List[Any]]:
        """Legacy format: [[thickness, nk, coherent, roughness, rough_type], …]"""
        sa = self.get_solver_inputs()
        n_layers = sa.thicknesses.shape[0]
        self.simple_layer_list = [
            [
                sa.thicknesses[i],
                sa.indices[i],
                not sa.incoherent_flags[i],
                sa.rough_vals[i],
                sa.rough_types[i],
            ]
            for i in range(n_layers)
        ]
        return self.simple_layer_list

    # -- serialisation (node-graph) ----------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Serialise to a dict (for node save/load)."""
        return {
            "layers": [layer.get_state() for layer in self.layer_list],
            "groups": {
                name: group.get_state()
                for name, group in self.group_dict.items()
            },
        }

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        materials: Optional[Union[MaterialProvider, Dict[str, Any]]] = None,
    ) -> "Navette_Structure":
        """Reconstruct from a serialised dict."""
        layers = [Layer.from_state(ls) for ls in state.get("layers", [])]
        groups = {
            name: Group.from_state(gs)
            for name, gs in state.get("groups", {}).items()
        }
        return cls(layer_list=layers, group_dict=groups, materials=materials)

    # -- clone (addresses aliasing) ---------------------------------------
    def clone(self) -> "Navette_Structure":
        """Create a deep copy of this structure (new layer and group objects).

        The MaterialProvider is shared by reference on purpose: providers are
        caches keyed by material name on a shared wavelength grid, so sharing
        is safe and lets clones reuse the same computed nk values.
        """
        new_layers = [layer.clone() for layer in self.layer_list]
        new_groups = {name: group.clone() for name, group in self.group_dict.items()}
        new_struct = Navette_Structure(
            layer_list=new_layers,
            group_dict=new_groups,
            materials=self._materials,
        )
        return new_struct

    def __repr__(self) -> str:
        return (
            f"Navette_Structure(layers={len(self.layer_list)}, "
            f"groups={len(self.group_dict)})"
        )

    def __len__(self) -> int:
        return len(self.layer_list)

    # ===== inside class Navette_Structure =====
    # (place these methods after __len__ and before __repr__)
    
    def __getitem__(self, index: int) -> Layer:
        return self.layer_list[index]
    
    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layer_list)
    
    def __bool__(self) -> bool:
        return len(self.layer_list) > 0
    
    def total_physical_thickness(self) -> float:
        """Sum of nominal thicknesses of all layers."""
        return sum(layer.thickness for layer in self.layer_list)
    
    def get_optimization_parameters(self) -> List[Layer]:
        """Return list of layers marked for optimisation."""
        return [layer for layer in self.layer_list if layer.optimize]
    
    def replace_material(self, old_name: str, new_name: str) -> int:
        """Replace all occurrences of old material name with new name. Returns count."""
        count = 0
        for layer in self.layer_list:
            if layer.material == old_name:
                layer.material = new_name
                count += 1
        return count
    
    def insert_layer(self, index: int, layer: Layer) -> None:
        self.layer_list.insert(index, layer)
    
    def remove_layer(self, index: int) -> Layer:
        return self.layer_list.pop(index)
    
    def replace_layer(self, index: int, new_layer: Layer) -> None:
        self.layer_list[index] = new_layer
    
    def prune_thin_layers(self, min_thickness: float = 0.001) -> int:
        """Remove layers thinner than min_thickness. Returns number removed."""
        before = len(self.layer_list)
        self.layer_list = [l for l in self.layer_list if l.thickness >= min_thickness]
        return before - len(self.layer_list)
    
    def total_sub_layers(self) -> int:
        """
        Number of physical slices after inhomogeneous subdivision (no errors).
        Accounts for interfaces (adds one slice per interface except for the first layer).
        """
        total = 0
        for i, layer in enumerate(self.layer_list):
            # Inhomogeneous splitting
            if layer.inhomogen and layer.sub_layer_count > 1:
                total += layer.sub_layer_count
            else:
                total += 1
            # Interface slice (if not the first layer)
            if layer.interface and i > 0:
                total += 1
        return total
    
    def find_layers_by_material(self, material_name: str) -> List[int]:
        """Return indices of all layers using the given material."""
        return [i for i, layer in enumerate(self.layer_list) if layer.material == material_name]
    
    def count_material(self, material_name: str) -> int:
        """Return how many layers use the given material."""
        return sum(1 for layer in self.layer_list if layer.material == material_name)
    
    def apply_to_all_layers(self, func: callable) -> None:
        """Apply a callable to every layer (e.g., thickness scaling)."""
        for layer in self.layer_list:
            func(layer)
    
    def __add__(self, other: "Navette_Structure") -> "Navette_Structure":
        """
        Concatenate two structures. Groups are merged; conflicting group definitions raise ValueError.
        """
        new = self.clone()
        # Add layers from other (cloned)
        new.layer_list.extend(other.clone().layer_list)
        # Merge group dicts with conflict detection
        for name, group in other.group_dict.items():
            if name in new.group_dict:
                if new.group_dict[name].get_state() != group.get_state():
                    raise ValueError(
                        f"Group '{name}' defined differently in the two structures. "
                        "Cannot merge automatically."
                    )
            else:
                new.group_dict[name] = group
        return new
    
    def get_group_for_material(self, material_name: str) -> Group:
        """Return the Group associated with a material, or a default Group."""
        from navette_structure import _DEFAULT_GROUP  # local import to avoid circular
        return self.group_dict.get(material_name, _DEFAULT_GROUP)
    
    def __contains__(self, material_name: str) -> bool:
        """Check if any layer uses the given material name."""
        return any(layer.material == material_name for layer in self.layer_list)