# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: loom_structure.py — Thin-film layer stack definition and solver-array generation.

Rework notes (v2):
  1.  MaterialProvider protocol decouples material lookup from a specific
      container.  Concrete adapters exist for the legacy dict format
      (DictMaterialProvider) and for OpticalWeaver (WeaverMaterialProvider).
  2.  Shared expansion logic lives in _LayerExpander — a stateless helper
      consumed by both Loom_Structure and Loom_Architect, eliminating the ~200
      lines of duplicated flatten-logic.
  3.  get_solver_inputs / get_error_solver_inputs are thin wrappers around
      _LayerExpander.expand(), differing only in whether stochastic errors
      are requested.
  4.  SolverArrays NamedTuple replaces the ad-hoc 5-tuple return, giving
      field names (.indices, .thicknesses, …) that are self-documenting and
      unpackable at call-sites that still want positional access.
"""

from __future__ import annotations

import warnings
from typing import (
    Any,
    Callable,
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

from optical_models.ema_models import looyenga_eps

# ═══════════════════════════════════════════════════════════════════════════════
# Standardised numeric types (Numba-friendly)
# ═══════════════════════════════════════════════════════════════════════════════
FLOAT_TYPE = np.float64
COMPLEX_TYPE = np.complex128
INT_TYPE = np.int32


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
    """
    __slots__ = (
        "_weaver", "_target_wl", "_cache",
        "_key_prefix", "_n_label", "_k_label",
    )

    def __init__(
        self,
        weaver: Any,  # OpticalWeaver | OpticalCollection
        target_wavelength: np.ndarray,
        key_prefix: float = 0.0,
        n_label: str = "n",
        k_label: str = "k",
    ) -> None:
        self._weaver = weaver
        self._target_wl = np.asarray(target_wavelength, dtype=np.float64)
        self._cache: Dict[str, np.ndarray] = {}
        self._key_prefix = key_prefix
        self._n_label = n_label
        self._k_label = k_label

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
        return np.interp(self._target_wl, src_wl, src_data).astype(np.float64)


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
# 3.  Layer  (unchanged in spirit; minor additions for node-graph compat)
# ═══════════════════════════════════════════════════════════════════════════════
class Layer:
    """
    Represents a single layer in a thin-film structure.

    Uses __slots__ for faster attribute access and reduced memory footprint.
    Designed to be serialisable (get_state / from_state) for node-graph
    persistence and cloneable (clone) for optimisation algorithms.
    """
    __slots__ = (
        "material", "coherent", "_inhomogen", "rough_type", "_inh_delta",
        "roughness", "interface", "interface_thickness", "_thickness",
        "optimize", "needle", "layer_typ", "mask", "sub_layer_count",
    )

    def __init__(
        self,
        thickness: float = 1.0,
        material_name: str = "",
        coherent: bool = True,
        roughness: float = 0.0,
        rough_type: int = 0,
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
        self.rough_type: int = rough_type
        self._inh_delta: float = inh_delta
        self.roughness: float = roughness
        self.interface: bool = interface
        self.interface_thickness: float = interface_thickness
        self._thickness: float = float(thickness)
        self.optimize: bool = optimize
        self.needle: bool = needle
        self.layer_typ: int = layer_typ

        self._initialize_mask()
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

    # -- internal helpers --------------------------------------------------
    def _initialize_mask(self) -> None:
        self.mask = np.array([
            1,
            int(self.coherent),
            int(self._inhomogen),
            1 if self.rough_type > 0 else 0,
        ], dtype=INT_TYPE)

    def _refine_layer_count(self) -> None:
        if self._inhomogen and self._thickness > 0:
            factor = 1.0 + (self._inh_delta / 0.1) * 0.5
            self.sub_layer_count = int(np.ceil(self._thickness ** 0.4) * factor) + 1
        else:
            self.sub_layer_count = 1

    # -- serialisation (node-graph friendly) -------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Full serialisable snapshot (replaces get_properties)."""
        return {
            "thickness": self._thickness,
            "material": self.material,
            "coherent": self.coherent,
            "inhomogen": self._inhomogen,
            "inh_delta": self._inh_delta,
            "rough_type": self.rough_type,
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
            rough_type=state.get("rough_type", 0),
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
        for key, value in properties.items():
            if hasattr(self, key):
                setattr(self, key, value)
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
        obj.mask = self.mask.copy()
        return obj

    def __repr__(self) -> str:
        return (
            f"Layer(mat='{self.material}', d={self._thickness:.2f}nm, "
            f"rough={self.roughness:.2f}A, opt={self.optimize})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Group  (unchanged — solid design already)
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

        self.error_mask: List[int] = [0] * 6
        self.optimization_mask: List[int] = [0] * 7

        self.thickness_error_type = 0
        self.n_error_type = 0
        self.k_error_type = 0
        self.inh_delta_error_type = 0
        self.roughness_error_type = 0
        self.interface_error_type = 0

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

    # -- stochastic error application -------------------------------------
    @staticmethod
    def _apply_error(
        value: Any, error_type: int, error_params: Dict[str, float]
    ) -> Any:
        """Apply a stochastic manufacturing error to *value*."""
        rng = np.random

        if error_type == 0:  # Gaussian
            abs_err = rng.normal(error_params["abs_mean_delta_g"],
                                 error_params["abs_std_dev"])
            rel_err = rng.normal(error_params["rel_mean_delta_g"],
                                 error_params["rel_std_dev"]) * value
            return value + abs_err + rel_err

        if error_type == 1:  # Uniform
            abs_err = rng.uniform(-error_params["abs_variance"],
                                   error_params["abs_variance"])
            rel_err = rng.uniform(-error_params["rel_variance"],
                                   error_params["rel_variance"]) * value
            return value + abs_err + rel_err

        if error_type == 2:  # Combined
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

    def thickness_error(self, value: float) -> float:
        return max(0.0, self._apply_error(
            value, self.thickness_error_type, self.thickness_error_params))

    def inh_delta_error(self, value: float) -> float:
        return self._apply_error(
            value, self.inh_delta_error_type, self.inh_delta_error_params)

    def sr_roughness_error(self, value: float, thickness: float) -> float:
        return max(0.0, self._apply_error(
            value, self.roughness_error_type, self.roughness_error_params))

    def interface_error(self, value: float, thickness: float) -> float:
        return max(0.0, self._apply_error(
            value, self.interface_error_type, self.interface_error_params))

    def nk_error(self, nk_value: complex) -> complex:
        n_val = self._apply_error(nk_value.real, self.n_error_type, self.n_error_params)
        k_val = self._apply_error(nk_value.imag, self.k_error_type, self.k_error_params)
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
        for key, value in properties.items():
            if hasattr(self, key):
                setattr(self, key, value)

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


class _LayerExpander:
    """
    Stateless helper that expands a logical layer sequence into flat columnar
    arrays for the solver.

    This is the *single* implementation of the expansion logic —
    Loom_Structure and Loom_Architect both delegate here.

    The caller provides an iterator of (layer, is_inverted) tuples.  The
    expander handles interface generation, inhomogeneous subdivision,
    group-factor application, and optional stochastic errors.
    """

    @staticmethod
    def expand(
        layers: Iterator[Layer],
        materials: MaterialProvider,
        group_dict: Dict[str, Group],
        *,
        apply_errors: bool = False,
        invert_inhomogen: bool = False,
    ) -> SolverArrays:
        """
        Expand a sequence of layers into solver-ready columnar arrays.

        Parameters
        ----------
        layers : iterable of Layer
            Ordered layer sequence (ambient first, substrate last).
        materials : MaterialProvider
            Source for complex refractive indices.
        group_dict : dict[str, Group]
            Material-name → Group mapping for factors/errors.
        apply_errors : bool
            If True, apply stochastic manufacturing errors from groups.
        invert_inhomogen : bool
            If True, flip the gradient direction of inhomogeneous layers
            (used when the owning structure is traversed in reverse).

        Returns
        -------
        SolverArrays
        """
        col_thick: List[float] = []
        col_nk: List[Union[complex, np.ndarray]] = []
        col_coh: List[bool] = []
        col_r_val: List[float] = []
        col_r_type: List[int] = []

        get_group = group_dict.get
        prev_eff_nk: Optional[np.ndarray] = None

        for layer in layers:
            mat_name = layer.material
            group = get_group(mat_name, _DEFAULT_GROUP)

            # --- Base nk and thickness with systematic group factors ------
            base_nk = materials.get_nk(mat_name)

            if group.n_factor != 1.0 or group.k_factor != 1.0:
                layer_nk = base_nk * group.nk_factor
            else:
                layer_nk = base_nk

            layer_thickness = (
                layer.thickness * group.thick_factor + group.thick_summand
            )

            # --- Stochastic errors (when requested) -----------------------
            current_roughness = layer.roughness

            if apply_errors:
                # Mask 0: Thickness
                if group.error_mask[0]:
                    layer_thickness = group.thickness_error(layer_thickness)

                # Mask 1/2: n and k
                if group.error_mask[1] or group.error_mask[2]:
                    n_part = layer_nk.real
                    k_part = layer_nk.imag
                    if group.error_mask[1]:
                        n_part = Group._apply_error(
                            n_part, group.n_error_type, group.n_error_params
                        )
                        n_part = np.maximum(0.0, n_part)
                    if group.error_mask[2]:
                        k_part = Group._apply_error(
                            k_part, group.k_error_type, group.k_error_params
                        )
                    layer_nk = n_part + 1j * k_part

                # Mask 3: Roughness
                if group.error_mask[3]:
                    current_roughness = group.sr_roughness_error(
                        current_roughness, layer_thickness
                    )
            else:
                current_roughness = layer.roughness

            if layer_thickness < 0.0:
                layer_thickness = 0.0

            # --- A. Interface Generation ----------------------------------
            if layer.interface and prev_eff_nk is not None:
                t_interface = layer.interface_thickness

                if apply_errors and group.error_mask[5]:
                    t_interface = group.interface_error(
                        t_interface, layer.thickness
                    )

                if t_interface > layer_thickness:
                    t_interface = layer_thickness
                layer_thickness -= t_interface

                interface_nk = looyenga_eps(layer_nk, prev_eff_nk, 0.5)

                col_thick.append(t_interface)
                col_nk.append(interface_nk)
                col_coh.append(True)
                col_r_val.append(0.0)
                col_r_type.append(0)

            # --- B. Inhomogeneity Generation ------------------------------
            if layer._inhomogen and layer.sub_layer_count > 1:
                sub_div = layer.sub_layer_count

                current_delta = (
                    (layer._inh_delta + group.inh_delta_summand) * 0.5
                )

                if apply_errors and group.error_mask[4]:
                    current_delta = group.inh_delta_error(current_delta)

                factors = np.linspace(
                    1.0 - current_delta, 1.0 + current_delta, sub_div
                )
                if invert_inhomogen:
                    factors = factors[::-1]

                step_t = layer_thickness / sub_div

                for ix, f in enumerate(factors):
                    col_thick.append(step_t)
                    col_nk.append(layer_nk * f)
                    col_coh.append(layer.coherent)

                    if ix == 0:
                        col_r_val.append(current_roughness)
                        col_r_type.append(layer.rough_type)
                    else:
                        col_r_val.append(0.0)
                        col_r_type.append(0)

            # --- C. Standard Layer ----------------------------------------
            else:
                col_thick.append(layer_thickness)
                col_nk.append(layer_nk)
                col_coh.append(layer.coherent)
                col_r_val.append(current_roughness)
                col_r_type.append(layer.rough_type)

            prev_eff_nk = layer_nk

        # --- Final conversion to strict-typed arrays ----------------------
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
# 6.  Loom_Structure
# ═══════════════════════════════════════════════════════════════════════════════
class Loom_Structure:
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

    # -- material provider access (settable for TF_Architect injection) ----
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

        return _LayerExpander.expand(
            iter(self.layer_list),
            self._materials,
            self.group_dict,
            apply_errors=False,
        )

    def get_error_solver_inputs(self) -> SolverArrays:
        """
        Generate flat solver arrays WITH stochastic manufacturing errors.

        Returns
        -------
        SolverArrays
        """
        if not self.layer_list:
            raise ValueError("Structure is empty.")
        if self._materials is None:
            raise ValueError("No material provider set.")

        return _LayerExpander.expand(
            iter(self.layer_list),
            self._materials,
            self.group_dict,
            apply_errors=True,
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
    ) -> "Loom_Structure":
        """Reconstruct from a serialised dict."""
        layers = [Layer.from_state(ls) for ls in state.get("layers", [])]
        groups = {
            name: Group.from_state(gs)
            for name, gs in state.get("groups", {}).items()
        }
        return cls(layer_list=layers, group_dict=groups, materials=materials)

    def __repr__(self) -> str:
        return (
            f"Loom_Structure(layers={len(self.layer_list)}, "
            f"groups={len(self.group_dict)})"
        )

    def __len__(self) -> int:
        return len(self.layer_list)
