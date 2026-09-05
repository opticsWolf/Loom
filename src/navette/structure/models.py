# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np

from .types import INT_TYPE, ErrorType, LayerType, OptMask, RoughnessType, ErrorMask, LayerMask

"""Layer and Group: the Python-side thin-film stack model.

A :class:`Layer` is one physical film (material name, thickness, coherence
and roughness flags); a :class:`Group` scales/couples layers for optimization
and error analysis. Both serialize via ``get_state``/``from_state`` for
config files, and the expander flattens them into solver arrays.
"""

class Layer:
    """One physical film: material, thickness [nm], coherence/roughness flags.

    Length units are nanometres everywhere: ``thickness``,
    ``interface_thickness`` and ``roughness`` (rms sigma, same unit as the
    solver wavelengths) are all [nm]. ``thickness``/``inhomogen``/
    ``inh_delta`` setters keep the solver sub-layer count in sync. Call
    the layer to get ``(material, thickness)``.
    """
    __slots__ = (
        "material", "coherent", "_inhomogen", "rough_type", "_inh_delta",
        "roughness", "interface", "interface_thickness", "_thickness",
        "optimize", "needle", "layer_type", "sub_layer_count",
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
        layer_type: Union[int, LayerType] = LayerType.FILM,
    ) -> None:
        self.material = material_name
        self.coherent = coherent
        self._inhomogen = inhomogen
        try:
            self.rough_type = RoughnessType(rough_type)
        except ValueError:
            raise ValueError(f"Layer: unknown rough_type {rough_type!r} (see RoughnessType).") from None
        self._inh_delta = inh_delta
        self.roughness = roughness
        self.interface = interface
        self.interface_thickness = interface_thickness
        self._thickness = float(thickness)
        self.optimize = optimize
        self.needle = needle
        try:
            self.layer_type = LayerType(layer_type)
        except ValueError:
            raise ValueError(f"Layer: unknown layer_type {layer_type!r} (see LayerType).") from None
        self._refine_layer_count()

    def __call__(self) -> Tuple[str, float]:
        return (self.material, self._thickness)

    @property
    def thickness(self) -> float:
        """Film thickness [nm]; setting it re-refines the sub-layer count."""
        return self._thickness
    @thickness.setter
    def thickness(self, value: float) -> None:
        self._thickness = float(value)
        if self._inhomogen: self._refine_layer_count()

    @property
    def inhomogen(self) -> bool:
        """Whether the film is graded (split into sub-layers for the solver)."""
        return self._inhomogen
    @inhomogen.setter
    def inhomogen(self, value: bool) -> None:
        self._inhomogen = bool(value)
        if self._inhomogen: self._refine_layer_count()

    @property
    def inh_delta(self) -> float:
        """Grading strength driving the sub-layer refinement."""
        return self._inh_delta
    @inh_delta.setter
    def inh_delta(self, value: float) -> None:
        self._inh_delta = float(value)
        if self._inhomogen: self._refine_layer_count()

    @property
    def mask(self) -> np.ndarray:
        """Per-layer status vector indexed by :class:`LayerMask`."""
        m = np.zeros(len(LayerMask), dtype=INT_TYPE)
        m[LayerMask.ACTIVE] = 1
        m[LayerMask.COHERENT] = int(self.coherent)
        m[LayerMask.INHOMOGEN] = int(self._inhomogen)
        m[LayerMask.ROUGHNESS] = int(self.rough_type != RoughnessType.NONE)
        return m

    def _refine_layer_count(self) -> None:
        if self._inhomogen and self._thickness > 0:
            factor = 1.0 + (self._inh_delta / 0.1) * 0.5
            self.sub_layer_count = int(np.ceil(self._thickness ** 0.4) * factor) + 1
        else:
            self.sub_layer_count = 1

    def get_state(self) -> Dict[str, Any]:
        """Serialize all layer properties to a plain dict (config files)."""
        return {
            "thickness": self._thickness,
            "material_name": self.material,
            "coherent": self.coherent,
            "inhomogen": self._inhomogen,
            "inh_delta": self._inh_delta,
            "rough_type": int(self.rough_type),
            "roughness": self.roughness,
            "interface": self.interface,
            "interface_thickness": self.interface_thickness,
            "optimize": self.optimize,
            "needle": self.needle,
            "layer_type": int(self.layer_type),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Layer":
        """Rebuild a layer from :meth:`get_state` output (unknown keys ignored)."""
        return cls(**{k: v for k, v in state.items() if k in cls.__init__.__code__.co_varnames})
        
    get_properties = get_state

    def set_properties(self, properties: Dict[str, Any]) -> None:
        """Bulk-set known properties; warns (not raises) on unknown/read-only keys."""
        for key, value in properties.items():
            if not hasattr(self, key):
                warnings.warn(f"Layer.set_properties: ignoring unknown attribute '{key}'.", stacklevel=2)
                continue
            if key in ("rough_type", "layer_type"):
                enum = RoughnessType if key == "rough_type" else LayerType
                try:
                    value = enum(value)
                except ValueError:
                    warnings.warn(f"Layer.set_properties: unknown {key} {value!r}; ignoring.", stacklevel=2)
                    continue
            try:
                setattr(self, key, value)
            except AttributeError:
                warnings.warn(f"Layer.set_properties: '{key}' is read-only; ignoring.", stacklevel=2)
        if self.interface or self._inhomogen:
            self._refine_layer_count()

    def clone(self) -> "Layer":
        """Independent copy sharing no mutable state."""
        obj = Layer.__new__(Layer)
        for attr in self.__slots__:
            setattr(obj, attr, getattr(self, attr))
        return obj

    def __repr__(self) -> str:
        return f"Layer(mat='{self.material}', d={self._thickness:.2f}nm, rough={self.roughness:.2f}nm, opt={self.optimize})"


class Group:
    """Scaling/error policy shared by a set of layers.

    Holds thickness/index multipliers plus per-channel fabrication-error
    models (:class:`ErrorType` + params). Error helpers (``*_error``)
    draw perturbed values; ``get_state``/``from_state`` serialize.
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
    # Roughness-channel defaults: identical shape, `abs_*` values x0.1 so
    # the physical error magnitude is unchanged by the A -> nm switch
    # (0.01 A == 0.001 nm). `rel_*` params are dimensionless: untouched.
    _DEFAULT_ROUGHNESS_ERROR_PARAMS: Dict[str, float] = {
        "abs_mean_delta_g": 0.0, "abs_std_dev": 0.001,
        "rel_mean_delta_g": 0.0, "rel_std_dev": 1.0,
        "abs_mean_delta_h": 0.0, "abs_variance": 0.001,
        "rel_mean_delta_h": 0.0, "rel_variance": 1.0,
    }

    def __init__(
        self, group_name: str, thick_factor: float = 1.0, thick_summand: float = 0.0,
        n_factor: float = 1.0, k_factor: float = 1.0, inh_delta_summand: float = 0.0,
        roughness_summand: float = 0.0, interface_summand: float = 0.0,
    ) -> None:
        self.group_name = group_name
        self.thick_factor, self.thick_summand = thick_factor, thick_summand
        self.n_factor, self.k_factor = n_factor, k_factor
        self.inh_delta_summand = inh_delta_summand
        self.roughness_summand = roughness_summand
        self.interface_summand = interface_summand

        self.error_mask = [0] * len(ErrorMask)
        self.optimization_mask = [1] * len(OptMask)
        
        for err_attr in ["thickness", "n", "k", "inh_delta", "roughness", "interface"]:
            setattr(self, f"{err_attr}_error_type", ErrorType.GAUSSIAN)
            if err_attr == "roughness":
                setattr(self, f"{err_attr}_error_params", self._DEFAULT_ROUGHNESS_ERROR_PARAMS.copy())
            else:
                setattr(self, f"{err_attr}_error_params", self._DEFAULT_ERROR_PARAMS.copy())

    @property
    def nk_factor(self) -> complex:
        return complex(self.n_factor, self.k_factor)

    def validate(self) -> List[str]:
        """Check factor domains (identity (1, 1); negatives unphysical)."""
        issues: List[str] = []
        if not (self.thick_factor >= 0.0):
            issues.append(f"Group '{self.group_name}': thick_factor {self.thick_factor} < 0 (NaN counts as invalid).")
        if not (self.n_factor >= 0.0):
            issues.append(f"Group '{self.group_name}': n_factor {self.n_factor} < 0 (no negative-index media).")
        if not (self.k_factor >= 0.0):
            issues.append(f"Group '{self.group_name}': k_factor {self.k_factor} < 0 (no gain media).")
        if len(self.optimization_mask) != len(OptMask) or any(v not in (0, 1) for v in self.optimization_mask):
            issues.append(f"Group '{self.group_name}': optimization_mask must be {len(OptMask)} binary entries (see OptMask).")
        return issues

    @staticmethod
    def _apply_error(value: Any, error_type: int, error_params: Dict[str, float], rng: Optional[np.random.Generator] = None) -> Any:
        rng = rng or np.random
        if error_type == ErrorType.GAUSSIAN:
            return value + rng.normal(error_params["abs_mean_delta_g"], error_params["abs_std_dev"]) + \
                   rng.normal(error_params["rel_mean_delta_g"], error_params["rel_std_dev"]) * value
        if error_type == ErrorType.UNIFORM:
            return value + rng.uniform(-error_params["abs_variance"], error_params["abs_variance"]) + \
                   rng.uniform(-error_params["rel_variance"], error_params["rel_variance"]) * value
        if error_type == ErrorType.COMBINED:
            return value + rng.normal(error_params["abs_mean_delta_g"], error_params["abs_std_dev"]) + \
                   rng.normal(error_params["rel_mean_delta_g"], error_params["rel_std_dev"]) * value + \
                   rng.uniform(-error_params["abs_variance"], error_params["abs_variance"]) + \
                   rng.uniform(-error_params["rel_variance"], error_params["rel_variance"]) * value
        return value

    def thickness_error(self, value: float, rng: Optional[np.random.Generator] = None) -> float:
        """Perturbed thickness (floored at 0)."""
        return max(0.0, self._apply_error(value, self.thickness_error_type, self.thickness_error_params, rng=rng))
    def inh_delta_error(self, value: float, rng: Optional[np.random.Generator] = None) -> float:
        """Perturbed grading strength."""
        return self._apply_error(value, self.inh_delta_error_type, self.inh_delta_error_params, rng=rng)
    def sr_roughness_error(self, value: float, thickness: float, rng: Optional[np.random.Generator] = None) -> float:
        """Perturbed surface roughness [nm] (floored at 0)."""
        return max(0.0, self._apply_error(value, self.roughness_error_type, self.roughness_error_params, rng=rng))
    def interface_error(self, value: float, thickness: float, rng: Optional[np.random.Generator] = None) -> float:
        """Perturbed interface width [nm] (floored at 0)."""
        return max(0.0, self._apply_error(value, self.interface_error_type, self.interface_error_params, rng=rng))
    def nk_error(self, nk_value: complex, rng: Optional[np.random.Generator] = None) -> complex:
        """Perturbed index with n floored at 0 (k untouched by the floor)."""
        n_val = self._apply_error(nk_value.real, self.n_error_type, self.n_error_params, rng=rng)
        k_val = self._apply_error(nk_value.imag, self.k_error_type, self.k_error_params, rng=rng)
        return complex(max(0.0, n_val), k_val)

    def get_state(self) -> Dict[str, Any]:
        """Serialize all slots to a plain dict (config files)."""
        return {attr: getattr(self, attr) for attr in self.__slots__}
    get_properties = get_state

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Group":
        """Rebuild a group from :meth:`get_state` output."""
        obj = cls(state.get("group_name", "default"))
        for key, value in state.items():
            if hasattr(obj, key): setattr(obj, key, value)
        return obj

    def set_properties(self, properties: Dict[str, Any]) -> None:
        """Bulk-set known properties (same warn-on-unknown policy as Layer)."""
        for key, value in properties.items():
            if not hasattr(self, key):
                warnings.warn(f"Group.set_properties: ignoring unknown attribute '{key}'.", stacklevel=2)
                continue
            try:
                setattr(self, key, value)
            except AttributeError:
                warnings.warn(f"Group.set_properties: '{key}' is read-only; ignoring.", stacklevel=2)

    def clone(self) -> "Group":
        obj = Group.__new__(Group)
        for attr in self.__slots__:
            val = getattr(self, attr)
            setattr(obj, attr, val.copy() if isinstance(val, (list, dict)) else val)
        return obj

    def __repr__(self) -> str:
        return f"Group(name='{self.group_name}', thick_factor={self.thick_factor:.3f})"