# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: material.py — Base class for optical materials.

v3 Updates:
  - Hybrid parameter API: supports both dict-based and individual parameters
  - Single source of truth: self.params is the only place parameters are stored
  - Centralized cache invalidation through set_param()
  - Validation helpers for consistent parameter checking
  - Config-file ready while maintaining ergonomic API
"""

import numpy as np
from numba import njit
from typing import Dict, Union, Optional, List


@njit(cache=True)
def compute_energy(wavelength: np.ndarray, h_c: float) -> np.ndarray:
    """
    Compute photon energy from wavelength.

    Args:
        wavelength: Array of wavelengths in nanometers.
        h_c: Product of Planck's constant and speed of light
             (h·c in J·nm, or h·c/eV in eV·nm).

    Returns:
        Photon energy array.
    """
    return h_c / wavelength


class Material:
    """
    Base class for optical materials with complex refractive index.

    Subclasses must override ``real_part()`` and ``imag_part()`` (or
    override ``complex_refractive_index()`` entirely, as Konstant and
    TableMaterial do).

    After ``complex_refractive_index(wavelength)`` is called, the
    result is cached in ``self.nk`` (complex128 ndarray).

    The class uses a hybrid parameter API supporting both dict-based
    and keyword argument initialization, with self.params as the single
    source of truth for all material parameters.

    Attributes:
        params : Dict[str, float]
            Dictionary of material parameters (single source of truth).
        wavelength : np.ndarray | None
            Current wavelength grid (nm). None if not yet set.
        nk : np.ndarray | None
            Cached complex refractive index. None until first
            call to ``complex_refractive_index()``.
    """

    def __init__(
        self,
        wavelength: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs: Union[float, int]
    ):
        """
        Initialize material with parameters.

        Args:
            wavelength: Optional wavelength array in nanometers.
            params: Dictionary of material parameters.
            **kwargs: Individual parameters (override params dict).
        
        Examples:
            # Dict-based (good for config files)
            Material(params={'n': 1.5, 'k': 0.01})
            
            # Keyword-based (good for interactive use)
            Material(n=1.5, k=0.01)
            
            # Hybrid (kwargs override params)
            Material(params={'n': 1.5, 'k': 0.01}, n=1.6)
        """
        # Merge logic: kwargs override params dict
        # Scalar values are cast to float; non-scalar values (e.g. lists
        # of oscillator tuples) are stored as-is for subclasses to handle.
        initial_params = params or {}
        merged = {**initial_params, **kwargs}
        self.params: Dict[str, Union[float, any]] = {}
        for k, v in merged.items():
            if isinstance(v, (int, float, np.number)):
                self.params[k] = float(v)
            else:
                self.params[k] = v
        
        # Ensure these always exist (prevent AttributeError in providers)
        self.wavelength: Optional[np.ndarray] = None
        self.nk: Optional[np.ndarray] = None

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def _validate_params(
        self,
        required: Optional[List[str]] = None,
        optional: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Validate and set defaults for parameters.

        Args:
            required: List of required parameter names.
            optional: Dict of {param_name: default_value} for optional params.
        
        Raises:
            ValueError: If a required parameter is missing.
        """
        if required:
            for param in required:
                if param not in self.params:
                    raise ValueError(
                        f"Parameter '{param}' is required for {self.__class__.__name__}."
                    )
        
        if optional:
            for param, default in optional.items():
                self.params.setdefault(param, default)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """
        Set the wavelength range for calculations.
        
        Invalidates cached nk when grid changes.
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        # Invalidate cached nk when grid changes
        self.nk = None

    def complex_refractive_index(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute and cache the complex refractive index (n + ik).

        If wavelength is provided, the internal grid is updated first.

        Returns:
            complex128 ndarray of shape (n_wavelengths,).
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        n = self.real_part()
        k = self.imag_part()
        self.nk = n + 1j * k
        return self.nk

    def real_part(self) -> np.ndarray:
        """Override in subclass: return n(λ) array."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement real_part()"
        )

    def imag_part(self) -> np.ndarray:
        """Override in subclass: return k(λ) array."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement imag_part()"
        )

    def get_params(self) -> Dict[str, float]:
        """
        Return a copy of material parameters.
        
        Returns:
            Dictionary copy of self.params.
        """
        return self.params.copy()

    def set_param(self, param_name: str, value: Union[float, int]) -> None:
        """
        Set a material parameter by name with cache invalidation.

        Args:
            param_name: Name of the parameter to set.
            value: New numeric value for the parameter.
        
        Raises:
            TypeError: If value is not numeric.
        """
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(
                f"Parameter '{param_name}' must be numeric, got {type(value).__name__}"
            )
        
        self.params[param_name] = float(value)
        # Invalidate cache when parameters change
        self.nk = None