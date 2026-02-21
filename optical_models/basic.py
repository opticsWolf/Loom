# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: basic.py — Constant and tabulated refractive index materials.

v3 Updates:
  - Hybrid parameter API for all classes
  - Single source of truth: self.params for all parameters
  - Property accessors for convenient attribute-style access
  - No redundant attributes (n, k, etc. are only in self.params)
  - Config-file ready while maintaining ergonomic API
  
Bug fixes from v2:
  1. TableMaterial.complex_refractive_index: self.k_factpr → self.k_factor
  2. TableMaterial._interpolate_data: `if data:` → `if data is not None:`
     (numpy truthiness trap — an array of length ≠ 1 raises ValueError)
"""

import numpy as np
from numba import njit
from typing import Dict, Union, Optional
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator

from .material import Material

__all__ = ["Konstant", "TableMaterial"]


class Konstant(Material):
    """
    Material with constant real (n) and imaginary (k) refractive indices.

    Useful for simple media or reference materials where optical properties
    don't vary with wavelength.

    Parameters:
        n: Real refractive index (required, must be > 0).
        k: Imaginary refractive index / extinction coefficient (optional, default 0.0).
        wavelength: Optional wavelength array in nanometers.
        params: Dictionary with 'n' and optionally 'k' (alternative to individual params).
        **kwargs: Additional parameters passed to base class.
    
    Examples:
        # Individual parameters (interactive use)
        mat = Konstant(n=1.5, k=0.01)
        
        # Dict-based (config files)
        mat = Konstant(params={'n': 1.5, 'k': 0.01})
        
        # Hybrid
        mat = Konstant(params={'n': 1.5}, k=0.01)
        
        # Access via properties
        print(mat.n)  # 1.5
        
        # Modify via set_param
        mat.set_param('n', 1.6)
        print(mat.n)  # 1.6
    """

    def __init__(
        self,
        n: Optional[Union[float, int]] = None,
        k: Optional[Union[float, int]] = None,
        wavelength: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs: Union[float, int]
    ):
        # Consolidate inputs: individual params override dict params
        p = params.copy() if params else {}
        if n is not None:
            p['n'] = n
        if k is not None:
            p['k'] = k
        p.update(kwargs)

        # Initialize base class with merged parameters
        super().__init__(wavelength=wavelength, params=p)
        
        # Validate required parameters and set defaults
        self._validate_params(required=['n'], optional={'k': 0.0})
        
        # Additional validation for physical constraints
        if self.params['n'] <= 0:
            raise ValueError(f"Refractive index n must be > 0, got {self.params['n']}")
        if self.params['k'] < 0:
            raise ValueError(f"Extinction coefficient k must be >= 0, got {self.params['k']}")

    @property
    def n(self) -> float:
        """Real refractive index (read-only convenience accessor)."""
        return self.params['n']

    @property
    def k(self) -> float:
        """Extinction coefficient (read-only convenience accessor)."""
        return self.params['k']

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Convert wavelength array and invalidate cached nk."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.nk = None

    def complex_refractive_index(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return the complex refractive index (n + ik).
        
        Uses self.params as single source of truth - no redundant attributes.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        # Use self.params directly - single source of truth
        n_arr = np.full(self.wavelength.shape, self.params['n'], dtype=np.float64)
        k_arr = np.full(self.wavelength.shape, self.params['k'], dtype=np.float64)
        
        self.nk = n_arr + 1j * k_arr
        return self.nk


class TableMaterial(Material):
    """
    Material with tabulated refractive index and extinction coefficient data.

    Supports multiple interpolation methods: linear, cubicspline, pchip,
    akima, makima.

    Parameters:
        n_data: 2-element tuple/list of (wavelength_array, n_values_array).
        k_data: Same format for extinction coefficient (optional, defaults to 0).
        n_factor: Scaling factor for n values (default 1.0).
        k_factor: Scaling factor for k values (default 1.0).
        interpolation_type_n: Interpolation method for n.
        interpolation_type_k: Interpolation method for k.
        wavelength: Optional target wavelength grid.
        params: Dictionary with 'n_factor' and 'k_factor' (alternative to individual params).
        **kwargs: Additional parameters passed to base class.
    
    Examples:
        # Individual parameters
        n_data = ([400, 500, 600], [1.5, 1.48, 1.46])
        mat = TableMaterial(n_data=n_data, n_factor=1.0)
        
        # Dict-based
        mat = TableMaterial(n_data=n_data, params={'n_factor': 1.0, 'k_factor': 1.0})
        
        # Access via properties
        print(mat.n_factor)  # 1.0
        
        # Modify via set_param
        mat.set_param('n_factor', 1.05)
    """

    def __init__(
        self,
        n_data: np.ndarray,
        k_data: Optional[np.ndarray] = None,
        n_factor: Optional[Union[float, int]] = None,
        k_factor: Optional[Union[float, int]] = None,
        interpolation_type_n: str = "linear",
        interpolation_type_k: str = "linear",
        wavelength: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs: Union[float, int]
    ):
        # Consolidate inputs: individual params override dict params
        p = params.copy() if params else {}
        if n_factor is not None:
            p['n_factor'] = n_factor
        if k_factor is not None:
            p['k_factor'] = k_factor
        p.update(kwargs)

        # Initialize base class with merged parameters
        super().__init__(wavelength=wavelength, params=p)
        
        # Validate and set defaults
        self._validate_params(optional={'n_factor': 1.0, 'k_factor': 1.0})

        # Store interpolation settings (not in params - these are method choices)
        self.interpolation_type_n = interpolation_type_n
        self.interpolation_type_k = interpolation_type_k

        # Store raw data (not in params - these are data arrays)
        self.n_data = n_data  # tuple of (wavelengths, values)
        self.k_data = k_data

        # Interpolation caches (invalidated on wavelength change)
        self.n_interp: Optional[np.ndarray] = None
        self.k_interp: Optional[np.ndarray] = None

    def _interpolate_data(
        self,
        data_type: str,
        data: Optional[object],
        interpolation_type: str = "linear",
    ) -> np.ndarray:
        """
        Interpolate tabulated data onto self.wavelength.

        Args:
            data_type: 'n' or 'k' (determines fallback: ones or zeros).
            data: Tuple/list of (wavelengths, values), or None for fallback.
            interpolation_type: 'linear', 'cubicspline', 'pchip', 'akima',
                                or 'makima'.

        Returns:
            Interpolated values at self.wavelength.
        """
        if self.wavelength is None:
            raise ValueError("self.wavelength must be set before interpolation.")

        # Use `is not None` instead of truthiness test (numpy array bug)
        if data is not None:
            wvl, vals = data

            methods = {
                "linear": lambda w, v: np.interp(self.wavelength, w, v),
                "cubicspline": lambda w, v: CubicSpline(
                    w, v, extrapolate=True
                )(self.wavelength),
                "pchip": lambda w, v: PchipInterpolator(
                    w, v, extrapolate=True
                )(self.wavelength),
                "akima": lambda w, v: Akima1DInterpolator(
                    w, v, method="akima", extrapolate=True
                )(self.wavelength),
                "makima": lambda w, v: Akima1DInterpolator(
                    w, v, method="makima", extrapolate=True
                )(self.wavelength),
            }

            if interpolation_type in methods:
                return methods[interpolation_type](wvl, vals)
            else:
                raise ValueError(
                    f"Unknown interpolation type '{interpolation_type}'. "
                    f"Choose from: {list(methods.keys())}"
                )
        else:
            # Fallback for missing data
            if data_type == "n":
                return np.ones_like(self.wavelength)
            else:  # 'k'
                return np.zeros_like(self.wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the wavelength range, invalidating interpolation caches."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.n_interp = None
        self.k_interp = None
        self.nk = None

    def complex_refractive_index(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return the complex refractive index (n + ik).
        
        Uses self.params as single source of truth for scaling factors.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            # Recursively call to compute nk with new wavelength
            return self.complex_refractive_index()

        # Lazy interpolation: compute only if cache is invalid
        if self.n_interp is None:
            self.n_interp = self._interpolate_data(
                "n", self.n_data, self.interpolation_type_n
            )
        if self.k_interp is None:
            self.k_interp = self._interpolate_data(
                "k", self.k_data, self.interpolation_type_k
            )

        # Use self.params directly - single source of truth
        self.nk = (
            self.params['n_factor'] * self.n_interp + 
            1j * self.params['k_factor'] * self.k_interp
        )
        return self.nk

    def set_param(self, param_name: str, value: Union[float, int]) -> None:
        """
        Set a material parameter by name.
        
        Invalidates both nk cache and interpolation caches since factors changed.
        """
        if param_name not in self.params:
            raise AttributeError(
                f"Parameter '{param_name}' does not exist in TableMaterial. "
                f"Available parameters: {list(self.params.keys())}"
            )
        
        # Call base class set_param for validation and cache invalidation
        super().set_param(param_name, value)
        
        # Note: We don't need to invalidate n_interp/k_interp here because
        # they are independent of the scaling factors. Only self.nk needs
        # to be recomputed, which is already handled by base class.
