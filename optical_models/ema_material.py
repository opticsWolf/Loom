# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from typing import Dict, Union, Optional, Any
from numba import njit, prange

# Import your base class
from .material import Material

# Import your optimized EMA kernels
from .ema_models import (
    lichtenecker_eps,
    looyenga_eps,
    general_power_law_eps,
    maxwell_garnett_eps,
    bruggeman_eps,
    mori_tanaka_eps,
    roughness_interface_eps,
    wiener_bounds
)

__all__ = ["EMAMaterial", "RoughnessMaterial"]


# --- Optimized Helper Kernels ---

@njit(cache=True, fastmath=True, parallel=True)
def _parallel_sqrt(arr: np.ndarray) -> np.ndarray:
    """
    Calculates the square root of a complex array in parallel.
    
    Standard np.sqrt is serial. For complex numbers, the sqrt operation 
    is computationally heavy, making parallelization highly effective 
    for arrays larger than ~1,000 elements.
    """
    n = arr.size
    res = np.empty_like(arr)
    
    # We use a flat loop to handle any array shape efficiently
    flat_arr = arr.ravel()
    flat_res = res.ravel()
    
    for i in prange(n):
        flat_res[i] = np.sqrt(flat_arr[i])
        
    return res


class EffectiveMaterial(Material):
    """
    A composite material that calculates its optical properties using Effective Medium Approximations (EMA).
    
    This class wraps two other Material objects (host and inclusion) and mixes them
    according to a specified physics model (e.g., Bruggeman, Maxwell-Garnett).
    
    Attributes:
        host (Material): The host material (or Material A).
        inclusion (Material): The inclusion material (or Material B).
        fraction (float): Volume fraction of the inclusion (0.0 to 1.0).
        model (str): The name of the EMA model to use.
    """

    # Dispatch table mapping model names to Numba functions
    _MODEL_DISPATCH = {
        'bruggeman': bruggeman_eps,
        'maxwell_garnett': maxwell_garnett_eps,
        'looyenga': looyenga_eps,
        'lichtenecker': lichtenecker_eps,
        'mori_tanaka': mori_tanaka_eps,
        'birchak': general_power_law_eps,
    }

    def __init__(self, 
                 host: Material, 
                 inclusion: Material, 
                 fraction: float = 0.5,
                 model: str = 'bruggeman',
                 model_args: Optional[Dict[str, float]] = None,
                 wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Effective Material.

        Args:
            host: The host material object.
            inclusion: The inclusion material object.
            fraction: Volume fraction of the inclusion (0.0 <= f <= 1.0).
                      MUST be a scalar float. Arrays are not supported.
            model: Name of the model ('bruggeman', 'maxwell_garnett', 'looyenga', etc.).
            model_args: Dict of extra arguments for specific models (e.g., {'L': 0.33} for Mori-Tanaka).
            wavelength: Optional initial wavelength array.
        """
        # Store constituents
        self.host = host
        self.inclusion = inclusion
        
        # Strict scalar validation
        if not isinstance(fraction, (float, int)):
            raise TypeError(f"Volume fraction must be a scalar number, got {type(fraction)}")
        
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"Volume fraction must be between 0.0 and 1.0, got {fraction}")
            
        self.fraction = float(fraction)

        if model not in self._MODEL_DISPATCH:
            raise ValueError(f"Unknown EMA model '{model}'. Available: {list(self._MODEL_DISPATCH.keys())}")

        self.model_name = model
        self.model_func = self._MODEL_DISPATCH[model]
        self.model_args = model_args if model_args else {}

        # Dummy params for base class compatibility
        super().__init__({'A': 0.0}, wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """
        Set the wavelength range and propagate it to child materials.
        
        Includes an optimization check: if the wavelength array is identical 
        (by reference or value) to the current one, the update is skipped.
        Also checks children individually to prevent redundant updates on shared materials.
        """
        # 1. Self Check: Skip if identical object or equal values
        if self.wavelength is wavelength or (
            self.wavelength is not None and wavelength is not None and np.array_equal(self.wavelength, wavelength)
        ):
            return

        # 2. Update Self (calls base class implementation)
        super().set_wavelength_range(wavelength)
        
        # 3. Propagate to children (Host & Inclusion) only if needed
        # We iterate to avoid code duplication.
        for child in (self.host, self.inclusion):
            # Check Identity or Value equality against the NEW self.wavelength
            if not (child.wavelength is self.wavelength or (
                child.wavelength is not None and np.array_equal(child.wavelength, self.wavelength)
            )):
                child.set_wavelength_range(self.wavelength)
                child.complex_refractive_index() # Warm cache

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the effective refractive index N_eff.
        """
        # The optimization check is now inside set_wavelength_range.
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        # 1. Fetch N (n + ik) from constituents
        # Optimization: Direct attribute access + copy=False
        # We trust that set_wavelength_range has already warmed the cache (nk).
        n_h = self.host.nk.astype(np.complex128, copy=False)
        n_i = self.inclusion.nk.astype(np.complex128, copy=False)
        
        # 2. Execute EMA Kernel
        # v5 models accept scalar fraction 'f', removing O(N) allocation of f_arr
        if self.model_args:
             eps_eff = self.model_func(n_i, n_h, self.fraction, **self.model_args)
        else:
             eps_eff = self.model_func(n_i, n_h, self.fraction)

        # 3. Convert Permittivity to Refractive Index
        # Optimization: Use parallel Numba kernel instead of serial np.sqrt
        self.nk = _parallel_sqrt(eps_eff)
        
        return self.nk

    def get_params(self) -> Dict[str, Any]:
        """Return parameters including sub-material parameters."""
        return {
            'fraction': self.fraction,
            'model': self.model_name,
            'host_params': self.host.get_params(),
            'inclusion_params': self.inclusion.get_params()
        }


class RoughnessMaterial(EffectiveMaterial):
    """
    Specialized Effective Material for Interface/Roughness layers.
    
    Automatically forces the mixing fraction to 0.5 and uses the 
    'roughness_interface_eps' optimized kernel (Looyenga).
    """
    def __init__(self, 
                 bottom_material: Material, 
                 top_material: Material, 
                 wavelength: Optional[np.ndarray] = None):
        
        # We initialize the parent with dummy values because we override the calc method
        super().__init__(
            host=bottom_material, 
            inclusion=top_material, 
            fraction=0.5, 
            model='looyenga', 
            wavelength=wavelength
        )

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimized calculation for 50:50 roughness."""
        # The optimization check is now inside set_wavelength_range (inherited).
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        # Optimization: Direct attribute access + copy=False
        n_bot = self.host.nk.astype(np.complex128, copy=False)
        n_top = self.inclusion.nk.astype(np.complex128, copy=False)

        # Use the specific kernel for interfaces (f=0.5 hardcoded internally)
        # Optimization: Combined parallel sqrt + roughness calculation
        self.nk = _parallel_sqrt(roughness_interface_eps(n_bot, n_top))
        return self.nk