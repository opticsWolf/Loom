# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: cauchy_sellmeier.py — Dispersion models (Cauchy, Sellmeier) with optional Urbach absorption.

Optimized Version without verbose properties:
  - Removed most @property decorators
  - Kept essential ones for cache management and key access
"""

import numpy as np
from numba import njit
from typing import Dict, Union, Optional

from .material import Material, compute_energy


__all__ = ["Cauchy", "CauchyUrbach", "Sellmeier", "SellmeierUrbach"]

# Physical constant: h·c in eV·nm
_HC_EV_NM: float = 1239.8419843320028

@njit(cache=True)
def compute_cauchy_n_part(wavelength_µm_2: np.ndarray,
                           A: float,
                           B: float,
                           C: float) -> np.ndarray:
    """
    Compute real part of refractive index from Cauchy model.

    The Cauchy model describes chromatic dispersion using a polynomial
    expansion: n(λ) = A + B/λ² + C/λ⁴

    Args:
        wavelength_µm_2: Array of wavelengths in µm squared
        A: Constant term coefficient
        B: 1/λ² term coefficient
        C: 1/λ⁴ term coefficient

    Returns:
        Refractive index (real part) for given wavelengths
    """
    return A + B / wavelength_µm_2 + C / (wavelength_µm_2 ** 2)


@njit(cache=True)
def compute_cauchy_complex_nk(wavelength_µm_2: np.ndarray,
                              A: float,
                              B: float,
                              C: float) -> np.ndarray:
    """
    Compute complex refractive index from Cauchy model with k=0.

    Args:
        wavelength_µm_2: Array of wavelengths in µm squared
        A, B, C: Cauchy model coefficients for real part (n)

    Returns:
        Complex refractive index array where real part is from Cauchy model
        and imaginary part is set to 0.
    """
    n = compute_cauchy_n_part(wavelength_µm_2, A, B, C)
    k = np.zeros_like(wavelength_µm_2)
    return n + 1j * k


@njit(cache=True)
def compute_sellmeier_n_part(wavelength_µm_2: np.ndarray,
                        B1: float, C1: float,
                        B2: float, C2: float,
                        B3: float, C3: float) -> np.ndarray:
    """
    Compute refractive index from Sellmeier equation.

    The Sellmeier equation describes chromatic dispersion:
    n²(λ) = 1 + Σᵢ[Bᵢλ²/(λ² - Cᵢ)]

    Args:
        wavelength_µm_2: Array of wavelengths in µm squared
        B1-B3: Numerator coefficients for the Sellmeier equation
        C1-C3: Denominator coefficients

    Returns:
        Refractive index (n) as numpy array
    """
    term1 = B1 * wavelength_µm_2 / (wavelength_µm_2 - C1)
    term2 = B2 * wavelength_µm_2 / (wavelength_µm_2 - C2)
    term3 = np.where(B3 != 0.0, B3 * wavelength_µm_2 / (wavelength_µm_2 - C3), 0.0)
    
    n_squared = 1.0 + term1 + term2 + term3
    return np.sqrt(n_squared)


@njit(cache=True)
def compute_sellmeier_complex_nk(wavelength_µm_2: np.ndarray,
                             B1: float, C1: float,
                             B2: float, C2: float,
                             B3: float, C3: float) -> np.ndarray:
    """
    Compute complex refractive index for Sellmeier model with k=0.

    Args:
        wavelength_µm_2: Array of wavelengths in µm squared
        B1-B3: Numerator coefficients for the Sellmeier equation
        C1-C3: Denominator coefficients

    Returns:
        Complex refractive index array where real part is from Sellmeier model
        and imaginary part is set to 0.
    """
    n = compute_sellmeier_n_part(wavelength_µm_2, B1, C1, B2, C2, B3, C3)
    k = np.zeros_like(wavelength_µm_2)
    return n + 1j * k


@njit(cache=True)
def compute_urbach_k_part(wavelength_m: np.ndarray,
                            E: np.ndarray,
                            alpha0: float,
                            Eu: float,
                            lambda_g: float,
                            h_c: float) -> np.ndarray:
    """
    Compute Urbach extinction coefficient.

    The Urbach model describes exponential absorption near band gaps:
    k(λ) = α₀·exp((E - E₉)/Eᵤ) · λ/(4π)

    Args:
        wavelength_m: Array of wavelengths in m
        E: Corresponding array of photon energies
        alpha0: Absorption coefficient at band gap energy (1/cm)
        Eu: Urbach energy parameter (eV)
        lambda_g: Band gap wavelength (nm)
        h_c: Product of Planck's constant and speed of light

    Returns:
        Extinction coefficient k for the given wavelengths
    """
    k_part = np.zeros_like(E)
    E_g = h_c / lambda_g
    
    mask = E < E_g
    if np.any(mask):
        exponent = (E[mask] - E_g) / Eu
        absorption_coeff = alpha0 * np.exp(exponent)
        k_part[mask] = absorption_coeff * wavelength_m[mask] / (4 * np.pi)
        
    return k_part


@njit(cache=True)
def compute_cauchy_urbach_complex_nk(wavelength_m: np.ndarray, 
                             wavelength_µm_2: np.ndarray,
                             E: np.ndarray,
                             A: float,
                             B: float,
                             C: float,
                             alpha0: float,
                             Eu: float,
                             lambda_g: float,
                             h_c) -> np.ndarray:
    """
    Compute complex refractive index from Cauchy and Urbach models.

    Combines the Cauchy model (dispersion) with Urbach absorption (extinction).

    Args:
        wavelength_m: Array of wavelengths in m
        wavelength_µm_2: Array of wavelengths in µm squared
        E: Corresponding photon energies
        A, B, C: Cauchy model coefficients for real part (n)
        alpha0: Urbach absorption coefficient at band gap energy (1/cm)
        Eu: Urbach energy parameter (eV)
        lambda_g: Band gap wavelength (nm)
        h_c: Product of Planck's constant and speed of light

    Returns:
        Complex refractive index array where real part is from Cauchy model
        and imaginary part is from Urbach model extinction coefficient.
    """
    n = compute_cauchy_n_part(wavelength_µm_2, A, B, C)
    k = compute_urbach_k_part(wavelength_m, E, alpha0, Eu, lambda_g, h_c)
    return n + 1j * k


@njit(cache=True)
def compute_sellmeier_urbach_complex_nk(wavelength_m: np.ndarray,
                             wavelength_µm_2: np.ndarray,
                             E: np.ndarray,
                             B1: float, C1: float,
                             B2: float, C2: float,
                             B3: float, C3: float,
                             alpha0: float,
                             Eu: float,
                             lambda_g: float,
                             h_c: float) -> np.ndarray:
    """
    Compute complex refractive index for Sellmeier model with Urbach k.

    Combines the Sellmeier model (dispersion) with Urbach absorption (extinction).

    Args:
        wavelength_m: Array of wavelengths in m
        wavelength_µm_2: Array of wavelengths in µm squared
        E: Corresponding photon energies
        B1-B3: Numerator coefficients for the Sellmeier equation
        C1-C3: Denominator coefficients
        alpha0: Absorption coefficient at band gap energy (1/cm)
        Eu: Urbach energy parameter (eV)
        lambda_g: Band gap wavelength (nm)
        h_c: Product of Planck's constant and speed of light

    Returns:
        Complex refractive index array where real part is from Sellmeier model
        and imaginary part is from Urbach model extinction coefficient.
    """
    n = compute_sellmeier_n_part(wavelength_µm_2, B1, C1, B2, C2, B3, C3)
    k = compute_urbach_k_part(wavelength_m, E, alpha0, Eu, lambda_g, h_c)
    return n + 1j * k


class Cauchy(Material):
    """
    Cauchy dispersion model for transparent optical materials.

    The Cauchy model describes chromatic dispersion using a polynomial
    expansion: n(λ) = A + B/λ² + C/λ⁴

    This model is suitable for transparent materials in their transparency
    region (k = 0).

    Parameters:
        A: Constant term coefficient (required).
        B: 1/λ² term coefficient (required).
        C: 1/λ⁴ term coefficient (required).
        wavelength: Optional wavelength array in nanometers.
        params: Dictionary with 'A', 'B', 'C' (alternative to individual params).
        **kwargs: Additional parameters passed to base class.

    Examples:
        # Individual parameters (interactive use)
        mat = Cauchy(A=1.5, B=0.01, C=0.0001)

        # Dict-based (config files)
        params = {'A': 1.5, 'B': 0.01, 'C': 0.0001}
        mat = Cauchy(params=params)

        # With wavelength range
        wavelength = np.linspace(400, 700, 100)
        mat = Cauchy(A=1.5, B=0.01, C=0.0001, wavelength=wavelength)
    """
    
    def __init__(
        self,
        A: Optional[Union[float, int]] = None,
        B: Optional[Union[float, int]] = None,
        C: Optional[Union[float, int]] = None,
        wavelength: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs: Union[float, int]
    ):
        """
        Initialize the Cauchy dispersion model.

        Args:
            A: Constant term coefficient (required).
            B: 1/λ² term coefficient (required).
            C: 1/λ⁴ term coefficient (required).
            wavelength: Optional wavelength array in nanometers.
            params: Dictionary with 'A', 'B', 'C' (alternative to individual params).
            **kwargs: Additional parameters passed to base class.
        """
        p = params.copy() if params else {}
        if A is not None:
            p['A'] = A
        if B is not None:
            p['B'] = B
        if C is not None:
            p['C'] = C
        p.update(kwargs)

        super().__init__(wavelength=wavelength, params=p)
        self._validate_params(required=['A', 'B', 'C'])

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the complex refractive index (n + ik) with k=0.
        
        Uses self.params as single source of truth.

        Args:
            wavelength: Optional wavelength array in nanometers. If provided,
                updates internal wavelength range and recomputes nk.

        Returns:
            Complex refractive index array where real part is from Cauchy model
            and imaginary part is set to 0.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        self.nk = compute_cauchy_complex_nk(
            self.wavelength_µm_2,
            self.params['A'],
            self.params['B'],
            self.params['C']
        )
        return self.nk

    def __getattr__(self, name: str):
        """
        Enable attribute-style access to parameters.
        
        Only fires for names not found via normal attribute lookup,
        so internal attributes (``wavelength``, ``nk``, etc.) are 
        never intercepted.
        """
        # Guard against recursion during init / unpickling
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """
        Route parameter setting through ``set_param()`` for consistency.

        All parameters (A, B, C) are handled through the params dict
        and cache invalidation.
        """
        # Always bypass for internal / infrastructure attributes
        if name.startswith("_") or name in ("params", "wavelength", "nk", "E"):
            super().__setattr__(name, value)
            return

        # Only route through set_param if params dict exists and name is
        # already a known key - prevents hijacking of new attributes 
        # and avoids crashes during __init__ before params is populated.
        params = self.__dict__.get("params")
        if params is not None and name in params:
            self.set_param(name, value)
        else:
            super().__setattr__(name, value)


class CauchyUrbach(Material):
    """
    Cauchy dispersion model with Urbach absorption tail.

    Combines Cauchy dispersion (n) with Urbach absorption (k) to model
    materials near their band gap. The Cauchy model describes the
    refractive index in the transparent region, while the Urbach tail
    accounts for exponential band-edge absorption.

    Parameters:
        A, B, C: Cauchy coefficients (required).
        alpha0: Absorption coefficient at band gap energy in 1/cm (required).
        Eu: Urbach energy parameter in eV (required).
        lambda_g: Band gap wavelength in nm (required).
        wavelength: Optional wavelength array in nanometers.
        params: Dictionary with all parameters (alternative to individual params).
        **kwargs: Additional parameters passed to base class.

    Examples:
        # Individual parameters
        mat = CauchyUrbach(
            A=1.5, B=0.01, C=0.0001,
            alpha0=1e3, Eu=0.05, lambda_g=400
        )

        # Dict-based
        params = {
            'A': 1.5, 'B': 0.01, 'C': 0.0001,
            'alpha0': 1e3, 'Eu': 0.05, 'lambda_g': 400
        }
        mat = CauchyUrbach(params=params)
    """
    
    def __init__(
        self,
        A: Optional[Union[float, int]] = None,
        B: Optional[Union[float, int]] = None,
        C: Optional[Union[float, int]] = None,
        alpha0: Optional[Union[float, int]] = None,
        Eu: Optional[Union[float, int]] = None,
        lambda_g: Optional[Union[float, int]] = None,
        wavelength: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs: Union[float, int]
    ):
        """
        Initialize the CauchyUrbach model.

        Args:
            A: Cauchy coefficient (required).
            B: Cauchy coefficient (required).
            C: Cauchy coefficient (required).
            alpha0: Absorption coefficient at band gap energy in 1/cm (required).
            Eu: Urbach energy parameter in eV (required).
            lambda_g: Band gap wavelength in nm (required).
            wavelength: Optional wavelength array in nanometers.
            params: Dictionary with all parameters (alternative to individual params).
            **kwargs: Additional parameters passed to base class.
        """
        p = params.copy() if params else {}
        if A is not None:
            p['A'] = A
        if B is not None:
            p['B'] = B
        if C is not None:
            p['C'] = C
        if alpha0 is not None:
            p['alpha0'] = alpha0
        if Eu is not None:
            p['Eu'] = Eu
        if lambda_g is not None:
            p['lambda_g'] = lambda_g
        p.update(kwargs)

        super().__init__(wavelength=wavelength, params=p)
        self._validate_params(required=['A', 'B', 'C', 'alpha0', 'Eu', 'lambda_g'])

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the complex refractive index (n + ik).
        
        Uses self.params as single source of truth.

        Args:
            wavelength: Optional wavelength array in nanometers. If provided,
                updates internal wavelength range and recomputes nk.

        Returns:
            Complex refractive index array where real part is from Cauchy model
            and imaginary part is from Urbach model extinction coefficient.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        self.nk = compute_cauchy_urbach_complex_nk(
            self.wavelength_m,
            self.wavelength_µm_2,
            self.E,
            self.params['A'],
            self.params['B'],
            self.params['C'],
            self.params['alpha0'],
            self.params['Eu'],
            self.params['lambda_g'],
            _HC_EV_NM
        )
        return self.nk

    def __getattr__(self, name: str):
        """
        Enable attribute-style access to parameters.
        
        Only fires for names not found via normal attribute lookup,
        so internal attributes (``wavelength``, ``nk``, etc.) are 
        never intercepted.
        """
        # Guard against recursion during init / unpickling
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """
        Route parameter setting through ``set_param()`` for consistency.

        All parameters (A, B, C, alpha0, Eu, lambda_g) are handled through 
        the params dict and cache invalidation.
        """
        # Always bypass for internal / infrastructure attributes
        if name.startswith("_") or name in ("params", "wavelength", "nk", "E"):
            super().__setattr__(name, value)
            return

        # Only route through set_param if params dict exists and name is
        # already a known key - prevents hijacking of new attributes 
        # and avoids crashes during __init__ before params is populated.
        params = self.__dict__.get("params")
        if params is not None and name in params:
            self.set_param(name, value)
        else:
            super().__setattr__(name, value)


class Sellmeier(Material):
    """
    Sellmeier dispersion model using Schott convention.

    The Sellmeier equation describes chromatic dispersion:
    n²(λ) = 1 + Σᵢ[Bᵢλ²/(λ² - Cᵢ)]

    This model is widely used for optical glasses and crystals in their
    transparency region (k = 0).

    Parameters:
        B1, B2, B3: Numerator coefficients (B1 and B2 required, B3 defaults to 0).
        C1, C2, C3: Denominator coefficients (C1 and C2 required, C3 defaults to 0).
        wavelength: Optional wavelength array in nanometers.
        params: Dictionary with coefficients (alternative to individual params).
        **kwargs: Additional parameters passed to base class.

    Examples:
        # Individual parameters (BK7 glass)
        mat = Sellmeier(
            B1=1.03961212, C1=0.00600069867,
            B2=0.231792344, C2=0.0200179144,
            B3=1.01046945, C3=103.560653
        )

        # Dict-based
        params = {
            'B1': 1.03961212, 'C1': 0.00600069867,
            'B2': 0.231792344, 'C2': 0.0200179144,
            'B3': 1.01046945, 'C3': 103.560653
        }
        mat = Sellmeier(params=params)

        # Minimal (two-term Sellmeier)
        mat = Sellmeier(B1=1.0, C1=0.01, B2=0.2, C2=0.02)
    """
    
    def __init__(
        self,
        B1: Optional[Union[float, int]] = None,
        B2: Optional[Union[float, int]] = None,
        B3: Optional[Union[float, int]] = None,
        C1: Optional[Union[float, int]] = None,
        C2: Optional[Union[float, int]] = None,
        C3: Optional[Union[float, int]] = None,
        wavelength: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs: Union[float, int]
    ):
        """
        Initialize the Sellmeier dispersion model.

        Args:
            B1: First numerator coefficient (required).
            B2: Second numerator coefficient (required).
            B3: Third numerator coefficient (optional, defaults to 0.0).
            C1: First denominator coefficient (required).
            C2: Second denominator coefficient (required).
            C3: Third denominator coefficient (optional, defaults to 0.0).
            wavelength: Optional wavelength array in nanometers.
            params: Dictionary with coefficients (alternative to individual params).
            **kwargs: Additional parameters passed to base class.
        """
        p = params.copy() if params else {}
        if B1 is not None:
            p['B1'] = B1
        if B2 is not None:
            p['B2'] = B2
        if B3 is not None:
            p['B3'] = B3
        if C1 is not None:
            p['C1'] = C1
        if C2 is not None:
            p['C2'] = C2
        if C3 is not None:
            p['C3'] = C3
        p.update(kwargs)

        super().__init__(wavelength=wavelength, params=p)
        self._validate_params(
            required=['B1', 'B2', 'C1', 'C2'],
            optional={'B3': 0.0, 'C3': 0.0}
        )

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the complex refractive index (n + ik) with k=0.
        
        Uses self.params as single source of truth.

        Args:
            wavelength: Optional wavelength array in nanometers. If provided,
                updates internal wavelength range and recomputes nk.

        Returns:
            Complex refractive index array where real part is from Sellmeier model
            and imaginary part is set to 0.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        self.nk = compute_sellmeier_complex_nk(
            self.wavelength_µm_2,
            self.params['B1'], self.params['C1'],
            self.params['B2'], self.params['C2'],
            self.params['B3'], self.params['C3']
        )
        return self.nk

    def __getattr__(self, name: str):
        """
        Enable attribute-style access to parameters.
        
        Only fires for names not found via normal attribute lookup,
        so internal attributes (``wavelength``, ``nk``, etc.) are 
        never intercepted.
        """
        # Guard against recursion during init / unpickling
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """
        Route parameter setting through ``set_param()`` for consistency.

        All parameters (B1, B2, B3, C1, C2, C3) are handled through 
        the params dict and cache invalidation.
        """
        # Always bypass for internal / infrastructure attributes
        if name.startswith("_") or name in ("params", "wavelength", "nk", "E"):
            super().__setattr__(name, value)
            return

        # Only route through set_param if params dict exists and name is
        # already a known key - prevents hijacking of new attributes 
        # and avoids crashes during __init__ before params is populated.
        params = self.__dict__.get("params")
        if params is not None and name in params:
            self.set_param(name, value)
        else:
            super().__setattr__(name, value)


class SellmeierUrbach(Material):
    """
    Sellmeier dispersion model with Urbach absorption tail.

    Combines Sellmeier dispersion (n) with Urbach absorption (k) to model
    materials near their band gap. The Sellmeier model describes the
    refractive index in the transparent region, while the Urbach tail
    accounts for exponential band-edge absorption.

    Parameters:
        B1, B2, B3: Sellmeier numerator coefficients.
        C1, C2, C3: Sellmeier denominator coefficients.
        alpha0: Absorption coefficient at band gap energy in 1/cm (required).
        Eu: Urbach energy parameter in eV (required).
        lambda_g: Band gap wavelength in nm (required).
        wavelength: Optional wavelength array in nanometers.
        params: Dictionary with all parameters (alternative to individual params).
        **kwargs: Additional parameters passed to base class.

    Examples:
        # Individual parameters
        mat = SellmeierUrbach(
            B1=1.0, C1=0.01, B2=0.2, C2=0.02, B3=0.0, C3=0.0,
            alpha0=1e3, Eu=0.05, lambda_g=400
        )

        # Dict-based
        params = {
            'B1': 1.0, 'C1': 0.01, 'B2': 0.2, 'C2': 0.02,
            'B3': 0.0, 'C3': 0.0,
            'alpha0': 1e3, 'Eu': 0.05, 'lambda_g': 400
        }
        mat = SellmeierUrbach(params=params)
    """
    
    def __init__(
        self,
        B1: Optional[Union[float, int]] = None,
        B2: Optional[Union[float, int]] = None,
        B3: Optional[Union[float, int]] = None,
        C1: Optional[Union[float, int]] = None,
        C2: Optional[Union[float, int]] = None,
        C3: Optional[Union[float, int]] = None,
        alpha0: Optional[Union[float, int]] = None,
        Eu: Optional[Union[float, int]] = None,
        lambda_g: Optional[Union[float, int]] = None,
        wavelength: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs: Union[float, int]
    ):
        """
        Initialize the SellmeierUrbach model.

        Args:
            B1: First numerator coefficient (required).
            B2: Second numerator coefficient (required).
            B3: Third numerator coefficient (optional, defaults to 0.0).
            C1: First denominator coefficient (required).
            C2: Second denominator coefficient (required).
            C3: Third denominator coefficient (optional, defaults to 0.0).
            alpha0: Absorption coefficient at band gap energy in 1/cm (required).
            Eu: Urbach energy parameter in eV (required).
            lambda_g: Band gap wavelength in nm (required).
            wavelength: Optional wavelength array in nanometers.
            params: Dictionary with all parameters (alternative to individual params).
            **kwargs: Additional parameters passed to base class.
        """
        p = params.copy() if params else {}
        if B1 is not None:
            p['B1'] = B1
        if B2 is not None:
            p['B2'] = B2
        if B3 is not None:
            p['B3'] = B3
        if C1 is not None:
            p['C1'] = C1
        if C2 is not None:
            p['C2'] = C2
        if C3 is not None:
            p['C3'] = C3
        if alpha0 is not None:
            p['alpha0'] = alpha0
        if Eu is not None:
            p['Eu'] = Eu
        if lambda_g is not None:
            p['lambda_g'] = lambda_g
        p.update(kwargs)

        super().__init__(wavelength=wavelength, params=p)
        self._validate_params(
            required=['B1', 'B2', 'C1', 'C2', 'alpha0', 'Eu', 'lambda_g'],
            optional={'B3': 0.0, 'C3': 0.0}
        )

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the complex refractive index (n + ik).
        
        Uses self.params as single source of truth.

        Args:
            wavelength: Optional wavelength array in nanometers. If provided,
                updates internal wavelength range and recomputes nk.

        Returns:
            Complex refractive index array where real part is from Sellmeier model
            and imaginary part is from Urbach model extinction coefficient.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        self.nk = compute_sellmeier_urbach_complex_nk(
            self.wavelength_m,
            self.wavelength_µm_2,
            self.E,
            self.params['B1'], self.params['C1'],
            self.params['B2'], self.params['C2'],
            self.params['B3'], self.params['C3'],
            self.params['alpha0'],
            self.params['Eu'],
            self.params['lambda_g'],
            _HC_EV_NM
        )
        return self.nk

    def __getattr__(self, name: str):
        """
        Enable attribute-style access to parameters.
        
        Only fires for names not found via normal attribute lookup,
        so internal attributes (``wavelength``, ``nk``, etc.) are 
        never intercepted.
        """
        # Guard against recursion during init / unpickling
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """
        Route parameter setting through ``set_param()`` for consistency.

        All parameters (B1, B2, B3, C1, C2, C3, alpha0, Eu, lambda_g) are 
        handled through the params dict and cache invalidation.
        """
        # Always bypass for internal / infrastructure attributes
        if name.startswith("_") or name in ("params", "wavelength", "nk", "E"):
            super().__setattr__(name, value)
            return

        # Only route through set_param if params dict exists and name is
        # already a known key - prevents hijacking of new attributes 
        # and avoids crashes during __init__ before params is populated.
        params = self.__dict__.get("params")
        if params is not None and name in params:
            self.set_param(name, value)
        else:
            super().__setattr__(name, value)
