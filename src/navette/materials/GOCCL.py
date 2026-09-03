# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from navette.materials._numba import njit
from typing import Dict, Union, Optional, List, Tuple

# Try importing from local structure, mock if missing for standalone usage
try:
    from .material import Material, compute_energy
except ImportError:
    # Mock for standalone testing if package structure isn't present
    class Material:
        def __init__(self, params, wavelength=None): 
            self.A = params.get('A', 0)
            if wavelength is not None: self.set_wavelength_range(wavelength)
        def set_wavelength_range(self, wl): self.wavelength = np.asarray(wl, dtype=np.float64)
        def get_params(self): return {'A': self.A}
        
    @njit(cache=True)
    def compute_energy(wavelength, h_c): return h_c / wavelength

__all__ = ["MultiOscillatorCodyLorentz"]


# --- Numba Accelerated Helper Functions ---

@njit(cache=True)
def kramers_kronig_principal_value(E_target: np.ndarray, 
                                   E_source: np.ndarray, 
                                   eps2_source: np.ndarray, 
                                   epsilon_infinity: float = 1.0) -> np.ndarray:
    """
    Performs the Kramers-Kronig transformation on eps2 to calculate eps1 (eps1_KK).
    Uses trapezoidal integration with Principal Value singularity handling.
    """
    n_target = E_target.shape[0]
    eps1_target = np.zeros(n_target, dtype=np.float64)
    
    for i in range(n_target):
        Ei = E_target[i]
        integral_sum = 0.0
        
        for j in range(E_source.shape[0] - 1):
            Ej = E_source[j]
            Ej_next = E_source[j+1]
            
            # Skip the integration interval(s) adjacent to the singularity
            if np.abs(Ej - Ei) < 1e-9 or np.abs(Ej_next - Ei) < 1e-9:
                continue
            
            val_j = eps2_source[j]
            val_j_next = eps2_source[j+1]
            
            # Integrand: I(E') = (E' * eps2(E')) / (E'^2 - E_i^2)
            denom1 = Ej**2 - Ei**2
            term1 = (Ej * val_j) / denom1
            
            denom2 = Ej_next**2 - Ei**2
            term2 = (Ej_next * val_j_next) / denom2
            
            # Trapezoidal Rule
            dE = Ej_next - Ej
            integral_sum += 0.5 * (term1 + term2) * dE
            
        eps1_target[i] = epsilon_infinity + (2.0 / np.pi) * integral_sum
        
    return eps1_target


@njit(cache=True)
def _calc_eps2_components(E: np.ndarray,
                          ccl_params: np.ndarray,
                          lorentz_params: np.ndarray,
                          gaussian_params: np.ndarray) -> np.ndarray:
    """
    Internal helper to calculate total eps2 from all components.
    
    ccl_params: [Eg, Et, Ep, A_b, Eu, Gamma_b]
    lorentz_params: Nx3 [[A, E0, Gamma], ...]
    gaussian_params: Mx3 [[A, E0, C], ...]
    """
    # Unpack CCL params
    Eg, Et, Ep, A_b, Eu, Gamma_b = ccl_params[0], ccl_params[1], ccl_params[2], ccl_params[3], ccl_params[4], ccl_params[5]
    
    eps2_total = np.zeros_like(E, dtype=np.float64)
    E_sq = E**2

    # --- 1. Lorentz Defects (Lf) ---
    # Equation (2): Sum [ A * E0 * Gamma * E ] / [ (E0^2 - E^2)^2 + Gamma^2 E^2 ]
    for j in range(lorentz_params.shape[0]):
        A_L, E0_L, Gamma_L = lorentz_params[j]
        num_L = A_L * E0_L * Gamma_L * E
        den_L = (E0_L**2 - E_sq)**2 + (Gamma_L * E)**2
        eps2_total += num_L / np.maximum(den_L, 1e-12)

    # --- 2. Gaussian Defects (Gf) ---
    # Equation (3): Sum A * [ exp(-(E-E0)^2/C^2) - exp(-(E+E0)^2/C^2) ]
    for k in range(gaussian_params.shape[0]):
        A_G, E0_G, C_G = gaussian_params[k]
        C_sq = C_G**2
        term1 = np.exp(-(E - E0_G)**2 / C_sq)
        term2 = np.exp(-(E + E0_G)**2 / C_sq)
        eps2_total += A_G * (term1 - term2)

    # --- 3. Cody-Lorentz Core (CCL) ---
    # Equation (1): Piecewise definition
    
    # Pre-calculate common terms
    # Cody factor: (E-Eg)^2 / E^2
    cody_factor = np.zeros_like(E)
    mask_above_Eg = E > Eg
    cody_factor[mask_above_Eg] = (E[mask_above_Eg] - Eg)**2 / E_sq[mask_above_Eg]

    # Lorentz-like Band Term: E*Gamma / ((E^2-Ep^2)^2 + (E*Gamma)^2)
    denom_band = (E_sq - Ep**2)**2 + (E * Gamma_b)**2
    lorentz_band = (E * Gamma_b) / np.maximum(denom_band, 1e-12)
    
    # Calculate Amplitude at Et for continuity (A_t)
    Et_sq = Et**2
    if Et > Eg:
        cf_Et = (Et - Eg)**2 / Et_sq
    else:
        cf_Et = 0.0
    den_Et = (Et_sq - Ep**2)**2 + (Et * Gamma_b)**2
    lb_Et = (Et * Gamma_b) / np.maximum(den_Et, 1e-12)
    A_t = A_b * cf_Et * lb_Et

    # Fill Eps2 for CCL
    for i in range(E.shape[0]):
        Ei = E[i]
        
        val_ccl = 0.0
        if Ei < Et:
            # Urbach Tail: A_t * (Et/E) * exp((E-Et)/Eu)
            if Ei > Eg and Ei > 0: # Usually Urbach extends below Eg, but strictly > 0
                val_ccl = A_t * (Et / Ei) * np.exp((Ei - Et) / Eu)
        else:
            # Band-to-Band: A_b * Gc(E) * Lb(E)
            val_ccl = A_b * cody_factor[i] * lorentz_band[i]
            
        eps2_total[i] += val_ccl
        
    return np.maximum(eps2_total, 0.0)


@njit(cache=True)
def compute_goccl_complex_nk(E: np.ndarray,
                             ccl_params: np.ndarray,
                             lorentz_params: np.ndarray,
                             gaussian_params: np.ndarray,
                             eps_inf: float) -> np.ndarray:
    """
    Main driver function to compute the complex refractive index.
    
    Steps:
    1. Calculate eps2(E) using the GOCCL model (Defects + CCL).
    2. Calculate eps1(E) using numerical Kramers-Kronig transform.
    3. Convert complex epsilon to n + ik.
    """
    # 1. Calculate imaginary part
    eps2 = _calc_eps2_components(E, ccl_params, lorentz_params, gaussian_params)
    
    # 2. Calculate real part via KK
    # Note: E_target and E_source are the same (E)
    eps1 = kramers_kronig_principal_value(E, E, eps2, eps_inf)
    
    # 3. Convert to n+ik
    eps_complex = eps1 + 1j * eps2
    return np.sqrt(eps_complex)


# --- Main Class ---

class MultiOscillatorCodyLorentz(Material):
    """
    Multi-Oscillator Continuous Cody-Lorentz (GOCCL) Model.

    A comprehensive dispersion model for amorphous semiconductors and high-K dielectrics,
    combining a continuous bandgap/Urbach tail model with specific defect oscillators.

    Physics:
        eps2(E) = Lf(E) [Lorentz Defects] + Gf(E) [Gaussian Defects] + CCL(E) [Bandgap]
        eps1(E) = eps_inf + Kramers-Kronig{ eps2(E) }

    Attributes:
        params (dict): Must contain:
            - 'epsilon_inf': High-freq constant.
            - 'Eg', 'Et', 'Ep', 'A_b', 'Eu', 'Gamma_b': CCL Core parameters.
            - 'lorentz_params': List of tuples [(A, E0, Gamma), ...].
            - 'gaussian_params': List of tuples [(A, E0, C), ...].
    """

    def __init__(self,
                 params: Dict[str, Union[float, int, List, Dict]],
                 wavelength: Optional[np.ndarray] = None):
        """
        Initialize the GOCCL model.

        Args:
            params: Dictionary of parameters.
            wavelength: Optional initial wavelength array (nm).
        """
        # Pass a representative amplitude to parent (A_b from CCL)
        super().__init__({'A': params.get('A_b', 0.0)}, wavelength)
        
        self.epsilon_inf = float(params.get('epsilon_inf', 1.0))
        
        # --- 1. CCL Core Parameters ---
        # Eg: Bandgap, Et: Urbach Transition, Ep: Lorentz Peak, Eu: Urbach Energy
        self.ccl_dict = {
            'Eg': float(params.get('Eg', 0.0)),
            'Et': float(params.get('Et', 0.0)),
            'Ep': float(params.get('Ep', 0.0)),
            'A_b': float(params.get('A_b', 0.0)),
            'Eu': float(params.get('Eu', 0.1)), # Default non-zero to avoid div/0
            'Gamma_b': float(params.get('Gamma_b', 0.0))
        }
        
        # Pack CCL params for Numba: [Eg, Et, Ep, A_b, Eu, Gamma_b]
        self._ccl_params_arr = np.array([
            self.ccl_dict['Eg'], self.ccl_dict['Et'], self.ccl_dict['Ep'],
            self.ccl_dict['A_b'], self.ccl_dict['Eu'], self.ccl_dict['Gamma_b']
        ], dtype=np.float64)

        # --- 2. Lorentz Defect Parameters ---
        # Expected list of tuples: (A, E0, Gamma)
        l_params = params.get('lorentz_params', [])
        self._lorentz_params_arr = np.array(l_params, dtype=np.float64)
        if self._lorentz_params_arr.size == 0:
            self._lorentz_params_arr = np.zeros((0, 3), dtype=np.float64)
        elif self._lorentz_params_arr.ndim == 1:
            self._lorentz_params_arr = self._lorentz_params_arr.reshape(-1, 3)

        # --- 3. Gaussian Defect Parameters ---
        # Expected list of tuples: (A, E0, C)
        g_params = params.get('gaussian_params', [])
        self._gaussian_params_arr = np.array(g_params, dtype=np.float64)
        if self._gaussian_params_arr.size == 0:
            self._gaussian_params_arr = np.zeros((0, 3), dtype=np.float64)
        elif self._gaussian_params_arr.ndim == 1:
            self._gaussian_params_arr = self._gaussian_params_arr.reshape(-1, 3)
            
        # Constants
        self.h_c_by_eV_nm = 1239.8419843320028

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Sets the spectral range and converts wavelength (nm) to Energy (eV).

        Args:
            wavelength: Array of wavelengths in nanometers.
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm) 

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculates the complex refractive index (n + ik).
        
        Note: Because KK integration is global, this calculates n+ik over the 
        stored internal energy grid and then interpolates to the requested wavelengths.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        if not hasattr(self, 'E'):
             raise AttributeError("Wavelength range must be set.")

        self.nk = compute_goccl_complex_nk(
            self.E,
            self._ccl_params_arr,
            self._lorentz_params_arr,
            self._gaussian_params_arr,
            self.epsilon_inf
        )

        return self.nk

    def get_params(self) -> Dict:
        """Returns parameters dictionary."""
        return {
            'epsilon_inf': self.epsilon_inf,
            **self.ccl_dict,
            'lorentz_params': self._lorentz_params_arr.tolist(),
            'gaussian_params': self._gaussian_params_arr.tolist()
        }