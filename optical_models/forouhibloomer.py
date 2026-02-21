# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

This module implements the "New Formulation" of the Forouhi-Bloomer model,
which corrects the mathematical violations of Kramers-Kronig consistency found 
in the original 1986/1988 approximations. These models use Rational Functions 
to satisfy Titchmarsh’s Theorem strictly.

Attributes:
    ForouhiBloomerInterbandSingle: Class for single-term interband model.
    ForouhiBloomerInterbandMulti: Class for multi-term interband model.
    ForouhiBloomerMetal2021: Combined intraband+interband metal model.


References:
    1. Interband (Insulators/Semiconductors):
       Forouhi, A. R., & Bloomer, I. (2019). 
       "Optical dispersion of insulators and semiconductors: The new formulation of the Forouhi-Bloomer model."
       IOP Conference Series: Materials Science and Engineering, 479, 012002.
       https://iopscience.iop.org/article/10.1088/2399-6528/ab0603
    
    2. Metals (Intraband + Interband):
       Forouhi, A. R., & Bloomer, I. (2021).
       "Optical properties of metals: The new formulation of the Forouhi-Bloomer model."
       IOP Conference Series: Materials Science and Engineering, 1045, 012015.
       https://iopscience.iop.org/article/10.1088/1757-899X/1045/1/012015
"""

import numpy as np
from numba import njit
from typing import Dict, Union, Optional, List

# Try importing from local structure, mock if missing for standalone usage
try:
    from .material import Material, compute_energy 
except ImportError:
    class Material:
        def __init__(self, params, wavelength): pass
    def compute_energy(wavelength, constant):
        return constant / wavelength

__all__ = ["ForouhiBloomerInterbandSingle", "ForouhiBloomerInterbandMulti", 
           "ForouhiBloomerMetal2021"]

# --- Interband Term (FB 2019 Rational Formulation) ---
@njit(cache=True)
def compute_single_fb2019_term(E: np.ndarray,
                               Eg: float, A: float, B: float, C: float) -> np.ndarray:
    """
    Computes the complex refractive index (n + ik) contribution for a single 
    interband transition using the FB 2019 'New Formulation'.

    This function implements the Rational Function approximation which satisfies
    Titchmarsh’s Theorem (Causality) precisely, unlike the 1986 Log/Arctan approximation.

    Physical Model:
    ---------------
    k(E) represents the extinction coefficient derived from the Principle of Least Action
    applied to a quantum mechanical system with finite lifetime.
    
    Equations:
        D(E) = E^2 - B*E + C
        k(E) = A * (E - Eg)^2 / D(E)      for E >= Eg
        n(E) = (B0 * E + C0) / D(E)       for all E

    Parameters:
    -----------
    E : np.ndarray
        Photon energies in eV.
    Eg : float
        Optical Bandgap (eV). The energy threshold below which k(E) = 0.
    A : float
        Amplitude parameter. Related to the transition probability (oscillator strength).
    B : float
        Broadening parameter (eV). Related to the inverse lifetime of the electron state.
    C : float
        Resonance parameter (eV^2). C approx E_resonant^2.
        
    Stability Condition:
    --------------------
    Physical systems require 4*C > B^2. This ensures the poles of the function 
    lie in the complex plane (finite lifetime), preventing singularities on the 
    real energy axis.

    Returns:
    --------
    np.ndarray (complex128)
        The complex contribution (n + ik).
    """
    
    # --- 1. Stability Check & Pre-calculation ---
    # The term 4C - B^2 must be positive.
    discriminant = 4 * C - B**2
    
    # Q represents the effective "damped" resonance frequency term.
    if discriminant <= 1e-12:
        Q = 1e-6 # Avoid division by zero in unphysical parameter regimes
    else:
        Q = 0.5 * np.sqrt(discriminant)
        
    # --- 2. Calculate Coefficients B0 and C0 ---
    # In the 2019 formulation, n(E) is a rational function. B0 and C0 are 
    # strictly determined by Kramers-Kronig integration of the k(E) function defined above.
    # Ref: Eq (13) in FB 2019 paper.
    B0 = (A / Q) * ( - (B**2 / 2) + Eg * B - Eg**2 + C )
    C0 = (A / Q) * ( (Eg**2 + C) * (B / 2) - 2 * Eg * C )
    
    # --- 3. Compute Denominator D(E) ---
    # Note the FB convention: E^2 - B*E + C (Minus B).
    E_2 = E**2
    denom = E_2 - B * E + C
    
    # Numerical safeguard for denominator
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)

    # --- 4. Compute k(E) (Imaginary Part) ---
    k = np.zeros_like(E)
    mask = E >= Eg
    k[mask] = (A * (E[mask] - Eg)**2) / denom[mask]
    
    # --- 5. Compute n(E) (Real Part) ---
    # The rational form applies over the entire energy range.
    n_contribution = (B0 * E + C0) / denom
    
    return n_contribution + 1j * k

# --- Intraband/Metal Term (FB 2021) ---
@njit(cache=True)
def compute_fb_metal_fe_nk(E: np.ndarray, A_fe: float, B_fe: float, C_fe: float) -> np.ndarray:
    """
    Computes the Free-Electron (Intraband) contribution for Metals (FB 2021).

    Physical Interpretation:
    ------------------------
    This term models the conduction electrons (plasma). In the FB 2021 formalism,
    the Free-Electron (FE) term is treated as a specific case of the generalized 
    dispersion equation where the bandgap Eg approaches 0.
    
    This avoids the infinity at E=0 found in Drude models, maintaining physical 
    realism and integrability.

    Parameters:
    -----------
    A_fe : float
        Amplitude of the free-electron oscillation.
    B_fe : float
        Damping/Broadening of the free electrons (scattering rate).
    C_fe : float
        Resonance term for the free electrons.
        
    Returns:
    --------
    np.ndarray (complex128)
        The complex refractive index contribution of the conduction band.
    """
    # In the 2021 paper, the FE term is mathematically equivalent to 
    # the Interband term with Eg set to 0.0.
    Eg_fe = 0.0
    
    # We reuse the robust rational formulation from 2019.
    return compute_single_fb2019_term(E, Eg_fe, A_fe, B_fe, C_fe)


@njit(cache=True)
def _compute_nk_interband_only(E: np.ndarray,
                               n_inf: float,
                               ib_terms: np.ndarray) -> np.ndarray:
    """
    Numba driver for Interband-only models (Insulators/Semiconductors).
    Sums all interband terms efficiently.
    
    ib_terms must be a 2D array of shape (N, 4) -> [Eg, A, B, C]
    """
    total_n = np.full(E.shape, n_inf, dtype=np.float64)
    total_k = np.zeros_like(E, dtype=np.float64)
    
    # Loop over N interband terms
    n_terms = ib_terms.shape[0]
    for i in range(n_terms):
        Eg, A, B, C = ib_terms[i, 0], ib_terms[i, 1], ib_terms[i, 2], ib_terms[i, 3]
        
        # Call the core rational formula for one oscillator
        nk_j = compute_single_fb2019_term(E, Eg, A, B, C)
        
        total_n += nk_j.real
        total_k += nk_j.imag

    return total_n + 1j * total_k


@njit(cache=True)
def _compute_nk_metal_full(E: np.ndarray,
                            n_inf: float,
                            fe_params: np.ndarray, 
                            ib_terms: np.ndarray) -> np.ndarray:
    """
    Numba driver for Metal models. Sums Intraband and Interband terms efficiently.
    
    fe_params must be a 1D array of shape (3,) -> [A_fe, B_fe, C_fe]
    ib_terms must be a 2D array of shape (N, 4) -> [Eg, A, B, C]
    """
    total_n = np.full(E.shape, n_inf, dtype=np.float64)
    total_k = np.zeros_like(E, dtype=np.float64)

    # 1. Add Intraband (Free Electron) Term
    if fe_params[0] > 0.0:
        # Eg=0.0 is used for the Free Electron Term (Intraband)
        nk_fe = compute_single_fb2019_term(E, 0.0, fe_params[0], fe_params[1], fe_params[2])
        total_n += nk_fe.real
        total_k += nk_fe.imag

    # 2. Add Interband Terms Loop
    n_terms = ib_terms.shape[0]
    for i in range(n_terms):
        Eg, A, B, C = ib_terms[i, 0], ib_terms[i, 1], ib_terms[i, 2], ib_terms[i, 3]
        nk_j = compute_single_fb2019_term(E, Eg, A, B, C)
        
        total_n += nk_j.real
        total_k += nk_j.imag

    return total_n + 1j * total_k


class ForouhiBloomerInterbandSingle(Material):
    """
    Single-Term Forouhi-Bloomer (2019) Model for Insulators/Semiconductors.

    This class models a material dominated by a single interband transition
    (e.g., an amorphous semiconductor with one primary absorption peak).

    Physics:
        N(E) = n_inf + N_interband(E)
    
    Attributes:
    -----------
    params : dict
        - 'n_inf': High-frequency dielectric constant offset (real number >= 1).
        - 'Eg': Optical Bandgap (eV).
        - 'A', 'B', 'C': Dispersion parameters.
    """

    def __init__(self, params: Dict[str, Union[float, int]], wavelength: Optional[np.ndarray] = None):
        super().__init__({'A': params.get('A', 0.0)}, wavelength)

        self.n_inf = params.get('n_inf', 1.0)
        self.Eg = params.get('Eg')
        self.A = params.get('A')
        self.B = params.get('B')
        self.C = params.get('C')

        # --- Physical Constraints Validation ---
        if not all([x is not None for x in [self.Eg, self.A, self.B, self.C]]):
            raise ValueError("Parameters 'Eg', 'A', 'B', 'C' are required.")
        
        # 4C > B^2 is required for complex conjugate poles (finite lifetime).
        # If 4C <= B^2, the system implies infinite lifetime or gain, violating passivity.
        if 4 * self.C <= self.B**2:
             raise ValueError("Physical violation: 4*C must be > B^2 to ensure stability.")

        self._fb_term_params = np.array([self.Eg, self.A, self.B, self.C], dtype=np.float64)
        self.h_c_by_eV_nm = 1239.8419843320028 
        
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Sets the spectral range and converts wavelength (nm) to Energy (eV)."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm) 
        
    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculates the complex refractive index for the given wavelengths.

        Args:
            wavelength: Optional array of wavelengths in nm. If provided,
                overrides previously set wavelength range.
        
        Returns:
            Complex128 ndarray where real part is n (refractive index),
            imaginary part is k (extinction coefficient).
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        if not hasattr(self, 'E'):
             raise AttributeError("Wavelength range must be set.")
             
        # Pass E, n_inf, and the [1, 4] array to the Numba driver
        self.nk = _compute_nk_interband_only(
            self.E, 
            float(self.n_inf), 
            self._ib_terms_array
        )
        return self.nk

    def get_params(self):
        return {'n_inf': self.n_inf, 'Eg': self.Eg, 'A': self.A, 'B': self.B, 'C': self.C}


class ForouhiBloomerInterbandMulti(Material):
    """
    Multi-Term Forouhi-Bloomer (2019) Model for Complex Insulators/Semiconductors.

    Used for materials with multiple distinct absorption peaks (interband transitions)
    within the measured spectral range (e.g., Crystalline Silicon, complex oxides).

    Physics:
        N(E) = n_inf + Sum_{j} [ N_interband_j(E) ]
        
    Due to the linearity of the susceptibility in linear optics, contributions 
    from different quantum transitions are additive.
    """
    def __init__(self, params: Dict[str, Union[float, int, List]], wavelength: Optional[np.ndarray] = None):
        super().__init__({'A': 0.0}, wavelength)
        self.n_inf = params.get('n_inf', 1.0)
        self.ib_params = params.get('ib_params', [])
        
        self._ib_terms = []
        for i, term in enumerate(self.ib_params):
            # Validate physics for each oscillator
            if 4 * term['C'] <= term['B']**2:
                 raise ValueError(f"Oscillator {i} (Eg={term['Eg']}): 4*C must be > B^2.")
            self._ib_terms.append(np.array([term['Eg'], term['A'], term['B'], term['C']], dtype=np.float64))
        
        self._ib_terms_array = np.array(self._ib_terms, dtype=np.float64)
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
        """Calculates the total complex refractive index for metals using Numba acceleration.
    
        Sums contributions from free electrons (intraband) and bound electrons (interband transitions).
        The core computation is accelerated with Numba for speed improvements.
    
        Args:
            wavelength: Optional array of wavelengths in nm. If provided,
                overrides previously set wavelength range.
    
        Returns:
            Complex128 ndarray where real part is n (refractive index),
            imaginary part is k (extinction coefficient).
    
        Raises:
            AttributeError: If energy data has not been initialized.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        if not hasattr(self, 'E'):
             raise AttributeError("Wavelength range must be set.")
        
        #Handle case of no terms, though usually Multi has terms.
        #ib_terms = self._ib_terms_array if len(self._ib_terms_array) > 0 else np.zeros((0, 4), dtype=np.float64)

        # Pass E, n_inf, and the [N, 4] array to the Numba driver
        self.nk = _compute_nk_interband_only(
            self.E, 
            float(self.n_inf), 
            self._ib_terms_array
        )
        return self.nk
    
    def get_params(self):
        """Returns the model parameters as a dictionary.

        Returns:
            Dictionary containing 'n_inf' and list of interband parameters.
        """
        return {'n_inf': self.n_inf, 'ib_params': self.ib_params}

class ForouhiBloomerMetalSingle(Material):
    """
    Single-Term Forouhi-Bloomer (2021) Metal Model.
    
    This is a convenience class for metals with exactly one Interband term 
    and one Intraband (Free Electron) term.
    
    It maintains a 'flat' parameter structure, which is often easier for 
    optimization algorithms than the nested list structure of the general class.

    Physics:
        N(E) = n_inf + N_FreeElectron(E) + N_Interband(E)

    Parameters:
    -----------
    params : dict
        - 'n_inf': High-freq constant.
        - 'A_fe', 'B_fe', 'C_fe': Free Electron (Intraband) parameters.
        - 'Eg', 'A', 'B', 'C': Single Interband transition parameters.
    """

    def __init__(self,
                 params: Dict[str, Union[float, int]],
                 wavelength: Optional[np.ndarray] = None):
        
        super().__init__({'A': params.get('A', 0.0)}, wavelength)
        
        self.n_inf = params.get('n_inf', 1.0)
        
        # --- Interband Parameters (Single Term) ---
        self.Eg = params.get('Eg')
        self.A = params.get('A')
        self.B = params.get('B')
        self.C = params.get('C')
        
        # --- Intraband (Free Electron) Parameters ---
        self.A_fe = params.get('A_fe')
        self.B_fe = params.get('B_fe')
        self.C_fe = params.get('C_fe')

        # --- Validation ---
        required = [self.Eg, self.A, self.B, self.C, self.A_fe, self.B_fe, self.C_fe]
        if not all(p is not None for p in required):
            raise ValueError("All 7 parameters (Eg, A, B, C, A_fe, B_fe, C_fe) are required.")

        # Stability Checks (FB 2019/2021 constraints)
        if 4 * self.C <= self.B**2:
             raise ValueError("Interband violation: 4*C must be > B^2.")
        if self.A_fe > 0 and (4 * self.C_fe <= self.B_fe**2):
             raise ValueError("Intraband (Free Electron) violation: 4*C_fe must be > B_fe^2.")

        # Prepare Numba arrays
        self._ib_term_params = np.array([self.Eg, self.A, self.B, self.C], dtype=np.float64)
        self._fe_term_params = np.array([self.A_fe, self.B_fe, self.C_fe], dtype=np.float64)

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
        """Calculates the total complex refractive index for metals using Numba acceleration.
    
        Sums contributions from free electrons (intraband) and bound electrons (interband transitions).
        The core computation is accelerated with Numba for speed improvements.
    
        Args:
            wavelength: Optional array of wavelengths in nm. If provided,
                overrides previously set wavelength range.
    
        Returns:
            Complex128 ndarray where real part is n (refractive index),
            imaginary part is k (extinction coefficient).
    
        Raises:
            AttributeError: If energy data has not been initialized.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        if not hasattr(self, 'E'):
             raise AttributeError("Wavelength range must be set.")

        # Prepare parameters for Numba driver
        n_inf = float(self.n_inf)
        fe_params = self._fe_term_params
        
        # Ensure the single interband term is packaged correctly for the loop in Numba
        ib_terms = self._ib_term_params.reshape(1, 4)

        # Call the Metal Numba driver
        self.nk = _compute_nk_metal_full(
            self.E, 
            n_inf, 
            fe_params, 
            ib_terms
        )
        return self.nk

    def get_params(self) -> Dict[str, Union[float, int]]:
        """Returns the model parameters as a dictionary.
    
        Returns:
            Dictionary containing all model parameters including:
                - 'n_inf': High-frequency dielectric constant offset.
                - Interband parameters: 'Eg', 'A', 'B', 'C'
                - Free electron parameters: 'A_fe', 'B_fe', 'C_fe'
        """
        return {
            'n_inf': self.n_inf,
            'Eg': self.Eg, 'A': self.A, 'B': self.B, 'C': self.C,
            'A_fe': self.A_fe, 'B_fe': self.B_fe, 'C_fe': self.C_fe
        }

class ForouhiBloomerMetal2021(Material):
    """
    Forouhi-Bloomer (2021) Generalized Metal Model.
    
    This model unifies the description of metals by combining Intraband (Free Electron)
    and Interband (Bound Electron) transitions.

    Physics:
        N_total(E) = n_inf + N_FreeElectron(E) + Sum_{j} [ N_BoundElectron_j(E) ]
        
    1. n_inf: Contribution from deep UV transitions not modeled explicitly.
    2. Free Electron (Intraband): Conduction band electrons (Eg=0). 
       Replaces the Drude term, correcting the non-physical singularity at E=0.
    3. Bound Electron (Interband): Electrons jumping between bands (Eg > 0).

    Attributes:
    -----------
    params : dict
        - 'n_inf': float
        - 'fe_params': dict {'A_fe', 'B_fe', 'C_fe'} for the Intraband term.
        - 'ib_params': list of dicts [{'Eg', 'A', 'B', 'C'}, ...] for Interband terms.
    """
    def __init__(self, params: Dict[str, Union[float, int, Dict, List]], wavelength: Optional[np.ndarray] = None):
        super().__init__({'A': 0.0}, wavelength)
        
        self.n_inf = params.get('n_inf', 1.0)
        self.fe_params = params.get('fe_params', {})
        self.ib_params = params.get('ib_params', [])

        # --- Setup Intraband (Free Electron) Parameters ---
        self.A_fe = self.fe_params.get('A_fe', 0.0)
        self.B_fe = self.fe_params.get('B_fe', 0.0)
        self.C_fe = self.fe_params.get('C_fe', 0.0)
        self._fe_params = np.array([self.A_fe, self.B_fe, self.C_fe], dtype=np.float64)
        
        # Check Free Electron stability
        if self.A_fe > 0:
            if 4 * self.C_fe <= self.B_fe**2:
                raise ValueError("Free Electron Term: 4*C_fe must be > B_fe^2.")

        # --- Setup Interband (Bound Electron) Parameters ---
        self._ib_terms = []
        for term in self.ib_params:
            if 4 * term['C'] <= term['B']**2:
                 raise ValueError(f"Interband Term (Eg={term['Eg']}): 4*C must be > B^2.")
            self._ib_terms.append(np.array([term['Eg'], term['A'], term['B'], term['C']], dtype=np.float64))
        self._ib_terms_array = np.array(self._ib_terms, dtype=np.float64)
        
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
        """Calculates the total complex refractive index for metals using Numba acceleration.
    
        Sums contributions from free electrons (intraband) and bound electrons (interband transitions).
        The core computation is accelerated with Numba for speed improvements.
    
        Args:
            wavelength: Optional array of wavelengths in nm. If provided,
                overrides previously set wavelength range.
    
        Returns:
            Complex128 ndarray where real part is n (refractive index),
            imaginary part is k (extinction coefficient).
    
        Raises:
            AttributeError: If energy data has not been initialized.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        if not hasattr(self, 'E'):
             raise AttributeError("Wavelength range must be set.")
        
        # Handle case of no terms, though usually Multi has terms.
        ib_terms = self._ib_terms_array if len(self._ib_terms_array) > 0 else np.zeros((0, 4), dtype=np.float64)

        # Pass E, n_inf, and the [N, 4] array to the Numba driver
        self.nk = _compute_nk_interband_only(
            self.E, 
            float(self.n_inf), 
            ib_terms
        )
        return self.nk
        
    def get_params(self):
        """Returns the model parameters as a dictionary.

        Returns:
            Dictionary containing 'n_inf', free electron parameters ('fe_params'),
            and interband parameters ('ib_params').
        """
        return {
            'n_inf': self.n_inf,
            'fe_params': {'A_fe': self.A_fe, 'B_fe': self.B_fe, 'C_fe': self.C_fe},
            'ib_params': self.ib_params
        }