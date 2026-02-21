# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from numba import njit
from typing import List, Tuple, Dict, Union, Optional

from .material import Material, compute_energy

__all__ = ["Drude", "DrudeLorentz"]

_HC_EV_NM: float = 1239.8419843320028
_PARAM_MAP: Dict[str, int] = {"E0": 0, "Gamma": 1, "f0": 2}


@njit(cache=True, fastmath=True)
def _to_nk(eps: np.ndarray) -> np.ndarray:
    """Helper to convert complex epsilon to n + ik safely."""
    return np.sqrt(eps)


@njit(cache=True, fastmath=True)
def compute_drude_complex_nk(E: np.ndarray, omega_p: float, gamma: float, eps_inf: float) -> np.ndarray:
    """
    Compute complex refractive index (n + ik) for Drude model only.

    Args:
        E: Array of photon energies in electron volts [eV]
        omega_p: Plasma frequency parameter for Drude term [eV]
        gamma_drude: Damping coefficient for Drude term [eV]
        eps_inf: High-frequency dielectric constant (default 1.0)

    Returns:
        Complex refractive index array where real parts are n and imaginary parts are k.

    Notes:
        - The function assumes proper units for all parameters (energies in eV)
        - For best performance, input energy array should be sorted
        - Only Drude term is computed without Lorentz oscillators
        - Square root calculation preserves physical consistency between n and k
    """
    # Guard against E=0 to avoid division by zero
    E_eff = E + 1e-12 
    eps = np.full(E.shape, eps_inf + 0j, dtype=np.complex128)
    
    # Drude term: -omega_p^2 / (E^2 + i * gamma * E)
    denom = (E_eff * E_eff) + 1j * (gamma * E_eff)
    eps -= (omega_p**2) / denom
    return _to_nk(eps)


@njit(cache=True, fastmath=True)
def compute_drude_lorentz_complex_nk(
    E: np.ndarray, 
    omega_p: float, 
    gamma_d: float, 
    eps_inf: float, 
    lorentz_params: np.ndarray
) -> np.ndarray:
    """
    Compute complex refractive index (n + ik) for Drude-Lorentz model.

    Args:
        E: Array of photon energies in electron volts [eV]
        omega_p: Plasma frequency parameter for Drude term [eV]
        gamma_drude: Damping coefficient for Drude term [eV]
        eps_inf: High-frequency dielectric constant (default 1.0)
        lorentz_params: Oscillator parameters as array of shape (N, 3) with
            - E0: Resonance energy position [eV]
            - Gamma: Damping constant [eV]
            - f0: Oscillator strength parameter

    Returns:
        Complex refractive index array where real parts are n and imaginary parts are k.

    Notes:
        - The function assumes proper units for all parameters (energies in eV)
        - For best performance, input energy array should be sorted
        - Lorentz terms are added to the Drude response with proper sign handling
        - Square root calculation preserves physical consistency between n and k
    """
    E_eff = E + 1e-12
    E_sq = E_eff * E_eff
    eps = np.full(E.shape, eps_inf + 0j, dtype=np.complex128)

    # Drude term: -omega_p^2 / (E^2 + i * gamma * E)
    eps -= (omega_p**2) / (E_sq + 1j * (gamma_d * E_eff))

    # Lorentz components
    for i in range(lorentz_params.shape[0]):
        e0, gamma_l, f0 = lorentz_params[i]
        e0_sq = e0 * e0
        # eps += (f0 * e0^2) / (e0^2 - e^2 - i*e*gamma)
        denom = (e0_sq - E_sq) - 1j * (E_eff * gamma_l)
        eps += (f0 * e0_sq) / denom

    return _to_nk(eps)


def _validate_oscillator(E0: float, Gamma: float, f0: float, label: str = "") -> None:
    """Validate physical constraints for a single oscillator."""
    prefix = f"Oscillator {label}: " if label else ""
    if E0 <= 0:
        raise ValueError(f"{prefix}E0 must be positive, got {E0}")
    if Gamma < 0:
        raise ValueError(f"{prefix}Gamma must be non-negative, got {Gamma}")
    if f0 <= 0:
        raise ValueError(f"{prefix}f0 must be positive, got {f0}")


def _parse_osc_key(name: str) -> Optional[Tuple[str, int]]:
    """
    Parse an oscillator flat-key name like 'E0_2' into ('E0', 2).

    Returns None if the name is not a valid oscillator key.
    """
    if "_" not in name:
        return None
    prefix, idx_str = name.rsplit("_", 1)
    if prefix in _PARAM_MAP and idx_str.isdigit():
        return prefix, int(idx_str)
    return None


class Drude(Material):
    """
    Implementation of the Drude dispersion model for free-electron response only.

    This model describes dielectric properties using only a Drude term representing
    intraband electronic transitions without any Lorentz oscillators.

    Attributes:
        omega_p: Plasma frequency for Drude term [eV]
        gamma_drude: Damping constant for Drude term [eV]
        epsilon_inf: High-frequency dielectric constant

    Args:
        params: Dictionary containing model parameters with keys:
            - 'omega_p': Plasma frequency [eV]
            - 'gamma_drude': Drude damping constant [eV]
            - Optional: 'epsilon_inf' High-frequency dielectric constant (default 1.0)
        wavelength: Optional array of wavelengths in nm

    Raises:
        ValueError: If invalid parameters are provided

    Example:
        >>> # Initialize with Drude parameters
        >>> params = {
        ...     'omega_p': 2.5,                # Plasma frequency [eV]
        ...     'gamma_drude': 0.3,            # Damping constant [eV]
        ...     'epsilon_inf': 3.5             # High-frequency dielectric constant
        ... }
        >>>
        >>> # Initialize with wavelength range
        >>> wavelengths = np.linspace(150, 600)  # nm from 150 to 600nm
        >>> material = Drude(params, wavelengths)
    """

    def __init__(self,
                params: Dict[str, Union[float, int]],
                wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Drude model.

        Args:
            params: Dictionary of parameters
            wavelength: Optional array of wavelengths in nm

        Raises:
            ValueError: If invalid parameters are provided
        """
        super().__init__(wavelength=wavelength, params=params)
        
        # Validate required parameters
        self._validate_params(required=['omega_p', 'gamma_drude'], optional={'epsilon_inf': 1.0})
        
        if self.params['omega_p'] <= 0:
            raise ValueError("Plasma frequency must be positive")
            
        if self.params['gamma_drude'] < 0:
            raise ValueError("Drude damping constant must be non-negative")

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """
        Set the wavelength range for calculations and convert to energies.

        Args:
            wavelength: Array of wavelengths in nm
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, _HC_EV_NM)

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        if self.nk is None:
            self.nk = compute_drude_complex_nk(
                E=self.E,
                omega_p=self.params['omega_p'],
                gamma_drude=self.params['gamma_drude'],
                eps_inf=self.params['epsilon_inf']
            )
        return self.nk

    def get_params(self) -> Dict[str, Union[float, int]]:
        """
        Return all model parameters as a dictionary.

        Returns:
            Dictionary containing all model parameters
        """
        return {
            'omega_p': self.params['omega_p'],
            'gamma_drude': self.params['gamma_drude'],
            'epsilon_inf': self.params['epsilon_inf']
        }


class DrudeLorentz(Material):
    """
    Implementation of the Drude-Lorentz dispersion model combining free-electron response
    with multiple Lorentz oscillators.

    This model describes dielectric properties by combining:
    - A Drude term representing intraband electronic transitions
    - Multiple Lorentz oscillators representing interband transitions

    Each Lorentz oscillator is characterized by its resonance energy (E0), damping constant (Gamma),
    and oscillator strength parameter (f0).

    Attributes:
        n_oscillators (int): Number of oscillators in the model
        omega_p: Plasma frequency for Drude term [eV]
        gamma_drude: Damping constant for Drude term [eV]
        epsilon_inf: High-frequency dielectric constant

    Args:
        params: Dictionary containing model parameters with keys:
            - 'omega_p': Plasma frequency [eV]
            - 'gamma_drude': Drude damping constant [eV]
            - Either:
                'osc_params': List of tuples (E0, Gamma, f0) for all oscillators
            - Optional: 'epsilon_inf' High-frequency dielectric constant (default 1.0)
        wavelength: Optional array of wavelengths in nm

    Raises:
        ValueError: If invalid parameters are provided or no oscillators specified

    Example:
        >>> # Initialize with Lorentz oscillator parameters
        >>> params = {
        ...     'omega_p': 2.5,                # Plasma frequency [eV]
        ...     'gamma_drude': 0.3,            # Damping constant [eV]
        ...     'osc_params': [
        ...         (4.0, 0.5, 1.0),           # Oscillator parameters
        ...         (6.0, 0.2, 0.8)
        ...     ],
        ...     'epsilon_inf': 3.5             # High-frequency dielectric constant
        ... }
        >>>
        >>> # Initialize with wavelength range
        >>> wavelengths = np.linspace(150, 600)  # nm from 150 to 600nm
        >>> material = DrudeLorentz(params, wavelengths)
    """

    def __init__(self,
                params: Dict[str, Union[float, int, List[Tuple[float, float, float]]]],
                wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Drude-Lorentz model.

        Args:
            params: Dictionary of parameters
            wavelength: Optional array of wavelengths in nm

        Raises:
            ValueError: If invalid parameters are provided or no oscillators are specified
        """
        p = params.copy()
        super().__init__(wavelength=wavelength, params=p)
        
        # Validate required parameters
        self._validate_params(required=['omega_p', 'gamma_drude'], 
                             optional={'epsilon_inf': 1.0})
        
        if self.params['omega_p'] <= 0:
            raise ValueError("Plasma frequency must be positive")
            
        if self.params['gamma_drude'] < 0:
            raise ValueError("Drude damping constant must be non-negative")

        # Extract oscillator parameters - now using osc_params like LorentzOscillator
        raw_lorentz_params = params.get('osc_params', [])  # Changed from 'lorentz_params'
        
        if not raw_lorentz_params:
            raise ValueError("At least one Lorentz oscillator must be specified")
            
        # Master store - list of tuples with validated parameters
        self._osc_params: List[Tuple[float, float, float]] = []
        for i, osc in enumerate(raw_lorentz_params):
            if len(osc) != 3:
                raise ValueError(f"Oscillator {i} must be (E0, Gamma, f0), got {osc}")
            _validate_oscillator(*osc, label=str(i))
            self._osc_params.append((float(osc[0]), float(osc[1]), float(osc[2])))

        # Build derived state 
        self._sync()
        
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def _sync(self) -> None:
        """Project master oscillators to params dict and Numba array."""
        self._lorentz_params = np.array(self._osc_params, dtype=np.float64)
        
        # Clean flat keys
        stale_keys = [k for k in self.params.keys() if _parse_osc_key(k) is not None]
        for k in stale_keys:
            del self.params[k]
            
        # Also clean up old parameter names (if they were set before the sync)
        param_keys_to_clean = ['omega_p', 'gamma_drude', 'epsilon_inf']
        for key in param_keys_to_clean:
            if key in self.params:
                del self.params[key]
            
        for i, (e0, g, f) in enumerate(self._osc_params):
            self.params[f"E0_{i}"] = e0
            self.params[f"Gamma_{i}"] = g  
            self.params[f"f0_{i}"] = f
            
        # Add the scalar parameters to params dict for consistency with LorentzOscillator 
        self.params['omega_p'] = self.params.get('omega_p')
        self.params['gamma_drude'] = self.params.get('gamma_drude')  
        self.params['epsilon_inf'] = self.params.get('epsilon_inf')
        
        self.nk = None

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators."""
        return len(self._osc_params)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """
        Set the wavelength range for calculations and convert to energies.

        Args:
            wavelength: Array of wavelengths in nm
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, _HC_EV_NM)

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        if self.nk is None:
            self.nk = compute_drude_lorentz_complex_nk(
                E=self.E,
                omega_p=self.params['omega_p'],
                gamma_drude=self.params['gamma_drude'],
                eps_inf=self.params['epsilon_inf'],  # Changed from eps_inf
                lorentz_params=self._lorentz_params
            )
        return self.nk

    def get_params(self) -> Dict[str, Union[float, int]]:
        """
        Return all model parameters as a flat dictionary.

        Returns:
            Dictionary containing all model parameters in a flat structure
        """
        # Use the same pattern as LorentzOscillator - flat dict with all params
        result = {
            'omega_p': self.params['omega_p'],
            'gamma_drude': self.params['gamma_drude'],
            'epsilon_inf': self.params['epsilon_inf']
        }
        
        # Add oscillator parameters in the same format as LorentzOscillator 
        for i, (e0, gamma, f0) in enumerate(self._osc_params):
            result[f"E0_{i}"] = e0
            result[f"Gamma_{i}"] = gamma
            result[f"f0_{i}"] = f0
            
        return result

    # Oscillator access methods - following lorentz.py pattern
    def get_oscillator_param(self, index: int, param: str) -> float:
        """
        Get one scalar from oscillator *index*.
        
        Args:
            index: 0-based oscillator index.  
            param: One of 'E0', 'Gamma', 'f0'.
        """
        self._check_osc_index(index)
        col = _PARAM_MAP[param]
        return self._osc_params[index][col]

    def set_oscillator_param(self, index: int, param: str, value: float) -> None:
        """
        Set one scalar on oscillator *index* (validates, syncs, invalidates).
        
        Args:
            index: 0-based oscillator index.
            param: One of 'E0', 'Gamma', 'f0'.
            value: New value (must satisfy positivity constraints).
        """
        self._check_osc_index(index)
        col = _PARAM_MAP[param]
        
        current = list(self._osc_params[index])
        current[col] = float(value)
        
        _validate_oscillator(*current, label=str(index))
        
        self._osc_params[index] = tuple(current)
        self._sync()

    def add_oscillator(self, E0: float, Gamma: float, f0: float) -> None:
        """Append a validated oscillator and synchronise state."""
        _validate_oscillator(E0, Gamma, f0)
        self._osc_params.append((float(E0), float(Gamma), float(f0)))
        self._sync()

    def remove_oscillator(self, index: int) -> Tuple[float, float, float]:
        """
        Remove oscillator at *index*, re-index state, return removed params.
        
        Raises:
            IndexError: If index is out of range.
            ValueError: If removing the last oscillator.
        """
        self._check_osc_index(index)
        
        if len(self._osc_params) == 1:
            raise ValueError("Cannot remove the last oscillator")
            
        removed = self._osc_params.pop(index)
        self._sync()  # re-indexes flat keys automatically
        return removed

    def clear_and_replace(self, osc_params: List[Tuple[float, float, float]]) -> None:
        """
        Replace all oscillators at once (bulk update).
        
        Useful when loading from config or during fitting.
        """
        if not osc_params:
            raise ValueError("At least one oscillator must be specified")
            
        for i, osc in enumerate(osc_params):
            if len(osc) != 3:
                raise ValueError(
                    f"Oscillator {i} must be (E0, Gamma, f0), got {osc}"
                )
            _validate_oscillator(*osc, label=str(i))

        self._osc_params = [
            (float(o[0]), float(o[1]), float(o[2])) for o in osc_params
        ]
        self._sync()

    def set_param(self, name: str, value: Union[float, int]) -> None:
        """
        Set any parameter by name with validation and state sync.
        
        Oscillator flat keys (``E0_0``, ``Gamma_1``, etc.) are routed
        through the primary store so all derived state stays consistent.
        
        Args:
            name: Parameter name (e.g. 'omega_p', 'gamma_drude', 'epsilon_inf', 'E0_0', 'Gamma_1').
            value: New numeric value.
        """
        # Try oscillator flat-key first
        parsed = _parse_osc_key(name)
        if parsed is not None:
            prefix, idx = parsed
            self._check_osc_index(idx)

            col = _PARAM_MAP[prefix]
            current = list(self._osc_params[idx])
            current[col] = float(value)
            _validate_oscillator(*current, label=str(idx))

            self._osc_params[idx] = tuple(current)
            self._sync()
            return

        # Scalar parameters (omega_p, gamma_drude, epsilon_inf)  
        super().set_param(name, value)

    def __getattr__(self, name: str):
        """
        Enable ``mat.E0_0``, ``mat.Gamma_1`` style reads for params.
        
        Only fires for names not found via normal attribute lookup,
        so internal attributes (``_osc_params``, ``params``, etc.)
        are never intercepted.
        """
        # Guard against recursion during init / unpickling using __dict__ check
        params = self.__dict__.get("params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """
        Route known param keys through ``set_param()`` for auto-sync;
        everything else goes through normal attribute setting.
        """
        # Always bypass for internal / infrastructure attributes
        if name.startswith("_") or name in ("params", "wavelength", "nk", "E"):
            super().__setattr__(name, value)
            return

        # Only route through set_param if params dict exists and name is
        # already a known key — prevents hijacking of new attributes and
        # avoids crashes during __init__ before params is populated.
        # Using __dict__.get("params") for more robustness vs hasattr()
        params = self.__dict__.get("params")
        if params is not None and name in params:
            self.set_param(name, value)
        else:
            super().__setattr__(name, value)

    def _check_osc_index(self, index: int) -> None:
        if not 0 <= index < len(self._osc_params):
            raise IndexError(
                f"Oscillator index {index} out of range "
                f"[0, {len(self._osc_params) - 1}]"
            )
