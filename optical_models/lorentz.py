# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: lorentz.py — Lorentz oscillator dispersion model.

Architecture:
  Primary store:    self._osc_params       (list of tuples — mutated directly)
  Derived (Numba):  self._lorentz_params   (contiguous float64 array)
  Derived (config): flat keys in self.params  (E0_0, Gamma_0, f0_0, …)

  All derived state is rebuilt by _sync().  Every mutation path must end
  with _sync() to keep the three representations consistent.
"""

import numpy as np
from numba import njit
from typing import List, Tuple, Dict, Union, Optional

from .material import Material, compute_energy

__all__ = ["LorentzOscillator"]

# Physical constant: h·c in eV·nm
_HC_EV_NM: float = 1239.8419843320028

_PARAM_MAP: Dict[str, int] = {"E0": 0, "Gamma": 1, "f0": 2}


@njit(cache=True, fastmath=True)
def compute_lorentz_complex_nk(
    E: np.ndarray,
    lorentz_params: np.ndarray,
    eps_inf: float,
) -> np.ndarray:
    """
    Compute complex refractive index (n + ik) for Lorentz oscillators.

    Args:
        E: Photon energies [eV].
        lorentz_params: Shape (N, 3) array of (E0, Gamma, f0) per oscillator.
        eps_inf: High-frequency dielectric constant.

    Returns:
        Complex refractive index array (n + ik).
    """
    eps = np.full(E.shape, eps_inf + 0j, dtype=np.complex128)
    E_sq = E * E

    for j in range(lorentz_params.shape[0]):
        E0, Gamma, f0 = lorentz_params[j]
        E0_sq = E0 * E0
        eps += (f0 * E0_sq) / ((E0_sq - E_sq) - 1j * (E * Gamma))

    return np.sqrt(eps)


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


class LorentzOscillator(Material):
    """
    Lorentz oscillator dispersion model with high-performance JIT integration.

    Primary storage is ``_osc_params`` (list of tuples).  Flat keys in
    ``self.params`` (``E0_0``, ``Gamma_0``, ``f0_0``, …) and the Numba-ready
    array ``_lorentz_params`` are derived views rebuilt atomically by ``_sync()``.

    Args:
        params: Dict containing 'osc_params' (list of (E0, Gamma, f0) tuples)
                and optionally 'epsilon_inf' (default 1.0).
        wavelength: Optional wavelength grid [nm].
        **kwargs: Extra scalar params forwarded to Material.

    Examples:
        >>> params = {
        ...     'epsilon_inf': 1.0,
        ...     'osc_params': [(3.0, 0.2, 0.5), (4.5, 0.1, 0.7)]
        ... }
        >>> mat = LorentzOscillator(params=params)

        >>> mat.epsilon_inf
        1.0
        >>> mat.n_oscillators
        2
        >>> mat.get_oscillator_param(0, 'E0')
        3.0

        >>> # Flat-key access for optimizer integration
        >>> mat.set_param('E0_0', 3.5)
        >>> mat.E0_0
        3.5

        >>> mat.add_oscillator(5.0, 0.12, 0.6)
        >>> mat.n_oscillators
        3
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        params: Optional[Dict[str, Union[float, List[Tuple]]]] = None,
        wavelength: Optional[np.ndarray] = None,
        **kwargs,
    ):
        p = params.copy() if params else {}
        p.update(kwargs)

        super().__init__(wavelength=wavelength, params=p)
        self._validate_params(required=["osc_params"], optional={"epsilon_inf": 1.0})

        # ── Build primary store from the 'osc_params' init key ───────
        raw_oscs = self.params.pop("osc_params")  # remove list-form from params
        if not raw_oscs:
            raise ValueError(
                "Lorentz model requires 'osc_params' list of (E0, Gamma, f0)."
            )

        for i, osc in enumerate(raw_oscs):
            if len(osc) != 3:
                raise ValueError(
                    f"Oscillator {i} must be (E0, Gamma, f0), got {osc}"
                )
            _validate_oscillator(*osc, label=str(i))

        self._osc_params: List[Tuple[float, float, float]] = [
            (float(o[0]), float(o[1]), float(o[2])) for o in raw_oscs
        ]

        self.E: Optional[np.ndarray] = None

        # Build derived state (flat keys + Numba array)
        self._sync()

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    # ── Derived-state synchronisation ────────────────────────────────

    def _sync(self) -> None:
        """
        Rebuild all derived state from ``_osc_params`` (the single primary store).

        Updates:
          - ``_lorentz_params``: contiguous float64 array for Numba.
          - Flat keys in ``self.params``: ``E0_0``, ``Gamma_0``, ``f0_0``, …
          - Invalidates ``self.nk`` computation cache.
        """
        # 1. Numba array
        self._lorentz_params = np.array(self._osc_params, dtype=np.float64)

        # 2. Scrub stale flat keys, then rebuild
        stale = [k for k in self.params if _parse_osc_key(k) is not None]
        for k in stale:
            del self.params[k]

        for i, (e0, gamma, f0) in enumerate(self._osc_params):
            self.params[f"E0_{i}"] = e0
            self.params[f"Gamma_{i}"] = gamma
            self.params[f"f0_{i}"] = f0

        # 3. Invalidate computation cache
        self.nk = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators (derived — never stored)."""
        return len(self._osc_params)

    @property
    def epsilon_inf(self) -> float:
        """High-frequency dielectric constant."""
        return self.params["epsilon_inf"]

    @property
    def osc_params(self) -> List[Tuple[float, float, float]]:
        """Read-only copy of oscillator parameters as list of tuples."""
        return list(self._osc_params)

    # ── Wavelength / computation ─────────────────────────────────────

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Update energy grid and invalidate computation cache."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, _HC_EV_NM)
        self.nk = None

    def complex_refractive_index(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return cached or freshly computed complex refractive index."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        if self.nk is None:
            self.nk = compute_lorentz_complex_nk(
                self.E,
                self._lorentz_params,
                float(self.params["epsilon_inf"]),
            )
        return self.nk

    # ── Individual oscillator access ─────────────────────────────────

    def get_oscillator_param(self, index: int, param: str) -> float:
        """
        Get one scalar from oscillator *index*.

        Args:
            index: 0-based oscillator index.
            param: One of 'E0', 'Gamma', 'f0'.
        """
        self._check_osc_index(index)
        return self._osc_params[index][self._param_col(param)]

    def get_all_osc_params(self) -> List[Tuple[float, float, float]]:
        """Return all oscillator parameters."""
        return self.osc_params  # Already available via property    

    def set_oscillator_param(
        self, index: int, param: str, value: float
    ) -> None:
        """
        Set one scalar on oscillator *index* (validates, syncs, invalidates).

        Args:
            index: 0-based oscillator index.
            param: One of 'E0', 'Gamma', 'f0'.
            value: New value (must satisfy positivity constraints).
        """
        self._check_osc_index(index)
        col = self._param_col(param)

        current = list(self._osc_params[index])
        current[col] = float(value)
        _validate_oscillator(*current, label=str(index))

        self._osc_params[index] = tuple(current)
        self._sync()

    # ── Bulk oscillator operations ───────────────────────────────────

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

    def clear_and_replace(
        self, osc_params: List[Tuple[float, float, float]]
    ) -> None:
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

    # ── Unified parameter mutation (flat-key API for optimizers) ─────

    def set_param(self, name: str, value: Union[float, int]) -> None:
        """
        Set any parameter by name with validation and state sync.

        Oscillator flat keys (``E0_0``, ``Gamma_1``, etc.) are routed
        through the primary store so all derived state stays consistent.

        Args:
            name: Parameter name (e.g. 'epsilon_inf', 'E0_0', 'Gamma_1').
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

        # Scalar parameters (epsilon_inf, etc.)
        super().set_param(name, value)

    # ── Attribute-style convenience access ───────────────────────────

    def __getattr__(self, name: str):
        """
        Enable ``mat.E0_0``, ``mat.Gamma_1`` style reads for params.

        Only fires for names not found via normal attribute lookup,
        so internal attributes (``_osc_params``, ``params``, etc.)
        are never intercepted.
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
        params = self.__dict__.get("params")
        if params is not None and name in params:
            self.set_param(name, value)
        else:
            super().__setattr__(name, value)

    # ── Serialisation ────────────────────────────────────────────────

    def get_params(self) -> Dict[str, float]:
        """
        Return a copy of all parameters.

        Contains ``epsilon_inf`` plus flat oscillator keys
        (``E0_0``, ``Gamma_0``, ``f0_0``, …).  Use the ``osc_params``
        property for the tuple-list form.
        """
        return self.params.copy()

    # ── Private helpers ──────────────────────────────────────────────

    def _check_osc_index(self, index: int) -> None:
        if not 0 <= index < len(self._osc_params):
            raise IndexError(
                f"Oscillator index {index} out of range "
                f"[0, {len(self._osc_params) - 1}]"
            )

    @staticmethod
    def _param_col(param: str) -> int:
        if param not in _PARAM_MAP:
            raise ValueError(
                f"param must be one of {list(_PARAM_MAP)}, got '{param}'"
            )
        return _PARAM_MAP[param]