# -*- coding: utf-8 -*-
"""
Navette: the mathematics of light in thin-film systems
(Rust-backed wrapper; kernels live in navette.materials._native)

Tauc–Lorentz dispersion model (Jellison & Modine, 1996).

A multi-oscillator amorphous-semiconductor model: ε₂ is the analytic
Tauc–Lorentz form with a shared optical gap ``Eg`` and one or more Lorentz
oscillators (A, E0, C); ε₁ is recovered by FFT Kramers–Kronig in the Rust core
(``navette.materials._native.tauc_lorentz_nk``), the same KK path used by the
Cody–Lorentz and UBF models.

Storage mirrors the other oscillator models:
  Primary store:    self._osc_params   (list of (A, E0, C) tuples)
  Derived (Rust):   self._tl_osc_array (contiguous (N, 3) float64 array)
  Scalars:          self.params['Eg'], self.params['epsilon_inf']
``_sync()`` rebuilds the derived array and resets the cached ``self.nk``.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .material import Material

try:
  from . import _native
except ImportError:  # pragma: no cover - native not built yet
  _native = None  # type: ignore[assignment]


class TaucLorentz(Material):
    """
    Tauc–Lorentz model with a shared gap and N Lorentz oscillators.

    ε₂(E) = Σⱼ Aⱼ·E0ⱼ·Cⱼ·(E−Eg)² / ( ((E²−E0ⱼ²)² + Cⱼ²E²)·E )   for E > Eg, else 0
    n̂(E)  = √(ε₁ + iε₂),  with ε₁ = ε∞ − Hilbert{ε₂} (FFT KK).

    Example
    -------
    >>> mat = TaucLorentz(params={
    ...     'Eg': 1.2,
    ...     'epsilon_inf': 1.0,
    ...     'oscillators': [
    ...         {'A': 100.0, 'E0': 4.0, 'C': 2.0},
    ...         {'A':  50.0, 'E0': 6.5, 'C': 1.5},
    ...     ],
    ... })
    >>> nk = mat.complex_refractive_index(np.linspace(300.0, 900.0, 601))
    """

    def __init__(
        self,
        params: Dict[str, Union[float, int, List]],
        wavelength: Optional[np.ndarray] = None,
    ):
        # Pull the oscillator list out before the base stores scalars.
        p = dict(params)
        raw_oscs = p.pop("oscillators", None)
        super().__init__(wavelength=None, params=p)
        self._validate_params(required=["Eg"], optional={"epsilon_inf": 1.0})

        if not raw_oscs:
            raise ValueError("TaucLorentz requires at least one oscillator.")

        self._osc_params: List[Tuple[float, float, float]] = [
            self._osc_dict_to_tuple(osc, idx=i) for i, osc in enumerate(raw_oscs)
        ]
        self.E: Optional[np.ndarray] = None
        self._sync()

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    # ------------------------------------------------------------------ #
    #  Oscillator bookkeeping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _osc_dict_to_tuple(osc: Dict, idx: int = 0) -> Tuple[float, float, float]:
        """Convert a user oscillator dict {A, E0, C} to the internal tuple."""
        try:
            A = float(osc["A"])
            E0 = float(osc["E0"])
            C = float(osc["C"])
        except KeyError as exc:
            raise ValueError(f"Oscillator {idx} missing key {exc}.") from exc
        if E0 <= 0.0:
            raise ValueError(f"Oscillator {idx}: E0 must be > 0.")
        if C <= 0.0:
            raise ValueError(f"Oscillator {idx}: C (broadening) must be > 0.")
        if A < 0.0:
            raise ValueError(f"Oscillator {idx}: A (amplitude) must be >= 0.")
        return (A, E0, C)

    def _sync(self) -> None:
        """Rebuild derived state from ``self._osc_params`` and flush the cache."""
        self._tl_osc_array = np.array(self._osc_params, dtype=np.float64)

        # Refresh flat per-oscillator keys for introspection/fitting parity.
        stale = [k for k in self.params if k[:2] in ("A_", "E_", "C_")]
        for k in stale:
            del self.params[k]
        for i, (A, E0, C) in enumerate(self._osc_params):
            self.params[f"A_{i}"] = A
            self.params[f"E0_{i}"] = E0
            self.params[f"C_{i}"] = C

        self.nk = None

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators."""
        return len(self._osc_params)

    def add_oscillator(self, A: float, E0: float, C: float) -> None:
        """Append an oscillator and resync derived state."""
        self._osc_params.append(self._osc_dict_to_tuple({"A": A, "E0": E0, "C": C}))
        self._sync()

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the spectral grid (nm). Rust converts to energy internally."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.nk = None

    def complex_refractive_index(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return cached or freshly computed complex refractive index n + ik."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
        if self.wavelength is None:
            raise AttributeError("Wavelength range must be set.")

        if self.nk is None:
            if _native is None:  # pragma: no cover
              raise ImportError(
                "TaucLorentz needs the compiled `navette.materials._native` extension. "
                "Build it with: maturin develop -m crates/navette-materials-py/Cargo.toml"
              )
            self.nk = _native.tauc_lorentz_nk(
                self.wavelength,
                float(self.params["Eg"]),
                self._tl_osc_array,
                float(self.params["epsilon_inf"]),
            )
        return self.nk

    def get_params(self) -> Dict[str, Union[float, int, List]]:
        """Return all model parameters as a dictionary."""
        return {
            "Eg": self.params["Eg"],
            "epsilon_inf": self.params["epsilon_inf"],
            "oscillators": [
                {"A": A, "E0": E0, "C": C} for (A, E0, C) in self._osc_params
            ],
        }
