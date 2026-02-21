# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Cody-Lorentz Dielectric Model — Multi-Oscillator
=================================================
Ferlauto continuous Cody-Lorentz with FFT Kramers-Kronig.
Numba-accelerated throughout; single clean API.

Architecture:
  Shared scalars:   Eg, Et, Eu, epsilon_inf   (in self.params)
  Primary store:    self._osc_params           (list of (E0, A, Gamma, Ep) tuples)
  Derived (Numba):  self._cl_osc_array         (contiguous float64 array for JIT)
  Derived (config): flat keys in self.params   (E0_0, A_0, Gamma_0, Ep_0, …)

  All derived state is rebuilt by _sync().  Every mutation path must end
  with _sync() to keep the three representations consistent.

Multi-oscillator ε₂:
  Band (E ≥ Et):
      ε₂(E) = Σⱼ Aⱼ · (E−Eg)² / ((E−Eg)² + Epⱼ²) · E·Γⱼ / ((E²−E0ⱼ²)² + (E·Γⱼ)²)

  Urbach tail (E < Et):
      ε₂(E) = ε₂_total(Et) · (Et / E) · exp((E − Et) / Eu)

  The Urbach tail is C⁰-matched to the total oscillator sum at Et.

Optimisation features:
    - Module-global cached energy grid (_E_FULL)
    - Precomputed KK frequency multiplier (_KK_H, _KK_M)
    - Power-of-2 FFT grid (8192 pts)
    - Numba parallel ε₂ and √ε loops
    - rfft for real-valued input (half spectrum)
    - Parameter validation at construction and mutation
    - Result caching when wavelength/params unchanged
"""

import numpy as np
from numba import njit, prange
from typing import Dict, List, Tuple, Union, Optional

from material import Material, compute_energy

__all__ = ["CodyLorentz"]

_HC: float = 1239.8419843320028          # h·c [eV·nm]
_GRID_N: int = 8192                      # power-of-2 → fastest FFT
_GRID_MAX: float = 80.0                  # eV  (captures UV poles)
_GRID_MIN: float = 0.01                  # eV

# Per-oscillator parameter layout: (E0, A, Gamma, Ep)
_OSC_PARAM_MAP: Dict[str, int] = {"E0": 0, "A": 1, "Gamma": 2, "Ep": 3}
_OSC_PARAM_NAMES: Tuple[str, ...] = ("E0", "A", "Gamma", "Ep")
_OSC_WIDTH: int = 4

# Shared (scalar) parameter names
_SHARED_PARAMS: Tuple[str, ...] = ("Eg", "Et", "Eu")


# =====================================================================
#  Module-global cached grid and KK kernel  (built once at import)
# =====================================================================

_E_FULL: np.ndarray = np.linspace(_GRID_MIN, _GRID_MAX, _GRID_N)

# Precompute padded FFT size, frequency array, and Hilbert multiplier
_M_RAW: int = 2 * _GRID_N + 1
_KK_M: int = 1
while _KK_M < _M_RAW:
    _KK_M <<= 1

_KK_dE: float = _E_FULL[1] - _E_FULL[0]
_KK_H: np.ndarray = np.empty((_KK_M // 2) + 1, dtype=np.complex128)
_KK_H[0] = 0.0
_KK_H[1:] = -1j


# =====================================================================
#  Validation helpers
# =====================================================================

def _validate_shared_params(params: Dict[str, float]) -> None:
    """Validate physical constraints on shared Cody-Lorentz parameters."""
    Eu = params.get("Eu", 0)
    Eg = params.get("Eg", 0)
    Et = params.get("Et", 0)

    if Eu <= 0:
        raise ValueError(f"Eu must be > 0, got {Eu}")
    if Et < Eg:
        raise ValueError(f"Et must be >= Eg ({Eg}), got {Et}")


def _validate_oscillator(E0: float, A: float, Gamma: float, Ep: float,
                         label: str = "") -> None:
    """Validate physical constraints for a single oscillator."""
    prefix = f"Oscillator {label}: " if label else ""
    if E0 <= 0:
        raise ValueError(f"{prefix}E0 must be positive, got {E0}")
    if A <= 0:
        raise ValueError(f"{prefix}A must be positive, got {A}")
    if Gamma <= 0:
        raise ValueError(f"{prefix}Gamma must be positive, got {Gamma}")
    if Ep < 0:
        raise ValueError(f"{prefix}Ep must be non-negative, got {Ep}")


def _parse_osc_key(name: str) -> Optional[Tuple[str, int]]:
    """
    Parse an oscillator flat-key name like 'E0_2' into ('E0', 2).

    Returns None if the name is not a valid oscillator key.
    """
    if "_" not in name:
        return None
    prefix, idx_str = name.rsplit("_", 1)
    if prefix in _OSC_PARAM_MAP and idx_str.isdigit():
        return prefix, int(idx_str)
    return None


# =====================================================================
#  ε₂ — Multi-oscillator Ferlauto Cody DOS + Lorentz
# =====================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _eps2_multi(E, Eg, Et, Eu, osc_params):
    """
    ε₂(E) for the multi-oscillator continuous Cody-Lorentz model.

    Parameters
    ----------
    E : (N,) float64
        Photon energies [eV].
    Eg : float
        Optical band gap [eV].
    Et : float
        Urbach–band transition energy [eV].
    Eu : float
        Urbach tail energy [eV].
    osc_params : (N_osc, 4) float64
        Per-oscillator parameters: columns [E0, A, Gamma, Ep].

    Returns
    -------
    (N,) float64
        ε₂ values.
    """
    n_E = E.shape[0]
    n_osc = osc_params.shape[0]
    out = np.zeros(n_E, dtype=np.float64)

    # Pre-compute Urbach amplitude: total ε₂ at Et, C⁰-matched
    Et2 = Et * Et
    A_t_total = 0.0
    for j in range(n_osc):
        E0j   = osc_params[j, 0]
        Aj    = osc_params[j, 1]
        Gamj  = osc_params[j, 2]
        Epj   = osc_params[j, 3]
        E0jsq = E0j * E0j

        if Et > Eg:
            d = Et - Eg
            cody_Et = (d * d) / (d * d + Epj * Epj)
        else:
            cody_Et = 0.0

        denom_Et = (Et2 - E0jsq) ** 2 + (Et * Gamj) ** 2
        A_t_total += Aj * cody_Et * (Et * Gamj) / denom_Et

    inv_Eu = 1.0 / Eu

    for i in prange(n_E):
        Ei = E[i]
        if Ei >= Et:
            if Ei > Eg:
                d = Ei - Eg
                dsq = d * d
                Ei2 = Ei * Ei
                val = 0.0
                for j in range(n_osc):
                    E0j   = osc_params[j, 0]
                    Aj    = osc_params[j, 1]
                    Gamj  = osc_params[j, 2]
                    Epj   = osc_params[j, 3]
                    E0jsq = E0j * E0j
                    cody  = dsq / (dsq + Epj * Epj)
                    denom = (Ei2 - E0jsq) ** 2 + (Ei * Gamj) ** 2
                    val  += Aj * cody * (Ei * Gamj) / denom
                out[i] = val
        elif Ei > 1e-9:
            out[i] = A_t_total * (Et / Ei) * np.exp((Ei - Et) * inv_Eu)

    return out


# =====================================================================
#  FFT Kramers-Kronig with odd extension (precomputed kernel)
# =====================================================================

def _kk_fft(eps2, eps_inf):
    """
    ε₁ via FFT Hilbert transform with proper odd extension.

    Uses module-global precomputed _KK_H multiplier and _KK_M pad size.
    Only the buffer fill and two FFT calls happen per invocation.
    """
    N = _GRID_N

    buf = np.zeros(_KK_M, dtype=np.float64)
    buf[N + 1 : N + 1 + N] = eps2
    buf[1 : N + 1] = -eps2[::-1]

    F = np.fft.rfft(buf)
    hilb = np.fft.irfft(F * _KK_H, n=_KK_M)

    return eps_inf - hilb[N : N + N]


# =====================================================================
#  Full pipeline: ε₂ → KK → interpolate → n̂
# =====================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _nk_from_eps(eps1, eps2):
    """Convert (ε₁, ε₂) → n̂ = √(ε₁ + iε₂), Numba-parallel."""
    n = eps1.shape[0]
    out = np.empty(n, dtype=np.complex128)
    for i in prange(n):
        z = complex(eps1[i], eps2[i])
        out[i] = np.sqrt(z)
    return out


def compute_nk(target_E: np.ndarray,
               Eg: float, Et: float, Eu: float,
               osc_params: np.ndarray,
               eps_inf: float) -> np.ndarray:
    """
    Compute n̂(E) for an arbitrary target energy array.

    Pipeline:
        1. ε₂ on cached 8192-pt grid  (Numba parallel, multi-oscillator)
        2. ε₁ via FFT KK              (O(N log N), precomputed kernel)
        3. Interpolate ε₁ to target energies
        4. ε₂ evaluated exactly at target energies
        5. n̂ = √ε
    """
    eps2_full = _eps2_multi(_E_FULL, Eg, Et, Eu, osc_params)
    eps1_full = _kk_fft(eps2_full, eps_inf)

    # Guard against silent extrapolation outside the KK integration grid.
    E_lo, E_hi = _E_FULL[0], _E_FULL[-1]
    if target_E.min() < E_lo or target_E.max() > E_hi:
        raise ValueError(
            f"target energies [{target_E.min():.4f}, {target_E.max():.4f}] eV "
            f"exceed KK grid [{E_lo:.4f}, {E_hi:.4f}] eV. "
            f"Adjust wavelength range or _GRID_MIN/_GRID_MAX."
        )

    eps1_t = np.interp(target_E, _E_FULL, eps1_full)
    eps2_t = _eps2_multi(target_E, Eg, Et, Eu, osc_params)
    return _nk_from_eps(eps1_t, eps2_t)


# =====================================================================
#  Class interface
# =====================================================================

class CodyLorentz(Material):
    """
    Multi-oscillator continuous Cody-Lorentz (CCL) dielectric model.

    Shared (scalar) parameters live in ``self.params``.  Per-oscillator
    parameters are stored in ``_osc_params`` (list of 4-tuples) and
    exposed as flat keys (``E0_0``, ``A_0``, ``Gamma_0``, ``Ep_0``, …)
    in ``self.params``, exactly mirroring the Lorentz oscillator
    architecture.  All derived state is rebuilt atomically by ``_sync()``.

    Parameters
    ----------
    params : dict
        Must contain:
            osc_params  : list of (E0, A, Gamma, Ep) tuples — at least one.
            Eg          : optical band gap [eV]
            Et          : Urbach–band transition energy [eV], must be ≥ Eg
            Eu          : Urbach tail energy [eV], must be > 0
        Optional:
            epsilon_inf : high-frequency dielectric constant  (default 1.0)
    wavelength : ndarray, optional
        Wavelength grid [nm].
    **kwargs
        Extra scalar params forwarded to Material (override params dict).

    Examples
    --------
    >>> params = {
    ...     'Eg': 1.64, 'Et': 1.80, 'Eu': 0.15, 'epsilon_inf': 1.0,
    ...     'osc_params': [
    ...         (3.40, 60.0, 2.4, 1.0),   # (E0, A, Gamma, Ep)
    ...         (4.70, 40.0, 1.8, 0.5),
    ...     ],
    ... }
    >>> mat = CodyLorentz(params=params, wavelength=np.linspace(200, 1200, 500))

    >>> mat.Eg
    1.64
    >>> mat.n_oscillators
    2
    >>> mat.get_oscillator_param(0, 'E0')
    3.40

    >>> # Flat-key access for optimizer integration
    >>> mat.set_param('E0_0', 3.5)
    >>> mat.E0_0
    3.5

    >>> mat.add_oscillator(5.5, 20.0, 1.2, 0.8)
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

        super().__init__(wavelength=None, params=p)
        self._validate_params(
            required=["osc_params", "Eg", "Et", "Eu"],
            optional={"epsilon_inf": 1.0},
        )

        # Validate shared scalar constraints
        _validate_shared_params(self.params)

        # ── Build primary store from the 'osc_params' init key ───────
        raw_oscs = self.params.pop("osc_params")  # remove list-form from params
        if not raw_oscs:
            raise ValueError(
                "CodyLorentz model requires 'osc_params' list of "
                "(E0, A, Gamma, Ep)."
            )

        for i, osc in enumerate(raw_oscs):
            if len(osc) != _OSC_WIDTH:
                raise ValueError(
                    f"Oscillator {i} must be (E0, A, Gamma, Ep), got {osc}"
                )
            _validate_oscillator(*osc, label=str(i))

        self._osc_params: List[Tuple[float, float, float, float]] = [
            (float(o[0]), float(o[1]), float(o[2]), float(o[3]))
            for o in raw_oscs
        ]

        self.E: Optional[np.ndarray] = None

        # Build derived state (flat keys + Numba array)
        self._sync()

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    # ── Derived-state synchronisation ────────────────────────────────

    def _sync(self) -> None:
        """
        Rebuild all derived state from ``_osc_params`` (the primary store).

        Updates:
          - ``_cl_osc_array``: contiguous (N_osc, 4) float64 array for Numba.
          - Flat keys in ``self.params``: ``E0_0``, ``A_0``, ``Gamma_0``,
            ``Ep_0``, …
          - Invalidates ``self.nk`` computation cache.
        """
        # 1. Numba array
        self._cl_osc_array = np.array(self._osc_params, dtype=np.float64)

        # 2. Scrub stale flat keys, then rebuild
        stale = [k for k in self.params if _parse_osc_key(k) is not None]
        for k in stale:
            del self.params[k]

        for i, (e0, a, gamma, ep) in enumerate(self._osc_params):
            self.params[f"E0_{i}"] = e0
            self.params[f"A_{i}"] = a
            self.params[f"Gamma_{i}"] = gamma
            self.params[f"Ep_{i}"] = ep

        # 3. Invalidate computation cache
        self.nk = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators (derived — never stored)."""
        return len(self._osc_params)

    @property
    def Eg(self) -> float:
        """Optical band gap [eV]."""
        return self.params["Eg"]

    @property
    def Et(self) -> float:
        """Urbach–band transition energy [eV]."""
        return self.params["Et"]

    @property
    def Eu(self) -> float:
        """Urbach tail energy [eV]."""
        return self.params["Eu"]

    @property
    def epsilon_inf(self) -> float:
        """High-frequency dielectric constant."""
        return self.params["epsilon_inf"]

    @property
    def osc_params(self) -> List[Tuple[float, float, float, float]]:
        """Read-only copy of oscillator parameters as list of tuples."""
        return list(self._osc_params)

    # ── Wavelength / computation ─────────────────────────────────────

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Update energy grid and invalidate computation cache."""
        self.wavelength = np.ascontiguousarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, _HC)
        self.nk = None

    def complex_refractive_index(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return cached or freshly computed complex refractive index."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        if self.nk is None:
            self.nk = compute_nk(
                self.E,
                float(self.params["Eg"]),
                float(self.params["Et"]),
                float(self.params["Eu"]),
                self._cl_osc_array,
                float(self.params["epsilon_inf"]),
            )
        return self.nk

    def dielectric_function(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return ε = n̂² as a complex array."""
        nk = self.complex_refractive_index(wavelength)
        return nk * nk

    # ── Individual oscillator access ─────────────────────────────────

    def get_oscillator_param(self, index: int, param: str) -> float:
        """
        Get one scalar from oscillator *index*.

        Args:
            index: 0-based oscillator index.
            param: One of 'E0', 'A', 'Gamma', 'Ep'.
        """
        self._check_osc_index(index)
        return self._osc_params[index][self._param_col(param)]

    def get_all_osc_params(self) -> List[Tuple[float, float, float, float]]:
        """Return all oscillator parameters."""
        return self.osc_params  # Already available via property

    def set_oscillator_param(
        self, index: int, param: str, value: float
    ) -> None:
        """
        Set one scalar on oscillator *index* (validates, syncs, invalidates).

        Args:
            index: 0-based oscillator index.
            param: One of 'E0', 'A', 'Gamma', 'Ep'.
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

    def add_oscillator(self, E0: float, A: float,
                       Gamma: float, Ep: float) -> None:
        """Append a validated oscillator and synchronise state."""
        _validate_oscillator(E0, A, Gamma, Ep)
        self._osc_params.append(
            (float(E0), float(A), float(Gamma), float(Ep))
        )
        self._sync()

    def remove_oscillator(self, index: int) -> Tuple[float, float, float, float]:
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
        self, osc_params: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Replace all oscillators at once (bulk update).

        Useful when loading from config or during fitting.
        """
        if not osc_params:
            raise ValueError("At least one oscillator must be specified")
        for i, osc in enumerate(osc_params):
            if len(osc) != _OSC_WIDTH:
                raise ValueError(
                    f"Oscillator {i} must be (E0, A, Gamma, Ep), got {osc}"
                )
            _validate_oscillator(*osc, label=str(i))

        self._osc_params = [
            (float(o[0]), float(o[1]), float(o[2]), float(o[3]))
            for o in osc_params
        ]
        self._sync()

    # ── Unified parameter mutation (flat-key API for optimizers) ─────

    def set_param(self, name: str, value: Union[float, int]) -> None:
        """
        Set any parameter by name with validation and state sync.

        Oscillator flat keys (``E0_0``, ``A_1``, ``Gamma_0``, ``Ep_2``,
        etc.) are routed through the primary store so all derived state
        stays consistent.  Shared scalar keys (``Eg``, ``Et``, ``Eu``,
        ``epsilon_inf``) are validated against physical constraints.

        Args:
            name: Parameter name (e.g. 'Eg', 'epsilon_inf', 'E0_0', 'Gamma_1').
            value: New numeric value.
        """
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(
                f"Parameter '{name}' must be numeric, got {type(value).__name__}"
            )

        # Try oscillator flat-key first
        parsed = _parse_osc_key(name)
        if parsed is not None:
            prefix, idx = parsed
            self._check_osc_index(idx)

            col = _OSC_PARAM_MAP[prefix]
            current = list(self._osc_params[idx])
            current[col] = float(value)
            _validate_oscillator(*current, label=str(idx))

            self._osc_params[idx] = tuple(current)
            self._sync()
            return

        # Shared scalar parameters — validate constraints before committing
        if name in _SHARED_PARAMS:
            trial = self.params.copy()
            trial[name] = float(value)
            _validate_shared_params(trial)

        super().set_param(name, value)

    # ── Attribute-style convenience access ───────────────────────────

    def __getattr__(self, name: str):
        """
        Enable ``mat.E0_0``, ``mat.Gamma_1``, ``mat.Eg`` style reads.

        Only fires for names not found via normal attribute lookup,
        so properties and internal attributes are never intercepted.
        """
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

        # Route through set_param if params dict exists and name is a known key
        params = self.__dict__.get("params")
        if params is not None and name in params:
            self.set_param(name, value)
        else:
            super().__setattr__(name, value)

    # ── Serialisation ────────────────────────────────────────────────

    def get_params(self) -> Dict[str, float]:
        """
        Return a copy of all parameters.

        Contains ``Eg``, ``Et``, ``Eu``, ``epsilon_inf`` plus flat
        oscillator keys (``E0_0``, ``A_0``, ``Gamma_0``, ``Ep_0``, …).
        Use the ``osc_params`` property for the tuple-list form.
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
        if param not in _OSC_PARAM_MAP:
            raise ValueError(
                f"param must be one of {list(_OSC_PARAM_MAP)}, got '{param}'"
            )
        return _OSC_PARAM_MAP[param]


# =====================================================================
#  Demo
# =====================================================================

if __name__ == "__main__":
    import time, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    wavelengths = np.linspace(200, 1200, 1001)

    # ── a-Si: single oscillator (backward-compatible use) ─────────
    si_params_1osc = dict(
        Eg=1.64, Et=1.80, Eu=0.15, epsilon_inf=1.0,
        osc_params=[
            (3.40, 60.0, 2.4, 1.0),       # (E0, A, Gamma, Ep)
        ],
    )

    # ── a-Si: two oscillators (extended UV structure) ─────────────
    si_params_2osc = dict(
        Eg=1.64, Et=1.80, Eu=0.15, epsilon_inf=1.0,
        osc_params=[
            (3.40, 45.0, 2.2, 1.0),       # primary absorption
            (4.70, 30.0, 1.8, 0.5),        # higher-energy shoulder
        ],
    )

    # ── SiO₂: single oscillator ──────────────────────────────────
    sio2_params = dict(
        Eg=8.0, Et=8.0, Eu=0.05, epsilon_inf=1.0,
        osc_params=[
            (11.0, 100.0, 0.5, 2.0),
        ],
    )

    # Warm-up JIT
    _ = CodyLorentz(si_params_1osc, wavelengths).complex_refractive_index()

    # Benchmark
    for label, p in [("a-Si 1", si_params_1osc),
                     ("a-Si 2", si_params_2osc),
                     ("SiO₂",   sio2_params)]:
        m = CodyLorentz(p, wavelengths)
        m.complex_refractive_index()

        t0 = time.perf_counter()
        N_ITER = 100
        for _ in range(N_ITER):
            m.complex_refractive_index(wavelengths)
        dt = (time.perf_counter() - t0) / N_ITER * 1000
        print(f"{label:8s}  {dt:.2f} ms / call  (avg of {N_ITER})")

    # ── Verify cache ──────────────────────────────────────────────
    m = CodyLorentz(si_params_1osc, wavelengths)
    nk1 = m.complex_refractive_index()
    nk2 = m.complex_refractive_index()
    assert nk1 is nk2, "Cache should return same object"
    print("Cache hit verified ✓")

    # ── Verify set_param invalidation ─────────────────────────────
    m = CodyLorentz(si_params_1osc, wavelengths)
    nk_before = m.complex_refractive_index()
    m.set_param('Eg', 1.70)
    assert m.nk is None, "Cache should be invalidated after set_param"
    nk_after = m.complex_refractive_index()
    assert nk_before is not nk_after, "Should recompute after param change"
    print("set_param invalidation verified ✓")

    # ── Verify oscillator flat-key round-trip ─────────────────────
    m = CodyLorentz(si_params_2osc, wavelengths)
    assert m.E0_0 == 3.40
    assert m.A_1 == 30.0
    m.set_param('E0_0', 3.50)
    assert m.E0_0 == 3.50
    assert m.get_oscillator_param(0, 'E0') == 3.50
    print("Flat-key round-trip verified ✓")

    # ── Verify add / remove ───────────────────────────────────────
    m = CodyLorentz(si_params_1osc, wavelengths)
    assert m.n_oscillators == 1
    m.add_oscillator(4.70, 30.0, 1.8, 0.5)
    assert m.n_oscillators == 2
    assert m.E0_1 == 4.70
    removed = m.remove_oscillator(1)
    assert m.n_oscillators == 1
    assert removed == (4.70, 30.0, 1.8, 0.5)
    print("add/remove oscillator verified ✓")

    # ── Verify clear_and_replace ──────────────────────────────────
    m = CodyLorentz(si_params_1osc, wavelengths)
    m.clear_and_replace([
        (3.50, 50.0, 2.0, 1.2),
        (5.00, 25.0, 1.5, 0.8),
        (6.20, 15.0, 1.0, 0.3),
    ])
    assert m.n_oscillators == 3
    assert m.E0_2 == 6.20
    print("clear_and_replace verified ✓")

    # ── Verify validation ─────────────────────────────────────────
    try:
        CodyLorentz(dict(
            Eg=1.64, Et=1.0, Eu=0.15,
            osc_params=[(3.4, 60.0, 2.4, 1.0)],
        ))
        print("ERROR: should have raised ValueError for Et < Eg")
    except ValueError as e:
        print(f"Shared param validation works ✓  ({e})")

    try:
        m = CodyLorentz(si_params_1osc, wavelengths)
        m.set_param('Eu', -0.1)
        print("ERROR: should have raised ValueError for Eu <= 0")
    except ValueError as e:
        print(f"set_param validation works ✓  ({e})")

    try:
        m = CodyLorentz(si_params_2osc, wavelengths)
        m.set_param('Gamma_0', -1.0)
        print("ERROR: should have raised ValueError for Gamma <= 0")
    except ValueError as e:
        print(f"Oscillator flat-key validation works ✓  ({e})")

    # ── Comparison table: 1-osc vs 2-osc ─────────────────────────
    nk1 = CodyLorentz(si_params_1osc, wavelengths).complex_refractive_index()
    nk2 = CodyLorentz(si_params_2osc, wavelengths).complex_refractive_index()
    sio = CodyLorentz(sio2_params, wavelengths).complex_refractive_index()

    print(f"\n{'λ nm':<8} {'1-osc n':>8} {'1-osc k':>8}"
          f"  {'2-osc n':>8} {'2-osc k':>8}"
          f"  {'SiO₂ n':>8} {'SiO₂ k':>8}")
    print("-" * 70)
    for wl in range(200, 1201, 100):
        i = np.argmin(np.abs(wavelengths - wl))
        print(f"{wavelengths[i]:<8.0f}"
              f" {nk1[i].real:8.4f} {nk1[i].imag:8.4f}"
              f"  {nk2[i].real:8.4f} {nk2[i].imag:8.4f}"
              f"  {sio[i].real:8.4f} {sio[i].imag:8.4f}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, nk, title in zip(axes,
                              [nk1, nk2, sio],
                              ['a-Si (1 osc)', 'a-Si (2 osc)', 'SiO₂']):
        ax.plot(wavelengths, nk.real, 'b-', lw=1.5, label='n')
        ax.plot(wavelengths, nk.imag, 'r--', lw=1.5, label='k')
        ax.set(title=title, xlabel='Wavelength (nm)', ylabel='n, k')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("codylorentz_multi_demo.png", dpi=150)
    print("\nPlot → codylorentz_multi_demo.png")