# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Universal Band-Fluctuation Cody-Lorentz Model — Optimised
==========================================================
Monolog-Lorentz ε₂ (Lizárraga et al. 2022) with FFT Kramers-Kronig.
Numba-accelerated throughout; multi-oscillator support.

Architecture (mirrors lorentz.py):
  Primary store:    self._osc_params       (list of tuples — mutated directly)
  Derived (Numba):  self._osc_array        (contiguous float64 array with
                                             β=1/Eu conversion applied)
  Derived (config): flat keys in self.params  (Eg_0, Ec_0, Eu_0, …)

  All derived state is rebuilt by _sync().  Every mutation path must end
  with _sync() to keep the three representations consistent.

Changes from original
---------------------
* KK: O(N²) bin-skipping trapezoidal → O(N log N) FFT Hilbert with
  proper odd extension.  Eliminates systematic ε₁ error from the
  discarded singularity bin and is ~100× faster.
* Extended grid: ε₂ is now evaluated on a cached 8192-pt [0.01, 80] eV
  grid before KK, so oscillator peaks outside the measurement window
  (e.g. UV poles at 6-11 eV) contribute correctly to ε₁.
* ε₂ loop: outer energy loop is now Numba `prange` (parallel).
  Softplus is inlined to avoid per-point function-call overhead.
* Softplus: added negative-x early-out (base → 0 when x ≪ 0).
* Parameter validation at construction.
* Result caching when wavelength is unchanged.
* Energy-bounds guard on interpolation (no silent extrapolation).

v3 Updates (material.py / lorentz.py compatibility)
---------------------------------------------------
* Hybrid parameter API: supports both dict-based and individual parameters.
* Single source of truth: self._osc_params is the primary store;
  self.params flat keys and self._osc_array are derived via _sync().
* Centralized cache invalidation through set_param().
* Attribute-style access (mat.Eg_0, mat.epsilon_inf) via __getattr__/__setattr__.
* Oscillator management: add_oscillator, remove_oscillator, clear_and_replace.
* Flat-key API for optimizer integration (set_param('Eg_0', 1.4)).
"""

import numpy as np
from typing import Dict, Union, Optional, List, Tuple

from .material import Material, compute_energy

try:
  from . import _native
except ImportError:  # pragma: no cover - native not built yet
  _native = None  # type: ignore[assignment]

_NATIVE_BUILD_HINT = "maturin develop -m crates/navette-materials-py/Cargo.toml"

__all__ = ["UBF_CodyLorentz"]

# Physical constant: h·c [eV·nm]
_HC: float = 1239.8419843320028

# Extended KK grid parameters
_GRID_N: int = 8192                      # power-of-2 → fastest FFT
_GRID_MAX: float = 80.0                  # eV
_GRID_MIN: float = 0.01                  # eV

# Oscillator parameter names → column indices in primary store tuples
# Primary store tuple order: (Eg, Ec, Eu, A, Gamma, gamma)
_PARAM_MAP: Dict[str, int] = {
    "Eg": 0, "Ec": 1, "Eu": 2, "A": 3, "Gamma": 4, "gamma": 5,
}


# =====================================================================
#  Validation helpers
# =====================================================================

def _validate_ubf_oscillator(
    Eg: float, Ec: float, Eu: float, A: float, Gamma: float,
    gamma: float, label: str = ""
) -> None:
    """Validate physical constraints for a single UBF-CL oscillator."""
    prefix = f"Oscillator {label}: " if label else ""
    if Ec <= 0:
        raise ValueError(f"{prefix}Ec must be positive, got {Ec}")
    if Eu <= 0:
        raise ValueError(f"{prefix}Eu must be > 0, got {Eu}")
    if Gamma <= 0:
        raise ValueError(f"{prefix}Gamma must be > 0, got {Gamma}")
    if gamma not in (0.5, 1.0, 2.0):
        if gamma <= 0:
            raise ValueError(f"{prefix}gamma must be positive, got {gamma}")


def _parse_osc_key(name: str) -> Optional[Tuple[str, int]]:
    """
    Parse an oscillator flat-key name like 'Eg_2' into ('Eg', 2).

    Returns None if the name is not a valid oscillator key.
    """
    if "_" not in name:
        return None
    prefix, idx_str = name.rsplit("_", 1)
    if prefix in _PARAM_MAP and idx_str.isdigit():
        return prefix, int(idx_str)
    return None


def _osc_dict_to_tuple(
    osc: Dict, idx: int = 0
) -> Tuple[float, float, float, float, float, float]:
    """
    Convert a user-facing oscillator dict to the internal tuple form.

    Dict keys: Eg, Ec, Eu, A, Gamma, Type (str).
    Tuple:     (Eg, Ec, Eu, A, Gamma, gamma_exp).
    """
    Eg = float(osc.get("Eg", 1.5))
    Ec = float(osc.get("Ec", 5.0))
    Eu = float(osc.get("Eu", 0.05))
    A = float(osc.get("A", 10.0))
    Gamma = float(osc.get("Gamma", 1.0))

    mat_type = osc.get("Type", "Indirect").strip().lower()
    gamma_exp = 0.5 if mat_type == "direct" else 2.0

    _validate_ubf_oscillator(Eg, Ec, Eu, A, Gamma, gamma_exp, label=str(idx))
    return (Eg, Ec, Eu, A, Gamma, gamma_exp)


# =====================================================================
#  Class interface
# =====================================================================

class UBF_CodyLorentz(Material):
    """
    Universal Band-Fluctuation Cody-Lorentz (UBF-CL) dielectric model.

    ε₂(E) = Σ_j (A_j/E) · [ln(1+exp(β_j(E−Eg_j)))]^γ_j · L_j(E)

    where L_j is a Lorentz oscillator and β_j = 1/Eu_j.

    Primary storage is ``_osc_params`` (list of tuples).  Flat keys in
    ``self.params`` (``Eg_0``, ``Ec_0``, ``Eu_0``, …) and the Numba-ready
    array ``_osc_array`` are derived views rebuilt atomically by ``_sync()``.

    Args:
        params: Dict containing 'oscillators' (list of dicts with Eg, Ec, Eu,
                A, Gamma, Type) and optionally 'epsilon_inf' (default 1.0).
        wavelength: Optional wavelength grid [nm].
        **kwargs: Extra scalar params forwarded to Material.

    Examples:
        >>> params = {
        ...     'epsilon_inf': 1.0,
        ...     'oscillators': [
        ...         dict(Eg=1.3, Ec=3.6, Eu=0.5, A=3.0, Gamma=2.5, Type='Indirect'),
        ...         dict(Eg=3.0, Ec=4.5, Eu=0.5, A=1.5, Gamma=2.0, Type='Indirect'),
        ...     ],
        ... }
        >>> mat = UBF_CodyLorentz(params=params)

        >>> mat.epsilon_inf
        1.0
        >>> mat.n_oscillators
        2
        >>> mat.get_oscillator_param(0, 'Eg')
        1.3

        >>> # Flat-key access for optimizer integration
        >>> mat.set_param('Eg_0', 1.4)
        >>> mat.Eg_0
        1.4

        >>> mat.add_oscillator(Eg=5.0, Ec=8.0, Eu=0.3, A=20.0, Gamma=1.5)
        >>> mat.n_oscillators
        3
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        params: Optional[Dict[str, Union[float, List[Dict]]]] = None,
        wavelength: Optional[np.ndarray] = None,
        **kwargs,
    ):
        p = params.copy() if params else {}
        p.update(kwargs)

        super().__init__(wavelength=None, params=p)
        self._validate_params(
            required=["oscillators"], optional={"epsilon_inf": 1.0}
        )

        # ── Build primary store from the 'oscillators' init key ──────
        raw_oscs = self.params.pop("oscillators")  # remove list-form from params
        if not raw_oscs:
            raise ValueError(
                "UBF-CL model requires 'oscillators' list of oscillator dicts."
            )

        self._osc_params: List[Tuple[float, float, float, float, float, float]] = [
            _osc_dict_to_tuple(osc, idx=i) for i, osc in enumerate(raw_oscs)
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
          - ``_osc_array``: contiguous float64 array for Numba
            (with β=1/Eu conversion in column 2).
          - Flat keys in ``self.params``: ``Eg_0``, ``Ec_0``, ``Eu_0``, …
          - Invalidates ``self.nk`` computation cache.
        """
        # 1. Numba array — apply β = 1/Eu conversion for the compute kernel
        raw = np.array(self._osc_params, dtype=np.float64)
        self._osc_array = raw.copy()
        self._osc_array[:, 2] = 1.0 / raw[:, 2]  # Eu → β = 1/Eu

        # 2. Scrub stale flat keys, then rebuild
        stale = [k for k in self.params if _parse_osc_key(k) is not None]
        for k in stale:
            del self.params[k]

        for i, (Eg, Ec, Eu, A, Gamma, gamma) in enumerate(self._osc_params):
            self.params[f"Eg_{i}"] = Eg
            self.params[f"Ec_{i}"] = Ec
            self.params[f"Eu_{i}"] = Eu
            self.params[f"A_{i}"] = A
            self.params[f"Gamma_{i}"] = Gamma
            self.params[f"gamma_{i}"] = gamma

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
    def osc_params(self) -> List[Tuple[float, float, float, float, float, float]]:
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
            if self.E is None:
                raise AttributeError(
                    "Wavelength range must be set before computing n̂."
                )
            if _native is None:  # pragma: no cover
              raise ImportError(
                "UBF_CodyLorentz needs the compiled `navette.materials._native` extension. "
                f"Build it with: {_NATIVE_BUILD_HINT}"
              )
            self.nk = _native.ubf_nk(
                self.wavelength,
                self._osc_array,
                float(self.params["epsilon_inf"]),
            )
        return self.nk

    def dielectric_function(
        self, wavelength: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return ε = n̂², computing n̂ first if needed."""
        nk = self.complex_refractive_index(wavelength)
        return nk * nk

    # ── Individual oscillator access ─────────────────────────────────

    def get_oscillator_param(self, index: int, param: str) -> float:
        """
        Get one scalar from oscillator *index*.

        Args:
            index: 0-based oscillator index.
            param: One of 'Eg', 'Ec', 'Eu', 'A', 'Gamma', 'gamma'.
        """
        self._check_osc_index(index)
        return self._osc_params[index][self._param_col(param)]

    def get_all_osc_params(self) -> List[Tuple[float, float, float, float, float, float]]:
        """Return all oscillator parameters."""
        return self.osc_params  # Already available via property

    def set_oscillator_param(
        self, index: int, param: str, value: float
    ) -> None:
        """
        Set one scalar on oscillator *index* (validates, syncs, invalidates).

        Args:
            index: 0-based oscillator index.
            param: One of 'Eg', 'Ec', 'Eu', 'A', 'Gamma', 'gamma'.
            value: New value (must satisfy physical constraints).
        """
        self._check_osc_index(index)
        col = self._param_col(param)

        current = list(self._osc_params[index])
        current[col] = float(value)
        _validate_ubf_oscillator(*current, label=str(index))

        self._osc_params[index] = tuple(current)
        self._sync()

    # ── Bulk oscillator operations ───────────────────────────────────

    def add_oscillator(
        self,
        Eg: float = 1.5,
        Ec: float = 5.0,
        Eu: float = 0.05,
        A: float = 10.0,
        Gamma: float = 1.0,
        gamma: float = 2.0,
        Type: Optional[str] = None,
    ) -> None:
        """
        Append a validated oscillator and synchronise state.

        Args:
            Eg: Optical band gap [eV].
            Ec: Lorentz resonance energy [eV].
            Eu: Urbach energy [eV], must be > 0.
            A: Oscillator amplitude.
            Gamma: Lorentz damping width [eV], must be > 0.
            gamma: Band-fluctuation exponent (2.0=indirect, 0.5=direct).
            Type: Alternative to gamma — 'Indirect' (γ=2) or 'Direct' (γ=0.5).
                  Overrides gamma if provided.
        """
        if Type is not None:
            gamma = 0.5 if Type.strip().lower() == "direct" else 2.0

        _validate_ubf_oscillator(
            float(Eg), float(Ec), float(Eu), float(A), float(Gamma), float(gamma)
        )
        self._osc_params.append(
            (float(Eg), float(Ec), float(Eu), float(A), float(Gamma), float(gamma))
        )
        self._sync()

    def remove_oscillator(
        self, index: int
    ) -> Tuple[float, float, float, float, float, float]:
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
        self, oscillators: List[Union[Tuple, Dict]]
    ) -> None:
        """
        Replace all oscillators at once (bulk update).

        Accepts either list of tuples (Eg, Ec, Eu, A, Gamma, gamma)
        or list of dicts (same format as __init__).
        Useful when loading from config or during fitting.
        """
        if not oscillators:
            raise ValueError("At least one oscillator must be specified")

        new_osc: List[Tuple[float, float, float, float, float, float]] = []
        for i, osc in enumerate(oscillators):
            if isinstance(osc, dict):
                new_osc.append(_osc_dict_to_tuple(osc, idx=i))
            else:
                if len(osc) != 6:
                    raise ValueError(
                        f"Oscillator {i} tuple must have 6 elements "
                        f"(Eg, Ec, Eu, A, Gamma, gamma), got {len(osc)}"
                    )
                vals = tuple(float(v) for v in osc)
                _validate_ubf_oscillator(*vals, label=str(i))
                new_osc.append(vals)

        self._osc_params = new_osc
        self._sync()

    # ── Unified parameter mutation (flat-key API for optimizers) ─────

    def set_param(self, name: str, value: Union[float, int]) -> None:
        """
        Set any parameter by name with validation and state sync.

        Oscillator flat keys (``Eg_0``, ``Ec_1``, etc.) are routed
        through the primary store so all derived state stays consistent.

        Args:
            name: Parameter name (e.g. 'epsilon_inf', 'Eg_0', 'Gamma_1').
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
            _validate_ubf_oscillator(*current, label=str(idx))

            self._osc_params[idx] = tuple(current)
            self._sync()
            return

        # Scalar parameters (epsilon_inf, etc.)
        super().set_param(name, value)

    # ── Attribute-style convenience access ───────────────────────────

    def __getattr__(self, name: str):
        """
        Enable ``mat.Eg_0``, ``mat.Gamma_1`` style reads for params.

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
        (``Eg_0``, ``Ec_0``, ``Eu_0``, …).  Use the ``osc_params``
        property for the tuple-list form.
        """
        return self.params.copy()

    def get_params_as_dicts(self) -> Dict:
        """
        Return parameters in the original dict-of-dicts format.

        Useful for serialisation to config files.
        """
        osc_out = []
        for Eg, Ec, Eu, A, Gamma, gamma_exp in self._osc_params:
            osc_out.append({
                "Eg": Eg,
                "Ec": Ec,
                "Eu": Eu,
                "A": A,
                "Gamma": Gamma,
                "Type": "Direct" if gamma_exp == 0.5 else "Indirect",
            })
        return {"epsilon_inf": self.params["epsilon_inf"], "oscillators": osc_out}

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


# =====================================================================
#  Demo
# =====================================================================

if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wl = np.linspace(200, 1200, 1001)

    # --- a-Si: main gap + high-energy shoulder ---
    asi_params = {
        "epsilon_inf": 1.0,
        "oscillators": [
            dict(Eg=1.3, Ec=3.6, Eu=0.5, A=3.0, Gamma=2.5, Type="Indirect"),
            dict(Eg=3.0, Ec=4.5, Eu=0.5, A=1.5, Gamma=2.0, Type="Indirect"),
        ],
    }

    # --- SiO₂: single far-UV oscillator ---
    sio2_params = {
        "epsilon_inf": 1.0,
        "oscillators": [
            dict(Eg=8.0, Ec=11.0, Eu=0.5, A=60.0, Gamma=1.0, Type="Direct"),
        ],
    }

    # Warm-up JIT
    _ = UBF_CodyLorentz(asi_params, wl).complex_refractive_index()

    # Benchmark
    for label, p in [("a-Si", asi_params), ("SiO₂", sio2_params)]:
        m = UBF_CodyLorentz(p, wl)
        m.complex_refractive_index()
        t0 = time.perf_counter()
        N_ITER = 100
        for _ in range(N_ITER):
            m.complex_refractive_index(wl)
        dt = (time.perf_counter() - t0) / N_ITER * 1000
        print(f"{label:6s}  {dt:.2f} ms / call  (avg of {N_ITER})")

    # Cache test
    m = UBF_CodyLorentz(asi_params, wl)
    nk1 = m.complex_refractive_index()
    nk2 = m.complex_refractive_index()
    assert nk1 is nk2
    print("Cache ✓")

    # Validation test
    try:
        UBF_CodyLorentz({"oscillators": [dict(Eg=1.0, Ec=3.0, Eu=0.0,
                                              A=10.0, Gamma=1.0)]})
        print("ERROR: Eu=0 accepted")
    except ValueError as e:
        print(f"Validation ✓  ({e})")

    # Flat-key API test
    m = UBF_CodyLorentz(asi_params, wl)
    print(f"\nFlat-key test: Eg_0 = {m.Eg_0}")
    m.set_param("Eg_0", 1.4)
    print(f"After set_param: Eg_0 = {m.Eg_0}")
    m.Eg_0 = 1.5
    print(f"After attr set:  Eg_0 = {m.Eg_0}")

    # Add/remove oscillator test
    m.add_oscillator(Eg=5.0, Ec=8.0, Eu=0.3, A=20.0, Gamma=1.5, Type="Direct")
    print(f"After add: n_oscillators = {m.n_oscillators}")
    removed = m.remove_oscillator(2)
    print(f"After remove: n_oscillators = {m.n_oscillators}, removed = {removed}")

    # Compute
    asi_nk = UBF_CodyLorentz(asi_params, wl).complex_refractive_index()
    sio2_nk = UBF_CodyLorentz(sio2_params, wl).complex_refractive_index()

    # Table
    print(f"\n{'λ nm':<8} {'Si n':>8} {'Si k':>8}  {'SiO₂ n':>8} {'SiO₂ k':>8}")
    print("-" * 50)
    for target in range(200, 1201, 100):
        i = np.argmin(np.abs(wl - target))
        print(f"{wl[i]:<8.0f} {asi_nk[i].real:8.4f} {asi_nk[i].imag:8.4f}"
              f"  {sio2_nk[i].real:8.4f} {sio2_nk[i].imag:8.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(wl, asi_nk.real, "b-", lw=1.5, label="n")
    ax1.plot(wl, asi_nk.imag, "r--", lw=1.5, label="k")
    ax1.set(title="a-Si  (UBF Monolog-Lorentz)", xlabel="λ (nm)", ylabel="n, k")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(wl, sio2_nk.real, "b-", lw=1.5, label="n")
    ax2.plot(wl, sio2_nk.imag, "r--", lw=1.5, label="k")
    ax2.set(title="SiO₂  (UBF Monolog-Lorentz)", xlabel="λ (nm)", ylabel="n, k",
            ylim=(0, 2.5))
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ubf_codylorentz_opt_demo.png", dpi=150)
    print("\nPlot → ubf_codylorentz_opt_demo.png")