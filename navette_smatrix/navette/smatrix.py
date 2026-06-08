"""navette_smatrix — a clean Python interface to the ``smatrix`` Rust extension.

This module is a thin, self-contained wrapper around the compiled Rust core. It
owns no physics: every quantity is produced by the request-driven engine
(``core_engine`` / func_6) or the eigenmode solver. The wrapper's job is to

  * hold a layer stack + wavelength/angle grid,
  * marshal them into the exact array layouts the Rust functions expect,
  * turn a ``Request`` mask into a dict of named, correctly-shaped results,
  * expose the eigenmode tools (landscape scan, mode finding, field profile).

Conventions
-----------
* ``layer_indices`` is complex, shape ``(n_layers, n_wavs)`` — row 0 is the
  ambient/incident medium, the last row is the substrate. A 1-D array of length
  ``n_layers`` is accepted and treated as wavelength-independent.
* Angles are in degrees by default (pass ``angles_in_radians=True`` to override).
* ``thicknesses``, ``incoherent_flags`` and the two roughness arrays are
  per-layer (length ``n_layers``). Roughness entry ``k`` describes the interface
  at the *front* of layer ``k`` (so ambient entries are unused); ambient and
  substrate thicknesses should be 0.
* Results are 2-D ``[n_angles, n_wavs]``, or 1-D ``[n_wavs]`` when a single angle
  was supplied (the angle axis is squeezed away).

The ``Request`` bit positions MUST stay in sync with the ``REQ_*`` constants in
``func_6.rs``.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# --- compiled Rust extension -------------------------------------------------
# Only the symbols this wrapper actually uses are imported; the legacy
# func_4/func_5 engines are intentionally not referenced.
try:
    from ._smatrix import (
        core_engine,
        w_function,
        redheffer_product_real,
        redheffer_product_cross,
        redheffer_product_complex_field,
        solve_coherent_block_fields,
        scan_landscape as _rs_scan_landscape,
        find_local_minima as _rs_find_local_minima,
        nelder_mead as _rs_nelder_mead,
        field_profile as _rs_field_profile,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Could not import the compiled `smatrix` extension. Build the Rust "
        "crate (e.g. `maturin develop` / `maturin build`) so that `smatrix` is "
        "importable, then retry."
    ) from exc


__all__ = [
    "Request",
    "CoherenceMode",
    "Pol",
    "RoughnessType",
    "ScatterMatrix",
    "EigenLandscape",
    # low-level Rust passthroughs
    "w_function",
    "redheffer_product_real",
    "redheffer_product_cross",
    "redheffer_product_complex_field",
    "solve_coherent_block_fields",
]


# ─── Flags / enums (mirror func_6.rs REQ_* and the coherence modes) ──────────
class Request(IntFlag):
    """Observable selectors. OR them together and pass to :meth:`ScatterMatrix.compute`."""

    RS = 1 << 0
    RP = 1 << 1
    TS = 1 << 2
    TP = 1 << 3
    R_AVG = 1 << 4
    T_AVG = 1 << 5
    A_S = 1 << 6
    A_P = 1 << 7
    A_AVG = 1 << 8
    PSI_R = 1 << 9
    PSI_T = 1 << 10
    DELTA_R = 1 << 11
    DELTA_T = 1 << 12
    DOP_R = 1 << 13
    DOP_T = 1 << 14
    DIATT_R = 1 << 15
    DIATT_T = 1 << 16
    S0_R = 1 << 17
    S1_R = 1 << 18
    S2_R = 1 << 19
    S3_R = 1 << 20
    S0_T = 1 << 21
    S1_T = 1 << 22
    S2_T = 1 << 23
    S3_T = 1 << 24
    PHI_RS = 1 << 25
    PHI_RP = 1 << 26
    PHI_TS = 1 << 27
    PHI_TP = 1 << 28
    RS_C = 1 << 29
    RP_C = 1 << 30
    TS_C = 1 << 31
    TP_C = 1 << 32
    CROSS_R = 1 << 33
    CROSS_T = 1 << 34
    RETARD_R = 1 << 35
    RETARD_T = 1 << 36
    DISP_R_S = 1 << 37  # emits GD_R_s, GDD_R_s, TOD_R_s, FOD_R_s
    DISP_R_P = 1 << 38
    DISP_T_S = 1 << 39
    DISP_T_P = 1 << 40

    # Convenience bundles
    PHOTOMETRY = RS | RP | TS | TP | R_AVG | T_AVG
    ELLIPSOMETRY = PSI_R | DELTA_R | DOP_R | PSI_T | DELTA_T | DOP_T
    ABSORPTION = A_S | A_P | A_AVG
    STOKES_R = S0_R | S1_R | S2_R | S3_R
    STOKES_T = S0_T | S1_T | S2_T | S3_T


class CoherenceMode(IntEnum):
    FRONT_BLOCK = 0       # incoherent gaps applied at flagged boundaries
    COHERENCY_MATRIX = 1  # tracks the complex p-s coherency channel (Mode B)
    FULLY_COHERENT = 2     # whole stack treated as one coherent block


class Pol(IntEnum):
    S = 0
    P = 1


class RoughnessType(IntEnum):
    NONE = 0
    LINEAR = 1
    STEP = 2
    EXPONENTIAL = 3
    GAUSSIAN = 4
    NEVOT_CROCE = 5


# Maps each request bit to the output dict key(s) the engine emits for it. Used
# for validation and introspection (the engine itself returns exactly these).
_DISP_KEYS = {
    Request.DISP_R_S: ("GD_R_s", "GDD_R_s", "TOD_R_s", "FOD_R_s"),
    Request.DISP_R_P: ("GD_R_p", "GDD_R_p", "TOD_R_p", "FOD_R_p"),
    Request.DISP_T_S: ("GD_T_s", "GDD_T_s", "TOD_T_s", "FOD_T_s"),
    Request.DISP_T_P: ("GD_T_p", "GDD_T_p", "TOD_T_p", "FOD_T_p"),
}
_SCALAR_KEYS = {
    Request.RS: "Rs", Request.RP: "Rp", Request.TS: "Ts", Request.TP: "Tp",
    Request.R_AVG: "R_avg", Request.T_AVG: "T_avg",
    Request.A_S: "A_s", Request.A_P: "A_p", Request.A_AVG: "A_avg",
    Request.PSI_R: "Psi_R", Request.PSI_T: "Psi_T",
    Request.DELTA_R: "Delta_R", Request.DELTA_T: "Delta_T",
    Request.DOP_R: "DOP_R", Request.DOP_T: "DOP_T",
    Request.DIATT_R: "Diattenuation_R", Request.DIATT_T: "Diattenuation_T",
    Request.S0_R: "S0_R", Request.S1_R: "S1_R", Request.S2_R: "S2_R", Request.S3_R: "S3_R",
    Request.S0_T: "S0_T", Request.S1_T: "S1_T", Request.S2_T: "S2_T", Request.S3_T: "S3_T",
    Request.RETARD_R: "Retardance_R", Request.RETARD_T: "Retardance_T",
    Request.PHI_RS: "phi_rs", Request.PHI_RP: "phi_rp",
    Request.PHI_TS: "phi_ts", Request.PHI_TP: "phi_tp",
    Request.RS_C: "rs_c", Request.RP_C: "rp_c", Request.TS_C: "ts_c", Request.TP_C: "tp_c",
    Request.CROSS_R: "cross_R", Request.CROSS_T: "cross_T",
}


def expected_keys(request: Union[int, Request]) -> List[str]:
    """Return the output dict keys the engine will emit for ``request``."""
    req = Request(int(request))
    keys: List[str] = []
    for bit, name in _SCALAR_KEYS.items():
        if req & bit:
            keys.append(name)
    for bit, names in _DISP_KEYS.items():
        if req & bit:
            keys.extend(names)
    return keys


# ─── Eigenmode result container ──────────────────────────────────────────────
class EigenLandscape:
    """Result of an eigenmode landscape scan over the complex effective index."""

    def __init__(self, n_real: np.ndarray, n_imag: np.ndarray, values: np.ndarray):
        self.n_real = np.asarray(n_real, dtype=np.float64)  # shape (points_real,)
        self.n_imag = np.asarray(n_imag, dtype=np.float64)  # shape (points_imag,)
        self.values = np.asarray(values, dtype=np.float64)   # shape (points_imag, points_real)

    def local_minima(self, median_factor: float = 0.1) -> List[complex]:
        """Coarse local minima below ``median_factor * median(values)``."""
        mins = _rs_find_local_minima(
            np.ascontiguousarray(self.values, dtype=np.float64),
            list(map(float, self.n_real)),
            list(map(float, self.n_imag)),
            float(median_factor),
        )
        return [complex(re, im) for (re, im) in mins]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EigenLandscape(real[{self.n_real.size}], imag[{self.n_imag.size}], "
            f"values{self.values.shape})"
        )


# ─── Main entry point ────────────────────────────────────────────────────────
class ScatterMatrix:
    """Multilayer optical solver over a (wavelength, angle) grid.

    Parameters
    ----------
    layer_indices : complex array, shape (n_layers, n_wavs) or (n_layers,)
        Complex refractive indices per layer. Row 0 = ambient, last row =
        substrate. A 1-D array is treated as wavelength-independent.
    thicknesses : float array, shape (n_layers,)
        Physical thickness per layer (same length unit as ``wavelengths``).
        Ambient and substrate should be 0.
    wavelengths : float array, shape (n_wavs,)
    angles : float or float array
        Angle(s) of incidence, degrees unless ``angles_in_radians=True``.
    incoherent_flags : int array, shape (n_layers,), optional
        Non-zero where a layer breaks phase coherence (thick substrate). Default
        all-zero (fully coherent boundaries).
    roughness_types : int array, shape (n_layers,), optional
        Per-interface roughness model (see :class:`RoughnessType`). Default none.
    roughness_values : float array, shape (n_layers,), optional
        Per-interface roughness sigma (same unit as ``wavelengths``). Default 0.
    coherence_mode : CoherenceMode, optional
        Default ``FRONT_BLOCK``.
    angles_in_radians : bool, optional
        Treat ``angles`` as radians. Default False (degrees).
    """

    def __init__(
        self,
        layer_indices: np.ndarray,
        thicknesses: Sequence[float],
        *,
        wavelengths: Sequence[float],
        angles: Union[float, Sequence[float]],
        incoherent_flags: Optional[Sequence[int]] = None,
        roughness_types: Optional[Sequence[int]] = None,
        roughness_values: Optional[Sequence[float]] = None,
        coherence_mode: Union[CoherenceMode, int] = CoherenceMode.FRONT_BLOCK,
        angles_in_radians: bool = False,
    ):
        self.wavls = np.ascontiguousarray(wavelengths, dtype=np.float64).ravel()
        self.n_wavs = self.wavls.size
        if self.n_wavs == 0:
            raise ValueError("`wavelengths` must be non-empty.")

        theta = np.atleast_1d(np.asarray(angles, dtype=np.float64)).ravel()
        if not angles_in_radians:
            theta = np.radians(theta)
        self.sin_theta = np.ascontiguousarray(np.sin(theta), dtype=np.float64)
        self.n_angles = self.sin_theta.size
        if self.n_angles == 0:
            raise ValueError("`angles` must be non-empty.")

        # Normalise layer_indices to (n_layers, n_wavs) complex.
        idx = np.asarray(layer_indices, dtype=np.complex128)
        if idx.ndim == 1:
            idx = np.repeat(idx[:, None], self.n_wavs, axis=1)
        elif idx.ndim != 2:
            raise ValueError("`layer_indices` must be 1-D or 2-D.")
        self.n_layers = idx.shape[0]
        if self.n_layers < 2:
            raise ValueError("Need at least 2 layers (ambient + substrate).")
        if idx.shape[1] != self.n_wavs:
            raise ValueError(
                f"layer_indices second axis ({idx.shape[1]}) must equal "
                f"number of wavelengths ({self.n_wavs})."
            )
        # Keep both views: (n_layers, n_wavs) for slicing, (n_wavs, n_layers) for the cache.
        self._indices = np.ascontiguousarray(idx)                 # (n_layers, n_wavs)
        self._indices_wav_major = np.ascontiguousarray(idx.T)     # (n_wavs, n_layers)

        self.thicknesses = self._as_layer_array(thicknesses, np.float64, "thicknesses", 0.0)
        self.incoherent_flags = self._as_layer_array(
            incoherent_flags, np.int32, "incoherent_flags", 0
        )
        self.roughness_types = self._as_layer_array(
            roughness_types, np.int32, "roughness_types", 0
        )
        self.roughness_values = self._as_layer_array(
            roughness_values, np.float64, "roughness_values", 0.0
        )
        self.coherence_mode = CoherenceMode(int(coherence_mode))

        # Flat re/im-interleaved cache the Rust core_engine expects:
        # layout per wavelength is [Re0, Im0, Re1, Im1, ...]; base = w * n_layers * 2.
        self._n_stack_cache = self._indices_wav_major.view(np.float64).ravel()

    # ---- input helpers ------------------------------------------------------
    def _as_layer_array(self, value, dtype, name, default):
        if value is None:
            return np.full(self.n_layers, default, dtype=dtype)
        arr = np.ascontiguousarray(value, dtype=dtype).ravel()
        if arr.size != self.n_layers:
            raise ValueError(
                f"`{name}` length {arr.size} must equal n_layers {self.n_layers}."
            )
        return arr

    def _squeeze(self, arr: np.ndarray) -> np.ndarray:
        """Drop the angle axis when a single angle was supplied."""
        return arr[0] if self.n_angles == 1 else arr

    # ---- core engine --------------------------------------------------------
    def compute(
        self, request: Union[int, Request], *, squeeze: bool = True
    ) -> Dict[str, np.ndarray]:
        """Run the engine for ``request`` and return ``{name: ndarray}``.

        The returned dict contains exactly the keys implied by ``request`` (see
        :func:`expected_keys`). Arrays are ``[n_angles, n_wavs]``, or ``[n_wavs]``
        when a single angle was supplied and ``squeeze=True``.
        """
        req = int(request)
        if req == 0:
            raise ValueError("Empty request mask: select at least one observable.")
        out = core_engine(
            self.wavls,
            self.sin_theta,
            int(self.n_layers),
            self._n_stack_cache,
            self.thicknesses,
            self.incoherent_flags,
            self.roughness_types,
            self.roughness_values,
            int(self.coherence_mode),
            int(req),
        )
        if not squeeze or self.n_angles != 1:
            return dict(out)
        return {k: self._squeeze(v) for k, v in out.items()}

    # ---- convenience views over `compute` -----------------------------------
    def reflectance_transmittance(self, pol: str = "u") -> Dict[str, np.ndarray]:
        """R/T spectra. ``pol`` is 's', 'p', or 'u' (unpolarized = both)."""
        if pol not in ("s", "p", "u"):
            raise ValueError("pol must be 's', 'p', or 'u'.")
        req = Request.R_AVG | Request.T_AVG
        if pol in ("s", "u"):
            req |= Request.RS | Request.TS
        if pol in ("p", "u"):
            req |= Request.RP | Request.TP
        return self.compute(req)

    def ellipsometry(self, *, transmission: bool = False) -> Dict[str, np.ndarray]:
        """Psi/Delta/DOP plus R (and optionally T) spectra."""
        req = (
            Request.PSI_R | Request.DELTA_R | Request.DOP_R
            | Request.RS | Request.RP | Request.R_AVG
        )
        if transmission:
            req |= (
                Request.PSI_T | Request.DELTA_T | Request.DOP_T
                | Request.TS | Request.TP | Request.T_AVG
            )
        return self.compute(req)

    def absorption(self) -> Dict[str, np.ndarray]:
        """Per-polarization and averaged absorptance (A = 1 - R - T)."""
        return self.compute(Request.ABSORPTION)

    def complex_amplitudes(self) -> Dict[str, np.ndarray]:
        """Complex r/t coefficients (rs_c, rp_c, ts_c, tp_c)."""
        return self.compute(Request.RS_C | Request.RP_C | Request.TS_C | Request.TP_C)

    def stokes(self, *, reflection: bool = True, transmission: bool = False) -> Dict[str, np.ndarray]:
        """Stokes parameters S0..S3 for reflection and/or transmission."""
        req = Request(0)
        if reflection:
            req |= Request.STOKES_R
        if transmission:
            req |= Request.STOKES_T
        if int(req) == 0:
            raise ValueError("Select reflection and/or transmission.")
        return self.compute(req)

    def dispersion(
        self, *, reflection: bool = True, transmission: bool = False,
        s_pol: bool = True, p_pol: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Group delay and higher orders (GD/GDD/TOD/FOD).

        Physically meaningful only for coherent stacks (``FULLY_COHERENT`` mode
        or a stack with no incoherent boundaries). Higher orders amplify
        numerical noise — validate against a fine-grid spline if precision
        matters.
        """
        req = Request(0)
        if reflection and s_pol:
            req |= Request.DISP_R_S
        if reflection and p_pol:
            req |= Request.DISP_R_P
        if transmission and s_pol:
            req |= Request.DISP_T_S
        if transmission and p_pol:
            req |= Request.DISP_T_P
        if int(req) == 0:
            raise ValueError("Select at least one channel for dispersion.")
        return self.compute(req)

    def energy_conservation(self) -> np.ndarray:
        """``max(|1 - Rs - Ts|, |1 - Rp - Tp|)`` per grid point.

        Exact and free — derived from intensities. ~0 for lossless stacks; for
        absorbing stacks it equals the absorptance and lies in [0, 1].
        """
        out = self.compute(
            Request.RS | Request.RP | Request.TS | Request.TP, squeeze=False
        )
        cons = np.maximum(
            np.abs(1.0 - out["Rs"] - out["Ts"]),
            np.abs(1.0 - out["Rp"] - out["Tp"]),
        )
        return self._squeeze(cons)

    # ---- eigenmode tools ----------------------------------------------------
    def _index_column(
        self, wavelength: Optional[float] = None, wav_index: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """Return ``(lam, per_layer_complex_indices)`` for one wavelength."""
        if wav_index is None:
            if wavelength is None:
                if self.n_wavs != 1:
                    raise ValueError(
                        "Specify `wavelength` or `wav_index` (grid has "
                        f"{self.n_wavs} wavelengths)."
                    )
                wav_index = 0
            else:
                wav_index = int(np.argmin(np.abs(self.wavls - float(wavelength))))
        lam = float(self.wavls[wav_index])
        col = np.ascontiguousarray(self._indices[:, wav_index], dtype=np.complex128)
        return lam, col

    def eigenmode_landscape(
        self,
        n_real_range: Tuple[float, float],
        n_imag_range: Tuple[float, float],
        *,
        resolution: Tuple[int, int] = (200, 200),
        pol: Union[Pol, int] = Pol.S,
        wavelength: Optional[float] = None,
        wav_index: Optional[int] = None,
    ) -> EigenLandscape:
        """Scan ``|1/r(n_eff)|^2`` over a complex effective-index box.

        ``resolution`` is ``(points_real, points_imag)``.
        """
        lam, col = self._index_column(wavelength, wav_index)
        points_real, points_imag = int(resolution[0]), int(resolution[1])
        real_vals, imag_vals, land = _rs_scan_landscape(
            col,
            self.thicknesses,
            self.roughness_types,
            self.roughness_values,
            lam,
            int(pol),
            float(n_real_range[0]), float(n_real_range[1]),
            float(n_imag_range[0]), float(n_imag_range[1]),
            points_real, points_imag,
        )
        return EigenLandscape(real_vals, imag_vals, np.asarray(land))

    def refine_mode(
        self,
        guess: complex,
        *,
        pol: Union[Pol, int] = Pol.S,
        wavelength: Optional[float] = None,
        wav_index: Optional[int] = None,
        step: float = 1e-3,
        tol: float = 1e-9,
        max_iter: int = 200,
    ) -> Tuple[complex, float]:
        """Nelder-Mead refine a single complex eigenmode guess.

        Returns ``(n_eff, characteristic_value)``.
        """
        lam, col = self._index_column(wavelength, wav_index)
        re, im, val = _rs_nelder_mead(
            col,
            self.thicknesses,
            self.roughness_types,
            self.roughness_values,
            lam,
            int(pol),
            (float(guess.real), float(guess.imag)),
            float(step), float(tol), int(max_iter),
        )
        return complex(re, im), float(val)

    def find_eigenmodes(
        self,
        n_real_range: Tuple[float, float],
        n_imag_range: Tuple[float, float],
        *,
        resolution: Tuple[int, int] = (200, 200),
        median_factor: float = 0.1,
        refine: bool = True,
        pol: Union[Pol, int] = Pol.S,
        wavelength: Optional[float] = None,
        wav_index: Optional[int] = None,
    ) -> List[complex]:
        """Scan, locate coarse minima, and (optionally) Nelder-Mead refine each."""
        land = self.eigenmode_landscape(
            n_real_range, n_imag_range, resolution=resolution,
            pol=pol, wavelength=wavelength, wav_index=wav_index,
        )
        seeds = land.local_minima(median_factor=median_factor)
        if not refine:
            return seeds
        refined: List[complex] = []
        for s in seeds:
            n_eff, _ = self.refine_mode(
                s, pol=pol, wavelength=wavelength, wav_index=wav_index
            )
            refined.append(n_eff)
        return refined

    def field_profile(
        self,
        n_eff: complex,
        *,
        pol: Union[Pol, int] = Pol.S,
        wavelength: Optional[float] = None,
        wav_index: Optional[int] = None,
        points_per_layer: int = 50,
    ) -> Dict[str, np.ndarray]:
        """``|E(z)|`` through the stack for a given eigenmode.

        Returns a dict with ``z`` (positions), ``E`` (normalised |E|, max=1),
        ``layer_start`` / ``layer_end`` (per finite layer), and ``layer_index``
        (complex n of each finite layer).
        """
        lam, col = self._index_column(wavelength, wav_index)
        z, e, lstart, lend, lidx = _rs_field_profile(
            col,
            self.thicknesses,
            self.roughness_types,
            self.roughness_values,
            lam,
            complex(n_eff),
            int(pol),
            int(points_per_layer),
        )
        return {
            "z": np.asarray(z, dtype=np.float64),
            "E": np.asarray(e, dtype=np.float64),
            "layer_start": np.asarray(lstart, dtype=np.float64),
            "layer_end": np.asarray(lend, dtype=np.float64),
            "layer_index": np.asarray(lidx, dtype=np.complex128),
        }

    # ---- misc ---------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ScatterMatrix(n_layers={self.n_layers}, n_wavs={self.n_wavs}, "
            f"n_angles={self.n_angles}, mode={self.coherence_mode.name})"
        )
