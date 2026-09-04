"""navette.needle — Python interface to the analytic needle operator.

Thin wrapper around ``needle_engine`` in the compiled Rust extension, which
exposes the Tikhonravov needle-operator sensitivities computed by
``needle_operator.rs``:

  * ``P(z)``      — coherent merit gradient per depth (sub-block confined)
  * ``Pmb(z)``    — merit gradient through incoherent stacks (Modes A/B),
                    routed around flagged spacer layers via intensity cascade
  * ``dphi/dgd/dgdd/dtod/dfod`` — phase-dispersion sensitivities
                    (∂ⁿφ/∂ωⁿ per δ) obtained by spectral differentiation

The ``NeedleRequest`` bit values are bound directly from the Rust module's
``NREQ_*`` constants, so they can never drift out of sync.

Conventions
-----------
* Depths ``z_grid`` are absolute, measured from the top of layer
  ``start_idx + 1`` (with ``start_idx = 0``, from the top of layer 1).
* Needle hosts must lie strictly interior to the coherent sub-block
  ``[start_idx, end_idx]``; insertion into an incoherent-flagged spacer is
  rejected by the engine.
* Merit convention (half-gradient): a single point contributes
  ``2·w·(R − R_target)·Re{conj(r)·∂r/∂δ}``; aggregate over points at the call
  site, e.g. for a GDD merit ``∂F/∂δ(z) = Σ_k 2·w_k·(GDD_k − GDD_t_k)·dgdd[k,z]``.
  The T/A siblings follow the same convention with flux-corrected
  ``T = |t_fwd|²·f`` and ``A = 1 − R − T`` (front incidence); the phase
  channel is the full gradient ``2·w·wrap(φ − φ_t)·Q`` (phase is not an
  intensity, so no half factor applies).
* ``Pmb`` stays reflectance-only (intensity cascade); T/A/phase gradients
  are coherent-path only.
* Results are returned as ``(n_angles, n_wavs, n_depths)`` float64 arrays.
"""

from __future__ import annotations

from enum import IntFlag
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

# --- compiled Rust extension -------------------------------------------------
try:
    from navette._smatrix import (
        needle_engine as _rs_needle_engine,
        NREQ_P as _NREQ_P,
        NREQ_P_MB as _NREQ_P_MB,
        NREQ_P_T as _NREQ_P_T,
        NREQ_P_A as _NREQ_P_A,
        NREQ_P_PHI as _NREQ_P_PHI,
        NREQ_DPHI as _NREQ_DPHI,
        NREQ_DGD as _NREQ_DGD,
        NREQ_DGDD as _NREQ_DGDD,
        NREQ_DTOD as _NREQ_DTOD,
        NREQ_DFOD as _NREQ_DFOD,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Could not import the compiled `_smatrix` extension. Build the Rust "
        "crate (e.g. `maturin develop`) so that `_smatrix` is importable."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover
    from .smatrix import ScatterMatrix


__all__ = ["NeedleRequest", "needle_gradient"]


class NeedleRequest(IntFlag):
    """Needle-operator selectors, OR-ed together and passed to
    :func:`needle_gradient`. Values are bound from the Rust constants."""

    P = _NREQ_P            # coherent merit gradient P(z)
    P_MB = _NREQ_P_MB      # multiblock P(z) through the intensity cascade
    P_T = _NREQ_P_T        # coherent transmission-merit gradient P_T(z)
    P_A = _NREQ_P_A        # coherent absorption-merit gradient P_A(z)
    P_PHI = _NREQ_P_PHI    # coherent phase-merit gradient P_PHI(z)
    DPHI = _NREQ_DPHI      # ∂φ/∂δ
    DGD = _NREQ_DGD        # ∂(dφ/dω)/∂δ  (group delay)
    DGDD = _NREQ_DGDD      # ∂(d²φ/dω²)/∂δ  (group-delay dispersion)
    DTOD = _NREQ_DTOD      # ∂TOD/∂δ
    DFOD = _NREQ_DFOD      # ∂FOD/∂δ

    # Convenience bundle: full dispersion ladder
    DISPERSION = DPHI | DGD | DGDD | DTOD | DFOD


# Output-key suffixes per dispersion derivative order (must match
# needle_engine.rs's DISP_KEYS).
_DISP_KEYS = ("dphi", "dgd", "dgdd", "dtod", "dfod")


def needle_gradient(
    stack: "ScatterMatrix",
    needle_n: Union[complex, Sequence[complex], np.ndarray],
    z_grid: Sequence[float],
    request: Union[int, NeedleRequest],
    *,
    targets_r: Union[float, Sequence[float], np.ndarray, None] = None,
    weights_r: Union[float, Sequence[float], np.ndarray, None] = None,
    targets_t: Union[float, Sequence[float], np.ndarray, None] = None,
    weights_t: Union[float, Sequence[float], np.ndarray, None] = None,
    targets_a: Union[float, Sequence[float], np.ndarray, None] = None,
    weights_a: Union[float, Sequence[float], np.ndarray, None] = None,
    targets_phi: Union[float, Sequence[float], np.ndarray, None] = None,
    weights_phi: Union[float, Sequence[float], np.ndarray, None] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    channel: int = 0,
    pol: str = "sp",
    host_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, np.ndarray]:
    """Analytic needle-operator gradients on ``stack``.

    Parameters
    ----------
    stack : ScatterMatrix
        The configured stack (wavelengths, angles, layers, roughness).
    needle_n : complex or (n_wavs,) complex array
        Candidate needle material index; scalar means wavelength-independent.
    z_grid : (n_depths,) float array
        Absolute depths at which sensitivities are evaluated.
    request : NeedleRequest or int
        Bit mask of desired observables.
    targets_r / weights_r : None, scalar, or (n_angles·n_wavs,) array
        Per-spectral-point reflectance targets / merit weights (angle-major).
        Defaults: target 0, weight 1.
    targets_t / weights_t : same layout, for ``P_T`` (flux-corrected
        transmittance ``|t_fwd|²·f``, front incidence).
    targets_a / weights_a : same layout, for ``P_A`` (absorptance
        ``1 − R − T``, front incidence).
    targets_phi / weights_phi : same layout, for ``P_PHI`` (phase of the
        ``channel`` element, in radians; residual wrapped to [-π, π]).
        Only honoured when ``P_PHI`` is requested.
    start_idx, end_idx : int
        Coherent sub-block confinement; ``end_idx`` is the *index* of the
        terminating medium (default: last layer). Hosts must lie strictly
        inside.
    channel : {0, 1, 2, 3}
        Which composed element drives the dispersion channels:
        0 = r_front, 1 = t_back, 2 = t_fwd, 3 = r_back.
    pol : {'s', 'p', 'sp'}
        Polarization branches to evaluate ('sp' = both in one sweep).
    host_mask : (n_layers,) bool array, optional
        Restricts admissible hosts for the multiblock path.

    Returns
    -------
    dict[str, ndarray]
        ``P_s``, ``P_p``, ``Pmb_s``, ``Pmb_p``, ``P_T_s``, ``P_A_p``,
        ``P_PHI_s``, ``dphi_s``, ``dgdd_p``, ... depending on ``request``
        and ``pol``; each shaped ``(n_angles, n_wavs, n_depths)``.
    """
    if pol not in ("s", "p", "sp"):
        raise ValueError("pol must be 's', 'p', or 'sp'.")
    req = int(request)
    if req == 0:
        raise ValueError("Empty request mask.")

    n_wavs, n_angles, n_layers = stack.n_wavs, stack.n_angles, stack.n_layers
    total_points = n_angles * n_wavs

    z = np.ascontiguousarray(z_grid, dtype=np.float64).ravel()
    if z.size == 0:
        raise ValueError("`z_grid` must be non-empty.")

    npw = np.broadcast_to(np.asarray(needle_n, dtype=np.complex128), (n_wavs,))
    npw = np.ascontiguousarray(npw)

    def _per_point(value, name, dtype):
        if value is None:
            return None
        arr = np.asarray(value, dtype=dtype).ravel()
        if arr.size == 1:
            arr = np.full(total_points, arr[0], dtype=dtype)
        elif arr.size != total_points:
            raise ValueError(
                f"`{name}` must be a scalar or have n_angles*n_wavs = "
                f"{total_points} entries (angle-major); got {arr.size}."
            )
        return np.ascontiguousarray(arr)

    tgt = _per_point(targets_r, "targets_r", np.float64)
    wgt = _per_point(weights_r, "weights_r", np.float64)
    tgt_t = _per_point(targets_t, "targets_t", np.float64)
    wgt_t = _per_point(weights_t, "weights_t", np.float64)
    tgt_a = _per_point(targets_a, "targets_a", np.float64)
    wgt_a = _per_point(weights_a, "weights_a", np.float64)
    tgt_phi = _per_point(targets_phi, "targets_phi", np.float64)
    wgt_phi = _per_point(weights_phi, "weights_phi", np.float64)
    mask = None if host_mask is None else np.ascontiguousarray(
        np.asarray(host_mask, dtype=bool).ravel()
    )

    if (req & NeedleRequest.P_MB) and not np.any(stack.incoherent_flags):
        pass  # legal: all-coherent flags reduce exactly onto the P path

    out = _rs_needle_engine(
        stack.wavls,
        stack.sin_theta,
        int(n_layers),
        stack._n_stack_cache,
        stack.thicknesses,
        stack.roughness_types,
        stack.roughness_values,
        npw,
        z,
        req,
        stack.incoherent_flags if (req & NeedleRequest.P_MB) else None,
        tgt,
        wgt,
        tgt_t,
        wgt_t,
        tgt_a,
        wgt_a,
        tgt_phi,
        wgt_phi,
        int(start_idx),
        None if end_idx is None else int(end_idx),
        int(channel),
        pol != "p",
        pol != "s",
        mask,
    )

    # Reshape flat [n_points, n_z] buffers into (n_angles, n_wavs, n_depths).
    nz = z.size
    return {
        k: np.asarray(v).reshape(n_angles, n_wavs, nz) for k, v in out.items()
    }
