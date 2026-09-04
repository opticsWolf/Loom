# -*- coding: utf-8 -*-
"""navette.synthesis — bridge from woven targets to optimizer-ready specs.

Converts a :class:`TargetCollection` (user-facing targets over free-form
spectral labels) into a native :class:`MeritSpec` (flat CurveId demands for
the thickness optimizer and the needle fold), and builds native
``SimCurves`` rows from solver outputs::

    from navette.synthesis import build_merit_spec, sim_curves_from_arrays

    spec = build_merit_spec(collection)          # MeritSpec (native)
    sim = sim_curves_from_arrays(angles, wl,    # SimCurves (native)
                                 {"Rs": rs, "Ts": ts})
    merit = spec.merit(sim, 1e6)
    folded = build_needle_targets(spec, angles, wl, sim)  # needle inputs

Spectral-label mapping (``(spectral, polarization)`` → CurveId):

===========  ===============================
label        CurveIds (s / p / u)
===========  ===============================
``"R"``      Rs / Rp / Ru (front reflectance)
``"T"``      Ts / Tp / Tu (front transmittance)
``"A"``      As / Ap / Au (absorptance 1 − R − T)
``"RB"``     RBs / RBp / RBu (back-reflectance)
``"TB"``     TBs / TBp / TBu (back-transmittance)
``"AB"``     ABs / ABp / ABu (back-absorptance)
===========  ===============================

Differential-phase labels (transmitted, synthesis-only quantities):

===========  =====================================================
label        meaning
===========  =====================================================
``"PDts"``   arg(t_s) minus the equivalent-medium reference
``"PDtp"``   arg(t_p) minus the equivalent-medium reference
===========  =====================================================

The reference is ``passes · 2π · n_inc · D · cosθ / λ`` (``passes = 1``
for transmitted): the propagation phase through a layer of incidence
medium of the coating's total thickness ``D``. ``PDts``/``PDtp`` map to
the ``Ts``/``Tp`` curves with ``phase=True`` (required — anything else
raises) plus the reference subtraction; ingestion forces phase
normalization (raw radians, ``nf = 1``). See
``docs/spectralweave-target-kinds.md`` for conventions (solver
forward-propagation sign, needle gain shift).

Any other label raises ``ValueError`` — the synthesis side speaks this fixed
vocabulary (the weaver itself accepts anything). Targets with
``phase=True`` become phase demands on the mapped curve's S-matrix element
(R → r_front, T → t_fwd, RB → r_back, TB → t_back; absorption and
unpolarized keys rejected); their values must be radians.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from navette._smatrix import (
    MeritSpec as _NativeMeritSpec,
    SimCurves as _NativeSimCurves,
    build_needle_targets as _native_fold,
)
from navette.spectralweave.target import TargetCollection

__all__ = [
    "MeritSpec",
    "SimCurves",
    "build_merit_spec",
    "sim_curves_from_arrays",
    "build_needle_targets",
    "apply_reference_rotation",
    "SPECTRAL_MAP",
    "DIFFERENTIAL_PASSES",
]

# Re-export native classes under friendly names.
MeritSpec = _NativeMeritSpec
SimCurves = _NativeSimCurves
build_needle_targets = _native_fold

SPECTRAL_MAP: Dict[Tuple[str, str], str] = {
    ("R", "s"): "Rs", ("R", "p"): "Rp", ("R", "u"): "Ru",
    ("T", "s"): "Ts", ("T", "p"): "Tp", ("T", "u"): "Tu",
    ("A", "s"): "As", ("A", "p"): "Ap", ("A", "u"): "Au",
    ("RB", "s"): "RBs", ("RB", "p"): "RBp", ("RB", "u"): "RBu",
    ("TB", "s"): "TBs", ("TB", "p"): "TBp", ("TB", "u"): "TBu",
    ("AB", "s"): "ABs", ("AB", "p"): "ABp", ("AB", "u"): "ABu",
}


# Differential-phase labels: spectral label → (host CurveId, polarization,
# reference passes). The label already encodes the polarization; a mismatch
# raises. `passes = 1` is single traversal (transmitted PDts/PDtp).
_DIFFERENTIAL: Dict[str, Tuple[str, str, float]] = {
    "PDts": ("Ts", "s", 1.0),
    "PDtp": ("Tp", "p", 1.0),
}
#: Reference passes per differential-phase label (``PDts``/``PDtp`` → 1.0).
DIFFERENTIAL_PASSES: Dict[str, float] = {
    label: passes for label, (_, _, passes) in _DIFFERENTIAL.items()
}


def _curve_id(spectral: str, polarization: str) -> str:
    if spectral in _DIFFERENTIAL:
        curve, pol, _ = _DIFFERENTIAL[spectral]
        if polarization != pol:
            raise ValueError(
                f"Cannot convert spectral={spectral!r}, polarization={polarization!r}: "
                f"{spectral!r} is {pol}-polarized (label encodes polarization)."
            )
        return curve
    try:
        return SPECTRAL_MAP[(spectral, polarization)]
    except KeyError:
        raise ValueError(
            f"Cannot convert spectral={spectral!r}, polarization={polarization!r}: "
            f"supported labels are {sorted({s for s, _ in SPECTRAL_MAP} | set(_DIFFERENTIAL))}, "
            f"polarizations 's'/'p'/'u'."
        ) from None


def build_merit_spec(collection: TargetCollection,
                     cache_size: int = 128,
                     tolerance_floor: float = 1e-12):
    """Compile a :class:`TargetCollection` into a native ``MeritSpec``.

    Ingestion (normalization, kind/band metadata) is shared verbatim with
    ``calculate_merit``: the collection is built into a ``TargetWeaver``
    and every entry exported, so merit values agree by construction.
    Phase-flagged targets become phase demands (radians, wrapped).
    """
    weaver = collection.build_weaver(cache_size=cache_size,
                                     tolerance_floor=tolerance_floor)
    entries = weaver.export_entries()
    # Join phase flags by creation order: spectral entries (uid order) align
    # with spectral_targets, angular frames (uid order) with angular_targets
    # (frames are pushed on creation; uids are monotonic).
    spec_targets = list(collection.spectral_targets)
    ang_targets = list(collection.angular_targets)
    spec_entries = []
    ang_frames: Dict[int, list] = {}
    for e in entries:
        uid = int(e["uid"])
        # Angular frames hold N single-point keys; spectral frames hold one
        # curve. Distinguish by grid length AND key count per frame below.
        ang_frames.setdefault(uid, []).append(e)
    # Spectral entries: frames with a multi-point grid.
    spectral_by_uid = sorted(
        ((uid, es[0]) for uid, es in ang_frames.items()
         if len(es) == 1 and np.asarray(es[0]["wavelengths"]).size > 1),
        key=lambda t: t[0],
    )
    angular_by_uid = sorted(
        ((uid, es) for uid, es in ang_frames.items()
         if not (len(es) == 1 and np.asarray(es[0]["wavelengths"]).size > 1)),
        key=lambda t: t[0],
    )
    if len(spectral_by_uid) != len(spec_targets):
        raise RuntimeError("build_merit_spec: spectral entry/target count mismatch.")
    if len(angular_by_uid) != len(ang_targets):
        raise RuntimeError("build_merit_spec: angular entry/target count mismatch.")

    spec = _NativeMeritSpec()
    keys: Dict[tuple, int] = {}

    def get_key(angle: float, curve: str) -> int:
        k = (float(angle), curve)
        idx = keys.get(k)
        if idx is None:
            idx = spec.add_key(float(angle), curve)
            keys[k] = idx
        return idx

    def convert_one(e, t) -> None:
        curve = _curve_id(t.spectral, t.polarization)
        ki = get_key(float(e["angle"]), curve)
        nf = float(e["norm_factor"])
        # PD labels are differential-phase: phase=True is required (a
        # differential intensity is meaningless) and ingestion already
        # forced the phase triple (raw radians, nf == 1).
        diff = _DIFFERENTIAL.get(t.spectral)
        if diff is not None and not t.phase:
            raise ValueError(
                f"spectral={t.spectral!r} is differential-phase: pass phase=True."
            )
        if t.phase:
            # Phase arm scales nothing (nf == 1 invariant): unscale the
            # resolved triple back to raw radians + raw band.
            nf = max(nf, 1e-300)
            norm = np.asarray(e["targets"], dtype=np.float64) / nf
            band = np.asarray(e["band"], dtype=np.float64) / nf
            mode, out_nf = "phase", 1.0
        else:
            norm = np.asarray(e["targets"], dtype=np.float64)
            band = np.asarray(e["band"], dtype=np.float64)
            mode, out_nf = str(e["mode"]), nf
        spec.add_target(
            ki, np.ascontiguousarray(e["wavelengths"], dtype=np.float64),
            np.ascontiguousarray(norm, dtype=np.float64),
            np.ascontiguousarray(e["tolerances"], dtype=np.float64),
            str(e["kind"]), mode, out_nf,
            band=np.ascontiguousarray(band, dtype=np.float64),
            phase=bool(t.phase),
            differential_passes=(diff[2] if diff is not None else None),
        )

    for (_, e), t in zip(spectral_by_uid, spec_targets):
        convert_one(e, t)

    for (_, es), t in zip(angular_by_uid, ang_targets):
        for e in sorted(es, key=lambda d: float(d["angle"])):
            convert_one(e, t)
    return spec


def apply_reference_rotation(cplx, wavelengths, angle_deg: float,
                             n_inc: float = 1.0, total_d: float = 0.0,
                             passes: float = 1.0):
    """Rotate complex amplitudes into differential-phase space.

    Multiplies by ``exp(-i·ref)`` with
    ``ref = passes · 2π · n_inc · total_d · cosθ / λ`` (``wavelengths`` and
    ``total_d`` share units; ``angle_deg`` is degrees in the incidence
    medium), so ``arg()`` of the result is the differential phase
    ``Δφ = arg(a) − ref``. Last axis is wavelength; leading axes (angles)
    broadcast. An absolute-phase demand on the rotated rows is exactly a
    differential-phase demand on the raw rows (the test oracle for
    ``PDts``/``PDtp`` — native and numpy paths must agree to 1e-12).
    """
    a = np.asarray(cplx)
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    if a.shape[-1] != wl.size:
        raise ValueError(
            f"last axis {a.shape[-1]} != {wl.size} wavelengths."
        )
    ref = (float(passes) * 2.0 * np.pi * float(n_inc) * float(total_d)
           * np.cos(np.radians(float(angle_deg))) / wl)
    rot = np.exp(-1j * ref).reshape((1,) * (a.ndim - 1) + (-1,))
    return a * rot


def sim_curves_from_arrays(angles, wavelengths,
                           curves: Dict[str, np.ndarray],
                           complex_curves: Dict[str, np.ndarray] | None = None,
                           total_d: float = 0.0,
                           n_front: float = 1.0,
                           n_back: float = 1.0):
    """Build a native ``SimCurves`` from row-major ``[n_angles, n_wavs]`` maps.

    ``curves`` maps CurveId codes (``"Rs"``…``"ABu"``) to float rows;
    ``complex_curves`` maps ``Rs/Rp/Ts/Tp/RBs/RBp/TBs/TBp`` to complex rows
    for phase demands. Lengths are validated (FFI safety).
    ``total_d``/``n_front``/``n_back`` are the stack metadata for
    differential-phase (``PDts``/``PDtp``) demands — total coating
    thickness (same units as ``wavelengths``) and the real
    incidence/exit indices. Defaults zero the reference.
    """
    angles = np.ascontiguousarray(np.asarray(angles, dtype=np.float64)).ravel()
    wavelengths = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64)).ravel()
    n = angles.size * wavelengths.size
    sim = _NativeSimCurves(angles, wavelengths, float(total_d),
                            float(n_front), float(n_back))
    for code, arr in curves.items():
        v = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).ravel()
        if v.size != n:
            raise ValueError(f"curve {code!r}: {v.size} entries != {n} grid points.")
        sim.set_curve(code, v)
    for code, arr in (complex_curves or {}).items():
        v = np.ascontiguousarray(np.asarray(arr, dtype=np.complex128)).ravel()
        if v.size != n:
            raise ValueError(f"complex curve {code!r}: {v.size} entries != {n} grid points.")
        sim.set_complex(code, v)
    return sim
