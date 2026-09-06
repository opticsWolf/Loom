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
    # Join each exported entry with its Python target by grid content:
    # spectral frames hold the target's wavelength grid at its angle,
    # angular frames hold one single-point entry per angle. (Matching by
    # grid length alone misclassifies single-point spectral targets.)
    spec_targets = list(collection.spectral_targets)
    ang_targets = list(collection.angular_targets)
    by_uid: Dict[int, list] = {}
    for e in entries:
        by_uid.setdefault(int(e["uid"]), []).append(e)
    used: set = set()

    def take_spectral(t) -> dict:
        grid = np.asarray(t.wavelengths, dtype=np.float64).ravel()
        for uid in sorted(by_uid):
            if uid in used:
                continue
            es = by_uid[uid]
            if (len(es) == 1 and es[0]["spectral"] == t.spectral
                    and es[0]["polarization"] == t.polarization
                    and float(es[0]["angle"]) == float(t.angle)
                    and np.allclose(np.asarray(es[0]["wavelengths"],
                                               dtype=np.float64).ravel(),
                                    grid)):
                used.add(uid)
                return es[0]
        raise RuntimeError(
            "build_merit_spec: no exported entry matches a spectral target "
            f"(spectral={t.spectral!r}, polarization={t.polarization!r}, "
            f"angle={t.angle!r})."
        )

    def take_angular(t) -> list:
        angs = np.asarray(t.angles, dtype=np.float64).ravel()
        for uid in sorted(by_uid):
            if uid in used:
                continue
            es = by_uid[uid]
            got = sorted(float(e["angle"]) for e in es)
            if (len(es) == angs.size
                    and np.allclose(got, np.sort(angs))
                    and all(e["spectral"] == t.spectral
                            and e["polarization"] == t.polarization
                            and np.asarray(e["wavelengths"]).size == 1
                            and np.allclose(float(e["wavelengths"][0]),
                                            float(t.wavelength))
                            for e in es)):
                used.add(uid)
                return sorted(es, key=lambda d: float(d["angle"]))
        raise RuntimeError(
            "build_merit_spec: no exported entries match an angular target "
            f"(spectral={t.spectral!r}, polarization={t.polarization!r}, "
            f"wavelength={t.wavelength!r})."
        )

    spectral_pairs = [(take_spectral(t), t) for t in spec_targets]
    angular_pairs = [(take_angular(t), t) for t in ang_targets]

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
            weight=float(t.weight),
            count_norm=(None if e["count_norm"] is None
                        else float(e["count_norm"])),
            integral=bool(e["integral"]),
        )

    for e, t in spectral_pairs:
        convert_one(e, t)

    for es, t in angular_pairs:
        for e in es:
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
    # Thin over the native kernel: factors computed in Rust, broadcast
    # multiply stays numpy (arbitrary leading axes are presentation).
    from navette._smatrix import reference_rotation as _ref_rot
    a = np.asarray(cplx)
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    if a.shape[-1] != wl.size:
        raise ValueError(
            f"last axis {a.shape[-1]} != {wl.size} wavelengths."
        )
    rot = np.asarray(_ref_rot(wl, float(angle_deg), float(n_inc),
                              float(total_d), float(passes)))
    return a * rot.reshape((1,) * (a.ndim - 1) + (-1,))


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
    # Thin: lengths + key rules validated natively in set_curve/set_complex.
    angles = np.ascontiguousarray(np.asarray(angles, dtype=np.float64)).ravel()
    wavelengths = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64)).ravel()
    sim = _NativeSimCurves(angles, wavelengths, float(total_d),
                            float(n_front), float(n_back))
    for code, arr in curves.items():
        sim.set_curve(code, np.ascontiguousarray(
            np.asarray(arr, dtype=np.float64)).ravel())
    for code, arr in (complex_curves or {}).items():
        sim.set_complex(code, np.ascontiguousarray(
            np.asarray(arr, dtype=np.complex128)).ravel())
    return sim


# Re-export the needle pipeline driver (submodule import kept last: it
# imports this package's converter, so it must run after the defs above).
from navette.synthesis.pipeline import (  # noqa: E402,F401
    DesignStack,
    LayerSpec,
    LmConfig,
    NeedleCycleConfig,
    NeedlePipeline,
    PipelineConfig,
    SmatrixContext,
    layer_from_material,
    run_needle,
    stack_from_layers,
)

__all__ += [
    "DesignStack",
    "LayerSpec",
    "LmConfig",
    "NeedleCycleConfig",
    "NeedlePipeline",
    "PipelineConfig",
    "SmatrixContext",
    "layer_from_material",
    "run_needle",
    "stack_from_layers",
]
