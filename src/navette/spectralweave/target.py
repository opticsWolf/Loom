# -*- coding: utf-8 -*-
"""
Navette: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

Optimized Python Wrapper over `spectralweave` Rust extension for Targets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np

# Import the ultra-fast Rust implementation
try:
  from navette._spectralweave import TargetWeaver
  from navette._spectralweave import OpticalWeaver
  from navette._spectralweave import calculate_merit as _calculate_merit
except ImportError as exc:  # pragma: no cover - native not built yet
  raise ImportError(
    "Targets need the compiled `navette._spectralweave` extension. "
    "Build it with: maturin develop  # from the repo root"
  ) from exc

TargetType = Literal["a", "b", "e", "r", "c"]
NormalizationMode = Literal["auto", "linear", "log", "phase", "complex"]

# ---------------------------------------------------------------------------
# 1. Target data classes (user-facing inputs)
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class SpectralTarget:
    """One constraint curve: value vs wavelength at fixed angle.

    Kinds ``e``/``a``/``b`` are exact/above/below; ``r`` is a hard range box
    (zero merit inside ±``band``, quadratic exceedance outside) and ``c`` a
    soft center box (reduced ``(d/band)^2`` inside, exceedance + 1 outside).
    ``band`` is a per-point half-width in raw units (scalar broadcasts);
    omit it and ``r`` falls back to ±``tolerances``, ``c`` to ``e``.
    Tolerances are fractions of the curve level (linear/log), absolute
    radians (phase) — see ``docs/spectralweave-target-kinds.md``.
    ``weight`` multiplies the target's merit sum (default 1; relative
    importance across targets — the 200-pt vs 10-pt density question).
    ``normalize_count=True`` additionally divides by the target-level point
    count (spectral grid size, angular angle count), turning the sum into a
    mean: equal say regardless of sampling density. Both flow verbatim into
    ``MeritSpec`` and the needle fold (missing-data penalties unaffected —
    drop a target to silence those, weight 0 only mutes present data).
    ``integral=True`` instead constrains the MEAN of the scaled diffs
    (single residual ``mean(d)/mean(tol)`` — kinds apply once to the mean,
    e.g. integral-``a`` is a lower bound on the average); it rejects
    ``normalize_count`` (the mean already is one). Regular and integral
    targets mix freely in one collection/run.
    Note for needle optimization: the folded needle evaluation drops the
    ``c`` +1 outside-level (constant offset — gradients exact, values read
    lower by the violated-point count); see
    ``docs/spectralweave-target-kinds.md``.
    """
    wavelengths:  np.ndarray
    values:       np.ndarray
    tolerances:   np.ndarray
    angle:        float
    polarization: str
    spectral:     str
    kind:         TargetType = "e"
    normalization_mode: NormalizationMode = "auto"
    band:         Union[np.ndarray, float, None] = None
    phase:        bool = False
    weight:       float = 1.0
    normalize_count: bool = False
    integral:     bool = False

    def _dump(self) -> dict:
        band = self.band
        if isinstance(band, (int, float)):
            band = float(band)
        elif band is not None:
            band = np.ascontiguousarray(np.asarray(band, dtype=np.float64)).ravel().tolist()
        return {
            "wavelengths": np.ascontiguousarray(np.asarray(self.wavelengths, dtype=np.float64)).ravel().tolist(),
            "values": np.ascontiguousarray(np.asarray(self.values, dtype=np.float64)).ravel().tolist(),
            "tolerances": np.ascontiguousarray(np.asarray(self.tolerances, dtype=np.float64)).ravel().tolist(),
            "angle": float(self.angle), "polarization": str(self.polarization),
            "spectral": str(self.spectral), "kind": str(self.kind),
            "normalization_mode": str(self.normalization_mode),
            "band": band, "phase": bool(self.phase),
            "weight": float(self.weight) if isinstance(self.weight, (int, float)) else self.weight,
            "normalize_count": bool(self.normalize_count),
            "integral": bool(self.integral),
        }

    def __post_init__(self) -> None:
        # Shaping (broadcast/contiguity) stays; all CHECKS run natively,
        # once, via validate_targets (single validator, no duplication).
        from navette._structure import validate_targets as _validate
        import json as _json
        band = self.band
        if isinstance(band, (int, float)):
            band = np.ascontiguousarray(np.full(np.shape(self.values), float(band), dtype=np.float64))
        elif band is not None:
            band = np.ascontiguousarray(np.asarray(band, dtype=np.float64))
        # FFI Guard: Ensure contiguous float64 memory to prevent Rust segfaults.
        # This is an O(1) no-op if the array is already correctly formatted.
        object.__setattr__(self, 'wavelengths', np.ascontiguousarray(self.wavelengths, dtype=np.float64))
        object.__setattr__(self, 'values', np.ascontiguousarray(self.values, dtype=np.float64))
        object.__setattr__(self, 'tolerances', np.ascontiguousarray(self.tolerances, dtype=np.float64))
        object.__setattr__(self, 'band', band)
        _validate(_json.dumps({"spectral": [self._dump()], "angular": []}))


@dataclass(slots=True, frozen=True)
class AngularTarget:
    """One constraint curve: value vs angle at fixed wavelength.

    Same ``kind``/``band`` semantics as :class:`SpectralTarget`.
    """
    wavelength:   float
    angles:       np.ndarray
    values:       np.ndarray
    tolerances:   np.ndarray
    polarization: str
    spectral:     str
    kind:         TargetType = "e"
    normalization_mode: NormalizationMode = "auto"
    band:         Union[np.ndarray, float, None] = None
    phase:        bool = False
    weight:       float = 1.0
    normalize_count: bool = False
    integral:     bool = False

    def _dump(self) -> dict:
        band = self.band
        if isinstance(band, (int, float)):
            band = float(band)
        elif band is not None:
            band = np.ascontiguousarray(np.asarray(band, dtype=np.float64)).ravel().tolist()
        return {
            "wavelength": float(self.wavelength),
            "angles": np.ascontiguousarray(np.asarray(self.angles, dtype=np.float64)).ravel().tolist(),
            "values": np.ascontiguousarray(np.asarray(self.values, dtype=np.float64)).ravel().tolist(),
            "tolerances": np.ascontiguousarray(np.asarray(self.tolerances, dtype=np.float64)).ravel().tolist(),
            "polarization": str(self.polarization),
            "spectral": str(self.spectral), "kind": str(self.kind),
            "normalization_mode": str(self.normalization_mode),
            "band": band, "phase": bool(self.phase),
            "weight": float(self.weight) if isinstance(self.weight, (int, float)) else self.weight,
            "normalize_count": bool(self.normalize_count),
            "integral": bool(self.integral),
        }

    def __post_init__(self) -> None:
        from navette._structure import validate_targets as _validate
        import json as _json
        band = self.band
        if isinstance(band, (int, float)):
            band = np.ascontiguousarray(np.full(np.shape(self.values), float(band), dtype=np.float64))
        elif band is not None:
            band = np.ascontiguousarray(np.asarray(band, dtype=np.float64))
        # FFI Guard
        object.__setattr__(self, 'angles', np.ascontiguousarray(self.angles, dtype=np.float64))
        object.__setattr__(self, 'values', np.ascontiguousarray(self.values, dtype=np.float64))
        object.__setattr__(self, 'tolerances', np.ascontiguousarray(self.tolerances, dtype=np.float64))
        object.__setattr__(self, 'band', band)
        _validate(_json.dumps({"spectral": [], "angular": [self._dump()]}))

ColorQuantity = Literal["Lab", "XyY", "LCh", "Oklab", "Y"]
ColorDistance = Literal["DeltaE2000", "DeltaE76", "Channels"]


@dataclass(slots=True, frozen=True)
class ColorTarget:
    """One colorimetric demand: front R/T intensity curve at fixed angle,
    integrated against an illuminant x observer pair.

    Defaults are D65 + 1931 2 deg; every field is overridable. ``reference``
    is a triple (e.g. Lab ``(62, 18, -34)``) — or a scalar for quantity
    ``"Y"`` (P2); mismatched shapes are refused natively. ``illuminant`` /
    ``observer`` are ``"D65"`` / ``"1931_2deg"`` or explicit table dicts
    (``{"wavelengths", "values"}`` / ``{"wavelengths", "xyz"}``);
    unknown names are refused natively (no registry to drift).
    Lab white point is always the demand illuminant's own white.
    Limits (all refused natively with named messages): front R/T curves
    only (no back, no absorption/phase curves), kind ``"Exact"`` only,
    ``"linear"`` only; quantities Lab|XyY|LCh|Oklab|Y — DeltaE2000|DeltaE76
    in Lab/LCh only (XyY and Oklab take Channels; Y is scalar Channels).
    Oklab references are D65-Oklab (non-D65 demands Bradford-adapt first;
    in-tree D65 white carries a pre-existing ~1e-4 b offset — systematic,
    far below demand relevance). LCh hue wraps before scaling; near-neutral
    hues are gradient-unstable by nature (weight hue tolerance generously
    near the achromatic axis). No ``integral`` /
    ``count_norm`` / ``phase`` / ``band`` — a color demand already is
    integral. ``weight`` scales the demand's merit sum (default 1).
    Needle note: ``Ru``/``Tu`` demands split half per polarization branch
    (Ru is the Rs/Rp mean); s/p demands ride their shared bucket, same
    convention as pointwise targets — see
    ``docs/plans/color_targets_optionB_plan.md`` D5.
    All checks run natively, once, via validate_targets (single validator,
    no duplication) — shaping here is normalization only.
    """
    curve:        str = "Ru"
    angle:        float = 0.0
    illuminant:   Union[str, dict] = "D65"
    observer:     Union[str, dict] = "1931_2deg"
    quantity:     str = "Lab"
    reference:    Union[tuple, list, float] = (62.0, 18.0, -34.0)
    distance:     str = "DeltaE2000"
    weight:       float = 1.0
    kind:         str = "Exact"
    transform:    str = "linear"

    def _dump(self) -> dict:
        return {
            "curve": str(self.curve), "angle": float(self.angle),
            "illuminant": _dump_table(self.illuminant, "illuminant"),
            "observer": _dump_table(self.observer, "observer"),
            "quantity": str(self.quantity),
            "reference": _dump_reference(self.reference),
            "distance": str(self.distance),
            "weight": float(self.weight) if isinstance(self.weight, (int, float)) else self.weight,
            "kind": str(self.kind), "transform": str(self.transform),
        }

    def __post_init__(self) -> None:
        from navette._structure import validate_targets as _validate
        import json as _json
        _validate(_json.dumps({"spectral": [], "angular": [], "color": [self._dump()]}))


def _dump_table(spec: Union[str, dict], role: str):
    """Name passthrough (native resolves ``D65``/``1931_2deg`` from the
    embedded defaults) or explicit table dict with array normalization.
    No rule checks here — the native validator owns them."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        out = {}
        for key, value in spec.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).ravel().tolist()
            out[key] = value
        return out
    return spec


def _dump_reference(ref: Union[tuple, list, float]):
    """Triple -> [f, f, f], scalar -> float; anything else passes through
    for the native validator to refuse with a named message."""
    if isinstance(ref, bool):
        return ref
    if isinstance(ref, (int, float)):
        return float(ref)
    if isinstance(ref, (tuple, list)):
        return [float(v) for v in ref]
    return ref


BaseTarget = Union[SpectralTarget, AngularTarget, ColorTarget]

# ---------------------------------------------------------------------------
# 2. TargetCollection (lightweight, standalone)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TargetCollection:
    """
    User-facing container for mixed Spectral and Angular targets.
    """
    _spectral_targets: list[SpectralTarget] = field(default_factory=list)
    _angular_targets:  list[AngularTarget]  = field(default_factory=list)
    _color_targets:    list[ColorTarget]    = field(default_factory=list)

    def add(self, target: BaseTarget) -> None:
        """Append a :class:`SpectralTarget`, :class:`AngularTarget`, or
        :class:`ColorTarget`."""
        if isinstance(target, SpectralTarget):
            self._spectral_targets.append(target)
        elif isinstance(target, AngularTarget):
            self._angular_targets.append(target)
        elif isinstance(target, ColorTarget):
            self._color_targets.append(target)
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

    def clear(self) -> None:
        """Remove all spectral, angular, and color targets."""
        self._spectral_targets.clear()
        self._angular_targets.clear()
        self._color_targets.clear()

    @property
    def spectral_targets(self) -> list[SpectralTarget]:
        """The ingested wavelength-domain targets."""
        return self._spectral_targets

    @property
    def angular_targets(self) -> list[AngularTarget]:
        """The ingested angle-domain targets."""
        return self._angular_targets

    @property
    def color_targets(self) -> list[ColorTarget]:
        """The ingested colorimetric demands."""
        return self._color_targets

    @property
    def count(self) -> int:
        """Total number of spectral plus angular plus color targets."""
        return len(self._spectral_targets) + len(self._angular_targets) + len(self._color_targets)

    def build_weaver(self, cache_size: int = 128, tolerance_floor: float = 1e-12) -> TargetWeaver:
        """
        Compiles the defined targets into a Rust-native TargetWeaver.
        """
        weaver = TargetWeaver(cache_size=cache_size, tolerance_floor=tolerance_floor)

        # Iterating in Python and passing to Rust here is fine because it
        # only happens ONCE during initialization, not inside the hot loop.
        # PDts/PDtp force phase normalization (raw radians, nf == 1): they
        # are differential-phase quantities and any other resolution would
        # not unscale back to radians in the synthesis converter.
        for t in self._spectral_targets:
            weaver.add_spectral_target(
                t.wavelengths, t.values, t.tolerances,
                t.angle, t.polarization, t.spectral, t.kind,
                "phase" if t.spectral in ("PDts", "PDtp") else t.normalization_mode,
                t.band,
                weight=t.weight, normalize_count=t.normalize_count,
                integral=t.integral,
            )

        for t in self._angular_targets:
            weaver.add_angular_target(
                t.wavelength, t.angles, t.values, t.tolerances,
                t.polarization, t.spectral, t.kind,
                "phase" if t.spectral in ("PDts", "PDtp") else t.normalization_mode,
                t.band,
                weight=t.weight, normalize_count=t.normalize_count,
                integral=t.integral,
            )

        return weaver


# ---------------------------------------------------------------------------
# 3. Merit Function
# ---------------------------------------------------------------------------
def calculate_merit(
    sim_weaver: OpticalWeaver,  # Will accept the unwrapped Rust object
    target_weaver: TargetWeaver,
    *,
    missing_penalty: float = 1e6,
) -> float:
    """
    Delegates the Merit Function computation completely to compiled Rust,
    bypassing the Python GIL entirely for true concurrency.
    """
    return _calculate_merit(sim_weaver, target_weaver, missing_penalty)