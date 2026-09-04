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

    def __post_init__(self) -> None:
        _validate_shapes(self.wavelengths, self.values, self.tolerances, label="SpectralTarget")
        _validate_weight(self.weight, label="SpectralTarget")
        _validate_integral(self.integral, self.normalize_count, label="SpectralTarget")
        band = _normalize_band(self.band, self.values.shape, label="SpectralTarget")
        # FFI Guard: Ensure contiguous float64 memory to prevent Rust segfaults.
        # This is an O(1) no-op if the array is already correctly formatted.
        object.__setattr__(self, 'wavelengths', np.ascontiguousarray(self.wavelengths, dtype=np.float64))
        object.__setattr__(self, 'values', np.ascontiguousarray(self.values, dtype=np.float64))
        object.__setattr__(self, 'tolerances', np.ascontiguousarray(self.tolerances, dtype=np.float64))
        object.__setattr__(self, 'band', band)


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

    def __post_init__(self) -> None:
        _validate_shapes(self.angles, self.values, self.tolerances, label="AngularTarget")
        _validate_weight(self.weight, label="AngularTarget")
        _validate_integral(self.integral, self.normalize_count, label="AngularTarget")
        band = _normalize_band(self.band, self.values.shape, label="AngularTarget")
        # FFI Guard
        object.__setattr__(self, 'angles', np.ascontiguousarray(self.angles, dtype=np.float64))
        object.__setattr__(self, 'values', np.ascontiguousarray(self.values, dtype=np.float64))
        object.__setattr__(self, 'tolerances', np.ascontiguousarray(self.tolerances, dtype=np.float64))
        object.__setattr__(self, 'band', band)

BaseTarget = Union[SpectralTarget, AngularTarget]

def _validate_weight(weight: float, label: str = "") -> None:
    """Trust boundary: weights scale merit sums — NaN/negative is rejected."""
    if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 0:
        raise ValueError(f"{label}: weight must be finite and >= 0 (got {weight!r}).")


def _validate_integral(integral: bool, normalize_count: bool, label: str = "") -> None:
    """Integral targets already are means — a count divisor would double-dilute."""
    if integral and normalize_count:
        raise ValueError(
            f"{label}: integral targets reject normalize_count (the mean already is one)."
        )


def _validate_shapes(*arrays: np.ndarray, label: str = "") -> None:
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"{label} shape mismatch: " + ", ".join(str(s) for s in shapes))

def _normalize_band(band: Union[np.ndarray, float, None], shape: tuple, label: str = "") -> Union[np.ndarray, None]:
    """Broadcast a scalar band to `shape`, validate arrays, pass None through."""
    if band is None:
        return None
    if isinstance(band, (int, float)):
        if band < 0:
            raise ValueError(f"{label}: band must be >= 0.")
        return np.ascontiguousarray(np.full(shape, float(band), dtype=np.float64))
    arr = np.ascontiguousarray(np.asarray(band, dtype=np.float64))
    if arr.shape != shape:
        raise ValueError(f"{label} band shape mismatch: {arr.shape} != {shape}.")
    if bool((arr < 0).any()):
        raise ValueError(f"{label}: band must be >= 0.")
    return arr

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

    def add(self, target: BaseTarget) -> None:
        """Append a :class:`SpectralTarget` or :class:`AngularTarget`."""
        if isinstance(target, SpectralTarget):
            self._spectral_targets.append(target)
        elif isinstance(target, AngularTarget):
            self._angular_targets.append(target)
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

    def clear(self) -> None:
        """Remove all spectral and angular targets."""
        self._spectral_targets.clear()
        self._angular_targets.clear()

    @property
    def spectral_targets(self) -> list[SpectralTarget]:
        """The ingested wavelength-domain targets."""
        return self._spectral_targets

    @property
    def angular_targets(self) -> list[AngularTarget]:
        """The ingested angle-domain targets."""
        return self._angular_targets

    @property
    def count(self) -> int:
        """Total number of spectral plus angular targets."""
        return len(self._spectral_targets) + len(self._angular_targets)

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