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
from navette.spectralweave import TargetWeaver
from navette.spectralweave import OpticalWeaver
from navette.spectralweave import calculate_merit as _calculate_merit

TargetType = Literal["a", "b", "e"]
NormalizationMode = Literal["auto", "linear", "log", "phase", "complex"]

# ---------------------------------------------------------------------------
# 1. Target data classes (user-facing inputs)
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class SpectralTarget:
    """One constraint curve: value vs wavelength at fixed angle."""
    wavelengths:  np.ndarray
    values:       np.ndarray
    tolerances:   np.ndarray
    angle:        float
    polarization: str
    spectral:     str
    kind:         TargetType = "e"
    normalization_mode: NormalizationMode = "auto"

    def __post_init__(self) -> None:
        _validate_shapes(self.wavelengths, self.values, self.tolerances, label="SpectralTarget")
        
        # FFI Guard: Ensure contiguous float64 memory to prevent Rust segfaults.
        # This is an O(1) no-op if the array is already correctly formatted.
        object.__setattr__(self, 'wavelengths', np.ascontiguousarray(self.wavelengths, dtype=np.float64))
        object.__setattr__(self, 'values', np.ascontiguousarray(self.values, dtype=np.float64))
        object.__setattr__(self, 'tolerances', np.ascontiguousarray(self.tolerances, dtype=np.float64))


@dataclass(slots=True, frozen=True)
class AngularTarget:
    """One constraint curve: value vs angle at fixed wavelength."""
    wavelength:   float
    angles:       np.ndarray
    values:       np.ndarray
    tolerances:   np.ndarray
    polarization: str
    spectral:     str
    kind:         TargetType = "e"
    normalization_mode: NormalizationMode = "auto"

    def __post_init__(self) -> None:
        _validate_shapes(self.angles, self.values, self.tolerances, label="AngularTarget")
        
        # FFI Guard
        object.__setattr__(self, 'angles', np.ascontiguousarray(self.angles, dtype=np.float64))
        object.__setattr__(self, 'values', np.ascontiguousarray(self.values, dtype=np.float64))
        object.__setattr__(self, 'tolerances', np.ascontiguousarray(self.tolerances, dtype=np.float64))

BaseTarget = Union[SpectralTarget, AngularTarget]

def _validate_shapes(*arrays: np.ndarray, label: str = "") -> None:
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"{label} shape mismatch: " + ", ".join(str(s) for s in shapes))

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
        if isinstance(target, SpectralTarget):
            self._spectral_targets.append(target)
        elif isinstance(target, AngularTarget):
            self._angular_targets.append(target)
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

    def clear(self) -> None:
        self._spectral_targets.clear()
        self._angular_targets.clear()

    @property
    def spectral_targets(self) -> list[SpectralTarget]:
        return self._spectral_targets

    @property
    def angular_targets(self) -> list[AngularTarget]:
        return self._angular_targets

    @property
    def count(self) -> int:
        return len(self._spectral_targets) + len(self._angular_targets)

    def build_weaver(self, cache_size: int = 128, tolerance_floor: float = 1e-12) -> TargetWeaver:
        """
        Compiles the defined targets into a Rust-native TargetWeaver.
        """
        weaver = TargetWeaver(cache_size=cache_size, tolerance_floor=tolerance_floor)
        
        # Iterating in Python and passing to Rust here is fine because it 
        # only happens ONCE during initialization, not inside the hot loop.
        for t in self._spectral_targets:
            weaver.add_spectral_target(
                t.wavelengths, t.values, t.tolerances,
                t.angle, t.polarization, t.spectral, t.kind, t.normalization_mode
            )
            
        for t in self._angular_targets:
            weaver.add_angular_target(
                t.wavelength, t.angles, t.values, t.tolerances,
                t.polarization, t.spectral, t.kind, t.normalization_mode
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