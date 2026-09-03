# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

Thin Python Wrapper over `spectralweave` Rust extension.
Maintains the beautiful @dataclass user API while deferring all
heavy lifting, normalisation, and merit calculation to compiled Rust.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Union

import numpy as np

# Import the ultra-fast Rust implementation
from spectralweave import TargetWeaver as RustTargetWeaver
from spectralweave import OpticalWeaver
from spectralweave import calculate_merit as _rust_calculate_merit

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
    normalization_mode: NormalizationMode = "auto"  # Mathematical residual metric

    def __post_init__(self) -> None:
        _validate_shapes(self.wavelengths, self.values, self.tolerances,
                         label="SpectralTarget")

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
    normalization_mode: NormalizationMode = "auto"  # Mathematical residual metric

    def __post_init__(self) -> None:
        _validate_shapes(self.angles, self.values, self.tolerances,
                         label="AngularTarget")

BaseTarget = Union[SpectralTarget, AngularTarget]

def _validate_shapes(*arrays: np.ndarray, label: str = "") -> None:
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"{label} shape mismatch: " + ", ".join(str(s) for s in shapes)
        )

# ---------------------------------------------------------------------------
# 2. TargetCollection (lightweight, standalone)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TargetCollection:
    """
    User-facing container for mixed Spectral and Angular targets.
    """
    _spectral_targets: List[SpectralTarget] = field(default_factory=list)
    _angular_targets:  List[AngularTarget]  = field(default_factory=list)

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
    def spectral_targets(self) -> List[SpectralTarget]:
        return self._spectral_targets

    @property
    def angular_targets(self) -> List[AngularTarget]:
        return self._angular_targets

    @property
    def count(self) -> int:
        return len(self._spectral_targets) + len(self._angular_targets)

    def build_weaver(self, cache_size: int = 128, tolerance_floor: float = 1e-12) -> RustTargetWeaver:
        """
        Compiles the defined targets into a Rust-native TargetWeaver.
        """
        weaver = RustTargetWeaver(cache_size=cache_size, tolerance_floor=tolerance_floor)
        
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
    sim_weaver: OpticalWeaver,
    target_weaver: RustTargetWeaver,
    *,
    missing_penalty: float = 1e6,
) -> float:
    """
    Delegates the Merit Function computation completely to compiled Rust,
    bypassing the Python GIL and numpy interpolator overhead entirely.
    """
    return _rust_calculate_merit(sim_weaver, target_weaver, missing_penalty)