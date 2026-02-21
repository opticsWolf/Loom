"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Changes from v5:
  * Imports from opticaldatastructure_v6 (native uid, no monkey-patch needed).
  * Uses frame.uid directly everywhere.
  * Uses frames_for_key() public API in iter_target_frames fallback.
  * Minor cleanup: removed _frame_uid helper, removed itertools/warnings imports
    that were only needed for the patch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import (
    Dict, Iterator, List, Literal, Optional, Tuple, TypeAlias, Union,
)

import numpy as np

from loom_spectraldata import (
    OpticalCollection,
    OpticalWeaver,
    SpectralDataFrame,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
OpticalKeyAlias: TypeAlias = tuple[float, str, str]
TargetType = Literal["a", "b", "e"]


# ---------------------------------------------------------------------------
# 1.  Target data classes (user-facing inputs)
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
# 2.  TargetCollection (lightweight, standalone)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TargetCollection:
    """
    User-facing container for mixed Spectral and Angular targets.
    Not an OpticalCollection — it holds definitions, not frames.
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

    def build_weaver(self, **weaver_kw) -> TargetWeaver:
        weaver = TargetWeaver(**weaver_kw)
        weaver.consume_collection(self)
        return weaver


# ---------------------------------------------------------------------------
# 3.  Per-key metadata
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TargetEntry:
    """Single constraint for one OpticalKey inside one frame."""
    kind:       TargetType
    tolerances: np.ndarray


@dataclass(slots=True)
class TargetMetadata:
    """Parallel metadata for one SpectralDataFrame, keyed by frame uid."""
    entries: Dict[OpticalKeyAlias, TargetEntry] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4.  TargetWeaver
# ---------------------------------------------------------------------------
class TargetWeaver(OpticalWeaver):
    """
    Optimization-engine-facing weaver for target constraints.

    * Each target gets a dedicated frame (no overwrites).
    * Metadata per (frame.uid, key) supports mixed constraint kinds.
    * Public iter_target_frames() for encapsulated access.
    """

    def __init__(self, cache_size: int = 128,
                 tolerance_floor: float = 1e-12) -> None:
        super().__init__(cache_size=cache_size)
        self._target_metadata: Dict[int, TargetMetadata] = {}  # uid → meta
        self._meta_lock = threading.RLock()
        self.tolerance_floor: float = tolerance_floor

    # -- ingestion ---------------------------------------------------------
    def consume_collection(self, collection: TargetCollection) -> None:
        for sp in collection.spectral_targets:
            self._ingest_spectral(sp)
        for ang in collection.angular_targets:
            self._ingest_angular(ang)

    def _ingest_spectral(self, t: SpectralTarget) -> None:
        frame = self._create_dedicated_frame(t.wavelengths)
        key: OpticalKeyAlias = (t.angle, t.polarization, t.spectral)

        frame.set_data(key, t.values, wavelength=t.wavelengths)
        with self._lock:
            self._map_frame_to_key(key, frame)

        self._register_metadata(frame, key, t.tolerances, t.kind)

    def _ingest_angular(self, t: AngularTarget) -> None:
        wl_point = np.array([t.wavelength], dtype=np.float64)
        frame = self._create_dedicated_frame(wl_point)

        for i, angle in enumerate(t.angles):
            key: OpticalKeyAlias = (float(angle), t.polarization, t.spectral)
            val_arr = np.array([t.values[i]], dtype=np.float64)
            tol_arr = np.array([t.tolerances[i]], dtype=np.float64)

            is_new = frame.set_data(key, val_arr, wavelength=wl_point)
            if is_new:
                with self._lock:
                    self._map_frame_to_key(key, frame)

            self._register_metadata(frame, key, tol_arr, t.kind)

    # -- frame creation ----------------------------------------------------
    def _create_dedicated_frame(self, wl: np.ndarray) -> SpectralDataFrame:
        """Always creates a new frame (targets are independent constraints)."""
        new_frame = SpectralDataFrame()
        with self._lock:
            self._frames.append(new_frame)
            self._gen += 1
        return new_frame

    # -- metadata ----------------------------------------------------------
    def _register_metadata(self, frame: SpectralDataFrame,
                           key: OpticalKeyAlias,
                           tolerances: np.ndarray,
                           kind: TargetType) -> None:
        with self._meta_lock:
            if frame.uid not in self._target_metadata:
                self._target_metadata[frame.uid] = TargetMetadata()
            self._target_metadata[frame.uid].entries[key] = TargetEntry(
                kind=kind,
                tolerances=np.maximum(tolerances, self.tolerance_floor),
            )

    def get_metadata(self, frame: SpectralDataFrame) -> Optional[TargetMetadata]:
        return self._target_metadata.get(frame.uid)

    # -- public iteration --------------------------------------------------
    def iter_target_frames(
        self, key: OpticalKeyAlias
    ) -> Iterator[Tuple[SpectralDataFrame, TargetEntry]]:
        """Yields (frame, entry) for every constraint on *key*."""
        for frm in self._key_map.get(key, []):
            meta = self._target_metadata.get(frm.uid)
            if meta is not None and key in meta.entries:
                yield frm, meta.entries[key]

    def target_keys(self) -> Iterator[OpticalKeyAlias]:
        return iter(self._key_map)


# ---------------------------------------------------------------------------
# 5.  Merit function
# ---------------------------------------------------------------------------
def calculate_merit(
    sim_weaver: OpticalWeaver,
    target_weaver: TargetWeaver,
    *,
    missing_penalty: float = 1e6,
) -> float:
    """
    Weighted Sum of Squared Errors.

        Σ ((sim − target) / tolerance)²

    Constraint types per (frame, key):
        'e'  exact  — penalise any deviation
        'a'  above  — penalise only sim < target
        'b'  below  — penalise only sim > target
    """
    total: float = 0.0

    for key in target_weaver.target_keys():

        try:
            sim_wl, sim_val = sim_weaver.get_weaved(key)
        except KeyError:
            total += missing_penalty
            continue

        if sim_wl.size == 0:
            total += missing_penalty
            continue

        for frame, entry in target_weaver.iter_target_frames(key):
            target_wl = frame.wavelength
            if target_wl is None or target_wl.size == 0:
                continue

            target_val = frame[key]

            # Range overlap check
            if sim_wl[-1] < target_wl[0] or sim_wl[0] > target_wl[-1]:
                continue

            # Slice simulation to relevant neighbourhood
            idx_lo = max(0, np.searchsorted(sim_wl, target_wl[0], side="left") - 1)
            idx_hi = min(len(sim_wl),
                         np.searchsorted(sim_wl, target_wl[-1], side="right") + 1)
            sub_wl  = sim_wl[idx_lo:idx_hi]
            sub_val = sim_val[idx_lo:idx_hi]

            if sub_wl.size < 2:
                continue

            sim_at_target = np.interp(target_wl, sub_wl, sub_val)
            total += _compute_residual(sim_at_target, target_val,
                                       entry.tolerances, entry.kind)

    return total


def _compute_residual(
    sim: np.ndarray,
    target: np.ndarray,
    tol: np.ndarray,
    kind: TargetType,
) -> float:
    """Vectorised residual for one (frame, key) pair."""
    diff = sim - target

    if kind == "e":
        return float(np.sum(np.square(diff / tol)))

    mask = diff < 0.0 if kind == "a" else diff > 0.0

    if not np.any(mask):
        return 0.0

    return float(np.sum(np.square(diff[mask] / tol[mask])))