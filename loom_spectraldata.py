"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Changes from v5:
  SpectralDataFrame
    1.  Native monotonic `uid` (stable identity, serialisable, GC-safe).
    2.  Shape validation on set_data — value length must match wavelength length.
    3.  Sorted-wavelength enforcement on initialisation.
    4.  Dropped MutableMapping — __setitem__ bypassed validation, creating a 
        foot-gun. Replaced with explicit dict-like read accessors and a single
        validated write path (set_data). Direct [] read still works.
    5.  Rich __repr__ for easier debugging.
    6.  Read-only properties for wavelength/bounds (no stale copies).

  OpticalCollection
    7.  Uses frame.uid for identity in _map_frame_to_key (O(1) set lookup 
        instead of O(n) list scan with `in`).
    8.  _length_count replaced by derived property (cannot drift).
    9.  __getitem__ split: [] for optical-key lookup, frame() / frames for 
        index-based access — removes ambiguous Union overload.
    10. keys() returns a KeysView (sized, iterable, supports `in`).

  OpticalWeaver
    11. Generation counter (_gen) — auto-incremented on structural changes,
        used to invalidate stale cache plans that hold frame references.
    12. _build_distribution_plan uses searchsorted intersection instead of 
        np.isin on floats (avoids floating-point equality pitfalls).
    13. get_weaved detects and warns on overlapping frame ranges.
    14. Extracted _resolve_cache_plan helper (DRY for unweave / unweave_collection).
"""

from __future__ import annotations

import itertools
import threading
import warnings
from collections import OrderedDict
from collections.abc import KeysView
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import numpy as np

# ---------------------------------------------------------------------------
# Unit system (mock-safe import)
# ---------------------------------------------------------------------------
try:
    from units import Unit, UnitCategory, UnitConverter
except ImportError:
    from enum import IntFlag, auto

    class UnitCategory(IntFlag):                                  # type: ignore[no-redef]
        SPECTRAL = auto()
        INTENSITY = auto()

    class Unit:                                                   # type: ignore[no-redef]
        NM  = type("Enum", (), {"category": UnitCategory.SPECTRAL,  "name": "NM"})
        RAW = type("Enum", (), {"category": UnitCategory.INTENSITY, "name": "RAW"})

    class UnitConverter:                                          # type: ignore[no-redef]
        @staticmethod
        def bridge(data: Any, u1: Any, u2: Any) -> Any:
            return data


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
OpticalKeyAlias: TypeAlias = tuple[float, str, str]
CollectionResult: TypeAlias = Tuple[List[np.ndarray], List[np.ndarray]]
DistributionPlan: TypeAlias = List[Tuple["SpectralDataFrame", Union[slice, np.ndarray]]]


# =============================================================================
# 1.  SpectralDataFrame
# =============================================================================
class SpectralDataFrame:
    """
    Atomic, thread-safe storage unit: one wavelength grid, many optical keys.

    Write path
    ----------
    All writes go through ``set_data`` which validates shape and grid
    consistency.  There is no ``__setitem__``; this prevents silent
    bypasses of validation.

    Read path
    ---------
    ``frame[key]`` → ``np.ndarray``  (fast, no lock — CPython GIL is
    sufficient for dict reads).

    Identity
    --------
    Every frame gets a process-unique, monotonically increasing ``uid``
    (int).  Use this instead of ``id(frame)`` for metadata keying,
    serialisation, and debugging.
    """

    __slots__ = (
        "uid",
        "_data",
        "_wavelength",
        "_wl_sig",
        "_wl_min",
        "_wl_max",
        "_lock",
    )

    _uid_gen: itertools.count = itertools.count()

    def __init__(self) -> None:
        self.uid: int = next(SpectralDataFrame._uid_gen)
        self._data: dict[OpticalKeyAlias, np.ndarray] = {}
        self._wavelength: Optional[np.ndarray] = None
        self._wl_sig: Optional[bytes] = None
        self._wl_min: float = -float("inf")
        self._wl_max: float = float("inf")
        self._lock = threading.RLock()

    # -- read interface ----------------------------------------------------
    def __getitem__(self, key: OpticalKeyAlias) -> np.ndarray:
        return self._data[key]

    def __contains__(self, key: OpticalKeyAlias) -> bool:          # type: ignore[override]
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[OpticalKeyAlias]:
        return iter(self._data)

    def keys(self) -> KeysView:
        return self._data.keys()

    @property
    def wavelength(self) -> Optional[np.ndarray]:
        """The shared wavelength grid (read-only reference)."""
        return self._wavelength

    @property
    def wl_bounds(self) -> tuple[float, float]:
        """(min, max) of the wavelength grid."""
        return self._wl_min, self._wl_max

    @property
    def wl_signature(self) -> Optional[bytes]:
        """Raw bytes fingerprint of the wavelength grid."""
        return self._wl_sig

    # -- write interface ---------------------------------------------------
    def set_data(
        self,
        key: OpticalKeyAlias,
        value: np.ndarray,
        wavelength: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Validated write.  Returns ``True`` if *key* is new to this frame.

        Raises
        ------
        ValueError
            If wavelength grid conflicts with the existing grid, or if
            *value* length does not match the grid length.
        """
        with self._lock:
            # 1. Wavelength grid initialisation / consistency
            if wavelength is not None:
                self._apply_wavelength(wavelength)

            # 2. Shape guard
            if self._wavelength is not None and value.shape[0] != self._wavelength.shape[0]:
                raise ValueError(
                    f"SpectralDataFrame(uid={self.uid}): value length "
                    f"{value.shape[0]} != wavelength length {self._wavelength.shape[0]}"
                )

            # 3. Store
            is_new = key not in self._data
            self._data[key] = value

        return is_new

    def remove(self, key: OpticalKeyAlias) -> None:
        """Remove a key. Raises KeyError if absent."""
        with self._lock:
            del self._data[key]

    # -- internals ---------------------------------------------------------
    def _apply_wavelength(self, wl: np.ndarray) -> None:
        """Set or verify the wavelength grid. Caller must hold _lock."""
        if self._wavelength is None:
            # First assignment — enforce sorted order
            if wl.size > 1 and not np.all(np.diff(wl) > 0):
                raise ValueError(
                    "SpectralDataFrame: wavelength array must be strictly "
                    "monotonically increasing."
                )
            self._wavelength = wl
            self._wl_sig = wl.tobytes()
            self._wl_min = float(wl[0])
            self._wl_max = float(wl[-1])
        else:
            # Subsequent — bit-exact match required
            if self._wl_sig != wl.tobytes():
                raise ValueError(
                    f"SpectralDataFrame(uid={self.uid}): wavelength grid "
                    "conflict (bit-exact match required)."
                )

    # -- display -----------------------------------------------------------
    def __repr__(self) -> str:
        n_keys = len(self._data)
        wl_len = self._wavelength.shape[0] if self._wavelength is not None else 0
        return (
            f"SpectralDataFrame(uid={self.uid}, keys={n_keys}, "
            f"wl_points={wl_len}, "
            f"range=[{self._wl_min:.2f}, {self._wl_max:.2f}])"
        )


# =============================================================================
# 2.  OpticalCollection
# =============================================================================
class OpticalCollection:
    """
    Mid-level manager: owns frames, handles unit conversion for GUI display.

    Frame identity uses ``frame.uid`` everywhere (O(1) set lookups instead
    of O(n) list scans with ``is``).
    """

    def __init__(self) -> None:
        # Ordered list of all frames
        self._frames: List[SpectralDataFrame] = []

        # wavelength-signature → frame  (deduplication registry)
        self._wl_fingerprints: Dict[bytes, SpectralDataFrame] = {}

        # optical-key → [frames]  (reverse index)
        self._key_map: Dict[OpticalKeyAlias, List[SpectralDataFrame]] = {}

        # uid set per key for O(1) duplicate guard in _map_frame_to_key
        self._key_uid_sets: Dict[OpticalKeyAlias, set[int]] = {}

        # Structural lock
        self._lock = threading.RLock()

        # Display units
        self._display_spectral: Any = Unit.NM
        self._display_intensity: Any = Unit.RAW

    # -- display unit properties -------------------------------------------
    @property
    def display_spectral(self) -> Any:
        return self._display_spectral

    @display_spectral.setter
    def display_spectral(self, unit: Any) -> None:
        if not (unit.category & UnitCategory.SPECTRAL):
            raise ValueError(f"Unit {unit.name} is not a valid Spectral unit.")
        self._display_spectral = unit

    @property
    def display_intensity(self) -> Any:
        return self._display_intensity

    @display_intensity.setter
    def display_intensity(self, unit: Any) -> None:
        if not (unit.category & UnitCategory.INTENSITY):
            raise ValueError(f"Unit {unit.name} is not a valid Intensity unit.")
        self._display_intensity = unit

    # -- sizing / iteration ------------------------------------------------
    @property
    def frame_count(self) -> int:
        """Total number of SpectralDataFrame objects held."""
        return len(self._frames)

    def __len__(self) -> int:
        """Number of distinct optical keys."""
        return len(self._key_map)

    def keys(self) -> KeysView:
        """All registered optical keys (sized, supports ``in``)."""
        return self._key_map.keys()

    def __contains__(self, key: OpticalKeyAlias) -> bool:          # type: ignore[override]
        return key in self._key_map

    # -- frame access (by index) -------------------------------------------
    def frame(self, index: int) -> SpectralDataFrame:
        """Single frame by position."""
        return self._frames[index]

    @property
    def frames(self) -> List[SpectralDataFrame]:
        """All frames (shallow copy for safe iteration)."""
        return list(self._frames)

    def frames_for_key(self, key: OpticalKeyAlias) -> List[SpectralDataFrame]:
        """All frames containing *key*. Raises KeyError if unknown."""
        lst = self._key_map.get(key)
        if lst is None:
            raise KeyError(f"Key {key} not found.")
        return list(lst)

    # -- data access (by optical key, with unit conversion) ----------------
    def get_converted(self, key: OpticalKeyAlias) -> CollectionResult:
        """
        Returns ``(data_list, wl_list)`` with display-unit conversion applied.
        Each list entry corresponds to one frame that holds *key*.
        """
        target_frames = self._key_map.get(key)
        if target_frames is None:
            raise KeyError(f"Key {key} not found.")

        count = len(target_frames)
        data_list: list = [None] * count
        wl_list:   list = [None] * count

        int_unit  = self._display_intensity
        spec_unit = self._display_spectral

        for i, frm in enumerate(target_frames):
            data_list[i] = UnitConverter.bridge(frm[key], Unit.RAW, int_unit)
            wl = frm.wavelength
            if wl is not None:
                wl_list[i] = UnitConverter.bridge(wl, Unit.NM, spec_unit)

        return cast(CollectionResult, (data_list, wl_list))

    # -- data write --------------------------------------------------------
    def set_data(
        self,
        key: OpticalKeyAlias,
        value: Union[np.ndarray, Iterable[Any]],
        wavelength: Union[np.ndarray, Iterable[Any]],
        input_spectral: Any = None,
        input_intensity: Any = None,
    ) -> None:
        """
        Public write path with unit conversion and frame deduplication.
        """
        if wavelength is None:
            raise ValueError("Wavelength is mandatory.")

        input_spectral  = input_spectral  or Unit.NM
        input_intensity = input_intensity or Unit.RAW

        arr    = np.asarray(value, dtype=np.float64)
        wl_arr = np.asarray(wavelength, dtype=np.float64)

        if arr.shape[0] != wl_arr.shape[0]:
            raise ValueError(
                f"Length mismatch: value ({arr.shape[0]}) vs "
                f"wavelength ({wl_arr.shape[0]})."
            )

        # Convert to base units
        base_wl   = UnitConverter.bridge(wl_arr, input_spectral, Unit.NM)
        base_data = UnitConverter.bridge(arr, input_intensity, Unit.RAW)

        target_frame = self._get_or_create_frame(base_wl)
        is_new = target_frame.set_data(key, base_data, wavelength=base_wl)

        if is_new:
            with self._lock:
                self._map_frame_to_key(key, target_frame)

    # -- internal helpers --------------------------------------------------
    def _map_frame_to_key(self, key: OpticalKeyAlias, frame: SpectralDataFrame) -> None:
        """
        Register *frame* against *key*.  Caller must hold ``self._lock``.

        Uses a uid set for O(1) duplicate detection (replaces the O(n) list
        ``in`` check from v5).
        """
        if key not in self._key_map:
            self._key_map[key] = []
            self._key_uid_sets[key] = set()

        if frame.uid not in self._key_uid_sets[key]:
            self._key_map[key].append(frame)
            self._key_uid_sets[key].add(frame.uid)

    def _get_or_create_frame(self, wl_arr: np.ndarray) -> SpectralDataFrame:
        """
        Return existing frame for this wavelength grid, or create a new one.
        Uses double-checked locking for thread safety.
        """
        wl_sig = wl_arr.tobytes()

        # Fast read (safe under CPython GIL for dict lookup)
        existing = self._wl_fingerprints.get(wl_sig)
        if existing is not None:
            return existing

        with self._lock:
            # Re-check after acquiring lock
            existing = self._wl_fingerprints.get(wl_sig)
            if existing is not None:
                return existing

            new_frame = SpectralDataFrame()
            self._frames.append(new_frame)
            self._wl_fingerprints[wl_sig] = new_frame
            return new_frame


# =============================================================================
# 3.  OpticalWeaver
# =============================================================================
class OpticalWeaver(OpticalCollection):
    """
    Simulation-facing interface with LRU-cached distribution plans for
    adaptive grids.

    Generation counter
    ------------------
    ``_gen`` is bumped on every structural change (new frame, new key
    registration).  Cached distribution plans store the generation at
    build time; a mismatch triggers a rebuild.  This prevents stale plans
    from referencing removed or reordered frames.
    """

    def __init__(self, cache_size: int = 128) -> None:
        super().__init__()
        self._distribution_cache: OrderedDict[
            bytes, Tuple[int, DistributionPlan]
        ] = OrderedDict()
        self._max_cache_size = cache_size
        self._cache_lock = threading.RLock()

        # Structural generation counter
        self._gen: int = 0

    # -- override to bump generation on structural changes -----------------
    def _map_frame_to_key(self, key: OpticalKeyAlias, frame: SpectralDataFrame) -> None:
        super()._map_frame_to_key(key, frame)
        self._gen += 1

    def _get_or_create_frame(self, wl_arr: np.ndarray) -> SpectralDataFrame:
        existing_count = len(self._frames)
        frame = super()._get_or_create_frame(wl_arr)
        if len(self._frames) != existing_count:
            # A new frame was created
            self._gen += 1
        return frame

    # -- weave (read: stitch fragments into continuous curves) -------------
    def get_weaved(self, key: OpticalKeyAlias) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns ``(wavelengths, values)`` — all fragments for *key* sorted
        and concatenated by wavelength.

        Warns if frame ranges overlap (which would produce a non-monotonic
        or duplicated grid after concatenation).
        """
        target_frames = self._key_map.get(key)
        if not target_frames:
            raise KeyError(f"Key {key} not found.")

        # Collect (min_wl, wl_array, data_array), skip frames without grid
        fragments: list[tuple[float, np.ndarray, np.ndarray]] = []
        for frm in target_frames:
            wl = frm.wavelength
            if wl is not None:
                fragments.append((frm.wl_bounds[0], wl, frm[key]))

        if not fragments:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        fragments.sort(key=lambda t: t[0])

        # Overlap detection
        for i in range(len(fragments) - 1):
            cur_max = fragments[i][1][-1]
            nxt_min = fragments[i + 1][0]
            if cur_max > nxt_min:
                warnings.warn(
                    f"get_weaved(key={key}): frames overlap at "
                    f"{cur_max:.4f} > {nxt_min:.4f}. Concatenated grid "
                    "may contain duplicates or be non-monotonic.",
                    stacklevel=2,
                )

        _, sorted_wls, sorted_data = zip(*fragments)
        return np.concatenate(sorted_wls), np.concatenate(sorted_data)

    def get_weaved_collections(
        self,
    ) -> List[Tuple[np.ndarray, Dict[OpticalKeyAlias, np.ndarray]]]:
        """
        Groups all stored data by shared combined wavelength geometry.

        Returns a list of ``(wavelength_array, {key: data_array})`` tuples,
        one per unique geometry.
        """
        groups: Dict[bytes, Tuple[np.ndarray, Dict[OpticalKeyAlias, np.ndarray]]] = {}

        for key in self.keys():
            try:
                wl, data = self.get_weaved(key)
            except KeyError:
                continue
            if wl.size == 0:
                continue

            sig = wl.tobytes()
            if sig not in groups:
                groups[sig] = (wl, {key: data})
            else:
                groups[sig][1][key] = data

        return list(groups.values())

    # -- unweave (write: distribute a full curve back into frames) ---------
    def unweave(
        self,
        key: OpticalKeyAlias,
        full_wavelength: np.ndarray,
        full_data: np.ndarray,
    ) -> int:
        """
        Distribute *full_data* on *full_wavelength* back into the
        constituent frames.  Returns the number of frames updated.
        """
        plan = self._resolve_plan(full_wavelength)
        updated = 0

        for frm, indices in plan:
            subset = full_data[indices]
            is_new = frm.set_data(key, subset)
            if is_new:
                with self._lock:
                    self._map_frame_to_key(key, frm)
            updated += 1

        return updated

    def unweave_collection(
        self,
        common_wavelength: np.ndarray,
        data_batch: Dict[OpticalKeyAlias, np.ndarray],
    ) -> int:
        """Batch-optimised unweave — one plan lookup for many keys."""
        if not data_batch:
            return 0

        plan = self._resolve_plan(common_wavelength)
        updated = 0

        for key, full_data in data_batch.items():
            for frm, indices in plan:
                subset = full_data[indices]
                is_new = frm.set_data(key, subset)
                if is_new:
                    with self._lock:
                        self._map_frame_to_key(key, frm)
                updated += 1

        return updated

    # -- cache management --------------------------------------------------
    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._distribution_cache.clear()

    # -- internals ---------------------------------------------------------
    def _resolve_plan(self, full_wavelength: np.ndarray) -> DistributionPlan:
        """
        Retrieve or build a distribution plan.  Handles LRU promotion,
        generation-based staleness detection, and eviction.
        """
        sig = full_wavelength.tobytes()

        with self._cache_lock:
            entry = self._distribution_cache.get(sig)
            if entry is not None:
                gen, plan = entry
                if gen == self._gen:
                    # Cache hit — promote to MRU
                    self._distribution_cache.move_to_end(sig)
                    return plan
                # Stale — drop it, rebuild below
                del self._distribution_cache[sig]

        # Cache miss or stale — build fresh plan
        plan = self._build_distribution_plan(full_wavelength)

        with self._cache_lock:
            if len(self._distribution_cache) >= self._max_cache_size:
                self._distribution_cache.popitem(last=False)  # evict LRU
            self._distribution_cache[sig] = (self._gen, plan)

        return plan

    def _build_distribution_plan(
        self, full_wavelength: np.ndarray
    ) -> DistributionPlan:
        """
        For each existing frame whose wavelength grid is a subset of
        *full_wavelength*, compute the index mapping (slice or array)
        needed to extract that subset.

        Uses searchsorted-based intersection instead of ``np.isin`` to
        avoid floating-point equality pitfalls on large grids.
        """
        plan: DistributionPlan = []

        with self._lock:
            frames_snapshot = list(self._frames)

        if full_wavelength.size == 0:
            return plan

        fw_min = full_wavelength[0]
        fw_max = full_wavelength[-1]

        for frm in frames_snapshot:
            frame_wl = frm.wavelength
            if frame_wl is None or frame_wl.size == 0:
                continue

            f_min, f_max = frm.wl_bounds

            # Fast disjoint rejection
            if f_min > fw_max or f_max < fw_min:
                continue

            # Narrow search window via searchsorted
            idx_lo = np.searchsorted(full_wavelength, f_min, side="left")
            idx_hi = np.searchsorted(full_wavelength, f_max, side="right")

            candidate = full_wavelength[idx_lo:idx_hi]

            # Try fast path: contiguous slice with exact match
            if candidate.size == frame_wl.size and np.array_equal(candidate, frame_wl):
                plan.append((frm, slice(idx_lo, idx_hi)))
                continue

            # Slow path: searchsorted intersection (tolerant to float layout
            # differences, unlike np.isin which uses hash-based equality)
            insert_pos = np.searchsorted(full_wavelength, frame_wl)

            # Clamp to valid range
            insert_pos = np.clip(insert_pos, 0, full_wavelength.size - 1)

            # Accept only positions where the value actually matches
            matched = np.abs(full_wavelength[insert_pos] - frame_wl) < 1e-12
            indices = insert_pos[matched]

            if indices.size > 0:
                # Check if indices form a contiguous range → use slice
                if indices.size > 1 and np.all(np.diff(indices) == 1):
                    plan.append((frm, slice(int(indices[0]), int(indices[-1]) + 1)))
                else:
                    plan.append((frm, indices))

        return plan