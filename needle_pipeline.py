# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: needle_pipeline.py — Continuous iterative needle synthesis pipeline.

Wraps the single-pass NeedleSynthesizer in a macro-loop that repeatedly
executes needle insertion cycles, optional cleanup, and optional QWOT-based
inflation until a design budget (layer count or total thickness) is reached
or the algorithm detects stagnation.

Key additions over needle_synthesis.py:

    ClampedNeedleSynthesizer
        Subclass with min/max per-layer thickness enforcement baked into
        every optimisation, cleanup, and inflate step.  The optimizer's
        ``bounds`` parameter and a post-optimisation ``clamp_all_layers()``
        sweep guarantee no film layer ever leaves the permitted range.

    StagnationDetector
        Sliding-window analysis of the merit-function trajectory.  Computes
        a normalised improvement gradient and detects three failure modes:
        plateau (gradient ≈ 0), oscillation (sign-alternating deltas over
        consecutive macro-cycles), and divergence (sustained MF increase).

    PipelineConfig / PipelinePhaseResult / PipelineResult
        Data classes for the full pipeline parameterisation and its output.

    NeedlePipeline
        The orchestrator.  Each macro-cycle is:
            1. needle pass  (several insertion cycles + local opt)
            2. cleanup      (merge → prune → re-opt)       [optional]
            3. inflate      (QWOT addon → re-opt)           [optional]
            4. stagnation check
        The loop terminates when any of these fires:
            - layer count ≥ target
            - total thickness ≥ target
            - stagnation detected
            - hard iteration cap reached

Usage:

    >>> from needle_pipeline import (
    ...     NeedlePipeline, PipelineConfig, ClampedNeedleSynthesizer,
    ... )
    >>> pipeline = NeedlePipeline.from_synthesizer(synth, PipelineConfig(...))
    >>> result = pipeline.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
from scipy.optimize import least_squares

from loom_structure import (
    Layer,
    Loom_Structure,
    MaterialProvider,
    SolverArrays,
)
from loom_matrix import LoomScatterMatrix
from loom_spectraldata import OpticalWeaver
from loom_targets import TargetWeaver, calculate_merit

from needle_synthesis import (
    NeedleSynthesizer,
    NeedleConfig,
    NeedleCycleResult,
    CleanupResult,
    InflateResult,
    ArrayMaterialProvider,
    _collect_target_angles,
    _RESULT_KEY_MAP,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Termination reason enum
# ═══════════════════════════════════════════════════════════════════════════════
class TerminationReason(Enum):
    """Why the pipeline loop stopped."""
    LAYER_BUDGET_REACHED = auto()
    THICKNESS_BUDGET_REACHED = auto()
    STAGNATION_PLATEAU = auto()
    STAGNATION_OSCILLATION = auto()
    STAGNATION_DIVERGENCE = auto()
    MAX_ITERATIONS_REACHED = auto()
    MERIT_TARGET_REACHED = auto()
    USER_ABORT = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Pipeline configuration
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class PipelineConfig:
    """
    Configuration for the continuous needle pipeline.

    Budgets
    -------
    max_film_layers : int
        Stop when the film-layer count reaches this.
    max_total_thickness_nm : float
        Stop when total film thickness (nm) reaches this.
    max_macro_cycles : int
        Hard cap on the number of macro-cycles (needle + cleanup + inflate).
    merit_target : float
        Stop early if the merit function drops below this value.

    Clamping
    --------
    clamp_min_nm : float
        Minimum allowed film-layer thickness (nm).  Layers below this
        are removed during cleanup; the optimiser lower bound is set here.
    clamp_max_nm : float
        Maximum allowed film-layer thickness (nm).  Applied as the
        optimiser upper bound and enforced after every inflate/cleanup.

    Needle pass
    -----------
    needles_per_cycle : int
        Number of needle insertions per macro-cycle (forwarded to
        ``NeedleConfig.max_needles`` each pass).

    Cleanup (optional)
    ------------------
    enable_cleanup : bool
        Run cleanup after each needle pass.
    cleanup_min_nm : float
        Minimum thickness for the prune step inside cleanup.
        Defaults to ``clamp_min_nm`` if not set.

    Inflation (optional)
    --------------------
    enable_inflate : bool
        Run QWOT inflation after cleanup (or after needle if cleanup
        is disabled).
    inflate_addon_qwot : float
        QWOT to add each macro-cycle.
    inflate_reference_wl : float
        Reference wavelength (nm) for QWOT calculation.

    Stagnation detection
    --------------------
    stagnation_window : int
        Number of recent macro-cycles to consider.
    stagnation_gradient_tol : float
        If the normalised MF improvement gradient is below this, the
        algorithm is considered stagnant (plateau).
    stagnation_oscillation_ratio : float
        Fraction of sign-alternating MF deltas within the window
        that triggers oscillation detection (0.0–1.0).
    stagnation_divergence_count : int
        Number of consecutive MF *increases* that triggers divergence
        detection.
    """
    # -- budgets --
    max_film_layers: int = 40
    max_total_thickness_nm: float = 5000.0
    max_macro_cycles: int = 50
    merit_target: float = 0.0

    # -- clamping --
    clamp_min_nm: float = 2.0
    clamp_max_nm: float = 1000.0

    # -- needle --
    needles_per_cycle: int = 3

    # -- cleanup --
    enable_cleanup: bool = True
    cleanup_min_nm: Optional[float] = None  # defaults to clamp_min_nm
    cleanup_max_removals: Optional[int] = None  # None = no cap

    # -- inflate --
    enable_inflate: bool = False
    inflate_addon_qwot: float = 2.0
    inflate_reference_wl: float = 550.0
    inflate_max_layers: Optional[int] = None  # None = all layers

    # -- stagnation --
    stagnation_window: int = 5
    stagnation_gradient_tol: float = 1e-4
    stagnation_oscillation_ratio: float = 0.75
    stagnation_divergence_count: int = 3

    def __post_init__(self) -> None:
        if self.cleanup_min_nm is None:
            self.cleanup_min_nm = self.clamp_min_nm
        if self.clamp_min_nm < 0:
            raise ValueError("clamp_min_nm must be non-negative.")
        if self.clamp_max_nm <= self.clamp_min_nm:
            raise ValueError("clamp_max_nm must be greater than clamp_min_nm.")
        if self.stagnation_window < 2:
            raise ValueError("stagnation_window must be ≥ 2.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Result data classes
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class PipelinePhaseResult:
    """
    Record of one macro-cycle (needle pass + optional cleanup + inflate).
    """
    macro_cycle: int
    mf_after_needle: float
    mf_after_cleanup: Optional[float]
    mf_after_inflate: Optional[float]
    mf_end: float
    layer_count: int
    total_thickness_nm: float
    needle_results: List[NeedleCycleResult]
    cleanup_result: Optional[CleanupResult]
    inflate_result: Optional[InflateResult]


@dataclass(slots=True)
class PipelineResult:
    """Full output of a pipeline run."""
    phases: List[PipelinePhaseResult]
    termination: TerminationReason
    final_mf: float
    final_layer_count: int
    final_total_thickness_nm: float
    stagnation_detail: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Stagnation detector
# ═══════════════════════════════════════════════════════════════════════════════
class StagnationDetector:
    """
    Analyses the merit-function trajectory to detect when the pipeline
    is no longer making meaningful progress.

    Three failure modes are detected independently:

    1.  **Plateau**
        The normalised improvement gradient (linear regression slope of
        the last *W* MF values, divided by their mean) falls below a
        tolerance.  A negative gradient means improvement; a near-zero
        gradient means the algorithm is flat-lining.

    2.  **Oscillation**
        The MF bounces up and down instead of monotonically decreasing.
        Measured as the fraction of sign-alternating consecutive deltas
        within the window.  A ratio > threshold means the algorithm is
        trapped in a cycle of insert ↔ prune that undoes its own work.

    3.  **Divergence**
        The MF increases for several consecutive macro-cycles, meaning
        cleanup/inflate is actively harming the design.

    Parameters
    ----------
    window : int
        Number of recent MF samples to keep.
    gradient_tol : float
        Plateau detection threshold on the normalised gradient.
    oscillation_ratio : float
        Oscillation detection threshold (0–1).
    divergence_count : int
        Number of consecutive increases that trigger divergence.
    """

    __slots__ = (
        "_window",
        "_gradient_tol",
        "_oscillation_ratio",
        "_divergence_count",
        "_history",
    )

    def __init__(
        self,
        window: int = 5,
        gradient_tol: float = 1e-4,
        oscillation_ratio: float = 0.75,
        divergence_count: int = 3,
    ) -> None:
        self._window = max(2, window)
        self._gradient_tol = gradient_tol
        self._oscillation_ratio = oscillation_ratio
        self._divergence_count = max(2, divergence_count)
        self._history: List[float] = []

    def record(self, mf: float) -> None:
        """Append a merit-function sample."""
        self._history.append(mf)

    @property
    def count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> List[float]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

    # -- analysis ----------------------------------------------------------
    def _recent(self) -> np.ndarray:
        """Return the last *window* samples as an array."""
        tail = self._history[-self._window:]
        return np.array(tail, dtype=np.float64)

    def normalised_gradient(self) -> float:
        """
        Linear-regression slope of the recent MF window, normalised
        by the mean MF value.

        Returns a dimensionless rate:
            < 0  →  MF is decreasing (good)
            ≈ 0  →  plateau
            > 0  →  MF is increasing (divergence)

        Returns ``-inf`` if insufficient data.
        """
        if self.count < 2:
            return float("-inf")

        recent = self._recent()
        n = len(recent)
        x = np.arange(n, dtype=np.float64)

        # Least-squares slope: cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = recent.mean()

        if y_mean == 0.0:
            return 0.0

        numerator = np.sum((x - x_mean) * (recent - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0.0:
            return 0.0

        slope = numerator / denominator
        return float(slope / abs(y_mean))

    def oscillation_fraction(self) -> float:
        """
        Fraction of consecutive MF deltas that alternate in sign.

        Returns 0.0 if insufficient data or no alternation, up to 1.0
        for perfect oscillation.
        """
        if self.count < 3:
            return 0.0

        recent = self._recent()
        deltas = np.diff(recent)

        if len(deltas) < 2:
            return 0.0

        # Count sign alternations
        signs = np.sign(deltas)
        # Remove zeros (no change) — treat as non-alternating
        nonzero_mask = signs != 0.0
        signs_nz = signs[nonzero_mask]

        if len(signs_nz) < 2:
            return 0.0

        alternations = np.sum(np.diff(signs_nz) != 0.0)
        max_possible = len(signs_nz) - 1

        return float(alternations / max_possible)

    def consecutive_increases(self) -> int:
        """
        Number of consecutive MF increases counting back from the most
        recent sample.
        """
        if self.count < 2:
            return 0

        streak = 0
        for i in range(len(self._history) - 1, 0, -1):
            if self._history[i] >= self._history[i - 1]:
                streak += 1
            else:
                break
        return streak

    def check(self) -> Optional[TerminationReason]:
        """
        Run all detectors.  Returns a ``TerminationReason`` if
        stagnation is detected, or ``None`` if the pipeline should
        continue.

        The check order is divergence → oscillation → plateau, because
        divergence is the most urgent signal and plateau is the most
        common (and most tolerant).
        """
        if self.count < self._window:
            return None

        # 1. Divergence: several consecutive MF increases
        if self.consecutive_increases() >= self._divergence_count:
            return TerminationReason.STAGNATION_DIVERGENCE

        # 2. Oscillation: bouncing up and down
        osc = self.oscillation_fraction()
        if osc >= self._oscillation_ratio:
            return TerminationReason.STAGNATION_OSCILLATION

        # 3. Plateau: near-zero gradient
        grad = self.normalised_gradient()
        # A gradient near zero or slightly positive means no improvement
        if abs(grad) < self._gradient_tol or grad > 0.0:
            return TerminationReason.STAGNATION_PLATEAU

        return None

    def summary(self) -> str:
        """Human-readable state summary."""
        grad = self.normalised_gradient()
        osc = self.oscillation_fraction()
        cons = self.consecutive_increases()
        return (
            f"StagnationDetector("
            f"samples={self.count}, "
            f"gradient={grad:+.2e}, "
            f"oscillation={osc:.0%}, "
            f"consecutive_up={cons})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ClampedNeedleSynthesizer
# ═══════════════════════════════════════════════════════════════════════════════
class ClampedNeedleSynthesizer(NeedleSynthesizer):
    """
    Extension of NeedleSynthesizer with hard min/max thickness clamping.

    Every operation that changes layer thicknesses — optimisation,
    cleanup, inflation — enforces the clamp range so that no film layer
    is ever thinner than ``clamp_min`` or thicker than ``clamp_max``
    when the method returns.

    Layers that fall below ``clamp_min`` after an optimisation step are
    removed (the optimizer is free to drive layers to zero, but they are
    pruned immediately).  Layers above ``clamp_max`` are hard-clamped.

    Parameters
    ----------
    clamp_min : float
        Minimum allowed film-layer thickness (nm).
    clamp_max : float
        Maximum allowed film-layer thickness (nm).
    **kwargs
        Forwarded to ``NeedleSynthesizer.__init__``.
    """

    def __init__(
        self,
        *,
        clamp_min: float = 2.0,
        clamp_max: float = 1000.0,
        structure: Loom_Structure,
        wavls: np.ndarray,
        target_weaver: TargetWeaver,
        contrasting_materials: Dict[str, str],
        config: Optional[NeedleConfig] = None,
    ) -> None:
        super().__init__(
            structure=structure,
            wavls=wavls,
            target_weaver=target_weaver,
            contrasting_materials=contrasting_materials,
            config=config,
        )
        if clamp_max <= clamp_min:
            raise ValueError(
                f"clamp_max ({clamp_max}) must exceed clamp_min ({clamp_min})."
            )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    @classmethod
    def from_synthesizer(
        cls,
        synth: NeedleSynthesizer,
        clamp_min: float = 2.0,
        clamp_max: float = 1000.0,
    ) -> "ClampedNeedleSynthesizer":
        """
        Wrap an existing NeedleSynthesizer with clamping.

        The returned object shares the same ``structure`` reference, so
        mutations propagate.  This is intentional — the pipeline operates
        on the same stack throughout.
        """
        obj = cls.__new__(cls)
        # Copy all base attributes by reference (shared state is desired)
        obj.structure = synth.structure
        obj.wavls = synth.wavls
        obj.target_weaver = synth.target_weaver
        obj.contrast_map = synth.contrast_map
        obj.cfg = synth.cfg
        obj.clamp_min = clamp_min
        obj.clamp_max = clamp_max
        return obj

    # -- clamping primitives -----------------------------------------------
    def clamp_all_layers(self) -> Tuple[int, int]:
        """
        Enforce [clamp_min, clamp_max] on every film layer.

        Layers below ``clamp_min`` are *removed* (not clamped up, which
        would inject unphysical thin layers the optimizer tried to
        eliminate).  Layers above ``clamp_max`` are hard-clamped.

        Returns
        -------
        (n_removed, n_capped)
            Count of layers removed (too thin) and capped (too thick).
        """
        layer_list = self.structure.layer_list
        ambient = layer_list[0]
        substrate = layer_list[-1]
        films = layer_list[1:-1]

        surviving: List[Layer] = []
        n_removed = 0
        n_capped = 0

        for layer in films:
            if layer.thickness < self.clamp_min:
                n_removed += 1
                continue
            if layer.thickness > self.clamp_max:
                layer.thickness = self.clamp_max
                n_capped += 1
            surviving.append(layer)

        self.structure.layer_list = [ambient] + surviving + [substrate]
        return n_removed, n_capped

    # -- overridden optimize_thicknesses with bounds -----------------------
    def optimize_thicknesses(self) -> float:
        """
        Bounded least-squares optimisation with [clamp_min, clamp_max]
        enforced both as optimizer bounds AND as a post-optimisation
        sweep.

        Layers driven below ``clamp_min`` by the optimizer are removed.
        """
        films = self.film_layers
        if not films:
            return self.evaluate_merit()

        opt_indices: List[int] = []
        for i, layer in enumerate(films):
            if layer.optimize:
                opt_indices.append(i + 1)

        if not opt_indices:
            return self.evaluate_merit()

        layer_list = self.structure.layer_list

        def residual_vector(thickness_vec: np.ndarray) -> np.ndarray:
            for k, idx in enumerate(opt_indices):
                layer_list[idx].thickness = float(thickness_vec[k])

            sa = self.structure.get_solver_inputs()
            target_angles = _collect_target_angles(self.target_weaver)
            solver = LoomScatterMatrix(
                sa.indices, sa.thicknesses, sa.incoherent_flags,
                sa.rough_types, sa.rough_vals,
                self.wavls, target_angles, theta_is_radians=True,
            )
            result = solver.compute_RT(mode="u")

            residuals: List[np.ndarray] = []

            for key in self.target_weaver.target_keys():
                angle, pol, spectral = key
                result_key = _RESULT_KEY_MAP.get((pol, spectral))
                if result_key is None:
                    continue
                data = result.get(result_key)
                if data is None:
                    continue
                if data.ndim == 2:
                    a_idx = int(np.argmin(np.abs(target_angles - angle)))
                    sim_vals = data[a_idx, :]
                else:
                    sim_vals = data

                for frame, entry in self.target_weaver.iter_target_frames(key):
                    target_wl = frame.wavelength
                    if target_wl is None or target_wl.size == 0:
                        continue
                    target_val = frame[key]
                    tol = entry.tolerances
                    sim_at_tgt = np.interp(target_wl, self.wavls, sim_vals)
                    diff = sim_at_tgt - target_val

                    if entry.kind == "e":
                        residuals.append(diff / tol)
                    elif entry.kind == "a":
                        residuals.append(np.where(diff < 0, diff / tol, 0.0))
                    elif entry.kind == "b":
                        residuals.append(np.where(diff > 0, diff / tol, 0.0))

            if not residuals:
                return np.zeros(1)
            return np.concatenate(residuals)

        d0 = np.array(
            [layer_list[i].thickness for i in opt_indices], dtype=np.float64
        )

        # --- Bounded optimisation ---
        # Use 0.0 as the true lower bound so the optimizer can drive
        # layers toward zero; clamp_all_layers removes them afterward.
        # The upper bound is hard-enforced by least_squares.
        lb = np.full_like(d0, 0.0)
        ub = np.full_like(d0, self.clamp_max)

        # Ensure initial guess is within bounds
        d0 = np.clip(d0, lb, ub)

        res = least_squares(
            residual_vector,
            d0,
            bounds=(lb, ub),
            method=self.cfg.optimizer_method,
            ftol=self.cfg.optimizer_ftol,
        )

        for k, idx in enumerate(opt_indices):
            layer_list[idx].thickness = float(res.x[k])

        # Enforce clamping (removes thin, caps thick)
        n_removed, n_capped = self.clamp_all_layers()
        if n_removed > 0 or n_capped > 0:
            logger.debug(
                "Post-optimisation clamp: removed %d, capped %d.",
                n_removed, n_capped,
            )

        return self.evaluate_merit()

    # -- overridden cleanup with clamping ----------------------------------
    def cleanup_design(
        self,
        min_thickness: Optional[float] = None,
        max_removals: Optional[int] = None,
        reoptimize: bool = True,
    ) -> CleanupResult:
        """
        Cleanup with clamping.  Uses ``self.clamp_min`` as the minimum
        thickness if *min_thickness* is not provided.
        """
        threshold = min_thickness if min_thickness is not None else self.clamp_min
        result = super().cleanup_design(
            min_thickness=threshold,
            max_removals=max_removals,
            reoptimize=reoptimize,
        )
        # Post-cleanup clamp (catches anything the re-optimisation
        # pushed outside bounds)
        self.clamp_all_layers()
        return result

    # -- overridden inflate with clamping ----------------------------------
    def inflate_design(
        self,
        addon_qwot: float,
        reference_wl: float,
        *,
        max_layers: Optional[int] = None,
        reoptimize: bool = True,
    ) -> InflateResult:
        """
        Impact-ranked inflate with pre/post clamping.

        Uses the base class's merit-impact scoring to select the best
        *max_layers* layers, then applies the QWOT addon with clamping
        enforced before and after re-optimisation.
        """
        mf_before = self.evaluate_merit()
        films = self.film_layers
        total_before = sum(L.thickness for L in films)

        # Determine which layers to inflate via merit-impact scoring
        if max_layers is not None and max_layers < len(films):
            scored: List[Tuple[int, float]] = []
            for i in range(len(films)):
                trial_mf = self._evaluate_inflate_impact(
                    i, addon_qwot, reference_wl,
                )
                scored.append((i, trial_mf))
            scored.sort(key=lambda t: t[1])
            inflate_indices = [idx for idx, _ in scored[:max_layers]]
        else:
            inflate_indices = list(range(len(films)))

        # Apply inflation to selected layers
        for idx in inflate_indices:
            layer = films[idx]
            qwot_nm = self._qwot_nm(layer.material, reference_wl)
            delta = addon_qwot * qwot_nm
            layer.thickness = max(0.0, layer.thickness + delta)

        # Clamp BEFORE re-optimisation
        self.clamp_all_layers()

        if reoptimize:
            mf_after = self.optimize_thicknesses()
        else:
            mf_after = self.evaluate_merit()

        # Final clamp
        self.clamp_all_layers()

        total_after = sum(L.thickness for L in self.film_layers)

        return InflateResult(
            merit_before=mf_before,
            merit_after=mf_after,
            total_thickness_before=total_before,
            total_thickness_after=total_after,
            layer_count=self.layer_count,
            addon_qwot=addon_qwot,
            reference_wavelength=reference_wl,
        )

    # -- helper properties -------------------------------------------------
    @property
    def total_film_thickness(self) -> float:
        """Sum of all film-layer thicknesses (nm)."""
        return sum(L.thickness for L in self.film_layers)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NeedlePipeline — the orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
class NeedlePipeline:
    """
    Continuous iterative needle synthesis pipeline.

    Each macro-cycle executes:
        1. Needle pass   — insert ``needles_per_cycle`` needles + optimise.
        2. Cleanup        — merge, prune, re-optimise.   (optional)
        3. Inflate        — QWOT addon, re-optimise.      (optional)
        4. Stagnation check — analyse MF trajectory.

    The loop terminates when any stopping condition fires.

    Parameters
    ----------
    synth : ClampedNeedleSynthesizer
        The synthesis engine (owns the structure, targets, materials).
    config : PipelineConfig
        All loop-control parameters.

    Example
    -------
    >>> pipeline = NeedlePipeline.from_synthesizer(base_synth, config)
    >>> result = pipeline.run()
    >>> print(result.termination, result.final_mf)
    """

    def __init__(
        self,
        synth: ClampedNeedleSynthesizer,
        config: PipelineConfig,
    ) -> None:
        self.synth = synth
        self.cfg = config

        self._detector = StagnationDetector(
            window=config.stagnation_window,
            gradient_tol=config.stagnation_gradient_tol,
            oscillation_ratio=config.stagnation_oscillation_ratio,
            divergence_count=config.stagnation_divergence_count,
        )

    @classmethod
    def from_synthesizer(
        cls,
        synth: NeedleSynthesizer,
        config: PipelineConfig,
    ) -> "NeedlePipeline":
        """
        Convenience factory: wraps a plain NeedleSynthesizer in a
        ClampedNeedleSynthesizer and builds the pipeline.
        """
        clamped = ClampedNeedleSynthesizer.from_synthesizer(
            synth,
            clamp_min=config.clamp_min_nm,
            clamp_max=config.clamp_max_nm,
        )
        return cls(clamped, config)

    # -- budget checks -----------------------------------------------------
    def _check_budgets(self) -> Optional[TerminationReason]:
        """Return a reason if any budget is exhausted, else None."""
        if self.synth.layer_count >= self.cfg.max_film_layers:
            return TerminationReason.LAYER_BUDGET_REACHED
        if self.synth.total_film_thickness >= self.cfg.max_total_thickness_nm:
            return TerminationReason.THICKNESS_BUDGET_REACHED
        mf = self.synth.evaluate_merit()
        if self.cfg.merit_target > 0.0 and mf <= self.cfg.merit_target:
            return TerminationReason.MERIT_TARGET_REACHED
        return None

    # -- main loop ---------------------------------------------------------
    def run(
        self,
        callback: Optional[
            Callable[
                [int, PipelinePhaseResult, StagnationDetector], None
            ]
        ] = None,
    ) -> PipelineResult:
        """
        Execute the continuous pipeline.

        Parameters
        ----------
        callback : callable, optional
            ``callback(macro_cycle, phase_result, detector)`` is called
            after each macro-cycle.  Useful for logging, plotting, or
            implementing a user-abort by raising ``KeyboardInterrupt``.

        Returns
        -------
        PipelineResult
        """
        phases: List[PipelinePhaseResult] = []
        self._detector.reset()

        # Adjust needle config for per-cycle budget
        original_max_needles = self.synth.cfg.max_needles
        self.synth.cfg.max_needles = self.cfg.needles_per_cycle

        termination = TerminationReason.MAX_ITERATIONS_REACHED
        stag_detail: Optional[str] = None

        try:
            for macro in range(1, self.cfg.max_macro_cycles + 1):
                logger.info(
                    "═══ Macro-cycle %d  (layers=%d, d=%.0f nm) ═══",
                    macro, self.synth.layer_count,
                    self.synth.total_film_thickness,
                )

                # -- Pre-flight budget check --
                reason = self._check_budgets()
                if reason is not None:
                    termination = reason
                    logger.info("Budget reached before needle pass: %s", reason.name)
                    break

                # ── Phase 1: Needle pass ──────────────────────────────
                needle_results = self.synth.run()
                mf_needle = self.synth.evaluate_merit()

                logger.info(
                    "  Needle pass: MF=%.6f, layers=%d",
                    mf_needle, self.synth.layer_count,
                )

                # Budget check after needle
                reason = self._check_budgets()
                if reason is not None:
                    phase = self._build_phase(
                        macro, mf_needle, None, None,
                        needle_results, None, None,
                    )
                    phases.append(phase)
                    self._detector.record(mf_needle)
                    if callback:
                        callback(macro, phase, self._detector)
                    termination = reason
                    break

                # ── Phase 2: Cleanup (optional) ───────────────────────
                cleanup_result: Optional[CleanupResult] = None
                mf_cleanup: Optional[float] = None

                if self.cfg.enable_cleanup:
                    cleanup_result = self.synth.cleanup_design(
                        min_thickness=self.cfg.cleanup_min_nm,
                        max_removals=self.cfg.cleanup_max_removals,
                        reoptimize=True,
                    )
                    mf_cleanup = cleanup_result.merit_after
                    logger.info(
                        "  Cleanup: MF=%.6f, layers=%d "
                        "(merged=%d, removed=%d)",
                        mf_cleanup, self.synth.layer_count,
                        cleanup_result.layers_merged,
                        cleanup_result.layers_removed_thin,
                    )

                # ── Phase 3: Inflate (optional) ───────────────────────
                inflate_result: Optional[InflateResult] = None
                mf_inflate: Optional[float] = None

                if self.cfg.enable_inflate:
                    inflate_result = self.synth.inflate_design(
                        addon_qwot=self.cfg.inflate_addon_qwot,
                        reference_wl=self.cfg.inflate_reference_wl,
                        max_layers=self.cfg.inflate_max_layers,
                        reoptimize=True,
                    )
                    mf_inflate = inflate_result.merit_after
                    logger.info(
                        "  Inflate: MF=%.6f, d=%.0f nm (+%.1f QWOT)",
                        mf_inflate, self.synth.total_film_thickness,
                        self.cfg.inflate_addon_qwot,
                    )

                # ── Record phase ──────────────────────────────────────
                phase = self._build_phase(
                    macro, mf_needle, mf_cleanup, mf_inflate,
                    needle_results, cleanup_result, inflate_result,
                )
                phases.append(phase)

                # ── Stagnation check ──────────────────────────────────
                self._detector.record(phase.mf_end)

                logger.info(
                    "  End of cycle: MF=%.6f  %s",
                    phase.mf_end, self._detector.summary(),
                )

                if callback:
                    callback(macro, phase, self._detector)

                stag = self._detector.check()
                if stag is not None:
                    termination = stag
                    stag_detail = self._detector.summary()
                    logger.info("Stagnation detected: %s", stag.name)
                    break

                # Post-cycle budget check
                reason = self._check_budgets()
                if reason is not None:
                    termination = reason
                    break

        except KeyboardInterrupt:
            termination = TerminationReason.USER_ABORT
            logger.info("Pipeline aborted by user.")

        finally:
            # Restore original config
            self.synth.cfg.max_needles = original_max_needles

        # ── Final optimisation ────────────────────────────────────────────
        logger.info("Final optimisation pass...")
        final_mf = self.synth.optimize_thicknesses()

        # Final clamp sweep
        self.synth.clamp_all_layers()
        final_mf = self.synth.evaluate_merit()

        logger.info(
            "Pipeline complete: %s  MF=%.6f  layers=%d  d=%.0f nm",
            termination.name, final_mf, self.synth.layer_count,
            self.synth.total_film_thickness,
        )

        return PipelineResult(
            phases=phases,
            termination=termination,
            final_mf=final_mf,
            final_layer_count=self.synth.layer_count,
            final_total_thickness_nm=self.synth.total_film_thickness,
            stagnation_detail=stag_detail,
        )

    # -- internal helpers --------------------------------------------------
    def _build_phase(
        self,
        macro: int,
        mf_needle: float,
        mf_cleanup: Optional[float],
        mf_inflate: Optional[float],
        needle_results: List[NeedleCycleResult],
        cleanup_result: Optional[CleanupResult],
        inflate_result: Optional[InflateResult],
    ) -> PipelinePhaseResult:
        mf_end = mf_inflate or mf_cleanup or mf_needle
        return PipelinePhaseResult(
            macro_cycle=macro,
            mf_after_needle=mf_needle,
            mf_after_cleanup=mf_cleanup,
            mf_after_inflate=mf_inflate,
            mf_end=mf_end,
            layer_count=self.synth.layer_count,
            total_thickness_nm=self.synth.total_film_thickness,
            needle_results=needle_results,
            cleanup_result=cleanup_result,
            inflate_result=inflate_result,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Usage example
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from loom_structure import Layer, Loom_Structure
    from loom_targets import SpectralTarget, TargetCollection

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s | %(message)s",
    )

    # ── Materials ─────────────────────────────────────────────────────────
    wavls = np.linspace(400, 800, 200)
    n_wavs = len(wavls)

    provider = ArrayMaterialProvider({
        "air": np.full(n_wavs, 1.00 + 0j),
        "H":   np.full(n_wavs, 2.35 + 0j),
        "L":   np.full(n_wavs, 1.46 + 0j),
        "sub": np.full(n_wavs, 1.52 + 0j),
    })

    # ── Structure (seed) ──────────────────────────────────────────────────
    structure = Loom_Structure(
        materials=provider,
        layer_list=[
            Layer(thickness=0.0, material_name="air",
                  optimize=False, needle=False, layer_typ=0),
            Layer(thickness=100.0, material_name="H"),
            Layer(thickness=0.0, material_name="sub",
                  optimize=False, needle=False, layer_typ=0),
        ],
    )

    # ── Target: broadband AR ──────────────────────────────────────────────
    targets = TargetCollection()
    targets.add(SpectralTarget(
        wavelengths=wavls,
        values=np.zeros(n_wavs, dtype=np.float64),
        tolerances=np.full(n_wavs, 0.01, dtype=np.float64),
        angle=0.0, polarization="u", spectral="R", kind="e",
    ))
    target_weaver = targets.build_weaver()

    # ── Base synthesizer ──────────────────────────────────────────────────
    base_synth = NeedleSynthesizer(
        structure=structure,
        wavls=wavls,
        target_weaver=target_weaver,
        contrasting_materials={"H": "L", "L": "H"},
        config=NeedleConfig(max_needles=4, scan_step_nm=2.0),
    )

    # ── Pipeline configuration ────────────────────────────────────────────
    pipe_cfg = PipelineConfig(
        # Budgets
        max_film_layers=20,
        max_total_thickness_nm=3000.0,
        max_macro_cycles=15,
        merit_target=0.0,

        # Clamping
        clamp_min_nm=3.0,
        clamp_max_nm=500.0,

        # Needle
        needles_per_cycle=3,

        # Cleanup
        enable_cleanup=True,
        cleanup_min_nm=3.0,
        cleanup_max_removals=2,           # remove at most 2 thin layers per cycle

        # Inflate
        enable_inflate=True,
        inflate_addon_qwot=2.0,
        inflate_reference_wl=550.0,
        inflate_max_layers=2,             # inflate only 2 most impactful layers

        # Stagnation
        stagnation_window=4,
        stagnation_gradient_tol=1e-4,
        stagnation_oscillation_ratio=0.75,
        stagnation_divergence_count=3,
    )

    # ── Run pipeline ──────────────────────────────────────────────────────
    pipeline = NeedlePipeline.from_synthesizer(base_synth, pipe_cfg)

    def on_phase(macro, phase, detector):
        """Per-cycle callback — print a summary line."""
        print(
            f"  ╰─ cycle {macro:2d}: MF={phase.mf_end:.6f}  "
            f"layers={phase.layer_count}  "
            f"d={phase.total_thickness_nm:.0f} nm  "
            f"{detector.summary()}"
        )

    result = pipeline.run(callback=on_phase)

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE RESULT")
    print("=" * 70)
    print(f"  Termination:     {result.termination.name}")
    if result.stagnation_detail:
        print(f"  Stagnation:      {result.stagnation_detail}")
    print(f"  Final MF:        {result.final_mf:.6f}")
    print(f"  Film layers:     {result.final_layer_count}")
    print(f"  Total thickness: {result.final_total_thickness_nm:.1f} nm")
    print(f"  Macro-cycles:    {len(result.phases)}")

    # Per-phase summary table
    print(f"\n  {'Cycle':<7} {'MF(needle)':<12} {'MF(clean)':<12} "
          f"{'MF(inflate)':<12} {'MF(end)':<12} {'Layers':<8} {'d (nm)':<10}")
    print("  " + "-" * 73)
    for p in result.phases:
        c_mf = f"{p.mf_after_cleanup:.6f}" if p.mf_after_cleanup is not None else "—"
        i_mf = f"{p.mf_after_inflate:.6f}" if p.mf_after_inflate is not None else "—"
        print(
            f"  {p.macro_cycle:<7d} {p.mf_after_needle:<12.6f} "
            f"{c_mf:<12} {i_mf:<12} {p.mf_end:<12.6f} "
            f"{p.layer_count:<8d} {p.total_thickness_nm:<10.0f}"
        )

    # Final design
    clamped = pipeline.synth
    clamped.print_design(reference_wl=550.0)
