# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Module: needle_synthesis.py — Needle optimisation (Tikhonravov method)
integrated with the Loom thin-film architecture.

This module implements the needle method for automated thin-film design
synthesis, fully leveraging the Loom class hierarchy:

    Loom_Structure / Layer   — stack definition and solver-array generation
    MaterialProvider         — pluggable material data source
    LoomScatterMatrix        — high-performance S-matrix TMM solver
    TargetWeaver / calculate_merit — spectral/angular target evaluation

The needle algorithm iteratively:
    1. Optimises existing layer thicknesses (local least-squares).
    2. Scans a virtual "test needle" through the optical depth of the stack
       to compute the P-function (merit function sensitivity profile).
    3. Inserts a thin seed layer at the position of maximum improvement.
    4. Repeats until convergence or a layer-count budget is exhausted.

Reference:
    Tikhonravov, A.V., Trubetskov, M.K., DeBell, G.W.,
    "Application of the needle optimization technique to the design of
    optical coatings," Appl. Opt. 35(28), 5493–5508 (1996).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

import numpy as np
from scipy.optimize import least_squares

from loom_structure import (
    Layer,
    Group,
    Loom_Structure,
    MaterialProvider,
    SolverArrays,
    FLOAT_TYPE,
    COMPLEX_TYPE,
)
from loom_matrix import LoomScatterMatrix
from loom_spectraldata import OpticalWeaver
from loom_targets import (
    SpectralTarget,
    AngularTarget,
    TargetCollection,
    TargetWeaver,
    calculate_merit,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ArrayMaterialProvider — lightweight provider for pre-computed nk arrays
# ═══════════════════════════════════════════════════════════════════════════════
class ArrayMaterialProvider:
    """
    MaterialProvider backed by a simple name → ndarray mapping.

    Useful for synthesis workflows where the complex refractive indices
    are already evaluated on the simulation wavelength grid (e.g. constant
    indices, or pre-interpolated dispersion data).

    Parameters
    ----------
    nk_dict : dict[str, np.ndarray]
        Mapping of material names to complex128 nk arrays,
        each of shape (n_wavs,).
    """
    __slots__ = ("_dict",)

    def __init__(self, nk_dict: Dict[str, np.ndarray]) -> None:
        self._dict = {
            name: np.asarray(nk, dtype=np.complex128)
            for name, nk in nk_dict.items()
        }

    def get_nk(self, material_name: str) -> np.ndarray:
        return self._dict[material_name]

    def contains(self, material_name: str) -> bool:
        return material_name in self._dict


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class NeedleConfig:
    """
    Tuning knobs for the needle synthesis algorithm.

    Attributes
    ----------
    max_needles : int
        Maximum number of needle insertion cycles.
    convergence_threshold : float
        Minimum merit-function improvement to continue inserting.
    min_layer_thickness : float
        Layers thinner than this (nm) are pruned after optimisation.
    needle_test_thickness : float
        Thickness (nm) of the virtual test needle during P-function scan.
    needle_seed_thickness : float
        Thickness (nm) of the seed layer inserted at the best position.
    scan_step_nm : float
        Step size (nm) for the P-function scan within each layer.
    optimizer_method : str
        Scipy least_squares method ('trf', 'dogbox', 'lm').
    optimizer_ftol : float
        Function tolerance for the least-squares optimiser.
    """
    max_needles: int = 10
    convergence_threshold: float = 1e-4
    min_layer_thickness: float = 0.5
    needle_test_thickness: float = 1.0
    needle_seed_thickness: float = 5.0
    scan_step_nm: float = 2.0
    optimizer_method: str = "trf"
    optimizer_ftol: float = 1e-4


@dataclass(slots=True)
class NeedleCycleResult:
    """Record of one needle insertion cycle."""
    cycle: int
    merit_before: float
    merit_after: float
    best_scan_mf: float
    layer_count: int
    insertion_index: Optional[int] = None
    insertion_material: Optional[str] = None


@dataclass(slots=True)
class CleanupResult:
    """Record of a post-synthesis design cleanup pass."""
    merit_before: float
    merit_after: float
    layers_before: int
    layers_after: int
    layers_removed_thin: int
    layers_merged: int


@dataclass(slots=True)
class InflateResult:
    """Record of a QWOT-based thickness inflation pass."""
    merit_before: float
    merit_after: float
    total_thickness_before: float
    total_thickness_after: float
    layer_count: int
    addon_qwot: float
    reference_wavelength: float


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Simulation helper
# ═══════════════════════════════════════════════════════════════════════════════
_RESULT_KEY_MAP: Dict[Tuple[str, str], str] = {
    ("s", "R"): "Rs",
    ("p", "R"): "Rp",
    ("u", "R"): "Ru",
    ("s", "T"): "Ts",
    ("p", "T"): "Tp",
    ("u", "T"): "Tu",
}


def _collect_target_angles(target_weaver: TargetWeaver) -> np.ndarray:
    """Extract unique angle values from all target keys."""
    angles = set()
    for key in target_weaver.target_keys():
        angles.add(key[0])
    return np.array(sorted(angles), dtype=np.float64)


def _simulate_to_weaver(
    structure: Loom_Structure,
    wavls: np.ndarray,
    target_weaver: TargetWeaver,
) -> OpticalWeaver:
    """
    Run the TMM solver for the given structure and store results in an
    OpticalWeaver, keyed to match the target weaver's keys.

    This is the bridge between the solver output and the merit-function
    evaluation in ``calculate_merit(sim_weaver, target_weaver)``.
    """
    sa: SolverArrays = structure.get_solver_inputs()
    target_angles = _collect_target_angles(target_weaver)

    # Build solver
    solver = LoomScatterMatrix(
        sa.indices,
        sa.thicknesses,
        sa.incoherent_flags,
        sa.rough_types,
        sa.rough_vals,
        wavls,
        target_angles,
        theta_is_radians=True,
    )

    # Run full unpolarised R/T (covers s, p, and u)
    result = solver.compute_RT(mode="u")

    # Distribute results into an OpticalWeaver
    sim_weaver = OpticalWeaver()

    for key in target_weaver.target_keys():
        angle, pol, spectral = key
        result_key = _RESULT_KEY_MAP.get((pol, spectral))
        if result_key is None:
            logger.warning("Unmapped target key %s — skipping.", key)
            continue

        data_array = result.get(result_key)
        if data_array is None:
            continue

        # If multiple angles, find the row index for this angle
        if data_array.ndim == 2:
            angle_idx = int(np.argmin(np.abs(target_angles - angle)))
            values = data_array[angle_idx, :]
        else:
            values = data_array

        sim_weaver.set_data(key, values, wavelength=wavls)

    return sim_weaver


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NeedleSynthesizer
# ═══════════════════════════════════════════════════════════════════════════════
class NeedleSynthesizer:
    """
    Automated thin-film design synthesis via the needle method.

    Operates on a Loom_Structure whose layer_list is:
        [ambient, film_0, film_1, …, film_N, substrate]

    The ambient (index 0) and substrate (index -1) are fixed; all film
    layers with ``layer.needle == True`` are candidates for needle
    insertion, and those with ``layer.optimize == True`` are included in
    thickness optimisation.

    Parameters
    ----------
    structure : Loom_Structure
        The initial thin-film stack.  Must have materials set.
    wavls : np.ndarray
        Simulation wavelength grid (nm).
    target_weaver : TargetWeaver
        Pre-built target weaver (from TargetCollection.build_weaver()).
    contrasting_materials : dict[str, str]
        Mapping from each film material name to the material that
        provides maximal index contrast for needle insertion.
        E.g. ``{'H': 'L', 'L': 'H'}``.
    config : NeedleConfig, optional
        Algorithm tuning parameters.

    Example
    -------
    >>> from loom_structure import Layer, Loom_Structure
    >>> from loom_targets import SpectralTarget, TargetCollection
    >>>
    >>> wavls = np.linspace(400, 800, 200)
    >>> nk = {'air': np.ones(200, dtype=complex),
    ...       'H':   np.full(200, 2.35+0j),
    ...       'L':   np.full(200, 1.46+0j),
    ...       'sub': np.full(200, 1.52+0j)}
    >>> provider = ArrayMaterialProvider(nk)
    >>>
    >>> struct = Loom_Structure(materials=provider, layer_list=[
    ...     Layer(thickness=0.0, material_name='air', optimize=False, needle=False),
    ...     Layer(thickness=150.0, material_name='H'),
    ...     Layer(thickness=0.0, material_name='sub', optimize=False, needle=False),
    ... ])
    >>>
    >>> targets = TargetCollection()
    >>> targets.add(SpectralTarget(
    ...     wavelengths=wavls, values=np.zeros(200),
    ...     tolerances=np.full(200, 0.01),
    ...     angle=0.0, polarization='u', spectral='R',
    ... ))
    >>> tw = targets.build_weaver()
    >>>
    >>> synth = NeedleSynthesizer(
    ...     struct, wavls, tw,
    ...     contrasting_materials={'H': 'L', 'L': 'H'},
    ... )
    >>> history = synth.run()
    """

    def __init__(
        self,
        structure: Loom_Structure,
        wavls: np.ndarray,
        target_weaver: TargetWeaver,
        contrasting_materials: Dict[str, str],
        config: Optional[NeedleConfig] = None,
    ) -> None:
        self.structure = structure
        self.wavls = np.asarray(wavls, dtype=np.float64)
        self.target_weaver = target_weaver
        self.contrast_map = contrasting_materials
        self.cfg = config or NeedleConfig()

        # Validate
        if structure.materials is None:
            raise ValueError("Structure has no MaterialProvider set.")
        if len(structure.layer_list) < 3:
            raise ValueError(
                "Structure must have at least ambient + 1 film + substrate."
            )

    # -- properties --------------------------------------------------------
    @property
    def film_layers(self) -> List[Layer]:
        """The film layers (excluding ambient and substrate)."""
        return self.structure.layer_list[1:-1]

    @property
    def layer_count(self) -> int:
        """Number of film layers."""
        return len(self.film_layers)

    # -- merit function ----------------------------------------------------
    def evaluate_merit(
        self, structure: Optional[Loom_Structure] = None,
    ) -> float:
        """
        Compute the weighted sum-of-squares merit function against targets.
        """
        struct = structure or self.structure
        sim_weaver = _simulate_to_weaver(struct, self.wavls, self.target_weaver)
        return calculate_merit(sim_weaver, self.target_weaver)

    # -- thickness optimisation --------------------------------------------
    def optimize_thicknesses(self) -> float:
        """
        Local least-squares optimisation of film-layer thicknesses.

        Only layers with ``layer.optimize == True`` are varied.  Layers
        thinner than ``cfg.min_layer_thickness`` are pruned afterwards.
        Pruning skips ambient (index 0) and substrate (index -1).

        Returns the post-optimisation merit function value.
        """
        films = self.film_layers
        if not films:
            return self.evaluate_merit()

        # Identify optimisable layers and their indices within layer_list
        opt_indices: List[int] = []
        for i, layer in enumerate(films):
            if layer.optimize:
                opt_indices.append(i + 1)  # +1 for ambient offset

        if not opt_indices:
            return self.evaluate_merit()

        layer_list = self.structure.layer_list

        def residual_vector(thickness_vec: np.ndarray) -> np.ndarray:
            """Return per-target-point residuals for least_squares."""
            # Set thicknesses (clamp to non-negative)
            for k, idx in enumerate(opt_indices):
                layer_list[idx].thickness = max(0.0, float(thickness_vec[k]))

            # Build simulation
            sa = self.structure.get_solver_inputs()
            target_angles = _collect_target_angles(self.target_weaver)
            solver = LoomScatterMatrix(
                sa.indices, sa.thicknesses, sa.incoherent_flags,
                sa.rough_types, sa.rough_vals,
                self.wavls, target_angles, theta_is_radians=True,
            )
            result = solver.compute_RT(mode="u")

            # Collect residuals: (sim - target) / tolerance for each key
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

                    # Interpolate simulation onto target wavelengths
                    sim_at_tgt = np.interp(target_wl, self.wavls, sim_vals)
                    diff = sim_at_tgt - target_val

                    # Apply constraint kind masking
                    if entry.kind == "e":
                        residuals.append(diff / tol)
                    elif entry.kind == "a":
                        masked = np.where(diff < 0, diff / tol, 0.0)
                        residuals.append(masked)
                    elif entry.kind == "b":
                        masked = np.where(diff > 0, diff / tol, 0.0)
                        residuals.append(masked)

            if not residuals:
                return np.zeros(1)
            return np.concatenate(residuals)

        # Initial guess
        d0 = np.array(
            [layer_list[i].thickness for i in opt_indices], dtype=np.float64
        )

        # Optimise
        res = least_squares(
            residual_vector,
            d0,
            bounds=(0.0, np.inf),
            method=self.cfg.optimizer_method,
            ftol=self.cfg.optimizer_ftol,
        )

        # Apply result
        for k, idx in enumerate(opt_indices):
            layer_list[idx].thickness = max(0.0, float(res.x[k]))

        # Prune dead layers (skip ambient[0] and substrate[-1])
        self.structure.layer_list = (
            [layer_list[0]]
            + [L for L in layer_list[1:-1]
               if L.thickness >= self.cfg.min_layer_thickness]
            + [layer_list[-1]]
        )

        return self.evaluate_merit()

    # -- P-function scan ---------------------------------------------------
    def compute_p_function(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float, str]]]:
        """
        Scan a virtual test needle through the optical depth of the stack.

        For each candidate position within layers marked ``needle=True``,
        a thin layer of the contrasting material is virtually inserted and
        the resulting merit function is evaluated.

        Returns
        -------
        positions : ndarray
            Cumulative physical depth (nm) of each scan point.
        mf_values : ndarray
            Merit function at each scan point.
        insertion_data : list of (layer_index, depth_into_layer, needle_material)
            Instructions for performing each insertion.
        """
        current_mf = self.evaluate_merit()
        layer_list = self.structure.layer_list
        films = layer_list[1:-1]

        positions: List[float] = []
        mf_values: List[float] = []
        insertion_data: List[Tuple[int, float, str]] = []

        step = self.cfg.scan_step_nm
        needle_d = self.cfg.needle_test_thickness
        cumulative_depth = 0.0

        for film_idx, layer in enumerate(films):
            global_idx = film_idx + 1  # offset for ambient

            if not layer.needle:
                cumulative_depth += layer.thickness
                continue

            mat_name = layer.material
            needle_mat = self.contrast_map.get(mat_name)
            if needle_mat is None:
                logger.warning(
                    "No contrasting material for '%s' — skipping layer %d.",
                    mat_name, global_idx,
                )
                cumulative_depth += layer.thickness
                continue

            d = layer.thickness
            n_steps = int(d / step)

            for k in range(1, n_steps):
                pos_in_layer = k * step
                d_top = pos_in_layer
                d_bot = d - pos_in_layer

                # Build temporary structure with needle inserted
                temp_layers = (
                    [L.clone() for L in layer_list[:global_idx]]
                    + [
                        Layer(
                            thickness=d_top,
                            material_name=mat_name,
                            optimize=False,
                            needle=False,
                        ),
                        Layer(
                            thickness=needle_d,
                            material_name=needle_mat,
                            optimize=False,
                            needle=False,
                        ),
                        Layer(
                            thickness=d_bot,
                            material_name=mat_name,
                            optimize=False,
                            needle=False,
                        ),
                    ]
                    + [L.clone() for L in layer_list[global_idx + 1:]]
                )

                temp_struct = Loom_Structure(
                    layer_list=temp_layers,
                    group_dict=self.structure.group_dict,
                    materials=self.structure.materials,
                )

                mf = self.evaluate_merit(temp_struct)

                positions.append(cumulative_depth + pos_in_layer)
                mf_values.append(mf)
                insertion_data.append((global_idx, d_top, needle_mat))

            cumulative_depth += d

        return (
            np.array(positions, dtype=np.float64),
            np.array(mf_values, dtype=np.float64),
            insertion_data,
        )

    # -- needle insertion --------------------------------------------------
    def _insert_needle(
        self,
        layer_index: int,
        depth_into_layer: float,
        needle_material: str,
    ) -> None:
        """
        Split the layer at `layer_index` and insert a seed layer.

        Replaces layer_list[layer_index] with:
            [top_portion, needle_seed, bottom_portion]
        """
        layer_list = self.structure.layer_list
        original = layer_list[layer_index]
        d_total = original.thickness
        d_bot = d_total - depth_into_layer

        top = original.clone()
        top.thickness = depth_into_layer

        seed = Layer(
            thickness=self.cfg.needle_seed_thickness,
            material_name=needle_material,
            optimize=True,
            needle=True,
        )

        bot = original.clone()
        bot.thickness = d_bot

        layer_list[layer_index: layer_index + 1] = [top, seed, bot]

    # -- main synthesis loop -----------------------------------------------
    def run(
        self,
        callback: Optional[
            Callable[[NeedleCycleResult, np.ndarray, np.ndarray], None]
        ] = None,
    ) -> List[NeedleCycleResult]:
        """
        Execute the full needle synthesis algorithm.

        Parameters
        ----------
        callback : callable, optional
            Called after each cycle with ``(cycle_result, scan_positions,
            scan_mf_values)``.  Useful for plotting or logging.

        Returns
        -------
        list of NeedleCycleResult
            History of each needle insertion cycle.
        """
        history: List[NeedleCycleResult] = []

        # Initial optimisation
        logger.info("Initial optimisation...")
        mf = self.optimize_thicknesses()
        logger.info(
            "Start MF: %.6f (Layers: %d)", mf, self.layer_count
        )

        for cycle in range(self.cfg.max_needles):
            # 1. Scan P-function
            positions, mf_values, instructions = self.compute_p_function()

            if mf_values.size == 0:
                logger.info("Stack too thin for needle insertion.")
                break

            # 2. Find best insertion point (minimum MF)
            best_idx = int(np.argmin(mf_values))
            best_mf = float(mf_values[best_idx])

            # 3. Check for convergence
            current_mf = self.evaluate_merit()
            improvement = current_mf - best_mf

            logger.info(
                "Cycle %d: best needle MF=%.6f  (improvement=%.2e)",
                cycle + 1, best_mf, improvement,
            )

            result = NeedleCycleResult(
                cycle=cycle + 1,
                merit_before=current_mf,
                merit_after=current_mf,  # updated below if we insert
                best_scan_mf=best_mf,
                layer_count=self.layer_count,
            )

            # Fire callback with scan data (e.g. for plotting)
            if callback is not None:
                callback(result, positions, mf_values)

            if improvement < self.cfg.convergence_threshold:
                logger.info("Convergence reached — stopping.")
                history.append(result)
                break

            # 4. Insert needle
            layer_idx, depth, needle_mat = instructions[best_idx]
            self._insert_needle(layer_idx, depth, needle_mat)
            result.insertion_index = layer_idx
            result.insertion_material = needle_mat

            # 5. Re-optimise
            logger.info("Optimising new stack...")
            new_mf = self.optimize_thicknesses()
            result.merit_after = new_mf
            result.layer_count = self.layer_count
            logger.info(
                "--> Optimised MF: %.6f  (Layers: %d)",
                new_mf, self.layer_count,
            )

            history.append(result)

        return history

    # -- QWOT helper -------------------------------------------------------
    def _qwot_nm(
        self,
        material_name: str,
        reference_wl: float,
    ) -> float:
        """
        Compute one quarter-wave optical thickness (QWOT) in nm for
        *material_name* at *reference_wl*.

            QWOT = λ₀ / (4 · n(λ₀))

        Uses the real part of the complex index at the wavelength grid
        point closest to *reference_wl*.
        """
        provider = self.structure.materials
        if provider is None:
            raise ValueError("No MaterialProvider on structure.")

        nk = provider.get_nk(material_name)
        idx = int(np.argmin(np.abs(self.wavls - reference_wl)))
        n_real = float(nk[idx].real)

        if n_real <= 0:
            raise ValueError(
                f"Material '{material_name}' has non-positive n={n_real} "
                f"at λ={self.wavls[idx]:.1f} nm."
            )
        return reference_wl / (4.0 * n_real)

    def thickness_to_qwot(
        self,
        thickness_nm: float,
        material_name: str,
        reference_wl: float,
    ) -> float:
        """Convert a physical thickness (nm) to QWOT units."""
        return thickness_nm / self._qwot_nm(material_name, reference_wl)

    def qwot_to_thickness(
        self,
        qwot: float,
        material_name: str,
        reference_wl: float,
    ) -> float:
        """Convert a QWOT value to physical thickness (nm)."""
        return qwot * self._qwot_nm(material_name, reference_wl)

    # ══════════════════════════════════════════════════════════════════════
    # POST-SYNTHESIS: Design cleanup
    # ══════════════════════════════════════════════════════════════════════

    def merge_adjacent_layers(self) -> int:
        """
        Merge consecutive film layers of the same material.

        When the needle algorithm splits a host layer and the optimiser
        drives the seed to zero, two same-material fragments may remain
        side by side.  This method combines them into a single layer
        whose thickness is the sum of the two, preserving the properties
        (optimize, needle, roughness, etc.) of the *first* layer in each
        merged pair.

        Only operates on film layers (ambient and substrate are untouched).

        Returns
        -------
        int
            Number of merges performed (each merge reduces layer count
            by one).
        """
        layer_list = self.structure.layer_list
        if len(layer_list) < 4:
            # Need at least ambient + 2 films + substrate
            return 0

        merged_films: List[Layer] = []
        films = layer_list[1:-1]
        merge_count = 0
        i = 0

        while i < len(films):
            current = films[i]

            # Accumulate consecutive same-material layers
            combined_d = current.thickness
            j = i + 1
            while j < len(films) and films[j].material == current.material:
                combined_d += films[j].thickness
                j += 1
                merge_count += 1

            # Keep the first layer's properties, update thickness
            result_layer = current.clone()
            result_layer.thickness = combined_d
            merged_films.append(result_layer)
            i = j

        self.structure.layer_list = (
            [layer_list[0]] + merged_films + [layer_list[-1]]
        )
        return merge_count

    def _evaluate_removal_impact(self, film_index: int) -> float:
        """
        Trial-remove a single film layer and return the resulting MF.

        The layer is removed from a temporary copy of the structure so
        the real stack is not mutated.

        Parameters
        ----------
        film_index : int
            Index into ``film_layers`` (0-based).

        Returns
        -------
        float
            Merit function value of the stack with the layer removed.
        """
        layer_list = self.structure.layer_list
        global_idx = film_index + 1  # +1 for ambient

        temp_layers = [L.clone() for i, L in enumerate(layer_list)
                       if i != global_idx]
        temp_struct = Loom_Structure(
            layer_list=temp_layers,
            group_dict=self.structure.group_dict,
            materials=self.structure.materials,
        )
        return self.evaluate_merit(temp_struct)

    def remove_thin_layers(
        self,
        min_thickness: Optional[float] = None,
        max_removals: Optional[int] = None,
    ) -> int:
        """
        Iteratively remove the lowest-impact thin film layers.

        On each iteration the method:
          1. Identifies all film layers below *min_thickness*.
          2. Trial-removes each candidate and evaluates the resulting MF.
          3. Removes the candidate whose removal causes the least
             increase (or greatest decrease) in MF.
          4. Re-optimises the remaining stack.
          5. Repeats until no thin layers remain or *max_removals* is
             exhausted.

        This conservative approach ensures the layer with the least
        structural importance is always removed first, and the stack is
        re-balanced after every removal so later decisions are based on
        fresh merit evaluations.

        Parameters
        ----------
        min_thickness : float, optional
            Thickness threshold in nm.  Defaults to
            ``self.cfg.min_layer_thickness``.
        max_removals : int, optional
            Maximum number of layers to remove.  If None, all layers
            below threshold may be removed (no cap).

        Returns
        -------
        int
            Total number of layers removed.
        """
        threshold = (
            min_thickness if min_thickness is not None
            else self.cfg.min_layer_thickness
        )
        budget = max_removals if max_removals is not None else self.layer_count
        removed = 0

        while removed < budget:
            films = self.film_layers

            # 1. Collect candidates (thin layers)
            candidates: List[int] = [
                i for i, L in enumerate(films)
                if L.thickness < threshold
            ]

            if not candidates:
                break

            # 2. Trial-remove each candidate, score by MF impact
            #    Lower MF = better design → we want the removal that
            #    yields the lowest post-removal MF.
            best_idx: Optional[int] = None
            best_mf = float("inf")

            for film_idx in candidates:
                trial_mf = self._evaluate_removal_impact(film_idx)
                if trial_mf < best_mf:
                    best_mf = trial_mf
                    best_idx = film_idx

            if best_idx is None:
                break

            # 3. Perform the removal
            global_idx = best_idx + 1  # +1 for ambient
            removed_layer = self.structure.layer_list[global_idx]
            logger.info(
                "Removing layer %d (%s, d=%.2f nm) — trial MF=%.6f",
                best_idx, removed_layer.material,
                removed_layer.thickness, best_mf,
            )
            del self.structure.layer_list[global_idx]
            removed += 1

            # 4. Re-optimise after removal
            if self.layer_count > 0:
                self.optimize_thicknesses()

        return removed

    def cleanup_design(
        self,
        min_thickness: Optional[float] = None,
        max_removals: Optional[int] = None,
        reoptimize: bool = True,
    ) -> CleanupResult:
        """
        Full post-synthesis cleanup: merge → prune → (optional) re-optimise.

        This should be called after ``run()`` completes.  The sequence is:

        1. **Merge** adjacent same-material layers (a common artefact of
           needle insertion where the seed collapses to zero and leaves
           two fragments of the host material side-by-side).
        2. **Remove** up to *max_removals* film layers thinner than
           *min_thickness*, one at a time in order of least merit
           impact, with re-optimisation after each removal.
        3. **Merge again** (removal can expose new same-material neighbours).
        4. **Re-optimise** thicknesses on the simplified stack (optional but
           highly recommended — the simplified topology often converges to
           a better local minimum).

        Parameters
        ----------
        min_thickness : float, optional
            Layers thinner than this (nm) are removed.  Defaults to
            ``self.cfg.min_layer_thickness``.
        max_removals : int, optional
            Maximum number of thin layers to remove per cleanup call.
            If None, all thin layers may be removed (no cap).
        reoptimize : bool
            If True (default), run ``optimize_thicknesses()`` after the
            full cleanup sequence.

        Returns
        -------
        CleanupResult
            Before/after metrics.

        Example
        -------
        >>> history = synth.run()
        >>> cleanup = synth.cleanup_design(min_thickness=3.0, max_removals=2)
        >>> print(f"Removed {cleanup.layers_removed_thin} thin layers, "
        ...       f"merged {cleanup.layers_merged}, MF: {cleanup.merit_after:.6f}")
        """
        mf_before = self.evaluate_merit()
        n_before = self.layer_count

        # Pass 1: merge adjacent same-material
        merges_1 = self.merge_adjacent_layers()

        # Pass 2: iterative impact-ranked removal of thin layers
        removed = self.remove_thin_layers(min_thickness, max_removals)

        # Pass 3: merge again (pruning can create new neighbours)
        merges_2 = self.merge_adjacent_layers()

        total_merged = merges_1 + merges_2

        logger.info(
            "Cleanup: merged %d pairs, removed %d thin layers "
            "(%d → %d film layers).",
            total_merged, removed, n_before, self.layer_count,
        )

        # Final re-optimise on the cleaned stack
        if reoptimize and self.layer_count > 0:
            logger.info("Re-optimising after cleanup...")
            mf_after = self.optimize_thicknesses()
        else:
            mf_after = self.evaluate_merit()

        logger.info(
            "Cleanup complete: MF %.6f → %.6f", mf_before, mf_after,
        )

        return CleanupResult(
            merit_before=mf_before,
            merit_after=mf_after,
            layers_before=n_before,
            layers_after=self.layer_count,
            layers_removed_thin=removed,
            layers_merged=total_merged,
        )

    # ══════════════════════════════════════════════════════════════════════
    # POST-SYNTHESIS: QWOT-based thickness inflation
    # ══════════════════════════════════════════════════════════════════════

    def _evaluate_inflate_impact(
        self,
        film_index: int,
        addon_qwot: float,
        reference_wl: float,
    ) -> float:
        """
        Trial-inflate a single film layer and return the resulting MF.

        The inflation is applied to a temporary copy of the structure
        so the real stack is not mutated.

        Parameters
        ----------
        film_index : int
            Index into ``film_layers`` (0-based).
        addon_qwot : float
            QWOT addon to apply.
        reference_wl : float
            Reference wavelength (nm).

        Returns
        -------
        float
            Merit function value with this single layer inflated.
        """
        layer_list = self.structure.layer_list
        temp_layers = [L.clone() for L in layer_list]
        global_idx = film_index + 1  # +1 for ambient

        mat = temp_layers[global_idx].material
        qwot_nm = self._qwot_nm(mat, reference_wl)
        delta = addon_qwot * qwot_nm
        temp_layers[global_idx].thickness = max(
            0.0, temp_layers[global_idx].thickness + delta
        )

        temp_struct = Loom_Structure(
            layer_list=temp_layers,
            group_dict=self.structure.group_dict,
            materials=self.structure.materials,
        )
        return self.evaluate_merit(temp_struct)

    def inflate_design(
        self,
        addon_qwot: float,
        reference_wl: float,
        *,
        max_layers: Optional[int] = None,
        reoptimize: bool = True,
    ) -> InflateResult:
        """
        Inflate the most impactful film layers by a QWOT addon, then
        re-optimise.

        Rather than inflating every layer, this method scores each film
        layer by the merit-function improvement that results from
        trial-inflating it individually.  The top *max_layers* layers
        (those whose inflation yields the lowest MF) are then inflated
        simultaneously and the stack is re-optimised.

        Each selected layer's thickness is increased by:

            Δd = addon_qwot × λ₀ / (4 · n_layer(λ₀))

        where n_layer is the real refractive index of that layer's
        material at *reference_wl*.

        Parameters
        ----------
        addon_qwot : float
            Number of QWOT to add to each selected layer.  Can be
            fractional (e.g. 0.5 QWOT).  Negative values thin the
            layers; the thickness is clamped to zero.
        reference_wl : float
            Reference wavelength (nm) used to compute QWOT.
        max_layers : int, optional
            Maximum number of layers to inflate.  If None, all film
            layers are inflated (original behaviour).  When set, layers
            are ranked by their individual merit-function impact and
            only the top *max_layers* are selected.
        reoptimize : bool
            If True (default), run ``optimize_thicknesses()`` after
            inflation.

        Returns
        -------
        InflateResult
            Before/after metrics.

        Example
        -------
        >>> # Inflate the 3 most impactful layers by 2 QWOT
        >>> inflate = synth.inflate_design(
        ...     addon_qwot=2.0, reference_wl=550.0, max_layers=3,
        ... )
        """
        mf_before = self.evaluate_merit()
        films = self.film_layers
        total_before = sum(L.thickness for L in films)

        # Determine which layers to inflate
        if max_layers is not None and max_layers < len(films):
            # Score every layer by trial-inflation MF
            scored: List[Tuple[int, float]] = []
            for i in range(len(films)):
                trial_mf = self._evaluate_inflate_impact(
                    i, addon_qwot, reference_wl,
                )
                scored.append((i, trial_mf))

            # Sort by MF ascending (best = lowest MF first)
            scored.sort(key=lambda t: t[1])
            inflate_indices = [idx for idx, _ in scored[:max_layers]]

            logger.info(
                "Inflate: scored %d layers, selected %d: %s",
                len(films), len(inflate_indices),
                [(i, films[i].material, f"{scored_mf:.6f}")
                 for i, scored_mf in scored[:max_layers]],
            )
        else:
            inflate_indices = list(range(len(films)))

        # Apply inflation to selected layers
        for idx in inflate_indices:
            layer = films[idx]
            qwot_nm = self._qwot_nm(layer.material, reference_wl)
            delta = addon_qwot * qwot_nm
            new_d = max(0.0, layer.thickness + delta)

            logger.debug(
                "  Layer %d (%s): %.2f nm + %.2f nm (%.2f QWOT) = %.2f nm",
                idx, layer.material, layer.thickness, delta,
                addon_qwot, new_d,
            )
            layer.thickness = new_d

        total_after_raw = sum(L.thickness for L in films)

        logger.info(
            "Inflated %d/%d layers by %.2f QWOT @ λ=%.1f nm: "
            "total thickness %.1f → %.1f nm",
            len(inflate_indices), len(films),
            addon_qwot, reference_wl, total_before, total_after_raw,
        )

        # Re-optimise
        if reoptimize:
            logger.info("Re-optimising after inflation...")
            mf_after = self.optimize_thicknesses()
        else:
            mf_after = self.evaluate_merit()

        total_after_opt = sum(L.thickness for L in self.film_layers)

        logger.info(
            "Inflate complete: MF %.6f → %.6f", mf_before, mf_after,
        )

        return InflateResult(
            merit_before=mf_before,
            merit_after=mf_after,
            total_thickness_before=total_before,
            total_thickness_after=total_after_opt,
            layer_count=self.layer_count,
            addon_qwot=addon_qwot,
            reference_wavelength=reference_wl,
        )

    def round_to_qwot(
        self,
        reference_wl: float,
        resolution: float = 0.25,
        *,
        reoptimize: bool = True,
    ) -> float:
        """
        Snap every film-layer thickness to the nearest QWOT multiple.

        Useful for generating "monitoring-friendly" designs where each
        layer is an integer (or half-integer) number of quarter-waves,
        simplifying optical monitoring during deposition.

        Parameters
        ----------
        reference_wl : float
            Reference wavelength (nm) for QWOT calculation.
        resolution : float
            QWOT granularity to snap to.  Default 0.25 means thicknesses
            are rounded to the nearest quarter-QWOT (i.e. to 0.25, 0.50,
            0.75, 1.00, 1.25, … QWOT).  Use 1.0 for whole-QWOT snapping,
            0.5 for half-QWOT, etc.
        reoptimize : bool
            If True (default), run ``optimize_thicknesses()`` after
            rounding to recover merit.

        Returns
        -------
        float
            Post-rounding merit function value.

        Example
        -------
        >>> synth.round_to_qwot(reference_wl=550.0, resolution=0.5)
        >>> synth.print_design(reference_wl=550.0)
        """
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        films = self.film_layers
        for layer in films:
            qwot_nm = self._qwot_nm(layer.material, reference_wl)
            step = resolution * qwot_nm

            # Round to nearest step, minimum one step (avoid zero)
            rounded = max(step, round(layer.thickness / step) * step)
            layer.thickness = rounded

        if reoptimize:
            logger.info("Re-optimising after QWOT rounding...")
            return self.optimize_thicknesses()

        return self.evaluate_merit()

    # ══════════════════════════════════════════════════════════════════════
    # POST-SYNTHESIS: Full pipeline convenience
    # ══════════════════════════════════════════════════════════════════════

    def run_full_pipeline(
        self,
        *,
        cleanup_min_thickness: float = 3.0,
        cleanup_max_removals: Optional[int] = None,
        inflate_qwot: Optional[float] = None,
        inflate_reference_wl: float = 550.0,
        inflate_max_layers: Optional[int] = None,
        callback: Optional[
            Callable[[NeedleCycleResult, np.ndarray, np.ndarray], None]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: run needle synthesis → cleanup → (optional)
        inflate, all in one call.

        Parameters
        ----------
        cleanup_min_thickness : float
            Minimum layer thickness (nm) for the cleanup pass.
        cleanup_max_removals : int, optional
            Maximum thin layers to remove per cleanup.
        inflate_qwot : float, optional
            If provided, run ``inflate_design`` with this QWOT addon
            after cleanup.
        inflate_reference_wl : float
            Reference wavelength for inflation (default 550 nm).
        inflate_max_layers : int, optional
            Maximum layers to inflate (ranked by merit impact).
        callback : callable, optional
            Forwarded to ``run()``.

        Returns
        -------
        dict
            ``{'needle_history': [...], 'cleanup': CleanupResult,
              'inflate': InflateResult | None, 'final_mf': float}``
        """
        # 1. Needle synthesis
        needle_history = self.run(callback=callback)

        # 2. Cleanup
        cleanup = self.cleanup_design(
            min_thickness=cleanup_min_thickness,
            max_removals=cleanup_max_removals,
            reoptimize=True,
        )

        # 3. Optional inflation
        inflate = None
        if inflate_qwot is not None and inflate_qwot != 0.0:
            inflate = self.inflate_design(
                addon_qwot=inflate_qwot,
                reference_wl=inflate_reference_wl,
                max_layers=inflate_max_layers,
                reoptimize=True,
            )

        return {
            "needle_history": needle_history,
            "cleanup": cleanup,
            "inflate": inflate,
            "final_mf": self.evaluate_merit(),
        }

    # -- utility -----------------------------------------------------------
    def get_final_design(self) -> List[Tuple[str, float]]:
        """
        Return the current film stack as a list of (material, thickness) tuples
        (excluding ambient and substrate).
        """
        return [(L.material, L.thickness) for L in self.film_layers]

    def print_design(
        self,
        reference_wl: Optional[float] = None,
    ) -> None:
        """
        Print a human-readable summary of the current design.

        Parameters
        ----------
        reference_wl : float, optional
            If provided, an additional column shows each layer's
            thickness in QWOT at this reference wavelength.
        """
        films = self.film_layers
        show_qwot = reference_wl is not None

        if show_qwot:
            hdr = (
                f"{'#':<4} {'Material':<12} {'d (nm)':<14} "
                f"{'QWOT':<10} {'QWOT×λ₀=' + f'{reference_wl:.0f}nm'}"
            )
            sep_len = len(hdr) + 2
        else:
            hdr = f"{'#':<4} {'Material':<12} {'Thickness (nm)':<16}"
            sep_len = 36

        print(f"\nDesign ({len(films)} film layers):")
        print(hdr)
        print("-" * sep_len)

        total_nm = 0.0
        total_qwot = 0.0

        for i, layer in enumerate(films, 1):
            total_nm += layer.thickness
            if show_qwot:
                q = self.thickness_to_qwot(
                    layer.thickness, layer.material, reference_wl
                )
                total_qwot += q
                print(
                    f"{i:<4} {layer.material:<12} {layer.thickness:>10.2f}    "
                    f"{q:>8.3f}"
                )
            else:
                print(f"{i:<4} {layer.material:<12} {layer.thickness:>12.2f}")

        print("-" * sep_len)
        if show_qwot:
            print(
                f"{'':4} {'Total':<12} {total_nm:>10.2f}    "
                f"{total_qwot:>8.3f}"
            )
        else:
            print(f"{'':4} {'Total':<12} {total_nm:>12.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Usage example
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    # ── Materials ─────────────────────────────────────────────────────────
    wavls = np.linspace(400, 800, 200)
    n_wavs = len(wavls)

    provider = ArrayMaterialProvider({
        "air": np.full(n_wavs, 1.00 + 0j),
        "H":   np.full(n_wavs, 2.35 + 0j),   # TiO2-like
        "L":   np.full(n_wavs, 1.46 + 0j),   # SiO2-like
        "sub": np.full(n_wavs, 1.52 + 0j),   # BK7-like
    })

    # ── Structure (seed: single H layer) ──────────────────────────────────
    structure = Loom_Structure(
        materials=provider,
        layer_list=[
            Layer(thickness=0.0, material_name="air",
                  optimize=False, needle=False, layer_typ=0),
            Layer(thickness=150.0, material_name="H",
                  optimize=True, needle=True),
            Layer(thickness=0.0, material_name="sub",
                  optimize=False, needle=False, layer_typ=0),
        ],
    )

    # ── Target: broadband anti-reflection (R = 0) ─────────────────────────
    targets = TargetCollection()
    targets.add(SpectralTarget(
        wavelengths=wavls,
        values=np.zeros(n_wavs, dtype=np.float64),
        tolerances=np.full(n_wavs, 0.01, dtype=np.float64),
        angle=0.0,
        polarization="u",
        spectral="R",
        kind="e",
    ))
    target_weaver = targets.build_weaver()

    # ── Synthesise ────────────────────────────────────────────────────────
    config = NeedleConfig(max_needles=6, scan_step_nm=2.0)

    synth = NeedleSynthesizer(
        structure=structure,
        wavls=wavls,
        target_weaver=target_weaver,
        contrasting_materials={"H": "L", "L": "H"},
        config=config,
    )

    # Callback: save P-function plots
    fig_p, axes_p = None, []

    def on_cycle(result, positions, mf_values):
        """Optional: collect P-function data for plotting."""
        pass  # Replace with plt.plot(...) for interactive use

    history = synth.run(callback=on_cycle)

    # ── Post-synthesis cleanup ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POST-SYNTHESIS CLEANUP")
    print("=" * 60)

    cleanup = synth.cleanup_design(min_thickness=3.0, max_removals=3)
    print(f"  Merged {cleanup.layers_merged} adjacent pairs")
    print(f"  Removed {cleanup.layers_removed_thin} thin layers (max 3)")
    print(f"  Layers: {cleanup.layers_before} → {cleanup.layers_after}")
    print(f"  MF:     {cleanup.merit_before:.6f} → {cleanup.merit_after:.6f}")

    synth.print_design(reference_wl=550.0)

    # ── Inflate by 2 QWOT and re-optimise ─────────────────────────────────
    print("\n" + "=" * 60)
    print("QWOT INFLATION (+2 QWOT @ 550 nm, top 2 layers)")
    print("=" * 60)

    inflate = synth.inflate_design(
        addon_qwot=2.0,
        reference_wl=550.0,
        max_layers=2,
        reoptimize=True,
    )
    print(f"  Total thickness: {inflate.total_thickness_before:.1f}"
          f" → {inflate.total_thickness_after:.1f} nm")
    print(f"  MF: {inflate.merit_before:.6f} → {inflate.merit_after:.6f}")

    synth.print_design(reference_wl=550.0)

    # ── QWOT rounding (optional) ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QWOT ROUNDING (0.5 QWOT resolution)")
    print("=" * 60)

    mf_rounded = synth.round_to_qwot(
        reference_wl=550.0, resolution=0.5, reoptimize=True,
    )
    print(f"  MF after rounding + re-opt: {mf_rounded:.6f}")
    synth.print_design(reference_wl=550.0)

    # ── Final evaluation & plot ───────────────────────────────────────────
    sa = structure.get_solver_inputs()
    solver = LoomScatterMatrix(
        sa.indices, sa.thicknesses, sa.incoherent_flags,
        sa.rough_types, sa.rough_vals,
        wavls, np.array([0.0]), theta_is_radians=True,
    )
    res = solver.compute_RT(mode="u")
    final_R = res["Ru"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavls, final_R, "k-", linewidth=2, label="Final Design")
    ax.axhline(0.0, color="r", linestyle="--", alpha=0.5, label="Target")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"BBAR Needle Synthesis ({synth.layer_count} film layers)")
    ax.set_ylim(-0.01, 0.10)
    ax.legend()
    fig.tight_layout()
    fig.savefig("needle_bbar_result.png", dpi=150)
    print(f"\nPlot saved to needle_bbar_result.png")

    synth.print_design(reference_wl=550.0)

    # Print convergence history
    print("\nNeedle convergence history:")
    for r in history:
        print(
            f"  Cycle {r.cycle}: MF {r.merit_before:.6f} → {r.merit_after:.6f} "
            f"({r.layer_count} layers)"
        )

    # ── Alternative: run_full_pipeline does everything in one call ─────────
    # results = synth.run_full_pipeline(
    #     cleanup_min_thickness=3.0,
    #     inflate_qwot=2.0,
    #     inflate_reference_wl=550.0,
    # )
    # print(f"Final MF: {results['final_mf']:.6f}")
