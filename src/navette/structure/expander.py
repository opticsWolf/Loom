# -*- coding: utf-8 -*-
"""Flatten layers/groups/materials into engine :class:`SolverArrays`.

The single (:class:`_LayerExpander`) traversal applies group scaling,
error draws, roughness/interface handling and index interpolation, so
``get_solver_inputs`` stays a one-liner. The Looyenga interface mix uses
the native kernel when built, else a NumPy fallback.

Two-phase design (MC-exact mirrors): phase 1 resolves every entry's bulk
(group scaling, no RNG); phase 2 emits rows in traversal order with
error draws. Each entry's boundary plane takes its flags from the
traversal predecessor when both are inverted (the mirror-image rule —
exact for whole-chain reversal), and from itself otherwise; an inverted
run's incident edge is clean, exactly like a forward chain start, so a
neighbor block's edge flags never teleport into the run. The slice width
— group summand and error draw included — is computed once per plane and
carved from the flag owner's bulk (donor side in inverted mode), so
forward and mirror receive bit-identical planes and bulk splits on the
deterministic path and identically-distributed draws under Monte-Carlo.
Forward (non-inverted) output is bit-identical to the legacy
single-pass traversal.
"""
from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

try:
  from navette._materials import ema_looyenga as _looyenga_eps
  from navette._materials import eps_to_nk as _eps_to_nk
  _NATIVE_EMA = True
except ImportError:  # pragma: no cover - native not built; numpy fallback below
  _looyenga_eps = None  # type: ignore[assignment]
  _eps_to_nk = None  # type: ignore[assignment]
  _NATIVE_EMA = False


def _looyenga_fallback(
  n_i: np.ndarray, n_h: np.ndarray, f: float
) -> np.ndarray:
  """Landau-Lifshitz-Looyenga mixing (pure numpy; elementwise)."""
  cbrt = (n_i * n_i) ** (1.0 / 3.0) * f + (n_h * n_h) ** (1.0 / 3.0) * (1.0 - f)
  return cbrt ** 3.0


def looyenga_n(
  n_i: np.ndarray, n_h: np.ndarray, f: float
) -> np.ndarray:
  """Interface index from LLL mixing (native kernel when built, else numpy).

  The kernel returns effective permittivity (documented native contract);
  the sqrt to refractive index happens here, at the single insertion
  point — the solver's `indices` column carries n, never eps.
  """
  if _NATIVE_EMA:
    assert _looyenga_eps is not None and _eps_to_nk is not None
    eps = _looyenga_eps(
      np.ascontiguousarray(n_i, dtype=np.complex128),
      np.ascontiguousarray(n_h, dtype=np.complex128),
      float(f),
    )
    return _eps_to_nk(np.ascontiguousarray(eps, dtype=np.complex128))
  return np.sqrt(_looyenga_fallback(
    np.asarray(n_i, dtype=np.complex128),
    np.asarray(n_h, dtype=np.complex128),
    float(f),
  ))


def looyenga_eps(
  n_i: np.ndarray, n_h: np.ndarray, f: float
) -> np.ndarray:
  """Deprecated alias of :func:`looyenga_n` (kept for compatibility)."""
  return looyenga_n(n_i, n_h, f)

from .types import COMPLEX_TYPE, FLOAT_TYPE, INT_TYPE, ErrorMask, RoughnessType, SolverArrays
from .materials import MaterialProvider
from .models import Group, Layer

_DEFAULT_GROUP = Group("_default_")
_NO_ROUGHNESS = int(RoughnessType.NONE)

class _LayerExpander:
    """Internal stack flattener (see module docstring)."""
    @staticmethod
    def expand(
        layers: Iterator[Tuple[Layer, bool]],
        materials: MaterialProvider,
        group_dict: Dict[str, Group],
        *,
        apply_errors: bool = False,
        rng: Optional[np.random.Generator] = None,
        return_spans: bool = False,
    ) -> Union[SolverArrays, Tuple[SolverArrays, List[Tuple[int, int, int]]]]:
        seq = list(layers)
        if not seq:
            raise ValueError("_LayerExpander.expand: No layers to expand. Empty layer sequence provided.")
        get_group = group_dict.get
        m = len(seq)

        # ---- Phase 1: deterministic bulk resolution (no RNG). ----
        bulk_nk: List[np.ndarray] = []
        bulk_t: List[float] = []
        owner_of: List[Optional[int]] = []  # entry holding this plane's flags
        for k, (layer, inv) in enumerate(seq):
            group = get_group(layer.material, _DEFAULT_GROUP)
            base_nk = materials.get_nk(layer.material)
            # Independent n/k scaling (identity (1, 1); matches the
            # table-material convention in navette-materials).
            bulk_nk.append(base_nk.real * group.n_factor + 1j * (base_nk.imag * group.k_factor)
                           if (group.n_factor != 1.0 or group.k_factor != 1.0) else base_nk)
            bulk_t.append(layer.thickness * group.thick_factor + group.thick_summand)
            # Plane flags: own layer forward (active for k > 0); traversal
            # predecessor when both entries are inverted (mirror rule
            # inside a run); a run's incident edge is clean, as is k == 0.
            if inv and k > 0 and seq[k - 1][1]:
                owner_of.append(k - 1)
            else:
                owner_of.append(k if (not inv and k > 0) else None)

        # ---- Phase 2: emission in traversal order (RNG draws here). ----
        # Per-entry bulk row spans into col_thick, so a donor-side carve
        # can rescale already-emitted rows uniformly (mirror of the
        # forward pre-split carve).
        col_thick: List[float] = []
        col_nk: List[Union[complex, np.ndarray]] = []
        col_coh: List[bool] = []
        col_r_val: List[float] = []
        col_r_type: List[int] = []
        spans: List[Tuple[int, int]] = []
        bulk_spans: List[Tuple[int, int]] = []  # carve ranges (bulk rows only)
        err_nk: List[np.ndarray] = []  # post-error ungraded bulk nk
        err_t: List[float] = []  # post-error bulk totals (pre-carve)
        prev_eff_nk: Optional[np.ndarray] = None

        for k, (layer, inv) in enumerate(seq):
            group = get_group(layer.material, _DEFAULT_GROUP)
            layer_nk = bulk_nk[k]
            layer_thickness = bulk_t[k]
            if apply_errors:
                if group.error_mask[ErrorMask.THICKNESS]:
                    layer_thickness = group.thickness_error(layer_thickness, rng=rng)
                if group.error_mask[ErrorMask.N_REAL] or group.error_mask[ErrorMask.N_IMAG]:
                    n_part, k_part = layer_nk.real, layer_nk.imag
                    if group.error_mask[ErrorMask.N_REAL]:
                        n_part = np.maximum(0.0, Group._apply_error(n_part, group.n_error_type, group.n_error_params, rng=rng))
                    if group.error_mask[ErrorMask.N_IMAG]:
                        k_part = Group._apply_error(k_part, group.k_error_type, group.k_error_params, rng=rng)
                    layer_nk = n_part + 1j * k_part
            layer_thickness = max(0.0, layer_thickness)
            err_nk.append(layer_nk)
            err_t.append(layer_thickness)
            o = owner_of[k]

            # Bulk roughness: the plane exists at every boundary, with or
            # without a slice — so inverted entries take the owner's value
            # whenever a predecessor exists (k == 0 is the clean incident
            # plane). Forward keeps the legacy own-value rule verbatim
            # (including the k == 0 ambient-row value the solver ignores).
            # Owner-group draws: the boundary belongs to the donor.
            # Drawn before the slice below so the RNG stream matches the
            # legacy per-layer order (thick, nk, rough, iface, inhg).
            if inv:
                if o is not None:
                    olayer = seq[o][0]
                    ogroup = get_group(olayer.material, _DEFAULT_GROUP)
                    current_roughness = max(0.0, olayer.roughness + ogroup.roughness_summand)
                    rtype = int(olayer.rough_type)
                    if apply_errors and ogroup.error_mask[ErrorMask.ROUGHNESS]:
                        current_roughness = ogroup.sr_roughness_error(current_roughness, layer_thickness, rng=rng)
                else:
                    current_roughness, rtype = 0.0, _NO_ROUGHNESS
            else:
                current_roughness = max(0.0, layer.roughness + group.roughness_summand)
                rtype = int(layer.rough_type)
                if apply_errors and group.error_mask[ErrorMask.ROUGHNESS]:
                    current_roughness = group.sr_roughness_error(current_roughness, layer_thickness, rng=rng)

            # Entry span starts here so a leading interface slice resolves
            # to its carrier's logical layer (see return_spans contract).
            start = len(col_thick)

            # Plane slice (flag owner's group governs summand + draws).
            t_interface = 0.0
            if o is not None:
                olayer = seq[o][0]
                ogroup = get_group(olayer.material, _DEFAULT_GROUP)
                if olayer.interface:
                    t_interface = olayer.interface_thickness + ogroup.interface_summand
                    if apply_errors and ogroup.error_mask[ErrorMask.INTERFACE]:
                        t_interface = ogroup.interface_error(t_interface, olayer.thickness, rng=rng)
                    # Carve follows the flag owner's material: own bulk
                    # forward, donor bulk (already buffered) inverted. Mix
                    # partners are the two sides of the plane: owner bulk
                    # against the previous entry forward, against the
                    # carrier bulk inverted (symmetric mix either way).
                    carve_total = layer_thickness if o == k else err_t[o]
                    t_interface = min(t_interface, carve_total)
                    if o == k:
                        layer_thickness -= t_interface
                        interface_nk = looyenga_n(layer_nk, prev_eff_nk, 0.5)
                    else:
                        start_o, end_o = bulk_spans[o]
                        if carve_total > 0.0 and end_o > start_o:
                            scale = (carve_total - t_interface) / carve_total
                            for j in range(start_o, end_o):
                                col_thick[j] *= scale
                        interface_nk = looyenga_n(err_nk[o], err_nk[k], 0.5)

                    col_thick.append(t_interface)
                    col_nk.append(interface_nk)
                    col_coh.append(True)
                    col_r_val.append(0.0)
                    col_r_type.append(_NO_ROUGHNESS)

            bulk_start = len(col_thick)
            if layer.inhomogen and layer.sub_layer_count > 1:
                sub_div = layer.sub_layer_count
                current_delta = (layer.inh_delta + group.inh_delta_summand) * 0.5
                if apply_errors and group.error_mask[ErrorMask.INH_DELTA]:
                    current_delta = group.inh_delta_error(current_delta, rng=rng)

                factors = np.linspace(1.0 - current_delta, 1.0 + current_delta, sub_div)
                if inv: factors = factors[::-1]

                step_t = layer_thickness / sub_div
                for ix, f in enumerate(factors):
                    col_thick.append(step_t)
                    col_nk.append(layer_nk * f)
                    col_coh.append(layer.coherent)
                    col_r_val.append(current_roughness if ix == 0 else 0.0)
                    col_r_type.append(rtype if ix == 0 else _NO_ROUGHNESS)

            else:
                col_thick.append(layer_thickness)
                col_nk.append(layer_nk)
                col_coh.append(layer.coherent)
                col_r_val.append(current_roughness)
                col_r_type.append(rtype)
            spans.append((start, len(col_thick)))
            bulk_spans.append((bulk_start, len(col_thick)))

            prev_eff_nk = layer_nk

        sa = SolverArrays(
            indices=np.vstack(col_nk).astype(COMPLEX_TYPE),
            thicknesses=np.array(col_thick, dtype=FLOAT_TYPE),
            incoherent_flags=np.array([not c for c in col_coh], dtype=np.bool_),
            rough_types=np.array(col_r_type, dtype=INT_TYPE),
            rough_vals=np.array(col_r_val, dtype=FLOAT_TYPE),
        )
        if return_spans:
            # (row_start, row_end, logical_index): interface slices belong
            # to their carrier entry; logical indices count _iter_layers
            # order (repeat-aware), matching map_global_index_to_layer.
            spans_out = [(s, e, k) for k, (s, e) in enumerate(spans)]
            return sa, spans_out
        return sa