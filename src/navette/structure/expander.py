# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

from navette.materials.ema_models import looyenga_eps

from .types import COMPLEX_TYPE, FLOAT_TYPE, INT_TYPE, ErrorMask, RoughnessType, SolverArrays
from .materials import MaterialProvider
from .models import Group, Layer

_DEFAULT_GROUP = Group("_default_")
_NO_ROUGHNESS = int(RoughnessType.NONE)

class _LayerExpander:
    @staticmethod
    def expand(
        layers: Iterator[Tuple[Layer, bool]],
        materials: MaterialProvider,
        group_dict: Dict[str, Group],
        *,
        apply_errors: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> SolverArrays:
        
        col_thick: List[float] = []
        col_nk: List[Union[complex, np.ndarray]] = []
        col_coh: List[bool] = []
        col_r_val: List[float] = []
        col_r_type: List[int] = []

        get_group = group_dict.get
        prev_eff_nk: Optional[np.ndarray] = None

        for layer, inv in layers:
            mat_name = layer.material
            group = get_group(mat_name, _DEFAULT_GROUP)

            base_nk = materials.get_nk(mat_name)
            layer_nk = base_nk * group.nk_factor if (group.n_factor != 1.0 or group.k_factor != 0.0) else base_nk
            layer_thickness = layer.thickness * group.thick_factor + group.thick_summand

            current_roughness = layer.roughness

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
                if group.error_mask[ErrorMask.ROUGHNESS]:
                    current_roughness = group.sr_roughness_error(current_roughness, layer_thickness, rng=rng)

            layer_thickness = max(0.0, layer_thickness)

            if layer.interface and prev_eff_nk is not None:
                t_interface = layer.interface_thickness
                if apply_errors and group.error_mask[ErrorMask.INTERFACE]:
                    t_interface = group.interface_error(t_interface, layer.thickness, rng=rng)
                
                t_interface = min(t_interface, layer_thickness)
                layer_thickness -= t_interface

                interface_nk = looyenga_eps(layer_nk, prev_eff_nk, 0.5)

                col_thick.append(t_interface)
                col_nk.append(interface_nk)
                col_coh.append(True)
                col_r_val.append(0.0)
                col_r_type.append(_NO_ROUGHNESS)

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
                    col_r_type.append(int(layer.rough_type) if ix == 0 else _NO_ROUGHNESS)

            else:
                col_thick.append(layer_thickness)
                col_nk.append(layer_nk)
                col_coh.append(layer.coherent)
                col_r_val.append(current_roughness)
                col_r_type.append(int(layer.rough_type))

            prev_eff_nk = layer_nk

        if not col_nk:
            raise ValueError("_LayerExpander.expand: No layers to expand. Empty layer sequence provided.")

        return SolverArrays(
            indices=np.vstack(col_nk).astype(COMPLEX_TYPE),
            thicknesses=np.array(col_thick, dtype=FLOAT_TYPE),
            incoherent_flags=np.array([not c for c in col_coh], dtype=np.bool_),
            rough_types=np.array(col_r_type, dtype=INT_TYPE),
            rough_vals=np.array(col_r_val, dtype=FLOAT_TYPE),
        )