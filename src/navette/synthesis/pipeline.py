# -*- coding: utf-8 -*-
"""navette.synthesis.pipeline — needle synthesis driver.

Thin wrapper over the native pipeline (``navette._smatrix``): target
definition stays on the spectralweave surface — pass a
:class:`~navette.spectralweave.target.TargetCollection` (or a ready-made
``MeritSpec``) and a layer list; this module evaluates materials onto the
simulation grid, builds the native ``DesignStack`` + ``NeedlePipeline``,
and runs the macro-loop. All target options (kind/band/phase/weight/
count/integral, PD labels) flow through ``build_merit_spec`` untouched.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from navette._smatrix import (
    DesignStack as _NativeDesignStack,
    LayerSpec as _NativeLayerSpec,
    LmConfig as _NativeLmConfig,
    NeedleCycleConfig as _NativeNeedleCycleConfig,
    NeedlePipeline as _NativePipeline,
    PipelineConfig as _NativePipelineConfig,
    SmatrixContext as _NativeContext,
)
from navette.materials import MaterialSpec, evaluate
from navette.spectralweave.target import TargetCollection
from navette.synthesis import build_merit_spec as _build_merit_spec

__all__ = [
    "LayerSpec",
    "DesignStack",
    "SmatrixContext",
    "LmConfig",
    "PipelineConfig",
    "NeedleCycleConfig",
    "NeedlePipeline",
    "layer_from_material",
    "stack_from_layers",
    "run_needle",
]

# Re-export native classes under friendly names.
LayerSpec = _NativeLayerSpec
DesignStack = _NativeDesignStack
SmatrixContext = _NativeContext
LmConfig = _NativeLmConfig
PipelineConfig = _NativePipelineConfig
NeedleCycleConfig = _NativeNeedleCycleConfig
NeedlePipeline = _NativePipeline


def _eval_nk(material: Any, wavelengths: np.ndarray) -> np.ndarray:
    """MaterialSpec (or mapping) → complex nk on ``wavelengths``."""
    if isinstance(material, MaterialSpec):
        return np.ascontiguousarray(evaluate(material, wavelengths))
    if isinstance(material, Mapping):
        return np.ascontiguousarray(
            evaluate(MaterialSpec(model=material["model"],
                                  params=dict(material.get("params", {}))),
                     wavelengths)
        )
    arr = np.ascontiguousarray(np.asarray(material, dtype=np.complex128))
    if arr.shape != wavelengths.shape:
        raise ValueError(
            f"nk array shape {arr.shape} != wavelengths shape {wavelengths.shape}."
        )
    return arr


def layer_from_material(material: Any, thickness: float, wavelengths,
                        name: str = "", **flags) -> Any:
    """One native ``LayerSpec`` from a spec/mapping/nk array + thickness.

    ``flags``: coherent, rough_type, rough_val, optimize, needle.
    """
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    return LayerSpec(str(name), _eval_nk(material, wl), float(thickness), **flags)


def stack_from_layers(layers: Sequence[Tuple[Any, float]],
                      wavelengths, contrast: Mapping[Any, Any],
                      ambient: Tuple[Any, str] = (1.0, "air"),
                      substrate: Tuple[Any, str] = (1.52, "sub"),
                      names: Optional[Sequence[str]] = None,
                      film_flags: Optional[Dict[str, Any]] = None,
                      groups: Optional[Mapping[str, Any]] = None,
                      ) -> Tuple[Any, Dict[str, Any]]:
    """Native ``(DesignStack, contrast_map)`` from ``[(material, d_nm)]`` films.

    Materials are ``MaterialSpec`` / mappings / nk arrays / constants
    (plain names are not resolvable here — pass specs). ``names`` gives
    the film material names (default ``film0…``); ``contrast`` maps host
    name *or film index* → seed material. ``ambient``/``substrate`` are
    ``(material, name)``; always fixed, non-hosts. ``film_flags`` are
    per-film ``LayerSpec`` defaults (``optimize``/``needle``/…).
    ``groups`` maps material name → bound ``Group`` (or param dict):
    thickness/nk scaling, roughness and interface policy expand here —
    the silent-drop limitation is gone. Graded profiles are refused
    (span-aware optimization is future work).
    """
    from navette._structure import Layer as RsLayer, Group as RsGroup
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    flags = dict(coherent=True, optimize=True, needle=True)
    flags.update(film_flags or {})
    layer_flags = {k: flags.pop(k) for k in
                   ("roughness", "rough_type", "interface", "interface_thickness")
                   if k in flags}
    if "rough_val" in flags:
        layer_flags["roughness"] = flags.pop("rough_val")

    def fixed(value, name):
        nk = (np.full(wl.shape, complex(value), dtype=np.complex128)
              if isinstance(value, (int, float, complex))
              else _eval_nk(value, wl))
        return LayerSpec(str(name), nk, 0.0,
                         coherent=True, optimize=False, needle=False)

    amb = fixed(ambient[0], ambient[1] if len(ambient) > 1 else "air")
    sub = fixed(substrate[0], substrate[1] if len(substrate) > 1 else "sub")
    names = list(names) if names is not None else [f"film{i}" for i in range(len(layers))]
    if len(names) != len(layers):
        raise ValueError("names length must match layers length.")
    if len(set(names)) != len(names):
        raise ValueError("film names must be unique (they key the nk table).")
    design, nk_map = [], {}
    for (mat, d), nm in zip(layers, names):
        nk_map[str(nm)] = _eval_nk(mat, wl)
        design.append(RsLayer(float(d), str(nm), **flags, **layer_flags))
    gmap = {}
    for name, g in (groups or {}).items():
        gmap[str(name)] = g if isinstance(g, RsGroup) else RsGroup(str(name), **dict(g))
    stack = DesignStack.from_design(amb, sub, design, nk_map, gmap, wl)
    cmap = {str(k): layer_from_material(v, 0.0, wl, name=f"{k}_seed",
                                        coherent=True, optimize=True,
                                        needle=True)
            for k, v in contrast.items()}
    return stack, cmap


def run_needle(layers: Sequence[Tuple[Any, float]],
               targets: Any,
               angles_deg, wavelengths,
               contrast: Mapping[Any, Any],
               pipeline_config=None, needle_config=None, lm_config=None,
               callback: Optional[Callable[[int, Dict], None]] = None,
               **stack_kwargs) -> Dict[str, Any]:
    """Design a coating with the needle pipeline, end to end.

    Parameters
    ----------
    layers : [(material, d_nm)] films (ambient/substrate via ``stack_kwargs``).
        Materials are ``MaterialSpec`` / mappings / nk arrays (constant
        complex also accepted).
    targets : TargetCollection or MeritSpec (native).
    angles_deg / wavelengths : solver grid (degrees / nm).
    contrast : {host name, film index, or "film{i}" name: seed material}.
        Hosts without an entry are never split (empty needle history when
        nothing matches — check names when a run inserts nothing).
    pipeline_config / needle_config / lm_config : native configs (or None).
    callback : ``(macro_cycle, phase_dict)``; raising aborts (USER_ABORT).
    stack_kwargs : ambient, substrate, per-film flags (see
        :func:`stack_from_layers`).

    Returns the native result dict (``termination``, ``final_mf``,
    ``phases``, final ``stack``).
    """
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    angs = np.ascontiguousarray(np.asarray(angles_deg, dtype=np.float64))
    names = stack_kwargs.pop("names", None)
    if names is None:
        names = [f"film{i}" for i in range(len(layers))]
    # Normalize contrast keys to host material names: int index, "film{i}"
    # strings, or names as given (unknown names match no host → no sites
    # there, by design).
    def _host_key(k):
        if isinstance(k, bool):
            return str(k)
        if isinstance(k, int) and 0 <= k < len(names):
            return names[k]
        s = str(k)
        if s in names:
            return s
        if s.startswith("film") and s[4:].isdigit() and int(s[4:]) < len(names):
            return names[int(s[4:])]
        return s
    cmap_in = {_host_key(k): v for k, v in contrast.items()}
    stack, cmap = stack_from_layers(layers, wl, cmap_in, names=names,
                                     **stack_kwargs)
    spec = (_build_merit_spec(targets) if isinstance(targets, TargetCollection)
            else targets)
    pipe = NeedlePipeline(stack, spec, angs, wl, cmap,
                          pipeline_config=pipeline_config,
                          needle_config=needle_config, lm=lm_config)
    return pipe.run(callback=callback)
