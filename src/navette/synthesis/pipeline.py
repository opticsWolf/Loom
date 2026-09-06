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


_LAYER_KEYS = ("roughness", "rough_type", "interface", "interface_thickness")
_FILM_DEFAULTS = dict(coherent=True, optimize=True, needle=True,
                      roughness=0.0, rough_type=0, inhomogen=False,
                      inh_delta=0.1, interface=False,
                      interface_thickness=0.0)


def _film_dicts(layers, names, wl, film_flags=None, per_film_flags=None):
    """Shape ``[(material, d)]`` films into native ``ArrayFilm`` dicts.

    Merge order per film: defaults < global ``film_flags`` < per-film
    override (``rough_val`` aliases ``roughness``); materials evaluate
    to nk on ``wl``. Shared by ``stack_from_layers`` and ``run_needle``.
    """
    flags = dict(film_flags or {})
    if "rough_val" in flags:
        flags["roughness"] = flags.pop("rough_val")
    unknown = set(flags) - set(_FILM_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown film flags {sorted(unknown)}.")
    overrides = per_film_flags or {}
    out = []
    for (mat, d), nm in zip(layers, names):
        fd = dict(_FILM_DEFAULTS)
        fd.update(flags)
        local = dict(overrides.get(nm, {}))
        if "rough_val" in local:
            local["roughness"] = local.pop("rough_val")
        unknown = set(local) - set(_FILM_DEFAULTS)
        if unknown:
            raise TypeError(f"film {nm!r}: unknown flags {sorted(unknown)}.")
        fd.update(local)
        out.append({
            "name": str(nm), "nk": _eval_nk(mat, wl), "d_nm": float(d),
            "coherent": bool(fd["coherent"]),
            "roughness": float(fd["roughness"]),
            "rough_type": int(fd["rough_type"]),
            "inhomogen": bool(fd["inhomogen"]),
            "inh_delta": float(fd["inh_delta"]),
            "interface": bool(fd["interface"]),
            "interface_thickness": float(fd["interface_thickness"]),
            "optimize": bool(fd["optimize"]),
            "needle": bool(fd["needle"]),
        })
    return out


def _half_nk(value, wl):
    """Half-space nk: constant indices broadcast (mirrors old ``fixed``)."""
    if isinstance(value, (int, float, complex)):
        return np.full(wl.shape, complex(value), dtype=np.complex128)
    return _eval_nk(value, wl)


def _group_map(groups):
    """Bound groups pass through; param dicts construct natively."""
    from navette._structure import Group as _RsGroup
    return {str(name): (g if isinstance(g, _RsGroup) else _RsGroup(str(name), **dict(g)))
            for name, g in (groups or {}).items()}


def stack_from_layers(layers: Sequence[Tuple[Any, float]],
                      wavelengths, contrast: Mapping[Any, Any],
                      ambient: Tuple[Any, str] = (1.0, "air"),
                      substrate: Tuple[Any, str] = (1.52, "sub"),
                      names: Optional[Sequence[str]] = None,
                      film_flags: Optional[Dict[str, Any]] = None,
                      groups: Optional[Mapping[str, Any]] = None,
                      per_film_flags: Optional[Mapping[str, Dict[str, Any]]] = None,
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
    the silent-drop limitation is gone. Graded films take one of two
    paths, implied by their flags (no separate declaration): a graded
    film with ``optimize=False`` *and* ``needle=False`` expands WITH
    the profile as pinned background (silent; excluded from needle
    candidacy and thickness optimization; merge/cleanup preserve the
    span). Any other graded film homogenizes with a warning (base
    index — pin it to keep the profile). ``per_film_flags`` overrides
    ``film_flags`` per film name (e.g. grade only the substrate).
    Span-aware graded optimization is future work (D2).
    """
    # Thin over native assemble_design: shaping here, expansion in Rust.
    from navette._smatrix import assemble_design as _assemble
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    names = list(names) if names is not None else [f"film{i}" for i in range(len(layers))]
    if len(names) != len(layers):
        raise ValueError("names length must match layers length.")
    films = _film_dicts(layers, names, wl, film_flags, per_film_flags)
    seeds = [(str(k), f"{k}_seed", _eval_nk(v, wl)) for k, v in contrast.items()]
    return _assemble(
        _half_nk(ambient[0], wl), str(ambient[1] if len(ambient) > 1 else "air"),
        _half_nk(substrate[0], wl), str(substrate[1] if len(substrate) > 1 else "sub"),
        films, _group_map(groups), seeds, wl)


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
    # Thin over native run_design: evaluate + key/flag shaping here,
    # assembly + macro-loop in Rust. Contrast-key normalization stays
    # (presentation over the film order).
    from navette._smatrix import run_design as _run_design
    wl = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64))
    angs = np.ascontiguousarray(np.asarray(angles_deg, dtype=np.float64))
    names = stack_kwargs.pop("names", None)
    if names is None:
        names = [f"film{i}" for i in range(len(layers))]
    names = list(names)
    if len(names) != len(layers):
        raise ValueError("names length must match layers length.")
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
    ambient = stack_kwargs.pop("ambient", (1.0, "air"))
    substrate = stack_kwargs.pop("substrate", (1.52, "sub"))
    film_flags = stack_kwargs.pop("film_flags", None)
    groups = stack_kwargs.pop("groups", None)
    per_film_flags = stack_kwargs.pop("per_film_flags", None)
    if stack_kwargs:
        raise TypeError(f"run_needle: unknown stack options {sorted(stack_kwargs)}.")
    films = _film_dicts(layers, names, wl, film_flags, per_film_flags)
    seeds = [(_host_key(k), f"{_host_key(k)}_seed", _eval_nk(v, wl))
             for k, v in contrast.items()]
    spec = (_build_merit_spec(targets) if isinstance(targets, TargetCollection)
            else targets)
    amb_nk = _half_nk(ambient[0], wl)
    sub_nk = _half_nk(substrate[0], wl)
    gmap = _group_map(groups)
    return _run_design(
        amb_nk, str(ambient[1] if len(ambient) > 1 else "air"),
        sub_nk, str(substrate[1] if len(substrate) > 1 else "sub"),
        films, gmap, seeds, wl, angs, spec,
        pipeline_config=pipeline_config, needle_config=needle_config,
        lm=lm_config, callback=callback)
