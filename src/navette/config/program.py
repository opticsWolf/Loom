# -*- coding: utf-8 -*-
"""Program documents: versioned envelopes for full or partial restore.

A *program* file restores a complete setup (materials, groups, named
structures, architect chain) in dependency order; every section is
schema-identical to its standalone document, so partial restore reuses
the same loaders. References are by name (layers → material codes,
blocks → structure labels); missing refs raise, never silently default.

Versioning matches the state discipline: ``schema_version`` is refused
when missing/stale/future. Legacy flat files (``materials:`` /
``layers:`` at top level, no envelope) keep loading.

Multi-load collisions: ``prefix`` prepends to every imported name (material
names/codes, group names, structure/block labels) with all references
rewritten consistently, so two programs coexist in one session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from .builders import (
    architect_from_config,
    group_from_config,
    layer_from_config,
    material_provider_from_library,
    structure_from_config,
)
from .io import load_json, load_yaml
from .models import (
    BlockConfig,
    GroupConfig,
    LayerConfig,
    MaterialDefinition,
    NamedStructureConfig,
)

PROGRAM_SCHEMA_VERSION = 1

Kind = Literal["materials", "groups", "structure", "architect", "program"]
KINDS: Tuple[str, ...] = ("materials", "groups", "structure", "architect", "program")


class ProgramDocument(BaseModel):
    """Versioned envelope: standalone section or full program."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(description="Must equal PROGRAM_SCHEMA_VERSION (1)")
    kind: Kind
    name: Optional[str] = None
    sections: Optional[Dict[str, Any]] = None  # kind == "program" only

    @classmethod
    def check_version(cls, v: Any) -> int:
        if v != PROGRAM_SCHEMA_VERSION:
            raise ValueError(
                f"program schema_version {v!r} unsupported "
                f"(code reads {PROGRAM_SCHEMA_VERSION})."
            )
        return v


def _gate(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Legacy-flat detection + version gate. Returns the raw mapping."""
    if "kind" not in raw:
        return raw  # legacy flat form (no envelope to gate)
    ProgramDocument.check_version(raw.get("schema_version"))
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(
            f"program kind {kind!r} unknown (expected one of {', '.join(KINDS)})."
        )
    # Envelope carries its section payload at top level (standalone docs);
    # strip known payload keys before strict validation so typos still fail.
    payload_keys = {
        "materials": {"materials"},
        "groups": {"groups"},
        "structure": {"label", "layers", "groups"},
        "architect": {"structures", "blocks"},
        "program": {"sections"},
    }[kind]
    unknown = set(raw) - {"schema_version", "kind", "name"} - payload_keys
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}.")
    doc = ProgramDocument.model_validate(
        {k: v for k, v in raw.items() if k in ("schema_version", "kind", "name", "sections")}
    )
    if doc.kind == "program" and not doc.sections:
        raise ValueError("program document needs a 'sections' mapping.")
    if doc.kind != "program" and doc.sections:
        raise ValueError(f"standalone {doc.kind!r} document must not carry 'sections'.")
    return raw


def load_document(
    path: Union[str, Path], fmt: Optional[str] = None
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Read + validate a program/section file.

    Returns ``(kind, name, payload)`` where payload is the section content:
    for ``kind == "program"`` the ``sections`` mapping, otherwise the
    document minus envelope keys. Legacy flat files map to
    ``("materials" | "structure", None, raw)``.
    """
    path = Path(path)
    if fmt is None:
        fmt = "json" if path.suffix.lower() == ".json" else "yaml"
    raw = load_json(path) if fmt == "json" else load_yaml(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top level must be a mapping.")
    gated = _gate(raw)
    if "kind" not in gated:  # legacy flat (raw section content, like nested)
        if "materials" in gated:
            return "materials", None, list(gated["materials"])
        if "layers" in gated:
            return "structure", None, {
                "label": "stack",
                "layers": list(gated["layers"]),
                "groups": list(gated.get("groups", [])),
            }
        raise ValueError(
            f"{path}: legacy document needs 'materials' or 'layers' at top level."
        )
    kind = gated["kind"]
    if kind == "program":
        return kind, gated.get("name"), dict(gated["sections"])
    # Standalone payload is the section content itself (identical to the
    # nested form): a list for materials/groups, a mapping otherwise.
    content_key = {"materials": "materials", "groups": "groups"}.get(kind)
    if content_key is not None:
        return kind, gated.get("name"), list(gated[content_key])
    payload = {k: v for k, v in gated.items()
               if k not in ("schema_version", "kind", "name")}
    return kind, gated.get("name"), payload


def _px(value: str, prefix: Optional[str]) -> str:
    return f"{prefix}{value}" if prefix else value


# -- section loaders (each usable standalone or nested) ---------------------

def load_materials(
    items: List[Mapping[str, Any]],
    wavelength: np.ndarray,
    prefix: Optional[str] = None,
) -> Any:
    """MaterialObjectProvider from a ``materials`` section (prefix-aware)."""
    defs = [MaterialDefinition.model_validate(m) for m in items]
    if prefix:
        defs = [d.model_copy(update={"name": _px(d.name, prefix),
                                     "code": _px(d.code or d.name, prefix)})
                for d in defs]
    return material_provider_from_library(defs, wavelength)


def load_groups(
    items: List[Mapping[str, Any]], prefix: Optional[str] = None
) -> Dict[str, Any]:
    """``{name: Group}`` from a ``groups`` section (prefix-aware)."""
    out = {}
    for raw in items:
        cfg = GroupConfig.model_validate(raw)
        if prefix:
            cfg = cfg.model_copy(update={"name": _px(cfg.name, prefix)})
        out[cfg.name] = group_from_config(cfg)
    return out


def load_structure(
    payload: Mapping[str, Any],
    materials: Any,
    library_groups: Optional[Mapping[str, Any]] = None,
    prefix: Optional[str] = None,
) -> Any:
    """Navette_Structure from a ``structure`` section (prefix-aware).

    Per-structure ``groups`` merge over ``library_groups`` (own wins).
    """
    if materials is None:
        raise KeyError("structure section needs materials: no provider given.")
    layers = [LayerConfig.model_validate(item) for item in payload["layers"]]
    own = [GroupConfig.model_validate(item) for item in payload.get("groups", [])]
    if prefix:
        layers = [item.model_copy(update={"material_code": _px(item.material_code, prefix)})
                  for item in layers]
        own = [item.model_copy(update={"name": _px(item.name, prefix)}) for item in own]
    merged: Dict[str, Any] = dict(library_groups or {})
    merged.update({c.name: group_from_config(c) for c in own})
    # Build directly (names already final — structure_from_config would
    # re-derive them from configs, so assemble here instead).
    from navette.structure import Navette_Structure
    from .builders import layer_from_config as _layer
    built = [_layer(c, materials) for c in layers]
    return Navette_Structure(layer_list=built, group_dict=merged, materials=materials)


def load_named_structures(
    items: List[Mapping[str, Any]],
    materials: Any,
    library_groups: Optional[Mapping[str, Any]] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """``{label: Navette_Structure}`` (duplicate labels raise)."""
    out: Dict[str, Any] = {}
    for raw in items:
        cfg = NamedStructureConfig.model_validate(raw)
        label = _px(cfg.label, prefix)
        if label in out:
            raise ValueError(f"duplicate structure label '{label}'.")
        out[label] = load_structure(
            {"layers": [item.model_dump() for item in cfg.layers],
             "groups": [item.model_dump() for item in cfg.groups]},
            materials, library_groups, prefix,
        )
    return out


def load_architect(
    payload: Mapping[str, Any],
    structures: Mapping[str, Any],
    materials: Any = None,
    prefix: Optional[str] = None,
) -> Any:
    """Navette_Architect: blocks reference ``structures`` by label."""
    blocks = [BlockConfig.model_validate(b) for b in payload["blocks"]]
    if prefix:
        blocks = [b.model_copy(update={"structure": _px(b.structure, prefix),
                                       "label": _px(b.label, prefix) if b.label else b.label})
                  for b in blocks]
    return architect_from_config(structures, blocks, materials)


# -- full program ------------------------------------------------------------

@dataclass
class LoadedProgram:
    """Everything a program file restores (absent sections stay empty)."""

    name: Optional[str] = None
    materials: Any = None
    groups: Dict[str, Any] = field(default_factory=dict)
    structures: Dict[str, Any] = field(default_factory=dict)
    architect: Any = None


def load_program(
    path: Union[str, Path],
    wavelength: np.ndarray,
    *,
    fmt: Optional[str] = None,
    prefix: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> LoadedProgram:
    """Restore a full program (or a standalone section) from file.

    Sections load in dependency order (materials → groups → structures →
    architect). File sections win; ``context`` (materials/groups/
    structures/architect) fills ABSENT sections only. A standalone
    section document loads just that part (same code path).
    """
    kind, name, payload = load_document(path, fmt)
    context = context or {}
    prog = LoadedProgram(name=name)

    sections = payload if kind == "program" else {kind: payload}

    if "materials" in sections:
        prog.materials = load_materials(sections["materials"], wavelength, prefix)
    elif "materials" in context:
        prog.materials = context["materials"]

    if "groups" in sections:
        prog.groups = load_groups(sections["groups"], prefix)
    elif "groups" in context:
        prog.groups = dict(context["groups"])

    # Program sections use the plural list form; the singular "structure"
    # kind exists only for standalone documents (loaded below as one entry).
    if kind == "structure":
        label = payload.get("label", "stack")
        prog.structures[_px(label, prefix)] = load_structure(
            payload, prog.materials, prog.groups, prefix)
    elif "structures" in sections:
        prog.structures = load_named_structures(sections["structures"],
                                                prog.materials, prog.groups, prefix)
    elif "structures" in context:
        prog.structures = dict(context["structures"])

    if kind == "architect":
        # Standalone architect documents carry their structures inline.
        if "structures" not in payload:
            raise ValueError("standalone architect document needs 'structures' + 'blocks'.")
        prog.structures = load_named_structures(payload["structures"],
                                                 prog.materials, prog.groups, prefix)
        sections = {"architect": {"blocks": payload["blocks"]}}

    arch_payload = sections.get("architect")
    if arch_payload is not None:
        prog.architect = load_architect(arch_payload, prog.structures,
                                        prog.materials, prefix)
    elif "architect" in context:
        prog.architect = context["architect"]
    return prog
