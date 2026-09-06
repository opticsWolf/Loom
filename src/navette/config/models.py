# -*- coding: utf-8 -*-
"""
Config models: thin validated holders over the native schema.

Native-validated holders (the pydantic models are gone). Each class validates its inputs
natively on construction (unknown fields refused, bounds enforced) and
exposes ``model_validate`` / ``model_dump`` / ``model_copy`` for the
loaders plus plain attribute access for the builders. No schema lives
here — ``navette._structure`` owns it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Re-export the schema version gate source (single canonical home is Rust;
# this mirrors the Python-side state constant it already had).
from navette.structure.types import SCHEMA_VERSION  # noqa: F401


def _plain(value: Any) -> Any:
    """Deep-convert holders to plain JSON values for native handover."""
    if isinstance(value, _NativeModel):
        return _plain(value.model_dump())
    if isinstance(value, _Params):
        return {k: _plain(v) for k, v in value.model_dump().items()}
    if isinstance(value, _TabData):
        return {"wavelengths": list(value.wavelengths), "values": list(value.values)}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class _Params:
    """Plain params mapping with attribute access + ``model_dump``."""

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged = dict(data or {})
        merged.update(kwargs)
        object.__setattr__(self, "_data", merged)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Params({self._data!r})"


class _TabData:
    """Tabulated grid holder (``wavelengths`` / ``values``)."""

    __slots__ = ("wavelengths", "values")

    def __init__(self, wavelengths: List[float], values: List[float]) -> None:
        self.wavelengths = list(wavelengths)
        self.values = list(values)


class _NativeModel:
    """Base: validated dict + attribute access + loader-compat surface."""

    _native_name: str = ""

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        from navette import _structure as _ext
        merged = dict(data or {})
        merged.update(kwargs)
        native_cls = getattr(_ext, self._native_name)
        self._data = native_cls(_plain(merged)).to_dict()
        self._post_init()

    def _post_init(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "_NativeModel":
        return cls(dict(data))

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return dict(self._data)

    def model_copy(self, update: Optional[Dict[str, Any]] = None) -> "_NativeModel":
        merged = dict(self._data)
        if update:
            merged.update(update)
        return type(self)(merged)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self._data!r})"


class MaterialDefinition(_NativeModel):
    """One named material: model selector plus params (natively validated)."""

    _native_name = "MaterialDefinition"

    def _post_init(self) -> None:
        self._data["params"] = _Params(self._data.get("params", {}))
        for key in ("n_data", "k_data"):
            raw = self._data.get(key)
            self._data[key] = _TabData(raw["wavelengths"], raw["values"]) if raw else None

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = dict(self._data)
        out["params"] = self._data["params"].model_dump()
        for key in ("n_data", "k_data"):
            raw = self._data.get(key)
            out[key] = ({"wavelengths": list(raw.wavelengths), "values": list(raw.values)}
                        if raw is not None else None)
        return out

    def model_copy(self, update: Optional[Dict[str, Any]] = None) -> "MaterialDefinition":
        return MaterialDefinition(self.model_dump() | dict(update or {}))


class LayerConfig(_NativeModel):
    """One stack layer (natively validated)."""

    _native_name = "LayerConfig"


class ErrorParams(_NativeModel):
    """Fabrication-error law and parameters (plain validated holder)."""

    _native_name = "_ErrorParams"

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged = dict(data or {})
        merged.update(kwargs)
        self._data = dict(merged)


class GroupConfig(_NativeModel):
    """Group scaling + per-channel errors (natively validated)."""

    _native_name = "GroupConfig"

    _channels = ("thickness_error_params", "inh_delta_error_params",
                 "roughness_error_params", "interface_error_params",
                 "n_error_params", "k_error_params")

    def _post_init(self) -> None:
        for key in self._channels:
            self._data[key] = _Params(self._data.get(key, {}))

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = dict(self._data)
        for key in self._channels:
            raw = self._data.get(key)
            out[key] = raw.model_dump() if isinstance(raw, _Params) else raw
        return out

    def model_copy(self, update: Optional[Dict[str, Any]] = None) -> "GroupConfig":
        return GroupConfig(self.model_dump() | dict(update or {}))


class BlockConfig(_NativeModel):
    """One architect block (natively validated)."""

    _native_name = "BlockConfig"


class NamedStructureConfig(_NativeModel):
    """A labelled structure (natively validated)."""

    _native_name = "NamedStructureConfig"

    def _post_init(self) -> None:
        self._data["layers"] = [v if isinstance(v, LayerConfig) else LayerConfig(v)
                                for v in self._data.get("layers", [])]
        self._data["groups"] = [v if isinstance(v, GroupConfig) else GroupConfig(v)
                                for v in self._data.get("groups", [])]

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = dict(self._data)
        out["layers"] = [v.model_dump() if isinstance(v, LayerConfig) else v
                         for v in self._data.get("layers", [])]
        out["groups"] = [v.model_dump() if isinstance(v, GroupConfig) else v
                         for v in self._data.get("groups", [])]
        return out

    def model_copy(self, update: Optional[Dict[str, Any]] = None) -> "NamedStructureConfig":
        return NamedStructureConfig(self.model_dump() | dict(update or {}))


# Param-model names kept as documentation aliases (validation is native;
# per-model classes no longer exist as types).
KonstantParams = _Params
CauchyParams = _Params
CauchyUrbachParams = _Params
SellmeierParams = _Params
SellmeierUrbachParams = _Params
TableMaterialParams = _Params
TabulatedData = _TabData


class StructureState(_NativeModel):
    """Serializable stack (version gate lives natively in load paths)."""

    _native_name = "_StructureState"

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged = dict(data or {})
        merged.update(kwargs)
        if merged.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"StructureState schema_version {merged.get('schema_version')} "
                f"unsupported (code reads {SCHEMA_VERSION}).")
        self._data = dict(merged)


class ArchitectState(_NativeModel):
    """Serializable chain (version gate lives natively in load paths)."""

    _native_name = "_ArchitectState"

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged = dict(data or {})
        merged.update(kwargs)
        if merged.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"ArchitectState schema_version {merged.get('schema_version')} "
                f"unsupported (code reads {SCHEMA_VERSION}).")
        self._data = dict(merged)
