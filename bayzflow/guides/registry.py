"""Trusted registry for built-in and locally registered Pyro guide factories.

Custom callables are only installed by trusted Python code. A serialized GuideSpec
can select and configure them, but cannot name an import path or carry code.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, get_type_hints

import pyro
from pyro.infer import autoguide

from ._fullrank import MAX_FULLRANK_LATENTS, BoundedFullRankGuide
from .specifications import GuideSpec


class GuideRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        config_schema: Any = None,
        version: str = "1",
        capabilities: Mapping[str, Any] | None = None,
        validator: Callable[[Mapping[str, Any]], None] | None = None,
        compatibility: Callable[[Any], bool] | None = None,
    ) -> None:
        if not name or ":" in name or name in self._entries:
            raise ValueError(f"invalid or duplicate guide name: {name!r}")
        if not callable(factory):
            raise TypeError("guide factory must be callable")
        self._entries[name] = {
            "factory": factory,
            "schema": config_schema,
            "version": str(version),
            "capabilities": dict(capabilities or {}),
            "validator": validator,
            "compatibility": compatibility,
            "identity": _factory_identity(factory),
        }

    def _entry(self, kind: str) -> dict[str, Any]:
        if kind in _BUILTINS:
            cls = getattr(autoguide, _BUILTINS[kind])
            return {"factory": cls, "schema": None, "version": pyro.__version__, "capabilities": {}}
        if kind.startswith("custom:"):
            name = kind.partition(":")[2]
            if name in self._entries:
                return self._entries[name]
        raise ValueError(f"unknown guide kind: {kind!r}")

    @staticmethod
    def _schema_values(schema: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        if schema is None:
            return dict(kwargs)
        if hasattr(schema, "model_validate"):
            obj = schema.model_validate(dict(kwargs))
            return obj.model_dump(mode="json")
        if hasattr(schema, "parse_obj"):
            obj = schema.parse_obj(dict(kwargs))
            return obj.dict()
        if inspect.isclass(schema) and is_dataclass(schema):
            fields = {field.name: field for field in __import__("dataclasses").fields(schema)}
            extra = set(kwargs) - set(fields)
            if extra:
                raise ValueError(f"unsupported custom guide config fields: {sorted(extra)}")
            try:
                obj = schema(**dict(kwargs))
            except TypeError as exc:
                raise ValueError(f"invalid custom guide configuration: {exc}") from exc
            hints = get_type_hints(schema)
            for key, value in vars(obj).items():
                expected = hints.get(key)
                if expected in (str, int, float, bool):
                    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
                        raise ValueError(f"custom guide field {key!r} must be {expected.__name__}")
                    if expected is float and not __import__("math").isfinite(value):
                        raise ValueError(f"custom guide field {key!r} must be finite")
            return asdict(obj)
        if callable(schema):
            result = schema(dict(kwargs))
            return dict(kwargs) if result is None else dict(result)
        raise TypeError("config_schema must be a Pydantic model, dataclass, or validator")

    def validate(self, spec: GuideSpec | str, kwargs: Mapping[str, Any] | None = None) -> GuideSpec:
        if isinstance(spec, str):
            spec = GuideSpec(spec, kwargs or {})
        elif kwargs is not None:
            raise ValueError("pass kwargs in GuideSpec or separately, not both")
        values = dict(spec.kwargs)
        if spec.kind == "auto_lowrank" and "rank" not in values:
            # Preserve the package's long-standing explicit rank=20 default.
            values["rank"] = 20
        entry = self._entry(spec.kind)
        values = self._schema_values(entry.get("schema"), values)
        factory = entry["factory"]
        signature = inspect.signature(factory)
        accepted = set(signature.parameters)
        accepted.discard("model")
        unsupported = set(values) - accepted
        if unsupported:
            raise ValueError(f"unsupported arguments for {spec.kind}: {sorted(unsupported)}")
        if spec.kind in _BUILTINS:
            if "init_loc_fn" in values:
                raise ValueError("init_loc_fn is not serializable")
            if values.get("create_plates") is not None:
                raise ValueError("create_plates must be null in a serialized guide specification")
            scale = values.get("init_scale", 0.1)
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError("init_scale must be a finite positive number")
            if "init_scale" in values:
                values["init_scale"] = float(scale)
        if entry.get("validator") is not None:
            entry["validator"](values)
        if spec.kind == "auto_lowrank" and "rank" in values:
            rank = values["rank"]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("auto_lowrank rank must be a positive integer")
        if spec.kind.startswith("custom:"):
            name = spec.kind.partition(":")[2]
            expected = self._entries[name]
            if spec.implementation_version and spec.implementation_version != expected["version"]:
                raise ValueError(f"custom guide version mismatch for {name!r}")
            if spec.implementation_id and spec.implementation_id != expected["identity"]:
                raise ValueError(f"custom guide implementation mismatch for {name!r}")
            return GuideSpec(spec.kind, values, expected["identity"], expected["version"])
        return GuideSpec(spec.kind, values, f"pyro.infer.autoguide.{_BUILTINS[spec.kind]}", entry["version"])

    def create(self, model: Callable[..., Any], spec: GuideSpec | str, kwargs: Mapping[str, Any] | None = None) -> Any:
        valid = self.validate(spec, kwargs)
        entry = self._entry(valid.kind)
        compatibility = entry.get("compatibility")
        if compatibility is not None and not compatibility(model):
            raise ValueError(f"guide {valid.kind!r} is incompatible with the supplied model")
        factory = BoundedFullRankGuide if valid.kind == "auto_fullrank" else entry["factory"]
        guide = factory(model, **dict(valid.kwargs))
        if not callable(guide):
            raise TypeError(f"guide factory for {valid.kind!r} did not return a callable Pyro guide")
        return guide

    def list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for kind, class_name in _BUILTINS.items():
            cls = getattr(autoguide, class_name)
            row: dict[str, Any] = {
                "kind": kind,
                "implementation": f"pyro.infer.autoguide.{class_name}",
                "version": pyro.__version__,
                "kwargs": [
                    name for name in _accepted_kwargs(cls) if name not in {"init_loc_fn", "create_plates"}
                ],
            }
            if kind == "auto_lowrank":
                row["defaults"] = {"rank": 20}
            if kind == "auto_fullrank":
                row["max_latent_dimension"] = MAX_FULLRANK_LATENTS
            rows.append(row)
        for name, entry in sorted(self._entries.items()):
            rows.append({"kind": f"custom:{name}", "implementation": entry["identity"],
                         "version": entry["version"], "kwargs": _accepted_kwargs(entry["factory"]),
                         "capabilities": entry["capabilities"],
                         "config_schema": getattr(entry["schema"], "__name__", None)})
        return rows


def _accepted_kwargs(factory: Callable[..., Any]) -> list[str]:
    return sorted(k for k in inspect.signature(factory).parameters if k != "model")


def _factory_identity(factory: Callable[..., Any]) -> str:
    target = factory if inspect.isfunction(factory) or inspect.isclass(factory) else type(factory)
    qualified = f"{target.__module__}.{target.__qualname__}"
    try:
        source = inspect.getsource(factory).encode("utf-8")
    except (OSError, TypeError):
        code = getattr(factory, "__code__", None)
        source = code.co_code if code is not None else qualified.encode("utf-8")
    digest = hashlib.sha256(source).hexdigest()
    return f"{qualified}#{digest}"


_BUILTINS = {
    "auto_normal": "AutoNormal",
    "auto_diag": "AutoDiagonalNormal",
    "auto_lowrank": "AutoLowRankMultivariateNormal",
    "auto_fullrank": "AutoMultivariateNormal",
}

default_registry = GuideRegistry()
