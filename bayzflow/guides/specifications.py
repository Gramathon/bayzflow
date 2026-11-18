"""Serializable descriptions of Pyro variational guides."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class GuideSpec:
    """A guide identifier and JSON-compatible constructor configuration."""

    kind: str = "auto_lowrank"
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    implementation_id: str | None = None
    implementation_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("guide kind must be a non-empty string")
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        # Fail early for callables, tensors and other non-portable config values.
        try:
            json.dumps(dict(self.kwargs), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("guide kwargs must contain only finite JSON-compatible values") from exc

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"kind": self.kind, "kwargs": dict(self.kwargs)}
        if self.implementation_id is not None:
            result["implementation_id"] = self.implementation_id
        if self.implementation_version is not None:
            result["implementation_version"] = self.implementation_version
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GuideSpec":
        allowed = {"kind", "kwargs", "implementation_id", "implementation_version"}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unsupported guide specification fields: {sorted(extra)}")
        return cls(**dict(value))

    @property
    def identity(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
