"""Application service shared by non-invasive MCP discovery tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pyro
import torch

from bayzflow import __version__
from bayzflow.guides import GuideSpec
from bayzflow.guides.registry import GuideRegistry, default_registry
from bayzflow.core.engine import find_candidate_modules


@dataclass
class BayzflowService:
    """Process-local registry for explicitly trusted objects and guides."""

    guide_registry: GuideRegistry = default_registry
    models: Dict[str, Any] = field(default_factory=dict)

    def register_model(self, name: str, model: torch.nn.Module) -> None:
        if not name or name in self.models:
            raise ValueError("model name must be non-empty and unique")
        self.models[name] = model

    def doctor(self) -> dict[str, Any]:
        return {"version": __version__, "python_torch": torch.__version__,
                "pyro": pyro.__version__, "cuda_available": torch.cuda.is_available(),
                "transport": "stdio", "models_registered": len(self.models)}

    def model_list(self) -> list[dict[str, str]]:
        return [{"id": name, "class": f"{type(model).__module__}.{type(model).__qualname__}"}
                for name, model in sorted(self.models.items())]

    def model_inspect(self, model_id: str) -> dict[str, Any]:
        model = self.models.get(model_id)
        if model is None:
            raise ValueError("unknown registered model id")
        return {"id": model_id, "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "parameters": sum(p.numel() for p in model.parameters()),
                "modules": [{"name": name, "class": type(module).__name__}
                            for name, module in list(model.named_modules())[1:257] if name],
                "modules_truncated": sum(1 for _ in model.named_modules()) > 257}

    def layer_candidates(self, model_id: str) -> list[dict[str, str]]:
        model = self.models.get(model_id)
        if model is None:
            raise ValueError("unknown registered model id")
        return [{"name": item.name, "class": type(item.module).__name__}
                for item in find_candidate_modules(model)[:256]]

    def guide_list(self) -> list[dict[str, Any]]:
        return self.guide_registry.list()

    def guide_schema(self, kind: str) -> dict[str, Any]:
        for row in self.guide_registry.list():
            if row["kind"] == kind:
                return row
        raise ValueError("unknown or unregistered guide")

    def guide_validate(self, kind: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        spec = self.guide_registry.validate(GuideSpec(kind, kwargs))
        return {"valid": True, "guide": spec.to_dict(), "identity": spec.identity}
