"""Application service shared by non-invasive MCP discovery tools."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pyro
import torch

from bayzflow import __version__
from bayzflow.core.engine import find_candidate_modules
from bayzflow.guides import GuideSpec
from bayzflow.guides._fullrank import check_fullrank_dimension
from bayzflow.guides.registry import GuideRegistry, default_registry

from .jobs import PUBLIC_ID, JobStore, fingerprint
from .workloads import Workload

FIT_FIELDS = {
    "model_id",
    "model_state_sha256",
    "workload_version",
    "layers",
    "guide",
    "epochs",
    "seed",
    "prior_scale",
    "lr",
}


@dataclass
class BayzflowService:
    """Process-local registry for explicitly trusted objects and guides."""

    guide_registry: GuideRegistry = default_registry
    models: dict[str, Any] = field(default_factory=dict)

    workspace: JobStore | None = None
    workloads: dict[str, Workload] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        model: torch.nn.Module,
        workload: Workload | None = None,
    ) -> None:
        if not PUBLIC_ID.fullmatch(name) or name in self.models:
            raise ValueError("model name must be non-empty and unique")
        self.models[name] = model
        if workload is not None:
            self.workloads[name] = workload

    def doctor(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "python_torch": torch.__version__,
            "pyro": pyro.__version__,
            "cuda_available": torch.cuda.is_available(),
            "transport": "stdio",
            "models_registered": len(self.models),
            "operational_workspace": self.workspace is not None,
        }

    def model_list(self) -> list[dict[str, str]]:
        return [
            {"id": name, "class": f"{type(model).__module__}.{type(model).__qualname__}"}
            for name, model in sorted(self.models.items())
        ]

    def model_inspect(self, model_id: str) -> dict[str, Any]:
        model = self.models.get(model_id)
        if model is None:
            raise ValueError("unknown registered model id")
        return {
            "id": model_id,
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameters": sum(p.numel() for p in model.parameters()),
            "modules": [
                {"name": name, "class": type(module).__name__}
                for name, module in list(model.named_modules())[1:257]
                if name
            ],
            "modules_truncated": sum(1 for _ in model.named_modules()) > 257,
        }

    def layer_candidates(self, model_id: str) -> list[dict[str, str]]:
        model = self.models.get(model_id)
        if model is None:
            raise ValueError("unknown registered model id")
        return [
            {"name": item.name, "class": type(item.module).__name__}
            for item in find_candidate_modules(model)[:256]
        ]

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

    def require_jobs(self) -> JobStore:
        if self.workspace is None:
            raise ValueError("operational workspace is disabled")
        return self.workspace

    def bayesian_plan(
        self,
        model_id: str,
        layers: list[str],
        kind: str,
        kwargs: dict[str, Any],
        epochs: int,
        seed: int,
        prior_scale: float,
        lr: float,
    ) -> dict[str, Any]:
        jobs = self.require_jobs()
        model = self.models.get(model_id)
        workload = self.workloads.get(model_id)
        if model is None or workload is None:
            raise ValueError("model is not approved for inference")
        limits = jobs.limits
        if type(epochs) is not int or not 1 <= epochs <= limits.max_epochs:
            raise ValueError("epoch budget exceeded")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if not 0 < prior_scale <= 10 or not 0 < lr <= 1:
            raise ValueError("prior scale and learning rate are outside allowed ranges")
        if len(layers) > 32 or any(len(name) > 128 for name in layers):
            raise ValueError("layer selection is too large")
        candidates = {item.name: item.module for item in find_candidate_modules(model)}
        if not layers or len(set(layers)) != len(layers) or not set(layers) <= candidates.keys():
            raise ValueError("select distinct registered Bayesian layers")
        if any(a.startswith(b + ".") for a in layers for b in layers):
            raise ValueError("overlapping Bayesian layers are not supported")
        parameters = sum(p.numel() for p in model.parameters())
        state = model.state_dict()
        size = sum(t.numel() * t.element_size() for t in state.values())
        if parameters > limits.max_parameters or size > limits.max_model_bytes:
            raise ValueError("model resource budget exceeded")
        if any(t.device.type != "cpu" or t.layout != torch.strided for t in state.values()):
            raise ValueError("the operational service supports dense CPU models only")
        latent_dimension = sum(
            p.numel() for name in layers for p in candidates[name].parameters(recurse=False)
        )
        if len(json.dumps(kwargs)) > 4096:
            raise ValueError("guide configuration is too large")
        if kind == "auto_lowrank" and "rank" not in kwargs:
            kwargs = {**kwargs, "rank": min(20, latent_dimension)}
        guide = self.guide_registry.validate(GuideSpec(kind, kwargs))
        if kind == "auto_fullrank":
            check_fullrank_dimension(latent_dimension)
        if kind == "auto_lowrank" and guide.kwargs["rank"] > min(512, latent_dimension):
            raise ValueError("guide rank exceeds the admitted latent dimension")
        guide_parameters = 2 * latent_dimension
        if kind == "auto_fullrank":
            guide_parameters += latent_dimension**2
        if kind == "auto_lowrank":
            guide_parameters += latent_dimension * guide.kwargs["rank"]
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        replaced = sum(
            p.numel() for name in layers for p in candidates[name].parameters(recurse=False)
            if p.requires_grad
        )
        estimated_trainable = trainable - replaced + guide_parameters
        if estimated_trainable > limits.max_parameters:
            raise ValueError("estimated trainable parameter budget exceeded")
        if 2 * size + 4 * guide_parameters > limits.max_artifact_bytes:
            raise ValueError("model and guide exceed the checkpoint artifact budget")
        model_digest = fingerprint(
            [
                (
                    name,
                    list(t.shape),
                    str(t.dtype),
                    hashlib.sha256(
                        t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                    ).hexdigest(),
                )
                for name, t in sorted(state.items())
            ]
        )
        request = {
            "model_id": model_id,
            "model_state_sha256": model_digest,
            "workload_version": workload.version,
            "layers": layers,
            "guide": guide.to_dict(),
            "epochs": epochs,
            "seed": seed,
            "prior_scale": prior_scale,
            "lr": lr,
        }
        return {
            "plan_id": fingerprint(request),
            "request": request,
            "model_bytes": size,
            "parameter_count": parameters,
            "latent_dimension": latent_dimension,
            "estimated_trainable": estimated_trainable,
            "limits": asdict(limits),
        }

    def fit(self, plan_id: str, request: dict[str, Any], key: str) -> dict:
        if set(request) != FIT_FIELDS:
            raise ValueError("invalid plan fields")
        guide = GuideSpec.from_dict(request["guide"])
        plan = self.bayesian_plan(
            request["model_id"],
            request["layers"],
            guide.kind,
            dict(guide.kwargs),
            request["epochs"],
            request["seed"],
            request["prior_scale"],
            request["lr"],
        )
        if plan["plan_id"] != plan_id or plan["request"] != request:
            raise ValueError("plan approval does not match the current request and model")
        model_id = request["model_id"]
        return self.require_jobs().submit(
            "fit",
            plan["request"],
            key,
            (self.models[model_id], self.workloads[model_id], self.guide_registry, None),
        )

    def analyse(
        self,
        operation: str,
        checkpoint_id: str,
        samples: int,
        seed: int,
        key: str,
        selection: list[str] | None = None,
    ) -> dict:
        jobs = self.require_jobs()
        if type(samples) is not int or not 2 <= samples <= jobs.limits.max_samples:
            raise ValueError("posterior sample budget exceeded")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if selection and (len(selection) > 64 or any(len(name) > 160 for name in selection)):
            raise ValueError("measurement selection is too large")
        checkpoint = jobs.verified_checkpoint(checkpoint_id)
        request = json.loads(jobs.row(checkpoint_id)["request"])
        model_id = request["model_id"]
        guide = GuideSpec.from_dict(request["guide"])
        current = self.bayesian_plan(
            model_id,
            request["layers"],
            guide.kind,
            dict(guide.kwargs),
            request["epochs"],
            request["seed"],
            request["prior_scale"],
            request["lr"],
        )
        if current["request"] != request:
            raise ValueError("checkpoint does not match the available model, workload and guide")
        request.update({"checkpoint_job_id": checkpoint_id, "samples": samples, "seed": seed})
        if operation == "layers":
            layers = selection or request["layers"]
            if not set(layers) <= set(request["layers"]):
                raise ValueError("measurement layers must be selected Bayesian layers")
            request["measure_layers"] = layers
        elif selection:
            request["sites"] = selection
        return jobs.submit(
            operation,
            request,
            key,
            (self.models[model_id], self.workloads[model_id], self.guide_registry, checkpoint),
        )

    def posterior_inspect(self, job_id: str) -> dict:
        jobs = self.require_jobs()
        jobs.verified_checkpoint(job_id)
        row = jobs.row(job_id)
        artifacts = json.loads(row["artifacts"])
        entry = next(artifact for artifact in artifacts if artifact["name"] == "meta.json")
        page = jobs.read(job_id, entry["artifact_id"], 0, 32768)
        if not page["eof"]:
            raise ValueError("posterior metadata exceeds the response limit")
        return json.loads(base64.b64decode(page["base64"]))
