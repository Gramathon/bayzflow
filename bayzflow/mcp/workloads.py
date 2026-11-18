"""Approved CPU inference adapters and their worker entry point."""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyro
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bayzflow import BayzFlow
from bayzflow.guides import GuideRegistry, GuideSpec

Batch = tuple[torch.Tensor, torch.Tensor]
CHECKPOINT_FILES = ("meta.json", "model_state.pt", "pyro_constraints.json", "pyro_params.pt")
OUTPUT_FILES = {
    "fit": CHECKPOINT_FILES,
    "sample": ("samples.pt", "samples.pt.json", "measurement.json"),
    "epistemic": ("samples.pt", "samples.pt.json", "measurement.json"),
    "layers": ("samples.pt", "samples.pt.json", "measurement.json"),
}


@dataclass(frozen=True)
class Workload:
    """An operator-approved, importable adapter with fixed data, likelihood and version."""

    version: str
    batch: Callable[[], Batch]
    predict: Callable[[nn.Module, Batch], torch.Tensor]
    loss: Callable[[torch.Tensor, Batch], torch.Tensor]


def regression_batch() -> Batch:
    x = torch.tensor([[-1.0, -1.1], [-0.5, -0.3], [0.5, 0.4], [1.0, 0.9], [1.5, 1.2]])
    y = x @ torch.tensor([[0.7], [-0.2]]) + torch.tensor([[0.1], [-0.2], [0.05], [0.2], [-0.1]])
    return x, y


def regression_model() -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.zero_()
    return model


def regression_predict(model: nn.Module, batch: Batch) -> torch.Tensor:
    return model(batch[0])


def regression_loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    # Gaussian likelihood with known observation standard deviation of 0.5.
    return 0.5 * ((logits - batch[1]) / 0.5).square().sum()


REGRESSION = Workload("1", regression_batch, regression_predict, regression_loss)


def run_job(
    directory: Path,
    operation: str,
    request: dict,
    model: nn.Module,
    workload: Workload,
    registry: GuideRegistry,
    checkpoint: Path | None,
    memory_limit: int,
    artifact_limit: int,
    deadline: float,
) -> None:
    import resource
    import signal

    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (artifact_limit, artifact_limit))
        signal.setitimer(signal.ITIMER_REAL, max(deadline - time.time(), 0.001))
        torch.set_num_threads(1)
        pyro.set_rng_seed(request["seed"])
        batch = workload.batch()
        inference = BayzFlow(
            model, guide=GuideSpec.from_dict(request["guide"]), guide_registry=registry
        )
        inference.selected_module_names = request["layers"]
        inference.prior_scales = dict.fromkeys(request["layers"], request["prior_scale"])
        inference.build_bayesian_model()
        if operation == "fit":
            inference.train_posterior(
                DataLoader(TensorDataset(*batch), batch_size=len(batch[0])),
                workload.loss,
                workload.predict,
                num_epochs=request["epochs"],
                lr=request["lr"],
                seed=request["seed"],
                save_dir=str(directory),
                experiment_metadata=request,
            )
            if not all(map(math.isfinite, inference.training_history)):
                raise ValueError("nonfinite training loss")
            return

        inference.load_posterior(str(checkpoint), workload.loss, workload.predict)
        count = request["samples"]
        if operation == "epistemic":
            prediction = inference.predict(batch, workload.predict, count, return_samples=True)
            tensors = {"logits": prediction["samples"]}
            summary = {
                "measurement": "predictive_epistemic_variance",
                "mean_marginal_variance": prediction["std"].square().mean().item(),
                "observation_noise_in_samples": False,
            }
        else:
            tensors = inference.posterior_samples(
                batch,
                count,
                site_names=request.get("sites"),
                max_output_bytes=artifact_limit,
            )
            tensors = {name: value for name, value in tensors.items() if name != "logits"}
            if operation == "layers":
                tensors = {
                    name: value
                    for name, value in tensors.items()
                    if name.rsplit(".", 1)[0] in request["measure_layers"]
                }
            if not tensors or (request.get("sites") and set(tensors) != set(request["sites"])):
                raise ValueError("requested sites do not match posterior latent sites")
            moments = inference.posterior_summary(tensors, max_covariance_dimension=32)
            summary = {
                "measurement": "parameter_variance",
                "sites": {
                    name: {
                        "layer": name.rsplit(".", 1)[0],
                        "mean_marginal_variance": row["variance"].mean().item(),
                        "covariance_representation": row["covariance_representation"],
                    }
                    for name, row in moments.items()
                },
            }
        if not all(torch.isfinite(value).all().item() for value in tensors.values()):
            raise ValueError("nonfinite posterior samples")
        inference.save_posterior_samples(
            tensors, str(directory / "samples.pt"), request["seed"], artifact_limit
        )
        (directory / "measurement.json").write_text(json.dumps(summary, allow_nan=False))
