"""Compare variational guides on a small Gaussian regression with a known posterior."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from torch.distributions import constraints
from torch.utils.data import DataLoader, TensorDataset

from bayzflow import BayzFlow, __version__
from bayzflow.guides import GuideRegistry, GuideSpec
from bayzflow.mcp.workloads import (
    regression_batch,
    regression_loss,
    regression_model,
    regression_predict,
)


@dataclass(frozen=True)
class CorrelatedConfig:
    init_scale: float = 0.2


class CorrelatedRegressionGuide(PyroModule):
    """A model-specific bivariate guide with a trainable Cholesky factor."""

    def __init__(self, init_scale: float):
        super().__init__()
        self.init_scale = init_scale

    def forward(self, batch):
        loc = pyro.param("correlated_loc", batch[0].new_zeros(2))
        scale = pyro.param(
            "correlated_scale",
            batch[0].new_full((2,), self.init_scale),
            constraint=constraints.positive,
        )
        lower = pyro.param("correlated_lower", batch[0].new_zeros(()))
        chol = torch.stack((scale[0], lower.new_zeros(()), lower, scale[1])).reshape(2, 2)
        posterior = dist.MultivariateNormal(loc, scale_tril=chol)
        return {"0.weight": pyro.sample("0.weight", posterior.expand((1,)).to_event(1))}


def correlated_guide(model, init_scale=0.2):
    return CorrelatedRegressionGuide(init_scale)


GUIDES = (
    GuideSpec("auto_normal"),
    GuideSpec("auto_diag"),
    GuideSpec("auto_lowrank", {"rank": 1}),
    GuideSpec("auto_fullrank"),
    GuideSpec("custom:correlated_regression"),
)


def evaluate(output: Path, seeds: list[int], steps: int, samples: int, kinds: list[str]) -> dict:
    torch.set_num_threads(1)
    batch = regression_batch()
    x, y = batch
    precision = torch.eye(x.shape[1]) + 4 * x.T @ x
    covariance = torch.linalg.inv(precision)
    mean = (4 * covariance @ x.T @ y).flatten()
    predictive_variance = (x @ covariance * x).sum(dim=1)
    registry = GuideRegistry()
    registry.register(
        "correlated_regression",
        correlated_guide,
        config_schema=CorrelatedConfig,
        version="1",
        capabilities={"covariance": "cholesky"},
    )
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for guide in GUIDES:
            if guide.kind not in kinds:
                continue
            model = BayzFlow(regression_model(), guide=guide, guide_registry=registry)
            model.selected_module_names = ["0"]
            model.prior_scales = {"0": 1.0}
            model.build_bayesian_model()
            checkpoint = output / f"{seed}-{guide.kind.replace(':', '_')}"
            began = time.perf_counter()
            with redirect_stdout(StringIO()):
                model.train_posterior(
                    DataLoader(TensorDataset(*batch), batch_size=len(x)),
                    regression_loss,
                    regression_predict,
                    num_epochs=steps,
                    seed=seed,
                    lr=0.025,
                    save_dir=str(checkpoint),
                    experiment_metadata={
                        "prior": "N(0,I)",
                        "observation_sd": 0.5,
                        "design": x.tolist(),
                        "observations": y.tolist(),
                        "learning_rate": 0.025,
                        "steps": steps,
                    },
                )
            train_seconds = time.perf_counter() - began
            # Exercise the saved guide and model rather than the just-trained instance.
            restored = BayzFlow(regression_model(), guide_registry=registry)
            restored.selected_module_names = ["0"]
            restored.prior_scales = {"0": 1.0}
            restored.build_bayesian_model()
            with redirect_stdout(StringIO()):
                restored.load_posterior(str(checkpoint), regression_loss, regression_predict)
            pyro.set_rng_seed(seed + 10_000)
            draws = restored.posterior_samples(batch, samples, site_names=["0.weight"])
            weights = draws["0.weight"].reshape(samples, 2)
            estimate = restored.posterior_summary(draws)["0.weight"]
            predictions = restored.predict(batch, regression_predict, 32, return_samples=True)
            conditional_mean = weights @ x.T
            assert torch.isfinite(predictions["samples"]).all()
            assert torch.isfinite(weights).all()
            artifact = restored.save_posterior_samples(
                draws, str(checkpoint / "posterior.pt"), seed + 10_000
            )
            history = model.training_history
            window = min(100, steps)
            rows.append(
                {
                    "guide": restored.guide_spec.to_dict(),
                    "seed": seed,
                    "train_seconds": train_seconds,
                    "elbo_loss_first_window": sum(history[:window]) / window,
                    "elbo_loss_last_window": sum(history[-window:]) / window,
                    "mean_l2_error": (weights.mean(0) - mean).norm().item(),
                    "covariance_frobenius_error": (estimate["covariance"] - covariance)
                    .norm()
                    .item(),
                    "covariance_representation": estimate["covariance_representation"],
                    "covariance": estimate["covariance"].tolist(),
                    "predictive_epistemic_rmse": (
                        conditional_mean.var(0, unbiased=False) - predictive_variance
                    )
                    .square()
                    .mean()
                    .sqrt()
                    .item(),
                    "sample_count": samples,
                    "posterior_sha256": artifact["artifact_sha256"],
                    "checkpoint_identity": restored.checkpoint_identity,
                }
            )
    report = {
        "environment": {
            "torch": str(torch.__version__),
            "pyro": pyro.__version__,
            "bayzflow": __version__,
        },
        "analytical_mean": mean.tolist(),
        "analytical_covariance": covariance.tolist(),
        "reverse_kl_diagonal_variances": precision.diagonal().reciprocal().tolist(),
        "observation_variance": 0.25,
        "steps": steps,
        "runs": rows,
        "max_mean_l2_error": max(row["mean_l2_error"] for row in rows),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--guide", action="append", choices=[guide.kind for guide in GUIDES])
    args = parser.parse_args()
    if args.steps < 1 or not 2 <= args.samples <= 1000:
        parser.error("use positive steps and 2–1000 samples")
    evaluate(
        args.output,
        args.seed or [7, 19, 43],
        args.steps,
        args.samples,
        args.guide or [guide.kind for guide in GUIDES],
    )


if __name__ == "__main__":
    main()
