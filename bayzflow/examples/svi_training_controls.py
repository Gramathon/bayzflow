"""Executable small examples: frozen/joint backbones, all selected Bayesian layers."""

from __future__ import annotations

import json

import torch

from bayzflow import BayzFlow


def predict(model, batch):
    return model(batch[0])


def loss(logits, batch):
    # Full-data summed likelihood. Minibatching needs explicit likelihood scaling.
    return -torch.distributions.Normal(logits, 0.25).log_prob(batch[1]).sum()


def example(policy: str, all_selected: bool = False) -> dict:
    torch.manual_seed(71)
    x = torch.randn(24, 2)
    batch = (x, x[:, :1] * 0.5)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    # Supply pretrained weights before constructing the engine for a pretrained backbone.
    bf = BayzFlow(model, guide="auto_fullrank", deterministic_update_policy=policy)
    bf.selected_module_names = ["0", "2"] if all_selected else ["2"]
    bf.build_bayesian_model(default_prior_scale=0.5)
    bf.setup_svi(loss, predict, lr=0.02)
    bf.fit([batch], num_epochs=5)
    bf.fit([batch], num_epochs=5, continue_optimizer=True)
    report = bf.diagnostics(batch)
    samples = bf.posterior_samples(batch, num_samples=64, site_names=["2.weight", "2.bias"])
    summary = bf.posterior_summary(samples)
    report["native_sample_shapes"] = {k: list(v["covariance"].shape) for k, v in summary.items()}
    return report


if __name__ == "__main__":
    torch.set_num_threads(1)
    for policy, selected in [("freeze", False), ("joint", False), ("freeze", True)]:
        print(json.dumps(example(policy, selected), indent=2))
