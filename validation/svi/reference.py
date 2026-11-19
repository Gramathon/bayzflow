"""Independent Gaussian algebra and direct Pyro model; no Bayzflow imports."""

from __future__ import annotations

import math
from operator import attrgetter

import numpy as np
import pyro
import pyro.distributions as dist
import torch


def dataset(seed: int, dtype=torch.float32):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 811]))
    z = rng.normal(size=48)
    x = np.column_stack([z, 0.92 * z + rng.normal(scale=0.12, size=48)])
    design = np.column_stack([x, np.ones(48)])
    y = design @ np.array([0.8, -0.45, 0.2]) + rng.normal(0, 0.25, 48)
    return torch.tensor(design, dtype=dtype), torch.tensor(y, dtype=dtype)


def exact_posterior(x, y, mean, covariance, sigma):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not all(torch.isfinite(t).all() for t in (x, y, mean, covariance)):
        raise ValueError("non-finite Gaussian input")
    identity = torch.eye(mean.numel(), dtype=x.dtype, device=x.device)
    precision0 = torch.cholesky_solve(identity, torch.linalg.cholesky(covariance))
    chol = torch.linalg.cholesky(precision0 + x.T @ x / sigma**2)
    covariance_p = torch.cholesky_solve(identity, chol)
    mean_p = torch.cholesky_solve((precision0 @ mean + x.T @ y / sigma**2)[:, None], chol)[:, 0]
    return mean_p, covariance_p


def fidelity(mean, covariance, target_mean, target_covariance, design):
    tensors = (mean, covariance, target_mean, target_covariance, design)
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("non-finite posterior moments")
    lp = torch.linalg.cholesky(target_covariance)
    lq = torch.linalg.cholesky(covariance)
    delta = (target_mean - mean)[:, None]
    logdet = lambda l: 2 * l.diag().log().sum()
    kl = 0.5 * (
        torch.cholesky_solve(covariance, lp).trace()
        + (delta.T @ torch.cholesky_solve(delta, lp)).squeeze()
        - mean.numel()
        + logdet(lp)
        - logdet(lq)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    _, target_vectors = torch.linalg.eigh(target_covariance)
    pred = (design @ covariance * design).sum(-1)
    target_pred = (design @ target_covariance * design).sum(-1)
    sd = covariance.diag().sqrt()
    return {
        "mean_error": float((mean - target_mean).norm()),
        "normalized_mean_error": float((mean - target_mean).norm() / target_covariance.trace().sqrt()),
        "relative_covariance_error": float(
            (covariance - target_covariance).norm() / target_covariance.norm()
        ),
        "gaussian_kl": float(kl),
        "trace": float(covariance.trace()),
        "logdet": float(logdet(lq)),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "leading_subspace_agreement": float((eigenvectors[:, -1] @ target_vectors[:, -1]).square()),
        "correlation": (covariance / (sd[:, None] * sd[None, :])).tolist(),
        "predictive_variance_error": float((pred - target_pred).norm() / target_pred.norm()),
    }


def nll(logits, batch):
    return -dist.Normal(logits.reshape(-1), 0.25).log_prob(batch[1].reshape(-1)).sum()


def direct_model(mean, scale):
    def model(batch):
        w = pyro.sample(
            "0.weight", dist.Normal(mean[:2].reshape(1, 2), scale[:2].reshape(1, 2)).to_event(2)
        )
        b = pyro.sample("0.bias", dist.Normal(mean[2:], scale[2:]).to_event(1))
        logits = batch[0] @ w.T + b
        pyro.deterministic("logits", logits)
        pyro.factor("nll", -nll(logits, batch))
        return logits

    return model


def guide_moments(guide, batch):
    guide(batch)
    if hasattr(guide, "get_posterior"):
        q = guide.get_posterior(batch)
        cov = q.covariance_matrix if hasattr(q, "covariance_matrix") else torch.diag(q.variance)
        return q.mean.detach().reshape(-1).double(), cov.detach().double()
    names = [n for n, s in guide.prototype_trace.iter_stochastic_nodes() if not s["is_observed"]]
    loc = torch.cat([attrgetter(n)(guide.locs).reshape(-1) for n in names])
    scale = torch.cat([attrgetter(n)(guide.scales).reshape(-1) for n in names])
    return loc.detach().double(), torch.diag(scale.detach().double().square())
