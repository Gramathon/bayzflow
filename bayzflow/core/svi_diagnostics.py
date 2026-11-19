"""Bounded scientific diagnostics for the engine's trusted Pyro guides."""

from __future__ import annotations

from operator import attrgetter
from typing import Any

import pyro
import torch
from torch.distributions import biject_to


def engine_diagnostics(engine: Any, batch: Any, max_dimension: int = 256) -> dict[str, Any]:
    if (
        isinstance(max_dimension, bool)
        or not isinstance(max_dimension, int)
        or not 1 <= max_dimension <= 2048
    ):
        raise ValueError("max_dimension must be between 1 and 2048")
    if engine.guide is None:
        raise RuntimeError("SVI not set up")
    guide = engine.guide
    with torch.no_grad():
        # Check the model's latent dimension before a fresh full-rank guide can
        # allocate its quadratic covariance parameters.
        if hasattr(guide, "prototype_trace") and guide.prototype_trace is None:
            trace = pyro.poutine.trace(engine._pyro_model).get_trace(batch)
            size = sum(
                biject_to(site["fn"].support).inv(site["value"]).numel()
                for _, site in trace.iter_stochastic_nodes()
                if not site["is_observed"]
            )
            if size > max_dimension:
                raise ValueError("guide dimension exceeds diagnostics dimension limit")
        guide(batch)
        sites = []
        dimension = 0
        if getattr(guide, "prototype_trace", None) is None:
            return {
                "training": {"guide": engine.guide_spec.to_dict()},
                "posterior": {"representation": "unavailable_custom_guide"},
                "numerical_warnings": ["Custom guide does not expose a prototype trace"],
            }
        for name, site in guide.prototype_trace.iter_stochastic_nodes():
            if site["is_observed"]:
                continue
            shape = list(site["value"].shape)
            unconstrained = biject_to(site["fn"].support).inv(site["value"])
            count = unconstrained.numel()
            sites.append(
                {
                    "name": name,
                    "shape": shape,
                    "unconstrained_shape": list(unconstrained.shape),
                    "start": dimension,
                    "stop": dimension + count,
                    "support": str(site["fn"].support),
                }
            )
            dimension += count
        if dimension > max_dimension:
            raise ValueError("guide dimension exceeds diagnostics dimension limit")
        representation = {
            "coordinate_system": "unconstrained_latent",
            "sites": sites,
            "dimension": dimension,
            "sample_count": 0,
            "native_weight_equivalence": "only for real-valued identity support transforms",
        }
        if hasattr(guide, "get_posterior"):
            q = guide.get_posterior(batch)
            mean = q.mean.reshape(-1)
            if hasattr(q, "cov_factor"):
                factor, diagonal = q.cov_factor, q.cov_diag
                covariance = factor @ factor.T + torch.diag(diagonal)
                representation.update(
                    representation="factor_plus_diagonal",
                    factor=factor.tolist(),
                    diagonal=diagonal.tolist(),
                )
            elif hasattr(q, "scale_tril"):
                covariance = q.scale_tril @ q.scale_tril.T
                representation.update(representation="cholesky", scale_tril=q.scale_tril.tolist())
            else:
                covariance = torch.diag(q.variance.reshape(-1))
                representation.update(
                    representation="diagonal", diagonal=q.variance.reshape(-1).tolist()
                )
        elif hasattr(guide, "locs") and hasattr(guide, "scales"):
            mean = torch.cat([attrgetter(s["name"])(guide.locs).reshape(-1) for s in sites])
            variance = torch.cat(
                [attrgetter(s["name"])(guide.scales).reshape(-1).square() for s in sites]
            )
            covariance = torch.diag(variance)
            representation.update(representation="site_diagonal", diagonal=variance.tolist())
        else:
            representation.update(representation="unavailable_custom_guide")
            mean = covariance = None
        if mean is not None:
            if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
                raise FloatingPointError("non-finite guide moments")
            representation.update(
                mean=mean.tolist(),
                marginal_variance=covariance.diag().tolist(),
                covariance=covariance.tolist(),
                eigenvalues=torch.linalg.eigvalsh(covariance).tolist(),
            )
    state = engine.optimizer.get_state() if engine.optimizer is not None else {}
    history = engine.training_history
    window = min(10, len(history))
    loss_windows = None
    if window:
        first = sum(history[:window]) / window
        recent = sum(history[-window:]) / window
        loss_windows = {
            "window_epochs": window,
            "first_mean": first,
            "recent_mean": recent,
            "relative_change": (recent - first) / max(abs(first), 1.0),
        }
    return {
        "training": {
            "selected_layers": list(engine.selected_module_names),
            "deterministic_update_policy": engine.deterministic_update_policy,
            "deterministic_parameters": [
                {"name": n, "shape": list(p.shape), "trainable": p.requires_grad}
                for n, p in engine.bayes_model.named_parameters()
            ],
            "priors": [
                {"name": n, "shape": list(b.shape), "value": b.detach().cpu().tolist()}
                for n, b in engine.bayes_model.named_buffers()
                if n.endswith(("_loc", "_scale"))
            ],
            "guide": engine.guide_spec.to_dict(),
            "optimizer": engine.optimization_config,
            "optimizer_parameter_names": sorted(state),
            "optimizer_steps": {
                name: [float(slot["step"]) for slot in entry["state"].values() if "step" in slot]
                for name, entry in state.items()
            },
            "svi_steps": engine.svi_steps,
            "elbo_per_epoch": list(engine.training_history),
            "elbo_window_summaries": loss_windows,
            "convergence_note": "ELBO stability does not establish posterior accuracy",
        },
        "parameter_updates": dict(engine.parameter_updates),
        "guide_parameter_updates": getattr(engine, "guide_parameter_updates", {}),
        "posterior": representation,
        "numerical_warnings": [],
        "optimizer_continuation": "in_process_only",
    }
