"""CPU scientific comparison. Run with an explicit --source and interpreter."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "svi"))
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO, autoguide
from reference import (
    dataset,
    direct_model,
    exact_posterior,
    fidelity,
    guide_moments,
    nll,
)


def run(args):
    if args.source:
        sys.path.insert(0, args.source)
    import bayzflow.core.engine as engine_module
    from bayzflow import BayzFlow

    torch.set_num_threads(1)
    rows = []
    for seed in args.seeds:
        dtype = getattr(torch, args.dtype)
        design, y = dataset(seed, dtype)
        torch.manual_seed(seed + 8900)
        base = torch.nn.Sequential(torch.nn.Linear(2, 1)).to(dtype)
        mean = torch.cat([base[0].weight.detach().flatten(), base[0].bias.detach()])
        scale = torch.full_like(mean, args.prior_sd)
        # General invertible feature change, with exactly transformed prior and moments.
        transform = torch.eye(3, dtype=dtype)
        if args.features != "original":
            if args.features == "standardized":
                transform[:2, :2] = torch.diag(1 / design[:, :2].std(0))
            else:
                _, singular, vh = torch.linalg.svd(design[:, :2], full_matrices=False)
                transform[:2, :2] = vh.T @ torch.diag(design.new_tensor(48.0).sqrt() / singular)
            # Isotropic original prior stays diagonal after SVD rotation/scaling.
            inverse = torch.linalg.solve(transform, torch.eye(3, dtype=dtype))
            mean = inverse @ mean
            scale = (inverse @ torch.diag(scale.square()) @ inverse.T).diag().sqrt()
        batch = (design[:, :2] @ transform[:2, :2], y[:, None])
        changed_design = design @ transform
        pm, pc = exact_posterior(
            changed_design.double(),
            y.double(),
            mean.double(),
            torch.diag(scale.double().square()),
            0.25,
        )
        for kind in args.guides:
            for implementation in ("direct", "bayzflow"):
                pyro.clear_param_store()
                pyro.set_rng_seed(seed + 8900)
                kwargs = {"init_scale": args.init_scale}
                if kind == "auto_lowrank":
                    kwargs["rank"] = 2
                start = time.monotonic()
                if implementation == "direct":
                    model = direct_model(mean, scale)
                    cls = {
                        "auto_normal": autoguide.AutoNormal,
                        "auto_diag": autoguide.AutoDiagonalNormal,
                        "auto_lowrank": autoguide.AutoLowRankMultivariateNormal,
                        "auto_fullrank": autoguide.AutoMultivariateNormal,
                    }[kind]
                    guide = cls(model, **kwargs)
                    svi = SVI(model, guide, pyro.optim.Adam({"lr": args.lr}), Trace_ELBO())
                    scope = contextlib.nullcontext
                else:
                    base = torch.nn.Sequential(torch.nn.Linear(2, 1)).to(dtype)
                    base[0].weight.data.copy_(mean[:2].reshape(1, 2))
                    base[0].bias.data.copy_(mean[2:])
                    pyro.set_rng_seed(seed + 8900)
                    bf = BayzFlow(base, guide=kind, guide_kwargs=kwargs)
                    bf.selected_module_names = ["0"]
                    bf.prior_scales = {"0": args.prior_sd}
                    bf.build_bayesian_model()
                    bf.bayes_model[0]._weight_scale.copy_(scale[:2].reshape(1, 2))
                    bf.bayes_model[0]._bias_scale.copy_(scale[2:])
                    bf.setup_svi(nll, lambda m, b: m(b[0]), lr=args.lr)
                    guide, svi = bf.guide, bf.svi
                    scope = lambda bf=bf: engine_module._engine_param_store(bf)
                checkpoints = sorted({0, 100, 400, 1000, 2000, args.steps})
                losses, curve = [], {}
                with scope():
                    # Initialization consumes identical random draws in both paths.
                    guide(batch)
                    for step in range(args.steps + 1):
                        if step in checkpoints:
                            qm, qc = guide_moments(guide, batch)
                            t = transform.double()
                            curve[str(step)] = fidelity(
                                t @ qm, t @ qc @ t.T, t @ pm, t @ pc @ t.T, design.double()
                            )
                            curve[str(step)].update(
                                mean=(t @ qm).tolist(), covariance=(t @ qc @ t.T).tolist()
                            )
                        if step < args.steps:
                            loss = float(svi.step(batch))
                            if not torch.isfinite(torch.tensor(loss)):
                                raise ValueError("nonfinite ELBO")
                            losses.append(loss)
                    qm, qc = guide_moments(guide, batch)
                    # Predictive diagnostics on fitting rows, explicitly not held-out calibration.
                    pv = (changed_design.double() @ qc * changed_design.double()).sum(-1) + 0.25**2
                    residual = y.double() - changed_design.double() @ qm
                    coverage = float(
                        (residual.abs() <= 1.6448536269514722 * pv.sqrt()).double().mean()
                    )
                    pll = float(
                        torch.distributions.Normal(changed_design.double() @ qm, pv.sqrt())
                        .log_prob(y.double())
                        .mean()
                    )
                rows.append(
                    {
                        "seed": seed,
                        "guide": kind,
                        "implementation": implementation,
                        "curve": curve,
                        "losses": losses,
                        "seconds": time.monotonic() - start,
                        "fitting_coverage_90": coverage,
                        "fitting_predictive_log_likelihood": pll,
                    }
                )
    source = Path(inspect.getfile(engine_module))
    return {
        "configuration": vars(args),
        "engine_path": str(source),
        "engine_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "pyro": pyro.__version__,
        "rows": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source")
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    p.add_argument("--guides", nargs="+", default=["auto_fullrank"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--prior-sd", type=float, default=0.5)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument(
        "--features", choices=["original", "standardized", "whitened"], default="original"
    )
    args = p.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        result = run(args)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
