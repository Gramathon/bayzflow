# core/wrapper.py
"""
Generic BayesianWrapper for PyTorch + Pyro models.

Designed to work with:
- A base PyTorch/PyroModule model (e.g. BayesianLSTM)
- A Pyro model callable (often the same as the model)
- An AutoGuide (e.g. AutoLowRankMultivariateNormal)

Features:
- SVI training loop (+ optional early stopping)
- Posterior predictive sampling via pyro.infer.Predictive
- Simple uncertainty decomposition helper for regression
- Checkpointing: model_state + Pyro param store (+ optional optim_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List
from pathlib import Path
from examples.fx.fx_utils.forecast import forecast_paths_primary_only
import torch
import torch.nn as nn

import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import PyroOptim
from pyro.infer.autoguide import (
    AutoLowRankMultivariateNormal,
    AutoNormal,
    AutoDiagonalNormal,
)


@dataclass
class UncertaintyReport:
    mean: torch.Tensor
    var_total: torch.Tensor
    var_epistemic: Optional[torch.Tensor] = None
    var_aleatoric: Optional[torch.Tensor] = None


class BayesianWrapper:
    def __init__(
        self,
        model: Optional[nn.Module],
        pyro_model: Callable[..., Any],
        guide: Optional[Callable[..., Any]] = None,
        guide_factory: Optional[Callable[[Callable[..., Any]], Any]] = None,
        *,
        default_guide_kind: str = "auto_lowrank",
        default_guide_kwargs: Optional[Dict[str, Any]] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        model:
            Base PyTorch/PyroModule model (e.g. BayesianLSTM). Can be None if
            your pyro_model doesn't close over a shared module.
        pyro_model:
            Callable Pyro model, e.g. your BayesianLSTM instance itself:
                pyro_model(x, y=None)
        guide:
            Optional instantiated guide.
        guide_factory:
            If guide is None, will be called as guide_factory(pyro_model)
            to construct a guide.
        default_guide_kind:
            If neither guide nor guide_factory is given, use this kind:
            "auto_lowrank", "auto_normal", "auto_diag".
        default_guide_kwargs:
            Extra kwargs for the default AutoGuide (e.g. rank=16).
        device:
            "cpu" or "cuda[:idx]".
        """
        self.model = model
        self.pyro_model = pyro_model

        # Normalize device and move model there
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)

        if default_guide_kwargs is None:
            default_guide_kwargs = {}
        self.default_guide_kind = default_guide_kind
        self.default_guide_kwargs = default_guide_kwargs

        # --------------- Guide creation --------------------
        if guide is not None:
            self.guide = guide

        elif guide_factory is not None:
            self.guide = guide_factory(self.pyro_model)

        else:
            kind = default_guide_kind.lower()
            if kind == "auto_lowrank":
                self.guide = AutoLowRankMultivariateNormal(
                    self.pyro_model, **default_guide_kwargs
                )
            elif kind in ("auto_normal", "auto_norm"):
                self.guide = AutoNormal(self.pyro_model, **default_guide_kwargs)
            elif kind in ("auto_diag", "auto_diagonal"):
                self.guide = AutoDiagonalNormal(
                    self.pyro_model, **default_guide_kwargs
                )
            else:
                raise ValueError(f"Unknown default_guide_kind: {default_guide_kind}")

        self.svi: Optional[SVI] = None
        self.optim: Optional[PyroOptim] = None

    # ----------------------------------------------------
    # Device helper
    # ----------------------------------------------------
    def _param_device(self) -> torch.device:
        """
        Use the model's parameter device if it has nn.Parameters;
        otherwise fall back to self.device.
        """
        if self.model is not None:
            for p in self.model.parameters():
                return p.device  # first param's device
        return self.device

    # ----------------------------------------------------
    # Plain SVI training (no early stopping)
    # ----------------------------------------------------
    def fit(
        self,
        train_loader: Iterable,
        num_epochs: int,
        lr: float = 1e-3,
        svi_loss: Optional[Trace_ELBO] = None,
        optimizer_cls: Callable[..., PyroOptim] = pyro.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_to_args_kwargs: Optional[
            Callable[[Any], Tuple[Tuple[Any, ...], Dict[str, Any]]]
        ] = None,
        progress: bool = True,
    ) -> None:
        """
        Plain SVI training (no early stopping).

        train_loader should yield:
          - (x, y) tuples, or
          - dicts that can be unpacked as **kwargs to pyro_model.
        """
        device = self._param_device()

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": lr}
        if svi_loss is None:
            svi_loss = Trace_ELBO()

        pyro.clear_param_store()
        self.optim = optimizer_cls(optimizer_kwargs)
        self.svi = SVI(self.pyro_model, self.guide, self.optim, loss=svi_loss)

        def default_batch_to_args_kwargs(batch):
            if isinstance(batch, dict):
                kwargs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                return (), kwargs
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                return (x, y), {}
            else:
                raise ValueError(
                    "Unsupported batch type. Provide custom `batch_to_args_kwargs`."
                )

        if batch_to_args_kwargs is None:
            batch_to_args_kwargs = default_batch_to_args_kwargs

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                args, kwargs = batch_to_args_kwargs(batch)
                try:
                    loss = self.svi.step(*args, **kwargs)
                except Exception as e:
                    print("[BayesianWrapper.fit] Error in svi.step()")
                    print("  args shapes:", [a.shape for a in args if isinstance(a, torch.Tensor)])
                    print("  device:", device)
                    raise
                epoch_loss += loss
                n_batches += 1

            if n_batches > 0:
                epoch_loss /= n_batches

            if progress:
                print(f"[Epoch {epoch:03d}] mean loss: {epoch_loss:.4f}")

    # ----------------------------------------------------
    # Early-stopping training
    # ----------------------------------------------------
    def fit_with_early_stopping(
        self,
        train_loader: Iterable,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        num_epochs: int,
        lr: float = 1e-3,
        patience: int = 20,
        svi_loss: Optional[Trace_ELBO] = None,
        optimizer_cls: Callable[..., PyroOptim] = pyro.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        progress: bool = True,
    ) -> Dict[str, list]:
        """
        Train with early stopping on validation ELBO and optional checkpointing.
        """
        device = self._param_device()

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": lr}
        if svi_loss is None:
            svi_loss = Trace_ELBO()

        pyro.clear_param_store()
        self.optim = optimizer_cls(optimizer_kwargs)
        self.svi = SVI(self.pyro_model, self.guide, self.optim, loss=svi_loss)

        best_val = float("inf")
        pat = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, num_epochs + 1):
            # ---- train ----
            if self.model is not None:
                self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                try:
                    loss = self.svi.step(xb, yb)
                except Exception as e:
                    print("[BayesianWrapper.fit_with_early_stopping] Error in svi.step()")
                    print("  xb.shape:", xb.shape, "yb.shape:", yb.shape)
                    print("  xb.device:", xb.device, "model.device:", device)
                    raise
                epoch_loss += loss
                n_batches += 1

            if n_batches > 0:
                epoch_loss /= n_batches

            # ---- val ----
            if self.model is not None:
                self.model.eval()
            with torch.no_grad():
                try:
                    val_loss = self.svi.evaluate_loss(
                        x_val.to(device), y_val.to(device)
                    )
                except Exception as e:
                    print("[BayesianWrapper.fit_with_early_stopping] Error in evaluate_loss()")
                    print("  x_val.shape:", x_val.shape, "y_val.shape:", y_val.shape)
                    print("  x_val.device:", x_val.device, "model.device:", device)
                    raise

            history["train_loss"].append(float(epoch_loss))
            history["val_loss"].append(float(val_loss))

            if progress:
                print(
                    f"[Epoch {epoch:03d}] train ELBO: {epoch_loss:.2f} | "
                    f"val ELBO: {val_loss:.2f}"
                )

            # ---- Early stopping + checkpoint ----
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                pat = 0
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, extra=extra_meta)
            else:
                pat += 1
                if pat >= patience:
                    if progress:
                        print(f"⏹ early stopping (patience={patience})")
                    break

        return history

    # ----------------------------------------------------
    # Posterior sampling
    # ----------------------------------------------------
    def sample_posterior(
        self,
        *model_args: Any,
        num_samples: int = 100,
        return_sites: Optional[List[str]] = None,
        **model_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Draw posterior predictive samples using pyro.infer.Predictive.

        Example for your BayesianLSTM:
            samples = wrapper.sample_posterior(
                x_test, y=None, num_samples=200, return_sites=["obs"]
            )

        Returns dict of site_name -> tensor with leading dim = num_samples.
        """
        if return_sites is None:
            return_sites = ["obs"]

        device = self._param_device()

        predictive = Predictive(
            self.pyro_model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=return_sites,
        )

        model_args = tuple(
            a.to(device) if isinstance(a, torch.Tensor) else a
            for a in model_args
        )
        model_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in model_kwargs.items()
        }

        samples = predictive(*model_args, **model_kwargs)
        return samples

    # ----------------------------------------------------
    # Uncertainty helper (regression)
    # ----------------------------------------------------
    def uncertainty_report(
        self,
        y_samples: torch.Tensor,
        aleatoric_var_samples: Optional[torch.Tensor] = None,
        dim: int = 0,
    ) -> UncertaintyReport:
        """
        Basic uncertainty decomposition from posterior samples.

        y_samples: [S, ...] where S = num_samples.
        """
        mean = y_samples.mean(dim=dim)
        var_total = y_samples.var(dim=dim, unbiased=False)

        var_ep = None
        var_al = None
        if aleatoric_var_samples is not None:
            var_al = aleatoric_var_samples.mean(dim=dim)
            var_ep = (var_total - var_al).clamp_min(0.0)

        return UncertaintyReport(
            mean=mean,
            var_total=var_total,
            var_epistemic=var_ep,
            var_aleatoric=var_al,
        )

    # ----------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------
    def save_checkpoint(
        self,
        path: str | Path,
        extra: Optional[Dict[str, Any]] = None,
        include_optim: bool = False,
    ) -> None:
        """
        Save:
        - model_state (if model is not None)
        - Pyro param_store state
        - optional optimizer state
        - extra metadata
        """
        if extra is None:
            extra = {}

        ckpt: Dict[str, Any] = {"extra": extra}

        if self.model is not None:
            ckpt["model_state"] = self.model.state_dict()

        ckpt["param_store_state"] = pyro.get_param_store().get_state()

        if include_optim and self.optim is not None:
            ckpt["optim_state"] = self.optim.get_state()

        torch.save(ckpt, path)
        print(f"[BayesianWrapper] Saved checkpoint to {path}")

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: Optional[str | torch.device] = None,
        load_optim: bool = False,
    ) -> Dict[str, Any]:
        """
        Load:
        - model_state into self.model
        - Pyro param_store state
        - optimizer state (if load_optim=True and present)

        Returns `extra` metadata dict.
        """
        if map_location is None:
            map_location = self._param_device()

        ckpt = torch.load(path, map_location=map_location)

        if self.model is not None and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self._param_device())

        if "param_store_state" in ckpt:
            pyro.clear_param_store()
            pyro.get_param_store().set_state(ckpt["param_store_state"])

        if load_optim and self.optim is not None and "optim_state" in ckpt:
            self.optim.set_state(ckpt["optim_state"])

        extra = ckpt.get("extra", {})
        print(f"[BayesianWrapper] Loaded checkpoint from {path}")
        return extra
