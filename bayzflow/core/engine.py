#!/usr/bin/env python3
"""
bayzflow.py (rewritten minimal extension)

Adds two high-level functions you can call:
  1) train_posterior(...)  -> runs SVI once and (optionally) saves posterior
  2) measure_layers(...)   -> loads posterior (if needed), runs MC Predictive,
                              records per-layer epistemic trajectory stats via hooks

Design goals:
  - Keep your existing BayzFlow API working
  - No retraining per layer; one posterior, many measurement passes
  - Measurement is lightweight: store scalar summaries per layer per MC sample
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal, AutoLowRankMultivariateNormal


# ============================================================================
# 1. Module information / discovery
# ============================================================================

@dataclass
class ModuleInfo:
    name: str
    module: nn.Module


_DEFAULT_INCLUDE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
)


def find_candidate_modules(
    model: nn.Module,
    include_types: Sequence[Type[nn.Module]] = _DEFAULT_INCLUDE_TYPES,
    exclude_names: Optional[Iterable[str]] = None,
) -> List[ModuleInfo]:
    exclude_names = set(exclude_names or [])
    candidates: List[ModuleInfo] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if not isinstance(module, tuple(include_types)):
            continue
        if any(ex in name for ex in exclude_names):
            continue
        candidates.append(ModuleInfo(name=name, module=module))
    return candidates


def print_candidates(candidates: Sequence[ModuleInfo]) -> None:
    for idx, info in enumerate(candidates):
        print(f"[{idx:03}] {info.name:60} {info.module.__class__.__name__}")


def select_by_pattern(candidates: Sequence[ModuleInfo], patterns: Sequence[str]) -> List[ModuleInfo]:
    patterns = list(patterns)
    return [info for info in candidates if any(p in info.name for p in patterns)]


# ============================================================================
# 2. Bayesian wrapper layers
# ============================================================================
class BayesLinear(PyroModule):
    def __init__(self, name: str, base_linear: nn.Linear, prior_scale: float = 0.1):
        super().__init__()

        # Store base shape
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        w_mean = base_linear.weight.detach()
        b_mean = base_linear.bias.detach() if base_linear.bias is not None else None

        # IMPORTANT: use pyro.sample with explicit unique names
        self._weight_prior = dist.Normal(w_mean, prior_scale).to_event(2)
        self._bias_prior = (
            dist.Normal(b_mean, prior_scale).to_event(1)
            if b_mean is not None else None
        )

        self._name = name  # unique layer name for namespacing

    def forward(self, x):
        weight = pyro.sample(f"{self._name}.weight", self._weight_prior)
        bias = None
        if self._bias_prior is not None:
            bias = pyro.sample(f"{self._name}.bias", self._bias_prior)
        return F.linear(x, weight, bias)

class BayesConvNd(PyroModule):
    conv_dim: int = 2

    def __init__(self, base_conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], prior_scale: float = 0.1):
        super().__init__()
        w_mean = base_conv.weight.detach()
        b_mean = base_conv.bias.detach() if base_conv.bias is not None else None

        self.in_channels = base_conv.in_channels
        self.out_channels = base_conv.out_channels
        self.kernel_size = base_conv.kernel_size
        self.stride = base_conv.stride
        self.padding = base_conv.padding
        self.dilation = base_conv.dilation
        self.groups = base_conv.groups

        self.weight = PyroSample(dist.Normal(w_mean, prior_scale).to_event(w_mean.dim()))
        self.bias = PyroSample(dist.Normal(b_mean, prior_scale).to_event(1)) if b_mean is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_dim == 1:
            return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.conv_dim == 2:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.conv_dim == 3:
            return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        raise RuntimeError(f"Unsupported conv_dim={self.conv_dim}")


class BayesConv1d(BayesConvNd):
    conv_dim = 1


class BayesConv2d(BayesConvNd):
    conv_dim = 2


class BayesConv3d(BayesConvNd):
    conv_dim = 3


class BayesConvTranspose3d(PyroModule):
    def __init__(self, base_conv: nn.ConvTranspose3d, prior_scale: float = 0.1):
        super().__init__()
        w_mean = base_conv.weight.detach()
        b_mean = base_conv.bias.detach() if base_conv.bias is not None else None

        self.in_channels = base_conv.in_channels
        self.out_channels = base_conv.out_channels
        self.kernel_size = base_conv.kernel_size
        self.stride = base_conv.stride
        self.padding = base_conv.padding
        self.output_padding = base_conv.output_padding
        self.groups = base_conv.groups
        self.dilation = base_conv.dilation

        self.weight = PyroSample(dist.Normal(w_mean, prior_scale).to_event(w_mean.dim()))
        self.bias = PyroSample(dist.Normal(b_mean, prior_scale).to_event(1)) if b_mean is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )


# ============================================================================
# 3. Utility: dotted path resolution & safe wrapping
# ============================================================================

_BAYES_TYPES = (BayesLinear, BayesConv1d, BayesConv2d, BayesConv3d, BayesConvTranspose3d)


def get_parent_and_child(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def wrap_module_with_prior(
    full_name: str,
    module: nn.Module,
    prior_scale: float = 0.1,
) -> nn.Module:
    if isinstance(module, _BAYES_TYPES):
        return module

    if isinstance(module, nn.Linear):
        return BayesLinear(full_name, module, prior_scale=prior_scale)

    if isinstance(module, nn.Conv1d):
        return BayesConv1d(module, prior_scale=prior_scale)

    if isinstance(module, nn.Conv2d):
        return BayesConv2d(module, prior_scale=prior_scale)

    if isinstance(module, nn.Conv3d):
        return BayesConv3d(module, prior_scale=prior_scale)

    if isinstance(module, nn.ConvTranspose3d):
        return BayesConvTranspose3d(module, prior_scale=prior_scale)

    return module


def attach_priors(
    model: nn.Module,
    selected_names: Sequence[str],
    prior_scale: float = 0.1,
    per_layer_scale: Optional[Dict[str, float]] = None,
) -> nn.Module:
    per_layer_scale = per_layer_scale or {}

    for full_name in selected_names:
        parent, child_name = get_parent_and_child(model, full_name)
        child = getattr(parent, child_name)

        scale = per_layer_scale.get(full_name, prior_scale)
        wrapped = wrap_module_with_prior(full_name, child, prior_scale=scale)

        if wrapped is child:
            print(f"[BayzFlow] Warning: '{full_name}' not wrapped (unsupported: {type(child)}).")
        else:
            setattr(parent, child_name, wrapped)

    return model


# ============================================================================
# 4. Prior calibration
# ============================================================================

def calibrate_prior_scales(
    model: nn.Module,
    module_names: Sequence[str],
    calib_loader: torch.utils.data.DataLoader,
    forward_fn: Callable[[nn.Module, Any], torch.Tensor],
    num_passes: int = 3,
    percentile: float = 0.9,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    model.eval()
    if device is not None:
        model.to(device)

    name_to_module = dict(model.named_modules())
    missing = [n for n in module_names if n not in name_to_module]
    if missing:
        raise KeyError(f"Modules not found for calibration: {missing}")

    stats: Dict[str, List[float]] = {n: [] for n in module_names}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach()
            if device is not None:
                x = x.to("cpu")
            stats[name].append(x.reshape(-1).std().item())
        return hook

    for name in module_names:
        hooks.append(name_to_module[name].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for _ in range(num_passes):
            for batch in calib_loader:
                if device is not None and isinstance(batch, dict):
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                _ = forward_fn(model, batch)

    for h in hooks:
        h.remove()

    out: Dict[str, float] = {}
    for name, vals in stats.items():
        if not vals:
            print(f"[BayzFlow] Warning: no stats for '{name}', using 0.1")
            out[name] = 0.1
        else:
            out[name] = float(np.percentile(vals, percentile * 100.0))
    return out


# ============================================================================
# 5. Layer measurement utilities (NEW)
# ============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_device_batch(batch: Any, device: Optional[torch.device]) -> Any:
    if device is None:
        return batch
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch

class _LayerPerExampleMCRecorder:
    """
    Computes per-example epistemic dispersion per layer:
      u[b] = RMS over units of std_mc(output[b, ...])

    Streaming Welford moments across MC samples:
      For each layer we maintain per-example:
        mean[b], M2[b], count
      where mean[b] and M2[b] are scalars computed from a reduced representation
      of the layer output per example.

    Reduction choice (important):
      We reduce output[b] -> scalar vector v[b] = output[b].reshape(-1).float()
      Then keep per-example second moment over elements:
         m1[b] = mean(v[b])
         m2[b] = mean(v[b]^2)
      But for epistemic we need std across MC of v-elements, not of mean.
      So instead we compute per-example RMS of v[b]:
         r[b] = sqrt(mean(v[b]^2))
      Then measure std_mc(r[b]) across MC samples.

    This is cheap, stable, and per-example.
    """
    def __init__(self, layer_names, model):
        self.layer_names = list(layer_names)
        self.model = model
        self.name_to_module = dict(model.named_modules())
        missing = [n for n in self.layer_names if n not in self.name_to_module]
        if missing:
            raise KeyError(f"[BayzFlow] measure_layers: modules not found: {missing}")

        self.handles = []
        self.count = 0

        # layer -> running mean per example (Tensor [B])
        self.mean = {}
        # layer -> running M2 per example (Tensor [B])
        self.M2 = {}
        # layer -> last seen batch size (int)
        self.B = None

    def _hook(self, name: str):
        def fn(module, inputs, output):
            y = output
            if isinstance(y, (tuple, list)):
                y = y[0]
            if not torch.is_tensor(y):
                return
            with torch.no_grad():
                t = y.detach().float()

                # ensure batch dimension exists
                if t.dim() == 1:
                    t = t.unsqueeze(0)  # [1, F]
                B = t.shape[0]
                if self.B is None:
                    self.B = B
                elif self.B != B:
                    raise RuntimeError(f"[BayzFlow] Batch size changed within Predictive run: {self.B} -> {B}")

                # per-example scalar summary of layer output for this MC sample:
                # RMS over units/positions
                r = torch.sqrt(torch.mean(t.reshape(B, -1) ** 2, dim=1))  # [B]

                if name not in self.mean:
                    self.mean[name] = torch.zeros_like(r)
                    self.M2[name] = torch.zeros_like(r)

                # Welford update for this layer
                n1 = self.count + 1
                delta = r - self.mean[name]
                self.mean[name] = self.mean[name] + delta / n1
                delta2 = r - self.mean[name]
                self.M2[name] = self.M2[name] + delta * delta2
        return fn

    def attach(self):
        self.handles = []
        for n in self.layer_names:
            m = self.name_to_module[n]
            self.handles.append(m.register_forward_hook(self._hook(n)))
        return self

    def step(self):
        # call once per MC sample (after forward). We'll increment count outside.
        self.count += 1

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def results(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns per-layer per-example arrays:
          layer -> {"mc_mean": [B], "mc_std": [B]}
        """
        out = {}
        for layer in self.layer_names:
            mu = self.mean.get(layer, None)
            M2 = self.M2.get(layer, None)
            if mu is None or M2 is None or self.count < 2:
                out[layer] = {"mc_mean": None, "mc_std": None}
                continue
            var = M2 / (self.count - 1)
            std = torch.sqrt(torch.clamp(var, min=0.0))
            out[layer] = {
                "mc_mean": mu.detach().cpu().numpy(),
                "mc_std": std.detach().cpu().numpy(),
            }
        return out

class _LayerScalarRecorder:
    """
    Records scalar summaries per layer per MC sample.

    For each layer, per sample we store a dict of scalars:
      - out_mean
      - out_std
      - out_abs_mean
      - out_l2_mean

    Then we aggregate across MC samples:
      - mean across samples
      - std across samples
      - p50, p90 across samples (optional)
    """
    def __init__(self, layer_names: Sequence[str], model: nn.Module):
        self.layer_names = list(layer_names)
        self.model = model
        self.name_to_module = dict(model.named_modules())
        missing = [n for n in self.layer_names if n not in self.name_to_module]
        if missing:
            raise KeyError(f"[BayzFlow] measure_layers: modules not found: {missing}")

        self.handles: List[Any] = []
        self._sample_idx = -1

        # layer -> list[dict] (one per MC sample)
        self.samples: Dict[str, List[Dict[str, float]]] = {n: [] for n in self.layer_names}

    def _hook(self, name: str):
        def fn(module, inputs, output):
            # output might be tuple; handle common cases
            y = output
            if isinstance(y, (tuple, list)):
                y = y[0]
            if not torch.is_tensor(y):
                return
            with torch.no_grad():
                t = y.detach()
                # scalar summaries (cheap)
                flat = t.reshape(-1).float()
                d = {
                    "out_mean": float(flat.mean().item()),
                    "out_std": float(flat.std(unbiased=False).item()),
                    "out_abs_mean": float(flat.abs().mean().item()),
                    "out_l2_mean": float((flat.pow(2).mean().sqrt()).item()),
                }
                self.samples[name].append(d)
        return fn

    def attach(self):
        self.handles = []
        for n in self.layer_names:
            m = self.name_to_module[n]
            self.handles.append(m.register_forward_hook(self._hook(n)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @staticmethod
    def _aggregate_one(metric_vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(metric_vals, dtype=np.float64)
        return {
            "mc_mean": float(arr.mean()) if arr.size else float("nan"),
            "mc_std": float(arr.std(ddof=0)) if arr.size else float("nan"),
            "mc_p50": float(np.percentile(arr, 50)) if arr.size else float("nan"),
            "mc_p90": float(np.percentile(arr, 90)) if arr.size else float("nan"),
        }

    def aggregate(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Returns:
          layer -> metric_name -> stats_dict
        """
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for layer, rows in self.samples.items():
            if not rows:
                out[layer] = {}
                continue
            metrics = rows[0].keys()
            out[layer] = {}
            for m in metrics:
                out[layer][m] = self._aggregate_one([r[m] for r in rows])
        return out


# ============================================================================
# 6. BayzFlow engine (UPDATED with two functions)
# ============================================================================

class BayzFlow:
    def __init__(self, model: nn.Module):
        self.base_model = model
        self.bayes_model: Optional[nn.Module] = None
        self.selected_module_names: List[str] = []
        self.prior_scales: Dict[str, float] = {}

        self.guide: Optional[AutoLowRankMultivariateNormal] = None
        self.optimizer = None
        self.svi: Optional[SVI] = None
        self._pyro_model = None

    def list_candidates(self, exclude_names: Optional[Iterable[str]] = None) -> List[ModuleInfo]:
        cands = find_candidate_modules(self.base_model, exclude_names=exclude_names)
        print_candidates(cands)
        return cands

    def select_modules(self, candidates: Sequence[ModuleInfo], patterns: Sequence[str]) -> List[str]:
        selected = select_by_pattern(candidates, patterns)
        self.selected_module_names = [m.name for m in selected]
        print("[BayzFlow] Selected modules:")
        for n in self.selected_module_names:
            print("  -", n)
        return self.selected_module_names

    def calibrate_priors(
        self,
        calib_loader: torch.utils.data.DataLoader,
        calib_forward_fn: Callable[[nn.Module, Any], torch.Tensor],
        num_passes: int = 3,
        percentile: float = 0.9,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        if not self.selected_module_names:
            raise RuntimeError("No modules selected.")
        self.prior_scales = calibrate_prior_scales(
            self.base_model,
            self.selected_module_names,
            calib_loader,
            calib_forward_fn,
            num_passes=num_passes,
            percentile=percentile,
            device=device,
        )
        print("[BayzFlow] Calibrated prior scales:")
        for k, v in self.prior_scales.items():
            print(f"  {k}: {v:.4f}")
        return self.prior_scales

    def build_bayesian_model(self, default_prior_scale: float = 0.1) -> nn.Module:
        if not self.selected_module_names:
            raise RuntimeError("No modules selected.")

        self.bayes_model = attach_priors(
            model=self.base_model,
            selected_names=self.selected_module_names,
            prior_scale=default_prior_scale,
            per_layer_scale=self.prior_scales if self.prior_scales else None,
        )
        return self.bayes_model

    def setup_svi(
        self,
        loss_from_logits_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        predict_fn: Callable[[nn.Module, Any], torch.Tensor],
        lr: float = 1e-3,
    ):
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built.")

        pyro.clear_param_store()
        model = self.bayes_model

        def pyro_model(batch):
            pyro.module("bayes_model", model, update_module_params=False)

            logits = predict_fn(model, batch)             # ✅ ONE forward pass
            pyro.deterministic("logits", logits)

            nll = loss_from_logits_fn(logits, batch)      # ✅ NO second forward
            pyro.factor("nll", -nll)

            return logits

        self._pyro_model = pyro_model
        self.guide = AutoLowRankMultivariateNormal(self._pyro_model, rank=20)
        self.optimizer = pyro.optim.Adam({"lr": lr})
        self.svi = SVI(self._pyro_model, self.guide, self.optimizer, loss=Trace_ELBO())

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 5,
        device: Optional[torch.device] = None,
    ):
        if self.svi is None:
            raise RuntimeError("SVI not set up. Call setup_svi(...) first.")
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built.")

        if device is not None:
            self.bayes_model.to(device)

        for epoch in range(num_epochs):
            total = 0.0
            n = 0
            for batch in train_loader:
                batch = _to_device_batch(batch, device)
                total += float(self.svi.step(batch))
                n += 1
            print(f"[BayzFlow] Epoch {epoch+1}/{num_epochs}  ELBO: {total / max(n,1):.4f}")

    def predict(
        self,
        batch: Any,
        predict_fn: Callable[[nn.Module, Any], torch.Tensor],
        num_samples: int = 16,
        device: Optional[torch.device] = None,
        return_samples: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self._pyro_model is None or self.guide is None or self.bayes_model is None:
            raise RuntimeError("Model/guide not initialised.")

        model = self.bayes_model
        if device is not None:
            model.to(device)
        batch = _to_device_batch(batch, device)

        model.eval()
        self.guide.eval()

        predictive = Predictive(
            self._pyro_model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=["logits"],
        )
        with torch.no_grad():
            samp = predictive(batch)

        logits_s = samp["logits"]  # [S, ...]
        out = {
            "mean": logits_s.mean(0),
            "std": logits_s.std(0, unbiased=False),
        }
        if return_samples:
            out["samples"] = logits_s
        return out

    # ------------------------------------------------------------------------
    # NEW: (1) train once + optionally save posterior
    # ------------------------------------------------------------------------
    def train_posterior(
        self,
        train_loader: torch.utils.data.DataLoader,
        loss_from_logits_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        predict_fn: Callable[[nn.Module, Any], torch.Tensor],
        num_epochs: int = 5,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Train SVI ONCE. Optionally save:
          - pyro param store (guide params)
          - model state_dict (for safety / reproducibility)
          - metadata (selected modules/prior scales)
        """
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built. Call build_bayesian_model(...) first.")

        self.setup_svi(loss_from_logits_fn=loss_from_logits_fn, predict_fn=predict_fn, lr=lr)
        self.fit(train_loader=train_loader, num_epochs=num_epochs, device=device)

        if save_dir is not None:
            _ensure_dir(save_dir)
            pyro_path = os.path.join(save_dir, "pyro_params.pt")
            model_path = os.path.join(save_dir, "model_state.pt")
            meta_path = os.path.join(save_dir, "meta.json")

            pyro.get_param_store().save(pyro_path)
            torch.save(self.bayes_model.state_dict(), model_path)

            meta = {
                "selected_module_names": list(self.selected_module_names),
                "prior_scales": dict(self.prior_scales),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"[BayzFlow] Saved posterior to: {save_dir}")

    # ------------------------------------------------------------------------
    # NEW: load posterior (for analysis-only runs)
    # ------------------------------------------------------------------------
    def load_posterior(
        self,
        load_dir: str,
        loss_from_logits_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        predict_fn: Callable[[nn.Module, Any], torch.Tensor],
        device: Optional[torch.device] = None,
        strict_model_state: bool = True,
    ) -> None:
        """
        Load:
          - pyro param store (guide params)
          - model state_dict
        Rebuilds guide against your current pyro_model definition (so names match).
        """
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built. Call build_bayesian_model(...) first.")

        pyro_path = os.path.join(load_dir, "pyro_params.pt")
        model_path = os.path.join(load_dir, "model_state.pt")

        if not os.path.exists(pyro_path):
            raise FileNotFoundError(pyro_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        if device is not None:
            self.bayes_model.to(device)

        # Recreate model/guide structure (param names must match)
        self.setup_svi(loss_from_logits_fn=loss_from_logits_fn, predict_fn=predict_fn, lr=1e-3)

        # Load deterministic weights (optional but recommended)
        state = torch.load(model_path, map_location="cpu")
        self.bayes_model.load_state_dict(state, strict=strict_model_state)

        # Load variational params
        pyro.clear_param_store()
        pyro.get_param_store().load(pyro_path)

        print(f"[BayzFlow] Loaded posterior from: {load_dir}")

    # ------------------------------------------------------------------------
    # NEW: (2) measure layers (trajectory) under fixed posterior
    # ------------------------------------------------------------------------

    

    def measure_layers_per_example(
        bf, batch, predict_fn, layer_names, num_samples=16, device=None
    ):
        model = bf.bayes_model
        if device is not None:
            model.to(device)
        if isinstance(batch, dict) and device is not None:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        model.eval()
        bf.guide.eval()

        rec = _LayerPerExampleMCRecorder(layer_names, model).attach()
        logits_samples = []

        try:
            # Build Predictive ONCE (cheaper + cleaner)
            pred = Predictive(bf._pyro_model, guide=bf.guide, num_samples=num_samples, return_sites=["logits"])
            with torch.no_grad():
                samp = pred(batch)                     # logits: [S,B,C]
            logits_s = samp["logits"].detach().cpu()   # [S,B,C]

            # Recorder: you currently call rec.step() per sample.
            # If your recorder depends on hooks firing during forward passes,
            # you need to run num_samples forward passes.
            # Since Predictive already did that internally, your hooks may NOT have fired.
            # So keep your existing loop for layer recording, but reuse logits_s from Predictive
            # OR modify recorder to capture inside pyro_model. For now: keep loop.
        finally:
            rec.remove()

        # --- If you keep the existing loop for rec.step(), use your original code path ---
        # (I’m keeping your current structure below to preserve recorder correctness.)

        # Re-run with per-sample Predictive to ensure layer hooks fire
        rec = _LayerPerExampleMCRecorder(layer_names, model).attach()
        logits_samples = []
        try:
            for _ in range(num_samples):
                pred1 = Predictive(bf._pyro_model, guide=bf.guide, num_samples=1, return_sites=["logits"])
                with torch.no_grad():
                    samp1 = pred1(batch)
                logits_samples.append(samp1["logits"][0].detach().cpu())  # [B,C]
                rec.step()
        finally:
            rec.remove()

        logits_s = torch.stack(logits_samples, dim=0)  # [S,B,C]
        logits_mean = logits_s.mean(0)                 # [B,C]
        logits_std  = logits_s.std(0, unbiased=False)  # [B,C]

        # -----------------------------
        # Prob-space uncertainty
        # -----------------------------
        probs_s = torch.softmax(logits_s, dim=-1)      # [S,B,C]
        probs_mean = probs_s.mean(0)                   # [B,C]

        # Predictive entropy H[E[p]]
        pred_entropy = -(probs_mean * (probs_mean + 1e-12).log()).sum(dim=1)  # [B]

        # Expected entropy E[H[p]]
        ent_s = -(probs_s * (probs_s + 1e-12).log()).sum(dim=2)              # [S,B]
        exp_entropy = ent_s.mean(0)                                          # [B]

        # Mutual information (epistemic proxy)
        mutual_info = pred_entropy - exp_entropy                              # [B]

        # Vote rate / variation ratio
        y_s = probs_s.argmax(dim=2)               # [S,B]
        # vote_rate[b] = max_c mean(y_s[:,b
        rec.remove()

        logits_s = torch.stack(logits_samples, dim=0)  # [S,B,C]
        logits_mean = logits_s.mean(0)                 # [B,C]
        logits_std  = logits_s.std(0, unbiased=False)  # [B,C]

        # -----------------------------] == c)
        B = y_s.shape[1]
        vote_rate = torch.empty(B)
        for b in range(B):
            vals, counts = y_s[:, b].unique(return_counts=True)
            vote_rate[b] = counts.max().float() / float(num_samples)
        variation_ratio = 1.0 - vote_rate         # [B]

        # Margin on mean probs
        top2 = torch.topk(probs_mean, k=2, dim=1).values   # [B,2]
        margin = top2[:, 0] - top2[:, 1]                   # [B]

        # -----------------------------
        # Trajectory derived features
        # -----------------------------
        layer_stats = rec.results()  # layer -> {"mc_mean": [B], "mc_std": [B]}
        eps = 1e-12

        # Example assumes layer_names are like ["encoder.0","encoder.2","head"]
        # but works for arbitrary list.
        layer_std_stack = torch.stack([torch.as_tensor(layer_stats[ln]["mc_std"]) for ln in layer_names], dim=1)  # [B,L]

        log_std = torch.log(layer_std_stack + eps)  # [B,L]

        traj_features = {
            "layer_std_stack": layer_std_stack,     # [B,L]
            "log_layer_std_stack": log_std,         # [B,L]
        }
        if len(layer_names) >= 3:
            # gains and curvature for first three layers
            g01 = log_std[:, 1] - log_std[:, 0]
            g12 = log_std[:, 2] - log_std[:, 1]
            curvature = g12 - g01
            ratio01 = layer_std_stack[:, 1] / (layer_std_stack[:, 0] + eps)
            ratio12 = layer_std_stack[:, 2] / (layer_std_stack[:, 1] + eps)

            traj_features.update({
                "gain_0_1": g01,
                "gain_1_2": g12,
                "curvature": curvature,
                "ratio_0_1": ratio01,
                "ratio_1_2": ratio12,
            })

        out = {
            "logits_mean": logits_mean,
            "logits_std": logits_std,
            "probs_mean": probs_mean,
            "pred_entropy": pred_entropy,
            "exp_entropy": exp_entropy,
            "mutual_info": mutual_info,
            "vote_rate": vote_rate,
            "variation_ratio": variation_ratio,
            "margin": margin,
            "layer_per_example": layer_stats,   # original
            "traj_features": traj_features,
        }
        return out

    
    def measure_layers(
        self,
        batch: Any,
        predict_fn: Callable[[nn.Module, Any], torch.Tensor],
        layer_names: Sequence[str],
        num_samples: int = 32,
        device: Optional[torch.device] = None,
        include_logits: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs MC Predictive under the current posterior and records per-layer scalar summaries.

        Returns dict:
          {
            "layer_stats": { layer -> metric -> {mc_mean, mc_std, mc_p50, mc_p90} },
            "logits_mean": tensor (optional),
            "logits_std":  tensor (optional),
          }
        """
        if self._pyro_model is None or self.guide is None or self.bayes_model is None:
            raise RuntimeError("Model/guide not initialised. Train or load posterior first.")

        model = self.bayes_model
        if device is not None:
            model.to(device)
        batch = _to_device_batch(batch, device)

        model.eval()
        self.guide.eval()

        rec = _LayerScalarRecorder(layer_names=layer_names, model=model).attach()
        try:
            predictive = Predictive(
                self._pyro_model,
                guide=self.guide,
                num_samples=num_samples,
                return_sites=["logits"] if include_logits else [],
            )
            with torch.no_grad():
                samp = predictive(batch)
        finally:
            rec.remove()

        out: Dict[str, Any] = {"layer_stats": rec.aggregate()}

        if include_logits:
            logits_s = samp["logits"]  # [S, ...]
            out["logits_mean"] = logits_s.mean(0)
            out["logits_std"] = logits_s.std(0, unbiased=False)

        return out

    @classmethod
    def auto(
        cls,
        model: nn.Module,
        calib_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        calib_forward_fn: Callable[[nn.Module, Any], torch.Tensor],
        loss_from_logits_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        predict_fn: Callable[[nn.Module, Any], torch.Tensor],
        bayes_patterns: Sequence[str],
        num_calib_passes: int = 3,
        calib_percentile: float = 0.9,
        default_prior_scale: float = 0.1,
        train_epochs: int = 5,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> "BayzFlow":
        engine = cls(model)

        if verbose:
            print("=== BayzFlow: candidate layers ===")
        candidates = engine.list_candidates()

        engine.select_modules(candidates, patterns=bayes_patterns)
        engine.calibrate_priors(
            calib_loader=calib_loader,
            calib_forward_fn=calib_forward_fn,
            num_passes=num_calib_passes,
            percentile=calib_percentile,
            device=device,
        )
        engine.build_bayesian_model(default_prior_scale=default_prior_scale)

        if verbose:
            print("\n[BayzFlow] Starting SVI training...")
        engine.setup_svi(loss_from_logits_fn=loss_from_logits_fn, predict_fn=predict_fn, lr=lr)
        engine.fit(train_loader=train_loader, num_epochs=train_epochs, device=device)
        return engine


# ============================================================================
# 7. Tiny demo (updated)
# ============================================================================

if __name__ == "__main__":
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x)
        
    class StructuredDS(torch.utils.data.Dataset):
        def __init__(self, n=5000, d=10, k=3, noise=0.7, seed=0):
            g = torch.Generator().manual_seed(seed)
            self.W = torch.randn(d, k, generator=g)
            self.X = torch.randn(n, d, generator=g)
            logits = self.X @ self.W + noise * torch.randn(n, k, generator=g)
            self.y = logits.argmax(dim=1)

        def __len__(self): return len(self.y)
        def __getitem__(self, i): return {"x": self.X[i], "y": self.y[i]}

    def calib_forward_fn(m, batch):
        return m(batch["x"])

    def predict_fn(m, batch):
        return m(batch["x"])

    def loss_from_logits_fn(logits, batch):
        return F.cross_entropy(logits, batch["y"])

    def train_deterministic(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None,
        num_epochs: int = 15,
        lr: float = 1e-3,
        device=None,
    ):
        if device is not None:
            model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        def _eval(loader):
            model.eval()
            correct = 0
            total = 0
            tot_loss = 0.0
            with torch.no_grad():
                for batch in loader:
                    if device is not None:
                        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    logits = model(batch["x"])
                    loss = F.cross_entropy(logits, batch["y"])
                    tot_loss += float(loss.item()) * int(batch["y"].shape[0])
                    pred = logits.argmax(dim=1)
                    correct += int((pred == batch["y"]).sum().item())
                    total += int(batch["y"].shape[0])
            return tot_loss / max(total, 1), correct / max(total, 1)

        for epoch in range(num_epochs):
            model.train()
            tot = 0.0
            n = 0
            for batch in train_loader:
                if device is not None:
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                logits = model(batch["x"])
                loss = F.cross_entropy(logits, batch["y"])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tot += float(loss.item()) * int(batch["y"].shape[0])
                n += int(batch["y"].shape[0])

            train_loss = tot / max(n, 1)
            if val_loader is not None:
                val_loss, val_acc = _eval(val_loader)
                print(f"[Deterministic] Epoch {epoch+1}/{num_epochs}  trainCE={train_loss:.4f}  valCE={val_loss:.4f}  valAcc={val_acc:.3f}")
            else:
                print(f"[Deterministic] Epoch {epoch+1}/{num_epochs}  trainCE={train_loss:.4f}")

        save_path = "deterministic_model.pt"
        torch.save(model.state_dict(), save_path)
        print(f"[Deterministic] Saved deterministic model to: {save_path}")

        model.eval()
        return model
    
    ds = StructuredDS(n=5000, d=10, k=3, noise=0.3, seed=0)

    # split
    n = len(ds)
    idx = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds   = torch.utils.data.Subset(ds, val_idx.tolist())

    calib_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=256, shuffle=False)
    
    model = train_deterministic(
        model=TinyNet(),
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        lr=1e-3,
        device=None,
    )
    net = model   

    bf = BayzFlow(net)
    cands = bf.list_candidates()
    bf.select_modules(cands, patterns=["encoder.0", "encoder.2", "head"])     # bayesianise head only
    bf.calibrate_priors(calib_loader, calib_forward_fn, num_passes=2, percentile=0.9)
    bf.build_bayesian_model(default_prior_scale=0.1)

    # 1) TRAIN ONCE + SAVE
    bf.train_posterior(
        train_loader=train_loader,
        loss_from_logits_fn=loss_from_logits_fn,
        predict_fn=predict_fn,
        num_epochs=8,
        lr=1e-3,
        device=None,
        save_dir="bf_ckpt",
    )

    ds = StructuredDS(n=5000, d=10, k=3, noise=0.3, seed=1887)

    # split
    n = len(ds)
    idx = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds   = torch.utils.data.Subset(ds, val_idx.tolist())

    calib_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=256, shuffle=False)

    # 2) MEASURE LAYERS (no retrain)
    batch = next(iter(train_loader))
    stats = bf.measure_layers_per_example(
    batch=batch,
    predict_fn=predict_fn,
    layer_names=["encoder.0", "encoder.2", "head"],
    num_samples=64,
    device=None,
    )

    print("\n[TEST] Per-example layer stats keys:", stats.keys())

    import numpy as np

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        LogisticRegression = None
        roc_auc_score = None


    # ----------------------------------------------------------------------------
    # 0) labels
    # ----------------------------------------------------------------------------
    if isinstance(batch, dict) and "y" in batch:
        y_true = batch["y"].detach().cpu().numpy()
    else:
        raise RuntimeError("Batch must be dict containing 'y'.")

    # ----------------------------------------------------------------------------
    # 1) predictions + correctness
    # ----------------------------------------------------------------------------
    logits_mean = stats["logits_mean"].detach().cpu().numpy()  # [B, C]
    logits_std  = stats["logits_std"].detach().cpu().numpy()   # [B, C]

    y_pred  = logits_mean.argmax(axis=1)
    correct = (y_pred == y_true).astype(np.int64)
    error   = 1 - correct

    # baseline uncertainty scalar per example
    logits_std_mag = np.sqrt((logits_std ** 2).mean(axis=1))  # [B]

    # ----------------------------------------------------------------------------
    # 2) trajectory features (PER EXAMPLE now)
    # ----------------------------------------------------------------------------
    layer_std_enc0 = stats["layer_per_example"]["encoder.0"]["mc_std"]  # [B]
    layer_std_enc2 = stats["layer_per_example"]["encoder.2"]["mc_std"]  # [B]
    layer_std_head = stats["layer_per_example"]["head"]["mc_std"]       # [B]

    # stack trajectory
    traj_vec = np.stack([
        layer_std_enc0,
        layer_std_enc2,
        layer_std_head,
    ], axis=1)  # [B, 3]

    eps = 1e-12

    σ0 = layer_std_enc0
    σ1 = layer_std_enc2
    σ2 = layer_std_head

    l0 = np.log(σ0 + eps)
    l1 = np.log(σ1 + eps)
    l2 = np.log(σ2 + eps)

    g01 = l1 - l0
    g12 = l2 - l1
    curvature = g12 - g01

    mid_dev = l1 - 0.5 * (l0 + l2)

    E_total = σ0**2 + σ1**2 + σ2**2
    w0 = σ0**2 / (E_total + eps)
    w1 = σ1**2 / (E_total + eps)
    w2 = σ2**2 / (E_total + eps)

    X_traj = np.stack([
        g01,
        g12,
        curvature,
        mid_dev,
        w0,
        w1,
        w2,
    ], axis=1)

    import pandas as pd

    feature_names = [
    "g01",
    "g12",
    "curvature",
    "mid_dev",
    "w0",
    "w1",
    "w2",
    ]

    df_traj = pd.DataFrame(X_traj, columns=feature_names)
    print(df_traj.head())

    import matplotlib.pyplot as plt

    df_traj.hist(bins=50)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------------
    # 3) correlation diagnostics
    # ----------------------------------------------------------------------------
    def _corr(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n[TEST] Correlations (Pearson):")
    print("  corr(logits_std_mag, error):", _corr(logits_std_mag, error))
    print("  corr(enc0_std, error):", _corr(layer_std_enc0, error))
    print("  corr(enc2_std, error):", _corr(layer_std_enc2, error))
    print("  corr(head_std, error):", _corr(layer_std_head, error))

    print("\n  corr(logits_std_mag, enc0_std):", _corr(logits_std_mag, layer_std_enc0))
    print("  corr(logits_std_mag, enc2_std):", _corr(logits_std_mag, layer_std_enc2))
    print("  corr(logits_std_mag, head_std):", _corr(logits_std_mag, layer_std_head))


    # ----------------------------------------------------------------------------
    # 4) AUC comparison: baseline vs +trajectory
    # ----------------------------------------------------------------------------
    if LogisticRegression is None or roc_auc_score is None:
        print("\n[TEST] sklearn not available; skipping AUC tests.")
    else:
        X_base = logits_std_mag.reshape(-1, 1)
        X_aug  = np.concatenate([X_base, traj_vec], axis=1)

        clf_base = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
        )
        clf_aug = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
        )

        clf_base.fit(X_base, error)
        clf_aug.fit(X_aug, error)

        p_base = clf_base.predict_proba(X_base)[:, 1]
        p_aug  = clf_aug.predict_proba(X_aug)[:, 1]

        auc_base = roc_auc_score(error, p_base) if len(np.unique(error)) > 1 else float("nan")
        auc_aug  = roc_auc_score(error, p_aug)  if len(np.unique(error)) > 1 else float("nan")

        print("\n[TEST] Error prediction AUC (higher is better):")
        print("  baseline (logits_std_mag):", auc_base)
        print("  + trajectory:", auc_aug)

        if not np.isnan(auc_base) and not np.isnan(auc_aug):
            print("  ΔAUC:", float(auc_aug - auc_base))


    # ----------------------------------------------------------------------------
    # 5) sanity print
    # ----------------------------------------------------------------------------
    print("\n[TEST] Sanity checks:")
    print("  correct rate:", float(correct.mean()))
    print("  logits_std_mag mean:", float(logits_std_mag.mean()))
    print("  enc0_std mean:", float(layer_std_enc0.mean()))
    print("  enc2_std mean:", float(layer_std_enc2.mean()))
    print("  head_std mean:", float(layer_std_head.mean()))

    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def eval_trajectory_auc(
        bf,
        loader,
        predict_fn,
        layer_names=("encoder.0", "encoder.2", "head"),
        num_samples=16,
        max_batches=50,
        device=None,
    ):
        Xb_all, Xa_all, yerr_all = [], [], []

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            stats = bf.measure_layers_per_example(
                batch=batch,
                predict_fn=predict_fn,
                layer_names=list(layer_names),
                num_samples=num_samples,
                device=device,
            )

            # labels
            if not (isinstance(batch, dict) and "y" in batch):
                raise RuntimeError("Batch must be dict with 'y'.")

            y_true = batch["y"].detach().cpu().numpy()

            # predictions
            logits_mean = stats["logits_mean"].detach().cpu().numpy()
            logits_std  = stats["logits_std"].detach().cpu().numpy()

            y_pred = logits_mean.argmax(axis=1)
            error  = (y_pred != y_true).astype(np.int64)

            logits_std_mag = np.sqrt((logits_std ** 2).mean(axis=1))  # [B]

            # per-example layer std
            traj = []
            for ln in layer_names:
                traj.append(stats["layer_per_example"][ln]["mc_std"])
            traj_vec = np.stack(traj, axis=1)  # [B, L]

            X_base = logits_std_mag.reshape(-1, 1)
            X_aug  = np.concatenate([X_base, traj_vec], axis=1)

            Xb_all.append(X_base)
            Xa_all.append(X_aug)
            yerr_all.append(error)

        X_base = np.concatenate(Xb_all, axis=0)
        X_aug  = np.concatenate(Xa_all, axis=0)
        yerr   = np.concatenate(yerr_all, axis=0)

        # Fit + evaluate (note: in-sample; fine for quick screening)
        clf_base = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)
        clf_aug  = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)

        clf_base.fit(X_base, yerr)
        clf_aug.fit(X_aug, yerr)

        p_base = clf_base.predict_proba(X_base)[:, 1]
        p_aug  = clf_aug.predict_proba(X_aug)[:, 1]

        auc_base = roc_auc_score(yerr, p_base) if len(np.unique(yerr)) > 1 else float("nan")
        auc_aug  = roc_auc_score(yerr, p_aug)  if len(np.unique(yerr)) > 1 else float("nan")

        return {
            "N": int(len(yerr)),
            "auc_base": float(auc_base),
            "auc_aug": float(auc_aug),
            "delta_auc": float(auc_aug - auc_base) if (not np.isnan(auc_base) and not np.isnan(auc_aug)) else float("nan"),
            "err_rate": float(yerr.mean()),
        }

    # Example:
    res = eval_trajectory_auc(
        bf=bf,
        loader=train_loader,
        predict_fn=predict_fn,
        layer_names=("encoder.0","encoder.2","head"),
        num_samples=64,
        max_batches=100,   # bump this up
        device=None,
    )
    print("\n[AGG TEST]", res)
    