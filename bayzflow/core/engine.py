#!/usr/bin/env python3
"""
bayzflow.py

BayzFlow engine — turn any PyTorch model into a calibrated Bayesian (Pyro) model
with transparent module selection and automatic prior calibration.

Key features:
    - Torch-model agnostic (any nn.Module)
    - Candidate layer listing (Linear / Conv / Norm)
    - Pattern-based selection of Bayesian layers
    - Automatic wrapping of selected layers with PyroModules (BayesLinear / BayesConv*)
    - Automatic prior calibration using dataset activations
    - SVI training with a proper Pyro model
    - Monte Carlo predictive inference with uncertainty

CRITICAL NOTE (fix for "Multiple sample sites named 'head.weight'"):
    Your old pyro_model ran TWO forward passes per trace:
        logits = predict_fn(model, batch)
        nll    = loss_fn(model, batch)   # <-- loss_fn usually calls model(...) again
    That causes PyroSample sites to be sampled twice in one trace, which is illegal.
    This script enforces ONE forward pass per trace by using:
        loss_from_logits_fn(logits, batch)

Requires:
    torch
    pyro-ppl
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal


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
    def __init__(self, base_linear: nn.Linear, prior_scale: float = 0.1):
        super().__init__()
        if base_linear.weight is None:
            raise ValueError("Base Linear has no weight.")

        w_mean = base_linear.weight.detach()
        b_mean = base_linear.bias.detach() if base_linear.bias is not None else None

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # Use the existing weights as prior means
        self.weight = PyroSample(dist.Normal(w_mean, prior_scale).to_event(2))
        self.bias = PyroSample(dist.Normal(b_mean, prior_scale).to_event(1)) if b_mean is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


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


def wrap_module_with_prior(module: nn.Module, prior_scale: float = 0.1) -> nn.Module:
    # Prevent double wrapping
    if isinstance(module, _BAYES_TYPES):
        return module

    if isinstance(module, nn.Linear):
        return BayesLinear(module, prior_scale=prior_scale)
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
        wrapped = wrap_module_with_prior(child, prior_scale=scale)

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
# 5. BayzFlow engine
# ============================================================================

class BayzFlow:
    def __init__(self, model: nn.Module):
        self.base_model = model
        self.bayes_model: Optional[nn.Module] = None
        self.selected_module_names: List[str] = []
        self.prior_scales: Dict[str, float] = {}

        self.guide: Optional[AutoNormal] = None
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

        # Important: do NOT force-convert the whole tree to PyroModule.
        # Your wrapped layers already provide PyroSample sites and will be named by their module path.
        self.bayes_model = attach_priors(
            model=self.base_model,
            selected_names=self.selected_module_names,
            prior_scale=default_prior_scale,
            per_layer_scale=self.prior_scales if self.prior_scales else None,
        )
        return self.bayes_model

    # ----------------------------
    # SVI (CRITICAL FIX HERE)
    # ----------------------------
    def setup_svi(
        self,
        loss_from_logits_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        predict_fn: Callable[[nn.Module, Any], torch.Tensor],
        lr: float = 1e-3,
    ):
        """
        Set up SVI so that the model is executed EXACTLY ONCE per trace.

        Args:
            loss_from_logits_fn: (logits, batch) -> scalar NLL (or any differentiable loss)
            predict_fn:          (model, batch) -> logits
        """
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built.")

        pyro.clear_param_store()
        model = self.bayes_model

        def pyro_model(batch):
            # Register model params but do NOT update them (PyroSample handles randomness)
            pyro.module("bayes_model", model, update_module_params=False)

            # ✅ ONE forward pass
            logits = predict_fn(model, batch)
            pyro.deterministic("logits", logits)

            # ✅ loss computed from logits (NO second forward)
            nll = loss_from_logits_fn(logits, batch)
            pyro.factor("nll", -nll)

            return logits

        self._pyro_model = pyro_model
        self.guide = AutoNormal(self._pyro_model)
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
                if device is not None and isinstance(batch, dict):
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
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
        if device is not None and isinstance(batch, dict):
            model.to(device)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        model.eval()
        self.guide.eval()

        predictive = Predictive(self._pyro_model, guide=self.guide, num_samples=num_samples, return_sites=["logits"])
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
# 6. Tiny demo
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

    class TinyDS(torch.utils.data.Dataset):
        def __len__(self):
            return 128

        def __getitem__(self, idx):
            x = torch.randn(10)
            y = torch.randint(0, 3, ()).long()
            return {"x": x, "y": y}

    def calib_forward_fn(m, batch):
        return m(batch["x"])

    def predict_fn(m, batch):
        return m(batch["x"])

    # ✅ loss computed from logits, NOT by calling m(...) again
    def loss_from_logits_fn(logits, batch):
        return F.cross_entropy(logits, batch["y"])

    ds = TinyDS()
    calib_loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

    net = TinyNet()

    bf = BayzFlow.auto(
        model=net,
        calib_loader=calib_loader,
        train_loader=train_loader,
        calib_forward_fn=calib_forward_fn,
        loss_from_logits_fn=loss_from_logits_fn,
        predict_fn=predict_fn,
        bayes_patterns=["head"],
        num_calib_passes=2,
        calib_percentile=0.9,
        default_prior_scale=0.1,
        train_epochs=3,
        lr=1e-3,
        device=None,
        verbose=True,
    )

    batch = next(iter(train_loader))
    out = bf.predict(batch, predict_fn=predict_fn, num_samples=16)
    print("MC mean:", out["mean"].shape, "MC std:", out["std"].shape)
