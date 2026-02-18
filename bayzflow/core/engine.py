"""
bayzflow.py

BayzFlow engine — turn any PyTorch model into a calibrated Bayesian (Pyro) model
with transparent module selection and automatic prior calibration.

Key fixes in this version:
- ✅ Prevents double-wrapping Bayesian layers (idempotent attach_priors)
- ✅ BayzFlow owns a deepcopy of the input model to avoid mutating user models
- ✅ Candidate discovery skips already-wrapped Bayesian layers
- ✅ setup_svi stores predict_fn and fit() no longer tries to call setup_svi without it
- ✅ predict() uses Pyro Predictive for correct posterior predictive sampling
- ✅ Removes brittle _pyro_name hack
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
from copy import deepcopy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal  # AutoLowRankMultivariateNormal optional


# ============================================================================
# 1. Module information / discovery
# ============================================================================

@dataclass
class ModuleInfo:
    """Simple container for a discovered module in the model tree."""
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

# Extended types for MONAI compatibility (optional)
try:
    import monai.networks.layers as monai_layers  # noqa: F401
    import monai.networks.blocks as monai_blocks  # noqa: F401

    _MONAI_INCLUDE_TYPES: Tuple[Type[nn.Module], ...] = (
        # Add MONAI types here if you want them auto-detected
    )
    _DEFAULT_INCLUDE_TYPES = _DEFAULT_INCLUDE_TYPES + _MONAI_INCLUDE_TYPES
except ImportError:
    pass


# ============================================================================
# 2. Module tree / map (unchanged)
# ============================================================================

def build_module_tree(model: nn.Module) -> Dict:
    tree = {"type": model.__class__.__name__, "children": {}}
    name_to_module = dict(model.named_modules())

    for full_name, module in name_to_module.items():
        if full_name == "":
            continue

        parts = full_name.split(".")
        node = tree
        for i, part in enumerate(parts):
            node.setdefault("children", {})
            node["children"].setdefault(part, {"type": None, "children": {}})
            node = node["children"][part]
            if i == len(parts) - 1:
                node["type"] = module.__class__.__name__

    return tree


def _print_tree_node(name: str, node: Dict, indent: str, is_last: bool):
    connector = "└── " if is_last else "├── "
    print(f"{indent}{connector}{name} ({node.get('type', 'Unknown')})")

    children = node.get("children", {})
    if not children:
        return

    next_indent = indent + ("    " if is_last else "│   ")
    child_items = list(children.items())
    for i, (child_name, child_node) in enumerate(child_items):
        _print_tree_node(child_name, child_node, next_indent, i == len(child_items) - 1)


def print_module_tree(model: nn.Module) -> None:
    tree = build_module_tree(model)
    print("root", f"({tree.get('type', 'Model')})")
    children = tree.get("children", {})
    items = list(children.items())
    for i, (name, node) in enumerate(items):
        _print_tree_node(name, node, indent="", is_last=(i == len(items) - 1))


# ============================================================================
# 3. Selection helpers
# ============================================================================

def print_candidates(candidates: Sequence[ModuleInfo]) -> None:
    for idx, info in enumerate(candidates):
        print(f"[{idx:03}] {info.name:60} {info.module.__class__.__name__}")


def select_by_indices(candidates: Sequence[ModuleInfo], indices: Sequence[int]) -> List[ModuleInfo]:
    selected: List[ModuleInfo] = []
    for i in indices:
        if i < 0 or i >= len(candidates):
            raise IndexError(f"Index {i} out of range for {len(candidates)} candidates.")
        selected.append(candidates[i])
    return selected


def select_by_pattern(candidates: Sequence[ModuleInfo], patterns: Sequence[str]) -> List[ModuleInfo]:
    patterns = list(patterns)
    return [info for info in candidates if any(p in info.name for p in patterns)]


def select_with_predicate(
    candidates: Sequence[ModuleInfo],
    predicate: Callable[[str, nn.Module], bool],
) -> List[ModuleInfo]:
    return [info for info in candidates if predicate(info.name, info.module)]


# ============================================================================
# 4. Bayesian wrapper layers
# ============================================================================

class BayesLinear(PyroModule):
    def __init__(self, base_linear: nn.Linear, prior_scale: float = 0.1):
        super().__init__()
        if base_linear.weight is None:
            raise ValueError("Base Linear has no weight tensor.")

        w_mean = base_linear.weight.detach()
        b_mean = base_linear.bias.detach() if base_linear.bias is not None else None

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.weight = PyroSample(dist.Normal(w_mean, prior_scale).to_event(2))
        if b_mean is not None:
            self.bias = PyroSample(dist.Normal(b_mean, prior_scale).to_event(1))
        else:
            self.bias = None

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
        if b_mean is not None:
            self.bias = PyroSample(dist.Normal(b_mean, prior_scale).to_event(1))
        else:
            self.bias = None

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
        if b_mean is not None:
            self.bias = PyroSample(dist.Normal(b_mean, prior_scale).to_event(1))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, dilation=self.dilation
        )


class BayesInstanceNorm3d(PyroModule):
    def __init__(self, base_norm: nn.InstanceNorm3d, prior_scale: float = 0.1):
        super().__init__()
        self.num_features = base_norm.num_features
        self.eps = base_norm.eps
        self.momentum = base_norm.momentum
        self.affine = base_norm.affine
        self.track_running_stats = base_norm.track_running_stats

        if self.affine:
            w_mean = base_norm.weight.detach()
            b_mean = base_norm.bias.detach()
            self.weight = PyroSample(dist.Normal(w_mean, prior_scale).to_event(1))
            self.bias = PyroSample(dist.Normal(b_mean, prior_scale).to_event(1))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.register_buffer("running_mean", base_norm.running_mean.clone())
            self.register_buffer("running_var", base_norm.running_var.clone())
            self.register_buffer("num_batches_tracked", base_norm.num_batches_tracked.clone())
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps
        )


# ✅ Used for guards (prevents double wrapping + candidate skip)
_BAYES_WRAPPER_TYPES: Tuple[Type[nn.Module], ...] = (
    BayesLinear, BayesConv1d, BayesConv2d, BayesConv3d, BayesConvTranspose3d, BayesInstanceNorm3d
)


# ============================================================================
# 5. Utility: dotted path resolution & wrapping
# ============================================================================

def get_parent_and_child(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(f"Model has no submodule '{p}' under path '{full_name}'.")
        parent = getattr(parent, p)
    return parent, parts[-1]


def wrap_module_with_prior(module: nn.Module, prior_scale: float = 0.1) -> nn.Module:
    # ✅ idempotent: if already wrapped, return as-is
    if isinstance(module, _BAYES_WRAPPER_TYPES):
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
    if isinstance(module, nn.InstanceNorm3d):
        return BayesInstanceNorm3d(module, prior_scale=prior_scale)

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

        # ✅ skip if already wrapped
        if isinstance(child, _BAYES_WRAPPER_TYPES):
            continue

        scale = per_layer_scale.get(full_name, prior_scale)
        wrapped = wrap_module_with_prior(child, prior_scale=scale)

        if wrapped is child:
            print(f"[BayzFlow] Warning: module '{full_name}' of type {type(child)} not wrapped (unsupported).")
        else:
            setattr(parent, child_name, wrapped)

    return model


# ============================================================================
# 6. Prior calibration from dataset activations
# ============================================================================

def calibrate_prior_scales(
    model: nn.Module,
    module_names: Sequence[str],
    calib_loader: torch.utils.data.DataLoader,
    forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
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

    stats: Dict[str, List[float]] = {name: [] for name in module_names}
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
                if device is not None:
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    elif isinstance(batch, (list, tuple)):
                        batch = tuple(v.to(device) if torch.is_tensor(v) else v for v in batch)
                _ = forward_fn(model, batch)

    for h in hooks:
        h.remove()

    prior_scales: Dict[str, float] = {}
    for name, values in stats.items():
        if not values:
            print(f"[BayzFlow] Warning: no activation stats for '{name}'. Using fallback scale 0.1.")
            prior_scales[name] = 0.1
        else:
            prior_scales[name] = float(np.percentile(values, percentile * 100.0))

    return prior_scales


# ============================================================================
# 7. Candidate discovery (updated to skip wrapped modules)
# ============================================================================

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

        # ✅ skip already-wrapped Bayesian layers
        if isinstance(module, _BAYES_WRAPPER_TYPES):
            continue

        if not isinstance(module, tuple(include_types)):
            continue

        if any(ex in name for ex in exclude_names):
            continue

        candidates.append(ModuleInfo(name=name, module=module))

    return candidates


# ============================================================================
# 8. BayzFlow engine
# ============================================================================

class BayzFlow:
    def __init__(self, model: nn.Module):
        # ✅ avoid mutating user model across calls
        self.base_model = deepcopy(model)

        self.bayes_model: Optional[nn.Module] = None
        self.selected_module_names: List[str] = []
        self.prior_scales: Dict[str, float] = {}
        self.guide = None
        self.optimizer = None
        self.svi: Optional[SVI] = None
        self._pyro_model = None
        self._predict_fn: Optional[Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor]] = None

    # --- inspection helpers ---

    def show_module_tree(self) -> None:
        print_module_tree(self.base_model)

    def list_candidates(
        self,
        include_types: Sequence[Type[nn.Module]] = _DEFAULT_INCLUDE_TYPES,
        exclude_names: Optional[Iterable[str]] = None,
    ) -> List[ModuleInfo]:
        candidates = find_candidate_modules(self.base_model, include_types=include_types, exclude_names=exclude_names)
        print_candidates(candidates)
        return candidates

    # --- selection ---

    def select_modules(
        self,
        candidates: Sequence[ModuleInfo],
        select: Union[Sequence[int], Sequence[str], None] = None,
        patterns: Optional[Sequence[str]] = None,
        selector: Optional[Callable[[str, nn.Module], bool]] = None,
    ) -> List[str]:
        if select is not None and len(select) > 0:
            if isinstance(select[0], int):
                selected_infos = select_by_indices(candidates, select)  # type: ignore[arg-type]
            elif isinstance(select[0], str):
                name_to_info = {c.name: c for c in candidates}
                selected_infos = []
                for nm in select:  # type: ignore[assignment]
                    if nm not in name_to_info:
                        raise KeyError(f"Requested module '{nm}' not in candidates.")
                    selected_infos.append(name_to_info[nm])
            else:
                raise TypeError("`select` must be list of int (indices) or str (names).")
        elif selector is not None:
            selected_infos = select_with_predicate(candidates, selector)
        elif patterns:
            selected_infos = select_by_pattern(candidates, patterns)
        else:
            raise ValueError("No selection criteria provided for select_modules.")

        self.selected_module_names = [info.name for info in selected_infos]
        print("[BayzFlow] Selected modules for priors:")
        for name in self.selected_module_names:
            print("   -", name)
        return self.selected_module_names

    # --- calibration & building ---

    def calibrate_priors(
        self,
        calib_loader: torch.utils.data.DataLoader,
        calib_forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        num_passes: int = 3,
        percentile: float = 0.9,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        if not self.selected_module_names:
            raise RuntimeError("No modules selected. Call select_modules(...) first.")

        print("[BayzFlow] Calibrating priors...")
        self.prior_scales = calibrate_prior_scales(
            self.base_model,
            module_names=self.selected_module_names,
            calib_loader=calib_loader,
            forward_fn=calib_forward_fn,
            num_passes=num_passes,
            percentile=percentile,
            device=device,
        )

        print("[BayzFlow] Calibrated scales:")
        for name, scale in self.prior_scales.items():
            print(f"   {name}: {scale:.4f}")

        return self.prior_scales

    def build_bayesian_model(self, default_prior_scale: float = 0.1) -> nn.Module:
        if not self.selected_module_names:
            raise RuntimeError("No modules selected. Call select_modules(...) first.")

        print("[BayzFlow] Building Bayesian model...")
        self.bayes_model = attach_priors(
            model=self.base_model,
            selected_names=self.selected_module_names,
            prior_scale=default_prior_scale,
            per_layer_scale=self.prior_scales if self.prior_scales else None,
        )
        return self.bayes_model

    # --- training (SVI) ---

    def setup_svi(
        self,
        loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        predict_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        lr: float = 1e-3,
    ):
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built. Call build_bayesian_model() first.")

        pyro.clear_param_store()
        model = self.bayes_model
        self._predict_fn = predict_fn  # ✅ store

        def pyro_model(batch):
            # NOTE: leaving pyro.module in is OK; collision was double-wrapping.
            pyro.module("bayes_model", model, update_module_params=False)

            logits = predict_fn(model, batch)
            pyro.deterministic("logits", logits)

            nll = loss_fn(model, batch)
            pyro.factor("nll", -nll)

            return logits

        self._pyro_model = pyro_model
        self.guide = AutoNormal(self._pyro_model)
        self.optimizer = pyro.optim.Adam({"lr": lr})
        self.svi = SVI(self._pyro_model, self.guide, self.optimizer, loss=Trace_ELBO())

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        device: Optional[torch.device] = None,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        patience: int = 10,
        checkpoint_path: Optional[str] = None,
        save_trajectory: bool = True,
    ):
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built.")
        if self.svi is None or self._pyro_model is None:
            raise RuntimeError("SVI not set up. Call setup_svi(loss_fn, predict_fn, lr) first.")

        model = self.bayes_model
        if device is not None:
            model.to(device)

        if save_trajectory:
            os.makedirs("bayz_artifacts/trajectory", exist_ok=True)
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Training
            total_loss = 0.0
            count = 0
            for batch in train_loader:
                if device is not None and isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss_val = self.svi.step(batch)
                total_loss += float(loss_val)
                count += 1

            avg_train_loss = total_loss / max(count, 1)

            # Validation
            if val_loader is not None:
                val_loss = 0.0
                val_count = 0
                for batch in val_loader:
                    if device is not None and isinstance(batch, dict):
                        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    loss_val = self.svi.evaluate_loss(batch)
                    val_loss += float(loss_val)
                    val_count += 1
                avg_val_loss = val_loss / max(val_count, 1)

                print(f"[BayzFlow] Epoch {epoch+1}/{num_epochs}  Train ELBO: {avg_train_loss:.4f}  Val ELBO: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss - 1e-6:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    if checkpoint_path:
                        self.save_checkpoint(checkpoint_path, epoch=epoch, val_loss=avg_val_loss)
                        print(f"[BayzFlow] Saved checkpoint to {checkpoint_path}")
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"[BayzFlow] Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
                    break
            else:
                print(f"[BayzFlow] Epoch {epoch+1}/{num_epochs}  ELBO: {avg_train_loss:.4f}")
                if checkpoint_path and avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    self.save_checkpoint(checkpoint_path, epoch=epoch, train_loss=avg_train_loss)

            if save_trajectory:
                pyro.get_param_store().save(f"bayz_artifacts/trajectory/posterior_t{epoch:04d}.pt")

    # --- inference ---

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 16,
        device: Optional[torch.device] = None,
        return_samples: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.bayes_model is None or self.guide is None or self._pyro_model is None:
            raise RuntimeError("Bayesian model/guide not initialised.")

        model = self.bayes_model
        if device is not None:
            model.to(device)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

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

        stacked = samp["logits"]  # [S, ...]
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)

        out = {"mean": mean, "std": std}
        if return_samples:
            out["samples"] = stacked
        return out

    # --- one-line helpers ---

    @classmethod
    def from_model_and_data(
        cls,
        model: nn.Module,
        calib_loader: torch.utils.data.DataLoader,
        calib_forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        bayes_patterns: Sequence[str],
        num_calib_passes: int = 3,
        calib_percentile: float = 0.9,
        default_prior_scale: float = 0.1,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> "BayzFlow":
        engine = cls(model)

        if verbose:
            print("=== BayzFlow: module tree ===")
            engine.show_module_tree()
            print("\n=== BayzFlow: candidate layers ===")

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
            print("\n[BayzFlow] Bayesian model ready for SVI / inference.")
        return engine

    @classmethod
    def auto(
        cls,
        model: nn.Module,
        calib_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        calib_forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        predict_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        bayes_patterns: Sequence[str],
        num_calib_passes: int = 3,
        calib_percentile: float = 0.9,
        default_prior_scale: float = 0.1,
        train_epochs: int = 5,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> "BayzFlow":
        engine = cls.from_model_and_data(
            model=model,
            calib_loader=calib_loader,
            calib_forward_fn=calib_forward_fn,
            bayes_patterns=bayes_patterns,
            num_calib_passes=num_calib_passes,
            calib_percentile=calib_percentile,
            default_prior_scale=default_prior_scale,
            device=device,
            verbose=verbose,
        )

        if verbose:
            print("\n[BayzFlow] Starting SVI training...")
        engine.setup_svi(loss_fn=loss_fn, predict_fn=predict_fn, lr=lr)
        engine.fit(train_loader=train_loader, num_epochs=train_epochs, device=device)
        return engine

    # --- checkpointing ---

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        checkpoint = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "base_model_state": self.base_model.state_dict() if self.base_model is not None else None,
            "bayes_model_state": self.bayes_model.state_dict() if self.bayes_model is not None else None,
            "param_store": pyro.get_param_store().get_state(),
            "selected_module_names": self.selected_module_names,
            "prior_scales": self.prior_scales,
        }
        if extra_metadata:
            checkpoint["metadata"] = extra_metadata
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location)

        if self.base_model is not None and checkpoint.get("base_model_state"):
            self.base_model.load_state_dict(checkpoint["base_model_state"], strict=False)

        if self.bayes_model is not None and checkpoint.get("bayes_model_state"):
            self.bayes_model.load_state_dict(checkpoint["bayes_model_state"], strict=False)

        if checkpoint.get("param_store"):
            pyro.get_param_store().set_state(checkpoint["param_store"])

        if checkpoint.get("selected_module_names"):
            self.selected_module_names = checkpoint["selected_module_names"]

        if checkpoint.get("prior_scales"):
            self.prior_scales = checkpoint["prior_scales"]

        print(f"[BayzFlow] Loaded checkpoint from {path}")
        return {
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "val_loss": checkpoint.get("val_loss"),
            "metadata": checkpoint.get("metadata", {}),
        }


# ============================================================================
# 9. Tiny demo
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
            return self.head(self.encoder(x))

    class TinyDS(torch.utils.data.Dataset):
        def __len__(self):
            return 128

        def __getitem__(self, idx):
            return {"x": torch.randn(10), "y": torch.randint(0, 3, ()).long()}

    def calib_forward_fn(model, batch):
        return model(batch["x"])

    def loss_fn(model, batch):
        logits = model(batch["x"])
        return F.cross_entropy(logits, batch["y"])

    def predict_fn(model, batch):
        return model(batch["x"])

    ds = TinyDS()
    calib_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

    net = TinyNet()

    bf = BayzFlow.auto(
        model=net,
        calib_loader=calib_loader,
        train_loader=train_loader,
        calib_forward_fn=calib_forward_fn,
        loss_fn=loss_fn,
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
    out = bf.predict(batch, num_samples=16, return_samples=False)
    print("\nMC mean shape:", out["mean"].shape, "std shape:", out["std"].shape)
