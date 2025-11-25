"""
bayzflow.py

BayzFlow engine — turn any PyTorch model into a calibrated Bayesian (Pyro) model
with transparent module selection and automatic prior calibration.

Key features:
    - Torch-model agnostic (any nn.Module)
    - Pretty-printed module tree
    - Candidate layer listing with indices (Linear / Conv / Norm, configurable)
    - User-driven or pattern-based selection of Bayesian layers
    - Automatic wrapping of selected layers with PyroModules (BayesLinear / BayesConv*)
    - Automatic prior calibration using dataset activations over N passes + percentile
    - SVI training with a proper Pyro model
    - Monte Carlo predictive inference with uncertainty
    - Human-in-the-loop "tighten posterior" updates
    - One-line high-level API via BayzFlow.auto(...)

UI can come later; this is pure Python.

Requires:
    torch
    pyro-ppl
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.infer import Predictive


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
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


def find_candidate_modules(
    model: nn.Module,
    include_types: Sequence[Type[nn.Module]] = _DEFAULT_INCLUDE_TYPES,
    exclude_names: Optional[Iterable[str]] = None,
) -> List[ModuleInfo]:
    """
    Walk the model tree and collect modules that are likely interesting
    for Bayesianisation (by default: Linear / Conv / Norm layers).
    """
    exclude_names = set(exclude_names or [])
    candidates: List[ModuleInfo] = []

    for name, module in model.named_modules():
        if name == "":  # skip root
            continue

        if not isinstance(module, tuple(include_types)):
            continue

        if any(ex in name for ex in exclude_names):
            continue

        candidates.append(ModuleInfo(name=name, module=module))

    return candidates


def print_candidates(candidates: Sequence[ModuleInfo]) -> None:
    """Pretty-print candidate modules with indices."""
    for idx, info in enumerate(candidates):
        module = info.module
        print(f"[{idx:03}] {info.name:60} {module.__class__.__name__}")


# ============================================================================
# 2. Module tree / map
# ============================================================================


def build_module_tree(model: nn.Module) -> Dict:
    """
    Build a nested dict representing the module hierarchy.
    """
    tree = {"type": model.__class__.__name__, "children": {}}
    name_to_module = dict(model.named_modules())

    for full_name, module in name_to_module.items():
        if full_name == "":
            continue

        parts = full_name.split(".")
        node = tree
        for i, part in enumerate(parts):
            if "children" not in node:
                node["children"] = {}

            if part not in node["children"]:
                node["children"][part] = {"type": None, "children": {}}

            node = node["children"][part]

            if i == len(parts) - 1:
                node["type"] = module.__class__.__name__

    return tree


def _print_tree_node(name: str, node: Dict, indent: str, is_last: bool):
    """Helper for pretty tree print."""
    connector = "└── " if is_last else "├── "
    line = f"{indent}{connector}{name} ({node.get('type', 'Unknown')})"
    print(line)

    children = node.get("children", {})
    if not children:
        return

    next_indent = indent + ("    " if is_last else "│   ")
    child_items = list(children.items())
    for i, (child_name, child_node) in enumerate(child_items):
        _print_tree_node(
            child_name,
            child_node,
            indent=next_indent,
            is_last=(i == len(child_items) - 1),
        )


def print_module_tree(model: nn.Module) -> None:
    """Pretty-print the full module hierarchy as a tree."""
    tree = build_module_tree(model)
    print("root", f"({tree.get('type', 'Model')})")
    children = tree.get("children", {})
    child_items = list(children.items())
    for i, (name, node) in enumerate(child_items):
        _print_tree_node(
            name,
            node,
            indent="",
            is_last=(i == len(child_items) - 1),
        )


# ============================================================================
# 3. Selection helpers
# ============================================================================


def select_by_indices(
    candidates: Sequence[ModuleInfo],
    indices: Sequence[int],
) -> List[ModuleInfo]:
    """Select subset of candidates by integer indices."""
    selected: List[ModuleInfo] = []
    for i in indices:
        if i < 0 or i >= len(candidates):
            raise IndexError(f"Index {i} out of range for {len(candidates)} candidates.")
        selected.append(candidates[i])
    return selected


def select_by_pattern(
    candidates: Sequence[ModuleInfo],
    patterns: Sequence[str],
) -> List[ModuleInfo]:
    """Select modules whose name contains any of the given patterns."""
    patterns = list(patterns)
    selected: List[ModuleInfo] = []
    for info in candidates:
        if any(p in info.name for p in patterns):
            selected.append(info)
    return selected


def select_with_predicate(
    candidates: Sequence[ModuleInfo],
    predicate: Callable[[str, nn.Module], bool],
) -> List[ModuleInfo]:
    """Select modules using a custom predicate(name, module) -> bool."""
    selected: List[ModuleInfo] = []
    for info in candidates:
        if predicate(info.name, info.module):
            selected.append(info)
    return selected


# ============================================================================
# 4. Bayesian wrapper layers (PyroModule-based)
# ============================================================================


class BayesLinear(PyroModule):
    """
    Bayesian replacement for nn.Linear.

    Uses the original weights as prior means, with configurable prior std.
    """

    def __init__(self, base_linear: nn.Linear, prior_scale: float = 0.1):
        super().__init__()
        if base_linear.weight is None:
            raise ValueError("Base Linear has no weight tensor.")

        weight_mean = base_linear.weight.detach()
        bias_mean = (
            base_linear.bias.detach() if base_linear.bias is not None else None
        )

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.weight = PyroSample(
            dist.Normal(weight_mean, prior_scale).to_event(2)
        )

        if bias_mean is not None:
            self.bias = PyroSample(
                dist.Normal(bias_mean, prior_scale).to_event(1)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class BayesConvNd(PyroModule):
    """
    Base class for Bayesian ConvNd replacement.

    conv_dim: 1, 2, or 3. Use BayesConv1d/2d/3d subclasses.
    """

    conv_dim: int = 2

    def __init__(self, base_conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], prior_scale: float = 0.1):
        super().__init__()

        weight_mean = base_conv.weight.detach()
        bias_mean = base_conv.bias.detach() if base_conv.bias is not None else None

        self.in_channels = base_conv.in_channels
        self.out_channels = base_conv.out_channels
        self.kernel_size = base_conv.kernel_size
        self.stride = base_conv.stride
        self.padding = base_conv.padding
        self.dilation = base_conv.dilation
        self.groups = base_conv.groups

        self.weight = PyroSample(
            dist.Normal(weight_mean, prior_scale).to_event(weight_mean.dim())
        )

        if bias_mean is not None:
            self.bias = PyroSample(
                dist.Normal(bias_mean, prior_scale).to_event(1)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_dim == 1:
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.conv_dim == 2:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.conv_dim == 3:
            return F.conv3d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            raise RuntimeError(f"Unsupported conv_dim={self.conv_dim}")


class BayesConv1d(BayesConvNd):
    conv_dim = 1


class BayesConv2d(BayesConvNd):
    conv_dim = 2


class BayesConv3d(BayesConvNd):
    conv_dim = 3


# ============================================================================
# 5. Utility: dotted path resolution & wrapping
# ============================================================================


def get_parent_and_child(
    model: nn.Module,
    full_name: str,
) -> Tuple[nn.Module, str]:
    """
    Given a dotted module path like "encoder.0", return (parent_module, "0").
    """
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(
                f"Model has no submodule '{p}' under path '{full_name}'."
            )
        parent = getattr(parent, p)
    child_name = parts[-1]
    return parent, child_name


def wrap_module_with_prior(
    module: nn.Module,
    prior_scale: float = 0.1,
) -> PyroModule:
    """
    Wrap a single nn.Module with a Bayesian PyroModule, if supported.

    Supported:
        - nn.Linear
        - nn.Conv1d/2d/3d

    Otherwise returns the original module unchanged.
    """
    if isinstance(module, nn.Linear):
        return BayesLinear(module, prior_scale=prior_scale)
    elif isinstance(module, nn.Conv1d):
        return BayesConv1d(module, prior_scale=prior_scale)
    elif isinstance(module, nn.Conv2d):
        return BayesConv2d(module, prior_scale=prior_scale)
    elif isinstance(module, nn.Conv3d):
        return BayesConv3d(module, prior_scale=prior_scale)
    else:
        return module  # not wrapped


def attach_priors(
    model: nn.Module,
    selected_names: Sequence[str],
    prior_scale: float = 0.1,
    per_layer_scale: Optional[Dict[str, float]] = None,
) -> nn.Module:
    """
    Replace selected submodules in the model with Bayesian PyroModules,
    attaching Normal priors centred at existing weights.

    Note: the root model can remain a plain nn.Module; the selected
    submodules become PyroModules with PyroSample parameters.
    """
    per_layer_scale = per_layer_scale or {}

    for full_name in selected_names:
        parent, child_name = get_parent_and_child(model, full_name)
        child = getattr(parent, child_name)

        scale = per_layer_scale.get(full_name, prior_scale)
        wrapped = wrap_module_with_prior(child, prior_scale=scale)

        if wrapped is child:
            print(
                f"[BayzFlow] Warning: module '{full_name}' of type {type(child)} "
                f"not wrapped (unsupported)."
            )
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
    """
    Estimate per-layer prior scales from activation statistics.

    forward_fn(model, batch) should simply "run the model" on the batch inputs
    (its return value is ignored here) so that hooks see real activations.
    """
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
            std_val = x.view(-1).std().item()
            stats[name].append(std_val)
        return hook

    for name in module_names:
        h = name_to_module[name].register_forward_hook(make_hook(name))
        hooks.append(h)

    with torch.no_grad():
        for _ in range(num_passes):
            for batch in calib_loader:
                if device is not None and isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                _ = forward_fn(model, batch)  # we only care that it runs

    for h in hooks:
        h.remove()

    prior_scales: Dict[str, float] = {}
    for name, values in stats.items():
        if not values:
            print(f"[BayzFlow] Warning: no activation stats for '{name}'. Using fallback scale 0.1.")
            prior_scales[name] = 0.1
        else:
            p = np.percentile(values, percentile * 100.0)
            prior_scales[name] = float(p)

    return prior_scales


# ============================================================================
# 7. BayzFlow engine
# ============================================================================


class BayzFlow:
    """
    Core BayzFlow engine.

    Orchestrates:
        - module tree display
        - candidate discovery
        - user selection of Bayesian modules
        - Bayesian wrapping
        - prior calibration
        - SVI training
        - Monte Carlo inference (uncertainty)
        - Online posterior tightening from human feedback
    """

    def __init__(self, model: nn.Module):
        self.base_model = model
        self.bayes_model: Optional[PyroModule] = None
        self.selected_module_names: List[str] = []
        self.prior_scales: Dict[str, float] = {}
        self.guide: Optional[AutoLowRankMultivariateNormal] = None
        self.optimizer = None
        self.svi: Optional[SVI] = None
        self._pyro_model: Optional[Callable] = None  # model(batch) used by SVI

    # --- inspection helpers ---

    def show_module_tree(self) -> None:
        print_module_tree(self.base_model)

    def list_candidates(
        self,
        include_types: Sequence[Type[nn.Module]] = _DEFAULT_INCLUDE_TYPES,
        exclude_names: Optional[Iterable[str]] = None,
    ) -> List[ModuleInfo]:
        candidates = find_candidate_modules(
            self.base_model,
            include_types=include_types,
            exclude_names=exclude_names,
        )
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
        """
        Choose which modules to Bayesianise.

        - select = [idx...]    -> by candidate index
        - select = ["name..."] -> by explicit dotted names
        - patterns = ["..."]   -> name contains pattern
        - selector = predicate(name, module) -> bool
        """
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
        """
        Run automatic prior calibration based on activation stats.

        calib_forward_fn(model, batch) should run the model on the batch inputs.
        """
        if not self.selected_module_names:
            raise RuntimeError("No modules selected. Call select_modules(...) first.")

        print("[BayzFlow] Calibrating priors...")
        prior_scales = calibrate_prior_scales(
            self.base_model,
            module_names=self.selected_module_names,
            calib_loader=calib_loader,
            forward_fn=calib_forward_fn,
            num_passes=num_passes,
            percentile=percentile,
            device=device,
        )

        self.prior_scales = prior_scales
        print("[BayzFlow] Calibrated scales:")
        for name, scale in prior_scales.items():
            print(f"   {name}: {scale:.4f}")

        return prior_scales

    def build_bayesian_model(
        self,
        default_prior_scale: float = 0.1,
    ) -> PyroModule:
        """
        Attach priors to selected modules and produce a PyroModule.
        """
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
        lr: float = 1e-3,
    ):
        """
        Setup SVI with a proper Pyro model embedding loss_fn.

        loss_fn(model, batch) must return a scalar negative log-likelihood
        (or any loss you want to minimise).
        """
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built. Call build_bayesian_model() first.")

        pyro.clear_param_store()
        model = self.bayes_model

        def pyro_model(batch):
            """
            Pyro model for SVI: runs the Bayesian model and applies loss_fn.
            The Bayesian layers create latent sites via PyroSample; loss_fn
            contributes a factor corresponding to -NLL.
            """
            pyro.module("bayes_model", model)
            nll = loss_fn(model, batch)
            pyro.factor("nll", -nll)
            return nll

        self._pyro_model = pyro_model
        self.guide = AutoLowRankMultivariateNormal(self._pyro_model)
        self.optimizer = pyro.optim.Adam({"lr": lr})
        self.svi = SVI(self._pyro_model, self.guide, self.optimizer, loss=Trace_ELBO())

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        num_epochs: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Run SVI training over the training loader.

        loss_fn(model, batch) should return a scalar NLL (or loss).
        """
        if self.bayes_model is None:
            raise RuntimeError("Bayesian model not built.")
        if self.svi is None:
            self.setup_svi(loss_fn)

        model = self.bayes_model
        if device is not None:
            model.to(device)

        for epoch in range(num_epochs):
            total_loss = 0.0
            count = 0
            for batch in train_loader:
                if device is not None and isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss_val = self.svi.step(batch)  # batch passed into pyro_model/guide
                total_loss += loss_val
                count += 1

            avg_loss = total_loss / max(count, 1)
            print(f"[BayzFlow] Epoch {epoch+1}/{num_epochs}  ELBO: {avg_loss:.4f}")

    # --- human-in-the-loop online tightening ---

    def tighten_posterior(
        self,
        feedback_loader: torch.utils.data.DataLoader,
        loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        num_epochs: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Online "tighten posterior" pass using new human-labelled feedback data.

        Reuses the existing guide and optimiser, performs extra SVI steps on
        the feedback_loader, effectively updating the posterior in-place.
        """
        if self.bayes_model is None or self.svi is None or self._pyro_model is None:
            # If SVI wasn't set up yet, initialise it now
            self.setup_svi(loss_fn)

        model = self.bayes_model
        if device is not None:
            model.to(device)

        print("[BayzFlow] Tightening posterior with feedback data...")
        for epoch in range(num_epochs):
            total_loss = 0.0
            count = 0
            for batch in feedback_loader:
                if device is not None and isinstance(batch, dict):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss_val = self.svi.step(batch)
                total_loss += loss_val
                count += 1
            avg_loss = total_loss / max(count, 1)
            print(f"[BayzFlow] Feedback epoch {epoch+1}/{num_epochs}  ELBO: {avg_loss:.4f}")

    # --- inference ---

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        predict_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
        num_samples: int = 16,
        device: Optional[torch.device] = None,
        return_samples: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo predictive inference with uncertainty.

        predict_fn(model, batch) -> prediction tensor (e.g. logits, masks, etc.)
        """
        if self.bayes_model is None or self.guide is None or self._pyro_model is None:
            raise RuntimeError("Bayesian model/guide not initialised.")

        model = self.bayes_model
        if device is not None:
            model.to(device)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        model.eval()
        samples = []

        # We'll use Predictive to sample posterior parameter sets, but it's
        # easiest with the autoguide directly:
        # For global-latent autoguides, guide(batch) returns a sample dict
        # that can be used to condition the model parameters.
        for _ in range(num_samples):
            # guide requires an input; for global latents it's usually ignored,
            # so using this same batch is fine.
            latent_sample = self.guide(batch)
            lifted_model = pyro.poutine.condition(model, data=latent_sample)

            with torch.no_grad():
                pred = predict_fn(lifted_model, batch)
            samples.append(pred.detach())

        stacked = torch.stack(samples, dim=0)  # [S, ...]
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

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
        """
        High-level helper (no training):

            bf = BayzFlow.from_model_and_data(
                model,
                calib_loader,
                calib_forward_fn=lambda m,b: m(b["x"]),
                bayes_patterns=["head", "decoder"],
            )
        """
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
        bayes_patterns: Sequence[str],
        num_calib_passes: int = 3,
        calib_percentile: float = 0.9,
        default_prior_scale: float = 0.1,
        train_epochs: int = 5,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> "BayzFlow":
        """
        Full one-line flow:

            bf = BayzFlow.auto(
                model,
                calib_loader,
                train_loader,
                calib_forward_fn=lambda m,b: m(b["x"]),
                loss_fn=classification_loss_fn,
                bayes_patterns=["head"],
            )

        Steps:
            1) Create BayzFlow
            2) Discover candidates
            3) Select modules by patterns
            4) Calibrate priors (num_calib_passes, percentile)
            5) Build Bayesian Pyro model
            6) Setup SVI and train for `train_epochs`
        """
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
        engine.setup_svi(loss_fn=loss_fn, lr=lr)
        engine.fit(
            train_loader=train_loader,
            loss_fn=loss_fn,
            num_epochs=train_epochs,
            device=device,
        )
        return engine


# ============================================================================
# 8. Tiny demo 
# ============================================================================


if __name__ == "__main__":
    # Minimal toy demo on a simple MLP for classification.

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

    # Fake dataset: dict batches {"x": tensor, "y": tensor}
    class TinyDS(torch.utils.data.Dataset):
        def __len__(self):
            return 128

        def __getitem__(self, idx):
            x = torch.randn(10)
            y = torch.randint(0, 3, ()).long()
            return {"x": x, "y": y}

    def calib_forward_fn(model, batch):
        # Just run the model so hooks see activations
        logits = model(batch["x"])
        return logits

    def loss_fn(model, batch):
        logits = model(batch["x"])      # [B, C]
        y = batch["y"]                  # [B]
        return F.cross_entropy(logits, y)


    def predict_fn(model, batch):
        # Normalize batch to dict
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                batch = {"x": batch[0], "y": batch[1]}
            else:
                raise ValueError("Unsupported tuple batch format for predict()")

        elif not isinstance(batch, dict):
            raise ValueError("Batch must be dict or (x, y) tuple")
        logits = model(batch["x"])
        return logits

    ds = TinyDS()
    calib_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

    net = TinyNet()

    # 🔥 ONE-LINE API
    bf = BayzFlow.auto(
        model=net,
        calib_loader=calib_loader,
        train_loader=train_loader,
        calib_forward_fn=calib_forward_fn,
        loss_fn=loss_fn,
        bayes_patterns=["head"],     # make only the final head Bayesian
        num_calib_passes=2,
        calib_percentile=0.9,
        default_prior_scale=0.1,
        train_epochs=3,
        lr=1e-3,
        device=None,
        verbose=True,
    )

    # Example: MC prediction on one batch
    batch = next(iter(train_loader))
    out = bf.predict(batch, predict_fn=predict_fn, num_samples=16)
    print("\nMC mean shape:", out["mean"].shape, "std shape:", out["std"].shape)
