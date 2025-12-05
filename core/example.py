import os
import sys

# ---------------------------------------------------------------------
# 1) Make project root importable so `core.engine` resolves
#    This file is .../bayzflow/core/example_resnet_fullflow.py
#    Project root is .../bayzflow
# ---------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
THIS_DIR = os.path.dirname(THIS_FILE)            # .../bayzflow/core
ROOT = os.path.dirname(THIS_DIR)                # .../bayzflow

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------
# 2) Project + library imports
# ---------------------------------------------------------------------
from core.engine import BayzFlow

import matplotlib.pyplot as plt
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# 3) Build a vanilla ResNet18 + CIFAR-10 loaders
# ---------------------------------------------------------------------
def build_resnet18_cifar10(batch_size: int = 8):
    """
    Build a plain ResNet18 classifier for CIFAR-10, plus train/calib loaders.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    data_root = os.path.join(ROOT, "data")

    train_ds = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_ds = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    # For a quick demo we use small subsets
    train_subset = Subset(train_ds, range(0, 5048))
    calib_subset = Subset(test_ds, range(0, 1012))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(calib_subset, batch_size=batch_size, shuffle=True)

    # Vanilla ResNet18, re-headed for CIFAR-10 (10 classes)
    num_classes = 10
    model = torchvision.models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.to(device)
    return model, train_loader, calib_loader, device

def plot_mc_bars_for_sample(mean_probs, std_probs, sample_idx=0, class_names=None):
    """
    mean_probs: [B, C]
    std_probs:  [B, C]
    """
    mean = mean_probs[sample_idx].detach().cpu().numpy()
    std  = std_probs[sample_idx].detach().cpu().numpy()
    C = mean.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(C)]

    plt.figure(figsize=(8, 4))
    x = range(C)
    plt.bar(x, mean, yerr=std, capsize=4)
    plt.xticks(x, class_names, rotation=45)
    plt.ylabel("P(class)")
    plt.xlabel("Class")
    top = mean.argmax()
    plt.title(
        f"Sample {sample_idx} — argmax={class_names[top]} "
        f"p={mean[top]:.3f}, ep_std={std[top]:.3f}"
    )
    plt.tight_layout()
    plt.show()

def plot_confidence_vs_epistemic(mean_probs, std_probs):
    """
    mean_probs, std_probs: [B, C]
    """
    mean = mean_probs.detach().cpu()
    std  = std_probs.detach().cpu()

    # max prob and its corresponding epistemic std per sample
    p_max, idx_max = mean.max(dim=1)
    ep_max = std.gather(1, idx_max.unsqueeze(1)).squeeze(1)

    plt.figure(figsize=(5, 5))
    plt.scatter(p_max.numpy(), ep_max.numpy(), alpha=0.7)
    plt.xlabel("max P(class)")
    plt.ylabel("epistemic std at argmax")
    plt.title("Confidence vs Epistemic Uncertainty (per sample)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    return p_max, ep_max



# ---------------------------------------------------------------------
# 4) Full BayzFlow manual flow
# ---------------------------------------------------------------------
def main():
    model, train_loader, calib_loader, device = build_resnet18_cifar10(batch_size=64)

    # ---- Step 1: show full module tree (diagram) ----
    print("\n=== [1] MODULE TREE ===")
    bf = BayzFlow(model)
    bf.show_module_tree()

    # ---- Step 2: list candidate modules with indices ----
    print("\n=== [2] CANDIDATE LAYERS (with indices) ===")
    candidates = bf.list_candidates()

    if not candidates:
        raise RuntimeError("No candidate modules found – check include_types / model definition.")

    # Simple heuristic: suggest the last candidate
    suggested_idx = len(candidates) - 1

    # Guard against out-of-range
    if suggested_idx < 0 or suggested_idx >= len(candidates):
        print("[BayzFlow] No valid suggested index, falling back to manual selection.")
        suggested_idx = None
    else:
        print(
            f"Suggested Bayesian layer index: {suggested_idx} "
            f"({candidates[suggested_idx].name})"
        )

    # When selecting modules:
    if suggested_idx is not None:
        selected = bf.select_modules(candidates, select=[suggested_idx])
    else:
        # fallback – e.g. select by name patterns
        selected = bf.select_modules(candidates, patterns=["head", "layer4.1", "fc"])
    print("\n=== [User Selection] Choose Bayesian layers ===")

    # Suggest index for "fc" or the last Linear layer
    suggested_idx = None
    for idx, info in enumerate(candidates):
        if info.name == "fc":
            suggested_idx = idx
            break
    # If no fc, fall back to last candidate
    if suggested_idx is None:
        suggested_idx = len(candidates) - 1

    print(f"Suggested Bayesian layer index: {suggested_idx} ({candidates[suggested_idx].name})")
    print("Enter indices separated by spaces to select modules.")
    print("Press ENTER to accept suggestion.")
    print("Or type '*' to switch to pattern-based mode.\n")

    user_input = input("Select indices (default = suggested): ").strip()

    # OPTION A fallback if user enters "*"
    if user_input == "*":
        print("\n[BayzFlow] Switching to pattern-based selection...")
        selected_names = bf.select_modules(candidates, patterns=["fc"])
        print("Selected via patterns:", selected_names)

    # OPTION B – index-based selection
    else:
        if user_input == "":
            indices = [suggested_idx]
            print(f"[BayzFlow] Using default index: {indices}")
        else:
            try:
                indices = [int(tok) for tok in user_input.split()]
            except ValueError:
                raise ValueError("Invalid input: only integers allowed for indices.")

        # Validate the indices
        for idx in indices:
            if idx < 0 or idx >= len(candidates):
                raise ValueError(f"Index {idx} is out of range.")

        selected_names = bf.select_modules(candidates, select=indices)
        print(f"Selected indices {indices} → modules:", selected_names)

    # You’ll see lines like:
    # [000] conv1 ...
    # [001] layer1.0.conv1 ...
    # ...
    # [XYZ] fc Linear

    # ---- Step 3: choose which modules get priors ----
    # Option A: select by pattern (name contains "fc")
    # selected_names = bf.select_modules(candidates, patterns=["fc"])

    # Option B: select manually by index after looking at printed list.
    # For a first run, we just Bayesianise the final classifier head.
    # Find the index where name == "fc":
    #fc_idx = None
    #for idx, info in enumerate(candidates):
    #    if info.name == "fc":
    #        fc_idx = idx
    #        break
    #if fc_idx is None:
    #    raise RuntimeError("Could not find 'fc' in candidates; check list_candidates output.")

    #print(f"\n[BayzFlow] Using candidate index {fc_idx} ('fc') as Bayesian layer.")
    #selected_names = bf.select_modules(candidates, select=[fc_idx])

    # ---- Step 4: define calib_forward_fn, loss_fn, predict_fn ----

    # a) Calibration forward: just run the model so hooks see activations
    def calib_forward_fn(m, batch):
        x, _ = batch
        x = x.to(device)
        _ = m(x)
        return _

    # b) Training loss: standard cross-entropy
    def loss_fn(m, batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = m(x)              # [B, 10]
        return F.cross_entropy(logits, y)

    # c) Prediction: what we want from BF.predict()
    def predict_fn(m, batch):
        x = batch["x"].to(device)
        logits = m(x)   # model already expects a tensor
        return F.softmax(logits, dim=-1)

    # ---- Step 5: calibrate priors from activation stats ----
    print("\n=== [3] PRIOR CALIBRATION ===")
    bf.calibrate_priors(
        calib_loader=calib_loader,
        calib_forward_fn=calib_forward_fn,
        num_passes=2,
        percentile=0.9,
        device=device,
    )

    # ---- Step 6: build Bayesian model (attaches priors to chosen modules) ----
    print("\n=== [4] BUILD BAYESIAN MODEL ===")
    bf.build_bayesian_model(default_prior_scale=0.1)

    # ---- Step 7: setup SVI + train ----
    print("\n=== [5] SVI TRAINING ===")
    bf.setup_svi(loss_fn=loss_fn, lr=1e-3)
    bf.fit(
        train_loader=train_loader,
        loss_fn=loss_fn,
        num_epochs=500,         # keep small for demo
        device=device,
    )

    # ---- Step 8: MC prediction with uncertainty ----
    print("\n=== [6] MONTE-CARLO PREDICTION ===")
    batch_tuple = next(iter(calib_loader))
    x, y = batch_tuple
    batch = {"x": x, "y": y}

    out = bf.predict(
        batch,
        predict_fn=predict_fn,
        num_samples=64,
        device=device,
    )

    mean_probs = out["mean"]   # [B, 10]
    std_probs = out["std"]     # [B, 10]
    print("MC mean probs shape:", mean_probs.shape)
    print("MC std shape:", std_probs.shape)

    # Show top-1 prediction + epistemic uncertainty for first few samples
    top_probs, top_idx = mean_probs.topk(1, dim=-1)
    for i in range(min(5, mean_probs.shape[0])):
        cls = top_idx[i].item()
        p = top_probs[i].item()
        ep_std = std_probs[i, cls].item()
        print(f"Sample {i}: class={cls}  p={p:.3f}  epistemic_std={ep_std:.3f}")
        mean_probs = out["mean"]
        std_probs  = out["std"]
        CLASS_NAMES = [str(i) for i in range(mean_probs.shape[1])] 
        plot_mc_bars_for_sample(mean_probs, std_probs, sample_idx=0, class_names=CLASS_NAMES)


if __name__ == "__main__":
    main()
