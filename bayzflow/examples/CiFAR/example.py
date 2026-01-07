import os
import sys
import matplotlib.pyplot as plt
import json
import os
import pyro

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


 


# A minimal ResNet-9 good for CIFAR-10
class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512 * 8 * 8, num_classes) 
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out + self.res2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# ---------------------------------------------------------------------
# 3) Build a vanilla ResNet18 + CIFAR-10 loaders
# ---------------------------------------------------------------------
def build_resnet18_cifar10(batch_size: int = 8):
    """
    Build a plain ResNet18 classifier for CIFAR-10, plus train/calib loaders.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
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
    model = ResNet9(num_classes=num_classes)
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

# ... all your imports and build_resnet18_cifar10 stay the same ...


# ---------------------------------------------------------------------
# (NEW) 3b) Plain deterministic pretraining for the base model
# ---------------------------------------------------------------------
  
def load_base_model(model: nn.Module,
                    device: torch.device,
                    ckpt_path: str = "checkpoints/resnet9_cifar10.pth"):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print(f"[Load] Loaded base model weights from {ckpt_path}")
    return model


def pretrain_base_model(
    model: nn.Module,
    train_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
):
    model.train()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        print(f"[Pretrain] Epoch {ep+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.3f}")
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), "checkpoints/resnet9_cifar10.pth")


# ---------------------------------------------------------------------
# 4) Full BayzFlow manual flow
# ---------------------------------------------------------------------
def main():
    model, train_loader, calib_loader, device = build_resnet18_cifar10(batch_size=8)

    # ---- (NEW) Step 0: deterministic pretraining of the whole ResNet ----
    print("\n=== [0] PRETRAIN BASE MODEL (deterministic) ===")
    #pretrain_base_model(model, train_loader, device, epochs=50, lr=1e-3)
    model = load_base_model(model, device, "checkpoints/resnet9_cifar10.pth")
    x_dbg, y_dbg = next(iter(calib_loader))
    x_dbg = x_dbg.to(device)

    with torch.no_grad():
        logits_det = model(x_dbg)
        probs_det = logits_det.softmax(dim=-1)

    print("Base probs[0]:", probs_det[0])
    entropy_det = -(probs_det[0] * probs_det[0].clamp_min(1e-8).log()).sum().item()
    print("Base entropy[0]:", entropy_det)
    acc_det = (probs_det.argmax(dim=1) == y_dbg.to(probs_det.device)).float().mean().item()
    print("Base batch acc:", acc_det)


    # Optional sanity check
    print("\n=== [DEBUG] Base model sanity check after pretrain ===")
    batch_tuple = next(iter(calib_loader))
    x_dbg, y_dbg = batch_tuple
    x_dbg = x_dbg.to(device)
    with torch.no_grad():
        base_logits = model(x_dbg)
        print("base model (post-pretrain): max diff logits[0] vs [1]:",
              (base_logits[0] - base_logits[1]).abs().max().item())

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
        selected = bf.select_modules(candidates, patterns=["head", "layer4.1", "fc"])
    print("\n=== [User Selection] Choose Bayesian layers ===")

    # (Unchanged: your interactive selection of fc)
    suggested_idx = None
    for idx, info in enumerate(candidates):
        if info.name == "fc":
            suggested_idx = idx
            break
    if suggested_idx is None:
        suggested_idx = len(candidates) - 1

    print(f"Suggested Bayesian layer index: {suggested_idx} ({candidates[suggested_idx].name})")
    print("Enter indices separated by spaces to select modules.")
    print("Press ENTER to accept suggestion.")
    print("Or type '*' to switch to pattern-based mode.\n")

    user_input = input("Select indices (default = suggested): ").strip()

    if user_input == "*":
        print("\n[BayzFlow] Switching to pattern-based selection...")
        selected_names = bf.select_modules(candidates, patterns=["fc"])
        print("Selected via patterns:", selected_names)
    else:
        if user_input == "":
            indices = [suggested_idx]
            print(f"[BayzFlow] Using default index: {indices}")
        else:
            indices = [int(tok) for tok in user_input.split()]
        selected_names = bf.select_modules(candidates, select=indices)
        print(f"Selected indices {indices} → modules:", selected_names)

    # ---- Step 4: define calib_forward_fn, loss_fn, predict_fn ----

    def calib_forward_fn(m, batch):
        x, _ = batch
        x = x.to(device)
        _ = m(x)
        return _

    def loss_fn(m, batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = m(x)              # [B, 10]
        return F.cross_entropy(logits, y)

    # (CHANGED) predict_fn now returns logits (not softmax) for clarity
    def predict_fn(m, batch):
        x = batch["x"].to(device)
        logits = m(x)
        return logits

    # ---- Step 5: calibrate priors from activation stats (on pretrained model) ----
    print("\n=== [3] PRIOR CALIBRATION ===")
    bf.calibrate_priors(
        calib_loader=calib_loader,
        calib_forward_fn=calib_forward_fn,
        num_passes=3,
        percentile=0.87,
        device=device,
    )

    # ---- Step 6: build Bayesian model (attaches priors to chosen modules) ----
    print("\n=== [4] BUILD BAYESIAN MODEL ===")
    bf.build_bayesian_model(default_prior_scale=0.1)

    # ---- (Optional) Step 6.5: verify base_model still varies across batch ----
    batch_tuple = next(iter(calib_loader))
    x_dbg, y_dbg = batch_tuple
    x_dbg = x_dbg.to(device)
    with torch.no_grad():
        logits_dbg = bf.base_model(x_dbg)
        print("base_model (after wrap): max diff logits[0] vs [1]:",
              (logits_dbg[0] - logits_dbg[1]).abs().max().item())

    # ---- Step 7: setup SVI + train posterior ONLY (backbone frozen by update_module_params=False) ----
    print("\n=== [5] SVI TRAINING (posterior over head only) ===")
    bf.setup_svi(loss_fn=loss_fn, lr=1e-3)
    bf.fit(
        train_loader=train_loader,
        loss_fn=loss_fn,
        num_epochs=15,         # this now only shapes the posterior
        device=device,
    )

    # ---- Step 8: MC prediction with uncertainty ----
    # (rest of your block is unchanged except we now softmax once)
    import pyro
    import json

    def show_image_with_title(img_tensor, title: str):
        img = img_tensor.detach().cpu()
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = img.permute(1, 2, 0).numpy()
        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # === [6] MONTE-CARLO PREDICTION & POSTERIOR SNAPSHOT ===
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

    mean_logits = out["mean"]         # [B, C] logits
    std_logits  = out["std"]          # [B, C]
    B, C = mean_logits.shape

    # Convert logits -> probs once
    mean_probs = mean_logits.softmax(dim=-1)
    std_probs = std_logits

    print("MC mean probs shape:", mean_probs.shape)
    print("MC std (logits) shape:", std_logits.shape)


    # Granular per-sample metrics
    top_probs, top_idx = mean_probs.topk(1, dim=-1)      # [B, 1]
    pred_classes = top_idx.squeeze(-1)                   # [B]
    max_prob = top_probs.squeeze(-1)                     # [B]
    epistemic_std_pred = std_probs[torch.arange(B), pred_classes]  # [B]

    # Predictive entropy per sample
    entropy = -(mean_probs * (mean_probs.clamp_min(1e-8).log())).sum(dim=-1)  # [B]

    print("\nSample-level uncertainty (first few examples):")
    for i in range(min(5, B)):
        cls = int(pred_classes[i].item())
        p = float(max_prob[i].item())
        ep = float(epistemic_std_pred[i].item())
        ent = float(entropy[i].item())
        true_y = int(y[i].item())

        print(
            f"Sample {i}: pred={cls} true={true_y}  "
            f"p_max={p:.3f}  epistemic_std={ep:.3f}  entropy={ent:.3f}"
        )

        # Optional: per-sample bar plot of class probs + std (if you already have this helper)
        CLASS_NAMES = [str(k) for k in range(C)]
        plot_mc_bars_for_sample(
            mean_probs,
            std_probs,
            sample_idx=i,
            class_names=CLASS_NAMES,
        )

        # Show the image with title
        title = f"Pred: {cls} (p={p:.2f})  True: {true_y}\n" \
                f"Epistemic std={ep:.3f}, Entropy={ent:.3f}"
        show_image_with_title(x[i], title=title)

    # --- Global summary over the batch ---
    print("\nBatch-level uncertainty summary:")
    print(f"  mean(max_prob)        = {float(max_prob.mean().item()):.3f}")
    print(f"  mean(epistemic_std)   = {float(epistemic_std_pred.mean().item()):.3f}")
    print(f"  mean(entropy)         = {float(entropy.mean().item()):.3f}")

    # --- Posterior (per-layer) summary + save to disk ---
    print("\n=== [7] POSTERIOR SNAPSHOT (PER LAYER) ===")
    #layer_stats = summarise_layer_posterior_scales()
    #for layer_name, s in layer_stats.items():
    #    print(
    #        f"{layer_name:40s}  "
    #        f"num={s['num_params']:7d}  "
    #        f"mean_sigma={s['mean_sigma']:.4f}  "
    #        f"max_sigma={s['max_sigma']:.4f}  "
    #        f"mean_var={s['mean_var']:.6f}  "
    #        f"max_var={s['max_var']:.6f}"
    #    )

    os.makedirs("bayz_artifacts", exist_ok=True)

    # Save full Pyro posterior (param store) for future analysis
    posterior_path = os.path.join("bayz_artifacts", "posterior_params.pt")
    pyro.get_param_store().save(posterior_path)
    print(f"\n[BayzFlow] Saved full posterior params to: {posterior_path}")

    # Save layer stats as JSON (for BayzBoard / offline plots / LinkedIn story)
    #layer_stats_path = os.path.join("bayz_artifacts", "posterior_layer_stats.json")
    #with open(layer_stats_path, "w") as f:
    #    json.dump(layer_stats, f, indent=2)
    #print(f"[BayzFlow] Saved layer-level posterior stats to: {layer_stats_path}")



if __name__ == "__main__":
    main()
