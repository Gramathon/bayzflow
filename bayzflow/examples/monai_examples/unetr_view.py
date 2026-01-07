from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from bayzflow import BayzFlow
from bayzflow.models import BayesianSwinUNETR
from bayzflow.examples.monai_examples import monai_dataset


# ------------------------------
# 1. Latent extraction utilities
# ------------------------------
def extract_latents_3d(bf, loaders, split="val", max_batches=None):
    """
    Run the (Bayesian) SwinUNETR on a split ("train"/"val"/"test"),
    hook a deep latent layer, and return 3D embeddings + IDs.
    """
    device = torch.device("cpu")

    # Prefer Bayesian model if present
    if bf.bayes_model is not None:
        model = bf.bayes_model.to(device)
    else:
        model = bf.base_model.to(device)

    model.eval()

    # pick loader
    if split == "train":
        loader = loaders["train"]
    elif split == "val":
        loader = loaders["val"]
    else:
        loader = loaders.get("test") or loaders["val"]

    latent_buf = {}
    all_latents = []
    all_ids = []
    running_idx = 0

    # Swin backbone
    backbone = getattr(model, "model", model)
    swin_module = backbone.swinViT

    def hook(module, inp, out):
        # out can be list/tuple of multi-scale features; take deepest
        if isinstance(out, (list, tuple)):
            t = out[-1]
        else:
            t = out
        latent_buf["z"] = t.detach().flatten(1)  # [B, -1]

    handle = swin_module.register_forward_hook(hook)

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch["image"].to(device)

            _ = model(x)

            if "z" not in latent_buf:
                raise RuntimeError(
                    "Forward hook did not capture any latent output. "
                    "Check that swinViT is used in forward()."
                )

            z = latent_buf["z"].cpu()
            bsz = z.shape[0]

            # IDs – use MONAI meta if present, otherwise synthetic
            if isinstance(batch, dict) and "image_meta_dict" in batch:
                meta = batch["image_meta_dict"]
                filenames = meta["filename_or_obj"]
                ids_this_batch = [str(fn) for fn in filenames]
            else:
                ids_this_batch = [f"{split}_{running_idx + i}" for i in range(bsz)]

            running_idx += bsz

            all_latents.append(z.numpy())
            all_ids.extend(ids_this_batch)

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    handle.remove()

    all_latents = np.concatenate(all_latents, axis=0)  # [N, D_latent]
    all_ids = np.array(all_ids)

    # PCA → 3D
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(all_latents)

    return latents_3d, all_ids


def plot_latent_cloud(latents_3d, ids, title="Latent space (PCA-3D)"):
    unique_ids, inv = np.unique(ids, return_inverse=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        latents_3d[:, 0],
        latents_3d[:, 1],
        latents_3d[:, 2],
        c=inv,
        s=30,
        alpha=0.8,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def build_val_loader():
    device = torch.device("cpu")  # or "cuda" if you ever want GPU

    dm = monai_dataset.MonaiDataset(
        data_dir="data/medical/decathlon",  # <-- change this
        task="Task01_BrainTumour",                    # default is fine
        section="training",                           # your class will split into train/val via train_frac/val_frac
        download=False,
        spatial_size=(96, 96, 96),
        cache_rate=1.0,
        num_workers=4,
        train_frac=0.8,
        val_frac=0.1,
        device=device,
    )


    val_loader = dm.train_loader()   
    return val_loader

# ------------------------------
# 2. Inference entrypoint
# ------------------------------
def main():
    device = torch.device("cpu")

    # 1) Rebuild the SwinUNETR exactly as in training
    model = BayesianSwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1,          # whatever you trained with
        feature_size=48,
        pretrained=True,        # weights will come from checkpoint
        freeze_encoder=False,
        device=device,
    )

    # 2) Wrap in BayzFlow
    bf = BayzFlow(model)

    # 3) Load checkpoint: restores base_model weights, param_store,
    #    selected_module_names, prior_scales
    bf.load_checkpoint(
        path="checkpoints_monai_swin/swin_unetr_best.pt",
        map_location=device,
    )

    # 4) Build Bayesian model using restored selection + prior scales
    #    (this attaches BayesConv/BayesLinear/etc. wrappers)
    if not bf.selected_module_names:
        raise RuntimeError(
            "Checkpoint did not contain selected_module_names; "
            "cannot rebuild Bayesian model."
        )
    bf.build_bayesian_model()   # uses bf.prior_scales if present

    
    val_loader = build_val_loader()

    loaders = {"val": val_loader}

    # 6) Extract and plot full latent cloud
    lat3d, ids = extract_latents_3d(bf, loaders, split="val", max_batches=None)
    plot_latent_cloud(lat3d, ids, title="Bayesian SwinUNETR latent space (new data)")


if __name__ == "__main__":
    main()
