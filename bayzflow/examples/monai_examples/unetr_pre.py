from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from bayzflow import BayzFlow


def extract_latents_3d(bf, loaders, split="val", max_batches=None):
    """
    Run the trained model on a split ("train"/"val"/"test"),
    hook a latent layer, and return 3D embeddings + IDs.
    """
    # ---- FORCE CPU ----
    device = torch.device("cpu")

    # --- pick a model from BayzFlow ---
    if hasattr(bf, "bayes_model") and bf.bayes_model is not None:
        model = bf.bayes_model.to(device)
    elif hasattr(bf, "base_model") and bf.base_model is not None:
        model = bf.base_model.to(device)
    else:
        raise AttributeError(
            "BayzFlow object has neither 'bayes_model' nor 'base_model'. "
            "Cannot extract latents."
        )

    model.eval()

    # --- choose which loader ---
    if split == "train":
        loader = loaders["train"]
    elif split == "val":
        loader = loaders["val"]
    else:
        loader = loaders.get("test") or loaders["val"]

    # --- buffer for latents + IDs ---
    latent_buf = {}
    all_latents = []
    all_ids = []
    running_idx = 0  # for synthetic IDs if no meta is present

    # --- locate a robust hook point for SwinUNETR ---
    backbone = getattr(model, "model", model)
    swin_module = backbone.swinViT

    def hook(module, inp, out):
        """
        out is often a list/tuple of multi-scale feature maps.
        We take the last one (deepest) and flatten it.
        """
        if isinstance(out, (list, tuple)):
            t = out[-1]  # deepest feature
        else:
            t = out      # already a tensor

        # t: [B, C, D, H, W] -> [B, -1]
        latent_buf["z"] = t.detach().flatten(1)

    handle = swin_module.register_forward_hook(hook)

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch["image"].to(device)

            # forward pass (on CPU)
            _ = model(x)

            if "z" not in latent_buf:
                raise RuntimeError(
                    "Forward hook did not capture any latent output. "
                    "Check that swinViT exists and is being used in forward()."
                )

            z = latent_buf["z"].cpu()  # [B, D_latent]
            bsz = z.shape[0]

            # ---- Try to get IDs from meta if available ----
            if isinstance(batch, dict) and "image_meta_dict" in batch:
                meta = batch["image_meta_dict"]
                filenames = meta["filename_or_obj"]
                ids_this_batch = [str(fn) for fn in filenames]
            else:
                # fallback: simple synthetic IDs
                ids_this_batch = [f"{split}_{running_idx + i}" for i in range(bsz)]

            running_idx += bsz

            all_latents.append(z.numpy())
            all_ids.extend(ids_this_batch)

            if (max_batches is not None) and (b_idx + 1 >= max_batches):
                break

    handle.remove()

    all_latents = np.concatenate(all_latents, axis=0)  # [N, D_latent]
    all_ids = np.array(all_ids)

    # --- reduce to 3D with PCA for visualization ---
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(all_latents)  # [N, 3]

    return latents_3d, all_ids


def plot_latent_cloud(latents_3d, ids, title="Latent space (PCA-3D)"):
    """
    Simple 3D scatter; colour by patient/file ID (collapsed to an index).
    """
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


def main():
    try:
        bf, loaders, cfg, extras = BayzFlow.exp(
            "bayzflow/bayzflow.yaml",
            exp="monai_swin_unetr_pretrained",
        )

        # --- AFTER training: extract & visualise latents ---
        lat3d, ids = extract_latents_3d(bf, loaders, split="val", max_batches=100)
        plot_latent_cloud(lat3d, ids, title="BayzFlow SwinUNETR latent space (val set)")

    except ImportError as e:
        print(f"[Main] MONAI not available: {e}")
        print("[Main] Install MONAI with: pip install monai scikit-learn")
        print("[Main] Skipping MONAI example...")

    except Exception as e:
        print(f"[Main] Error running MONAI experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
