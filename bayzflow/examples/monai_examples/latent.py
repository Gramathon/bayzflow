from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import pyro
import pyro.poutine

from bayzflow import BayzFlow
from bayzflow.models import BayesianSwinUNETR
from bayzflow.examples.monai_examples import monai_dataset
from sklearn.decomposition import PCA

torch.set_grad_enabled(False)



# ------------------------------
# 0. Utility: build val loader
# ------------------------------
def build_val_loader():
    device = torch.device("cpu")

    dm = monai_dataset.MonaiDataset(
        data_dir="data/medical/decathlon",   # <-- adjust to your path
        task="Task09_Spleen",                # or your chosen task
        section="training",                  # splits into train/val via train_frac/val_frac
        download=False,
        spatial_size=(96, 96, 96),
        cache_rate=1.0,
        num_workers=4,
        train_frac=0.8,
        val_frac=0.1,
        device=device,
    )

    val_loader = dm.val_loader()
    return val_loader


# ----------------------------------------
# 1. Latent extraction + burden (Exp A)
# ----------------------------------------
def extract_latents_3d_with_burden(bf, loaders, split="val", max_batches=None):
    """
    Run SwinUNETR on a split, hook a deep latent layer, and return:
        - 3D latent points (PCA)
        - IDs
        - lesion/tumour burden (voxels per case)
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
    all_burden = []
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
            y = batch.get("label", None)

            _ = model(x)

            if "z" not in latent_buf:
                raise RuntimeError(
                    "Forward hook did not capture any latent output. "
                    "Check that swinViT is used in forward()."
                )

            z = latent_buf["z"].cpu()
            bsz = z.shape[0]

            # ----- IDs -----
            if isinstance(batch, dict) and "image_meta_dict" in batch:
                meta = batch["image_meta_dict"]
                filenames = meta["filename_or_obj"]
                ids_this_batch = [str(fn) for fn in filenames]
            else:
                ids_this_batch = [f"{split}_{running_idx + i}" for i in range(bsz)]

            running_idx += bsz

            # ----- lesion / tumour burden -----
            if y is not None:
                # y shape can be [B, 1, D, H, W] or [B, D, H, W]
                y_t = y
                if y_t.ndim == 5:  # [B, C, ...]
                    # assume foreground is any non-zero voxel in all channels
                    y_bin = (y_t > 0).float()
                    # collapse channel dim
                    y_bin = y_bin.sum(dim=1)
                else:  # [B, D, H, W]
                    y_bin = (y_t > 0).float()

                # burden per case = number of positive voxels
                burden_batch = y_bin.view(bsz, -1).sum(dim=1).cpu().numpy()
            else:
                burden_batch = np.zeros(bsz, dtype=np.float32)

            all_latents.append(z.numpy())
            all_ids.extend(ids_this_batch)
            all_burden.extend(list(burden_batch))

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    handle.remove()

    all_latents = np.concatenate(all_latents, axis=0)  # [N, D_latent]
    all_ids = np.array(all_ids)
    all_burden = np.array(all_burden, dtype=np.float32)

    # PCA → 3D
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(all_latents)

    return latents_3d, all_ids, all_burden


def plot_latent_cloud_burden(latents_3d, burden, title="Latent space coloured by lesion volume"):
    """
    3D scatter where colour encodes lesion/tumour burden per case.
    """
    # simple normalisation for colour scale
    if np.max(burden) > 0:
        burden_norm = burden / (np.max(burden) + 1e-6)
    else:
        burden_norm = burden

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        latents_3d[:, 0],
        latents_3d[:, 1],
        latents_3d[:, 2],
        c=burden_norm,
        s=30,
        alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Normalised lesion volume")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# 2. Simple segmentation loss (for guide)
# ----------------------------------------
def seg_loss_fn(model, batch):
    """
    Simple BCEWithLogits loss for 3D segmentation.
    Used only to rebuild the Pyro guide structure; we do NOT re-train.
    """
    x = batch["image"]
    y = batch["label"]

    logits = model(x)  # [B, 1, D, H, W] or [B, C, ...]

    # make shapes consistent
    if y.ndim == 4:  # [B, D, H, W]
        y = y.unsqueeze(1)

    y = y.float()
    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss

def extract_two_patients_latents(bf, loader, patient_indices, max_batches=20):
    """
    Extract latent vectors for TWO chosen patients from the same hook layer.

    Args:
        bf: BayzFlow engine (with bayes_model loaded)
        loader: DataLoader containing "image" (+ optionally meta)
        patient_indices: list/tuple of length 2 with GLOBAL dataset indices, e.g. [0, 5]
        max_batches: how many batches to iterate over at most

    Returns:
        feats3d: np.ndarray of shape [N, 3] – PCA projection for plotting
        labels: np.ndarray of shape [N] with values 0 (patient A) or 1 (patient B)
    """
    device = torch.device("cpu")

    # Pick the actual model
    if getattr(bf, "bayes_model", None) is not None:
        model = bf.bayes_model.to(device)
    else:
        model = bf.base_model.to(device)

    model.eval()

    # Unpack target indices
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 5]")
    target1, target2 = patient_indices

    feats = []
    labels = []
    latent_buf = {}

    # Access Swin backbone hook
    backbone = getattr(model, "model", model)
    swin = backbone.swinViT

    def hook(module, inp, out):
        # out might be a tuple/list – take the deepest feature
        t = out[-1] if isinstance(out, (list, tuple)) else out
        latent_buf["z"] = t.detach().flatten(1)  # [B, D]

    handle = swin.register_forward_hook(hook)

    found1 = False
    found2 = False
    global_idx = 0  # sample index across all batches

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch["image"].to(device)

            # Forward pass → fills latent_buf["z"]
            _ = model(x)
            z = latent_buf["z"].cpu()  # [B, D]

            batch_size = z.shape[0]

            for i in range(batch_size):
                if global_idx == target1:
                    feats.append(z[i].numpy())
                    labels.append(0)
                    found1 = True

                if global_idx == target2:
                    feats.append(z[i].numpy())
                    labels.append(1)
                    found2 = True

                global_idx += 1

                if found1 and found2:
                    break

            if found1 and found2:
                break

            if max_batches and (b_idx + 1) >= max_batches:
                break

    handle.remove()

    feats = np.array(feats)
    labels = np.array(labels, dtype=np.int64)

    if feats.shape[0] == 0:
        raise RuntimeError(
            f"No features collected for patient_indices={patient_indices}. "
            "Check that your indices are within range."
        )

    # ----------------------
    # Robust PCA to 3D
    # ----------------------
    n_samples, n_features = feats.shape
    # number of meaningful components we can compute
    k = min(3, n_samples, n_features)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    proj = pca.fit_transform(feats)  # [N, k]

    if k < 3:
        # Pad to 3 dims for plotting
        pad = np.zeros((n_samples, 3 - k), dtype=proj.dtype)
        feats3d = np.concatenate([proj, pad], axis=1)
    else:
        feats3d = proj

    return feats3d, labels


def plot_two_patient_cloud(feats3d, labels, patient_ids):
    """
    Plot red vs blue latent clouds.
    labels: 0 = patient A, 1 = patient B
    """

    patientA, patientB = patient_ids

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # plot patient A in red
    ax.scatter(
        feats3d[labels == 0, 0],
        feats3d[labels == 0, 1],
        feats3d[labels == 0, 2],
        c="red",
        s=40,
        label=f"Patient A: {patientA}",
        alpha=0.7,
    )

    # plot patient B in blue
    ax.scatter(
        feats3d[labels == 1, 0],
        feats3d[labels == 1, 1],
        feats3d[labels == 1, 2],
        c="blue",
        s=40,
        label=f"Patient B: {patientB}",
        alpha=0.7,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Comparison of Latent Clouds (Two Patients)")

    ax.legend()
    plt.tight_layout()
    plt.show()



# ----------------------------------------
# 3. Experiment B – per-patient belief cloud
# ----------------------------------------
def sample_posterior_latent_cloud_for_case(
    bf,
    batch,
    num_samples: int = 100,
    device: str = "cpu",
):
    """
    For a single case (batch with B=1), sample `num_samples` posterior weight sets
    and collect latent vectors from the *deepest Bayesianised module*.

    Returns:
        latents_3d : [num_samples, 3] PCA projection of the posterior latent cloud
    """
    device = torch.device(device)

    if bf.bayes_model is None:
        raise RuntimeError("BayzFlow.bayes_model is None. Did you call build_bayesian_model()?")

    if bf.guide is None:
        raise RuntimeError("BayzFlow.guide is None. Did you call setup_svi(...) before?")

    model = bf.bayes_model.to(device)
    model.eval()

    # Move batch to device, assume dict with "image" (and maybe "label")
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    # We expect a single case (B=1) for a clean patient-level cloud
    if batch["image"].shape[0] != 1:
        raise ValueError(
            f"Expected batch['image'] with B=1, got shape {batch['image'].shape}"
        )

    # -------------------------------------------------------
    # 1) Find the deepest Bayesianised module to hook
    # -------------------------------------------------------
    name_to_module = dict(model.named_modules())

    # Filter selected_module_names to those actually present
    bayes_names = [n for n in bf.selected_module_names if n in name_to_module]
    if not bayes_names:
        raise RuntimeError(
            "No selected_module_names from checkpoint are present in bayes_model. "
            "Did the architecture change?"
        )

    # Heuristic: deepest = most dots in name
    deepest_name = max(bayes_names, key=lambda s: s.count("."))
    target_module = name_to_module[deepest_name]

    print(f"[BayzFlow] Posterior cloud hook attached to Bayesian module: '{deepest_name}'")

    latents = []

    def hook(module, inp, out):
        """
        Capture the output of the deepest Bayesianised layer for this forward pass.
        out: [B, C, ...] or list/tuple thereof.
        We flatten spatial dims -> [B, D_latent] and store for B=1.
        """
        if isinstance(out, (list, tuple)):
            t = out[-1]
        else:
            t = out

        if not torch.is_tensor(t):
            return

        z = t.detach().flatten(1)      # [B, D_latent]
        latents.append(z.cpu())        # append [1, D_latent]

    handle = target_module.register_forward_hook(hook)

    # -------------------------------------------------------
    # 2) Sample posterior weight sets and run forward passes
    # -------------------------------------------------------
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample global latents / weights from autoguide
            latent_sample = bf.guide(batch)  # uses loaded param_store

            # Condition model on this sample
            lifted_model = pyro.poutine.condition(model, data=latent_sample)

            # Forward pass; hook will capture the latent for this sample
            _ = lifted_model(batch["image"])

    handle.remove()

    if not latents:
        raise RuntimeError("No latents were captured. Hook may not be firing.")

    # latents: list of [1, D] -> [num_samples, D]
    Z = torch.cat(latents, dim=0).numpy()   # [S, D_latent]

    # -------------------------------------------------------
    # 3) PCA → 3D for the posterior cloud
    # -------------------------------------------------------
    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(Z)       # [S, 3]

    return latents_3d


def plot_single_case_cloud(latents_3d, title="Posterior latent cloud (single patient)"):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        latents_3d[:, 0],
        latents_3d[:, 1],
        latents_3d[:, 2],
        s=20,
        alpha=0.8,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def sample_posterior_latents_for_single_case(
    bf,
    batch,
    num_samples: int = 100,
    device: str = "cpu",
):
    """
    For a single case (batch with B=1), sample `num_samples` posterior weight sets
    and collect latent vectors from the deepest Bayesianised module.

    Returns:
        Z : [num_samples, D_latent] – raw latent vectors (no PCA)
    """
    device = torch.device(device)

    if bf.bayes_model is None:
        raise RuntimeError("BayzFlow.bayes_model is None. Did you call build_bayesian_model()?")

    if bf.guide is None:
        raise RuntimeError("BayzFlow.guide is None. Did you call setup_svi(...) before?")

    model = bf.bayes_model.to(device)
    model.eval()

    # Move batch to device, assume dict with "image" (and maybe "label")
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    # We expect a single case (B=1) for a clean patient-level cloud
    if batch["image"].shape[0] != 1:
        raise ValueError(
            f"Expected batch['image'] with B=1, got shape {batch['image'].shape}"
        )

    # -------------------------------------------------------
    # 1) Find the deepest Bayesianised module to hook
    # -------------------------------------------------------
    name_to_module = dict(model.named_modules())

    bayes_names = [n for n in bf.selected_module_names if n in name_to_module]
    if not bayes_names:
        raise RuntimeError(
            "No selected_module_names from checkpoint are present in bayes_model. "
            "Did the architecture change?"
        )

    # Heuristic: deepest = most dots in name
    deepest_name = max(bayes_names, key=lambda s: s.count("."))
    target_module = name_to_module[deepest_name]

    print(f"[BayzFlow] Posterior cloud hook attached to Bayesian module: '{deepest_name}'")

    latents = []

    def hook(module, inp, out):
        """
        Capture the output of the deepest Bayesianised layer for this forward pass.
        out: [B, C, ...] or list/tuple thereof.
        We flatten spatial dims -> [B, D_latent] and store for B=1.
        """
        if isinstance(out, (list, tuple)):
            t = out[-1]
        else:
            t = out

        if not torch.is_tensor(t):
            return

        z = t.detach().flatten(1)      # [B, D_latent]
        latents.append(z.cpu())        # append [1, D_latent]

    handle = target_module.register_forward_hook(hook)

    # -------------------------------------------------------
    # 2) Sample posterior weight sets and run forward passes
    # -------------------------------------------------------
    with torch.no_grad():
        for _ in range(num_samples):
            latent_sample = bf.guide(batch)  # uses loaded param_store
            lifted_model = pyro.poutine.condition(model, data=latent_sample)
            _ = lifted_model(batch["image"])

    handle.remove()

    if not latents:
        raise RuntimeError("No latents were captured. Hook may not be firing.")

    # latents: list of [1, D] -> [num_samples, D]
    Z = torch.cat(latents, dim=0).numpy()   # [S, D_latent]
    return Z

def extract_two_patients_posterior_clouds(
    bf,
    loader,
    patient_indices,
    num_samples_per_patient: int = 200,
    max_batches: int | None = None,
    device: str = "cpu",
):
    """
    Build two dense posterior latent clouds for two patients/studies,
    and project them into a shared 3D PCA space.

    Args:
        bf: BayzFlow engine (with bayes_model + guide built)
        loader: DataLoader yielding dicts with "image" and "label"
        patient_indices: [idx_a, idx_b] – GLOBAL dataset indices (order of loader)
        num_samples_per_patient: MC samples per patient
        max_batches: optional cap on batches to scan
        device: "cpu" or "cuda"

    Returns:
        feats3d: [N, 3] – PCA-projected latents for both patients
        labels: [N] – 0 for patient A, 1 for patient B
    """
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 5]")
    idx_a, idx_b = patient_indices

    device = torch.device(device)

    # 1) Find the two cases in the loader by global index
    batch_a = None
    batch_b = None
    global_idx = 0

    for b_idx, batch in enumerate(loader):
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == idx_a:
                ba = {"image": imgs[i:i+1]}
                if labels is not None:
                    ba["label"] = labels[i:i+1]
                batch_a = ba

            if global_idx == idx_b:
                bb = {"image": imgs[i:i+1]}
                if labels is not None:
                    bb["label"] = labels[i:i+1]
                batch_b = bb

            global_idx += 1

            if batch_a is not None and batch_b is not None:
                break

        if batch_a is not None and batch_b is not None:
            break

        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    if batch_a is None or batch_b is None:
        raise RuntimeError(
            f"Could not find both patient indices {patient_indices} "
            f"within max_batches={max_batches}."
        )

    # 2) Sample posterior latents for each patient separately
    Z_a = sample_posterior_latents_for_single_case(
        bf,
        batch_a,
        num_samples=num_samples_per_patient,
        device=device.type,
    )  # [Sa, D]

    Z_b = sample_posterior_latents_for_single_case(
        bf,
        batch_b,
        num_samples=num_samples_per_patient,
        device=device.type,
    )  # [Sb, D]

    # 3) Joint PCA → shared 3D space
    Z_all = np.vstack([Z_a, Z_b])   # [Sa+Sb, D]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_all = pca.fit_transform(Z_all)  # [Sa+Sb, 3]

    Sa = Z_a.shape[0]
    X_a = X_all[:Sa]
    X_b = X_all[Sa:]

    feats3d = np.vstack([X_a, X_b])
    labels = np.concatenate([
        np.zeros(Sa, dtype=np.int64),
        np.ones(X_b.shape[0], dtype=np.int64),
    ])

    return feats3d, labels

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from sklearn.decomposition import PCA

def compute_layer_trajectory_for_case(
    bf,
    batch,
    module_names=None,
    device="cpu",
):
    """
    For a single case (B=1), capture the flattened activations from a list of
    modules in forward order -> a 'trajectory' through depth.

    Returns:
        traj: np.ndarray of shape [L, D_pad]  (L = #layers, D_pad = max layer dim)
        ordered_module_names: list[str] of length L
    """
    device = torch.device(device)

    # choose model
    if getattr(bf, "bayes_model", None) is not None:
        model = bf.bayes_model.to(device)
    else:
        model = bf.base_model.to(device)
    model.eval()

    # move batch to device, expect {"image": ..., "label": ...}
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    if batch["image"].shape[0] != 1:
        raise ValueError(f"Expect B=1 for clean trajectory, got {batch['image'].shape}")

    # build name -> module and index maps
    name_to_module = dict(model.named_modules())
    name_to_idx = {name: idx for idx, (name, _) in enumerate(model.named_modules())}

    # default: all selected Bayesian modules, in forward order
    if module_names is None:
        if not bf.selected_module_names:
            raise RuntimeError("No bf.selected_module_names; build_bayesian_model() first?")
        module_names = [n for n in bf.selected_module_names if n in name_to_module]

    if not module_names:
        raise RuntimeError("No valid module_names to hook.")

    ordered_module_names = sorted(
        module_names,
        key=lambda n: name_to_idx.get(n, 10_000),
    )

    # prepare buffers + hooks
    latent_buf = {name: None for name in ordered_module_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out[-1] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t):
                # flatten all non-batch dims → [1, D_layer]
                latent_buf[name] = t.detach().flatten(1).cpu()
        return hook

    for name in ordered_module_names:
        m = name_to_module[name]
        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    # forward pass
    with torch.no_grad():
        _ = model(batch["image"])

    # remove hooks
    for h in handles:
        h.remove()

    # collect in order, as 1D numpy arrays of possibly different lengths
    latents_raw = []
    final_names = []
    for name in ordered_module_names:
        z = latent_buf[name]
        if z is None:
            continue
        latents_raw.append(z.numpy()[0])   # shape [D_layer]
        final_names.append(name)

    if not latents_raw:
        raise RuntimeError("No layer activations captured – check module_names selection.")

    # ---- NEW BIT: pad to common dimension ----
    dims = [v.shape[0] for v in latents_raw]
    max_d = max(dims)

    traj = np.zeros((len(latents_raw), max_d), dtype=latents_raw[0].dtype)
    for i, v in enumerate(latents_raw):
        d = v.shape[0]
        traj[i, :d] = v  # zero-pad the rest

    return traj, final_names


def extract_layer_trajectories_two_patients(
    bf,
    loader,
    patient_indices,
    module_names=None,
    device="cpu",
    max_batches=None,
):
    """
    Find two cases by GLOBAL dataset index, compute their layer-wise trajectories,
    and project everything into a shared 3D PCA space.

    Returns:
        trajA_3d: [L_A, 3]
        trajB_3d: [L_B, 3]
        layer_names: list[str] (for the common set we used)
    """
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 2]")
    idx_a, idx_b = patient_indices

    n_cases = len(loader.dataset)
    for idx in patient_indices:
        if idx < 0 or idx >= n_cases:
            raise ValueError(
                f"patient index {idx} out of range for val set of size {n_cases}"
            )

    device = torch.device(device)

    # 1) find the two single-case batches
    batch_a = None
    batch_b = None
    global_idx = 0

    for b_idx, batch in enumerate(loader):
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == idx_a:
                ba = {"image": imgs[i:i+1]}
                if labels is not None:
                    ba["label"] = labels[i:i+1]
                batch_a = ba

            if global_idx == idx_b:
                bb = {"image": imgs[i:i+1]}
                if labels is not None:
                    bb["label"] = labels[i:i+1]
                batch_b = bb

            global_idx += 1

            if batch_a is not None and batch_b is not None:
                break

        if batch_a is not None and batch_b is not None:
            break

        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    if batch_a is None or batch_b is None:
        raise RuntimeError(
            f"Could not find both patient indices {patient_indices} within max_batches={max_batches}."
        )

    # 2) compute layer-wise latents for each
    trajA, layer_namesA = compute_layer_trajectory_for_case(
        bf, batch_a, module_names=module_names, device=device.type
    )
    trajB, layer_namesB = compute_layer_trajectory_for_case(
        bf, batch_b, module_names=module_names, device=device.type
    )

    # we’ll only use the common prefix of layer names (usually identical)
    L = min(trajA.shape[0], trajB.shape[0])
    trajA = trajA[:L]
    trajB = trajB[:L]
    layer_names = layer_namesA[:L]

    # 3) joint PCA to 3D
    Z_all = np.vstack([trajA, trajB])  # [2L, D]
    pca = PCA(n_components=3)
    X_all = pca.fit_transform(Z_all)   # [2L, 3]

    trajA_3d = X_all[:L]
    trajB_3d = X_all[L:]

    return trajA_3d, trajB_3d, layer_names

def plot_layer_trajectories_3d(trajA_3d, trajB_3d, layer_names, patient_indices):
    """
    Plot two layer-wise trajectories as 3D polylines with markers.
    """
    idx_a, idx_b = patient_indices

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Patient A – red line with markers
    ax.plot(
        trajA_3d[:, 0],
        trajA_3d[:, 1],
        trajA_3d[:, 2],
        "-o",
        c="red",
        label=f"Patient A: {idx_a}",
        alpha=0.9,
    )

    # Patient B – blue line with markers
    ax.plot(
        trajB_3d[:, 0],
        trajB_3d[:, 1],
        trajB_3d[:, 2],
        "-o",
        c="blue",
        label=f"Patient B: {idx_b}",
        alpha=0.9,
    )

    # Optionally annotate a few layers
    for i, name in enumerate(layer_names):
        if i in (0, len(layer_names) - 1):  # first & last only to avoid clutter
            ax.text(
                trajA_3d[i, 0],
                trajA_3d[i, 1],
                trajA_3d[i, 2],
                f"{i}",
                color="red",
                fontsize=8,
            )
            ax.text(
                trajB_3d[i, 0],
                trajB_3d[i, 1],
                trajB_3d[i, 2],
                f"{i}",
                color="blue",
                fontsize=8,
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Layer-wise Latent Trajectories (Two Patients)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def sample_layer_trajectories_posterior_case(
    bf,
    batch,
    num_samples: int = 50,
    module_names=None,
    device: str = "cpu",
):
    """
    For a single case (B=1), sample `num_samples` posterior weight sets and
    capture layer-wise latents each time.

    Returns:
        traj_samples: np.ndarray of shape [S, L, D_pad]
            S = num_samples, L = #layers, D_pad = max latent dim across layers
        layer_names: list[str] of length L
    """
    device = torch.device(device)

    if bf.bayes_model is None:
        raise RuntimeError("BayzFlow.bayes_model is None. Call build_bayesian_model() first.")
    if bf.guide is None:
        raise RuntimeError("BayzFlow.guide is None. Call setup_svi(...) first.")

    model = bf.bayes_model.to(device)
    model.eval()

    # move batch to device, expect {"image": ..., "label": ...}
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    if batch["image"].shape[0] != 1:
        raise ValueError(f"Expected B=1, got {batch['image'].shape}")

    # --- module selection (same as before) ---
    name_to_module = dict(model.named_modules())
    name_to_idx = {name: idx for idx, (name, _) in enumerate(model.named_modules())}

    if module_names is None:
        if not bf.selected_module_names:
            raise RuntimeError("No bf.selected_module_names; build_bayesian_model() first?")
        module_names = [n for n in bf.selected_module_names if n in name_to_module]

    if not module_names:
        raise RuntimeError("No valid module_names to hook.")

    ordered_module_names = sorted(
        module_names,
        key=lambda n: name_to_idx.get(n, 10_000),
    )

    # --- prepare hooks ---
    latent_buf = {name: None for name in ordered_module_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out[-1] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t):
                latent_buf[name] = t.detach().flatten(1).cpu()  # [1, D_layer]
        return hook

    for name in ordered_module_names:
        m = name_to_module[name]
        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    all_samples_raw = []
    base_layer_names = None

    with torch.no_grad():
        for s in range(num_samples):
            # reset buffer
            for k in latent_buf.keys():
                latent_buf[k] = None

            # sample weights from guide and condition the model
            latent_sample = bf.guide(batch)  # uses param_store from checkpoint
            lifted_model = pyro.poutine.condition(model, data=latent_sample)

            _ = lifted_model(batch["image"])

            # collect this sample's layer latents
            sample_latents = []
            sample_layer_names = []
            for name in ordered_module_names:
                z = latent_buf[name]
                if z is None:
                    continue
                sample_latents.append(z.numpy()[0])  # [D_layer]
                sample_layer_names.append(name)

            if not sample_latents:
                raise RuntimeError("No layer activations captured; check hooks.")

            if base_layer_names is None:
                base_layer_names = sample_layer_names
            else:
                # enforce consistent layer set/order
                if sample_layer_names != base_layer_names:
                    raise RuntimeError(
                        "Layer name mismatch across posterior samples. "
                        "For now, we expect consistent module outputs."
                    )

            all_samples_raw.append(sample_latents)

    # remove hooks
    for h in handles:
        h.remove()

    # all_samples_raw: list of length S, each is list of length L with arrays [D_layer]
    S = len(all_samples_raw)
    L = len(base_layer_names)

    # compute padding dim
    dims = [v.shape[0] for sample in all_samples_raw for v in sample]
    max_d = max(dims)

    traj_samples = np.zeros((S, L, max_d), dtype=all_samples_raw[0][0].dtype)
    for s in range(S):
        for l in range(L):
            v = all_samples_raw[s][l]
            d = v.shape[0]
            traj_samples[s, l, :d] = v

    return traj_samples, base_layer_names

def extract_layer_trajectories_two_patients_posterior(
    bf,
    loader,
    patient_indices,
    num_samples_per_patient: int = 50,
    module_names=None,
    device: str = "cpu",
    max_batches=None,
):
    """
    For two patients, sample posterior layer-wise trajectories and project into
    a shared 3D PCA space.

    Returns:
        XA_3d: [S_a, L, 3]
        XB_3d: [S_b, L, 3]
        layer_names: list[str] of length L
    """
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 2]")
    idx_a, idx_b = patient_indices

    n_cases = len(loader.dataset)
    for idx in patient_indices:
        if idx < 0 or idx >= n_cases:
            raise ValueError(
                f"patient index {idx} out of range for val set of size {n_cases}"
            )

    device = torch.device(device)

    # --- find two single-case batches by global index ---
    batch_a = None
    batch_b = None
    global_idx = 0

    for b_idx, batch in enumerate(loader):
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == idx_a:
                ba = {"image": imgs[i:i+1]}
                if labels is not None:
                    ba["label"] = labels[i:i+1]
                batch_a = ba

            if global_idx == idx_b:
                bb = {"image": imgs[i:i+1]}
                if labels is not None:
                    bb["label"] = labels[i:i+1]
                batch_b = bb

            global_idx += 1

            if batch_a is not None and batch_b is not None:
                break

        if batch_a is not None and batch_b is not None:
            break

        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    if batch_a is None or batch_b is None:
        raise RuntimeError(
            f"Could not find both patient indices {patient_indices} within max_batches={max_batches}."
        )

    # --- posterior trajectories for each patient ---
    trajA_samples, layer_namesA = sample_layer_trajectories_posterior_case(
        bf,
        batch_a,
        num_samples=num_samples_per_patient,
        module_names=module_names,
        device=device.type,
    )  # [S_a, L, D]

    trajB_samples, layer_namesB = sample_layer_trajectories_posterior_case(
        bf,
        batch_b,
        num_samples=num_samples_per_patient,
        module_names=module_names,
        device=device.type,
    )  # [S_b, L, D]

    # use common prefix if needed
    if layer_namesA != layer_namesB:
        L = min(len(layer_namesA), len(layer_namesB))
        layer_names = layer_namesA[:L]
        trajA_samples = trajA_samples[:, :L, :]
        trajB_samples = trajB_samples[:, :L, :]
    else:
        layer_names = layer_namesA
        L = len(layer_names)

    S_a, _, D = trajA_samples.shape
    S_b, _, _ = trajB_samples.shape

    # --- joint PCA to 3D ---
    Z_all = np.concatenate(
        [
            trajA_samples.reshape(S_a * L, D),
            trajB_samples.reshape(S_b * L, D),
        ],
        axis=0,
    )  # [S_a*L + S_b*L, D]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_all = pca.fit_transform(Z_all)  # [total, 3]

    XA = X_all[: S_a * L].reshape(S_a, L, 3)
    XB = X_all[S_a * L :].reshape(S_b, L, 3)

    return XA, XB, layer_names

def plot_layer_trajectories_posterior(
    XA_3d,
    XB_3d,
    layer_names,
    patient_indices,
    max_trajs_to_show: int = 20,
):
    """
    XA_3d: [S_a, L, 3]
    XB_3d: [S_b, L, 3]
    """
    idx_a, idx_b = patient_indices
    S_a, L, _ = XA_3d.shape
    S_b, _, _ = XB_3d.shape

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # limit clutter
    Sa_vis = min(S_a, max_trajs_to_show)
    Sb_vis = min(S_b, max_trajs_to_show)

    # posterior samples (faint)
    for s in range(Sa_vis):
        ax.plot(
            XA_3d[s, :, 0],
            XA_3d[s, :, 1],
            XA_3d[s, :, 2],
            "-", c="red", alpha=0.15, linewidth=1.0,
        )

    for s in range(Sb_vis):
        ax.plot(
            XB_3d[s, :, 0],
            XB_3d[s, :, 1],
            XB_3d[s, :, 2],
            "-", c="blue", alpha=0.15, linewidth=1.0,
        )

    # mean trajectories (bold)
    meanA = XA_3d.mean(axis=0)
    meanB = XB_3d.mean(axis=0)

    ax.plot(
        meanA[:, 0],
        meanA[:, 1],
        meanA[:, 2],
        "-o",
        c="red",
        label=f"Patient A mean: {idx_a}",
        linewidth=3.0,
        alpha=0.9,
    )

    ax.plot(
        meanB[:, 0],
        meanB[:, 1],
        meanB[:, 2],
        "-o",
        c="blue",
        label=f"Patient B mean: {idx_b}",
        linewidth=3.0,
        alpha=0.9,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Posterior Layer-wise Latent Trajectories (Two Patients)")
    ax.legend()
    plt.tight_layout()
    plt.show()


from pathlib import Path
import gc

def dump_layer_latents_posterior_case(
    bf,
    batch,
    out_dir: str,
    num_samples: int = 20,
    module_names=None,
    device: str = "cpu",
):

    """
    For a single case (B=1), sample posterior weights and save each
    layer latent as a separate .npy file on disk.

    Directory layout:
      out_dir/
        sample_000_layer_00.npy
        sample_000_layer_01.npy
        ...
        sample_001_layer_00.npy
        ...

    No big arrays are kept in RAM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    if bf.bayes_model is None:
        raise RuntimeError("Need bf.bayes_model")
    if bf.guide is None:
        raise RuntimeError("Need bf.guide (call setup_svi)")

    model = bf.bayes_model.to(device)
    model.eval()

    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    assert batch["image"].shape[0] == 1

    name_to_module = dict(model.named_modules())
    name_to_idx = {name: idx for idx, (name, _) in enumerate(model.named_modules())}

    if module_names is None:
        if not bf.selected_module_names:
            raise RuntimeError("No bf.selected_module_names")
        module_names = [n for n in bf.selected_module_names if n in name_to_module]

    ordered_module_names = sorted(
        module_names, key=lambda n: name_to_idx.get(n, 10_000)
    )

    latent_buf = {name: None for name in ordered_module_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out[-1] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t):
                t = t.detach()
                if t.ndim > 2:
                    # global average pool to keep dim small
                    t = t.mean(dim=(2, 3, 4))
                latent_buf[name] = t.cpu()
        return hook

    for name in ordered_module_names:
        h = name_to_module[name].register_forward_hook(make_hook(name))
        handles.append(h)

    with torch.no_grad():
        for s in range(num_samples):
            # clear buffer
            for k in latent_buf.keys():
                latent_buf[k] = None

            latent_sample = bf.guide(batch)
            lifted_model = pyro.poutine.condition(model, data=latent_sample)
            _ = lifted_model(batch["image"])

            # dump each layer latent immediately
            for l, name in enumerate(ordered_module_names):
                z = latent_buf[name]
                if z is None:
                    continue
                v = z.numpy()[0].astype(np.float32)  # [D_layer]
                np.save(out_dir / f"sample_{s:03d}_layer_{l:02d}.npy", v)

            # free Python references
            gc.collect()

    for h in handles:
        h.remove()

    # also save the ordered module names for later
    np.save(out_dir / "layer_names.npy", np.array(ordered_module_names, dtype=object))

def get_single_case_batch(loader, target_idx: int):
    """
    Scan through loader to find the sample with GLOBAL index target_idx,
    and return a single-case batch dict: {"image": [1,...], "label": [1,...]}.
    """
    global_idx = 0
    for batch in loader:
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == target_idx:
                out = {"image": imgs[i:i+1]}
                if labels is not None:
                    out["label"] = labels[i:i+1]
                return out
            global_idx += 1

    raise RuntimeError(f"Index {target_idx} not found in loader (max global_idx={global_idx-1}).")


from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import pyro
import pyro.poutine

from bayzflow import BayzFlow
from bayzflow.models import BayesianSwinUNETR
from bayzflow.examples.monai_examples import monai_dataset
from sklearn.decomposition import PCA

torch.set_grad_enabled(False)



# ------------------------------
# 0. Utility: build val loader
# ------------------------------
def build_val_loader():
    device = torch.device("cpu")

    dm = monai_dataset.MonaiDataset(
        data_dir="data/medical/decathlon",   # <-- adjust to your path
        task="Task09_Spleen",                # or your chosen task
        section="training",                  # splits into train/val via train_frac/val_frac
        download=False,
        spatial_size=(96, 96, 96),
        cache_rate=1.0,
        num_workers=4,
        train_frac=0.8,
        val_frac=0.1,
        device=device,
    )

    val_loader = dm.val_loader()
    return val_loader


# ----------------------------------------
# 1. Latent extraction + burden (Exp A)
# ----------------------------------------
def extract_latents_3d_with_burden(bf, loaders, split="val", max_batches=None):
    """
    Run SwinUNETR on a split, hook a deep latent layer, and return:
        - 3D latent points (PCA)
        - IDs
        - lesion/tumour burden (voxels per case)
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
    all_burden = []
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
            y = batch.get("label", None)

            _ = model(x)

            if "z" not in latent_buf:
                raise RuntimeError(
                    "Forward hook did not capture any latent output. "
                    "Check that swinViT is used in forward()."
                )

            z = latent_buf["z"].cpu()
            bsz = z.shape[0]

            # ----- IDs -----
            if isinstance(batch, dict) and "image_meta_dict" in batch:
                meta = batch["image_meta_dict"]
                filenames = meta["filename_or_obj"]
                ids_this_batch = [str(fn) for fn in filenames]
            else:
                ids_this_batch = [f"{split}_{running_idx + i}" for i in range(bsz)]

            running_idx += bsz

            # ----- lesion / tumour burden -----
            if y is not None:
                # y shape can be [B, 1, D, H, W] or [B, D, H, W]
                y_t = y
                if y_t.ndim == 5:  # [B, C, ...]
                    # assume foreground is any non-zero voxel in all channels
                    y_bin = (y_t > 0).float()
                    # collapse channel dim
                    y_bin = y_bin.sum(dim=1)
                else:  # [B, D, H, W]
                    y_bin = (y_t > 0).float()

                # burden per case = number of positive voxels
                burden_batch = y_bin.view(bsz, -1).sum(dim=1).cpu().numpy()
            else:
                burden_batch = np.zeros(bsz, dtype=np.float32)

            all_latents.append(z.numpy())
            all_ids.extend(ids_this_batch)
            all_burden.extend(list(burden_batch))

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    handle.remove()

    all_latents = np.concatenate(all_latents, axis=0)  # [N, D_latent]
    all_ids = np.array(all_ids)
    all_burden = np.array(all_burden, dtype=np.float32)

    # PCA → 3D
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(all_latents)

    return latents_3d, all_ids, all_burden


def plot_latent_cloud_burden(latents_3d, burden, title="Latent space coloured by lesion volume"):
    """
    3D scatter where colour encodes lesion/tumour burden per case.
    """
    # simple normalisation for colour scale
    if np.max(burden) > 0:
        burden_norm = burden / (np.max(burden) + 1e-6)
    else:
        burden_norm = burden

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        latents_3d[:, 0],
        latents_3d[:, 1],
        latents_3d[:, 2],
        c=burden_norm,
        s=30,
        alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Normalised lesion volume")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# 2. Simple segmentation loss (for guide)
# ----------------------------------------
def seg_loss_fn(model, batch):
    """
    Simple BCEWithLogits loss for 3D segmentation.
    Used only to rebuild the Pyro guide structure; we do NOT re-train.
    """
    x = batch["image"]
    y = batch["label"]

    logits = model(x)  # [B, 1, D, H, W] or [B, C, ...]

    # make shapes consistent
    if y.ndim == 4:  # [B, D, H, W]
        y = y.unsqueeze(1)

    y = y.float()
    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss

def extract_two_patients_latents(bf, loader, patient_indices, max_batches=20):
    """
    Extract latent vectors for TWO chosen patients from the same hook layer.

    Args:
        bf: BayzFlow engine (with bayes_model loaded)
        loader: DataLoader containing "image" (+ optionally meta)
        patient_indices: list/tuple of length 2 with GLOBAL dataset indices, e.g. [0, 5]
        max_batches: how many batches to iterate over at most

    Returns:
        feats3d: np.ndarray of shape [N, 3] – PCA projection for plotting
        labels: np.ndarray of shape [N] with values 0 (patient A) or 1 (patient B)
    """
    device = torch.device("cpu")

    # Pick the actual model
    if getattr(bf, "bayes_model", None) is not None:
        model = bf.bayes_model.to(device)
    else:
        model = bf.base_model.to(device)

    model.eval()

    # Unpack target indices
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 5]")
    target1, target2 = patient_indices

    feats = []
    labels = []
    latent_buf = {}

    # Access Swin backbone hook
    backbone = getattr(model, "model", model)
    swin = backbone.swinViT

    def hook(module, inp, out):
        # out might be a tuple/list – take the deepest feature
        t = out[-1] if isinstance(out, (list, tuple)) else out
        latent_buf["z"] = t.detach().flatten(1)  # [B, D]

    handle = swin.register_forward_hook(hook)

    found1 = False
    found2 = False
    global_idx = 0  # sample index across all batches

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch["image"].to(device)

            # Forward pass → fills latent_buf["z"]
            _ = model(x)
            z = latent_buf["z"].cpu()  # [B, D]

            batch_size = z.shape[0]

            for i in range(batch_size):
                if global_idx == target1:
                    feats.append(z[i].numpy())
                    labels.append(0)
                    found1 = True

                if global_idx == target2:
                    feats.append(z[i].numpy())
                    labels.append(1)
                    found2 = True

                global_idx += 1

                if found1 and found2:
                    break

            if found1 and found2:
                break

            if max_batches and (b_idx + 1) >= max_batches:
                break

    handle.remove()

    feats = np.array(feats)
    labels = np.array(labels, dtype=np.int64)

    if feats.shape[0] == 0:
        raise RuntimeError(
            f"No features collected for patient_indices={patient_indices}. "
            "Check that your indices are within range."
        )

    # ----------------------
    # Robust PCA to 3D
    # ----------------------
    n_samples, n_features = feats.shape
    # number of meaningful components we can compute
    k = min(3, n_samples, n_features)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    proj = pca.fit_transform(feats)  # [N, k]

    if k < 3:
        # Pad to 3 dims for plotting
        pad = np.zeros((n_samples, 3 - k), dtype=proj.dtype)
        feats3d = np.concatenate([proj, pad], axis=1)
    else:
        feats3d = proj

    return feats3d, labels


def plot_two_patient_cloud(feats3d, labels, patient_ids):
    """
    Plot red vs blue latent clouds.
    labels: 0 = patient A, 1 = patient B
    """

    patientA, patientB = patient_ids

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # plot patient A in red
    ax.scatter(
        feats3d[labels == 0, 0],
        feats3d[labels == 0, 1],
        feats3d[labels == 0, 2],
        c="red",
        s=40,
        label=f"Patient A: {patientA}",
        alpha=0.7,
    )

    # plot patient B in blue
    ax.scatter(
        feats3d[labels == 1, 0],
        feats3d[labels == 1, 1],
        feats3d[labels == 1, 2],
        c="blue",
        s=40,
        label=f"Patient B: {patientB}",
        alpha=0.7,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Comparison of Latent Clouds (Two Patients)")

    ax.legend()
    plt.tight_layout()
    plt.show()



# ----------------------------------------
# 3. Experiment B – per-patient belief cloud
# ----------------------------------------
def sample_posterior_latent_cloud_for_case(
    bf,
    batch,
    num_samples: int = 100,
    device: str = "cpu",
):
    """
    For a single case (batch with B=1), sample `num_samples` posterior weight sets
    and collect latent vectors from the *deepest Bayesianised module*.

    Returns:
        latents_3d : [num_samples, 3] PCA projection of the posterior latent cloud
    """
    device = torch.device(device)

    if bf.bayes_model is None:
        raise RuntimeError("BayzFlow.bayes_model is None. Did you call build_bayesian_model()?")

    if bf.guide is None:
        raise RuntimeError("BayzFlow.guide is None. Did you call setup_svi(...) before?")

    model = bf.bayes_model.to(device)
    model.eval()

    # Move batch to device, assume dict with "image" (and maybe "label")
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    # We expect a single case (B=1) for a clean patient-level cloud
    if batch["image"].shape[0] != 1:
        raise ValueError(
            f"Expected batch['image'] with B=1, got shape {batch['image'].shape}"
        )

    # -------------------------------------------------------
    # 1) Find the deepest Bayesianised module to hook
    # -------------------------------------------------------
    name_to_module = dict(model.named_modules())

    # Filter selected_module_names to those actually present
    bayes_names = [n for n in bf.selected_module_names if n in name_to_module]
    if not bayes_names:
        raise RuntimeError(
            "No selected_module_names from checkpoint are present in bayes_model. "
            "Did the architecture change?"
        )

    # Heuristic: deepest = most dots in name
    deepest_name = max(bayes_names, key=lambda s: s.count("."))
    target_module = name_to_module[deepest_name]

    print(f"[BayzFlow] Posterior cloud hook attached to Bayesian module: '{deepest_name}'")

    latents = []

    def hook(module, inp, out):
        """
        Capture the output of the deepest Bayesianised layer for this forward pass.
        out: [B, C, ...] or list/tuple thereof.
        We flatten spatial dims -> [B, D_latent] and store for B=1.
        """
        if isinstance(out, (list, tuple)):
            t = out[-1]
        else:
            t = out

        if not torch.is_tensor(t):
            return

        z = t.detach().flatten(1)      # [B, D_latent]
        latents.append(z.cpu())        # append [1, D_latent]

    handle = target_module.register_forward_hook(hook)

    # -------------------------------------------------------
    # 2) Sample posterior weight sets and run forward passes
    # -------------------------------------------------------
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample global latents / weights from autoguide
            latent_sample = bf.guide(batch)  # uses loaded param_store

            # Condition model on this sample
            lifted_model = pyro.poutine.condition(model, data=latent_sample)

            # Forward pass; hook will capture the latent for this sample
            _ = lifted_model(batch["image"])

    handle.remove()

    if not latents:
        raise RuntimeError("No latents were captured. Hook may not be firing.")

    # latents: list of [1, D] -> [num_samples, D]
    Z = torch.cat(latents, dim=0).numpy()   # [S, D_latent]

    # -------------------------------------------------------
    # 3) PCA → 3D for the posterior cloud
    # -------------------------------------------------------
    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(Z)       # [S, 3]

    return latents_3d


def plot_single_case_cloud(latents_3d, title="Posterior latent cloud (single patient)"):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        latents_3d[:, 0],
        latents_3d[:, 1],
        latents_3d[:, 2],
        s=20,
        alpha=0.8,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def sample_posterior_latents_for_single_case(
    bf,
    batch,
    num_samples: int = 100,
    device: str = "cpu",
):
    """
    For a single case (batch with B=1), sample `num_samples` posterior weight sets
    and collect latent vectors from the deepest Bayesianised module.

    Returns:
        Z : [num_samples, D_latent] – raw latent vectors (no PCA)
    """
    device = torch.device(device)

    if bf.bayes_model is None:
        raise RuntimeError("BayzFlow.bayes_model is None. Did you call build_bayesian_model()?")

    if bf.guide is None:
        raise RuntimeError("BayzFlow.guide is None. Did you call setup_svi(...) before?")

    model = bf.bayes_model.to(device)
    model.eval()

    # Move batch to device, assume dict with "image" (and maybe "label")
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    # We expect a single case (B=1) for a clean patient-level cloud
    if batch["image"].shape[0] != 1:
        raise ValueError(
            f"Expected batch['image'] with B=1, got shape {batch['image'].shape}"
        )

    # -------------------------------------------------------
    # 1) Find the deepest Bayesianised module to hook
    # -------------------------------------------------------
    name_to_module = dict(model.named_modules())

    bayes_names = [n for n in bf.selected_module_names if n in name_to_module]
    if not bayes_names:
        raise RuntimeError(
            "No selected_module_names from checkpoint are present in bayes_model. "
            "Did the architecture change?"
        )

    # Heuristic: deepest = most dots in name
    deepest_name = max(bayes_names, key=lambda s: s.count("."))
    target_module = name_to_module[deepest_name]

    print(f"[BayzFlow] Posterior cloud hook attached to Bayesian module: '{deepest_name}'")

    latents = []

    def hook(module, inp, out):
        """
        Capture the output of the deepest Bayesianised layer for this forward pass.
        out: [B, C, ...] or list/tuple thereof.
        We flatten spatial dims -> [B, D_latent] and store for B=1.
        """
        if isinstance(out, (list, tuple)):
            t = out[-1]
        else:
            t = out

        if not torch.is_tensor(t):
            return

        z = t.detach().flatten(1)      # [B, D_latent]
        latents.append(z.cpu())        # append [1, D_latent]

    handle = target_module.register_forward_hook(hook)

    # -------------------------------------------------------
    # 2) Sample posterior weight sets and run forward passes
    # -------------------------------------------------------
    with torch.no_grad():
        for _ in range(num_samples):
            latent_sample = bf.guide(batch)  # uses loaded param_store
            lifted_model = pyro.poutine.condition(model, data=latent_sample)
            _ = lifted_model(batch["image"])

    handle.remove()

    if not latents:
        raise RuntimeError("No latents were captured. Hook may not be firing.")

    # latents: list of [1, D] -> [num_samples, D]
    Z = torch.cat(latents, dim=0).numpy()   # [S, D_latent]
    return Z

def extract_two_patients_posterior_clouds(
    bf,
    loader,
    patient_indices,
    num_samples_per_patient: int = 200,
    max_batches: int | None = None,
    device: str = "cpu",
):
    """
    Build two dense posterior latent clouds for two patients/studies,
    and project them into a shared 3D PCA space.

    Args:
        bf: BayzFlow engine (with bayes_model + guide built)
        loader: DataLoader yielding dicts with "image" and "label"
        patient_indices: [idx_a, idx_b] – GLOBAL dataset indices (order of loader)
        num_samples_per_patient: MC samples per patient
        max_batches: optional cap on batches to scan
        device: "cpu" or "cuda"

    Returns:
        feats3d: [N, 3] – PCA-projected latents for both patients
        labels: [N] – 0 for patient A, 1 for patient B
    """
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 5]")
    idx_a, idx_b = patient_indices

    device = torch.device(device)

    # 1) Find the two cases in the loader by global index
    batch_a = None
    batch_b = None
    global_idx = 0

    for b_idx, batch in enumerate(loader):
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == idx_a:
                ba = {"image": imgs[i:i+1]}
                if labels is not None:
                    ba["label"] = labels[i:i+1]
                batch_a = ba

            if global_idx == idx_b:
                bb = {"image": imgs[i:i+1]}
                if labels is not None:
                    bb["label"] = labels[i:i+1]
                batch_b = bb

            global_idx += 1

            if batch_a is not None and batch_b is not None:
                break

        if batch_a is not None and batch_b is not None:
            break

        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    if batch_a is None or batch_b is None:
        raise RuntimeError(
            f"Could not find both patient indices {patient_indices} "
            f"within max_batches={max_batches}."
        )

    # 2) Sample posterior latents for each patient separately
    Z_a = sample_posterior_latents_for_single_case(
        bf,
        batch_a,
        num_samples=num_samples_per_patient,
        device=device.type,
    )  # [Sa, D]

    Z_b = sample_posterior_latents_for_single_case(
        bf,
        batch_b,
        num_samples=num_samples_per_patient,
        device=device.type,
    )  # [Sb, D]

    # 3) Joint PCA → shared 3D space
    Z_all = np.vstack([Z_a, Z_b])   # [Sa+Sb, D]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_all = pca.fit_transform(Z_all)  # [Sa+Sb, 3]

    Sa = Z_a.shape[0]
    X_a = X_all[:Sa]
    X_b = X_all[Sa:]

    feats3d = np.vstack([X_a, X_b])
    labels = np.concatenate([
        np.zeros(Sa, dtype=np.int64),
        np.ones(X_b.shape[0], dtype=np.int64),
    ])

    return feats3d, labels

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from sklearn.decomposition import PCA

def compute_layer_trajectory_for_case(
    bf,
    batch,
    module_names=None,
    device="cpu",
):
    """
    For a single case (B=1), capture the flattened activations from a list of
    modules in forward order -> a 'trajectory' through depth.

    Returns:
        traj: np.ndarray of shape [L, D_pad]  (L = #layers, D_pad = max layer dim)
        ordered_module_names: list[str] of length L
    """
    device = torch.device(device)

    # choose model
    if getattr(bf, "bayes_model", None) is not None:
        model = bf.bayes_model.to(device)
    else:
        model = bf.base_model.to(device)
    model.eval()

    # move batch to device, expect {"image": ..., "label": ...}
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    if batch["image"].shape[0] != 1:
        raise ValueError(f"Expect B=1 for clean trajectory, got {batch['image'].shape}")

    # build name -> module and index maps
    name_to_module = dict(model.named_modules())
    name_to_idx = {name: idx for idx, (name, _) in enumerate(model.named_modules())}

    # default: all selected Bayesian modules, in forward order
    if module_names is None:
        if not bf.selected_module_names:
            raise RuntimeError("No bf.selected_module_names; build_bayesian_model() first?")
        module_names = [n for n in bf.selected_module_names if n in name_to_module]

    if not module_names:
        raise RuntimeError("No valid module_names to hook.")

    ordered_module_names = sorted(
        module_names,
        key=lambda n: name_to_idx.get(n, 10_000),
    )

    # prepare buffers + hooks
    latent_buf = {name: None for name in ordered_module_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out[-1] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t):
                # flatten all non-batch dims → [1, D_layer]
                latent_buf[name] = t.detach().flatten(1).cpu()
        return hook

    for name in ordered_module_names:
        m = name_to_module[name]
        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    # forward pass
    with torch.no_grad():
        _ = model(batch["image"])

    # remove hooks
    for h in handles:
        h.remove()

    # collect in order, as 1D numpy arrays of possibly different lengths
    latents_raw = []
    final_names = []
    for name in ordered_module_names:
        z = latent_buf[name]
        if z is None:
            continue
        latents_raw.append(z.numpy()[0])   # shape [D_layer]
        final_names.append(name)

    if not latents_raw:
        raise RuntimeError("No layer activations captured – check module_names selection.")

    # ---- NEW BIT: pad to common dimension ----
    dims = [v.shape[0] for v in latents_raw]
    max_d = max(dims)

    traj = np.zeros((len(latents_raw), max_d), dtype=latents_raw[0].dtype)
    for i, v in enumerate(latents_raw):
        d = v.shape[0]
        traj[i, :d] = v  # zero-pad the rest

    return traj, final_names


def extract_layer_trajectories_two_patients(
    bf,
    loader,
    patient_indices,
    module_names=None,
    device="cpu",
    max_batches=None,
):
    """
    Find two cases by GLOBAL dataset index, compute their layer-wise trajectories,
    and project everything into a shared 3D PCA space.

    Returns:
        trajA_3d: [L_A, 3]
        trajB_3d: [L_B, 3]
        layer_names: list[str] (for the common set we used)
    """
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 2]")
    idx_a, idx_b = patient_indices

    n_cases = len(loader.dataset)
    for idx in patient_indices:
        if idx < 0 or idx >= n_cases:
            raise ValueError(
                f"patient index {idx} out of range for val set of size {n_cases}"
            )

    device = torch.device(device)

    # 1) find the two single-case batches
    batch_a = None
    batch_b = None
    global_idx = 0

    for b_idx, batch in enumerate(loader):
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == idx_a:
                ba = {"image": imgs[i:i+1]}
                if labels is not None:
                    ba["label"] = labels[i:i+1]
                batch_a = ba

            if global_idx == idx_b:
                bb = {"image": imgs[i:i+1]}
                if labels is not None:
                    bb["label"] = labels[i:i+1]
                batch_b = bb

            global_idx += 1

            if batch_a is not None and batch_b is not None:
                break

        if batch_a is not None and batch_b is not None:
            break

        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    if batch_a is None or batch_b is None:
        raise RuntimeError(
            f"Could not find both patient indices {patient_indices} within max_batches={max_batches}."
        )

    # 2) compute layer-wise latents for each
    trajA, layer_namesA = compute_layer_trajectory_for_case(
        bf, batch_a, module_names=module_names, device=device.type
    )
    trajB, layer_namesB = compute_layer_trajectory_for_case(
        bf, batch_b, module_names=module_names, device=device.type
    )

    # we’ll only use the common prefix of layer names (usually identical)
    L = min(trajA.shape[0], trajB.shape[0])
    trajA = trajA[:L]
    trajB = trajB[:L]
    layer_names = layer_namesA[:L]

    # 3) joint PCA to 3D
    Z_all = np.vstack([trajA, trajB])  # [2L, D]
    pca = PCA(n_components=3)
    X_all = pca.fit_transform(Z_all)   # [2L, 3]

    trajA_3d = X_all[:L]
    trajB_3d = X_all[L:]

    return trajA_3d, trajB_3d, layer_names

def plot_layer_trajectories_3d(trajA_3d, trajB_3d, layer_names, patient_indices):
    """
    Plot two layer-wise trajectories as 3D polylines with markers.
    """
    idx_a, idx_b = patient_indices

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Patient A – red line with markers
    ax.plot(
        trajA_3d[:, 0],
        trajA_3d[:, 1],
        trajA_3d[:, 2],
        "-o",
        c="red",
        label=f"Patient A: {idx_a}",
        alpha=0.9,
    )

    # Patient B – blue line with markers
    ax.plot(
        trajB_3d[:, 0],
        trajB_3d[:, 1],
        trajB_3d[:, 2],
        "-o",
        c="blue",
        label=f"Patient B: {idx_b}",
        alpha=0.9,
    )

    # Optionally annotate a few layers
    for i, name in enumerate(layer_names):
        if i in (0, len(layer_names) - 1):  # first & last only to avoid clutter
            ax.text(
                trajA_3d[i, 0],
                trajA_3d[i, 1],
                trajA_3d[i, 2],
                f"{i}",
                color="red",
                fontsize=8,
            )
            ax.text(
                trajB_3d[i, 0],
                trajB_3d[i, 1],
                trajB_3d[i, 2],
                f"{i}",
                color="blue",
                fontsize=8,
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Layer-wise Latent Trajectories (Two Patients)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def sample_layer_trajectories_posterior_case(
    bf,
    batch,
    num_samples: int = 50,
    module_names=None,
    device: str = "cpu",
):
    """
    For a single case (B=1), sample `num_samples` posterior weight sets and
    capture layer-wise latents each time.

    Returns:
        traj_samples: np.ndarray of shape [S, L, D_pad]
            S = num_samples, L = #layers, D_pad = max latent dim across layers
        layer_names: list[str] of length L
    """
    device = torch.device(device)

    if bf.bayes_model is None:
        raise RuntimeError("BayzFlow.bayes_model is None. Call build_bayesian_model() first.")
    if bf.guide is None:
        raise RuntimeError("BayzFlow.guide is None. Call setup_svi(...) first.")

    model = bf.bayes_model.to(device)
    model.eval()

    # move batch to device, expect {"image": ..., "label": ...}
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    if batch["image"].shape[0] != 1:
        raise ValueError(f"Expected B=1, got {batch['image'].shape}")

    # --- module selection (same as before) ---
    name_to_module = dict(model.named_modules())
    name_to_idx = {name: idx for idx, (name, _) in enumerate(model.named_modules())}

    if module_names is None:
        if not bf.selected_module_names:
            raise RuntimeError("No bf.selected_module_names; build_bayesian_model() first?")
        module_names = [n for n in bf.selected_module_names if n in name_to_module]

    if not module_names:
        raise RuntimeError("No valid module_names to hook.")

    ordered_module_names = sorted(
        module_names,
        key=lambda n: name_to_idx.get(n, 10_000),
    )

    # --- prepare hooks ---
    latent_buf = {name: None for name in ordered_module_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out[-1] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t):
                latent_buf[name] = t.detach().flatten(1).cpu()  # [1, D_layer]
        return hook

    for name in ordered_module_names:
        m = name_to_module[name]
        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    all_samples_raw = []
    base_layer_names = None

    with torch.no_grad():
        for s in range(num_samples):
            # reset buffer
            for k in latent_buf.keys():
                latent_buf[k] = None

            # sample weights from guide and condition the model
            latent_sample = bf.guide(batch)  # uses param_store from checkpoint
            lifted_model = pyro.poutine.condition(model, data=latent_sample)

            _ = lifted_model(batch["image"])

            # collect this sample's layer latents
            sample_latents = []
            sample_layer_names = []
            for name in ordered_module_names:
                z = latent_buf[name]
                if z is None:
                    continue
                sample_latents.append(z.numpy()[0])  # [D_layer]
                sample_layer_names.append(name)

            if not sample_latents:
                raise RuntimeError("No layer activations captured; check hooks.")

            if base_layer_names is None:
                base_layer_names = sample_layer_names
            else:
                # enforce consistent layer set/order
                if sample_layer_names != base_layer_names:
                    raise RuntimeError(
                        "Layer name mismatch across posterior samples. "
                        "For now, we expect consistent module outputs."
                    )

            all_samples_raw.append(sample_latents)

    # remove hooks
    for h in handles:
        h.remove()

    # all_samples_raw: list of length S, each is list of length L with arrays [D_layer]
    S = len(all_samples_raw)
    L = len(base_layer_names)

    # compute padding dim
    dims = [v.shape[0] for sample in all_samples_raw for v in sample]
    max_d = max(dims)

    traj_samples = np.zeros((S, L, max_d), dtype=all_samples_raw[0][0].dtype)
    for s in range(S):
        for l in range(L):
            v = all_samples_raw[s][l]
            d = v.shape[0]
            traj_samples[s, l, :d] = v

    return traj_samples, base_layer_names

def extract_layer_trajectories_two_patients_posterior(
    bf,
    loader,
    patient_indices,
    num_samples_per_patient: int = 50,
    module_names=None,
    device: str = "cpu",
    max_batches=None,
):
    """
    For two patients, sample posterior layer-wise trajectories and project into
    a shared 3D PCA space.

    Returns:
        XA_3d: [S_a, L, 3]
        XB_3d: [S_b, L, 3]
        layer_names: list[str] of length L
    """
    if len(patient_indices) != 2:
        raise ValueError("patient_indices must be length 2, e.g. [0, 2]")
    idx_a, idx_b = patient_indices

    n_cases = len(loader.dataset)
    for idx in patient_indices:
        if idx < 0 or idx >= n_cases:
            raise ValueError(
                f"patient index {idx} out of range for val set of size {n_cases}"
            )

    device = torch.device(device)

    # --- find two single-case batches by global index ---
    batch_a = None
    batch_b = None
    global_idx = 0

    for b_idx, batch in enumerate(loader):
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == idx_a:
                ba = {"image": imgs[i:i+1]}
                if labels is not None:
                    ba["label"] = labels[i:i+1]
                batch_a = ba

            if global_idx == idx_b:
                bb = {"image": imgs[i:i+1]}
                if labels is not None:
                    bb["label"] = labels[i:i+1]
                batch_b = bb

            global_idx += 1

            if batch_a is not None and batch_b is not None:
                break

        if batch_a is not None and batch_b is not None:
            break

        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    if batch_a is None or batch_b is None:
        raise RuntimeError(
            f"Could not find both patient indices {patient_indices} within max_batches={max_batches}."
        )

    # --- posterior trajectories for each patient ---
    trajA_samples, layer_namesA = sample_layer_trajectories_posterior_case(
        bf,
        batch_a,
        num_samples=num_samples_per_patient,
        module_names=module_names,
        device=device.type,
    )  # [S_a, L, D]

    trajB_samples, layer_namesB = sample_layer_trajectories_posterior_case(
        bf,
        batch_b,
        num_samples=num_samples_per_patient,
        module_names=module_names,
        device=device.type,
    )  # [S_b, L, D]

    # use common prefix if needed
    if layer_namesA != layer_namesB:
        L = min(len(layer_namesA), len(layer_namesB))
        layer_names = layer_namesA[:L]
        trajA_samples = trajA_samples[:, :L, :]
        trajB_samples = trajB_samples[:, :L, :]
    else:
        layer_names = layer_namesA
        L = len(layer_names)

    S_a, _, D = trajA_samples.shape
    S_b, _, _ = trajB_samples.shape

    # --- joint PCA to 3D ---
    Z_all = np.concatenate(
        [
            trajA_samples.reshape(S_a * L, D),
            trajB_samples.reshape(S_b * L, D),
        ],
        axis=0,
    )  # [S_a*L + S_b*L, D]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_all = pca.fit_transform(Z_all)  # [total, 3]

    XA = X_all[: S_a * L].reshape(S_a, L, 3)
    XB = X_all[S_a * L :].reshape(S_b, L, 3)

    return XA, XB, layer_names

def plot_layer_trajectories_posterior(
    XA_3d,
    XB_3d,
    layer_names,
    patient_indices,
    max_trajs_to_show: int = 20,
):
    """
    XA_3d: [S_a, L, 3]
    XB_3d: [S_b, L, 3]
    """
    idx_a, idx_b = patient_indices
    S_a, L, _ = XA_3d.shape
    S_b, _, _ = XB_3d.shape

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # limit clutter
    Sa_vis = min(S_a, max_trajs_to_show)
    Sb_vis = min(S_b, max_trajs_to_show)

    # posterior samples (faint)
    for s in range(Sa_vis):
        ax.plot(
            XA_3d[s, :, 0],
            XA_3d[s, :, 1],
            XA_3d[s, :, 2],
            "-", c="red", alpha=0.15, linewidth=1.0,
        )

    for s in range(Sb_vis):
        ax.plot(
            XB_3d[s, :, 0],
            XB_3d[s, :, 1],
            XB_3d[s, :, 2],
            "-", c="blue", alpha=0.15, linewidth=1.0,
        )

    # mean trajectories (bold)
    meanA = XA_3d.mean(axis=0)
    meanB = XB_3d.mean(axis=0)

    ax.plot(
        meanA[:, 0],
        meanA[:, 1],
        meanA[:, 2],
        "-o",
        c="red",
        label=f"Patient A mean: {idx_a}",
        linewidth=3.0,
        alpha=0.9,
    )

    ax.plot(
        meanB[:, 0],
        meanB[:, 1],
        meanB[:, 2],
        "-o",
        c="blue",
        label=f"Patient B mean: {idx_b}",
        linewidth=3.0,
        alpha=0.9,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Posterior Layer-wise Latent Trajectories (Two Patients)")
    ax.legend()
    plt.tight_layout()
    plt.show()


from pathlib import Path
import gc

def dump_layer_latents_posterior_case(
    bf,
    batch,
    out_dir: str,
    num_samples: int = 20,
    module_names=None,
    device: str = "cpu",
):

    """
    For a single case (B=1), sample posterior weights and save each
    layer latent as a separate .npy file on disk.

    Directory layout:
      out_dir/
        sample_000_layer_00.npy
        sample_000_layer_01.npy
        ...
        sample_001_layer_00.npy
        ...

    No big arrays are kept in RAM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    if bf.bayes_model is None:
        raise RuntimeError("Need bf.bayes_model")
    if bf.guide is None:
        raise RuntimeError("Need bf.guide (call setup_svi)")

    model = bf.bayes_model.to(device)
    model.eval()

    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    assert batch["image"].shape[0] == 1

    name_to_module = dict(model.named_modules())
    name_to_idx = {name: idx for idx, (name, _) in enumerate(model.named_modules())}

    if module_names is None:
        if not bf.selected_module_names:
            raise RuntimeError("No bf.selected_module_names")
        module_names = [n for n in bf.selected_module_names if n in name_to_module]

    ordered_module_names = sorted(
        module_names, key=lambda n: name_to_idx.get(n, 10_000)
    )

    latent_buf = {name: None for name in ordered_module_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out[-1] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t):
                t = t.detach()
                if t.ndim > 2:
                    # global average pool to keep dim small
                    t = t.mean(dim=(2, 3, 4))
                latent_buf[name] = t.cpu()
        return hook

    for name in ordered_module_names:
        h = name_to_module[name].register_forward_hook(make_hook(name))
        handles.append(h)

    with torch.no_grad():
        for s in range(num_samples):
            # clear buffer
            for k in latent_buf.keys():
                latent_buf[k] = None

            latent_sample = bf.guide(batch)
            lifted_model = pyro.poutine.condition(model, data=latent_sample)
            _ = lifted_model(batch["image"])

            # dump each layer latent immediately
            for l, name in enumerate(ordered_module_names):
                z = latent_buf[name]
                if z is None:
                    continue
                v = z.numpy()[0].astype(np.float32)  # [D_layer]
                np.save(out_dir / f"sample_{s:03d}_layer_{l:02d}.npy", v)

            # free Python references
            gc.collect()

    for h in handles:
        h.remove()

    # also save the ordered module names for later
    np.save(out_dir / "layer_names.npy", np.array(ordered_module_names, dtype=object))

def get_single_case_batch(loader, target_idx: int):
    """
    Scan through loader to find the sample with GLOBAL index target_idx,
    and return a single-case batch dict: {"image": [1,...], "label": [1,...]}.
    """
    global_idx = 0
    for batch in loader:
        imgs = batch["image"]
        labels = batch.get("label", None)
        B = imgs.shape[0]

        for i in range(B):
            if global_idx == target_idx:
                out = {"image": imgs[i:i+1]}
                if labels is not None:
                    out["label"] = labels[i:i+1]
                return out
            global_idx += 1

    raise RuntimeError(f"Index {target_idx} not found in loader (max global_idx={global_idx-1}).")


def save_case_volume(batch, out_path: str | Path):
    """
    Save a single-case 3D image volume from a MONAI batch dict
    as a (D, H, W) numpy array.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = batch["image"]  # shape [1, C, D, H, W]
    if img.ndim != 5:
        raise ValueError(f"Expected image with shape [1, C, D, H, W], got {img.shape}")

    vol = img[0, 0].cpu().numpy()  # take first channel -> (D, H, W)
    np.save(out_path, vol)
    print(f"[INFO] Saved volume to {out_path} with shape {vol.shape}")
def compute_layer_belief_map(model, batch, layer_name, device="cpu"):
    """
    Computes ∂ ||latent|| / ∂ image for a given Bayesian layer.
    Returns a saliency volume shaped (D, H, W).
    """
    device = torch.device(device)

    model.eval()
    img = batch["image"].to(device)
    img.requires_grad_(True)

    # Find layer
    name_to_module = dict(model.named_modules())
    target = name_to_module[layer_name]

    latent_out = {}

    def hook(module, inp, out):
        t = out[-1] if isinstance(out, (list, tuple)) else out
        latent_out["z"] = t.flatten(1)     # [1, D]
    h = target.register_forward_hook(hook)

    # Forward
    out = model(img)

    h.remove()

    z = latent_out["z"]     # [1,D]
    # scalar belief objective: norm of latent activation
    scalar = z.norm()

    scalar.backward()

    sal = img.grad.detach().abs()[0,0]   # [D,H,W]
    sal = sal.cpu().numpy()

    # normalise
    sal = sal / (sal.max() + 1e-6)

    return sal


# ------------------------------
# 4. Inference entrypoint
# ------------------------------
def main():
    device = torch.device("cpu")

    # 1) Rebuild the SwinUNETR exactly as in training
    model = BayesianSwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,     # must match training
        out_channels=1,    # must match training
        feature_size=48,
        pretrained=False,  # weights will come from checkpoint
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
    if not bf.selected_module_names:
        raise RuntimeError(
            "Checkpoint did not contain selected_module_names; "
            "cannot rebuild Bayesian model."
        )
    bf.build_bayesian_model()   # uses bf.prior_scales if present

    # 5) Build val loader
    val_loader = build_val_loader()

    # -----------------------------
    # Experiment B:
    #   Posterior latent cloud + belief maps for two patients
    # -----------------------------

    # Rebuild Pyro model + guide structure using the same loss
    bf.setup_svi(loss_fn=seg_loss_fn, lr=1e-3)

    # Choose two global dataset indices
    patient_indices = [0, 2]  # global dataset indices for the two patients/studies

    batch_0 = get_single_case_batch(val_loader, patient_indices[0])
    batch_2 = get_single_case_batch(val_loader, patient_indices[1])

    # Save anatomy volumes for VTK
    save_case_volume(batch_0, "volumes/patient_0_volume.npy")
    save_case_volume(batch_2, "volumes/patient_2_volume.npy")

    # optional: choose a subset of layers (e.g. last 5 Bayesian swinViT layers)
    all_bayes = [n for n in bf.selected_module_names if "swinViT" in n]
    all_bayes = sorted(all_bayes)
    module_names = all_bayes[-5:] if all_bayes else None

    # Dump posterior layer latents for each patient (for your VTK latent gimbal)
    dump_layer_latents_posterior_case(
        bf,
        batch_0,
        out_dir="latent_dump/patient_0",
        num_samples=200,
        module_names=module_names,  # or None for all
        device="cpu",
    )

    dump_layer_latents_posterior_case(
        bf,
        batch_2,
        out_dir="latent_dump/patient_2",
        num_samples=200,
        module_names=module_names,  # or None for all
        device="cpu",
    )

    # -----------------------------
    # Belief maps for one layer
    # -----------------------------
    if not all_bayes:
        raise RuntimeError("No Bayesian swinViT layers found in bf.selected_module_names")

    # pick a layer to explain – deepest swinViT Bayesian layer
    layer_name = all_bayes[-1]
    print(f"[INFO] Computing belief maps for layer: {layer_name}")

    belief_0 = compute_layer_belief_map(
        bf.bayes_model,
        batch_0,
        layer_name=layer_name,
        device="cpu",
    )
    belief_2 = compute_layer_belief_map(
        bf.bayes_model,
        batch_2,
        layer_name=layer_name,
        device="cpu",
    )

    Path("volumes").mkdir(parents=True, exist_ok=True)
    np.save("volumes/patient_0_belief.npy", belief_0)
    np.save("volumes/patient_2_belief.npy", belief_2)

    print("[INFO] Saved belief maps:",
          belief_0.shape, belief_0.min(), belief_0.max(),
          "and",
          belief_2.shape, belief_2.min(), belief_2.max())


if __name__ == "__main__":
    main()
