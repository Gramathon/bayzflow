# bayzflow/core/experiment.py

import importlib
from typing import Any, Dict, Tuple

import torch
# import torch.nn.functional as F  # Not needed currently
import yaml

from .engine import BayzFlow


def _resolve_class(relative_path: str):
    """
    'models.blstm.BayesianLSTM' -> bayzflow.models.blstm.BayesianLSTM
    """
    full = f"bayzflow.{relative_path}"
    module_path, cls_name = full.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def exp(config_path: str, exp: str | None = None) -> Tuple[BayzFlow, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    High-level, multi-experiment loader.

    - config_path: path to YAML file
    - experiment:  which experiment key to run from bayzflow.experiments
                   (if None, uses bayzflow.default_experiment)

    Returns:
      bf      : trained BayzFlow engine
      loaders : {'train': ..., 'calib': ..., 'test': ...}
      cfg     : full parsed YAML dict
      extras  : any extra domain-specific outputs from data builder
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = cfg["bayzflow"]

    # 1) pick experiment name -----------------------------------
    exp_name = exp or root.get("default_experiment")
    if exp_name is None:
        raise ValueError("No experiment specified and no default_experiment in YAML.")

    if exp_name not in root["experiments"]:
        raise KeyError(f"Experiment '{exp_name}' not found in bayzflow.experiments")

    exp_cfg = root["experiments"][exp_name]
    kind = exp_cfg.get("kind", None)

    # 2) build model --------------------------------------------
    model_cfg = exp_cfg["model"]
    model_cls = _resolve_class(model_cfg["module"])
    model_kwargs = {
        k: v for k, v in model_cfg.items()
        if k not in ("module", "type")
    }
    data_cfg = exp_cfg["data"]
    train_cfg = exp_cfg["training"]

    # For MONAI models, pass spatial_size from data config if available
    if kind == "monai_segmentation":
        if "spatial_size" in data_cfg:
            # For BasicUNet, use spatial_size parameter
            if model_cfg.get("type") == "monai_unet_3d":
                model_kwargs["spatial_size"] = data_cfg["spatial_size"]
            # For SwinUNETR, use img_size parameter (if not already specified)
            elif model_cfg.get("type") == "swin_unetr" and "img_size" not in model_kwargs:
                model_kwargs["img_size"] = data_cfg["spatial_size"]

    model = model_cls(**model_kwargs)

    # 3) build data + loaders -----------------------------------
    #data_cfg = exp_cfg["data"]
    #train_cfg = exp_cfg["training"]
    # dataset_cls = _resolve_class(data_cfg["module"])  # Not used directly

    # Create dataset and loaders based on experiment kind
    if kind == "fx":
        from bayzflow.examples.fx.fx_utils.fx_dataloader import FXDataset

        # Extract data config parameters
        dataset_kwargs = {
            k: v for k, v in data_cfg.items()
            if k not in ("module",)
        }

        # Create FX dataset
        fx_dataset = FXDataset(**dataset_kwargs)

        # Create loaders
        batch_size = train_cfg.get("batch_size", 32)
        train_loader = fx_dataset.train_loader(batch_size=batch_size, shuffle=True)
        val_loader = fx_dataset.val_loader(batch_size=batch_size, shuffle=False) if hasattr(fx_dataset, 'val_loader') else None
        calib_loader = fx_dataset.train_loader(batch_size=batch_size, shuffle=False)  # Use training data for calibration

        # Store dataset for later use
        dataset = fx_dataset
        
    elif kind == "monai_segmentation":
        try:
            # Create MONAI dataset and loaders
            from bayzflow.examples.monai_examples.monai_dataset import MonaiDataset

            # Get device configuration
            device_str = exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_str)

            # Extract data config parameters
            dataset_kwargs = {
                k: v for k, v in data_cfg.items()
                if k not in ("module",)
            }
            dataset_kwargs['device'] = device

            # Create MONAI dataset
            monai_dataset = MonaiDataset(**dataset_kwargs)

            # Create loaders
            batch_size = train_cfg.get("batch_size", 2)
            train_loader = monai_dataset.train_loader(batch_size=batch_size, shuffle=True)
            val_loader = monai_dataset.val_loader(batch_size=batch_size, shuffle=False) if hasattr(monai_dataset, 'val_loader') else None
            calib_loader = monai_dataset.train_loader(batch_size=1, shuffle=False)

            # Store dataset for later use
            dataset = monai_dataset
            
        except ImportError as e:
            raise ImportError(f"MONAI not available for monai_segmentation experiments. Install MONAI: pip install monai. Error: {e}")
        
    else:
        raise NotImplementedError(f"Dataset creation for kind='{kind}' not implemented yet")

    # 4) training config & functions -----------------------------
    # train_cfg already defined above

    # Define forward and loss functions based on experiment kind
    if kind == "fx":
        def calib_forward_fn(model, batch):
            """Forward function for calibration - just run the model."""
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch[0], batch[1]  # TensorDataset returns tuple
            return model(x, y)
        
        def loss_fn(model, batch):
            """Loss function for FX training."""
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch[0], batch[1]  # TensorDataset returns tuple
            
            # BayesianLSTM returns predictions and handles pyro.sample internally
            model(x, y)  # y is passed for pyro.sample("obs", ..., obs=y)
            
            # The model already handles the probabilistic loss via pyro.sample
            # We return a dummy loss since the actual loss is handled by Pyro
            return torch.tensor(0.0, requires_grad=True)
    
    elif kind == "monai_segmentation":
        # Get device configuration early for use in functions
        device_str_monai = exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device_monai = torch.device(device_str_monai)

        def calib_forward_fn(model, batch):
            """Forward function for MONAI segmentation calibration."""
            if isinstance(batch, dict):
                x = batch["image"]
            else:
                x = batch[0]  # Assume first element is image
            return model(x)

        def loss_fn(model, batch):
            """Loss function for MONAI segmentation training."""
            import torch.nn.functional as F

            if isinstance(batch, dict):
                x, y = batch["image"], batch["label"]
            else:
                x, y = batch[0], batch[1]

            # Get model predictions
            logits = model(x)
            
            # Compute segmentation loss (cross-entropy or Dice)
            if y.dtype == torch.long:
                # Multi-class segmentation
                # cross_entropy expects target shape (N, d1, d2, ...) without channel dim
                # If labels have shape (N, 1, d1, d2, ...), squeeze the channel dimension
                if y.ndim == logits.ndim and y.shape[1] == 1:
                    y = y.squeeze(1)
                loss = F.cross_entropy(logits, y)
            else:
                # Binary segmentation with sigmoid + BCE
                loss = F.binary_cross_entropy_with_logits(logits, y.float())
            
            return loss
    
    else:
        raise NotImplementedError(f"Functions for kind='{kind}' not implemented yet")

    

    # Extract training parameters
    bayes_patterns      = train_cfg.get("bayes_patterns", ["head"])  # Default to head layers
    num_calib_passes    = train_cfg.get("calib_passes", 3)
    calib_percentile    = train_cfg.get("calib_percentile", 0.9)
    default_prior_scale = train_cfg.get("default_prior_scale", 0.1)
    train_epochs        = train_cfg.get("epochs", 10)
    lr                  = float(train_cfg.get("lr", 1e-3))
    patience            = train_cfg.get("patience", 10)

    device_str = exp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Get checkpoint configuration
    checkpoint_cfg = exp_cfg.get("checkpoints", {})
    checkpoint_dir = checkpoint_cfg.get("outdir", "checkpoints")
    checkpoint_name = checkpoint_cfg.get("best_name", "best_model.pt")
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}" if checkpoint_cfg else None

    # 5) run BayzFlow workflow with checkpointing --------------
    # Use from_model_and_data for setup, then fit with checkpoint support
    bf = BayzFlow.from_model_and_data(
        model=model,
        calib_loader=calib_loader,
        calib_forward_fn=calib_forward_fn,
        bayes_patterns=bayes_patterns,
        num_calib_passes=num_calib_passes,
        calib_percentile=calib_percentile,
        default_prior_scale=default_prior_scale,
        device=device,
        verbose=True,
    )

    # Setup SVI and train with checkpoint support
    print("\n[BayzFlow] Starting SVI training...")
    bf.setup_svi(loss_fn=loss_fn, lr=lr)
    bf.fit(
        train_loader=train_loader,
        loss_fn=loss_fn,
        num_epochs=train_epochs,
        device=device,
        val_loader=val_loader,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )

    # Prepare loaders dict for return
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "calib": calib_loader,
        "test": getattr(dataset, 'test_loader', lambda: None)(batch_size) if hasattr(dataset, 'test_loader') else None
    }
    
    # Get posterior configuration
    posterior_cfg = exp_cfg.get("posterior", {})
    save_latents = posterior_cfg.get("save_latents", False)
    save_path = posterior_cfg.get("save_path", None)
    
    # Prepare extras dict with posterior configuration
    extras = {
        "dataset": dataset,
        "device": device,
        "posterior_cfg": {
            "save_latents": save_latents,
            "save_path": save_path,
            "num_samples": posterior_cfg.get("num_samples", 16)
        }
    }

    return bf, loaders, cfg, extras
