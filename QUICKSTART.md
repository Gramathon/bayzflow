# BayzFlow Quick Start Guide

Get up and running with BayzFlow in minutes!

## Installation

```bash
# Navigate to the repository
cd bayzflow

# Install with all dependencies
pip install -e ".[all]"

# Or install only what you need
pip install -e ".[medical,viz]"  # For medical imaging examples
```

## 1. Basic Usage - High-Level API

```python
from bayzflow import BayzFlow
import torch
from torch.utils.data import DataLoader

# Your PyTorch model
model = YourModel()

# Your data loaders (must return dict batches: {"x": tensor, "y": tensor})
train_loader = DataLoader(train_dataset, batch_size=32)
calib_loader = DataLoader(calib_dataset, batch_size=32)

# Define forward and loss functions
def calib_forward_fn(model, batch):
    return model(batch["x"])

def loss_fn(model, batch):
    logits = model(batch["x"])
    return F.cross_entropy(logits, batch["y"])

# Create Bayesian model in one line!
bf = BayzFlow.auto(
    model=model,
    calib_loader=calib_loader,
    train_loader=train_loader,
    calib_forward_fn=calib_forward_fn,
    loss_fn=loss_fn,
    bayes_patterns=["head"],  # Which layers to make Bayesian
    train_epochs=10
)

# Predict with uncertainty
def predict_fn(model, batch):
    return model(batch["x"])

predictions = bf.predict(
    batch=test_batch,
    predict_fn=predict_fn,
    num_samples=16,
    return_samples=True
)

print(f"Mean prediction: {predictions['mean']}")
print(f"Uncertainty (std): {predictions['std']}")
```

## 2. Using Experiment Configs (Recommended for MONAI)

### Step 1: Edit Configuration

Edit `bayzflow/bayzflow.yaml`:

```yaml
bayzflow:
  default_experiment: "monai_swin_unetr_pretrained"  # Change this

  experiments:
    monai_swin_unetr_pretrained:
      # ... experiment configuration
```

### Step 2: Run Experiment

```python
from bayzflow import BayzFlow

# Load and run experiment
bf, loaders, cfg, extras = BayzFlow.exp(
    "bayzflow/bayzflow.yaml",
    exp="monai_swin_unetr_pretrained"
)

# Model is already trained!
# Use it for predictions
test_batch = next(iter(loaders['test']))

predictions = bf.predict(
    batch=test_batch,
    predict_fn=lambda m, b: m(b["image"]),
    num_samples=16,
    device=extras['device']
)
```

## 3. Medical Imaging with MONAI

### BasicUNet Example

```bash
# Make sure MONAI is installed
pip install -e ".[medical]"

# Run the example
python bayzflow/examples/monai_examples/monai_unet_train.py
```

### SwinUNETR with Pretrained Encoder

```bash
# Edit bayzflow.yaml first:
# Set default_experiment: "monai_swin_unetr_pretrained"

# Optional: Download pretrained weights and set path in config
# pretrained_path: "/path/to/swin_unetr_pretrained.pt"

# Run the example
python bayzflow/examples/monai_examples/monai_unet_train.py
```

Key features of SwinUNETR experiment:
- ✓ Pretrained Swin Transformer encoder
- ✓ Encoder weights frozen (SVI-only training)
- ✓ Only decoder/head layers are Bayesian
- ✓ Efficient training (no encoder backprop)
- ✓ Automatic checkpointing with early stopping

## 4. Working with Checkpoints

### Training with Checkpoints

```python
# Through experiment config (automatic)
# checkpoints:
#   outdir: "checkpoints"
#   best_name: "model_best.pt"

# Or manually in code
bf.fit(
    train_loader=train_loader,
    loss_fn=loss_fn,
    num_epochs=20,
    val_loader=val_loader,  # Optional for early stopping
    patience=8,
    checkpoint_path="checkpoints/my_model.pt"
)
```

### Loading Checkpoints

```python
from bayzflow import BayzFlow

# Create model
model = YourModel()
bf = BayzFlow(model)

# Load checkpoint
metadata = bf.load_checkpoint("checkpoints/my_model.pt")
print(f"Loaded from epoch {metadata['epoch']}")
print(f"Validation loss: {metadata['val_loss']}")

# Use for inference
predictions = bf.predict(...)
```

## 5. Customizing Spatial Sizes

For medical imaging, you can easily change input spatial sizes:

```yaml
data:
  spatial_size: [128, 128, 128]  # Change from [96, 96, 96]
  # ... rest of data config

model:
  # For SwinUNETR, img_size will auto-match spatial_size
  # Or explicitly set: img_size: [128, 128, 128]
```

The model will automatically:
- Validate spatial size compatibility
- Warn if size may be too small
- Adapt to the specified dimensions

## 6. Manual Workflow (Advanced)

For fine-grained control:

```python
from bayzflow import BayzFlow

# 1. Create engine
bf = BayzFlow(model)

# 2. Inspect model structure
bf.show_module_tree()

# 3. Find candidate layers
candidates = bf.list_candidates()

# 4. Select layers to make Bayesian
bf.select_modules(patterns=["decoder", "head"])
# Or by index: bf.select_modules(select=[3, 5, 7])

# 5. Calibrate priors using your data
bf.calibrate_priors(
    calib_loader=calib_loader,
    calib_forward_fn=lambda m, b: m(b["x"]),
    num_passes=3,
    percentile=0.9
)

# 6. Build Bayesian model
bf.build_bayesian_model(default_prior_scale=0.1)

# 7. Setup SVI training
bf.setup_svi(loss_fn=loss_fn, lr=1e-4)

# 8. Train with checkpointing
bf.fit(
    train_loader=train_loader,
    loss_fn=loss_fn,
    num_epochs=20,
    device=torch.device("cuda"),
    val_loader=val_loader,
    patience=8,
    checkpoint_path="checkpoint.pt"
)

# 9. Predict with uncertainty
predictions = bf.predict(
    batch=test_batch,
    predict_fn=predict_fn,
    num_samples=16
)
```

## Common Patterns

### Pattern Selection Examples

```python
# Make only the final output layer Bayesian
bf.select_modules(patterns=["head"])

# Make decoder and output layers Bayesian (recommended for U-Nets)
bf.select_modules(patterns=["decoder", "head"])

# Make specific module types Bayesian
bf.select_modules(patterns=["conv", "linear"])

# Combine patterns
bf.select_modules(patterns=["decoder.conv", "head.linear"])
```

### Loss Functions

```python
# Classification
def loss_fn(model, batch):
    logits = model(batch["x"])
    return F.cross_entropy(logits, batch["y"])

# Segmentation
def loss_fn(model, batch):
    logits = model(batch["image"])
    targets = batch["label"]
    return F.cross_entropy(logits, targets)

# Regression
def loss_fn(model, batch):
    pred = model(batch["x"])
    return F.mse_loss(pred, batch["y"])
```

## Tips and Best Practices

1. **Start with small models** - Test your workflow on a small model first
2. **Use validation sets** - Enable early stopping to prevent overfitting
3. **Save checkpoints** - Always use checkpointing for long training runs
4. **Monitor uncertainty** - High uncertainty indicates need for more data
5. **Pattern selection** - Start with `["head"]`, expand if needed
6. **Frozen encoders** - Use pretrained frozen encoders when possible (faster training)
7. **Batch size** - Medical imaging may require small batch sizes (1-2)
8. **Learning rate** - Use lower LR for SVI-only training (5e-5 typical)

## Troubleshooting

### "Module not found" errors
```bash
pip install -e .  # Reinstall in editable mode
```

### CUDA out of memory
```yaml
training:
  batch_size: 1  # Reduce batch size
model:
  use_checkpoint: true  # Enable gradient checkpointing
```

### Model not learning
- Check your loss function returns proper gradients
- Try different `bayes_patterns` (more or fewer layers)
- Adjust learning rate
- Check data normalization

### Validation loss not improving
- Increase `patience` for early stopping
- Check for data leakage between train/val
- Try different prior scales

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [INSTALL.md](INSTALL.md) for installation options
- See [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) for new features
- Review example scripts in `bayzflow/examples/`

## Getting Help

If you encounter issues:
1. Check this guide and the README
2. Review the example scripts
3. Examine the YAML configuration files
4. Check the CLAUDE.md for development guidelines

Happy Bayesian modeling! 🎯
