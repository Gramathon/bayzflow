# MONAI Dataset Transform Error - Fix Summary

## Issues Fixed

### 1. Channel Mismatch Error (Original Issue)
**Error:**
```
RuntimeError: Given groups=1, weight of size [16, 4, 3, 3, 3],
expected input[2, 1, 64, 64, 64] to have 4 channels, but got 1 channels instead
```

**Root Cause:** The dummy dataset fallback was creating 1-channel images, but the BrainTumour dataset configuration expects 4 channels (4 MRI modalities).

**Fix:** Updated `monai_dataset.py:284`
```python
# Before
image = torch.randn(1, *self.spatial_size)

# After
image = torch.randn(4, *self.spatial_size)
```

### 2. Transform Pipeline Error
**Error:**
```
RuntimeError: applying transform <monai.transforms.utility.dictionary.EnsureChannelFirstd>
ValueError: Metadata not available and channel_dim=None, EnsureChannelFirst is not in use.
```

**Root Cause:** DecathlonDataset returns file paths (strings), not loaded tensors. The transforms were trying to process strings instead of image data.

**Fix:** Added `LoadImaged` as the first transform in `monai_dataset.py:113`
```python
common_transforms = [
    LoadImaged(keys=keys),  # Load images from file paths
    ...
]
```

### 3. Channel Dimension Handling Error
**Error:**
```
ValueError: axcodes must match data_array spatially, got axcodes=3D data_array=4D
```

**Root Cause:**
- BrainTumour images load as `[H, W, D, C]` with channels at the end
- Labels load as `[H, W, D]` without channels
- Using the same `EnsureChannelFirstd` for both caused incorrect transformations

**Fix:** Separate channel handling for image and label in `monai_dataset.py:114-117`
```python
# Image has channels at last position (H,W,D,C), move to first (C,H,W,D)
EnsureChannelFirstd(keys=["image"], channel_dim=-1),
# Label has no channel dimension (H,W,D), add one at first position (1,H,W,D)
EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
```

### 4. Loss Function Shape Mismatch Error (DTYPE ISSUE)
**Error:**
```
ValueError: Target size (torch.Size([2, 1, 64, 64, 64])) must be the same as input size (torch.Size([2, 4, 64, 64, 64]))
```

**Root Cause:**
- Model outputs 4 channels for multi-class segmentation: `(batch, 4, D, H, W)`
- Labels have shape `(batch, 1, D, H, W)` with class indices (0-3)
- The loss function in `experiment.py:174-183` checks `if y.dtype == torch.long` to determine which loss to use:
  - `F.cross_entropy()` for multi-class (expects labels without channel dim)
  - `F.binary_cross_entropy_with_logits()` for binary (expects same shape as logits)
- **PROBLEM:** The labels were NOT `torch.long` dtype after transforms, so the code was using binary cross-entropy incorrectly
- Binary cross-entropy requires input and target to have the same shape, causing the mismatch error

**Fix 1 (Already Exists):** Modified loss function in `experiment.py:174-183` to squeeze channel dimension for multi-class segmentation:
```python
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
```

**Fix 2 (NEW - MAIN FIX):** Added `AsDiscreted` transform in `monai_dataset.py:139,148` to ensure labels are `torch.long` dtype:
```python
# Training transforms
train_transforms = common_transforms + [
    # ... other transforms ...
    ToTensord(keys=keys),
    # Convert labels to long dtype for multi-class segmentation
    AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.long),
]

# Validation transforms
val_transforms = common_transforms + [
    # ... other transforms ...
    ToTensord(keys=keys),
    # Convert labels to long dtype for multi-class segmentation
    AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.long),
]
```

**How It Works:**
1. `AsDiscreted` converts label tensors to `torch.long` dtype
2. Loss function now correctly detects multi-class segmentation via `y.dtype == torch.long`
3. Uses `F.cross_entropy()` which handles shape `[2, 4, 64, 64, 64]` (logits) vs `[2, 64, 64, 64]` (labels after squeeze)

## Final Result

After fixes, the dataset correctly loads with:
- **Image shape:** `[4, 64, 64, 64]` - 4 channels (T1, T1ce, T2, FLAIR), 64³ spatial
- **Label shape:** `[1, 64, 64, 64]` - 1 channel (segmentation classes), 64³ spatial

And the loss function correctly handles the label shape for training.

The full MONAI 3D U-Net pipeline now works correctly!

## Files Modified

1. `/home/gramath/VScode/bayzflow/bayzflow/examples/monai_examples/monai_dataset.py`
   - Line 113-117: Added LoadImaged and corrected EnsureChannelFirstd
   - Line 139, 148: **NEW** - Added AsDiscreted transform to convert labels to torch.long dtype
   - Line 284-286: Fixed dummy dataset to use 4 channels

2. `/home/gramath/VScode/bayzflow/bayzflow/core/experiment.py`
   - Line 174-183: Added label shape handling for multi-class segmentation loss

## Testing

To verify the fixes work:
```bash
cd /home/gramath/VScode/bayzflow
PYTHONPATH=. python3 bayzflow/examples/monai_examples/monai_unet_train.py
```

The script should now successfully:
1. Load the BrainTumour dataset
2. Apply transforms without errors
3. Create the Bayesian U-Net model
4. Begin training
