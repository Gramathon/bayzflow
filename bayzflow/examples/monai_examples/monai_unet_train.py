#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MONAI 3D U-Net Bayesian Segmentation Example

This example demonstrates how to use BayzFlow with MONAI's 3D U-Net
for medical image segmentation with uncertainty quantification.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from bayzflow import BayzFlow


def visualize_3d_segmentation(volume, gt_mask, pred_mask, uncertainty, num_slices=6):
    """
    Visualize 3D volume and segmentation masks as 2D slices.

    Args:
        volume: Input 3D volume [C, D, H, W] or [D, H, W]
        gt_mask: Ground truth segmentation [C, D, H, W] or [D, H, W]
        pred_mask: Predicted segmentation [C, D, H, W] or [D, H, W]
        uncertainty: Uncertainty map [C, D, H, W] or [D, H, W]
        num_slices: Number of evenly-spaced slices to display
    """
    # Convert to numpy and extract first channel if needed
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(uncertainty):
        uncertainty = uncertainty.cpu().numpy()

    # Handle channel dimension
    if volume.ndim == 4:
        volume = volume[0]  # Take first channel
    if gt_mask.ndim == 4:
        gt_mask = np.argmax(gt_mask, axis=0)  # Convert to class labels
    if pred_mask.ndim == 4:
        pred_mask = np.argmax(pred_mask, axis=0)
    if uncertainty.ndim == 4:
        uncertainty = uncertainty.mean(axis=0)  # Average across channels

    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    # Create figure with subplots
    fig, axes = plt.subplots(4, num_slices, figsize=(3 * num_slices, 12))

    # Create custom colormap for segmentation overlay
    colors = ['none', 'red', 'blue', 'green', 'yellow']
    n_colors = max(int(gt_mask.max()) + 1, int(pred_mask.max()) + 1, 2)
    cmap = ListedColormap(colors[:n_colors])

    for idx, slice_idx in enumerate(slice_indices):
        # Row 1: Original volume
        axes[0, idx].imshow(volume[slice_idx], cmap='gray')
        axes[0, idx].set_title(f'Slice {slice_idx}')
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_ylabel('Input Volume', fontsize=12, fontweight='bold')

        # Row 2: Ground truth segmentation overlay
        axes[1, idx].imshow(volume[slice_idx], cmap='gray')
        gt_overlay = np.ma.masked_where(gt_mask[slice_idx] == 0, gt_mask[slice_idx])
        axes[1, idx].imshow(gt_overlay, cmap=cmap, alpha=0.5, vmin=0, vmax=n_colors-1)
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_ylabel('Ground Truth', fontsize=12, fontweight='bold')

        # Row 3: Predicted segmentation overlay
        axes[2, idx].imshow(volume[slice_idx], cmap='gray')
        pred_overlay = np.ma.masked_where(pred_mask[slice_idx] == 0, pred_mask[slice_idx])
        axes[2, idx].imshow(pred_overlay, cmap=cmap, alpha=0.5, vmin=0, vmax=n_colors-1)
        axes[2, idx].axis('off')
        if idx == 0:
            axes[2, idx].set_ylabel('Prediction', fontsize=12, fontweight='bold')

        # Row 4: Uncertainty map
        im = axes[3, idx].imshow(uncertainty[slice_idx], cmap='hot')
        axes[3, idx].axis('off')
        if idx == 0:
            axes[3, idx].set_ylabel('Uncertainty', fontsize=12, fontweight='bold')

        # Add colorbar for the last uncertainty plot
        if idx == num_slices - 1:
            cbar = plt.colorbar(im, ax=axes[3, idx], fraction=0.046, pad=0.04)
            cbar.set_label('Std Dev', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig


def main():
    """
    Train a Bayesian MONAI 3D U-Net for medical image segmentation.
    """
    print("=== BayzFlow MONAI 3D U-Net Example ===")
    
    # Run the MONAI U-Net experiment using BayzFlow.exp
    try:
        bf, loaders, cfg, extras = BayzFlow.exp(
            "bayzflow/bayzflow.yaml", 
            exp="monai_unet_3d"
        )
        
        print(f"\n[Main] Training completed successfully!")
        print(f"[Main] Posterior configuration: {extras['posterior_cfg']}")
        
        # Demo prediction with posterior sampling
        print(f"\n=== Demonstration: Prediction with Uncertainty ===")
        
        # Get a test batch
        test_loader = loaders.get('test') or loaders['train']  # Fallback to train if no test
        batch = next(iter(test_loader))
        
        # Define prediction function for segmentation
        def predict_fn(model, batch):
            if isinstance(batch, dict):
                x = batch["image"]
            else:
                x = batch[0]
            return model(x)  # Returns segmentation logits
        
        # Get posterior samples with uncertainty
        posterior_cfg = extras['posterior_cfg']
        predictions = bf.predict(
            batch=batch,
            predict_fn=predict_fn,
            num_samples=posterior_cfg['num_samples'],
            device=extras['device'],
            return_samples=True,
            save_posterior_latents=posterior_cfg['save_latents'],
            posterior_save_path=posterior_cfg['save_path']
        )
        
        print(f"[Main] Prediction mean shape: {predictions['mean'].shape}")
        print(f"[Main] Prediction std shape: {predictions['std'].shape}")
        print(f"[Main] Prediction samples shape: {predictions['samples'].shape}")
        
        # Compute uncertainty statistics
        mean_uncertainty = predictions['std'].mean().item()
        max_uncertainty = predictions['std'].max().item()
        
        print(f"[Main] Mean voxel uncertainty: {mean_uncertainty:.4f}")
        print(f"[Main] Max voxel uncertainty: {max_uncertainty:.4f}")

        # Visualize results
        print(f"\n=== Visualizing 3D Segmentation Results ===")

        # Extract data for visualization
        if isinstance(batch, dict):
            input_volume = batch["image"][0]  # First sample in batch [C, D, H, W]
            gt_mask = batch["label"][0]  # Ground truth [C, D, H, W]
        else:
            input_volume = batch[0][0]
            gt_mask = batch[1][0]

        pred_mean = predictions['mean'][0]  # Predicted logits [C, D, H, W]
        pred_std = predictions['std'][0]  # Uncertainty [C, D, H, W]

        # Create visualization
        fig = visualize_3d_segmentation(
            volume=input_volume,
            gt_mask=gt_mask,
            pred_mask=pred_mean,
            uncertainty=pred_std,
            num_slices=6
        )

        # Save figure
        output_dir = Path("monai_artifacts")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "segmentation_visualization.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Main] Visualization saved to: {output_path}")

        # Display the plot
        plt.show()

        print(f"\n[Main] MONAI U-Net example completed!")
        
    except ImportError as e:
        print(f"[Main] MONAI not available: {e}")
        print("[Main] Install MONAI with: pip install monai")
        print("[Main] Skipping MONAI example...")
    
    except Exception as e:
        print(f"[Main] Error running MONAI experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()