# models/monai_unet.py

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    import monai.networks.nets as monai_nets
    from monai.networks.nets import BasicUNet
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    BasicUNet = None


class BayesianMonaiUNet(nn.Module):
    """
    Bayesian wrapper around MONAI's BasicUNet for 3D medical image segmentation.

    This wrapper creates a standard MONAI U-Net and exposes it for BayzFlow
    to discover and bayesianize specific layers (e.g., decoder heads).

    The model automatically adapts to different input spatial sizes. Ensure
    that the spatial_size is compatible with the stride configuration:
    - Minimum spatial size should be 2^(num_strides) * base_size
    - Each stride reduces spatial dimensions by factor of stride value
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2),
        num_res_units: int = 2,
        norm: Union[str, tuple] = "INSTANCE",
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        spatial_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        
        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for BayesianMonaiUNet. "
                "Install with: pip install monai"
            )

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.spatial_size = spatial_size

        # Validate spatial size compatibility with strides
        if spatial_size is not None:
            self._validate_spatial_size(spatial_size, strides, spatial_dims)

        print(
            "[BayesianMonaiUNet.__init__] spatial_dims=", spatial_dims,
            "in_channels=", in_channels, "out_channels=", out_channels,
            "channels=", channels, "strides=", strides,
            "num_res_units=", num_res_units, "norm=", norm,
            "dropout=", dropout, "device=", device,
            "spatial_size=", spatial_size,
        )
        
        # Create the base MONAI U-Net
        self.unet = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=channels,
            act=("LeakyReLU", {"inplace": True}),
            norm=norm,
            bias=True,
            dropout=dropout,
        ).to(self.device)
        
        # The U-Net structure exposes encoder and decoder parts
        # BayzFlow can discover and bayesianize layers by pattern matching
        # Common patterns: "head" (final layers), "decoder" (upsampling path)
        
        # For easier identification, we can alias the final layer as 'head'
        self.head = self.unet[-1] if hasattr(self.unet, '__getitem__') else None

    def _validate_spatial_size(
        self,
        spatial_size: Sequence[int],
        strides: Sequence[int],
        spatial_dims: int
    ):
        """
        Validate that the spatial size is compatible with the stride configuration.

        Args:
            spatial_size: Input spatial dimensions
            strides: Stride values for each downsampling level
            spatial_dims: Number of spatial dimensions (2 or 3)
        """
        if len(spatial_size) != spatial_dims:
            raise ValueError(
                f"spatial_size length ({len(spatial_size)}) must match "
                f"spatial_dims ({spatial_dims})"
            )

        # Calculate minimum required spatial size
        total_downsampling = 1
        for stride in strides:
            total_downsampling *= stride

        min_size = total_downsampling * 2  # Add some margin

        for i, size in enumerate(spatial_size):
            if size < min_size:
                print(
                    f"[BayesianMonaiUNet] Warning: spatial_size[{i}]={size} "
                    f"may be too small for stride configuration (recommended >= {min_size}). "
                    f"This could cause issues with the U-Net architecture."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        The model handles arbitrary spatial sizes as long as they're compatible
        with the stride configuration.

        Args:
            x: Input tensor of shape [B, C, D, H, W] for 3D or [B, C, H, W] for 2D

        Returns:
            Segmentation logits of shape [B, out_channels, D, H, W] or [B, out_channels, H, W]
        """
        # Don't move input to self.device - let the training framework handle device placement
        # The model's parameters will be on the correct device after model.to(device) is called
        return self.unet(x)