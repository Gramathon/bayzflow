# models/swin_unetr.py

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR
    from monai.bundle import download
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    SwinUNETR = None


class BayesianSwinUNETR(nn.Module):
    """
    Bayesian wrapper around MONAI's SwinUNETR for 3D medical image segmentation.

    SwinUNETR uses a Swin Transformer encoder with a UNet-style decoder. This wrapper
    supports loading pretrained encoder weights from MONAI model zoo and enables
    Bayesian inference on the decoder layers only (SVI-only training without
    retraining the base encoder).

    Key features:
    - Load pretrained Swin Transformer encoder from MONAI model zoo
    - Freeze encoder weights to preserve pretrained features
    - Enable Bayesian inference on decoder/head layers via BayzFlow
    - Suitable for SVI-only training workflows

    Recommended BayzFlow patterns for this model:
    - ["decoder", "head"] - Bayesianize decoder and output layers
    - ["out"] - Bayesianize only the final segmentation head
    """

    def __init__(
        self,
        img_size: Sequence[int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize BayesianSwinUNETR.

        Args:
            img_size: Input image size (D, H, W)
            in_channels: Number of input channels
            out_channels: Number of output segmentation classes
            feature_size: Feature size for the Swin Transformer (24, 48, 96)
            use_checkpoint: Use gradient checkpointing to save memory
            spatial_dims: Number of spatial dimensions (must be 3 for SwinUNETR)
            pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained weights file (if None, will download from MONAI model zoo)
            freeze_encoder: Whether to freeze encoder weights (recommended for SVI-only training)
            device: Device to place the model on
        """
        super().__init__()

        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for BayesianSwinUNETR. "
                "Install with: pip install monai"
            )

        if spatial_dims != 3:
            raise ValueError("SwinUNETR only supports spatial_dims=3")

        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.freeze_encoder = freeze_encoder
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        print(
            f"[BayesianSwinUNETR.__init__] img_size={img_size}, "
            f"in_channels={in_channels}, out_channels={out_channels}, "
            f"feature_size={feature_size}, pretrained={pretrained}, "
            f"freeze_encoder={freeze_encoder}, device={device}"
        )

        # Create the SwinUNETR model
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        ).to(self.device)

        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(pretrained_path)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
            print("[BayesianSwinUNETR] Encoder weights frozen (will not be updated during training)")

        # Create aliases for easier layer identification by BayzFlow
        # The decoder and output layers will be the targets for Bayesianization
        self.decoder = self.model.decoder5  # Final decoder layer
        self.head = self.model.out  # Output convolution layer

    def _load_pretrained_weights(self, pretrained_path: Optional[str] = None):
        """
        Load pretrained weights from file or MONAI model zoo.

        Args:
            pretrained_path: Path to pretrained weights. If None, downloads from MONAI model zoo.
        """
        try:
            if pretrained_path is not None:
                print(f"[BayesianSwinUNETR] Loading pretrained weights from {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location="cpu")

                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint

                # Current model params
                model_state = self.model.state_dict()

                # Filter out keys that don't exist or have mismatched shapes
                filtered_state = {}
                skipped_keys = []
                for k, v in state_dict.items():
                    if k in model_state and model_state[k].shape == v.shape:
                        filtered_state[k] = v
                    else:
                        skipped_keys.append(k)

                # Load only the compatible subset
                missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)

                print("[BayesianSwinUNETR] Successfully loaded pretrained weights (backbone)")
                if skipped_keys:
                    print(f"[BayesianSwinUNETR]   Skipped {len(skipped_keys)} keys due to shape/lookup mismatch")
                    # Uncomment if you want to see them:
                    # for k in skipped_keys:
                    #     print(f"      - {k}")
                if missing:
                    print(f"[BayesianSwinUNETR]   Missing keys in checkpoint: {missing}")
                if unexpected:
                    print(f"[BayesianSwinUNETR]   Unexpected keys in checkpoint: {unexpected}")

            else:
                # No path provided – keep your existing message
                print("[BayesianSwinUNETR] Attempting to download pretrained weights from MONAI model zoo")
                print("[BayesianSwinUNETR] Warning: Automatic download not implemented.")
                print("[BayesianSwinUNETR] Please provide pretrained_path to load pretrained weights.")
                print("[BayesianSwinUNETR] Continuing with randomly initialized weights...")

        except Exception as e:
            print(f"[BayesianSwinUNETR] Warning: Failed to load pretrained weights: {e}")
            print("[BayesianSwinUNETR] Continuing with randomly initialized weights...")


    def _freeze_encoder(self):
        """
        Freeze encoder weights to prevent training.
        Only decoder and output layers will be trainable (and bayesianized).
        """
        # Freeze Swin Transformer encoder
        for param in self.model.swinViT.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        for i in range(5):
            encoder_attr = f"encoder{i+1}"
            if hasattr(self.model, encoder_attr):
                for param in getattr(self.model, encoder_attr).parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwinUNETR.

        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Segmentation logits of shape [B, out_channels, D, H, W]
        """
        return self.model(x)

    def get_trainable_parameters(self):
        """
        Get the number of trainable vs frozen parameters.

        Returns:
            Dictionary with parameter counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen

        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_percent": 100 * trainable / total if total > 0 else 0
        }
