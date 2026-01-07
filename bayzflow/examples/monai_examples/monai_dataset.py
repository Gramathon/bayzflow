# examples/monai/monai_dataset.py

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import monai
    from monai.data import CacheDataset, DataLoader as MonaiDataLoader
    from monai.apps import DecathlonDataset
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandFlipd,
        RandRotate90d,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropd,
        ResizeWithPadOrCropd, 
        Resized,
        Spacingd,
        ToTensord,
        AsDiscreted,
        EnsureTyped,
        Lambdad,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    CacheDataset = None
    DecathlonDataset = None
    Compose = None


class MonaiDataset:
    """
    MONAI-based dataset for 3D medical image segmentation using DecathlonDataset.

    Provides train/val/test splits with appropriate transforms for medical imaging.
    Designed to work with BayzFlow's experiment framework.
    """

    def __init__(
        self,
        data_dir: str,
        task: str = "Task01_BrainTumour",
        section: str = "training",
        download: bool = True,
        spatial_size: Sequence[int] = (96, 96, 96),
        cache_rate: float = 1.0,
        num_workers: int = 4,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            data_dir: Root directory where MONAI will download/store the dataset
            task: Decathlon task name (e.g., 'Task01_BrainTumour', 'Task02_Heart', etc.)
            section: Dataset section ('training', 'validation', 'test')
            download: Whether to download the dataset if not present
            spatial_size: Target spatial size for resampling
            cache_rate: Fraction of data to cache in memory (0.0 to 1.0)
            num_workers: Number of workers for data loading
            train_frac: Fraction of data to use for training
            val_frac: Fraction of data to use for validation
            device: Torch device to use
        """
        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for MonaiDataset. "
                "Install with: pip install monai"
            )

        self.data_dir = data_dir
        self.task = task
        self.section = section
        self.download = download
        self.spatial_size = spatial_size
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1.0 - train_frac - val_frac

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Initialize transforms
        self._setup_transforms()

        # Create datasets using DecathlonDataset
        self._create_datasets()
    
    
    def _setup_transforms(self):
        """
        Setup MONAI transforms for training and validation.
        BrainTumour dataset has 4 modalities and 4 label classes.
        """
        # Keys for dictionary-based transforms
        keys = ["image", "label"]

        # Common transforms - must start with LoadImaged to load from file paths
        common_transforms = [
            LoadImaged(keys=keys),  # Load images from file paths
            # Image has channels at last position (H,W,D,C), need to move to first (C,H,W,D)
            #EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            Lambdad(keys=["image"], func=lambda x: x[:1, ...]),
            # Label has no channel dimension (H,W,D), need to add one at first position (1,H,W,D)
            #EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            #Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(1.5, 1.5, 2.0),  # Downsample for memory efficiency
                mode=("bilinear", "nearest")
            ),
            CropForegroundd(keys=keys, source_key="image"),
        ]

        # Training transforms (with augmentation)
        train_transforms = common_transforms + [
            RandSpatialCropd(keys=keys, roi_size=self.spatial_size, random_size=False),
            RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
            RandFlipd(keys=keys, spatial_axis=[1], prob=0.5),
            RandFlipd(keys=keys, spatial_axis=[2], prob=0.5),
            RandRotate90d(keys=keys, prob=0.1, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.spatial_size),
            ToTensord(keys=keys),
            # Convert labels to long dtype for multi-class segmentation
            AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.long),
        ]

        # Validation transforms (no augmentation)
        val_transforms = common_transforms + [
            Resized(keys=keys, spatial_size=self.spatial_size, mode=("trilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=keys),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.spatial_size),
            # Convert labels to long dtype for multi-class segmentation
            AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.long),
        ]

        self.train_transform = Compose(train_transforms)
        self.val_transform = Compose(val_transforms)
    
    def _create_datasets(self):
        """
        Create train/val/test datasets using MONAI's DecathlonDataset.
        This version is self-contained and guarantees 1-channel images
        with fixed spatial size for SwinUNETR (in_channels=1).
        """
        print(f"[MonaiDataset] Loading {self.task} from {self.data_dir}")
        print(f"[MonaiDataset] Download mode: {self.download}")

        keys = ["image", "label"]

        # --------------------------------------------------
        # 1) Define transforms (train & val/test)
        # --------------------------------------------------
        common_transforms = [
            # Load images/labels from file paths in the dict
            LoadImaged(keys=keys),

            # Make channels first for both image and label:
            #   image: (H,W,D[,C]) -> (C,H,W,D)
            #   label: (H,W,D) -> (1,H,W,D)
            EnsureChannelFirstd(keys=keys),

            # 🔴 Force single-channel input for the model:
            # For CT (Spleen): C=1 → no-op.
            # For any multi-channel data: C>1 → keep only first channel.
            Lambdad(keys="image", func=lambda x: x[:1, ...]),

            Orientationd(keys=keys, axcodes="RAS"),

            Spacingd(
                keys=keys,
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),

            CropForegroundd(keys=keys, source_key="image"),
        ]

        train_transform = Compose(
            common_transforms
            + [
                # Enforce fixed spatial size for SwinUNETR
                ResizeWithPadOrCropd(keys=keys, spatial_size=self.spatial_size),
                EnsureTyped(keys=keys),
            ]
        )

        val_transform = Compose(
            common_transforms
            + [
                ResizeWithPadOrCropd(keys=keys, spatial_size=self.spatial_size),
                EnsureTyped(keys=keys),
            ]
        )

        try:
            # --------------------------------------------------
            # 2) Load full Decathlon dataset (no transform here)
            # --------------------------------------------------
            full_dataset = DecathlonDataset(
                root_dir=self.data_dir,
                task=self.task,
                section=self.section,
                download=self.download,
                transform=None,      # transforms applied in CacheDataset
                cache_rate=0.0,
                num_workers=0,
            )

            print(f"[MonaiDataset] Loaded {len(full_dataset)} samples from {self.task}")

            # Get all data dictionaries
            all_data = [full_dataset[i] for i in range(len(full_dataset))]

            # --------------------------------------------------
            # 3) Split into train / val / test
            # --------------------------------------------------
            n_total = len(all_data)
            n_train = int(n_total * self.train_frac)
            n_val = int(n_total * self.val_frac)

            train_data = all_data[:n_train]
            val_data = all_data[n_train:n_train + n_val]
            test_data = all_data[n_train + n_val:]

            print(
                f"[MonaiDataset] Split: "
                f"{len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
            )

            # --------------------------------------------------
            # 4) Wrap in CacheDataset with the transforms
            # --------------------------------------------------
            if train_data:
                self.train_ds = CacheDataset(
                    data=train_data,
                    transform=train_transform,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
            else:
                self.train_ds = None

            if val_data:
                self.val_ds = CacheDataset(
                    data=val_data,
                    transform=val_transform,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
            else:
                self.val_ds = None

            if test_data:
                self.test_ds = CacheDataset(
                    data=test_data,
                    transform=val_transform,  # val transforms for test
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
            else:
                self.test_ds = None

        except Exception as e:
            print(f"[MonaiDataset] Error loading dataset: {e}")
            print("[MonaiDataset] Falling back to dummy datasets")
            self.train_ds = None
            self.val_ds = None
            self.test_ds = None

    
    def train_loader(self, batch_size: int = 2, shuffle: bool = True) -> DataLoader:
        """
        Create training data loader.
        """
        if self.train_ds is None:
            # Return a dummy loader for testing
            print("[MonaiDataset] Warning: No training data available, returning dummy loader")
            return self._create_dummy_loader(batch_size)
        
        return DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Use 0 for compatibility
            pin_memory=True if self.device.type == "cuda" else False,
        )
    
    def val_loader(self, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        """
        Create validation data loader.
        """
        if self.val_ds is None:
            print("[MonaiDataset] Warning: No validation data available, returning dummy loader")
            return self._create_dummy_loader(batch_size)
        
        return DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )
    
    def test_loader(self, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        """
        Create test data loader.
        """
        if self.test_ds is None:
            print("[MonaiDataset] Warning: No test data available, returning dummy loader")
            return self._create_dummy_loader(batch_size)
        
        return DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )
    
    def _create_dummy_loader(self, batch_size: int) -> DataLoader:
        """
        Create a dummy data loader with synthetic data for testing.
        Uses 4 channels to match BrainTumour dataset (4 MRI modalities).
        """
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, spatial_size):
                self.spatial_size = spatial_size

            def __len__(self):
                return 10  # Small dummy dataset

            def __getitem__(self, idx):
                # Create dummy 3D volume and segmentation mask on CPU
                # The training loop will move them to the appropriate device
                # BrainTumour dataset has 4 MRI modalities, so use 4 channels
                image = torch.randn(4, *self.spatial_size)
                # 4 label classes for brain tumor segmentation
                label = torch.randint(0, 4, self.spatial_size, dtype=torch.long)
                return {"image": image, "label": label}

        dummy_ds = DummyDataset(self.spatial_size)
        return DataLoader(dummy_ds, batch_size=batch_size, shuffle=False)
    
    def meta_for_checkpoint(self) -> dict:
        """
        Return metadata for checkpointing.
        """
        return {
            "data_dir": self.data_dir,
            "task": self.task,
            "section": self.section,
            "spatial_size": self.spatial_size,
            "cache_rate": self.cache_rate,
            "num_workers": self.num_workers,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "n_train": len(self.train_ds) if self.train_ds is not None else 0,
            "n_val": len(self.val_ds) if self.val_ds is not None else 0,
            "n_test": len(self.test_ds) if self.test_ds is not None else 0,
        }