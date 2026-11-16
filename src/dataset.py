import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from typing import List, Tuple, Optional

from preprocessing import load_image, preprocess_image

class SurfaceDefectDataset(Dataset):
    """
    PyTorch Dataset for Industrial Surface Defect Detection.
    Connects the raw image paths and labels to the PyTorch training loop.
    """
    def __init__(self,image_paths: List[str],labels: List[int],image_size: int = 224,transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.

        Args:
            image_paths (List[str]): List of file paths to the images.
            labels (List[int]): List of corresponding labels (0 for OK, 1 for NG).
            image_size (int): Target size for resizing the image.
            transform (torchvision.transforms.Compose, optional): PyTorch transform pipeline.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform
        
        # Fallback transform if none is provided: 
        # Converts NumPy array (H, W, C) to Tensor (C, H, W) and scales to [0.0, 1.0].
        self.default_transform = transforms.ToTensor()

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch the image and label at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image_tensor, label_tensor)
        """
        
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        
        image_np = load_image(img_path, as_gray=False)

        image_preprocessed = preprocess_image(image_np, size=self.image_size)

        if self.transform is not None:
            image_tensor = self.transform(image_preprocessed)
        else:
            image_tensor = self.default_transform(image_preprocessed)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Get the data transformation pipeline for training or inference.

    Args:
        is_train (bool): If True, applies data augmentations (flips, etc.) to prevent overfitting.
                         If False, only applies essential tensor conversion and normalization.

    Returns:
        transforms.Compose: A sequence of PyTorch transformations.
    """
    # ImageNet normalization statistics. 
    # Required when using pre-trained models like ResNet18.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        # Pipeline for Training (Includes Data Augmentation)
        return transforms.Compose([
            transforms.ToTensor(),                  # Convert to tensor and scale to [0, 1]
            transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip horizontally
            transforms.RandomVerticalFlip(p=0.5),   # 50% chance to flip vertically
            normalize                               # Normalize with ImageNet stats
        ])
    else:
        # Pipeline for Validation / Testing / Inference (No Data Augmentation)
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])