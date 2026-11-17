import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SurfaceDefectModel(nn.Module):
    """
    PyTorch Model for Industrial Surface Defect Detection.
    It uses a pre-trained ResNet18 as the backbone (feature extractor) 
    and replaces the final classification head for our specific task.
    """
    def __init__(self, num_classes: int = 2, fine_tune: bool = True):
        """
        Initialize the model.

        Args:
            num_classes (int): Number of output classes (e.g., 2 for OK/NG).
            fine_tune (bool): If True, all layers are trainable. 
                              If False, the backbone is frozen (Feature Extraction mode),
                              and only the new classification head will be trained.
        """
        super(SurfaceDefectModel, self).__init__()

        # Load the pre-trained ResNet18 model using modern torchvision API
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze the backbone weights if fine_tune is set to False
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False  # Lock the weights, no gradient calculation

        # Replace the final Classification Head
        # Get the number of input features to the original fully connected (fc) layer
        in_features = self.backbone.fc.in_features
        
        # Replace the original 1000-class head with our new N-class head (2 classes: OK/NG)
        # By default, a newly created nn.Linear layer has requires_grad=True
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images, shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: The output logits, shape (Batch_Size, num_classes).
        """
        return self.backbone(x)

def get_model(num_classes: int = 2, fine_tune: bool = True) -> nn.Module:
    """
    Helper function to instantiate and return the model.
    """
    return SurfaceDefectModel(num_classes=num_classes, fine_tune=fine_tune)