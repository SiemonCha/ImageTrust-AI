import torch
import torch.nn as nn
from torchvision import models


def build_model(pretrained=True):
    """
    Builds ResNet18 for binary image classification (Real vs AI-Generated).

    Transfer learning approach:
    - Load ResNet18 pretrained on ImageNet (1000 classes)
    - Freeze all layers so pretrained features are preserved
    - Replace the final FC layer with our custom binary classifier head
    - During training, only layer4 and fc are unfrozen (see train.py)

    Architecture change:
    - Original: ResNet18 -> 512 features -> 1000 classes
    - Ours:     ResNet18 -> 512 features -> 256 -> 1 (binary output)

    Output: single logit — apply sigmoid to get probability (0.0 to 1.0)
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Freeze all pretrained layers — prevents destroying learned ImageNet features
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer for binary output
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(0.6),           # aggressive dropout to prevent overfitting
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),           # second dropout layer for regularisation
        nn.Linear(256, 1)          # single output for binary classification
    )

    return model


def build_efficientnet(pretrained=True):
    """
    Builds EfficientNet-B0 for binary image classification.
    Used for model comparison against ResNet18 in V2.

    EfficientNet-B0 vs ResNet18:
    - EfficientNet final features: 1280 (richer representation)
    - ResNet18 final features: 512
    - EfficientNet uses compound scaling (depth + width + resolution)
    - In our experiments, ResNet18 outperformed EfficientNet on this dataset
      at 30k samples — EfficientNet needs more data to show its advantage

    Output: single logit — apply sigmoid to get probability (0.0 to 1.0)
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head for binary output
    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )

    return model


if __name__ == "__main__":
    model = build_model()
    print(model.fc)

    # Quick shape test — batch of 4 images at 224x224
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # should be [4, 1]