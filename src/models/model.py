import torch
import torch.nn as nn
from torchvision import models

def build_model(pretrained=True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
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
    
    # Quick shape test
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # should be [4, 1]