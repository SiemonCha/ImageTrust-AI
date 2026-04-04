from torchvision import transforms

# ResNet18 and EfficientNet-B0 both expect 224x224 input
# This matches the size used during ImageNet pretraining
IMAGE_SIZE = 224

# Training transforms — includes augmentation to improve generalisation
train_transforms = transforms.Compose([
    # Resize all images to consistent size required by the model
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Data augmentation — artificially increases training data diversity
    # Applied randomly to each image during training
    transforms.RandomHorizontalFlip(),           # mirror image 50% of the time
    transforms.RandomRotation(10),               # rotate up to 10 degrees
    transforms.ColorJitter(brightness=0.2,       # slightly vary brightness
                           contrast=0.2,         # slightly vary contrast
                           saturation=0.2),      # slightly vary colour saturation

    # Convert PIL image (0-255 uint8) to PyTorch tensor (0.0-1.0 float)
    # Also changes shape from (H, W, C) to (C, H, W) as PyTorch expects
    transforms.ToTensor(),

    # Normalize using ImageNet mean and std values
    # Required because ResNet18 was pretrained on ImageNet with these statistics
    # Without this, the pretrained weights produce meaningless activations
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation/test transforms — NO augmentation
# Val and test must reflect real conditions to get accurate metrics
# Augmentation on val would make evaluation results unreliable
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])