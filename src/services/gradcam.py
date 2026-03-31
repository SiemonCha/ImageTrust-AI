import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.model import build_model
from src.data.transforms import val_transforms

MODEL_PATH = "saved_models/best_model.pth"


def load_model_for_gradcam():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def generate_gradcam(image_path: str, model=None) -> np.ndarray:
    if model is None:
        model = load_model_for_gradcam()

    # Enable gradients for GradCAM
    for param in model.parameters():
        param.requires_grad = True

    # Target layer — last conv block of ResNet18
    target_layer = [model.layer4[-1]]

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)

    # Convert image to numpy for overlay (0.0-1.0 range, H x W x C)
    img_resized = image.resize((224, 224))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Generate CAM
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=None)
        grayscale_cam = grayscale_cam[0]

    # Overlay heatmap on image
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_images/test_fake.jpg"
    result = generate_gradcam(image_path)

    plt.imshow(result)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.savefig("sample_images/gradcam_output.jpg")
    print("Saved to sample_images/gradcam_output.jpg")