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
    """
    Loads the binary classifier for Grad-CAM use.
    Separate from inference.load_model() because GradCAM needs
    gradients enabled on all parameters, which we handle in generate_gradcam().
    """
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def generate_gradcam(image_path: str, model=None) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap showing which image regions influenced the prediction.

    How Grad-CAM works:
    1. Run forward pass through the model
    2. Compute gradients of the output with respect to the last conv layer (layer4)
    3. Weight the feature maps by their gradients
    4. Average the weighted maps to get a single heatmap
    5. Overlay the heatmap on the original image

    Red/yellow = high activation (model paid attention here)
    Blue = low activation (model ignored this region)

    Why layer4:
    - Last convolutional block in ResNet18
    - Contains the most semantically rich spatial features
    - Standard choice for Grad-CAM in ResNet architectures

    Why we enable gradients:
    - model.eval() disables dropout but doesn't disable gradient tracking
    - However our frozen layers have requires_grad=False
    - Grad-CAM needs gradients to flow through all layers
    - We re-enable them here specifically for visualization

    Returns: RGB numpy array (H x W x 3) with heatmap overlaid on image
    """
    if model is None:
        model = load_model_for_gradcam()

    # Re-enable gradients on all parameters — required for Grad-CAM
    # Without this, gradients are None and Grad-CAM throws AttributeError
    for param in model.parameters():
        param.requires_grad = True

    # Target the last convolutional block — richest spatial features
    target_layer = [model.layer4[-1]]

    # Preprocess image for model input
    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Also prepare raw image for overlay (must be float32, range 0.0-1.0, HxWxC)
    img_resized = image.resize((224, 224))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Generate the class activation map
    # targets=None means Grad-CAM uses the highest-scoring class automatically
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=None)
        grayscale_cam = grayscale_cam[0]  # remove batch dimension

    # Overlay heatmap on the original image
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization  # numpy array ready for PIL or matplotlib


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