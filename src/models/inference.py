import torch
from PIL import Image
from src.models.model import build_model
from src.data.transforms import val_transforms
from src.data.generator_loader import CLASS_NAMES

# Default paths for saved model checkpoints
MODEL_PATH = "saved_models/best_model.pth"
GENERATOR_MODEL_PATH = "saved_models/generator_model.pth"


# --- Binary Classifier (Real vs Fake) ---

def load_model(model_path=MODEL_PATH):
    """
    Loads the binary classifier (ResNet18) from a saved checkpoint.
    pretrained=False because we're loading our own trained weights, not ImageNet.
    map_location="cpu" ensures the model loads on any machine regardless of GPU availability.
    model.eval() disables dropout for deterministic inference.
    """
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(image_path: str, model=None):
    """
    Predicts whether an image is real or AI-generated.

    Flow:
    1. Load image and apply val_transforms (resize to 224x224, normalize)
    2. unsqueeze(0) adds batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    3. Forward pass returns raw logit (unbounded number)
    4. sigmoid converts logit to probability (0.0 to 1.0)
    5. prob >= 0.5 means AI-Generated, else Real
    6. Confidence = how far from 0.5 the probability is

    Returns dict with label, confidence percentage, and raw sigmoid score.
    """
    if model is None:
        model = load_model()

    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():  # disable gradient tracking for inference
        output = model(tensor)
        prob = torch.sigmoid(output).item()  # convert logit to probability

    label = "AI-Generated" if prob >= 0.5 else "Real"
    # Confidence = distance from decision boundary (0.5)
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(prob, 4)
    }


# --- Generator Type Classifier (4-class) ---

def load_generator_model(model_path=GENERATOR_MODEL_PATH):
    """
    Loads the 4-class generator type classifier from a saved checkpoint.
    Classes: Real, GAN, Diffusion, Other (defined in generator_loader.CLASS_NAMES)
    Handles DataParallel prefix (module.) if model was trained with multiple GPUs.
    """
    from src.models.train_generator import build_multiclass_model
    model = build_multiclass_model(num_classes=4, pretrained=False)

    state_dict = torch.load(model_path, map_location="cpu")

    # Remove 'module.' prefix added by DataParallel when training on multiple GPUs
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_generator(image_path: str, model=None):
    """
    Predicts the generator type of an image (Real, GAN, Diffusion, Other).

    Flow:
    1. Same preprocessing as binary classifier
    2. Forward pass returns 4 raw logits (one per class)
    3. softmax converts logits to probabilities summing to 1.0
    4. argmax picks the class with highest probability
    5. Returns predicted class, confidence, and all class probabilities

    Unlike binary classifier which uses sigmoid (single output),
    multi-class uses softmax (4 outputs) so probabilities sum to 100%.
    """
    if model is None:
        model = load_generator_model()

    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]  # convert logits to probabilities
        pred_class = probs.argmax().item()        # index of highest probability class
        confidence = probs[pred_class].item()

    return {
        "generator_type": CLASS_NAMES[pred_class],
        "confidence": round(confidence * 100, 2),
        # All 4 class probabilities for display in UI
        "class_probabilities": {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
            for i in range(4)
        }
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/models/inference.py <image_path>")
    else:
        result = predict(sys.argv[1])
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']}%")