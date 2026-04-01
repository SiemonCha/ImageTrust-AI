import torch
from PIL import Image
from src.models.model import build_model
from src.data.transforms import val_transforms
from src.data.generator_loader import CLASS_NAMES

MODEL_PATH = "saved_models/best_model.pth"

def load_model(model_path=MODEL_PATH):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict(image_path: str, model=None):
    if model is None:
        model = load_model()

    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    label = "AI-Generated" if prob >= 0.5 else "Real"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(prob, 4)
    }

GENERATOR_MODEL_PATH = "saved_models/generator_model.pth"

def load_generator_model(model_path=GENERATOR_MODEL_PATH):
    from src.models.train_generator import build_multiclass_model
    model = build_multiclass_model(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict_generator(image_path: str, model=None):
    if model is None:
        model = load_generator_model()

    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    return {
        "generator_type": CLASS_NAMES[pred_class],
        "confidence": round(confidence * 100, 2),
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