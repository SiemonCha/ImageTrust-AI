from src.models.inference import load_model, predict

# Load model once at startup, reuse for all predictions
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def run_prediction(image_path: str) -> dict:
    model = get_model()
    result = predict(image_path, model=model)
    return result