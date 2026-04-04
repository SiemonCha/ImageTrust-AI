from src.models.inference import load_model, predict

# Singleton pattern — model loaded once and reused for all predictions
# Without this, every API call would reload the 44MB model from disk
# which would make the API extremely slow (~2-3 seconds per request)
_model = None


def get_model():
    """
    Returns the loaded binary classifier, loading it on first call only.
    Subsequent calls return the cached model instance.
    This is the singleton pattern — ensures only one model lives in memory.
    """
    global _model
    if _model is None:
        _model = load_model()
    return _model


def run_prediction(image_path: str) -> dict:
    """
    Service layer wrapper around inference.predict().
    Called by the FastAPI endpoint — separates API logic from ML logic.
    Returns dict with label, confidence, and raw_score.
    """
    model = get_model()
    result = predict(image_path, model=model)
    return result