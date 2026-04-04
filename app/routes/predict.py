import shutil
import uuid
import os
import base64
from fastapi import APIRouter, UploadFile, File
from src.services.predictor import run_prediction
from src.services.metadata_checker import get_metadata
from src.services.gradcam import generate_gradcam
from src.models.inference import predict_generator, load_generator_model
from PIL import Image
import io

router = APIRouter()

# Temporary directory for uploaded files during processing
# Files are deleted immediately after prediction is complete
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Singleton for generator model — same pattern as predictor.py
# Prevents reloading the model on every API request
_generator_model = None


def get_generator_model():
    """Returns cached generator model, loading it on first call only."""
    global _generator_model
    if _generator_model is None:
        _generator_model = load_generator_model()
    return _generator_model


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    POST /predict endpoint — accepts an image file and returns full analysis.

    Request: multipart/form-data with image file
    Response JSON:
        - prediction: binary real/fake result with confidence
        - generator: 4-class generator type with class probabilities
        - metadata: file info and EXIF data
        - gradcam: base64-encoded JPEG heatmap image
        - note: disclaimer about model limitations

    File handling:
    - Save uploaded file to temp_uploads/ with UUID prefix to avoid collisions
    - UUID prevents filename conflicts if multiple users upload simultaneously
    - Delete temp file in finally block — runs even if prediction fails
    - We save to disk (not memory) because our inference functions take file paths
    """
    # Save uploaded file temporarily with unique filename
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run all analysis on the temp file
        prediction = run_prediction(temp_path)
        generator_result = predict_generator(temp_path, model=get_generator_model())
        metadata = get_metadata(temp_path)

        # Generate Grad-CAM heatmap
        # REST APIs return JSON (text only) — images must be encoded as base64 string
        # Base64 converts binary image bytes to ASCII text safe for JSON transport
        # The Streamlit/Gradio UI decodes it back to display the image
        cam_image = generate_gradcam(temp_path)
        pil_cam = Image.fromarray(cam_image)
        buffer = io.BytesIO()
        pil_cam.save(buffer, format="JPEG")
        cam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    finally:
        # Always delete temp file — runs even if an exception occurs above
        os.remove(temp_path)

    return {
        "prediction": prediction,
        "generator": generator_result,
        "metadata": metadata,
        "gradcam": cam_b64,
        "note": "This is a model-based estimate, not definitive proof."
    }