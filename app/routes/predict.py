import shutil
import uuid
import os
import base64
from fastapi import APIRouter, UploadFile, File
from src.services.predictor import run_prediction
from src.services.metadata_checker import get_metadata
from src.services.gradcam import generate_gradcam
from PIL import Image
import io

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction = run_prediction(temp_path)
        metadata = get_metadata(temp_path)

        # Generate Grad-CAM and encode as base64 for API response
        cam_image = generate_gradcam(temp_path)
        pil_cam = Image.fromarray(cam_image)
        buffer = io.BytesIO()
        pil_cam.save(buffer, format="JPEG")
        cam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    finally:
        os.remove(temp_path)

    return {
        "prediction": prediction,
        "metadata": metadata,
        "gradcam": cam_b64,
        "note": "This is a model-based estimate, not definitive proof."
    }