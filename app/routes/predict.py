import shutil
import uuid
import os
from fastapi import APIRouter, UploadFile, File
from src.services.predictor import run_prediction
from src.services.metadata_checker import get_metadata

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction = run_prediction(temp_path)
        metadata = get_metadata(temp_path)
    finally:
        os.remove(temp_path)  # Clean up temp file

    return {
        "prediction": prediction,
        "metadata": metadata,
        "note": "This is a model-based estimate, not definitive proof."
    }