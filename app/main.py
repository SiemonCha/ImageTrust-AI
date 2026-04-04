from fastapi import FastAPI
from app.routes.predict import router

# FastAPI app entry point
# Run with: PYTHONPATH=. uvicorn app.main:app --reload
# API docs auto-generated at: http://127.0.0.1:8000/docs
app = FastAPI(title="ImageTrust-AI", version="1.0")

# Register the predict router — adds POST /predict endpoint
app.include_router(router)


@app.get("/")
def root():
    """Health check endpoint — confirms API is running."""
    return {"message": "ImageTrust-AI API is running"}