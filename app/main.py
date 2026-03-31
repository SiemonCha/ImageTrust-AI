from fastapi import FastAPI
from app.routes.predict import router

app = FastAPI(title="ImageTrust-AI", version="1.0")
app.include_router(router)

@app.get("/")
def root():
    return {"message": "ImageTrust-AI API is running"}