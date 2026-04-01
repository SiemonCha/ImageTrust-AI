import gradio as gr
import torch
import base64
import io
from PIL import Image
from src.models.inference import load_model, predict, predict_generator, load_generator_model
from src.services.metadata_checker import get_metadata
from src.services.gradcam import generate_gradcam
import tempfile
import os

# Load model once
model = load_model()
generator_model = load_generator_model()

def analyze_image(image):
    if image is None:
        return "Please upload an image.", None

    # Save PIL image to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        prediction = predict(tmp_path, model=model)
        metadata = get_metadata(tmp_path)
        cam_array = generate_gradcam(tmp_path, model=model)
        cam_image = Image.fromarray(cam_array)
        generator_result = predict_generator(tmp_path, model=generator_model)
    finally:
        os.remove(tmp_path)

    label = prediction["label"]
    confidence = prediction["confidence"]

    result_text = f"**{label}** — Confidence: {confidence}%\n\n"
    result_text += f"⚠️ This is a model-based estimate, not definitive proof.\n\n"

    # Generator type
    gen_type = generator_result["generator_type"]
    gen_conf = generator_result["confidence"]
    result_text += f"**Generator Type:** {gen_type} ({gen_conf}%)\n\n"
    result_text += "**Class Probabilities:**\n"
    for cls, prob in generator_result["class_probabilities"].items():
        result_text += f"- {cls}: {prob}%\n"

    result_text += f"\n**Metadata:**\n"
    result_text += f"- Format: {metadata['format']}\n"
    result_text += f"- Dimensions: {metadata['dimensions']}\n"
    result_text += f"- File Size: {metadata['file_size_kb']} KB\n"
    result_text += f"- EXIF Data: {'Yes' if metadata['has_exif'] else 'No'}\n"

    if not metadata["has_exif"]:
        result_text += f"\n_{metadata['exif_note']}_"

    return result_text, cam_image


demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Markdown(label="Result"),
        gr.Image(type="pil", label="Grad-CAM Explanation")
    ],
    title="🔍 ImageTrust-AI",
    description="Upload an image to check if it's real or AI-generated. Powered by ResNet18 trained on ArtiFact dataset.",
    examples=[],
)

if __name__ == "__main__":
    demo.launch()