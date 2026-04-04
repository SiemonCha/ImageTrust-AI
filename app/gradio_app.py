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

# Load both models once at startup — singleton pattern
# Models stay in memory for the lifetime of the app
# Avoids 2-3 second reload penalty on every user request
model = load_model()
generator_model = load_generator_model()


def analyze_image(image):
    """
    Main function called by Gradio when user submits an image.

    Flow:
    1. Save PIL image to a temp file (inference functions require file paths)
    2. Run binary prediction (real vs fake)
    3. Run generator type prediction (Real/GAN/Diffusion/Other)
    4. Extract image metadata and EXIF
    5. Generate Grad-CAM heatmap
    6. Build formatted markdown result string
    7. Delete temp file
    8. Return (result_text, cam_image) — matches Gradio output components

    Why temp file:
    Gradio passes images as PIL objects but our inference functions
    expect file paths. We save temporarily and clean up after.
    """
    if image is None:
        return "Please upload an image.", None

    # Save PIL image to temp file with unique name
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Run all analysis pipelines
        prediction = predict(tmp_path, model=model)
        metadata = get_metadata(tmp_path)
        cam_array = generate_gradcam(tmp_path, model=model)
        cam_image = Image.fromarray(cam_array)  # convert numpy array to PIL for Gradio
        generator_result = predict_generator(tmp_path, model=generator_model)
    finally:
        # Always clean up temp file regardless of success or failure
        os.remove(tmp_path)

    label = prediction["label"]
    confidence = prediction["confidence"]

    # Build markdown-formatted result string for Gradio Markdown component
    result_text = f"**{label}** — Confidence: {confidence}%\n\n"
    result_text += f"⚠️ This is a model-based estimate, not definitive proof.\n\n"
    result_text += f"📌 **Note:** Model performs best on Stable Diffusion, StyleGAN, and DDPM images. "
    result_text += f"Performance drops on unseen generators (DALL-E, MidJourney).\n\n"

    # Generator type section — shows architecture family
    gen_type = generator_result["generator_type"]
    gen_conf = generator_result["confidence"]
    result_text += f"**Generator Type:** {gen_type} ({gen_conf}%)\n\n"
    result_text += "**Class Probabilities:**\n"
    for cls, prob in generator_result["class_probabilities"].items():
        result_text += f"- {cls}: {prob}%\n"

    # Metadata section
    result_text += f"\n**Metadata:**\n"
    result_text += f"- Format: {metadata['format']}\n"
    result_text += f"- Dimensions: {metadata['dimensions']}\n"
    result_text += f"- File Size: {metadata['file_size_kb']} KB\n"
    result_text += f"- EXIF Data: {'Yes' if metadata['has_exif'] else 'No'}\n"

    # Show EXIF note in italics — honest disclaimer about what missing EXIF means
    if not metadata["has_exif"]:
        result_text += f"\n_{metadata['exif_note']}_"

    # Gradio expects return values matching the outputs list order:
    # outputs[0] = Markdown → result_text
    # outputs[1] = Image → cam_image
    return result_text, cam_image


# Gradio Interface — simplest Gradio pattern
# fn: function to call on submit
# inputs: single image upload component
# outputs: markdown text + image side by side
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