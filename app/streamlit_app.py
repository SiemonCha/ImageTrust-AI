import streamlit as st
import requests
from PIL import Image
import io

# Local FastAPI endpoint — must be running before using this UI
# Start with: PYTHONPATH=. uvicorn app.main:app --reload
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="ImageTrust-AI", page_icon="🔍", layout="centered")

st.title("🔍 ImageTrust-AI")
st.subheader("AI Image Authenticity Checker")
st.markdown("Upload an image to check if it's **real** or **AI-generated**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show preview before analysis
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            # Reset file pointer — Streamlit moves it after image preview
            uploaded_file.seek(0)
            # Send image to FastAPI as multipart form data
            response = requests.post(API_URL, files={"file": uploaded_file})

        if response.status_code == 200:
            data = response.json()
            prediction = data["prediction"]
            metadata = data["metadata"]

            st.markdown("---")
            label = prediction["label"]
            confidence = prediction["confidence"]

            # Color-coded result — red for AI, green for real
            if label == "AI-Generated":
                st.error(f"🤖 **{label}** — Confidence: {confidence}%")
            else:
                st.success(f"✅ **{label}** — Confidence: {confidence}%")

            # Generator type detection section (V4 feature)
            if "generator" in data:
                gen = data["generator"]
                st.markdown("### 🔬 Generator Type Detection")
                st.write(f"**Detected:** {gen['generator_type']} — {gen['confidence']}%")
                st.write("**Class Probabilities:**")
                # Progress bars for each class probability
                for cls, prob in gen["class_probabilities"].items():
                    st.progress(int(prob), text=f"{cls}: {prob}%")

            # Image metadata panel
            st.markdown("### 📋 Image Metadata")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Format:** {metadata['format']}")
                st.write(f"**Dimensions:** {metadata['dimensions']}")
            with col2:
                st.write(f"**File Size:** {metadata['file_size_kb']} KB")
                st.write(f"**EXIF Data:** {'Yes' if metadata['has_exif'] else 'No'}")

            # EXIF note — shown when no EXIF found (common in AI images)
            if not metadata["has_exif"]:
                st.info(metadata["exif_note"])

            # Grad-CAM heatmap section
            if "gradcam" in data:
                st.markdown("### 🔥 Grad-CAM Explanation")
                st.caption("Highlighted regions influenced the model's decision most.")
                import base64
                # Decode base64 string back to image bytes
                # API sends heatmap as base64 because JSON can't contain binary data
                cam_bytes = base64.b64decode(data["gradcam"])
                cam_image = Image.open(io.BytesIO(cam_bytes))
                st.image(cam_image, width=400)

            # Always show disclaimer — model is probabilistic, not definitive
            st.markdown("---")
            st.warning(f"⚠️ {data['note']}")

        else:
            st.error("Something went wrong. Make sure the API is running.")