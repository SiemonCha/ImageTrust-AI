import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="ImageTrust-AI", page_icon="🔍", layout="centered")

st.title("🔍 ImageTrust-AI")
st.subheader("AI Image Authenticity Checker")
st.markdown("Upload an image to check if it's **real** or **AI-generated**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            uploaded_file.seek(0)
            response = requests.post(API_URL, files={"file": uploaded_file})

        if response.status_code == 200:
            data = response.json()
            prediction = data["prediction"]
            metadata = data["metadata"]

            # Prediction result
            st.markdown("---")
            label = prediction["label"]
            confidence = prediction["confidence"]

            if label == "AI-Generated":
                st.error(f"🤖 **{label}** — Confidence: {confidence}%")
            else:
                st.success(f"✅ **{label}** — Confidence: {confidence}%")

            # Metadata
            st.markdown("### 📋 Image Metadata")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Format:** {metadata['format']}")
                st.write(f"**Dimensions:** {metadata['dimensions']}")
            with col2:
                st.write(f"**File Size:** {metadata['file_size_kb']} KB")
                st.write(f"**EXIF Data:** {'Yes' if metadata['has_exif'] else 'No'}")

            if not metadata["has_exif"]:
                st.info(metadata["exif_note"])

            # Grad-CAM
            if "gradcam" in data:
                st.markdown("### 🔥 Grad-CAM Explanation")
                st.caption("Highlighted regions influenced the model's decision most.")
                import base64
                from PIL import Image
                import io
                cam_bytes = base64.b64decode(data["gradcam"])
                cam_image = Image.open(io.BytesIO(cam_bytes))
                st.image(cam_image, width=400)

            # Disclaimer
            st.markdown("---")
            st.warning(f"⚠️ {data['note']}")

        else:
            st.error("Something went wrong. Make sure the API is running.")