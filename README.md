# ImageTrust-AI

An AI-powered image authenticity checker that detects whether an image is real or AI-generated, and identifies the generator type (GAN, Diffusion, or Other).

## Live Demo

[Try it on HuggingFace Spaces](https://huggingface.co/spaces/SiemonCha/ImageTrust-AI)

## Screenshots

### AI-Generated Image Detection

![Fake Image Result](sample_images/screenshots/fake_result.png)

### Real Image Detection

![Real Image Result](sample_images/screenshots/real_result.png)

---

## Problem Statement

With the rise of AI image generation tools like Stable Diffusion, MidJourney, and DALL-E,
it's becoming increasingly difficult to distinguish real photos from synthetic ones.
ImageTrust-AI helps identify AI-generated images using deep learning — and goes further by identifying which type of AI architecture produced them.

## Features

- Real vs AI-generated classification with confidence score
- **Generator type detection** — identifies GAN, Diffusion, or Other architecture
- Grad-CAM visual explanation — see which regions influenced the decision
- Image metadata extraction and EXIF analysis
- REST API endpoint
- Local Streamlit UI + deployed Gradio interface

## Tech Stack

- Python 3.10
- PyTorch + TorchVision (ResNet18)
- FastAPI
- Streamlit (local) / Gradio (deployed)
- Pillow, OpenCV
- scikit-learn
- pytorch-grad-cam

## Models

### Binary Classifier (Real vs Fake)

- Architecture: ResNet18 (transfer learning, pretrained on ImageNet)
- Dataset: ArtiFact (30k subset - 15k real, 15k fake)
- Validation Accuracy: ~94.5%

### Generator Type Classifier (4-class)

- Architecture: ResNet18 (transfer learning)
- Classes: Real, GAN, Diffusion, Other
- Dataset: ArtiFact (40k - 10k per class, manually curated)
- Test Accuracy: ~94.7%

## Project Structure

```
ImageTrust-AI/
├── app/
│   ├── main.py
│   ├── streamlit_app.py
│   ├── gradio_app.py
│   └── routes/predict.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── transforms.py
│   │   └── generator_loader.py
│   ├── models/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── train_efficientnet.py
│   │   ├── train_cross_validation.py
│   │   ├── train_generator.py
│   │   └── inference.py
│   ├── services/
│   │   ├── predictor.py
│   │   ├── metadata_checker.py
│   │   └── gradcam.py
│   └── utils/
├── notebooks/
│   └── failure_analysis.ipynb
├── saved_models/
├── sample_images/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/SiemonCha/ImageTrust-AI.git
cd ImageTrust-AI
conda create -n imagetrust-ai python=3.10
conda activate imagetrust-ai
pip install -r requirements.txt
```

## Dataset

Download ArtiFact dataset from Kaggle:

```bash
kaggle datasets download -d awsaf49/artifact-dataset
```

Place it outside the repo. Update `DATASET_ROOT` in `src/data/loader.py` and `src/data/generator_loader.py`.

## How to Run Locally

Start the API:

```bash
PYTHONPATH=. uvicorn app.main:app --reload
```

Start the UI (new terminal):

```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```

Or run Gradio directly:

```bash
PYTHONPATH=. python app/gradio_app.py
```

## Results

### Binary Classification (V1)

| Metric              | Score              |
| ------------------- | ------------------ |
| Validation Accuracy | 94.5%              |
| Best Val Loss       | 0.151              |
| Training Epochs     | 5 (early stopping) |

### Generator Type Detection (V4)

| Class       | Precision | Recall   | F1       |
| ----------- | --------- | -------- | -------- |
| Real        | 0.95      | 0.84     | 0.90     |
| GAN         | 1.00      | 1.00     | 1.00     |
| Diffusion   | 0.89      | 0.95     | 0.92     |
| Other       | 0.95      | 1.00     | 0.97     |
| **Overall** | **0.95**  | **0.95** | **0.95** |

### Model Comparison (V2)

| Model           | Val Acc | Val Loss | Epochs |
| --------------- | ------- | -------- | ------ |
| ResNet18        | 94.5%   | 0.151    | 5      |
| EfficientNet-B0 | 91.3%   | 0.207    | 17     |

ResNet18 selected as production model — higher accuracy, faster convergence.

## Cross-Dataset Validation (V3)

- **Seen (train):** Stable Diffusion, StyleGAN2, DDPM
- **Unseen (test):** Glide, Latent Diffusion
- **Train Accuracy:** ~94% | **Unseen Test Accuracy:** ~57%

Significant generalisation gap revealed — model learns generator-specific patterns rather than universal AI artifacts. Consistent with findings in published synthetic image detection research.

## Real-World Testing

| Image                | Expected | Predicted    | Confidence | Correct? |
| -------------------- | -------- | ------------ | ---------- | -------- |
| DALL-E generated     | AI       | Real         | 72.57%     | No       |
| Heavily edited photo | Real     | AI-Generated | 99.78%     | Partial  |
| iPhone camera photo  | Real     | Real         | 99.96%     | Yes      |
| MidJourney generated | AI       | Real         | 99.33%     | No       |
| Screenshot           | Real     | Real         | 100%       | Yes      |

Real images detected correctly. AI images from unseen generators (DALL-E, MidJourney) were misclassified — consistent with V3 cross-dataset validation findings.

## Failure Analysis

See `notebooks/failure_analysis.ipynb` for full analysis.

- Total test failures: 93 / ~3,750 (~2.5% error rate)
- Mean confidence on failures: 76.6%
- High confidence wrong predictions (>90%): 26
- Most common failure: real images with unusual lighting/texture classified as AI

## Limitations

- Generalisation drops on unseen AI generators (DALL-E, MidJourney)
- Missing EXIF data does not prove an image is fake
- Model confidence is not definitive proof
- Generator type classifier uses manually curated labels — may not generalise perfectly

## Future Improvements

- Train on all ArtiFact generators for better generalisation
- Frequency-domain features (FFT) for generator-agnostic detection
- Docker deployment
- Adversarial robustness testing
