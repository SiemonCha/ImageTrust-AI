# ImageTrust-AI

An AI-powered image authenticity checker that detects whether an image is real or AI-generated.

## Live Demo

[Try it on HuggingFace Spaces](https://huggingface.co/spaces/SiemonCha/ImageTrust-AI)

## Problem Statement

With the rise of AI image generation tools like Stable Diffusion, MidJourney, and DALL-E,
it's becoming increasingly difficult to distinguish real photos from synthetic ones.
ImageTrust-AI helps identify AI-generated images using deep learning.

## Features

- Upload any image and get a real vs AI-generated prediction
- Confidence score with each prediction
- Grad-CAM visual explanation — see which regions influenced the decision
- Basic image metadata extraction
- EXIF data analysis
- REST API endpoint
- Clean web interface

## Tech Stack

- Python 3.10
- PyTorch + TorchVision (ResNet18)
- FastAPI
- Streamlit (local) / Gradio (deployed)
- Pillow, OpenCV
- scikit-learn
- pytorch-grad-cam

## Model

- Architecture: ResNet18 (transfer learning, pretrained on ImageNet)
- Dataset: ArtiFact (30k subset - 15k real, 15k fake)
- Validation Accuracy: ~94.5%
- Training: MPS (Apple Silicon)

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
│   │   └── transforms.py
│   ├── models/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── train_efficientnet.py
│   │   ├── train_cross_validation.py
│   │   └── inference.py
│   ├── services/
│   │   ├── predictor.py
│   │   ├── metadata_checker.py
│   │   └── gradcam.py
│   └── utils/
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

Place it outside the repo. Update `DATASET_ROOT` in `src/data/loader.py`.

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

| Metric              | Score              |
| ------------------- | ------------------ |
| Validation Accuracy | 94.5%              |
| Best Val Loss       | 0.151              |
| Training Epochs     | 5 (early stopping) |

## Model Comparison (V2)

| Model           | Val Acc | Val Loss | Epochs |
| --------------- | ------- | -------- | ------ |
| ResNet18        | 94.5%   | 0.151    | 5      |
| EfficientNet-B0 | 91.3%   | 0.207    | 17     |

ResNet18 was selected as the production model — higher accuracy, faster convergence.

## Cross-Dataset Validation (V3)

To evaluate generalisation, the model was trained on seen generators and tested on unseen generators.

- **Seen (train):** Stable Diffusion, StyleGAN2, DDPM
- **Unseen (test):** Glide, Latent Diffusion
- **Train Accuracy:** ~94% | **Unseen Test Accuracy:** ~57%

Cross-dataset validation revealed a significant generalisation gap — the model learns generator-specific patterns rather than universal AI artifacts. This is consistent with findings in published synthetic image detection research.

## Real-World Testing

| Image                | Expected | Predicted    | Confidence | Correct? |
| -------------------- | -------- | ------------ | ---------- | -------- |
| DALL-E generated     | AI       | Real         | 72.57%     | No       |
| Heavily edited photo | Real     | AI-Generated | 99.78%     | n/a      |
| iPhone camera photo  | Real     | Real         | 99.96%     | Yes      |
| MidJourney generated | AI       | Real         | 99.33%     | No       |
| Screenshot           | Real     | Real         | 100%       | Yes      |

Real images detected correctly. AI images from unseen generators (DALL-E, MidJourney) were misclassified — consistent with V3 cross-dataset validation findings.

## Limitations

- Trained on a subset of ArtiFact dataset
- Generalisation drops significantly on unseen AI generators
- Missing EXIF data does not prove an image is fake
- Model confidence is not definitive proof

## Future Improvements

- Train on all ArtiFact generators for better generalisation
- Docker deployment
- Expanded augmentation strategies
- Test on MidJourney and DALL-E 3 images

## Screenshots

### Real Image Detection

![Real Image Result](sample_images/screenshots/real_result.png)

### AI-Generated Image Detection

![Fake Image Result](sample_images/screenshots/fake_result.png)
