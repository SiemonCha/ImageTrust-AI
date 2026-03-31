# ImageTrust-AI 🔍

An AI-powered image authenticity checker that detects whether an image is real or AI-generated.

## Problem Statement

With the rise of AI image generation tools like Stable Diffusion, MidJourney, and DALL-E,
it's becoming increasingly difficult to distinguish real photos from synthetic ones.
ImageTrust-AI helps identify AI-generated images using deep learning.

## Features

- Upload any image and get a real vs AI-generated prediction
- Confidence score with each prediction
- Basic image metadata extraction
- EXIF data analysis
- REST API endpoint
- Clean web interface

## Tech Stack

- Python 3.10
- PyTorch + TorchVision (ResNet18)
- FastAPI
- Streamlit
- Pillow, OpenCV
- scikit-learn

## Model

- Architecture: ResNet18 (transfer learning)
- Dataset: ArtiFact (30k subset - 15k real, 15k fake)
- Validation Accuracy: ~94.5%
- Training: MPS (Apple Silicon)

## Project Structure

```
ImageTrust-AI/
├── app/
│   ├── main.py
│   ├── streamlit_app.py
│   └── routes/predict.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── services/
│   │   ├── predictor.py
│   │   └── metadata_checker.py
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

## How to Run

Start the API:

```bash
PYTHONPATH=. uvicorn app.main:app --reload
```

Start the UI (new terminal):

```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```

## Limitations

- Trained on a subset of ArtiFact dataset
- May not generalise to all AI generators
- Missing EXIF data does not prove an image is fake
- Model confidence is not definitive proof

## Future Improvements (V2)

- Grad-CAM visual explanation
- EfficientNet-B0 comparison
- Cross-dataset validation
- Docker deployment
- Expanded training data

## Results

| Metric              | Score              |
| ------------------- | ------------------ |
| Validation Accuracy | 94.5%              |
| Best Val Loss       | 0.151              |
| Training Epochs     | 5 (early stopping) |
