import os
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from src.data.transforms import train_transforms, val_transforms

# Absolute path to ArtiFact dataset root — update when running on different machine
DATASET_ROOT = "/Users/siemoncha/Desktop/CS/datasets/artifact-dataset"

# Manual source-to-class mapping based on architecture research
# We cannot use the ArtiFact target column directly because target=6 is a mixed bag
# containing both GANs (StyleGAN1, BigGAN) and diffusion models (Stable Diffusion, DDPM)
# Each source was researched individually to determine its correct architecture family
SOURCE_CLASS_MAP = {
    # Class 0 - Real (genuine photographs from public datasets)
    "coco": 0,        # COCO 2017 dataset — real photos
    "ffhq": 0,        # Flickr-Faces-HQ — real face photos
    "lsun": 0,        # Large Scale Scene Understanding — real scenes
    "imagenet": 0,    # ImageNet — real object photos
    "landscape": 0,   # Landscape photos — real
    "afhq": 0,        # Animal Faces HQ — real animal photos
    "celebahq": 0,    # CelebA-HQ — real celebrity photos
    "metfaces": 0,    # MetFaces — real artwork faces

    # Class 1 - GAN (Generative Adversarial Network — generator vs discriminator)
    "stylegan1": 1,             # NVIDIA StyleGAN (2019)
    "stylegan2": 1,             # NVIDIA StyleGAN2 (2020)
    "stylegan3": 1,             # NVIDIA StyleGAN3 (2021)
    "pro_gan": 1,               # Progressive GAN
    "big_gan": 1,               # BigGAN — class-conditional GAN
    "star_gan": 1,              # StarGAN — multi-domain image translation
    "cycle_gan": 1,             # CycleGAN — unpaired image translation
    "gansformer": 1,            # GANsformer — Transformer-based GAN (still GAN family)
    "generative_inpainting": 1, # DeepFill v1/v2 — GAN-based inpainting
    "lama": 1,                  # LaMa — GAN with Fast Fourier Convolutions for inpainting
    "mat": 1,                   # MAT — Mask-Aware Transformer inpainting (uses GAN training)
    "sfhq": 1,                  # SFHQ — StyleGAN2-based synthetic faces
    "cips": 1,                  # CIPS — pixel-wise GAN synthesis
    "projected_gan": 1,         # Projected GAN — faster GAN convergence
    "gau_gan": 1,               # GauGAN — NVIDIA semantic image synthesis GAN

    # Class 2 - Diffusion (iterative denoising from noise to image)
    "stable_diffusion": 2,       # Stability AI Stable Diffusion
    "ddpm": 2,                   # Denoising Diffusion Probabilistic Models
    "glide": 2,                  # OpenAI GLIDE — guided diffusion
    "latent_diffusion": 2,       # Latent Diffusion Models (precursor to Stable Diffusion)
    "vq_diffusion": 2,           # VQ-Diffusion — discrete diffusion
    "denoising_diffusion_gan": 2, # DDG — diffusion with GAN acceleration (diffusion family)
    "diffusion_gan": 2,           # Diffusion-GAN — diffusion-based generator
    "palette": 2,                 # Palette — image-to-image diffusion

    # Class 3 - Other (does not fit GAN or Diffusion paradigm)
    "taming_transformer": 3,  # VQGAN + autoregressive Transformer (codebook-based, not diffusion)
    "face_synthetics": 3,     # Microsoft 3D graphics pipeline — not neural generation at all
}

# Human-readable class names for display and reporting
CLASS_NAMES = {0: "Real", 1: "GAN", 2: "Diffusion", 3: "Other"}

# Maximum samples per class — keeps dataset balanced
MAX_PER_CLASS = 10000


class GeneratorDataset(Dataset):
    """
    PyTorch Dataset for 4-class generator type classification.
    Accepts a pre-built list of (image_path, class_label) tuples.
    Must be defined at module level for multiprocessing compatibility with DataLoader.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_generator_samples():
    """
    Loads image paths and class labels from all sources in SOURCE_CLASS_MAP.
    Caps each class at MAX_PER_CLASS for balanced training.
    Returns a flat list of (image_path, class_label) tuples.
    """
    class_samples = {0: [], 1: [], 2: [], 3: []}

    for source, cls in SOURCE_CLASS_MAP.items():
        csv_path = os.path.join(DATASET_ROOT, source, "metadata.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {source} - no metadata.csv")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(DATASET_ROOT, source, row["image_path"])
            class_samples[cls].append((img_path, cls))

    # Cap each class and report counts
    for cls in class_samples:
        class_samples[cls] = class_samples[cls][:MAX_PER_CLASS]
        print(f"Class {cls} ({CLASS_NAMES[cls]}): {len(class_samples[cls])} samples")

    # Flatten into single list
    all_samples = []
    for cls in class_samples:
        all_samples.extend(class_samples[cls])

    print(f"Total: {len(all_samples)}")
    return all_samples


def get_generator_dataloaders(batch_size=32):
    """
    Returns train, val, and test DataLoaders for 4-class generator classification.
    Split: 75% train / 12.5% val / 12.5% test.

    Critical: shuffle before splitting.
    Without shuffling, samples are ordered by class (all Real, then all GAN, etc.)
    which causes the test set to contain only the last class — 0% accuracy on others.
    """
    all_samples = load_generator_samples()

    # Must shuffle before slicing — samples are ordered by class
    random.shuffle(all_samples)

    train_size = int(0.75 * len(all_samples))
    val_size = int(0.125 * len(all_samples))
    test_size = len(all_samples) - train_size - val_size

    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]

    train_set = GeneratorDataset(train_samples, transform=train_transforms)
    val_set = GeneratorDataset(val_samples, transform=val_transforms)   # no augmentation
    test_set = GeneratorDataset(test_samples, transform=val_transforms)  # no augmentation

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader