import os
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from src.data.transforms import train_transforms, val_transforms

DATASET_ROOT = "/Users/siemoncha/Desktop/CS/datasets/artifact-dataset"

# Manual class mapping based on architecture knowledge
SOURCE_CLASS_MAP = {
    # Class 0 - Real
    "coco": 0,
    "ffhq": 0,
    "lsun": 0,
    "imagenet": 0,
    "landscape": 0,
    "afhq": 0,
    "celebahq": 0,
    "metfaces": 0,

    # Class 1 - GAN
    "stylegan1": 1,
    "stylegan2": 1,
    "stylegan3": 1,
    "pro_gan": 1,
    "big_gan": 1,
    "star_gan": 1,
    "cycle_gan": 1,
    "gansformer": 1,
    "generative_inpainting": 1,
    "lama": 1,
    "mat": 1,
    "sfhq": 1,
    "cips": 1,
    "projected_gan": 1,
    "gau_gan": 1,

    # Class 2 - Diffusion
    "stable_diffusion": 2,
    "ddpm": 2,
    "glide": 2,
    "latent_diffusion": 2,
    "vq_diffusion": 2,
    "denoising_diffusion_gan": 2,
    "diffusion_gan": 2,
    "palette": 2,

    # Class 3 - Other
    "taming_transformer": 3,
    "face_synthetics": 3,
}

CLASS_NAMES = {0: "Real", 1: "GAN", 2: "Diffusion", 3: "Other"}
MAX_PER_CLASS = 10000


class GeneratorDataset(Dataset):
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

    # Balance classes
    for cls in class_samples:
        class_samples[cls] = class_samples[cls][:MAX_PER_CLASS]
        print(f"Class {cls} ({CLASS_NAMES[cls]}): {len(class_samples[cls])} samples")

    all_samples = []
    for cls in class_samples:
        all_samples.extend(class_samples[cls])

    print(f"Total: {len(all_samples)}")
    return all_samples


def get_generator_dataloaders(batch_size=32):
    all_samples = load_generator_samples()

    # Shuffle before splitting
    random.shuffle(all_samples)

    train_size = int(0.75 * len(all_samples))
    val_size = int(0.125 * len(all_samples))
    test_size = len(all_samples) - train_size - val_size

    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]

    train_set = GeneratorDataset(train_samples, transform=train_transforms)
    val_set = GeneratorDataset(val_samples, transform=val_transforms)
    test_set = GeneratorDataset(test_samples, transform=val_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader