import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split
from src.data.transforms import train_transforms, val_transforms

# Absolute path to the ArtiFact dataset root on local machine
# Update this path when running on a different machine (e.g. Kaggle)
DATASET_ROOT = "/Users/siemoncha/Desktop/CS/datasets/artifact-dataset"

# Sources where target == 0 (real photographs)
REAL_SOURCES = ["coco", "ffhq", "lsun", "imagenet", "landscape", "afhq"]

# Sources where target != 0 (AI-generated images)
FAKE_SOURCES = ["stable_diffusion", "stylegan2", "ddpm", "glide", "latent_diffusion"]

# Maximum samples per class — keeps dataset balanced and manageable for training
MAX_PER_CLASS = 15000  # 15k real + 15k fake = 30k total


class ArtiFact(Dataset):
    """
    PyTorch Dataset for the ArtiFact dataset.
    Loads image paths and binary labels (0=real, 1=fake) from metadata CSVs.
    Balances real and fake samples up to MAX_PER_CLASS each.
    """

    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []  # list of (image_path, label) tuples
        self._load_metadata()

    def _load_metadata(self):
        """
        Reads metadata.csv from each source folder.
        ArtiFact stores label info in CSVs, not folder names.
        target == 0 means real, anything else means AI-generated.
        """
        real, fake = [], []

        for source in REAL_SOURCES + FAKE_SOURCES:
            csv_path = os.path.join(DATASET_ROOT, source, "metadata.csv")
            if not os.path.exists(csv_path):
                print(f"Skipping {source} - no metadata.csv")
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # image_path in CSV is relative to the source folder
                img_path = os.path.join(DATASET_ROOT, source, row["image_path"])
                if row["target"] == 0:
                    real.append((img_path, 0))
                else:
                    fake.append((img_path, 1))

        # Cap both classes to MAX_PER_CLASS to ensure balanced training
        real = real[:MAX_PER_CLASS]
        fake = fake[:MAX_PER_CLASS]
        self.samples = real + fake

        print(f"Real: {len(real)} | Fake: {len(fake)} | Total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load image from disk, apply transforms, return (image_tensor, label)."""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SampleDataset(Dataset):
    """
    Lightweight Dataset that accepts a pre-built list of (image_path, label) tuples.
    Used for cross-dataset validation where we need custom source splits.
    Must be defined at module level (not inside a function) for multiprocessing compatibility.
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


def get_dataloaders(batch_size=32):
    """
    Returns train, val, and test DataLoaders for binary classification.
    Split: 75% train / 12.5% val / 12.5% test.
    Val and test use val_transforms (no augmentation) for accurate evaluation.
    """
    dataset = ArtiFact(transform=train_transforms)

    train_size = int(0.75 * len(dataset))
    val_size = int(0.125 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Override transforms for val/test — no augmentation, only resize + normalize
    val_set.dataset.transform = val_transforms
    test_set.dataset.transform = val_transforms

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def get_cross_dataset_loaders(batch_size=32):
    """
    Returns train and test DataLoaders for cross-dataset validation (V3).
    Trains on seen generators, tests on unseen generators to measure generalisation.

    Seen (train): stable_diffusion, stylegan2, ddpm
    Unseen (test): glide, latent_diffusion

    This reveals whether the model learns generator-specific patterns
    or universal AI artifacts.
    """
    # Generators the model will see during training
    SEEN_FAKE = ["stable_diffusion", "stylegan2", "ddpm"]
    # Generators the model will never see — used only for testing
    UNSEEN_FAKE = ["glide", "latent_diffusion"]

    def load_sources(real_sources, fake_sources, max_per_class=10000):
        """Helper to load image paths from specified sources."""
        real, fake = [], []
        for source in real_sources:
            csv_path = os.path.join(DATASET_ROOT, source, "metadata.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(DATASET_ROOT, source, row["image_path"])
                if row["target"] == 0:
                    real.append((img_path, 0))
        for source in fake_sources:
            csv_path = os.path.join(DATASET_ROOT, source, "metadata.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(DATASET_ROOT, source, row["image_path"])
                if row["target"] != 0:
                    fake.append((img_path, 1))
        real = real[:max_per_class]
        fake = fake[:max_per_class]
        return real + fake

    from torch.utils.data import DataLoader

    train_samples = load_sources(REAL_SOURCES, SEEN_FAKE)
    test_samples = load_sources(REAL_SOURCES, UNSEEN_FAKE, max_per_class=5000)

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    train_set = SampleDataset(train_samples, transform=train_transforms)
    test_set = SampleDataset(test_samples, transform=val_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader