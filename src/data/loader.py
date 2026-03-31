import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split
from src.data.transforms import train_transforms, val_transforms

DATASET_ROOT = "/Users/siemoncha/Desktop/CS/datasets/artifact-dataset"

REAL_SOURCES = ["coco", "ffhq", "lsun", "imagenet", "landscape", "afhq"]
FAKE_SOURCES = ["stable_diffusion", "stylegan2", "ddpm", "glide", "latent_diffusion"]

MAX_PER_CLASS = 15000  # 15k real + 15k fake = 30k total


class ArtiFact(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        self._load_metadata()

    def _load_metadata(self):
        real, fake = [], []

        for source in REAL_SOURCES + FAKE_SOURCES:
            csv_path = os.path.join(DATASET_ROOT, source, "metadata.csv")
            if not os.path.exists(csv_path):
                print(f"Skipping {source} - no metadata.csv")
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(DATASET_ROOT, source, row["image_path"])
                if row["target"] == 0:
                    real.append((img_path, 0))
                else:
                    fake.append((img_path, 1))

        # Balance and subsample
        real = real[:MAX_PER_CLASS]
        fake = fake[:MAX_PER_CLASS]
        self.samples = real + fake

        print(f"Real: {len(real)} | Fake: {len(fake)} | Total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(batch_size=32):
    dataset = ArtiFact(transform=train_transforms)

    train_size = int(0.75 * len(dataset))
    val_size = int(0.125 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Val and test use val_transforms
    val_set.dataset.transform = val_transforms
    test_set.dataset.transform = val_transforms

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader