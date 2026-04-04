import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.model import build_efficientnet
from src.data.loader import get_dataloaders


def train(epochs=20, batch_size=32, lr=1e-4):
    """
    Trains EfficientNet-B0 for binary classification — used for V2 model comparison.
    Identical training setup to train.py for fair comparison against ResNet18.

    V2 Comparison Results:
    - ResNet18:       94.5% val acc, 0.151 val loss, converged at epoch 5
    - EfficientNet:   91.3% val acc, 0.207 val loss, converged at epoch 17

    Conclusion: ResNet18 outperforms EfficientNet on this 30k dataset.
    EfficientNet needs more data/epochs to show its advantage.
    This file is kept for reproducibility — not used in production.

    Key difference from train.py:
    - Unfreezes features.8 (last EfficientNet block) instead of layer4
    - EfficientNet uses features.N naming vs ResNet's layerN naming
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_efficientnet().to(device)
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Unfreeze last conv block (features.8) + classifier
    # EfficientNet uses features.0 through features.8 naming convention
    for name, param in model.named_parameters():
        if "features.8" in name or "classifier" in name:
            param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=2)

    best_val_loss = float("inf")
    early_stop_patience = 3
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), "saved_models/efficientnet_best.pth")
            print(f"  -> Best model saved")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    train()