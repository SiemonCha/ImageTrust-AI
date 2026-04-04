import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.model import build_model
from src.data.loader import get_dataloaders


def train(epochs=10, batch_size=32, lr=1e-3):
    """
    Trains the binary ResNet18 classifier (Real vs AI-Generated).
    Run from project root: PYTHONPATH=. python src/models/train.py

    Key decisions:
    - Only layer4 and fc are unfrozen — lower layers keep ImageNet features
    - BCEWithLogitsLoss for binary classification (applies sigmoid internally)
    - Adam optimizer with lr=1e-4 (low lr to avoid destroying pretrained weights)
    - ReduceLROnPlateau halves lr when val loss plateaus
    - Early stopping after 3 epochs without improvement
    - Saves best model checkpoint based on val loss (not val accuracy)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)

    # Unfreeze last conv block + classifier head
    # Lower layers (layer1-3) stay frozen — their ImageNet features are still useful
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # BCEWithLogitsLoss = Binary Cross Entropy + sigmoid in one numerically stable operation
    # More stable than applying sigmoid manually then using BCELoss
    criterion = nn.BCEWithLogitsLoss()

    # Only pass parameters that require gradients — frozen layers excluded
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Reduce lr by factor 0.1 when val loss doesn't improve for 2 epochs
    scheduler = ReduceLROnPlateau(optimizer, patience=2)

    best_val_loss = float("inf")
    early_stop_patience = 3
    no_improve_count = 0

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()  # enables dropout
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            # unsqueeze(1) reshapes labels from [batch] to [batch, 1] to match output shape
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()           # clear gradients from previous batch
            outputs = model(images)         # forward pass — raw logits
            loss = criterion(outputs, labels)
            loss.backward()                 # compute gradients
            optimizer.step()               # update weights

            train_loss += loss.item()
            # sigmoid converts logit to probability, threshold at 0.5
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()   # disables dropout for deterministic evaluation
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():  # no gradient tracking needed for validation
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
        scheduler.step(avg_val_loss)  # adjust lr based on val loss

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save checkpoint only when val loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print(f"  -> Best model saved")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    train()