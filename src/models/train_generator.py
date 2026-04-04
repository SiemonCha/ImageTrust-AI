import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from src.data.generator_loader import get_generator_dataloaders, CLASS_NAMES


def build_multiclass_model(num_classes=4, pretrained=True):
    """
    Builds ResNet18 for 4-class generator type classification.
    Same base architecture as binary classifier but with multi-class output head.

    Key difference from build_model() in model.py:
    - Output: num_classes neurons (4) instead of 1
    - Loss: CrossEntropyLoss instead of BCEWithLogitsLoss
    - Prediction: argmax over 4 outputs instead of sigmoid threshold

    Classes: Real (0), GAN (1), Diffusion (2), Other (3)
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last conv block + classifier for fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)  # 4 output neurons for 4 classes
    )
    return model


def evaluate(model, loader, device):
    """
    Returns predicted and true labels for a dataloader.
    Used for both per-epoch val accuracy and final test evaluation.
    Multi-class version uses argmax instead of sigmoid threshold.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # argmax picks the class with highest logit score
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def train(epochs=15, batch_size=32, lr=1e-4):
    """
    Trains the 4-class generator type classifier.

    Key differences from binary train.py:
    - CrossEntropyLoss instead of BCEWithLogitsLoss
      (CrossEntropy handles multi-class, BCE is binary only)
    - Labels are long integers (class indices), not floats
    - No unsqueeze(1) needed — labels shape is [batch] not [batch, 1]
    - Prediction uses argmax over 4 outputs instead of sigmoid threshold
    - Saves to generator_model.pth (used in production)

    V4 Results (40k samples, 10k per class):
    - Overall test accuracy: 94.7%
    - GAN detection: perfect (1.00 F1)
    - Real images: hardest class (0.84 recall)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_generator_dataloaders(batch_size=batch_size)
    model = build_multiclass_model().to(device)

    # CrossEntropyLoss for multi-class — combines log_softmax + NLLLoss
    # Expects raw logits as input (no softmax needed before this)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=2)

    best_val_acc = 0
    early_stop_patience = 3
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)  # long integers, not floats

            optimizer.zero_grad()
            outputs = model(images)      # shape: [batch, 4]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
        # Pass 1-val_acc because scheduler expects a loss (lower=better)
        scheduler.step(1 - val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), "saved_models/generator_model.pth")
            print(f"  -> Best model saved")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation with full classification report
    print("\n--- Final Evaluation ---")
    test_preds, test_labels = evaluate(model, test_loader, device)
    test_acc = sum(p == l for p, l in zip(test_preds, test_labels)) / len(test_labels)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=list(CLASS_NAMES.values())))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    train()