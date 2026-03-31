import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.models.model import build_model
from src.data.loader import get_cross_dataset_loaders


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, prec, rec, f1, cm


def train(epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cross_dataset_loaders(batch_size=batch_size)

    model = build_model().to(device)
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=2)

    best_loss = float("inf")
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

        # Validate on unseen generators
        val_loss, acc, prec, rec, f1, _ = evaluate(model, test_loader, device, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Unseen Test Acc: {acc:.4f} | F1: {f1:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), "saved_models/cross_val_best.pth")
            print(f"  -> Best model saved")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\n--- Final Evaluation on Unseen Generators ---")
    _, acc, prec, rec, f1, cm = evaluate(model, test_loader, device, criterion)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    train()