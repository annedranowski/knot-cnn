import torch
from torch import nn
from torch.optim import Adam
from pathlib import Path
from .models import KnotCNN
from .data.dataset import get_dataloaders
from .eval import evaluate

def train_model(data_root, num_epochs=10, lr=1e-3, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl = get_dataloaders(data_root, batch_size=batch_size)

    model = KnotCNN(num_classes=len(train_dl.dataset.classes)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate(model, val_dl, device=device)
        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {running_loss/len(train_dl):.4f} "
              f"Val Acc: {val_metrics['accuracy']:.2%}")

    return model

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_model(args.data_root, num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
