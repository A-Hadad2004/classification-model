"""Training script for the furniture classifier.

Usage:
    python -m src.train --train-dir ./project_img --test-dir ./sample_images
"""
import argparse
import logging
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from .model import FurnitureClassifier
from .dataset import train_transforms, inference_transforms

log = logging.getLogger(__name__)


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def train_epoch(
    model: FurnitureClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
        total_acc += accuracy_fn(y, y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    n = len(loader)
    return total_loss / n, total_acc / n


def eval_epoch(
    model: FurnitureClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            total_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    n = len(loader)
    return total_loss / n, total_acc / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FurnitureClassifier")
    parser.add_argument("--train-dir", required=True, help="ImageFolder-structured training directory")
    parser.add_argument("--test-dir", default="sample_images", help="ImageFolder-structured test directory")
    parser.add_argument("--output", default="models/furniture_classifier.pth", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden-units", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    train_data = datasets.ImageFolder(args.train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(args.test_dir, transform=inference_transforms)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    log.info("Classes (%d): %s", len(train_data.classes), train_data.classes)
    log.info("Train: %d samples | Test: %d samples", len(train_data), len(test_data))

    model = FurnitureClassifier(
        in_channels=3,
        hidden_units=args.hidden_units,
        num_classes=len(train_data.classes),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    start = timer()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
        log.info(
            "Epoch %d/%d | train loss: %.4f  acc: %.1f%% | test loss: %.4f  acc: %.1f%%",
            epoch + 1, args.epochs, train_loss, train_acc, test_loss, test_acc,
        )

    log.info("Training complete in %.1fs", timer() - start)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    log.info("Weights saved to %s", output)


if __name__ == "__main__":
    main()
