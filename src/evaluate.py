"""Evaluate the trained classifier on a labelled dataset.

Computes overall accuracy and a confusion matrix, saves the matrix as a PNG,
and writes a metrics.json summary — making the reported results reproducible
from the repository.

Usage:
    python -m src.evaluate --data-dir ./sample_images
    python -m src.evaluate --data-dir ./test_split --out assets/confusion_matrix.png
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .dataset import CLASS_NAMES, inference_transforms
from .predict import load_model

log = logging.getLogger(__name__)


def evaluate(data_dir: str, batch_size: int = 32) -> dict:
    """Run the model over an ImageFolder dataset and return a metrics dict."""
    model, device = load_model()
    data = datasets.ImageFolder(data_dir, transform=inference_transforms)

    if data.classes != CLASS_NAMES:
        raise ValueError(
            f"Dataset classes {data.classes} do not match the model's "
            f"CLASS_NAMES {CLASS_NAMES}. Each class needs a matching subfolder."
        )

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    n = len(CLASS_NAMES)
    confusion = np.zeros((n, n), dtype=int)

    with torch.inference_mode():
        for X, y in loader:
            preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
            for true, pred in zip(y.numpy(), preds):
                confusion[true, pred] += 1

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    support = confusion.sum(axis=1)
    per_class_acc = {
        CLASS_NAMES[i]: (float(confusion[i, i] / support[i]) if support[i] else None)
        for i in range(n)
    }
    return {
        "num_images": total,
        "accuracy": correct / total if total else 0.0,
        "per_class_accuracy": per_class_acc,
        "support": {CLASS_NAMES[i]: int(support[i]) for i in range(n)},
        "confusion_matrix": confusion.tolist(),
        "classes": CLASS_NAMES,
    }


def save_confusion_matrix(confusion: list[list[int]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(confusion)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the FurnitureClassifier")
    parser.add_argument("--data-dir", default="sample_images", help="ImageFolder-structured dataset")
    parser.add_argument("--out", default="eval_confusion_matrix.png", help="Confusion-matrix PNG path")
    parser.add_argument("--metrics", default="metrics.json", help="Metrics JSON output path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    metrics = evaluate(args.data_dir, batch_size=args.batch_size)
    log.info("Evaluated %d images | accuracy: %.1f%%", metrics["num_images"], metrics["accuracy"] * 100)
    for cls, acc in metrics["per_class_accuracy"].items():
        log.info("  %-8s %s", cls, f"{acc:.1%}" if acc is not None else "n/a")

    save_confusion_matrix(metrics["confusion_matrix"], Path(args.out))
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    log.info("Confusion matrix -> %s | metrics -> %s", args.out, args.metrics)


if __name__ == "__main__":
    main()
