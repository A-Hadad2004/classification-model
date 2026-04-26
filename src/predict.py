"""Inference utilities for the furniture classifier."""
from pathlib import Path

import os

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .model import FurnitureClassifier
from .dataset import CLASS_NAMES, inference_transforms

HF_REPO_ID = os.getenv("HF_REPO_ID", "ArtInSoul/furniture-classifier")
HF_FILENAME = "furniture_classifier.pth"

_cached_model: FurnitureClassifier | None = None


def load_model(weights_path: str | Path | None = None) -> tuple[FurnitureClassifier, str]:
    """Load the model weights, downloading from HuggingFace Hub if no local path is given.

    Args:
        weights_path: Path to a local .pth file. If None, downloads from HF Hub.

    Returns:
        (model, device) tuple — model is in eval mode on the selected device.
    """
    global _cached_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _cached_model is not None:
        return _cached_model, device

    if weights_path is None:
        weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

    model = FurnitureClassifier(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _cached_model = model
    return model, device


def predict(image: Image.Image) -> dict[str, float]:
    """Classify a PIL image.

    Args:
        image: RGB PIL image of any size (resized internally to 64×64).

    Returns:
        Dict mapping each class name to its predicted probability (0–1).
    """
    model, device = load_model()
    tensor = inference_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor).squeeze(), dim=0).cpu()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
