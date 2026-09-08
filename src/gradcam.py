"""Grad-CAM explainability for the furniture classifier.

Produces a heatmap over the input image highlighting the regions that most
influenced the prediction, using gradients of the target class score with
respect to the last convolutional layer's feature maps.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import IMAGE_SIZE, inference_transforms
from .model import FurnitureClassifier


def _target_layer(model: FurnitureClassifier) -> torch.nn.Module:
    """The last convolutional layer — its feature maps carry the most semantics."""
    return model.block_3[2]


def gradcam(
    model: FurnitureClassifier,
    image: Image.Image,
    device: str = "cpu",
    class_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compute a Grad-CAM heatmap for ``image``.

    Args:
        model: a trained FurnitureClassifier.
        image: input PIL image (any size; resized internally).
        device: torch device string.
        class_idx: class to explain. Defaults to the model's top prediction.

    Returns:
        ``(cam, class_idx)`` where ``cam`` is a float array in [0, 1] of shape
        ``(IMAGE_SIZE, IMAGE_SIZE)`` and ``class_idx`` is the explained class.
    """
    model.eval()
    tensor = inference_transforms(image.convert("RGB")).unsqueeze(0).to(device)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = _target_layer(model)

    def forward_hook(_module, _inputs, output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        logits = model(tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        model.zero_grad()
        logits[0, class_idx].backward()
    finally:
        handle.remove()

    acts = activations[0].detach()[0]          # (C, H, W)
    grads = gradients[0].detach()[0]           # (C, H, W)
    weights = grads.mean(dim=(1, 2))           # (C,) — importance of each channel
    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))   # (H, W)

    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = F.interpolate(
        cam[None, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )[0, 0]
    return cam.cpu().numpy(), class_idx


def _colorize(cam: np.ndarray) -> Image.Image:
    """Map a [0, 1] array to an RGB jet-style heatmap (no matplotlib dependency)."""
    x = np.clip(cam, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype("uint8")
    return Image.fromarray(rgb)


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.5, display_size: int = 256
) -> Image.Image:
    """Blend a Grad-CAM heatmap over the original image for display."""
    base = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    heat = _colorize(cam)
    blended = Image.blend(base, heat, alpha)
    return blended.resize((display_size, display_size), Image.BILINEAR)
