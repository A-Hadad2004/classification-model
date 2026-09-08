"""Gradio demo for the furniture classifier.

Run from the project root:
    python app/app.py

Shows the predicted class probabilities, a Grad-CAM heatmap of where the model
looked, and an out-of-distribution warning when the top prediction is weak.
"""
import sys
from pathlib import Path

# Ensure project root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from PIL import Image

from src.dataset import CLASS_NAMES
from src.gradcam import gradcam, overlay_heatmap
from src.predict import CONFIDENCE_THRESHOLD, is_confident, load_model, predict


def classify(image: Image.Image):
    if image is None:
        return {}, None, ""

    probs = predict(image)

    # Grad-CAM overlay for the top class.
    model, device = load_model()
    cam, _ = gradcam(model, image, device)
    heatmap = overlay_heatmap(image, cam)

    top = max(probs, key=probs.get)
    confidence = probs[top]
    if is_confident(probs):
        note = f"### ✅ **{top}** — {confidence:.0%} confidence"
    else:
        classes = ", ".join(CLASS_NAMES)
        note = (
            f"### ⚠️ Low confidence ({confidence:.0%})\n"
            f"This may not be one of the furniture types I know ({classes}). "
            f"The best guess is **{top}**, but I'm not sure."
        )
    return probs, heatmap, note


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a furniture image"),
    outputs=[
        gr.Label(num_top_classes=4, label="Predictions"),
        gr.Image(label="Where the model looked (Grad-CAM)"),
        gr.Markdown(),
    ],
    title="Furniture Classifier",
    description=(
        "Upload an image of indoor furniture and the model will classify it "
        "into one of 4 categories: bed, chair, sofa, or table. "
        "The heatmap shows which regions drove the prediction, and low-confidence "
        f"inputs (top class below {CONFIDENCE_THRESHOLD:.0%}) are flagged as uncertain."
    ),
    article="**Author:** Atara Hadad",
    examples=[
        ["examples/image1684.jpeg"],
        ["examples/image1689.jpeg"],
        ["examples/image1694.jpeg"],
        ["examples/image1716.jpeg"],
    ],
)

if __name__ == "__main__":
    demo.launch()
