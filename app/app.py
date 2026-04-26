"""Gradio demo for the furniture classifier.

Run from the project root:
    python app/app.py
"""
import sys
from pathlib import Path

# Ensure project root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from PIL import Image

from src.predict import predict


def classify(image: Image.Image) -> dict[str, float]:
    if image is None:
        return {}
    return predict(image)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a furniture image"),
    outputs=gr.Label(num_top_classes=8, label="Predictions"),
    title="Furniture Classifier",
    description=(
        "Upload an image of indoor furniture and the model will classify it "
        "into one of 8 categories: bed, chair, closet, dresser, "
        "library, mirror, sofa, or table."
    ),
    examples=[
        ["try/image1684.jpeg"],
        ["try/image1689.jpeg"],
        ["try/image1694.jpeg"],
        ["try/image1716.jpeg"],
    ],
)

if __name__ == "__main__":
    demo.launch()
