# Furniture Classifier

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-HuggingFace-yellow.svg)](https://huggingface.co/ArtInSoul/furniture-classifier)
[![Built with Gradio](https://img.shields.io/badge/built%20with-Gradio-ff7c00.svg)](https://www.gradio.app/)

A CNN-based image classifier that recognises 4 categories of indoor furniture from a single photograph — **trained from scratch** in PyTorch (no pretrained backbone).

**Classes:** bed · chair · sofa · table

The interactive demo shows the class probabilities, a **Grad-CAM heatmap** of *where* the model looked, and an **out-of-distribution warning** when the top prediction is weak.

> ▶ **Live demo:** deploy in one step to [Hugging Face Spaces](#deploy) — the `app/app.py` demo is Spaces-ready.

---

## Model architecture

`FurnitureClassifier` is a custom 3-block CNN trained from scratch with PyTorch. Each block uses two convolutional layers with ReLU activations, then a MaxPool to progressively reduce spatial dimensions while increasing feature depth.

```
Input  3 x 64 x 64
  Block 1   Conv(3->60)   -> ReLU -> Conv(60->120, stride=2) -> ReLU -> MaxPool(2) -> 120x16x16
  Block 2   Conv(120->80) -> ReLU -> Conv(80->120)            -> ReLU -> MaxPool(2) -> 120x8x8
  Block 3   Conv(120->80) -> ReLU -> Conv(80->10)             -> ReLU -> MaxPool(2) ->  10x4x4
  Head      Flatten(160)  -> Linear(160, 4)
```

| Param         | Value  |
|---------------|--------|
| hidden_units  | 10     |
| Input size    | 64×64  |
| Optimizer     | SGD    |
| Learning rate | 0.1    |
| Batch size    | 32     |

---

## Results

### On the synthetic test split

Trained for 10 epochs on a T4 GPU (Google Colab):

![Training metrics](assets/training_metrics.png)

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.3658    | 86.1%     | 0.1702   | 94.4%   |
| 5     | 0.0612    | 98.2%     | 0.0750   | 97.6%   |
| **6** | **0.0589**| **98.1%** | **0.0507**| **98.5%** |
| 10    | 0.0381    | 98.8%     | 0.0933   | 96.8%   |

**Best test accuracy: 98.5%** — train and test curves stay close throughout, indicating minimal overfitting.

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

### On real photos (the sim-to-real gap)

The model was trained purely on *synthetic* renders, so it's worth measuring how it holds up on real
photographs. The `sample_images/` folder contains ~190 labelled real photos; evaluating on them is a
single command:

```bash
python -m src.evaluate --data-dir ./sample_images
```

Accuracy drops from **98.5%** (synthetic) to roughly **~54%** on these real photos — a textbook
**sim-to-real domain gap**. The model generalises best on `sofa`/`bed` and struggles most on `chair`.
This is an honest limitation of training on synthetic data alone; closing it would mean fine-tuning on
real images or adding stronger augmentation. `src.evaluate` regenerates the confusion matrix and a
`metrics.json` for any labelled dataset, so these numbers are fully reproducible.

---

## Repository layout

```
classification-model/
├── src/
│   ├── __init__.py     # Public API: FurnitureClassifier, load_model, predict
│   ├── model.py        # FurnitureClassifier definition
│   ├── dataset.py      # Transforms and CLASS_NAMES
│   ├── predict.py      # Inference + confidence (downloads weights from HF Hub)
│   ├── gradcam.py      # Grad-CAM explainability heatmaps
│   ├── evaluate.py     # Accuracy + confusion matrix on a labelled set (CLI)
│   └── train.py        # Training script (CLI)
├── app/
│   └── app.py          # Gradio demo (predictions + Grad-CAM + OOD warning)
├── check_data.py       # Dataset sanity-checker (class balance, corrupt/small images)
├── assets/
│   ├── training_metrics.png
│   └── confusion_matrix.png
├── examples/           # Sample images used by the demo app
├── sample_images/      # Labelled images for spot-checking predictions
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo app

```bash
python app/app.py
```

The app downloads the model weights from HuggingFace Hub on first launch (cached locally afterwards),
then opens a local web UI where you can upload any furniture image and see the predictions.

### 3. Run inference from Python

```python
from PIL import Image
from src.predict import predict

image = Image.open("your_image.jpg")
results = predict(image)          # {"bed": 0.91, "chair": 0.03, ...}
top_class = max(results, key=results.get)
print(top_class, f"{results[top_class]:.1%}")
```

### 4. Retrain the model

```bash
python -m src.train \
  --train-dir ./project_img \
  --test-dir  ./sample_images \
  --epochs    10 \
  --output    models/furniture_classifier.pth
```

---

## Dataset

Trained on [filnow/furniture-synthetic-dataset-30k](https://huggingface.co/datasets/filnow/furniture-synthetic-dataset-30k) — 30,000 synthetic furniture images.
Weights are hosted on [ArtInSoul/furniture-classifier](https://huggingface.co/ArtInSoul/furniture-classifier).

---

## Deploy

The Gradio demo runs as a free [Hugging Face Space](https://huggingface.co/spaces) with no code changes:

1. Create a new Space (SDK: **Gradio**).
2. Push this repo to it, adding a one-line Space config that points at the app:
   set **`app_file: app/app.py`** in the Space's settings (or its `README.md` front-matter).
3. The Space installs `requirements.txt` and pulls the weights from HF Hub on first launch.

Once live, drop the URL into the **Live demo** badge at the top of this README.

---

## License

MIT

