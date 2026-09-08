# Furniture Classifier

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-HuggingFace-yellow.svg)](https://huggingface.co/ArtInSoul/furniture-classifier)
[![Built with Gradio](https://img.shields.io/badge/built%20with-Gradio-ff7c00.svg)](https://www.gradio.app/)

> **A personal learning project.** I built this to teach myself how to make an image
> classifier *from scratch* — designing the network layer by layer, wrangling the data,
> training it, figuring out how to tell whether it was any good, and shipping a small demo.
> I learned by experimenting and getting things wrong, not by following a recipe. The
> trained model isn't meant to be production-grade — it's the trail I left while learning.
> **The result was never the goal; the learning was.**

A CNN that recognises 4 categories of indoor furniture — **bed · chair · sofa · table** — from a
single photograph, **trained from scratch** in PyTorch (no pretrained backbone).

Run `python app/app.py` for a local Gradio demo that shows the class probabilities and an
out-of-distribution warning when the top guess is weak.

---

## What I set out to learn

- How a convolutional network is actually assembled and debugged — building my *own* model rather than calling a pretrained one.
- The whole loop end to end: **data → training → evaluation → inference → a usable demo**.
- How to judge whether a model is genuinely good — and what its headline numbers quietly hide.

---

## Model architecture

`FurnitureClassifier` is a custom 3-block CNN. Each block uses two convolutional layers with ReLU
activations, then a MaxPool to progressively shrink the spatial dimensions while increasing feature depth.

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
| Optimizer     | Adam   |
| Learning rate | 0.001  |
| Weight decay  | 1e-4   |
| Batch size    | 32     |

These are the settings I landed on by trying things and seeing what happened — the architecture and
hyperparameters were a big part of the experimentation, not something handed to me.

---

## Results — and what they taught me

### On the synthetic test split

Trained for 10 epochs on a T4 GPU (Google Colab):

![Training metrics](assets/training_metrics.png)

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.3658    | 86.1%     | 0.1702   | 94.4%   |
| 5     | 0.0612    | 98.2%     | 0.0750   | 97.6%   |
| **6** | **0.0589**| **98.1%** | **0.0507**| **98.5%** |
| 10    | 0.0381    | 98.8%     | 0.0933   | 96.8%   |

The runs reached ~98.5% on the synthetic test split, and watching the train and test curves stay
close together was my first real lesson in what *healthy* training looks like versus overfitting.

![Confusion Matrix](assets/confusion_matrix.png)

### On real photos — the lesson that mattered most

98.5% looked great on paper, so I tested the model on real photographs (the `sample_images/` folder,
~190 labelled real photos):

```bash
python -m src.evaluate --data-dir ./sample_images
```

Accuracy fell to roughly **~54%**. Watching a model that looked excellent collapse on real images was
the single most useful thing I took from this project — a first-hand encounter with the **sim-to-real
domain gap**: a network trained only on synthetic renders doesn't transfer to real photos for free.
`src/evaluate.py` reproduces these numbers (accuracy, confusion matrix, `metrics.json`) on any labelled
dataset, so the finding is something you can re-run rather than take on faith.

---

## What I took away

- **Building a CNN by hand** — how the layers, channel counts, and spatial sizes fit together, and how to debug them when the shapes don't line up.
- **Clean training curves ≠ a good model.** You have to evaluate on data that matches how the model will actually be used.
- **The sim-to-real gap is real and large.** Synthetic data got me most of the way in training and almost nowhere on real photos.
- **Data quality matters**, so I wrote `check_data.py` to catch corrupt, tiny, or imbalanced classes before training.
- **A model should be able to say "I'm not sure"** — hence the confidence / out-of-distribution flag in the demo.
- **Shipping the whole thing** — reproducible evaluation, a Gradio demo, and weights hosted on the HF Hub — was its own set of lessons.

---

## Repository layout

```
classification-model/
├── src/
│   ├── __init__.py     # Public API: FurnitureClassifier, load_model, predict
│   ├── model.py        # FurnitureClassifier definition
│   ├── dataset.py      # Transforms and CLASS_NAMES
│   ├── predict.py      # Inference + confidence (downloads weights from HF Hub)
│   ├── evaluate.py     # Accuracy + confusion matrix on a labelled set (CLI)
│   └── train.py        # Training script (CLI)
├── app/
│   └── app.py          # Gradio demo (predictions + OOD warning)
├── check_data.py       # Dataset sanity-checker (class balance, corrupt/small images)
├── assets/
│   ├── training_metrics.png
│   └── confusion_matrix.png
├── examples/           # Sample images used by the demo app
├── sample_images/      # Labelled real photos for spot-checking predictions
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
then opens a local web UI where you can upload a furniture image and see the predictions.

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

## License

MIT
