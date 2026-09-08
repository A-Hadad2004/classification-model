"""Furniture Classifier — public API."""
from .model import FurnitureClassifier
from .predict import load_model, predict

__all__ = ["FurnitureClassifier", "load_model", "predict"]
