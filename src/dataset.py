"""Transforms and class definitions for the furniture classifier."""
from torchvision import transforms

# Class order must match the alphabetical sort used by ImageFolder during training.
# Uppercase names (Library, Table) sort before lowercase names in Python.
CLASS_NAMES = ["Library", "Table", "bed", "chair", "closet", "dresser", "mirror", "sofa"]

IMAGE_SIZE = 64

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

inference_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
