"""
FurnitureClassifier — 3-block custom CNN for 4-class indoor furniture recognition.

Architecture (default hidden_units=10, input 3×64×64):
  Block 1 : Conv(3→60,  3×3, p=1)       → ReLU
            Conv(60→120, 3×3, s=2, p=1) → ReLU → MaxPool(2) → 120×16×16
  Block 2 : Conv(120→80, 3×3, p=1)      → ReLU
            Conv(80→120, 3×3, p=1)      → ReLU → MaxPool(2) → 120×8×8
  Block 3 : Conv(120→80, 3×3, p=1)      → ReLU
            Conv(80→10,  3×3, p=1)      → ReLU → MaxPool(2) → 10×4×4
  Head    : Flatten(160) → Linear(160, 4)
"""
import torch
from torch import nn


class FurnitureClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_units: int = 10, num_classes: int = 4):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_units * 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 6, hidden_units * 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units * 12, hidden_units * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 8, hidden_units * 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 12, hidden_units * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 8, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 4 * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return self.classifier(x)
