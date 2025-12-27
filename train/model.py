"""
The model is a function: N sequential images -> edge weights.
    (B, N, C, H, W) -> (B, 4).
Images are assumed to be consecutive past frames captured, with
some fixed frame step.
"""

import torch
import torch.nn as nn

from constants import *


class DiscamModel(nn.Module):
    def __init__(self):
        super().__init__()

        # (B, C, H, W) -> (B, 1, H/4, W/4)
        # Same conv independently on each frame.
        self.conv_xy = nn.Sequential(
            nn.Conv2d(3, 8, 5, padding=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 1, 3, padding=1),
        )

        # (B, N, H/4, W/4) -> (B, 4, H/16, W/16)
        self.conv_time = nn.Sequential(
            nn.Conv2d(MODEL_NUM_FRAMES, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(8, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            nn.MaxPool2d(2),
        )

        # (B, H * W / 64) -> (B, 4)
        self.head = nn.Sequential(
            nn.Linear(MODEL_INPUT_RES[0] * MODEL_INPUT_RES[1] // 64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        b, n, c, h, w = x.shape

        # Concatenate N.
        x = x.view(b, n * c, h, w)
        x = self.conv_xy(x)

        # Split N.
        x = x.view(b, n, h // 4, w // 4)
        x = self.conv_time(x)

        x = x.view(b, -1)
        x = self.head(x)

        x = torch.tanh(x * MODEL_OUTPUT_TEMP)

        return x
