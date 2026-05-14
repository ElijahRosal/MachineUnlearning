from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """Small classifier for the 2D synthetic unlearning experiment."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        # A compact MLP is enough for two Gaussian blobs and keeps the demo fast.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Return raw logits; CrossEntropyLoss applies softmax internally.
        return self.net(inputs)
