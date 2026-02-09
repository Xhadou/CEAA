"""Feature encoder module for CAAA model."""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureEncoder(nn.Module):
    """Encodes the 36-dimensional feature vector into a dense hidden
    representation via a multi-layer perceptron.

    Note: despite operating on features derived from time series, this
    module does not perform sequential or temporal processing — temporal
    patterns are captured in the feature extraction stage.

    Attributes:
        layers: Sequential stack of linear transformation layers.
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the FeatureEncoder.

        Args:
            input_dim: Dimensionality of input feature vectors.
            hidden_dim: Dimensionality of hidden layers.
            num_layers: Number of Linear + ReLU + Dropout blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers: list[nn.Module] = []
        # First layer: input_dim -> hidden_dim
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        # Additional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)
        logger.debug(
            "FeatureEncoder initialized: input_dim=%d, hidden_dim=%d, num_layers=%d",
            input_dim, hidden_dim, num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Encoded tensor of shape (batch_size, hidden_dim).
        """
        return self.layers(x)
