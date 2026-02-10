"""Main CAAA model combining feature encoding with context integration."""

import logging
from typing import Tuple

import torch
import torch.nn as nn

from src.features.feature_schema import CONTEXT_START, CONTEXT_END
from src.models.context_module import ContextIntegrationModule
from src.models.feature_encoder import FeatureEncoder

logger = logging.getLogger(__name__)


class CAAAModel(nn.Module):
    """Context-Aware Anomaly Attribution model.

    Combines MLP-based feature encoding with context-aware integration
    to produce classification logits.

    Attributes:
        feature_encoder: MLP that projects raw features into a dense
            hidden representation.
        context_module: Integrates context features with the encoded
            representation via attention and confidence gating.
        classifier: Classification head producing logits.
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 64,
        context_dim: int = 5,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the CAAAModel.

        Args:
            input_dim: Dimensionality of input feature vectors.
            hidden_dim: Hidden dimensionality for feature encoder.
            context_dim: Number of context features.
            n_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.feature_encoder = FeatureEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.context_module = ContextIntegrationModule(
            temporal_dim=hidden_dim,
            context_dim=context_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        logger.info(
            "CAAAModel initialized: input_dim=%d, hidden_dim=%d, n_classes=%d",
            input_dim, hidden_dim, n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full model.

        Args:
            x: Input tensor of shape (batch, input_dim) containing
                the full 36-dim feature vector.

        Returns:
            Logits tensor of shape (batch, n_classes).
        """
        # Split context features using centralized schema
        context_features = x[:, CONTEXT_START:CONTEXT_END]

        # Feature encoding of full feature vector
        encoded_features = self.feature_encoder(x)

        # Context integration
        integrated = self.context_module(encoded_features, context_features)

        # Classification
        logits = self.classifier(integrated)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate embeddings before the classifier head.

        These embeddings live in the ``hidden_dim``-dimensional space
        produced by the feature encoder + context integration module.
        They are used by the supervised contrastive loss to build a
        representation space where same-class samples cluster together.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Embedding tensor of shape (batch, hidden_dim).
        """
        context_features = x[:, CONTEXT_START:CONTEXT_END]
        encoded_features = self.feature_encoder(x)
        integrated = self.context_module(encoded_features, context_features)
        return integrated

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class predictions (argmax of softmax).

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Predicted class indices of shape (batch,).
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            return torch.argmax(probabilities, dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor, confidence_threshold: float = 0.6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns predictions with UNKNOWN class for low-confidence predictions.

        The model produces 2-class predictions (FAULT/EXPECTED_LOAD). When the
        max softmax probability is below ``confidence_threshold``, the prediction
        is changed to class 2 (UNKNOWN). UNKNOWN is a post-hoc decision, not
        a trained class.

        Args:
            x: Input tensor of shape (batch, input_dim).
            confidence_threshold: Minimum softmax probability to accept a
                prediction. Predictions below this become UNKNOWN (class 2).

        Returns:
            Tuple of (predictions, confidences) where:
                - predictions: class indices of shape (batch,), with values
                  in {0, 1, 2} (0=FAULT, 1=EXPECTED_LOAD, 2=UNKNOWN).
                - confidences: max softmax probability for each sample,
                  shape (batch,).

        See Also:
            :meth:`src.models.classifier.AnomalyClassifier.predict_with_confidence`
                for the sklearn-based equivalent using string labels.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)
            # Set low-confidence predictions to UNKNOWN (class 2)
            predictions[confidences < confidence_threshold] = 2
        return predictions, confidences
