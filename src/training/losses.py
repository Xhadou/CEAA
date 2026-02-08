"""Custom loss functions for CAAA training."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ContextConsistencyLoss(nn.Module):
    """Custom loss combining classification, context consistency, and calibration.

    Components:
        1. Classification loss: Standard CrossEntropyLoss.
        2. Context consistency loss: Penalizes predictions contradicting
           available context signals.
        3. Confidence calibration loss: Penalizes high entropy when
           context confidence is high.

    Attributes:
        alpha: Weight for the context consistency loss component.
        beta: Weight for the confidence calibration loss component.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1) -> None:
        """Initializes the ContextConsistencyLoss.

        Args:
            alpha: Weight for context consistency loss.
            beta: Weight for confidence calibration loss.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        context_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss.

        Args:
            logits: Model output logits of shape (batch, n_classes).
            labels: Ground truth labels of shape (batch,).
            context_features: Context features of shape (batch, 5).
                Indices: [event_active, event_expected_impact,
                time_seasonality, recent_deployment, context_confidence].

        Returns:
            Tuple of (total_loss, component_dict) where component_dict has
            keys 'cls_loss', 'consistency_loss', 'calibration_loss'.
        """
        # 1. Classification loss
        cls_loss = self.ce_loss(logits, labels)

        # 2. Context consistency loss
        probs = torch.softmax(logits, dim=-1)
        event_active = context_features[:, 0]  # 1.0 = load event happening

        # When event_active == 1.0, penalize predicting FAULT (class 0)
        fault_prob = probs[:, 0]
        penalty_when_event = event_active * fault_prob

        # When event_active == 0.0, penalize predicting EXPECTED_LOAD (class 1)
        load_prob = probs[:, 1]
        penalty_when_no_event = (1.0 - event_active) * load_prob

        consistency_loss = torch.mean(penalty_when_event + penalty_when_no_event)

        # 3. Confidence calibration loss
        # When context_confidence is high, penalize high entropy (uncertainty)
        context_confidence = context_features[:, 4]
        # Compute entropy of softmax distribution
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch,)
        calibration_loss = torch.mean(context_confidence * entropy)

        # Combined loss
        total_loss = cls_loss + self.alpha * consistency_loss + self.beta * calibration_loss

        components = {
            "cls_loss": cls_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "calibration_loss": calibration_loss.item(),
        }

        return total_loss, components
