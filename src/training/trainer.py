"""Trainer for the CAAA model."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models import CAAAModel

logger = logging.getLogger(__name__)


class CAAATrainer:
    """Trainer for the CAAA model.

    Handles training, evaluation, and prediction for the CAAAModel
    using PyTorch with Adam optimizer and CrossEntropyLoss.

    Attributes:
        model: The CAAAModel instance.
        device: Device to run computations on.
        optimizer: Adam optimizer.
        criterion: CrossEntropyLoss criterion.
    """

    def __init__(
        self,
        model: CAAAModel,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        """Initializes the CAAATrainer.

        Args:
            model: The CAAAModel to train.
            learning_rate: Learning rate for the Adam optimizer.
            weight_decay: Weight decay (L2 regularization) for the optimizer.
            device: Device to run computations on ('cpu' or 'cuda').
        """
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """Trains the CAAA model.

        Args:
            X_train: Training features of shape (n_samples, input_dim).
            y_train: Training labels of shape (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation labels.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            early_stopping_patience: Number of epochs without val_loss
                improvement before stopping.

        Returns:
            Dictionary with 'train_loss' and optionally 'val_loss' lists.
        """
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.device)

        history: Dict[str, List[float]] = {"train_loss": []}
        if has_val:
            history["val_loss"] = []

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        n_samples = X_train_t.shape[0]

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            if has_val:
                val_loss = self._compute_loss(X_val_t, y_val_t)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1,
                        early_stopping_patience,
                    )
                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                    break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.4f}"
                if has_val:
                    msg += f" - val_loss: {history['val_loss'][-1]:.4f}"
                logger.info(msg)

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluates the model on the given data.

        Args:
            X: Feature array of shape (n_samples, input_dim).
            y: Label array of shape (n_samples,).

        Returns:
            Dictionary with 'loss' and 'accuracy'.
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            loss = self.criterion(logits, y_t).item()
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == y_t).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels.

        Args:
            X: Feature array of shape (n_samples, input_dim).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model.predict(X_t)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probabilities.

        Args:
            X: Feature array of shape (n_samples, input_dim).

        Returns:
            Predicted probabilities of shape (n_samples, 2).
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=-1)
        return proba.cpu().numpy()

    def _compute_loss(self, X_t: torch.Tensor, y_t: torch.Tensor) -> float:
        """Computes loss on given tensors.

        Args:
            X_t: Feature tensor.
            y_t: Label tensor.

        Returns:
            Loss value as a float.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            loss = self.criterion(logits, y_t)
        return loss.item()
