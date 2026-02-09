"""Trainer for the CAAA model."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.features.feature_schema import CONTEXT_START as _CONTEXT_START, CONTEXT_END as _CONTEXT_END
from src.models import CAAAModel
from src.training.losses import ContextConsistencyLoss

logger = logging.getLogger(__name__)


class CAAATrainer:
    """Trainer for the CAAA model.

    Handles training, evaluation, and prediction for the CAAAModel
    using PyTorch with Adam optimizer and CrossEntropyLoss.

    **Important**: Input features should be standardized (zero mean, unit
    variance) before passing to ``train()``. Use
    ``sklearn.preprocessing.StandardScaler`` fitted on training data only.

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
        use_context_loss: bool = True,
    ) -> None:
        """Initializes the CAAATrainer.

        Args:
            model: The CAAAModel to train.
            learning_rate: Learning rate for the Adam optimizer.
            weight_decay: Weight decay (L2 regularization) for the optimizer.
            device: Device to run computations on ('cpu' or 'cuda').
            use_context_loss: Whether to use ContextConsistencyLoss.
        """
        self.device = device
        self.model = model.to(self.device)
        self.use_context_loss = use_context_loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if self.use_context_loss:
            self.criterion = ContextConsistencyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.temperature = 1.0  # default: no scaling; updated by calibrate_temperature()

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
        if self.use_context_loss:
            history["cls_loss"] = []
            history["consistency_loss"] = []
            history["calibration_loss"] = []
        if has_val:
            history["val_loss"] = []

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        n_samples = X_train_t.shape[0]

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_consistency_loss = 0.0
            epoch_calibration_loss = 0.0
            n_batches = 0

            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                if self.use_context_loss:
                    context = X_batch[:, _CONTEXT_START:_CONTEXT_END]
                    loss, components = self.criterion(logits, y_batch, context)
                else:
                    loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if self.use_context_loss:
                    epoch_cls_loss += components["cls_loss"]
                    epoch_consistency_loss += components["consistency_loss"]
                    epoch_calibration_loss += components["calibration_loss"]
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)
            if self.use_context_loss:
                history["cls_loss"].append(epoch_cls_loss / max(n_batches, 1))
                history["consistency_loss"].append(
                    epoch_consistency_loss / max(n_batches, 1)
                )
                history["calibration_loss"].append(
                    epoch_calibration_loss / max(n_batches, 1)
                )

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
            if self.use_context_loss:
                context = X_t[:, _CONTEXT_START:_CONTEXT_END]
                loss_tensor, _ = self.criterion(logits, y_t, context)
                loss = loss_tensor.item()
            else:
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

    def calibrate_temperature(
        self, X_val: np.ndarray, y_val: np.ndarray,
        lr: float = 0.01, max_iter: int = 50,
    ) -> float:
        """Learn temperature parameter on validation data.

        Minimizes NLL on the validation set by optimizing a single scalar T
        such that ``softmax(logits / T)`` produces calibrated probabilities.
        Must be called AFTER training, BEFORE ``predict_with_confidence``.

        Reference: Guo et al., "On Calibration of Modern Neural Networks",
        ICML 2017.

        Args:
            X_val: Validation features of shape (n_samples, input_dim).
            y_val: Validation labels of shape (n_samples,).
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Maximum optimizer iterations.

        Returns:
            Learned temperature value.
        """
        self.model.eval()
        temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        X_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_val, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t)

        def eval_fn():
            optimizer.zero_grad()
            # Clamp to avoid division by zero or negative temperature
            t = temperature.clamp(min=0.01)
            scaled = logits / t
            loss = nn.CrossEntropyLoss()(scaled, y_t)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        self.temperature = temperature.item()
        logger.info("Calibrated temperature: %.4f", self.temperature)
        return self.temperature

    def predict_with_confidence(
        self, X: np.ndarray, confidence_threshold: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predictions with UNKNOWN class for low-confidence cases.

        When ``calibrate_temperature()`` has been called, logits are divided
        by the learned temperature before applying softmax, producing
        better-calibrated confidence estimates.

        Args:
            X: Feature array of shape (n_samples, input_dim).
            confidence_threshold: Minimum probability threshold. Below this,
                predictions become class 2 (UNKNOWN).

        Returns:
            Tuple of (predictions, confidences) as numpy arrays.
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            scaled_logits = logits / self.temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)
            predictions[confidences < confidence_threshold] = 2
        return predictions.cpu().numpy(), confidences.cpu().numpy()

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

    def save_model(self, path: str) -> None:
        """Save model state dict, optimizer state, and training config.

        Args:
            path: File path to save the checkpoint (.pt file).
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "n_classes": self.model.classifier[-1].out_features,
            },
        }
        torch.save(checkpoint, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load model from checkpoint.

        Args:
            path: File path to the checkpoint (.pt file).
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Model loaded from %s", path)

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
            if self.use_context_loss:
                context = X_t[:, _CONTEXT_START:_CONTEXT_END]
                loss, _ = self.criterion(logits, y_t, context)
            else:
                loss = self.criterion(logits, y_t)
        return loss.item()
