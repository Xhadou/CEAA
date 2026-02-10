"""Trainer for the CAAA model."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.features.feature_schema import CONTEXT_START as _CONTEXT_START, CONTEXT_END as _CONTEXT_END
from src.models import CAAAModel
from src.training.losses import ContextConsistencyLoss, SupConContextLoss

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
        loss_type: str = "context_consistency",
    ) -> None:
        """Initializes the CAAATrainer.

        Args:
            model: The CAAAModel to train.
            learning_rate: Learning rate for the Adam optimizer.
            weight_decay: Weight decay (L2 regularization) for the optimizer.
            device: Device to run computations on ('cpu' or 'cuda').
            use_context_loss: Whether to use ContextConsistencyLoss.
                Ignored when *loss_type* is explicitly set to a value other
                than ``"context_consistency"``.
            loss_type: Loss function variant:
                ``"context_consistency"`` — ContextConsistencyLoss (default).
                ``"contrastive"`` — SupConContextLoss.
                ``"cross_entropy"`` — plain CrossEntropyLoss.
        """
        self.device = device
        self.model = model.to(self.device)
        self.loss_type = loss_type

        # Backward compat: use_context_loss=False overrides to cross_entropy
        if not use_context_loss and loss_type == "context_consistency":
            self.loss_type = "cross_entropy"

        self.use_context_loss = self.loss_type == "context_consistency"
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if self.loss_type == "contrastive":
            self.criterion = SupConContextLoss()
        elif self.loss_type == "context_consistency":
            self.criterion = ContextConsistencyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.temperature = 1.0  # default: no scaling; updated by calibrate_temperature()
        self._class_centroids: Optional[np.ndarray] = None

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
        elif self.loss_type == "contrastive":
            history["contrastive_loss"] = []
            history["cls_loss"] = []
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
            epoch_contrastive_loss = 0.0
            n_batches = 0

            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                if self.loss_type == "contrastive":
                    context = X_batch[:, _CONTEXT_START:_CONTEXT_END]
                    embeddings = self.model.get_embeddings(X_batch)
                    loss, components = self.criterion(
                        embeddings, logits, y_batch, context,
                    )
                elif self.use_context_loss:
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
                elif self.loss_type == "contrastive":
                    epoch_contrastive_loss += components["contrastive_loss"]
                    epoch_cls_loss += components["cls_loss"]
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
            elif self.loss_type == "contrastive":
                history["contrastive_loss"].append(
                    epoch_contrastive_loss / max(n_batches, 1)
                )
                history["cls_loss"].append(epoch_cls_loss / max(n_batches, 1))

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
        self,
        X: np.ndarray,
        base_threshold: float = 0.7,
        context_sensitivity: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with context-adaptive UNKNOWN thresholding.

        The effective per-sample threshold is::

            threshold = base_threshold + context_sensitivity * (context_confidence - 0.5)

        When *context_confidence* is high (0.8+) the threshold rises and the
        model is more decisive (fewer UNKNOWN).  When *context_confidence* is
        low (0.2−) the threshold drops and the model is more cautious (more
        UNKNOWN).

        Inspired by USAD's tunable anomaly score (KDD 2020).

        When ``calibrate_temperature()`` has been called, logits are divided
        by the learned temperature before applying softmax.

        Args:
            X: Feature array of shape (n_samples, input_dim).
            base_threshold: Base confidence threshold.
            context_sensitivity: How much context_confidence shifts the
                threshold around *base_threshold*.

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

            # Context-adaptive threshold: context_confidence is the last
            # feature in the context group (see feature_schema.CONTEXT_NAMES)
            ctx_conf = X_t[:, _CONTEXT_END - 1]
            threshold = base_threshold + context_sensitivity * (ctx_conf - 0.5)
            # Clamp to [0.5, 0.95] to avoid degenerate thresholds
            threshold = torch.clamp(threshold, 0.5, 0.95)

            predictions[confidences < threshold] = 2
        return predictions.cpu().numpy(), confidences.cpu().numpy()

    def predict_with_confidence_fixed(
        self, X: np.ndarray, confidence_threshold: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with a fixed (non-adaptive) confidence threshold.

        Provided for backward compatibility and ablation studies comparing
        fixed vs adaptive thresholding.

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

    def predict_with_embeddings(
        self,
        X: np.ndarray,
        distance_threshold: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify using embedding cluster distances.

        Computes cosine distance from each sample's embedding to pre-computed
        class centroids.  If the nearest centroid is farther than
        *distance_threshold*, the prediction is UNKNOWN (class 2).

        Call :meth:`compute_class_centroids` after training to set up the
        centroids from training data.

        Args:
            X: Feature array of shape (n_samples, input_dim).
            distance_threshold: Maximum cosine distance to nearest centroid.
                Samples beyond this become UNKNOWN.

        Returns:
            Tuple of (predictions, distances) where distances is the cosine
            distance to the nearest centroid.
        """
        if self._class_centroids is None:
            raise RuntimeError(
                "Class centroids not computed. Call compute_class_centroids() "
                "after training."
            )

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            emb = self.model.get_embeddings(X_t)
            emb = torch.nn.functional.normalize(emb, dim=1)

        centroids = torch.tensor(
            self._class_centroids, dtype=torch.float32, device=self.device
        )
        # Cosine distance: 1 - cosine_similarity
        cos_sim = torch.matmul(emb, centroids.T)  # (n, n_classes)
        distances = 1.0 - cos_sim  # (n, n_classes)

        min_dist, predictions = torch.min(distances, dim=1)
        predictions[min_dist > distance_threshold] = 2

        return predictions.cpu().numpy(), min_dist.cpu().numpy()

    def compute_class_centroids(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Compute mean embedding per class from training data.

        Must be called after training and before
        :meth:`predict_with_embeddings`.

        Args:
            X_train: Training features of shape (n_samples, input_dim).
            y_train: Training labels of shape (n_samples,).
        """
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            emb = self.model.get_embeddings(X_t)
            emb = torch.nn.functional.normalize(emb, dim=1)

        centroids = []
        for cls in sorted(set(y_train.tolist())):
            mask = torch.tensor(y_train == cls, device=self.device)
            cls_emb = emb[mask].mean(dim=0)
            cls_emb = torch.nn.functional.normalize(cls_emb, dim=0)
            centroids.append(cls_emb.cpu().numpy())
        self._class_centroids = np.stack(centroids)

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
            if self.loss_type == "contrastive":
                context = X_t[:, _CONTEXT_START:_CONTEXT_END]
                embeddings = self.model.get_embeddings(X_t)
                loss, _ = self.criterion(embeddings, logits, y_t, context)
            elif self.use_context_loss:
                context = X_t[:, _CONTEXT_START:_CONTEXT_END]
                loss, _ = self.criterion(logits, y_t, context)
            else:
                loss = self.criterion(logits, y_t)
        return loss.item()
