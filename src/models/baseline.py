"""Baseline models for comparison with CAAA model."""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """Simple RandomForest baseline for ablation comparisons.

    Uses integer labels (0=FAULT, 1=EXPECTED_LOAD) and numpy arrays.
    Used by ``scripts/ablation.py`` for systematic variant comparison.

    For a more feature-rich sklearn classifier with string labels,
    cross-validation, model persistence, and multi-backend support,
    see :class:`src.models.classifier.AnomalyClassifier`.

    Attributes:
        model: Underlying RandomForestClassifier instance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        """Initializes the BaselineClassifier.

        Args:
            n_estimators: Number of trees in the random forest.
            max_depth: Maximum depth of trees.
            random_state: Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        logger.info(
            "BaselineClassifier initialized: n_estimators=%d, max_depth=%d",
            n_estimators, max_depth,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """
        self.model.fit(X, y)
        logger.info("BaselineClassifier fitted on %d samples", len(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        return self.model.predict_proba(X)


class NaiveBaseline:
    """Baseline that labels everything as FAULT (no discrimination).

    Always predicts class 0 (FAULT) regardless of input, providing
    a lower-bound baseline for model comparison.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts all samples as FAULT (class 0).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of zeros of shape (n_samples,).
        """
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability [1.0, 0.0] for all samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, 2).
        """
        proba = np.zeros((len(X), 2))
        proba[:, 0] = 1.0
        return proba
