"""Baseline models for comparison with CAAA model."""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_schema import ALL_FEATURE_NAMES

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


class RuleBasedBaseline:
    """Rule-based baseline using simple feature thresholds.

    Predicts EXPECTED_LOAD (1) if ``event_active > 0.5`` AND
    ``error_rate_delta < 0.3``; predicts FAULT (0) otherwise.

    This tests whether a simple decision rule can match the neural model.
    """

    def __init__(self) -> None:
        self._event_active_idx = ALL_FEATURE_NAMES.index("event_active")
        self._error_rate_delta_idx = ALL_FEATURE_NAMES.index("error_rate_delta")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op: rule-based model has no learnable parameters."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using simple threshold rules.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        event_active = X[:, self._event_active_idx]
        error_delta = X[:, self._error_rate_delta_idx]
        is_load = (event_active > 0.5) & (error_delta < 0.3)
        return np.where(is_load, 1, 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns hard probabilities based on rule predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, 2).
        """
        preds = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[preds == 0, 0] = 1.0
        proba[preds == 1, 1] = 1.0
        return proba


class XGBoostBaseline:
    """XGBoost baseline classifier.

    Wraps ``xgboost.XGBClassifier`` with sensible defaults for the CAAA
    anomaly attribution task.

    Attributes:
        model: Underlying XGBClassifier instance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initializes the XGBoostBaseline.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            random_state: Random seed for reproducibility.
        """
        from xgboost import XGBClassifier

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        logger.info(
            "XGBoostBaseline initialized: n_estimators=%d, max_depth=%d, lr=%.3f",
            n_estimators, max_depth, learning_rate,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """
        self.model.fit(X, y)
        logger.info("XGBoostBaseline fitted on %d samples", len(y))

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
