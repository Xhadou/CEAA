"""Anomaly classifier: FAULT vs EXPECTED_LOAD vs UNKNOWN.

This is the CORE novel component — a multi-model sklearn classifier
that supports RandomForest, GradientBoosting, and MLP backends with
confidence-based UNKNOWN predictions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class AnomalyClassifier:
    """Classifier to distinguish between FAULT, EXPECTED_LOAD, and UNKNOWN.

    Supports multiple sklearn backends via *model_type*:
        - ``random_forest`` (default)
        - ``gradient_boosting``
        - ``mlp``

    Attributes:
        model_type: Type of sklearn classifier.
        model: The underlying sklearn estimator.
        label_encoder: Fitted :class:`~sklearn.preprocessing.LabelEncoder`.
        feature_names: List of feature names seen during :meth:`fit`.
        is_fitted: Whether the model has been fitted.
    """

    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "mlp": MLPClassifier,
    }

    def __init__(self, model_type: str = "random_forest", **model_kwargs) -> None:
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from {list(self.SUPPORTED_MODELS)}"
            )

        self.model_type = model_type
        self.model_kwargs = model_kwargs

        # Sensible defaults per model type
        if model_type == "random_forest":
            model_kwargs.setdefault("n_estimators", 100)
            model_kwargs.setdefault("max_depth", 10)
            model_kwargs.setdefault("random_state", 42)
            model_kwargs.setdefault("class_weight", "balanced")
        elif model_type == "gradient_boosting":
            model_kwargs.setdefault("n_estimators", 100)
            model_kwargs.setdefault("max_depth", 5)
            model_kwargs.setdefault("random_state", 42)
        elif model_type == "mlp":
            model_kwargs.setdefault("hidden_layer_sizes", (64, 32))
            model_kwargs.setdefault("max_iter", 500)
            model_kwargs.setdefault("random_state", 42)

        self.model = self.SUPPORTED_MODELS[model_type](**model_kwargs)
        self.label_encoder = LabelEncoder()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validate: bool = True,
    ) -> Dict:
        """Train the classifier.

        Args:
            X: Feature matrix (DataFrame with named columns).
            y: Labels (string or integer array).
            validate: Whether to run 5-fold cross-validation.

        Returns:
            Dictionary with training results including optional CV scores
            and feature importance.
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        y_encoded = self.label_encoder.fit_transform(y)
        X_clean = X.fillna(0) if isinstance(X, pd.DataFrame) else np.nan_to_num(X)

        results: Dict = {}

        if validate:
            cv_scores = cross_val_score(
                self.model, X_clean, y_encoded, cv=5, scoring="f1_weighted",
            )
            results["cv_f1_mean"] = float(cv_scores.mean())
            results["cv_f1_std"] = float(cv_scores.std())
            logger.info("CV F1 Score: %.3f (+/- %.3f)", cv_scores.mean(), cv_scores.std())

        self.model.fit(X_clean, y_encoded)
        self.is_fitted = True

        if hasattr(self.model, "feature_importances_") and self.feature_names:
            importance = pd.Series(
                self.model.feature_importances_, index=self.feature_names,
            ).sort_values(ascending=False)
            results["feature_importance"] = importance.to_dict()

        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix (DataFrame or ndarray).

        Returns:
            Array of predicted label strings.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        X_clean = X.fillna(0) if isinstance(X, pd.DataFrame) else np.nan_to_num(X)
        y_encoded = self.model.predict(X_clean)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X) -> pd.DataFrame:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with one column per class.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        X_clean = X.fillna(0) if isinstance(X, pd.DataFrame) else np.nan_to_num(X)
        probas = self.model.predict_proba(X_clean)
        return pd.DataFrame(probas, columns=self.label_encoder.classes_)

    def predict_with_confidence(
        self,
        X,
        confidence_threshold: float = 0.6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence threshold — returns UNKNOWN below threshold.

        Args:
            X: Feature matrix.
            confidence_threshold: Minimum max-class probability to commit
                to a prediction; below this the label is ``"UNKNOWN"``.

        Returns:
            Tuple of (predictions array, confidence scores array).
        """
        probas = self.predict_proba(X)
        predictions: List[str] = []
        confidences: List[float] = []

        for _, row in probas.iterrows():
            max_prob = float(row.max())
            max_class = str(row.idxmax())
            predictions.append(max_class if max_prob >= confidence_threshold else "UNKNOWN")
            confidences.append(max_prob)

        return np.array(predictions), np.array(confidences)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X, y: np.ndarray) -> Dict:
        """Evaluate classifier performance.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Dictionary with ``classification_report`` and ``confusion_matrix``.
        """
        y_pred = self.predict(X)
        return {
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(
                y, y_pred, labels=self.label_encoder.classes_,
            ).tolist(),
        }

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance (tree-based models only)."""
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return None
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
            },
            path,
        )
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "AnomalyClassifier":
        """Load a classifier from disk."""
        data = joblib.load(path)
        obj = cls(model_type=data["model_type"])
        obj.model = data["model"]
        obj.label_encoder = data["label_encoder"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted = True
        return obj


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    model_type: str = "random_forest",
) -> Tuple["AnomalyClassifier", Dict]:
    """Train and evaluate a classifier in one call.

    Args:
        X: Feature matrix.
        y: Labels.
        test_size: Test-set fraction.
        model_type: Classifier backend.

    Returns:
        Tuple of (trained classifier, results dict).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42,
    )

    classifier = AnomalyClassifier(model_type=model_type)
    train_results = classifier.fit(X_train, y_train)
    eval_results = classifier.evaluate(X_test, y_test)

    return classifier, {"train": train_results, "eval": eval_results}
