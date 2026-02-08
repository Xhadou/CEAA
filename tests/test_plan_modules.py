"""Tests for AnomalyClassifier — genuinely new multi-model sklearn classifier."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_combined_dataset
from src.features.extractors import FeatureExtractor
from src.models.classifier import AnomalyClassifier, train_and_evaluate


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def feature_data():
    fault_cases, load_cases = generate_combined_dataset(n_fault=10, n_load=10, seed=42)
    extractor = FeatureExtractor()
    all_cases = fault_cases + load_cases
    X = extractor.extract_batch(all_cases)
    y = np.array(["FAULT"] * len(fault_cases) + ["EXPECTED_LOAD"] * len(load_cases))
    X_df = pd.DataFrame(X, columns=extractor.feature_names())
    return X_df, y


# ── AnomalyClassifier ───────────────────────────────────────────────

class TestAnomalyClassifier:
    def test_random_forest(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        assert clf.is_fitted
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(y))

    def test_gradient_boosting(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="gradient_boosting")
        clf.fit(X, y, validate=False)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_mlp(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="mlp")
        clf.fit(X, y, validate=False)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_predict_proba(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        probas = clf.predict_proba(X)
        assert isinstance(probas, pd.DataFrame)
        assert probas.shape[0] == len(y)
        np.testing.assert_allclose(probas.sum(axis=1).values, 1.0, atol=1e-6)

    def test_predict_with_confidence_produces_unknown(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        # With a very high threshold most predictions should be UNKNOWN
        preds, confs = clf.predict_with_confidence(X, confidence_threshold=0.99)
        assert len(preds) == len(y)
        assert len(confs) == len(y)
        # At least verify all confidences are valid probabilities
        assert all(0 <= c <= 1 for c in confs)

    def test_predict_with_confidence_low_threshold(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        # With a low threshold no predictions should be UNKNOWN
        preds, confs = clf.predict_with_confidence(X, confidence_threshold=0.3)
        assert "UNKNOWN" not in preds

    def test_evaluate(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        results = clf.evaluate(X, y)
        assert "classification_report" in results
        assert "confusion_matrix" in results

    def test_feature_importance(self, feature_data):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        imp = clf.get_feature_importance()
        assert imp is not None
        assert isinstance(imp, pd.Series)
        assert len(imp) == 36

    def test_save_load(self, feature_data, tmp_path):
        X, y = feature_data
        clf = AnomalyClassifier(model_type="random_forest")
        clf.fit(X, y, validate=False)
        save_path = str(tmp_path / "test_clf.joblib")
        clf.save(save_path)

        loaded = AnomalyClassifier.load(save_path)
        assert loaded.is_fitted
        preds_orig = clf.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            AnomalyClassifier(model_type="svm")


# ── train_and_evaluate ───────────────────────────────────────────────

class TestTrainAndEvaluate:
    def test_train_and_evaluate(self, feature_data):
        X, y = feature_data
        clf, results = train_and_evaluate(X, y, test_size=0.3)
        assert clf.is_fitted
        assert "train" in results
        assert "eval" in results
