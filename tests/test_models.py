"""Tests for model components."""

import pytest
import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn

from src.models.feature_encoder import FeatureEncoder
from src.models.context_module import ContextIntegrationModule
from src.models.caaa_model import CAAAModel
from src.models.baseline import BaselineClassifier, NaiveBaseline
from src.data_loader.dataset import generate_combined_dataset
from src.features.extractors import FeatureExtractor, N_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def feature_data():
    """Generate features for model tests."""
    fault_cases, load_cases = generate_combined_dataset(
        n_fault=10, n_load=10, seed=42
    )
    extractor = FeatureExtractor()
    all_cases = fault_cases + load_cases
    X = extractor.extract_batch(all_cases)
    y = np.array([0] * len(fault_cases) + [1] * len(load_cases))
    return X, y


@pytest.fixture
def feature_encoder():
    return FeatureEncoder(input_dim=36, hidden_dim=64)


@pytest.fixture
def context_module():
    return ContextIntegrationModule(temporal_dim=64, context_dim=5)


@pytest.fixture
def caaa_model():
    return CAAAModel(input_dim=36, hidden_dim=64, context_dim=5, n_classes=2)


# ── FeatureEncoder ───────────────────────────────────────────────────

class TestFeatureEncoder:
    def test_feature_encoder_output_shape(self, feature_encoder):
        x = torch.randn(4, 36)
        feature_encoder.eval()
        out = feature_encoder(x)
        assert out.shape == (4, 64)

    def test_feature_encoder_single_sample(self, feature_encoder):
        x = torch.randn(1, 36)
        feature_encoder.eval()
        out = feature_encoder(x)
        assert out.shape == (1, 64)


# ── ContextIntegrationModule ─────────────────────────────────────────

class TestContextModule:
    def test_context_module_output_shape(self, context_module):
        temporal = torch.randn(4, 64)
        context = torch.randn(4, 5)
        context_module.eval()
        out = context_module(temporal, context)
        assert out.shape == (4, 64)


# ── CAAAModel ─────────────────────────────────────────────────────────

class TestCAAAModel:
    def test_caaa_model_forward(self, caaa_model):
        x = torch.randn(4, 36)
        caaa_model.eval()
        logits = caaa_model(x)
        assert logits.shape == (4, 2)

    def test_caaa_model_predict(self, caaa_model):
        x = torch.randn(8, 36)
        preds = caaa_model.predict(x)
        assert preds.shape == (8,)
        assert all(p in (0, 1) for p in preds.tolist())

    def test_caaa_model_training_step(self):
        torch.manual_seed(0)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(16, 36)
        y = torch.randint(0, 2, (16,))

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Average of last 5 losses should be lower than average of first 5
        avg_first = np.mean(losses[:5])
        avg_last = np.mean(losses[-5:])
        assert avg_last < avg_first, (
            f"Loss did not decrease: first_5_avg={avg_first:.4f} last_5_avg={avg_last:.4f}"
        )

    def test_model_handles_various_batch_sizes(self, caaa_model):
        caaa_model.eval()
        for bs in [1, 8, 32]:
            x = torch.randn(bs, 36)
            logits = caaa_model(x)
            assert logits.shape == (bs, 2)
            preds = caaa_model.predict(x)
            assert preds.shape == (bs,)


# ── BaselineClassifier ───────────────────────────────────────────────

class TestBaselineClassifier:
    def test_baseline_classifier(self, feature_data):
        X, y = feature_data
        clf = BaselineClassifier(n_estimators=10, max_depth=5, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset({0, 1})

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        npt.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ── NaiveBaseline ────────────────────────────────────────────────────

class TestNaiveBaseline:
    def test_naive_baseline(self):
        nb = NaiveBaseline()
        X = np.random.randn(10, 36)
        preds = nb.predict(X)
        npt.assert_array_equal(preds, 0)
        assert preds.shape == (10,)

        proba = nb.predict_proba(X)
        assert proba.shape == (10, 2)
        npt.assert_array_equal(proba[:, 0], 1.0)
        npt.assert_array_equal(proba[:, 1], 0.0)
