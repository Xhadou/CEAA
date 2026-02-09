"""Tests for RCAEval pipeline integration and AnomalyDetector."""

import os

import pytest

from src.data_loader.dataset import generate_rcaeval_dataset

RCAEVAL_DIR = "data/raw"
HAS_RCAEVAL = os.path.exists(os.path.join(RCAEVAL_DIR, "RE1", "online-boutique"))


@pytest.mark.skipif(not HAS_RCAEVAL, reason="RCAEval data not downloaded")
class TestRCAEvalPipeline:
    def test_generate_rcaeval_dataset(self):
        fault_cases, load_cases = generate_rcaeval_dataset(
            dataset="RE1", system="online-boutique", n_load_per_fault=1,
        )
        assert len(fault_cases) > 0
        assert len(load_cases) == len(fault_cases)
        assert all(c.label == "FAULT" for c in fault_cases)
        assert all(c.label == "EXPECTED_LOAD" for c in load_cases)

    def test_rcaeval_with_anomaly_detector(self):
        from src.models.anomaly_detector import AnomalyDetector

        fault_cases, load_cases = generate_rcaeval_dataset(
            dataset="RE1", system="online-boutique", n_load_per_fault=1,
        )
        # Use load cases as normal baseline
        normal_metrics = [svc.metrics for c in load_cases[:5] for svc in c.services]
        detector = AnomalyDetector(hidden_dim=32, latent_dim=8, seq_length=20)
        detector.fit(normal_metrics, epochs=10)

        # Test detection on a fault case
        _, max_score = detector.detect(fault_cases[0].services[0].metrics)
        assert isinstance(max_score, float)


class TestAnomalyDetectorSynthetic:
    def test_anomaly_detector_train_detect(self):
        """Test AnomalyDetector end-to-end with synthetic data."""
        from src.data_loader import generate_combined_dataset
        from src.models.anomaly_detector import AnomalyDetector

        fault_cases, load_cases = generate_combined_dataset(n_fault=5, n_load=5, seed=42)

        # Train on normal (load) data
        normal_metrics = [svc.metrics for c in load_cases for svc in c.services]
        detector = AnomalyDetector(hidden_dim=16, latent_dim=8, seq_length=10)
        detector.fit(normal_metrics, epochs=5)

        # Detect on fault data
        for case in fault_cases[:3]:
            scores, max_score = detector.detect(case.services[0].metrics)
            assert isinstance(max_score, float)
            assert len(scores) > 0

    def test_anomaly_detector_import_from_models(self):
        """Test that AnomalyDetector is exported from src.models."""
        from src.models import AnomalyDetector, LSTMAutoencoder

        assert AnomalyDetector is not None
        assert LSTMAutoencoder is not None
