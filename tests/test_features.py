"""Tests for feature extraction."""

import pytest
import numpy as np

from src.data_loader.dataset import generate_combined_dataset
from src.data_loader.data_types import AnomalyCase
from src.features.extractors import FeatureExtractor, N_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    """Generate a dataset for feature tests (large enough for majority checks)."""
    fault_cases, load_cases = generate_combined_dataset(
        n_fault=30, n_load=30, seed=42
    )
    return fault_cases, load_cases


@pytest.fixture
def extractor():
    return FeatureExtractor()


# ── Single extraction ─────────────────────────────────────────────────

class TestFeatureExtractorSingle:
    def test_feature_extractor_single(self, extractor, dataset):
        fault_cases, _ = dataset
        feats = extractor.extract(fault_cases[0])
        assert feats.shape == (N_FEATURES,)
        assert feats.shape == (36,)

    def test_features_are_finite(self, extractor, dataset):
        fault_cases, load_cases = dataset
        for case in fault_cases + load_cases:
            feats = extractor.extract(case)
            assert np.all(np.isfinite(feats)), "Feature vector contains NaN or Inf"


# ── Batch extraction ──────────────────────────────────────────────────

class TestFeatureExtractorBatch:
    def test_feature_extractor_batch(self, extractor, dataset):
        fault_cases, load_cases = dataset
        all_cases = fault_cases + load_cases
        feats = extractor.extract_batch(all_cases)
        assert feats.shape == (len(all_cases), N_FEATURES)


# ── Feature names ─────────────────────────────────────────────────────

class TestFeatureNames:
    def test_feature_names(self, extractor):
        names = extractor.feature_names()
        assert len(names) == N_FEATURES
        assert len(names) == 36
        assert all(isinstance(n, str) for n in names)
        # Spot check some known names
        assert "global_load_ratio" in names
        assert "event_active" in names
        assert "max_error_rate" in names
        assert "n_services" in names


# ── Fault vs Load feature differences ────────────────────────────────

class TestFaultVsLoad:
    def test_fault_vs_load_features_differ(self, extractor, dataset):
        fault_cases, load_cases = dataset
        fault_feats = extractor.extract_batch(fault_cases)
        load_feats = extractor.extract_batch(load_cases)
        names = extractor.feature_names()

        # error_rate_delta should be higher for fault cases on average
        err_idx = names.index("error_rate_delta")
        assert np.mean(fault_feats[:, err_idx]) > np.mean(load_feats[:, err_idx])

        # event_active: majority (>80%) of load cases should have event_active > 0.5
        # and majority (>80%) of fault cases should have event_active < 0.5
        # (no longer universally 1.0 / 0.0 due to label-leakage prevention)
        event_idx = names.index("event_active")
        load_active_frac = np.mean(load_feats[:, event_idx] > 0.5)
        fault_inactive_frac = np.mean(fault_feats[:, event_idx] < 0.5)
        assert load_active_frac >= 0.80
        assert fault_inactive_frac >= 0.80


# ── Context features ─────────────────────────────────────────────────

class TestContextFeatures:
    def test_context_features_for_load(self, extractor, dataset):
        _, load_cases = dataset
        names = extractor.feature_names()
        event_idx = names.index("event_active")
        conf_idx = names.index("context_confidence")

        # Majority (>80%) of load cases should have event_active == 1.0
        # (some may have empty context due to label-leakage prevention)
        active_count = 0
        for case in load_cases:
            feats = extractor.extract(case)
            if feats[event_idx] == 1.0:
                active_count += 1
        assert active_count / len(load_cases) >= 0.80

    def test_context_features_for_fault(self, extractor, dataset):
        fault_cases, _ = dataset
        names = extractor.feature_names()
        event_idx = names.index("event_active")

        # Majority (>80%) of fault cases should have event_active == 0.0
        # (some may have fake context due to label-leakage prevention)
        inactive_count = 0
        for case in fault_cases:
            feats = extractor.extract(case)
            if feats[event_idx] == 0.0:
                inactive_count += 1
        assert inactive_count / len(fault_cases) >= 0.80


# ── Change point features ────────────────────────────────────────────

class TestChangePointFeatures:
    def test_onset_gradient_computed_for_all_cases(self, extractor, dataset):
        """onset_gradient should be a finite number for all cases."""
        fault_cases, load_cases = dataset
        names = extractor.feature_names()
        onset_idx = names.index("onset_gradient")
        for case in fault_cases + load_cases:
            feats = extractor.extract(case)
            assert np.isfinite(feats[onset_idx]), "onset_gradient is not finite"

    def test_change_point_magnitude_computed_for_all_cases(self, extractor, dataset):
        """change_point_magnitude should be a finite number for all cases."""
        fault_cases, load_cases = dataset
        names = extractor.feature_names()
        mag_idx = names.index("change_point_magnitude")
        for case in fault_cases + load_cases:
            feats = extractor.extract(case)
            assert np.isfinite(feats[mag_idx]), "change_point_magnitude is not finite"
            assert feats[mag_idx] >= 0.0, "change_point_magnitude should be non-negative"

    def test_change_point_magnitude_in_schema(self, extractor):
        """change_point_magnitude should replace memory_trend_uniformity."""
        names = extractor.feature_names()
        assert "change_point_magnitude" in names
        assert "memory_trend_uniformity" not in names
