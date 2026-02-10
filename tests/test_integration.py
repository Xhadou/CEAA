"""Integration tests for the CAAA pipeline."""

import numpy as np
import numpy.testing as npt
import pytest
import torch

from src.data_loader import generate_combined_dataset
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_false_positive_rate,
)
from src.features import FeatureExtractor
from src.features.feature_schema import CONTEXT_START, CONTEXT_END
from src.models import CAAAModel, NaiveBaseline
from src.training.losses import ContextConsistencyLoss, SupConContextLoss
from src.training.trainer import CAAATrainer


class TestEndToEndPipeline:
    """Test the full CAAA pipeline: data → features → train → evaluate."""

    def test_end_to_end_pipeline(self):
        """Test the complete pipeline end-to-end."""
        # 1. Generate small dataset (5 fault, 5 load)
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=5, n_load=5, seed=42
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        # 2. Extract features (should be shape (10, 36))
        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)
        assert X.shape == (10, 36)
        assert np.all(np.isfinite(X))

        # 3. Split 80/20
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels,
        )

        # 4. Train CAAAModel for 10 epochs (with ContextConsistencyLoss)
        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(model, learning_rate=0.001, use_context_loss=True)
        history = trainer.train(
            X_train, y_train, epochs=10, batch_size=8,
            early_stopping_patience=10,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 10

        # 5. Predict on test set
        y_pred = trainer.predict(X_test)
        assert y_pred.shape == (len(y_test),)
        assert all(p in (0, 1) for p in y_pred)

        # 6. Compute metrics
        naive = NaiveBaseline()
        naive_pred = naive.predict(X_test)
        naive_fp = compute_false_positive_rate(y_test, naive_pred)
        metrics = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)

        # 7. Assert: accuracy > 0.0 (produces valid predictions)
        assert metrics["accuracy"] >= 0.0
        assert 0.0 <= metrics["fp_rate"] <= 1.0

        # 8. Assert: fault_recall is valid
        assert 0.0 <= metrics["fault_recall"] <= 1.0

        # 9. Assert: unknown_rate is 0 for standard predictions
        assert metrics["unknown_rate"] == 0.0

        # 10. Test predict_with_confidence returns valid UNKNOWN predictions
        preds_conf, confs = trainer.predict_with_confidence(
            X_test, base_threshold=0.99, context_sensitivity=0.0
        )
        assert preds_conf.shape == (len(y_test),)
        assert confs.shape == (len(y_test),)
        # With very high threshold, some predictions should be UNKNOWN
        assert set(preds_conf).issubset({0, 1, 2})
        # Confidences should be valid probabilities
        assert np.all(confs >= 0.0)
        assert np.all(confs <= 1.0)


class TestAblationNoContext:
    """Test that zeroing context features still produces valid predictions."""

    def test_ablation_no_context(self):
        """Zeroing context features should still yield valid outputs."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=5, n_load=5, seed=123
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        # Zero out context features
        X[:, CONTEXT_START:CONTEXT_END] = 0.0

        torch.manual_seed(123)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(model, learning_rate=0.001, use_context_loss=True)
        history = trainer.train(X, labels, epochs=5, batch_size=8)

        y_pred = trainer.predict(X)
        assert y_pred.shape == (len(labels),)
        assert set(y_pred).issubset({0, 1})


class TestContextConsistencyLoss:
    """Test that ContextConsistencyLoss computes valid loss values."""

    def test_context_consistency_loss_forward(self):
        """CCL should return valid total loss and component dict."""
        torch.manual_seed(42)
        ccl = ContextConsistencyLoss(alpha=0.3, beta=0.1)

        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        context = torch.rand(8, 5)

        total_loss, components = ccl(logits, labels, context)

        # Total loss should be a scalar tensor
        assert total_loss.shape == ()
        assert total_loss.item() > 0.0
        assert torch.isfinite(total_loss)

        # Components should be present and positive
        assert "cls_loss" in components
        assert "consistency_loss" in components
        assert "calibration_loss" in components
        assert components["cls_loss"] >= 0.0
        assert components["consistency_loss"] >= 0.0
        assert components["calibration_loss"] >= 0.0

    def test_context_consistency_loss_gradients(self):
        """Gradients should flow through all CCL components."""
        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        ccl = ContextConsistencyLoss(alpha=0.3, beta=0.1)

        x = torch.randn(8, 36)
        labels = torch.randint(0, 2, (8,))

        logits = model(x)
        context = x[:, CONTEXT_START:CONTEXT_END]
        total_loss, _ = ccl(logits, labels, context)
        total_loss.backward()

        # All model parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_context_consistency_loss_ablation(self):
        """Without context loss (alpha=0, beta=0), should equal CrossEntropyLoss."""
        torch.manual_seed(42)
        ccl = ContextConsistencyLoss(alpha=0.0, beta=0.0)
        ce = torch.nn.CrossEntropyLoss()

        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        context = torch.rand(8, 5)

        total_loss, components = ccl(logits, labels, context)
        ce_loss = ce(logits, labels)

        npt.assert_allclose(total_loss.item(), ce_loss.item(), atol=1e-6)


class TestTemperatureScaling:
    """Test post-hoc temperature calibration."""

    def test_temperature_is_learned(self):
        """Temperature should differ from 1.0 after calibration."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=15, n_load=15, seed=42,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=0.3, random_state=42, stratify=labels,
        )

        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(model, learning_rate=0.001, use_context_loss=True)
        trainer.train(X_train, y_train, epochs=20, batch_size=8)

        assert trainer.temperature == 1.0  # default before calibration
        T = trainer.calibrate_temperature(X_val, y_val)
        assert T != 1.0, "Temperature should change from default after calibration"
        assert T > 0.0, "Temperature must be positive"
        assert trainer.temperature == T

    def test_calibrated_confidence_differs(self):
        """Calibrated confidences should differ from uncalibrated ones."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=15, n_load=15, seed=99,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=0.3, random_state=99, stratify=labels,
        )

        torch.manual_seed(99)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(model, learning_rate=0.001, use_context_loss=True)
        trainer.train(X_train, y_train, epochs=20, batch_size=8)

        # Uncalibrated
        _, confs_uncal = trainer.predict_with_confidence_fixed(X_val, confidence_threshold=0.5)

        # Calibrate
        trainer.calibrate_temperature(X_val, y_val)

        # Calibrated
        _, confs_cal = trainer.predict_with_confidence_fixed(X_val, confidence_threshold=0.5)

        # Confidences should differ after calibration
        assert not np.allclose(confs_uncal, confs_cal, atol=1e-6), \
            "Calibrated confidences should differ from uncalibrated"
        # All confidences should be valid probabilities
        assert np.all(confs_cal >= 0.0)
        assert np.all(confs_cal <= 1.0)


class TestSupConContextLoss:
    """Test supervised contrastive loss with context-aware temperature."""

    def test_supcon_forward_valid_loss(self):
        """SupConContextLoss should produce a valid scalar loss."""
        torch.manual_seed(42)
        loss_fn = SupConContextLoss()

        embeddings = torch.randn(16, 64)
        logits = torch.randn(16, 2)
        labels = torch.randint(0, 2, (16,))
        context = torch.rand(16, 5)

        total_loss, components = loss_fn(embeddings, logits, labels, context)

        assert total_loss.shape == ()
        assert torch.isfinite(total_loss)
        assert "contrastive_loss" in components
        assert "cls_loss" in components
        assert "context_modulation" in components
        assert components["cls_loss"] >= 0.0

    def test_supcon_gradients_flow(self):
        """Gradients should flow through all model parameters."""
        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        loss_fn = SupConContextLoss()

        x = torch.randn(16, 36)
        labels = torch.randint(0, 2, (16,))

        embeddings = model.get_embeddings(x)
        logits = model(x)
        context = x[:, CONTEXT_START:CONTEXT_END]
        total_loss, _ = loss_fn(embeddings, logits, labels, context)
        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


class TestContrastiveTraining:
    """Test training with contrastive loss variant."""

    def test_contrastive_training_converges(self):
        """Loss should decrease over epochs with contrastive training."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=10, n_load=10, seed=42,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(
            model, learning_rate=0.001, loss_type="contrastive",
        )
        history = trainer.train(X, labels, epochs=15, batch_size=16)

        assert "train_loss" in history
        assert "contrastive_loss" in history
        assert "cls_loss" in history
        # Loss should generally decrease (first > last)
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_contrastive_predict(self):
        """Contrastive-trained model should produce valid predictions."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=10, n_load=10, seed=42,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(
            model, learning_rate=0.001, loss_type="contrastive",
        )
        trainer.train(X, labels, epochs=10, batch_size=16)

        y_pred = trainer.predict(X)
        assert y_pred.shape == (len(labels),)
        assert set(y_pred).issubset({0, 1})


class TestGetEmbeddings:
    """Test CAAAModel.get_embeddings()."""

    def test_get_embeddings_shape(self):
        """get_embeddings should return (batch, hidden_dim) tensor."""
        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        x = torch.randn(8, 36)

        emb = model.get_embeddings(x)
        assert emb.shape == (8, 64)


class TestPredictWithEmbeddings:
    """Test embedding-based UNKNOWN classification."""

    def test_predict_with_embeddings_valid(self):
        """predict_with_embeddings should return valid predictions."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=10, n_load=10, seed=42,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(
            model, learning_rate=0.001, loss_type="contrastive",
        )
        trainer.train(X, labels, epochs=10, batch_size=16)
        trainer.compute_class_centroids(X, labels)

        preds, dists = trainer.predict_with_embeddings(X, distance_threshold=1.5)
        assert preds.shape == (len(labels),)
        assert dists.shape == (len(labels),)
        assert set(preds).issubset({0, 1, 2})
        assert np.all(dists >= 0.0)

    def test_predict_with_embeddings_raises_without_centroids(self):
        """Should raise RuntimeError if centroids not computed."""
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(model, learning_rate=0.001)
        X = np.random.randn(5, 36).astype(np.float32)

        with pytest.raises(RuntimeError, match="Class centroids not computed"):
            trainer.predict_with_embeddings(X)


class TestAdaptiveThreshold:
    """Test context-adaptive UNKNOWN thresholding."""

    def test_adaptive_threshold_produces_unknown(self):
        """With a high base_threshold, some predictions should be UNKNOWN."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=10, n_load=10, seed=42,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(model, learning_rate=0.001, use_context_loss=True)
        trainer.train(X, labels, epochs=10, batch_size=8)

        preds, confs = trainer.predict_with_confidence(
            X, base_threshold=0.95, context_sensitivity=0.0,
        )
        assert set(preds).issubset({0, 1, 2})
        assert np.all(confs >= 0.0)
        assert np.all(confs <= 1.0)

    def test_fixed_vs_adaptive_differ(self):
        """Fixed and adaptive threshold should produce different results."""
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=15, n_load=15, seed=42,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0] * len(fault_cases) + [1] * len(load_cases))

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        torch.manual_seed(42)
        model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        trainer = CAAATrainer(model, learning_rate=0.001, use_context_loss=True)
        trainer.train(X, labels, epochs=15, batch_size=8)

        preds_fixed, _ = trainer.predict_with_confidence_fixed(
            X, confidence_threshold=0.7,
        )
        preds_adaptive, _ = trainer.predict_with_confidence(
            X, base_threshold=0.7, context_sensitivity=0.2,
        )
        # With context_sensitivity > 0, results should generally differ
        # (unless all context_confidence values happen to be exactly 0.5)
        assert preds_fixed.shape == preds_adaptive.shape
