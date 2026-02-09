#!/usr/bin/env python3
"""CAAA Training - Full training script with baseline comparison."""

import argparse
import logging
import sys
import os

import numpy as np
import yaml
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fallback for running without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset, generate_rcaeval_dataset
from src.features import FeatureExtractor
from src.models import CAAAModel, BaselineClassifier, NaiveBaseline
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_false_positive_rate,
    print_evaluation_summary,
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="CAAA Full Training")
    parser.add_argument("--n-fault", type=int, default=50)
    parser.add_argument("--n-load", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="Also train and compare with BaselineClassifier")
    parser.add_argument("--systems", nargs="+", default=["online-boutique"])
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")

    # Data source
    parser.add_argument(
        "--data", type=str, default="synthetic",
        choices=["synthetic", "rcaeval"],
        help="Data source: synthetic (default) or rcaeval (real faults)",
    )
    parser.add_argument("--dataset", type=str, default="RE1",
                        choices=["RE1", "RE2"], help="RCAEval dataset")
    parser.add_argument("--system", type=str, default="online-boutique",
                        choices=["online-boutique", "sock-shop", "train-ticket"],
                        help="Microservice system (for rcaeval)")
    parser.add_argument("--load-ratio", type=int, default=1,
                        help="Synthetic loads per RCAEval fault")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="RCAEval data directory")

    # Anomaly detector
    parser.add_argument("--anomaly-detector", action="store_true",
                        help="Enable LSTM-AE anomaly detection pre-stage")
    parser.add_argument("--ad-epochs", type=int, default=50,
                        help="Anomaly detector training epochs")
    parser.add_argument("--ad-threshold", type=float, default=95,
                        help="Anomaly detector threshold percentile")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        training_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        # Apply config values only when CLI arg uses the default
        if args.epochs == 50:
            args.epochs = training_cfg.get("epochs", args.epochs)
        if args.batch_size == 32:
            args.batch_size = training_cfg.get("batch_size", args.batch_size)
        if args.lr == 0.001:
            args.lr = training_cfg.get("learning_rate", args.lr)
        if args.systems == ["online-boutique"]:
            args.systems = data_cfg.get("systems", args.systems)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 50)
    print("CAAA TRAINING")
    print("=" * 50)

    # Generate dataset
    if args.data == "rcaeval":
        fault_cases, load_cases = generate_rcaeval_dataset(
            dataset=args.dataset, system=args.system,
            n_load_per_fault=args.load_ratio,
            data_dir=args.data_dir, seed=args.seed,
        )
    else:
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=args.n_fault, n_load=args.n_load,
            systems=args.systems, seed=args.seed,
        )

    # Optional anomaly detection pre-filter
    if args.anomaly_detector:
        from src.models.anomaly_detector import AnomalyDetector

        print("Training anomaly detector (LSTM-AE)...")
        normal_metrics = [svc.metrics for c in load_cases for svc in c.services]
        detector = AnomalyDetector(
            hidden_dim=64, latent_dim=16,
            seq_length=min(30, min(len(m) for m in normal_metrics) - 1),
            threshold_percentile=args.ad_threshold,
        )
        detector.fit(normal_metrics, epochs=args.ad_epochs, batch_size=args.batch_size)

        all_pre = fault_cases + load_cases
        detected = []
        for case in all_pre:
            _, max_score = detector.detect(case.services[0].metrics)
            if max_score > 1.0:
                detected.append(case)
        fault_cases = [c for c in detected if c.label == "FAULT"]
        load_cases = [c for c in detected if c.label == "EXPECTED_LOAD"]
        print(f"Anomaly detector kept {len(fault_cases)} faults, {len(load_cases)} loads")

    all_cases = fault_cases + load_cases
    labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])
    print(f"Dataset: {len(fault_cases)} faults, {len(load_cases)} load spikes")

    # Extract features
    extractor = FeatureExtractor()
    X = extractor.extract_batch(all_cases).astype(np.float32)
    print(f"Features: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=args.seed, stratify=labels,
    )

    # Scale features (fit on train only) for neural models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train CAAA model
    print(f"Training CAAA model for {args.epochs} epochs...")
    model = CAAAModel(input_dim=36)
    trainer = CAAATrainer(model, learning_rate=args.lr, device="cpu")
    trainer.train(
        X_train, y_train, X_val=X_test, y_val=y_test,
        epochs=args.epochs, batch_size=args.batch_size,
        early_stopping_patience=10,
    )

    # Evaluate CAAA
    caaa_pred = trainer.predict(X_test)

    # Naive baseline (always predicts FAULT=0)
    naive = NaiveBaseline()
    naive_pred = naive.predict(X_test)
    naive_fp = compute_false_positive_rate(y_test, naive_pred)

    caaa_metrics = compute_all_metrics(y_test, caaa_pred, baseline_fp_rate=naive_fp)

    print()
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:           {caaa_metrics.get('accuracy', 0):.2f}")
    print(f"F1 Score:           {caaa_metrics.get('f1', 0):.2f}")
    print(f"FP Rate:            {caaa_metrics.get('fp_rate', 0):.2f}")
    fp_red = caaa_metrics.get("fp_reduction", 0) * 100
    print(f"FP Reduction:       >{fp_red:.0f}%")
    print(f"Fault Recall:       >{caaa_metrics.get('fault_recall', 0):.2f}")
    print("=" * 50)

    # Save model
    os.makedirs("models/final", exist_ok=True)
    save_path = "models/final/caaa_model.pt"
    trainer.save_model(save_path)
    print(f"Model saved to: {save_path}")

    # Baseline comparison
    if args.baseline:
        print()
        print("Training BaselineClassifier...")
        bl = BaselineClassifier(random_state=args.seed)
        bl.fit(X_train, y_train)
        bl_pred = bl.predict(X_test)
        bl_metrics = compute_all_metrics(y_test, bl_pred, baseline_fp_rate=naive_fp)

        naive_metrics = compute_all_metrics(y_test, naive_pred, baseline_fp_rate=naive_fp)

        print()
        print("=" * 50)
        print("BASELINE COMPARISON")
        print("=" * 50)
        print(f"{'':16s}{'CAAA':>8s}{'Baseline':>12s}{'Naive':>8s}")
        print(f"{'Accuracy:':<16s}{caaa_metrics['accuracy']:>8.2f}{bl_metrics['accuracy']:>12.2f}{naive_metrics['accuracy']:>8.2f}")
        print(f"{'FP Rate:':<16s}{caaa_metrics['fp_rate']:>8.2f}{bl_metrics['fp_rate']:>12.2f}{naive_metrics['fp_rate']:>8.2f}")
        print(f"{'Fault Recall:':<16s}{caaa_metrics['fault_recall']:>8.2f}{bl_metrics['fault_recall']:>12.2f}{naive_metrics['fault_recall']:>8.2f}")
        caaa_fpr = caaa_metrics.get("fp_reduction", 0) * 100
        bl_fpr = bl_metrics.get("fp_reduction", 0) * 100
        naive_fpr = naive_metrics.get("fp_reduction", 0) * 100
        print(f"{'FP Reduction:':<16s}{caaa_fpr:>7.1f}%{bl_fpr:>11.1f}%{naive_fpr:>7.1f}%")
        print("=" * 50)


if __name__ == "__main__":
    main()
