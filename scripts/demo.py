#!/usr/bin/env python3
"""CAAA Demo - Quick demonstration of Context-Aware Anomaly Attribution."""

import argparse
import sys
import os

import numpy as np
import yaml
import torch
from sklearn.model_selection import train_test_split

# Fallback for running without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset
from src.features import FeatureExtractor
from src.models import CAAAModel, NaiveBaseline
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import compute_all_metrics, compute_false_positive_rate


def main():
    parser = argparse.ArgumentParser(description="CAAA Quick Demo")
    parser.add_argument("--n-fault", type=int, default=10)
    parser.add_argument("--n-load", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        training_cfg = config.get("training", {})
        if args.epochs == 30:
            args.epochs = training_cfg.get("epochs", args.epochs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 50)
    print("CAAA DEMO - Context-Aware Anomaly Attribution")
    print("=" * 50)

    # Generate dataset
    print("Generating dataset...")
    fault_cases, load_cases = generate_combined_dataset(
        n_fault=args.n_fault, n_load=args.n_load, seed=args.seed
    )
    all_cases = fault_cases + load_cases
    labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])
    print(f"  Faults: {args.n_fault}, Load spikes: {args.n_load}")

    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor()
    X = extractor.extract_batch(all_cases).astype(np.float32)
    print(f"  Feature matrix: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )

    # Train CAAA model
    print("Training CAAA model...")
    model = CAAAModel(input_dim=36)
    trainer = CAAATrainer(model, learning_rate=0.001, device="cpu")
    trainer.train(
        X_train, y_train, X_val=X_test, y_val=y_test,
        epochs=args.epochs, batch_size=16, early_stopping_patience=10,
    )

    # Evaluate
    print("Evaluating...")
    y_pred = trainer.predict(X_test)

    # Naive baseline for FP reduction
    naive = NaiveBaseline()
    naive_pred = naive.predict(X_test)
    naive_fp = compute_false_positive_rate(y_test, naive_pred)

    metrics = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)

    print()
    print("=" * 50)
    print("DEMO RESULTS")
    print("=" * 50)
    print(f"Accuracy:           {metrics.get('accuracy', 0):.2f}")
    print(f"F1 Score:           {metrics.get('f1', 0):.2f}")
    print(f"FP Rate:            {metrics.get('fp_rate', 0):.2f}")
    print(f"Fault Recall:       {metrics.get('fault_recall', 0):.2f}")
    fp_red = metrics.get("fp_reduction", 0) * 100
    print(f"FP Reduction:       {fp_red:.1f}%")
    print("=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    main()
