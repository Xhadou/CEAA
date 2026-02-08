#!/usr/bin/env python3
"""CAAA Ablation Study - Systematic evaluation of model variants."""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset
from src.features import FeatureExtractor
from src.models import CAAAModel, BaselineClassifier, NaiveBaseline
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_false_positive_rate,
)


def run_caaa_variant(
    X_train, y_train, X_test, y_test, naive_fp, epochs, batch_size, lr,
    use_context_loss=True, seed=42
):
    """Train and evaluate a CAAA model variant.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        use_context_loss: Whether to use ContextConsistencyLoss.
        seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    torch.manual_seed(seed)
    model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
    trainer = CAAATrainer(
        model, learning_rate=lr, device="cpu",
        use_context_loss=use_context_loss,
    )
    trainer.train(
        X_train, y_train, epochs=epochs,
        batch_size=batch_size, early_stopping_patience=epochs,
    )
    y_pred = trainer.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_baseline_rf(X_train, y_train, X_test, y_test, naive_fp, seed=42):
    """Train and evaluate RandomForest baseline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.
        seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    clf = BaselineClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_naive(X_test, y_test, naive_fp):
    """Evaluate the naive baseline (always predicts FAULT).

    Args:
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.

    Returns:
        Dictionary of evaluation metrics.
    """
    nb = NaiveBaseline()
    y_pred = nb.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def main():
    parser = argparse.ArgumentParser(description="CAAA Ablation Study")
    parser.add_argument("--n-fault", type=int, default=50)
    parser.add_argument("--n-load", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--systems", nargs="+", default=["online-boutique"])
    args = parser.parse_args()

    metrics_to_track = ["accuracy", "f1", "fp_rate", "fault_recall", "fp_reduction"]

    # Variant definitions
    variants = [
        "Full CAAA",
        "No Context Features",
        "No Context Loss",
        "No Behavioral",
        "Context Only",
        "Baseline RF",
        "Naive",
    ]

    # Collect results: {variant: {metric: [values across runs]}}
    all_results = {v: {m: [] for m in metrics_to_track} for v in variants}

    for run_idx in range(args.n_runs):
        run_seed = args.base_seed + run_idx
        print(f"\n--- Run {run_idx + 1}/{args.n_runs} (seed={run_seed}) ---")

        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        # Generate dataset
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=args.n_fault, n_load=args.n_load,
            systems=args.systems, seed=run_seed,
        )
        all_cases = fault_cases + load_cases
        labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=run_seed, stratify=labels,
        )

        # Naive baseline FP rate
        naive = NaiveBaseline()
        naive_pred = naive.predict(X_test)
        naive_fp = compute_false_positive_rate(y_test, naive_pred)

        # --- Full CAAA ---
        print("  Full CAAA...")
        m = run_caaa_variant(
            X_train, y_train, X_test, y_test, naive_fp,
            args.epochs, args.batch_size, args.lr,
            use_context_loss=True, seed=run_seed,
        )
        for k in metrics_to_track:
            all_results["Full CAAA"][k].append(m.get(k, 0.0))

        # --- No Context Features ---
        print("  No Context Features...")
        X_train_nc = X_train.copy()
        X_test_nc = X_test.copy()
        X_train_nc[:, 12:17] = 0.0
        X_test_nc[:, 12:17] = 0.0
        m = run_caaa_variant(
            X_train_nc, y_train, X_test_nc, y_test, naive_fp,
            args.epochs, args.batch_size, args.lr,
            use_context_loss=True, seed=run_seed,
        )
        for k in metrics_to_track:
            all_results["No Context Features"][k].append(m.get(k, 0.0))

        # --- No Context Loss ---
        print("  No Context Loss...")
        m = run_caaa_variant(
            X_train, y_train, X_test, y_test, naive_fp,
            args.epochs, args.batch_size, args.lr,
            use_context_loss=False, seed=run_seed,
        )
        for k in metrics_to_track:
            all_results["No Context Loss"][k].append(m.get(k, 0.0))

        # --- No Behavioral Features ---
        print("  No Behavioral...")
        X_train_nb = X_train.copy()
        X_test_nb = X_test.copy()
        X_train_nb[:, 6:12] = 0.0
        X_test_nb[:, 6:12] = 0.0
        m = run_caaa_variant(
            X_train_nb, y_train, X_test_nb, y_test, naive_fp,
            args.epochs, args.batch_size, args.lr,
            use_context_loss=True, seed=run_seed,
        )
        for k in metrics_to_track:
            all_results["No Behavioral"][k].append(m.get(k, 0.0))

        # --- Context Features Only ---
        print("  Context Only...")
        X_train_co = X_train.copy()
        X_test_co = X_test.copy()
        X_train_co[:, :12] = 0.0
        X_train_co[:, 17:] = 0.0
        X_test_co[:, :12] = 0.0
        X_test_co[:, 17:] = 0.0
        m = run_caaa_variant(
            X_train_co, y_train, X_test_co, y_test, naive_fp,
            args.epochs, args.batch_size, args.lr,
            use_context_loss=True, seed=run_seed,
        )
        for k in metrics_to_track:
            all_results["Context Only"][k].append(m.get(k, 0.0))

        # --- Baseline RF ---
        print("  Baseline RF...")
        m = run_baseline_rf(X_train, y_train, X_test, y_test, naive_fp, seed=run_seed)
        for k in metrics_to_track:
            all_results["Baseline RF"][k].append(m.get(k, 0.0))

        # --- Naive ---
        print("  Naive...")
        m = run_naive(X_test, y_test, naive_fp)
        for k in metrics_to_track:
            all_results["Naive"][k].append(m.get(k, 0.0))

    # Compute mean ± std
    summary = {}
    for v in variants:
        summary[v] = {}
        for m in metrics_to_track:
            vals = all_results[v][m]
            summary[v][m + "_mean"] = np.mean(vals)
            summary[v][m + "_std"] = np.std(vals)

    # Print table
    print()
    print("=" * 100)
    print(f"ABLATION STUDY RESULTS (mean ± std over {args.n_runs} runs)")
    print("=" * 100)
    header = f"{'Variant':<22s}{'Accuracy':>14s}{'F1 Score':>14s}{'FP Rate':>14s}{'Fault Recall':>14s}{'FP Reduction':>14s}"
    print(header)
    print("-" * 100)
    for v in variants:
        s = summary[v]
        acc = f"{s['accuracy_mean']:.2f}±{s['accuracy_std']:.2f}"
        f1 = f"{s['f1_mean']:.2f}±{s['f1_std']:.2f}"
        fpr = f"{s['fp_rate_mean']:.2f}±{s['fp_rate_std']:.2f}"
        fr = f"{s['fault_recall_mean']:.2f}±{s['fault_recall_std']:.2f}"
        fpred = f"{s['fp_reduction_mean']*100:.1f}±{s['fp_reduction_std']*100:.1f}%"
        print(f"{v:<22s}{acc:>14s}{f1:>14s}{fpr:>14s}{fr:>14s}{fpred:>14s}")
    print("=" * 100)

    # Save CSV
    csv_dir = "outputs/results"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = ["variant"]
        for m in metrics_to_track:
            header_row.extend([m + "_mean", m + "_std"])
        writer.writerow(header_row)
        for v in variants:
            row = [v]
            for m in metrics_to_track:
                row.append(f"{summary[v][m + '_mean']:.4f}")
                row.append(f"{summary[v][m + '_std']:.4f}")
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
