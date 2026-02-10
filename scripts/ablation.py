#!/usr/bin/env python3
"""CAAA Ablation Study - Systematic evaluation of model variants."""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold

# Fallback for running without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset, generate_rcaeval_dataset
from src.features import FeatureExtractor
from src.models import CAAAModel, BaselineClassifier, NaiveBaseline, RuleBasedBaseline, XGBoostBaseline
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_false_positive_rate,
)


def run_caaa_variant(
    X_train, y_train, X_test, y_test, naive_fp, epochs, batch_size, lr,
    use_context_loss=True, loss_type="context_consistency", seed=42
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
        loss_type: Loss function variant (context_consistency, contrastive,
            or cross_entropy).
        seed: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    torch.manual_seed(seed)
    model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
    trainer = CAAATrainer(
        model, learning_rate=lr, device="cpu",
        use_context_loss=use_context_loss,
        loss_type=loss_type,
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


def run_rule_based(X_train, y_train, X_test, y_test, naive_fp):
    """Train and evaluate the rule-based baseline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        naive_fp: Naive baseline FP rate.

    Returns:
        Dictionary of evaluation metrics.
    """
    clf = RuleBasedBaseline()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)


def run_xgboost(X_train, y_train, X_test, y_test, naive_fp, seed=42):
    """Train and evaluate the XGBoost baseline.

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
    clf = XGBoostBaseline(random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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

    # Data source
    parser.add_argument(
        "--data", type=str, default="synthetic",
        choices=["synthetic", "rcaeval"],
        help="Data source: synthetic (default) or rcaeval (real faults)",
    )
    parser.add_argument("--include-hard", action="store_true",
                        help="Include hard/adversarial scenarios in dataset")
    parser.add_argument("--dataset", type=str, default="RE1",
                        choices=["RE1", "RE2"], help="RCAEval dataset")
    parser.add_argument("--system", type=str, default="online-boutique",
                        choices=["online-boutique", "sock-shop", "train-ticket"],
                        help="Microservice system (for rcaeval)")
    parser.add_argument("--load-ratio", type=int, default=1,
                        help="Synthetic loads per RCAEval fault")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="RCAEval data directory")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="Number of CV folds (0 = use n-runs with train/test split)")
    parser.add_argument("--shap", action="store_true",
                        help="Generate SHAP plots for Full CAAA and Baseline RF")

    args = parser.parse_args()

    metrics_to_track = ["accuracy", "f1", "fp_rate", "fault_recall", "fp_reduction"]

    # Variant definitions
    variants = [
        "Full CAAA",
        "CAAA + Contrastive",
        "No Context Features",
        "No Context Loss",
        "No Behavioral",
        "Context Only",
        "Baseline RF",
        "XGBoost",
        "Rule-Based",
        "Naive",
    ]

    # Collect results: {variant: {metric: [values across runs]}}
    all_results = {v: {m: [] for m in metrics_to_track} for v in variants}

    use_cv = args.cv_folds > 0
    n_iterations = args.cv_folds if use_cv else args.n_runs

    for run_idx in range(n_iterations if not use_cv else 1):
        run_seed = args.base_seed + run_idx
        if not use_cv:
            print(f"\n--- Run {run_idx + 1}/{args.n_runs} (seed={run_seed}) ---")

        np.random.seed(run_seed if not use_cv else args.base_seed)
        torch.manual_seed(run_seed if not use_cv else args.base_seed)

        # Generate dataset
        if args.data == "rcaeval":
            fault_cases, load_cases = generate_rcaeval_dataset(
                dataset=args.dataset, system=args.system,
                n_load_per_fault=args.load_ratio,
                data_dir=args.data_dir, seed=run_seed if not use_cv else args.base_seed,
            )
        else:
            fault_cases, load_cases = generate_combined_dataset(
                n_fault=args.n_fault, n_load=args.n_load,
                systems=args.systems, seed=run_seed if not use_cv else args.base_seed,
                include_hard=args.include_hard,
            )
        all_cases = fault_cases + load_cases
        labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])

        extractor = FeatureExtractor()
        X = extractor.extract_batch(all_cases).astype(np.float32)

        if use_cv:
            # Generate all CV folds once — shared across all variants
            skf = StratifiedKFold(
                n_splits=args.cv_folds, shuffle=True, random_state=args.base_seed,
            )
            folds = list(skf.split(X, labels))
        else:
            # Single train/test split
            from sklearn.model_selection import train_test_split as tts
            train_idx, test_idx = tts(
                np.arange(len(labels)), test_size=0.2,
                random_state=run_seed, stratify=labels,
            )
            folds = [(train_idx, test_idx)]

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if use_cv:
                print(f"\n--- Fold {fold_idx + 1}/{args.cv_folds} ---")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

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

            # --- CAAA + Contrastive ---
            print("  CAAA + Contrastive...")
            m = run_caaa_variant(
                X_train, y_train, X_test, y_test, naive_fp,
                args.epochs, max(args.batch_size, 16), args.lr,
                loss_type="contrastive", seed=run_seed,
            )
            for k in metrics_to_track:
                all_results["CAAA + Contrastive"][k].append(m.get(k, 0.0))

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

            # --- XGBoost ---
            print("  XGBoost...")
            m = run_xgboost(X_train, y_train, X_test, y_test, naive_fp, seed=run_seed)
            for k in metrics_to_track:
                all_results["XGBoost"][k].append(m.get(k, 0.0))

            # --- Rule-Based ---
            print("  Rule-Based...")
            m = run_rule_based(X_train, y_train, X_test, y_test, naive_fp)
            for k in metrics_to_track:
                all_results["Rule-Based"][k].append(m.get(k, 0.0))

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
    n_evals = args.cv_folds if use_cv else args.n_runs
    eval_label = f"{args.cv_folds}-fold CV" if use_cv else f"{args.n_runs} runs"
    print()
    print("=" * 100)
    print(f"ABLATION STUDY RESULTS (mean ± std over {eval_label})")
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

    # SHAP analysis (optional, on last fold/run's data)
    if args.shap:
        from src.features.feature_schema import ALL_FEATURE_NAMES
        from src.evaluation.visualization import plot_shap_summary, plot_shap_by_class

        shap_dir = os.path.join(csv_dir, "shap")
        os.makedirs(shap_dir, exist_ok=True)
        print("\nGenerating SHAP plots on last fold data...")

        # Baseline RF (fast)
        print("  Baseline RF SHAP...")
        rf = BaselineClassifier(random_state=args.base_seed)
        rf.fit(X_train, y_train)
        plot_shap_summary(
            rf, X_test, ALL_FEATURE_NAMES,
            save_path=os.path.join(shap_dir, "shap_baseline_rf.png"),
        )
        plot_shap_by_class(
            rf, X_test, y_test, ALL_FEATURE_NAMES,
            save_path=os.path.join(shap_dir, "shap_baseline_rf_by_class.png"),
        )
        print(f"  Saved to {shap_dir}/shap_baseline_rf*.png")

        # Full CAAA (uses KernelExplainer, slower)
        print("  Full CAAA SHAP (KernelExplainer, may take a moment)...")
        torch.manual_seed(args.base_seed)
        caaa_model = CAAAModel(input_dim=36, hidden_dim=64, n_classes=2)
        caaa_trainer = CAAATrainer(
            caaa_model, learning_rate=args.lr, device="cpu",
            use_context_loss=True,
        )
        caaa_trainer.train(
            X_train, y_train, epochs=args.epochs,
            batch_size=args.batch_size, early_stopping_patience=args.epochs,
        )
        plot_shap_summary(
            caaa_trainer, X_test, ALL_FEATURE_NAMES,
            save_path=os.path.join(shap_dir, "shap_full_caaa.png"),
            X_background=X_train[:50],
        )
        print(f"  Saved to {shap_dir}/shap_full_caaa.png")


if __name__ == "__main__":
    main()
