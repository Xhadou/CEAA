"""Main CAAA Pipeline — end-to-end workflow for anomaly attribution.

Generates synthetic data, extracts features, trains the CAAA model,
evaluates against baselines, and prints a results summary.

Usage::

    python -m src.main --n-fault 50 --n-load 50 --model caaa
    python -m src.main --model random_forest --output outputs/results
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset
from src.features import FeatureExtractor
from src.models import CAAAModel, BaselineClassifier, NaiveBaseline
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_false_positive_rate,
    print_evaluation_summary,
)


def run_pipeline(
    n_fault: int = 50,
    n_load: int = 50,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_type: str = "caaa",
    systems: list = None,
    seed: int = 42,
    output_dir: str = "outputs/results",
) -> dict:
    """Run the complete CAAA pipeline.

    Args:
        n_fault: Number of fault cases.
        n_load: Number of expected-load cases.
        epochs: Training epochs (used for CAAA model).
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        model_type: ``"caaa"`` or ``"random_forest"``.
        systems: Microservice system names.
        seed: Random seed.
        output_dir: Directory for saved results.

    Returns:
        Dictionary of evaluation metrics.
    """
    if systems is None:
        systems = ["online-boutique"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print("CAAA: Context-Aware Anomaly Attribution")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Fault cases:  {n_fault}")
    print(f"  Load cases:   {n_load}")
    print(f"  Model:        {model_type}")
    print(f"  Epochs:       {epochs}")

    # ------------------------------------------------------------------
    # Step 1: Generate dataset
    # ------------------------------------------------------------------
    print("\n[1/5] Generating dataset...")
    fault_cases, load_cases = generate_combined_dataset(
        n_fault=n_fault, n_load=n_load, systems=systems, seed=seed,
    )
    all_cases = fault_cases + load_cases
    labels = np.array([0 if c.label == "FAULT" else 1 for c in all_cases])
    print(f"  {len(fault_cases)} faults, {len(load_cases)} load spikes")

    # ------------------------------------------------------------------
    # Step 2: Extract features
    # ------------------------------------------------------------------
    print("\n[2/5] Extracting features...")
    extractor = FeatureExtractor()
    X = extractor.extract_batch(all_cases).astype(np.float32)
    print(f"  Feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # Step 3: Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=seed, stratify=labels,
    )

    # ------------------------------------------------------------------
    # Step 4: Train
    # ------------------------------------------------------------------
    print(f"\n[3/5] Training {model_type} model...")
    if model_type == "caaa":
        model = CAAAModel(input_dim=X_train.shape[1])
        trainer = CAAATrainer(model, learning_rate=learning_rate, device="cpu")
        trainer.train(
            X_train, y_train, X_val=X_test, y_val=y_test,
            epochs=epochs, batch_size=batch_size, early_stopping_patience=10,
        )
        y_pred = trainer.predict(X_test)
    else:
        bl = BaselineClassifier(random_state=seed)
        bl.fit(X_train, y_train)
        y_pred = bl.predict(X_test)

    # ------------------------------------------------------------------
    # Step 5: Evaluate
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating...")
    naive = NaiveBaseline()
    naive_pred = naive.predict(X_test)
    naive_fp = compute_false_positive_rate(y_test, naive_pred)
    model_metrics = compute_all_metrics(y_test, y_pred, baseline_fp_rate=naive_fp)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n[5/5] Results")
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (all anomalies = FAULT):")
    print(f"  FP Rate: {naive_fp:.3f}")
    print(f"\nCAAA Model ({model_type}):")
    print(f"  Accuracy:     {model_metrics['accuracy']:.3f}")
    print(f"  F1 Score:     {model_metrics['f1']:.3f}")
    print(f"  FP Rate:      {model_metrics['fp_rate']:.3f}")
    fp_red = model_metrics.get("fp_reduction", 0)
    print(f"  FP Reduction: {fp_red * 100:.1f}%")
    print(f"  Fault Recall: {model_metrics['fault_recall']:.3f}")
    print("=" * 60)

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if model_type == "caaa":
        save_path = str(out_path / "caaa_model.pt")
        trainer.save_model(save_path)
        print(f"\nModel saved to {save_path}")

    return model_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAAA: Context-Aware Anomaly Attribution",
    )
    parser.add_argument("--n-fault", type=int, default=50)
    parser.add_argument("--n-load", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", type=str, default="caaa",
        choices=["caaa", "random_forest"],
        help="Model type",
    )
    parser.add_argument("--systems", nargs="+", default=["online-boutique"])
    parser.add_argument("--output", type=str, default="outputs/results")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # Override from config file if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        tc = cfg.get("training", {})
        args.epochs = tc.get("epochs", args.epochs)
        args.batch_size = tc.get("batch_size", args.batch_size)
        args.lr = tc.get("learning_rate", args.lr)

    run_pipeline(
        n_fault=args.n_fault,
        n_load=args.n_load,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_type=args.model,
        systems=args.systems,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
