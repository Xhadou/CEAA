"""Main CAAA Pipeline — unified entry point supporting all model backends.

This module provides a single ``run_pipeline()`` function that supports both
the novel CAAA neural model and traditional sklearn classifiers. For quick
experiments with the CAAA model alone, use ``scripts/train.py`` or
``scripts/demo.py`` instead.

Differences from scripts/train.py:
    - Supports ``--model caaa|random_forest|gradient_boosting|mlp``
    - Uses ``AnomalyClassifier`` for sklearn backends (with cross-validation,
      save/load, string labels)
    - Produces a unified results summary across model types

Usage::

    python -m src.main --n-fault 50 --n-load 50 --model caaa
    python -m src.main --n-fault 50 --n-load 50 --model random_forest
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fallback for running without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_combined_dataset, generate_rcaeval_dataset
from src.features import FeatureExtractor
from src.models import CAAAModel, BaselineClassifier, NaiveBaseline, RuleBasedBaseline, XGBoostBaseline
from src.models.anomaly_detector import AnomalyDetector
from src.training.trainer import CAAATrainer
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_false_positive_rate,
    cross_validate_model,
    print_evaluation_summary,
)


def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger for the CAAA pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
    # RCAEval data params
    data_source: str = "synthetic",
    dataset: str = "RE1",
    system: str = "online-boutique",
    n_load_per_fault: int = 1,
    data_dir: str = "data/raw",
    # Anomaly detector params
    use_anomaly_detector: bool = False,
    ad_epochs: int = 50,
    ad_threshold_percentile: float = 95,
    # Hard scenarios
    include_hard: bool = False,
    # Cross-validation
    cv_folds: int = 1,
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
        data_source: ``"synthetic"`` or ``"rcaeval"``.
        dataset: RCAEval dataset identifier (``"RE1"`` or ``"RE2"``).
        system: Microservice system for RCAEval.
        n_load_per_fault: Synthetic loads per RCAEval fault case.
        data_dir: Path to downloaded RCAEval data.
        use_anomaly_detector: Enable LSTM-AE anomaly detection pre-stage.
        ad_epochs: Anomaly detector training epochs.
        ad_threshold_percentile: Anomaly detector threshold percentile.
        include_hard: Include hard/adversarial scenarios in dataset.
        cv_folds: Number of cross-validation folds (1 = single split).

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
    # Step 1: Load data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading data (source: {data_source})...")

    if data_source == "rcaeval":
        fault_cases, load_cases = generate_rcaeval_dataset(
            dataset=dataset,
            system=system,
            n_load_per_fault=n_load_per_fault,
            data_dir=data_dir,
            seed=seed,
        )
        print(f"  RCAEval: {len(fault_cases)} real faults + {len(load_cases)} synthetic loads")
    else:
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=n_fault, n_load=n_load, systems=systems, seed=seed,
            include_hard=include_hard,
        )
        print(f"  Synthetic: {len(fault_cases)} faults + {len(load_cases)} loads")

    # ------------------------------------------------------------------
    # Step 1b (optional): Anomaly detection pre-filter
    # ------------------------------------------------------------------
    if use_anomaly_detector:
        print(f"\n  Training anomaly detector (LSTM-AE)...")

        # Use load cases as "normal" training data for the autoencoder
        normal_metrics = []
        for case in load_cases:
            for svc in case.services:
                normal_metrics.append(svc.metrics)

        detector = AnomalyDetector(
            hidden_dim=64,
            latent_dim=16,
            seq_length=min(30, min(len(m) for m in normal_metrics) - 1),
            threshold_percentile=ad_threshold_percentile,
        )
        detector.fit(normal_metrics, epochs=ad_epochs, batch_size=batch_size)

        # Score all cases — anomalous cases proceed to attribution
        all_cases = fault_cases + load_cases
        detected_cases = []
        detected_labels = []
        missed = 0

        for case in all_cases:
            # Use first service's metrics for detection
            _, max_score = detector.detect(case.services[0].metrics)
            if max_score > 1.0:
                detected_cases.append(case)
                detected_labels.append(case.label)
            else:
                missed += 1

        print(f"  Anomaly detector: {len(detected_cases)} detected, {missed} filtered as normal")
        print(f"  Detection breakdown: "
              f"{sum(1 for l in detected_labels if l == 'FAULT')} faults, "
              f"{sum(1 for l in detected_labels if l == 'EXPECTED_LOAD')} loads detected")

        # Split back into fault/load for feature extraction
        fault_cases = [c for c in detected_cases if c.label == "FAULT"]
        load_cases = [c for c in detected_cases if c.label == "EXPECTED_LOAD"]

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
    # Cross-validation path (when cv_folds > 1)
    # ------------------------------------------------------------------
    if cv_folds > 1:
        print(f"\n[3/5] Running {cv_folds}-fold cross-validation ({model_type})...")

        def _model_factory():
            if model_type == "xgboost":
                return XGBoostBaseline(random_state=seed)
            elif model_type == "rule_based":
                return RuleBasedBaseline()
            elif model_type in ("random_forest",):
                return BaselineClassifier(random_state=seed)
            else:
                raise ValueError(
                    f"CV not supported for model_type={model_type!r}; "
                    "use cv_folds=1 for CAAA neural model."
                )

        fold_metrics = cross_validate_model(
            model_factory=_model_factory,
            X=X, y=labels, n_splits=cv_folds, seed=seed,
        )

        model_metrics = {}
        for key, values in fold_metrics.items():
            model_metrics[key] = float(np.mean(values))
            model_metrics[key + "_std"] = float(np.std(values))

        print(f"\n[4/5] CV Results ({cv_folds} folds)")
        print("=" * 60)
        print(f"  Accuracy:     {model_metrics['accuracy']:.3f} ± {model_metrics['accuracy_std']:.3f}")
        print(f"  F1 Score:     {model_metrics['f1']:.3f} ± {model_metrics['f1_std']:.3f}")
        print(f"  FP Rate:      {model_metrics['fp_rate']:.3f} ± {model_metrics['fp_rate_std']:.3f}")
        fp_red = model_metrics.get("fp_reduction", 0)
        fp_red_std = model_metrics.get("fp_reduction_std", 0)
        print(f"  FP Reduction: {fp_red * 100:.1f}% ± {fp_red_std * 100:.1f}%")
        print(f"  Fault Recall: {model_metrics['fault_recall']:.3f} ± {model_metrics['fault_recall_std']:.3f}")
        print("=" * 60)

        return model_metrics

    # ------------------------------------------------------------------
    # Step 3: Split (single train/test)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=seed, stratify=labels,
    )

    # Step 3b: Scale AFTER split (fit on train only) for neural models
    scaler = None
    if model_type in ("caaa", "mlp"):
        print("  Applying StandardScaler (fit on train split only)...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
        # Post-hoc temperature calibration on validation set
        trainer.calibrate_temperature(X_test, y_test)
        y_pred = trainer.predict(X_test)
    elif model_type == "xgboost":
        bl = XGBoostBaseline(random_state=seed)
        bl.fit(X_train, y_train)
        y_pred = bl.predict(X_test)
    elif model_type == "rule_based":
        bl = RuleBasedBaseline()
        bl.fit(X_train, y_train)
        y_pred = bl.predict(X_test)
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
    _setup_logging()
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
        choices=["caaa", "random_forest", "xgboost", "rule_based"],
        help="Model type",
    )
    parser.add_argument("--systems", nargs="+", default=["online-boutique"])
    parser.add_argument("--output", type=str, default="outputs/results")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--include-hard", action="store_true",
                        help="Include hard/adversarial scenarios in dataset")
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Number of cross-validation folds (1 = single split)")

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

    # Download helper
    parser.add_argument("--download-data", action="store_true",
                        help="Download RCAEval dataset and exit")

    args = parser.parse_args()

    # Override from config file if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            tc = cfg.get("training", {})
            args.epochs = tc.get("epochs", args.epochs)
            args.batch_size = tc.get("batch_size", args.batch_size)
            args.lr = tc.get("learning_rate", args.lr)
        else:
            logging.warning("Config file %s not found; using CLI defaults.", args.config)

    if args.download_data:
        from src.data_loader.download_data import download_rcaeval_dataset
        download_rcaeval_dataset(args.dataset, [args.system], args.data_dir)
        return

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
        data_source=args.data,
        dataset=args.dataset,
        system=args.system,
        n_load_per_fault=args.load_ratio,
        data_dir=args.data_dir,
        use_anomaly_detector=args.anomaly_detector,
        ad_epochs=args.ad_epochs,
        ad_threshold_percentile=args.ad_threshold,
        include_hard=args.include_hard,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
