"""Evaluation metrics for the CAAA system."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def compute_false_positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 0
) -> float:
    """Computes false positive rate for the given positive class.

    A false positive occurs when predicting FAULT (0) when the true label
    is EXPECTED_LOAD (1).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        positive_class: The class considered positive (default: 0 = FAULT).

    Returns:
        False positive rate: FP / (FP + TN).
    """
    negatives = y_true != positive_class
    n_negatives = negatives.sum()
    if n_negatives == 0:
        return 0.0
    false_positives = ((y_pred == positive_class) & negatives).sum()
    return float(false_positives / n_negatives)


def compute_false_positive_reduction(
    baseline_fp_rate: float, model_fp_rate: float
) -> float:
    """Computes false positive reduction percentage.

    This is the KEY metric: FP reduction = (baseline_fp - model_fp) / baseline_fp.

    Args:
        baseline_fp_rate: False positive rate of the baseline model.
        model_fp_rate: False positive rate of the CAAA model.

    Returns:
        False positive reduction as a fraction (0.0 to 1.0).
    """
    if baseline_fp_rate == 0.0:
        return 0.0
    return (baseline_fp_rate - model_fp_rate) / baseline_fp_rate


def compute_fault_recall(
    y_true: np.ndarray, y_pred: np.ndarray, fault_class: int = 0
) -> float:
    """Computes recall for the fault class.

    Recall = TP_fault / (TP_fault + FN_fault). Target: >90%.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        fault_class: The class representing faults (default: 0).

    Returns:
        Fault recall value.
    """
    fault_mask = y_true == fault_class
    n_faults = fault_mask.sum()
    if n_faults == 0:
        return 0.0
    true_positives = ((y_pred == fault_class) & fault_mask).sum()
    return float(true_positives / n_faults)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_fp_rate: Optional[float] = None,
) -> Dict[str, float]:
    """Computes all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        baseline_fp_rate: Optional baseline false positive rate for
            computing FP reduction.

    Returns:
        Dictionary with:
            - accuracy: Overall accuracy.
            - precision: Weighted precision.
            - recall: Weighted recall.
            - f1: Weighted F1 score.
            - fp_rate: False positive rate for FAULT class.
            - fault_recall: Recall for the fault class.
            - fp_reduction: FP reduction vs baseline (if baseline_fp_rate provided).
            - attribution_accuracy: Same as accuracy for our binary case.
    """
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fp_rate": compute_false_positive_rate(y_true, y_pred),
        "fault_recall": compute_fault_recall(y_true, y_pred),
    }

    if baseline_fp_rate is not None:
        metrics["fp_reduction"] = compute_false_positive_reduction(
            baseline_fp_rate, metrics["fp_rate"]
        )

    metrics["attribution_accuracy"] = metrics["accuracy"]

    return metrics


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    """Prints a formatted evaluation summary.

    Args:
        metrics: Dictionary of metric names to values, as returned
            by compute_all_metrics.
    """
    summary_lines = [
        "=" * 50,
        "Evaluation Summary",
        "=" * 50,
        f"  Accuracy:            {metrics.get('accuracy', 0.0):.4f}",
        f"  Precision (weighted): {metrics.get('precision', 0.0):.4f}",
        f"  Recall (weighted):    {metrics.get('recall', 0.0):.4f}",
        f"  F1 (weighted):        {metrics.get('f1', 0.0):.4f}",
        "-" * 50,
        f"  False Positive Rate:  {metrics.get('fp_rate', 0.0):.4f}",
        f"  Fault Recall:         {metrics.get('fault_recall', 0.0):.4f}",
    ]

    if "fp_reduction" in metrics:
        summary_lines.append(
            f"  FP Reduction:         {metrics['fp_reduction']:.4f}"
        )

    summary_lines.append("=" * 50)

    summary = "\n".join(summary_lines)
    logger.info("\n%s", summary)
    print(summary)
