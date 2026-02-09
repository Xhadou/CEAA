"""Visualization utilities for CAAA evaluation results."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.features.feature_schema import FEATURE_GROUPS as _FEATURE_GROUPS

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for *path* if it does not exist."""
    parent = os.path.dirname(path)
    os.makedirs(parent if parent else ".", exist_ok=True)


# Consistent color palette
_PALETTE = {
    "FAULT": "#e74c3c",
    "EXPECTED_LOAD": "#3498db",
    "UNKNOWN": "#95a5a6",
    "correct": "#2ecc71",
    "incorrect": "#e74c3c",
}

_FEATURE_GROUP_COLORS = {
    "workload": "#3498db",
    "behavioral": "#e74c3c",
    "context": "#2ecc71",
    "statistical": "#f39c12",
    "service-level": "#9b59b6",
}




def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix heatmap with counts and percentages.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Names for each class. Defaults to
            ["FAULT", "EXPECTED_LOAD"] or includes "UNKNOWN" if present.
        save_path: Path to save the figure. If None, displays the plot.
    """
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        unique_labels = sorted(set(y_true.tolist() + y_pred.tolist()))
        default_names = {0: "FAULT", 1: "EXPECTED_LOAD", 2: "UNKNOWN"}
        class_names = [default_names.get(i, str(i)) for i in unique_labels]

    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    # Add counts and percentages as annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = 100.0 * count / total if total > 0 else 0.0
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if count > cm.max() / 2 else "black",
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300)
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close(fig)


def plot_feature_importance(
    model: object,
    feature_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot feature importance as a horizontal bar chart grouped by category.

    For BaselineClassifier (RandomForest), uses ``.feature_importances_``.
    For other models, uses permutation-style random importance as placeholder.

    Args:
        model: A fitted model with ``feature_importances_`` attribute
            or ``model`` attribute containing such.
        feature_names: List of feature names (length 36).
        save_path: Path to save the figure.
    """
    # Extract importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        importances = model.model.feature_importances_
    else:
        logger.warning("Model does not have feature_importances_; using random placeholder values.")
        importances = np.random.uniform(0, 0.1, len(feature_names))

    # Sort by importance
    indices = np.argsort(importances)

    # Assign colors by group
    colors = []
    for idx in indices:
        for group_name, group_indices in _FEATURE_GROUPS.items():
            if idx in group_indices:
                colors.append(_FEATURE_GROUP_COLORS[group_name])
                break
        else:
            colors.append("#7f8c8d")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color=colors,
        align="center",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance by Category")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=n.capitalize())
        for n, c in _FEATURE_GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300)
        logger.info("Feature importance plot saved to %s", save_path)
    plt.close(fig)


def plot_fp_reduction_comparison(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot FP reduction comparison across model variants.

    Args:
        results_dict: Mapping of variant name to dict with keys
            'fp_reduction_mean' and 'fp_reduction_std'.
        save_path: Path to save the figure.
    """
    variants = list(results_dict.keys())
    means = [results_dict[v].get("fp_reduction_mean", 0.0) * 100 for v in variants]
    stds = [results_dict[v].get("fp_reduction_std", 0.0) * 100 for v in variants]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(variants)),
        means,
        yerr=stds,
        capsize=5,
        color=sns.color_palette("viridis", len(variants)),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel("FP Reduction (%)")
    ax.set_title("False Positive Reduction Comparison")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300)
        logger.info("FP reduction comparison saved to %s", save_path)
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves.

    If ContextConsistencyLoss component losses are present, plots them
    as well on a secondary subplot.

    Args:
        history: Training history dict with keys like 'train_loss',
            'val_loss', 'cls_loss', 'consistency_loss', 'calibration_loss'.
        save_path: Path to save the figure.
    """
    has_components = "cls_loss" in history

    if has_components:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

    # Main loss curves
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#3498db")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Component losses
    if has_components:
        if "cls_loss" in history:
            ax2.plot(epochs, history["cls_loss"], label="Classification", color="#e74c3c")
        if "consistency_loss" in history:
            ax2.plot(epochs, history["consistency_loss"], label="Consistency", color="#3498db")
        if "calibration_loss" in history:
            ax2.plot(epochs, history["calibration_loss"], label="Calibration", color="#2ecc71")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss Component")
        ax2.set_title("Loss Components")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300)
        logger.info("Training curves saved to %s", save_path)
    plt.close(fig)


def plot_confidence_distribution(
    confidences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot histogram of prediction confidences colored by correctness.

    Args:
        confidences: Max softmax probability per sample.
        y_true: Ground truth labels.
        y_pred: Predicted labels (may include class 2 = UNKNOWN).
        save_path: Path to save the figure.
    """
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    fig, ax = plt.subplots(figsize=(8, 5))

    if correct_mask.sum() > 0:
        ax.hist(
            confidences[correct_mask],
            bins=20,
            alpha=0.6,
            color=_PALETTE["correct"],
            label="Correct",
            edgecolor="black",
            linewidth=0.5,
        )
    if incorrect_mask.sum() > 0:
        ax.hist(
            confidences[incorrect_mask],
            bins=20,
            alpha=0.6,
            color=_PALETTE["incorrect"],
            label="Incorrect",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution by Correctness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300)
        logger.info("Confidence distribution saved to %s", save_path)
    plt.close(fig)


# ── SHAP-based feature importance ─────────────────────────────────────


def _get_shap_explainer(model: object, X_background: np.ndarray):
    """Create an appropriate SHAP explainer for the given model.

    Uses ``TreeExplainer`` for tree-based models (RandomForest, XGBoost)
    and ``KernelExplainer`` for neural models or other opaque models.

    Args:
        model: A fitted model with a ``predict_proba`` method.
        X_background: Background dataset for KernelExplainer (typically
            50 representative training samples).

    Returns:
        A SHAP explainer instance.
    """
    import shap

    # Tree-based models (sklearn RF, XGBoost)
    inner = getattr(model, "model", model)
    if hasattr(inner, "estimators_") or type(inner).__name__ == "XGBClassifier":
        return shap.TreeExplainer(inner)

    # Fallback: KernelExplainer for neural/opaque models
    return shap.KernelExplainer(model.predict_proba, X_background, silent=True)


def plot_shap_summary(
    model: object,
    X_test: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    X_background: Optional[np.ndarray] = None,
    nsamples: int = 100,
) -> None:
    """Generate SHAP beeswarm summary plot showing global feature importance.

    For sklearn tree models, uses ``TreeExplainer`` (fast).
    For CAAA neural model, uses ``KernelExplainer`` with a background of
    50 samples (slower — use ``nsamples`` to control runtime).

    Args:
        model: Fitted model with ``predict_proba`` method.
        X_test: Test features of shape (n_samples, n_features).
        feature_names: List of feature names (length 36).
        save_path: Path to save the figure.
        X_background: Background samples for KernelExplainer.
            Defaults to first 50 samples of X_test.
        nsamples: Number of samples for KernelExplainer (ignored for trees).
    """
    import shap

    if X_background is None:
        X_background = X_test[:50]

    explainer = _get_shap_explainer(model, X_background)

    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(X_test)
    else:
        shap_values = explainer.shap_values(X_test, nsamples=nsamples)

    # For binary classification, use class-1 SHAP values if list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig = plt.figure(figsize=(10, 10))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names,
        show=False, plot_size=None,
    )
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("SHAP summary plot saved to %s", save_path)
    plt.close("all")


def plot_shap_by_class(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    X_background: Optional[np.ndarray] = None,
    nsamples: int = 100,
) -> None:
    """Show SHAP importance split by true class (FAULT vs EXPECTED_LOAD).

    Reveals whether the model relies on context features vs metric features
    differently for each class.

    Args:
        model: Fitted model with ``predict_proba`` method.
        X_test: Test features.
        y_test: True labels (0=FAULT, 1=EXPECTED_LOAD).
        feature_names: Feature names.
        save_path: Path to save the figure.
        X_background: Background samples for KernelExplainer.
        nsamples: Number of KernelExplainer samples.
    """
    import shap

    if X_background is None:
        X_background = X_test[:50]

    explainer = _get_shap_explainer(model, X_background)

    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(X_test)
    else:
        shap_values = explainer.shap_values(X_test, nsamples=nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fault_mask = y_test == 0
    load_mask = y_test == 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # FAULT class
    if fault_mask.sum() > 0:
        mean_abs_fault = np.abs(shap_values[fault_mask]).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_fault)[-15:]
        axes[0].barh(
            range(len(sorted_idx)),
            mean_abs_fault[sorted_idx],
            color="#e74c3c",
        )
        axes[0].set_yticks(range(len(sorted_idx)))
        axes[0].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
        axes[0].set_title("FAULT — Top 15 Features")
        axes[0].set_xlabel("Mean |SHAP value|")

    # EXPECTED_LOAD class
    if load_mask.sum() > 0:
        mean_abs_load = np.abs(shap_values[load_mask]).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_load)[-15:]
        axes[1].barh(
            range(len(sorted_idx)),
            mean_abs_load[sorted_idx],
            color="#3498db",
        )
        axes[1].set_yticks(range(len(sorted_idx)))
        axes[1].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
        axes[1].set_title("EXPECTED_LOAD — Top 15 Features")
        axes[1].set_xlabel("Mean |SHAP value|")

    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("SHAP by-class plot saved to %s", save_path)
    plt.close(fig)


def plot_shap_by_fault_type(
    model: object,
    X_test: np.ndarray,
    fault_types: List[Optional[str]],
    feature_names: List[str],
    save_path: Optional[str] = None,
    X_background: Optional[np.ndarray] = None,
    nsamples: int = 100,
) -> None:
    """Show SHAP breakdown per fault type.

    For example, cpu_hog faults should show high importance on cpu-related
    features. Only fault samples (those with a non-None fault_type) are shown.

    Args:
        model: Fitted model with ``predict_proba`` method.
        X_test: Test features.
        fault_types: Per-sample fault type string (None for load cases).
        feature_names: Feature names.
        save_path: Path to save the figure.
        X_background: Background samples for KernelExplainer.
        nsamples: Number of KernelExplainer samples.
    """
    import shap

    if X_background is None:
        X_background = X_test[:50]

    explainer = _get_shap_explainer(model, X_background)

    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(X_test)
    else:
        shap_values = explainer.shap_values(X_test, nsamples=nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Group by fault type
    unique_types = sorted(set(ft for ft in fault_types if ft is not None))
    if not unique_types:
        logger.warning("No fault types available; skipping per-fault-type SHAP plot.")
        return

    n_types = min(len(unique_types), 6)  # limit to 6 subplots
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 6))
    if n_types == 1:
        axes = [axes]

    for ax, ft in zip(axes, unique_types[:n_types]):
        mask = np.array([t == ft for t in fault_types])
        if mask.sum() == 0:
            continue
        mean_abs = np.abs(shap_values[mask]).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[-10:]
        ax.barh(range(len(sorted_idx)), mean_abs[sorted_idx], color="#e74c3c")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=7)
        ax.set_title(ft, fontsize=9)
        ax.set_xlabel("Mean |SHAP|")

    plt.suptitle("SHAP by Fault Type", fontsize=12)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("SHAP by-fault-type plot saved to %s", save_path)
    plt.close(fig)
