"""
Model Evaluation Module
=======================
Calculates classification metrics, balanced accuracy, per-class F1,
confusion matrices, and performs model candidate selection.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from tradepilot.config import DEFAULT_DATA_CONFIG

logger = logging.getLogger(__name__)


def evaluate_classifier(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    target_classes: List[str] = DEFAULT_DATA_CONFIG.target_classes,
) -> Dict[str, Any]:
    """
    Evaluate a fitted classifier pipeline on feature matrix X and target y.

    Parameters:
        pipeline: Fitted scikit-learn Pipeline.
        X: Feature DataFrame.
        y: True target labels.
        target_classes: Ordered list of target class names.

    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics dictionary.
    """
    preds = pipeline.predict(X)

    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y, preds))
    acc = float(accuracy_score(y, preds))

    report = classification_report(
        y, preds, labels=target_classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, preds, labels=target_classes)

    metrics = {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "accuracy": acc,
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1_score": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in target_classes
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def select_best_model(
    val_metrics: Dict[str, Dict[str, Any]],
    selection_metric: str = "macro_f1",
) -> str:
    """
    Select the top candidate model based on validation metric, excluding baseline.

    Parameters:
        val_metrics: Dictionary mapping model names to evaluation metrics.
        selection_metric: Metric key to maximize (default 'macro_f1').

    Returns:
        str: Name of the winning candidate model.
    """
    candidates = [k for k in val_metrics.keys() if k != "Majority_Baseline"]
    if not candidates:
        raise ValueError("No candidate models available for selection.")

    best_model_name = max(candidates, key=lambda k: val_metrics[k][selection_metric])
    logger.info(
        "Selected best candidate model '%s' based on validation %s: %.4f",
        best_model_name,
        selection_metric,
        val_metrics[best_model_name][selection_metric],
    )
    return best_model_name
