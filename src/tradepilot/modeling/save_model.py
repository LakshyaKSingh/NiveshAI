"""
Model Serialization Module
==========================
Persists fitted pipelines, training configurations, metadata, and evaluation metrics
to the models/ and reports/ directories.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from tradepilot.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DataConfig,
    ModelConfig,
)

logger = logging.getLogger(__name__)


def save_model_artifacts(
    best_model_name: str,
    best_pipeline: Pipeline,
    val_metrics: Dict[str, Dict[str, Any]],
    test_metrics: Dict[str, Any],
    split_info: Dict[str, Any],
    data_config: DataConfig = DEFAULT_DATA_CONFIG,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> Dict[str, Path]:
    """
    Serialize fitted pipeline, configuration metadata, and evaluation reports.

    Returns:
        Dict[str, Path]: Paths to all persisted artifacts.
    """
    model_config.models_dir.mkdir(parents=True, exist_ok=True)
    model_config.reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save fitted pipeline
    joblib.dump(best_pipeline, model_config.best_model_path)
    logger.info("Saved fitted pipeline to: %s", model_config.best_model_path)

    # 2. Save metadata JSON
    metadata = {
        "project": "TradePilot ML Stock Direction Prediction",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": model_config.random_seed,
        "selected_model": best_model_name,
        "features": data_config.feature_cols,
        "feature_count": len(data_config.feature_cols),
        "target_column": data_config.target_col,
        "target_classes": data_config.target_classes,
        "target_definition": (
            f"future_return_5d > {data_config.target_threshold} -> UP, "
            f"future_return_5d < -{data_config.target_threshold} -> DOWN, "
            "otherwise NEUTRAL"
        ),
        "split_periods": split_info,
        "pipeline_steps": [name for name, _ in best_pipeline.steps],
        "model_parameters": {
            k: str(v) for k, v in best_pipeline.named_steps["classifier"].get_params().items()
        },
        "validation_macro_f1": val_metrics[best_model_name]["macro_f1"],
        "validation_balanced_accuracy": val_metrics[best_model_name]["balanced_accuracy"],
        "validation_accuracy": val_metrics[best_model_name]["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
    }
    with open(model_config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to: %s", model_config.metadata_path)

    # 3. Save validation comparison CSV
    val_rows = []
    for model_name, m in val_metrics.items():
        row = {
            "model": model_name,
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "accuracy": m["accuracy"],
            "precision_DOWN": m["per_class"]["DOWN"]["precision"],
            "recall_DOWN": m["per_class"]["DOWN"]["recall"],
            "f1_DOWN": m["per_class"]["DOWN"]["f1_score"],
            "precision_NEUTRAL": m["per_class"]["NEUTRAL"]["precision"],
            "recall_NEUTRAL": m["per_class"]["NEUTRAL"]["recall"],
            "f1_NEUTRAL": m["per_class"]["NEUTRAL"]["f1_score"],
            "precision_UP": m["per_class"]["UP"]["precision"],
            "recall_UP": m["per_class"]["UP"]["recall"],
            "f1_UP": m["per_class"]["UP"]["f1_score"],
        }
        val_rows.append(row)
    val_df_report = pd.DataFrame(val_rows)
    val_df_report.to_csv(model_config.val_results_path, index=False)
    logger.info("Saved validation results to: %s", model_config.val_results_path)

    # 4. Save test results JSON
    with open(model_config.test_results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": best_model_name,
                "split": f"test_{model_config.test_year}",
                "metrics": test_metrics,
            },
            f,
            indent=2,
        )
    logger.info("Saved test results to: %s", model_config.test_results_path)

    # 5. Save confusion matrix CSV
    cm_records = []
    for model_name, m in val_metrics.items():
        cm = m["confusion_matrix"]
        for i, true_cls in enumerate(data_config.target_classes):
            cm_records.append({
                "model": model_name,
                "split": f"validation_{model_config.val_year}",
                "true_class": true_cls,
                "pred_DOWN": cm[i][0],
                "pred_NEUTRAL": cm[i][1],
                "pred_UP": cm[i][2],
            })

    test_cm = test_metrics["confusion_matrix"]
    for i, true_cls in enumerate(data_config.target_classes):
        cm_records.append({
            "model": best_model_name,
            "split": f"test_{model_config.test_year}",
            "true_class": true_cls,
            "pred_DOWN": test_cm[i][0],
            "pred_NEUTRAL": test_cm[i][1],
            "pred_UP": test_cm[i][2],
        })

    cm_df = pd.DataFrame(cm_records)
    cm_df.to_csv(model_config.confusion_matrix_path, index=False)
    logger.info("Saved confusion matrices to: %s", model_config.confusion_matrix_path)

    return {
        "best_model": model_config.best_model_path,
        "metadata": model_config.metadata_path,
        "validation_results": model_config.val_results_path,
        "test_results": model_config.test_results_path,
        "confusion_matrix": model_config.confusion_matrix_path,
    }
