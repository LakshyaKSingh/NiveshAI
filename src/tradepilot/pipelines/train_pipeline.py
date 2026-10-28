"""
Training Pipeline Orchestrator
==============================
Command-line entry point:
    python -m tradepilot.pipelines.train_pipeline

Coordinates data validation, chronological splitting, model fitting,
validation comparison, and final test evaluation.
"""

import logging
import sys
from typing import Any, Dict

import numpy as np

from tradepilot.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DataConfig,
    ModelConfig,
)
from tradepilot.data.load_data import load_processed_features
from tradepilot.data.validate_data import validate_features_dataset
from tradepilot.modeling.evaluate_model import evaluate_classifier, select_best_model
from tradepilot.modeling.save_model import save_model_artifacts
from tradepilot.modeling.split_data import chronological_split
from tradepilot.modeling.train_model import get_candidate_pipelines, train_candidates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradepilot.train_pipeline")


def run_train_pipeline(
    data_config: DataConfig = DEFAULT_DATA_CONFIG,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> Dict[str, Any]:
    """
    Execute the end-to-end model training, validation, and evaluation pipeline.
    """
    logger.info("=" * 80)
    logger.info("Starting TradePilot ML Training Pipeline")
    logger.info("=" * 80)

    # 1. Load and validate processed dataset
    df = load_processed_features(data_config.processed_features_path)
    validate_features_dataset(
        df=df,
        feature_cols=data_config.feature_cols,
        target_col=data_config.target_col,
        prohibited_cols=data_config.prohibited_cols,
        expected_classes=data_config.target_classes,
    )

    # 2. Chronological split with boundary purging
    train_df, val_df, test_df, split_info = chronological_split(
        df=df,
        train_end_year=model_config.train_end_year,
        val_year=model_config.val_year,
        test_year=model_config.test_year,
        purge_days=data_config.purge_boundary_days,
    )

    X_train = train_df[data_config.feature_cols]
    y_train = train_df[data_config.target_col]

    X_val = val_df[data_config.feature_cols]
    y_val = val_df[data_config.target_col]

    X_test = test_df[data_config.feature_cols]
    y_test = test_df[data_config.target_col]

    # 3. Fit candidate models on Training fold
    candidate_pipelines = get_candidate_pipelines(random_seed=model_config.random_seed)
    fitted_pipelines = train_candidates(candidate_pipelines, X_train, y_train)

    # 4. Evaluate candidates on Validation fold (2024)
    val_metrics = {}
    for name, pipe in fitted_pipelines.items():
        val_metrics[name] = evaluate_classifier(
            pipeline=pipe,
            X=X_val,
            y=y_val,
            target_classes=data_config.target_classes,
        )

    # Display comparison table
    print("\n" + "=" * 90)
    print("                     VALIDATION COMPARISON TABLE (2024)")
    print("=" * 90)
    print(f"{'Model':<22} | {'Macro F1':>9} | {'Bal Acc':>8} | {'Accuracy':>8} | {'DOWN F1':>8} | {'NEU F1':>8} | {'UP F1':>8}")
    print("-" * 90)
    for name, m in val_metrics.items():
        down_f1 = m["per_class"]["DOWN"]["f1_score"]
        neu_f1 = m["per_class"]["NEUTRAL"]["f1_score"]
        up_f1 = m["per_class"]["UP"]["f1_score"]
        print(
            f"{name:<22} | {m['macro_f1']:>9.4f} | {m['balanced_accuracy']:>8.4f} | "
            f"{m['accuracy']:>8.4f} | {down_f1:>8.4f} | {neu_f1:>8.4f} | {up_f1:>8.4f}"
        )
    print("=" * 90)

    # 5. Select best candidate based on validation Macro F1
    best_model_name = select_best_model(val_metrics, selection_metric="macro_f1")
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\nSelected Candidate: '{best_model_name}' (Validation Macro F1: {val_metrics[best_model_name]['macro_f1']:.4f})")

    # 6. Single evaluation on final test set (2025)
    print("\n" + "=" * 70)
    print(f"          FINAL 2025 TEST PERFORMANCE: {best_model_name}")
    print("=" * 70)
    test_metrics = evaluate_classifier(
        pipeline=best_pipeline,
        X=X_test,
        y=y_test,
        target_classes=data_config.target_classes,
    )
    print(f"  Macro F1           : {test_metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy  : {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Overall Accuracy   : {test_metrics['accuracy']:.4f}")
    print("\n  Per-Class Metrics:")
    for cls in data_config.target_classes:
        p = test_metrics["per_class"][cls]["precision"]
        r = test_metrics["per_class"][cls]["recall"]
        f = test_metrics["per_class"][cls]["f1_score"]
        s = test_metrics["per_class"][cls]["support"]
        print(f"    {cls:>7} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f} | Support: {s:,}")

    cm_arr = np.array(test_metrics["confusion_matrix"])
    print("\n  Confusion Matrix (Rows: True, Cols: Predicted [DOWN, NEUTRAL, UP]):")
    print(f"             Pred DOWN  Pred NEU   Pred UP")
    print(f"    True DOWN  {cm_arr[0, 0]:>9} {cm_arr[0, 1]:>9} {cm_arr[0, 2]:>9}")
    print(f"    True NEU   {cm_arr[1, 0]:>9} {cm_arr[1, 1]:>9} {cm_arr[1, 2]:>9}")
    print(f"    True UP    {cm_arr[2, 0]:>9} {cm_arr[2, 1]:>9} {cm_arr[2, 2]:>9}")
    print("=" * 70)

    # 7. Persist model and report artifacts
    saved_paths = save_model_artifacts(
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        split_info=split_info,
        data_config=data_config,
        model_config=model_config,
    )

    logger.info("Training pipeline completed successfully.")
    return {
        "best_model_name": best_model_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": saved_paths,
    }


def main() -> None:
    try:
        run_train_pipeline()
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
