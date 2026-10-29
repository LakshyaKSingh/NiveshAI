"""
MLflow Experiment Tracking Module for TradePilot
=================================================
Provides centralized tracking utilities for recording machine-learning parameters,
classification metrics, backtesting performance, and artifact files into local MLflow runs.
All tracking paths are local and portable (stored in PROJECT_ROOT / mlruns).
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from tradepilot.config import (
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    PROJECT_ROOT,
    BacktestConfig,
    DataConfig,
    ModelConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "TradePilot-Stock-Direction"


def setup_mlflow_tracking(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_dir: Optional[Path] = None,
) -> str:
    """
    Configure local-first MLflow tracking URI using SQLite backend (sqlite:///mlflow.db)
    and local artifact directory, ensuring full compatibility with modern MLflow.

    Parameters:
        experiment_name: Name of the MLflow experiment.
        tracking_dir: Directory where mlruns/ is stored. Defaults to project root.

    Returns:
        str: Active experiment ID.
    """
    db_path = (PROJECT_ROOT / "mlflow.db").resolve()
    artifacts_dir = (PROJECT_ROOT / "mlruns").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_loc = f"file:///{artifacts_dir.as_posix()}"
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_loc,
        )
        logger.info("Created MLflow experiment '%s' (ID: %s, URI: %s)", experiment_name, experiment_id, tracking_uri)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_training_to_mlflow(
    best_model_name: str,
    best_pipeline: Pipeline,
    val_metrics: Dict[str, Dict[str, Any]],
    test_metrics: Dict[str, Any],
    split_info: Dict[str, Any],
    data_config: DataConfig = DEFAULT_DATA_CONFIG,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> Optional[str]:
    """
    Log an end-to-end model training experiment to MLflow.

    Logs:
        - Parameters: Model hyperparameters, feature list, thresholds, split dates/rows.
        - Metrics: Validation & test accuracy, balanced accuracy, macro F1, and per-class stats.
        - Artifacts: Reports, confusion matrix, metadata JSON, feature importances, and model artifact.

    Returns:
        Optional[str]: Active MLflow run_id if successful, None otherwise.
    """
    try:
        setup_mlflow_tracking(experiment_name=experiment_name)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{best_model_name}_{timestamp_str}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info("Started MLflow run '%s' (ID: %s)", run_name, run_id)

            # ------------------------------------------------------------------
            # 1. Tags
            # ------------------------------------------------------------------
            mlflow.set_tags({
                "project": "TradePilot",
                "framework": "scikit-learn",
                "model_type": best_model_name,
                "dataset_version": "2021_2025_daily",
                "dvc_pipeline_stage": "train",
            })

            # ------------------------------------------------------------------
            # 2. Parameters
            # ------------------------------------------------------------------
            classifier = best_pipeline.named_steps.get("classifier", best_pipeline)
            clf_params = classifier.get_params() if hasattr(classifier, "get_params") else {}

            params_to_log: Dict[str, Any] = {
                "selected_model": best_model_name,
                "random_seed": model_config.random_seed,
                "feature_count": len(data_config.feature_cols),
                "features_list": ",".join(data_config.feature_cols),
                "target_col": data_config.target_col,
                "target_classes": ",".join(data_config.target_classes),
                "target_threshold": data_config.target_threshold,
                "prediction_horizon_days": data_config.prediction_horizon_days,
                "purge_boundary_days": data_config.purge_boundary_days,
                "train_start": split_info.get("train_start", ""),
                "train_end": split_info.get("train_end", ""),
                "train_rows": split_info.get("train_rows", 0),
                "val_start": split_info.get("val_start", ""),
                "val_end": split_info.get("val_end", ""),
                "val_rows": split_info.get("val_rows", 0),
                "test_start": split_info.get("test_start", ""),
                "test_end": split_info.get("test_end", ""),
                "test_rows": split_info.get("test_rows", 0),
            }

            # Add model-specific hyperparameters (filter out overly verbose objects)
            for k, v in clf_params.items():
                if isinstance(v, (int, float, str, bool)):
                    params_to_log[f"hparam_{k}"] = v

            mlflow.log_params(params_to_log)

            # ------------------------------------------------------------------
            # 3. Validation Metrics across Candidates
            # ------------------------------------------------------------------
            for cand_name, cm in val_metrics.items():
                mlflow.log_metric(f"val_{cand_name}_macro_f1", float(cm["macro_f1"]))
                mlflow.log_metric(f"val_{cand_name}_balanced_acc", float(cm["balanced_accuracy"]))
                mlflow.log_metric(f"val_{cand_name}_accuracy", float(cm["accuracy"]))

            # ------------------------------------------------------------------
            # 4. Winning Model Validation & Test Metrics
            # ------------------------------------------------------------------
            win_val = val_metrics[best_model_name]
            mlflow.log_metrics({
                "val_macro_f1": float(win_val["macro_f1"]),
                "val_balanced_accuracy": float(win_val["balanced_accuracy"]),
                "val_accuracy": float(win_val["accuracy"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            })

            # Per-class metrics on test set
            for cls in data_config.target_classes:
                if cls in test_metrics.get("per_class", {}):
                    pc = test_metrics["per_class"][cls]
                    mlflow.log_metric(f"test_precision_{cls}", float(pc["precision"]))
                    mlflow.log_metric(f"test_recall_{cls}", float(pc["recall"]))
                    mlflow.log_metric(f"test_f1_{cls}", float(pc["f1_score"]))

            # ------------------------------------------------------------------
            # 5. Feature Importances (if supported, e.g. Random Forest)
            # ------------------------------------------------------------------
            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
                feat_names = data_config.feature_cols
                imp_df = pd.DataFrame({
                    "feature": feat_names,
                    "importance": importances,
                }).sort_values(by="importance", ascending=False)

                imp_path = model_config.reports_dir / "feature_importances.csv"
                imp_df.to_csv(imp_path, index=False)
                mlflow.log_artifact(str(imp_path), artifact_path="feature_importance")

                for _, row in imp_df.iterrows():
                    mlflow.log_metric(f"feat_imp_{row['feature']}", float(row["importance"]))

            # ------------------------------------------------------------------
            # 6. Artifact Files
            # ------------------------------------------------------------------
            for report_path in [
                model_config.metadata_path,
                model_config.val_results_path,
                model_config.test_results_path,
                model_config.confusion_matrix_path,
            ]:
                if report_path.exists():
                    mlflow.log_artifact(str(report_path), artifact_path="evaluation_reports")

            if model_config.best_model_path.exists():
                mlflow.log_artifact(str(model_config.best_model_path), artifact_path="model_joblib")

            try:
                mlflow.sklearn.log_model(
                    sk_model=best_pipeline,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                )
            except Exception as model_err:
                logger.warning("Could not log model artifact via mlflow.sklearn: %s", model_err)

            logger.info("Successfully logged training experiment to MLflow run: %s", run_id)
            return run_id

    except Exception as exc:
        logger.warning("MLflow tracking encountered an error (execution continued): %s", exc)
        return None


def log_backtest_to_mlflow(
    summary_data: Dict[str, Any],
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Log historical backtesting metrics and artifact files to MLflow.

    Parameters:
        summary_data: Backtest results dictionary containing validation and test metrics.
        backtest_config: Backtesting configuration parameters.
        experiment_name: MLflow experiment name.
        run_id: Optional parent training run_id to attach backtest metrics to.
    """
    try:
        setup_mlflow_tracking(experiment_name=experiment_name)

        # If a run_id is supplied, log backtest directly into that existing run
        active_context = (
            mlflow.start_run(run_id=run_id)
            if run_id
            else mlflow.start_run(run_name=f"Backtest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        )

        with active_context as run:
            actual_run_id = run.info.run_id

            # Parameters
            mlflow.log_params({
                "backtest_initial_capital": backtest_config.initial_capital,
                "backtest_max_positions": backtest_config.max_positions,
                "backtest_allocation_per_pos": backtest_config.allocation_per_position,
                "backtest_entry_threshold": backtest_config.entry_threshold,
                "backtest_holding_period_days": backtest_config.holding_period_days,
                "backtest_slippage_bps": backtest_config.slippage_bps,
                "backtest_commission_bps": backtest_config.commission_bps,
                "backtest_allow_short": backtest_config.allow_short,
                "backtest_benchmark_name": backtest_config.benchmark_name,
            })

            # Strategy & Benchmark Metrics for Test 2025
            test_data = summary_data.get("test_2025", {})
            sm = test_data.get("strategy_metrics", {})
            bm = test_data.get("benchmark_metrics", {})

            metrics_to_log = {}
            for k, v in sm.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    metrics_to_log[f"test_{k}"] = float(v)

            for k, v in bm.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    metrics_to_log[f"test_bench_{k}"] = float(v)

            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log)

            # Artifacts
            for artifact_path in [
                backtest_config.summary_metrics_path,
                backtest_config.equity_curve_path,
                backtest_config.trade_ledger_path,
                backtest_config.benchmark_comparison_path,
            ]:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path), artifact_path="backtest_reports")

            logger.info("Successfully logged backtest results to MLflow run: %s", actual_run_id)
            return actual_run_id

    except Exception as exc:
        logger.warning("MLflow backtest tracking encountered an error: %s", exc)
        return None
