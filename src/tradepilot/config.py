"""
TradePilot Central Configuration
================================
Centralized paths, feature definitions, model settings, and backtest configurations.
All paths are resolved relative to the project root using pathlib.Path.
Secrets and API credentials are kept out of this configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml


# Project root resolution: src/tradepilot/config.py -> parent x 3 = project root
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


@dataclass(frozen=True)
class DataConfig:
    """Paths and schemas for data loading and feature processing."""
    raw_ohlcv_path: Path = PROJECT_ROOT / "data" / "raw" / "stock_ohlcv_2021_2025.csv"
    processed_features_path: Path = PROJECT_ROOT / "data" / "processed" / "stock_features_labeled.csv"
    cleaned_eda_path: Path = PROJECT_ROOT / "data" / "processed" / "stock_eda_cleaned.csv"

    # Permitted backward-looking feature columns
    feature_cols: List[str] = field(default_factory=lambda: [
        "return_1d",
        "return_5d",
        "return_10d",
        "volatility_10d",
        "volume_change",
        "RSI_14",
        "MACD",
        "MACD_signal",
    ])

    # Columns strictly prohibited from being model inputs (leaks or identifiers)
    prohibited_cols: Set[str] = field(default_factory=lambda: {
        "symbol",
        "timestamp",
        "target",
        "future_return_5d",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "SMA_10",
        "SMA_20",
        "SMA_50",
    })

    target_col: str = "target"
    target_classes: List[str] = field(default_factory=lambda: ["DOWN", "NEUTRAL", "UP"])
    target_threshold: float = 0.01  # 1.0% return over 5 trading days
    prediction_horizon_days: int = 5
    purge_boundary_days: int = 5


@dataclass(frozen=True)
class ModelConfig:
    """Chronological split periods, seeds, and artifact paths."""
    train_end_year: int = 2023
    val_year: int = 2024
    test_year: int = 2025
    random_seed: int = 42

    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    best_model_path: Path = PROJECT_ROOT / "models" / "best_model.joblib"
    metadata_path: Path = PROJECT_ROOT / "models" / "model_metadata.json"
    val_results_path: Path = PROJECT_ROOT / "reports" / "validation_results.csv"
    test_results_path: Path = PROJECT_ROOT / "reports" / "test_results.json"
    confusion_matrix_path: Path = PROJECT_ROOT / "reports" / "confusion_matrix.csv"


@dataclass(frozen=True)
class BacktestConfig:
    """Trading strategy, execution friction, and portfolio constraints."""
    initial_capital: float = 100_000.0
    max_positions: int = 10
    allocation_per_position: float = 0.10  # Max 10% equity per position
    entry_threshold: float = 0.40  # Model P(UP) required to trigger entry
    holding_period_days: int = 5  # Fixed holding horizon matching prediction target
    slippage_bps: float = 5.0  # 5 bps (0.05%) adverse execution slippage
    commission_bps: float = 5.0  # 5 bps (0.05%) transaction cost per trade
    allow_short: bool = False  # Long-only strategy
    benchmark_name: str = "Buy_and_Hold_Equal_Weight"

    # Output paths for backtest reports
    equity_curve_path: Path = PROJECT_ROOT / "reports" / "backtest_equity_curve.csv"
    trade_ledger_path: Path = PROJECT_ROOT / "reports" / "trade_ledger.csv"
    summary_metrics_path: Path = PROJECT_ROOT / "reports" / "backtest_summary.json"
    benchmark_comparison_path: Path = PROJECT_ROOT / "reports" / "benchmark_comparison.csv"


# Default singleton instances for convenient imports
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()


def load_params(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load pipeline parameters from params.yaml if available.
    Falls back to an empty dictionary if the file does not exist.
    """
    params_file = path or (PROJECT_ROOT / "params.yaml")
    if params_file.exists():
        try:
            with open(params_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def get_data_config(params_path: Optional[Path] = None) -> DataConfig:
    """Instantiate DataConfig with overrides from params.yaml if present."""
    params = load_params(params_path).get("features", {})
    horizon = params.get("horizon_days", DEFAULT_DATA_CONFIG.prediction_horizon_days)
    threshold = params.get("target_threshold", DEFAULT_DATA_CONFIG.target_threshold)
    return DataConfig(
        prediction_horizon_days=horizon,
        target_threshold=threshold,
    )


def get_model_config(params_path: Optional[Path] = None) -> ModelConfig:
    """Instantiate ModelConfig with overrides from params.yaml if present."""
    params = load_params(params_path).get("train", {})
    return ModelConfig(
        random_seed=params.get("random_seed", DEFAULT_MODEL_CONFIG.random_seed),
        train_end_year=params.get("train_end_year", DEFAULT_MODEL_CONFIG.train_end_year),
        val_year=params.get("val_year", DEFAULT_MODEL_CONFIG.val_year),
        test_year=params.get("test_year", DEFAULT_MODEL_CONFIG.test_year),
    )


def get_backtest_config(params_path: Optional[Path] = None) -> BacktestConfig:
    """Instantiate BacktestConfig with overrides from params.yaml if present."""
    params = load_params(params_path).get("backtest", {})
    return BacktestConfig(
        initial_capital=float(params.get("initial_capital", DEFAULT_BACKTEST_CONFIG.initial_capital)),
        max_positions=int(params.get("max_positions", DEFAULT_BACKTEST_CONFIG.max_positions)),
        allocation_per_position=float(params.get("allocation_per_position", DEFAULT_BACKTEST_CONFIG.allocation_per_position)),
        entry_threshold=float(params.get("entry_threshold", DEFAULT_BACKTEST_CONFIG.entry_threshold)),
        holding_period_days=int(params.get("holding_period_days", DEFAULT_BACKTEST_CONFIG.holding_period_days)),
        slippage_bps=float(params.get("slippage_bps", DEFAULT_BACKTEST_CONFIG.slippage_bps)),
        commission_bps=float(params.get("commission_bps", DEFAULT_BACKTEST_CONFIG.commission_bps)),
        allow_short=bool(params.get("allow_short", DEFAULT_BACKTEST_CONFIG.allow_short)),
        benchmark_name=str(params.get("benchmark_name", DEFAULT_BACKTEST_CONFIG.benchmark_name)),
    )
