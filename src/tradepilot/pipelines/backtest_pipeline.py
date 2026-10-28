"""
Backtest Pipeline Orchestrator
==============================
Command-line entry point:
    python -m tradepilot.pipelines.backtest_pipeline

Orchestrates historical out-of-sample strategy backtesting on validation and test periods,
computes financial risk and return metrics, and serializes auditable reports.
"""

import json
import logging
import sys
from typing import Any, Dict

import joblib
import pandas as pd

from tradepilot.backtesting.engine import BacktestEngine
from tradepilot.backtesting.signals import SignalGenerator
from tradepilot.config import (
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    BacktestConfig,
    DataConfig,
    ModelConfig,
)
from tradepilot.data.load_data import load_processed_features, load_raw_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradepilot.backtest_pipeline")


def run_backtest_pipeline(
    data_config: DataConfig = DEFAULT_DATA_CONFIG,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
) -> Dict[str, Any]:
    """
    Execute out-of-sample backtest evaluation on validation (2024) and test (2025) periods.
    """
    logger.info("=" * 80)
    logger.info("Starting TradePilot Backtesting Pipeline")
    logger.info("=" * 80)

    # 1. Check model artifact existence
    if not model_config.best_model_path.exists():
        raise FileNotFoundError(
            f"Fitted model artifact not found at {model_config.best_model_path}. "
            "Please run train_pipeline first."
        )

    logger.info("Loading fitted model from: %s", model_config.best_model_path)
    model_pipeline = joblib.load(model_config.best_model_path)

    # 2. Load market datasets
    raw_ohlcv = load_raw_ohlcv(data_config.raw_ohlcv_path)
    processed_features = load_processed_features(data_config.processed_features_path)

    # 3. Generate model signals across full timeline using backward-looking features
    logger.info("Generating model probability signals...")
    signal_gen = SignalGenerator(
        pipeline=model_pipeline,
        feature_cols=data_config.feature_cols,
        entry_threshold=backtest_config.entry_threshold,
    )
    signals_df = signal_gen.generate_signals(processed_features)

    # 4. Initialize engine
    engine = BacktestEngine(config=backtest_config)

    # 5. Run Validation Backtest (2024)
    val_start = pd.Timestamp(f"{model_config.val_year}-01-01", tz="UTC")
    val_end = pd.Timestamp(f"{model_config.val_year}-12-31", tz="UTC")
    logger.info("Executing Validation Period Backtest (2024)...")
    val_result = engine.run(
        ohlcv_df=raw_ohlcv,
        signals_df=signals_df,
        start_date=val_start,
        end_date=val_end,
    )

    # 6. Run Final Test Backtest (2025) using frozen settings
    test_start = pd.Timestamp(f"{model_config.test_year}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{model_config.test_year}-12-31", tz="UTC")
    logger.info("Executing Final Test Period Backtest (2025)...")
    test_result = engine.run(
        ohlcv_df=raw_ohlcv,
        signals_df=signals_df,
        start_date=test_start,
        end_date=test_end,
    )

    # 7. Display formatted summary tables
    print("\n" + "=" * 90)
    print("                     BACKTEST PERFORMANCE SUMMARY: 2025 FINAL TEST")
    print("=" * 90)
    sm = test_result.strategy_metrics
    bm = test_result.benchmark_metrics
    print(f"  Initial Capital            : ${backtest_config.initial_capital:,.2f}")
    print(f"  Final Equity               : ${sm['final_equity']:,.2f}")
    print(f"  Total Return               : {sm['total_return'] * 100:>7.2f}% (Benchmark: {bm['total_return'] * 100:>7.2f}%)")
    print(f"  Annualized Return (CAGR)   : {sm['annualized_return'] * 100:>7.2f}% (Benchmark: {bm['annualized_return'] * 100:>7.2f}%)")
    print(f"  Annualized Volatility      : {sm['annualized_volatility'] * 100:>7.2f}% (Benchmark: {bm['annualized_volatility'] * 100:>7.2f}%)")
    print(f"  Sharpe Ratio (Rf=0%)       : {sm['sharpe_ratio']:>7.2f}  (Benchmark: {bm['sharpe_ratio']:>7.2f})")
    print(f"  Maximum Drawdown           : {sm['max_drawdown'] * 100:>7.2f}% (Benchmark: {bm['max_drawdown'] * 100:>7.2f}%)")
    print("-" * 90)
    print(f"  Total Closed Trades        : {sm['total_trades']:,}")
    print(f"  Win Rate (% Profitable)    : {sm['win_rate'] * 100:>7.2f}%")
    print(f"  Profit Factor              : {sm['profit_factor']:>7.2f}")
    print(f"  Average Win Return         : {sm['avg_win_pct'] * 100:>7.2f}%")
    print(f"  Average Loss Return        : {sm['avg_loss_pct'] * 100:>7.2f}%")
    print(f"  Portfolio Turnover         : {sm['turnover']:>7.2f}x")
    print(f"  Total Modeled Costs        : ${sm['total_costs']:,.2f} (Slippage: {backtest_config.slippage_bps} bps, Comm: {backtest_config.commission_bps} bps)")
    print(f"  Average Capital Exposure   : {sm['avg_exposure'] * 100:>7.2f}%")
    print(f"  Time in Market             : {sm['time_in_market_pct'] * 100:>7.2f}%")
    print("=" * 90)

    # 8. Save artifacts to reports/
    backtest_config.equity_curve_path.parent.mkdir(parents=True, exist_ok=True)

    # Save daily equity curve
    test_result.equity_curve.to_csv(backtest_config.equity_curve_path, index=False)
    logger.info("Saved daily equity curve to: %s", backtest_config.equity_curve_path)

    # Save trade ledger
    test_result.trade_ledger.to_csv(backtest_config.trade_ledger_path, index=False)
    logger.info("Saved trade ledger to: %s", backtest_config.trade_ledger_path)

    # Save benchmark comparison
    bench_records = [
        {
            "strategy": "TradePilot_ML_LongOnly",
            "period": "2025_test",
            "total_return": sm["total_return"],
            "annualized_return": sm["annualized_return"],
            "annualized_volatility": sm["annualized_volatility"],
            "sharpe_ratio": sm["sharpe_ratio"],
            "max_drawdown": sm["max_drawdown"],
        },
        {
            "strategy": "Equal_Weight_Buy_and_Hold",
            "period": "2025_test",
            "total_return": bm["total_return"],
            "annualized_return": bm["annualized_return"],
            "annualized_volatility": bm["annualized_volatility"],
            "sharpe_ratio": bm["sharpe_ratio"],
            "max_drawdown": bm["max_drawdown"],
        },
        {
            "strategy": "Cash_No_Trade",
            "period": "2025_test",
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        },
    ]
    pd.DataFrame(bench_records).to_csv(backtest_config.benchmark_comparison_path, index=False)
    logger.info("Saved benchmark comparison to: %s", backtest_config.benchmark_comparison_path)

    # Save full summary JSON
    summary_data = {
        "validation_2024": {
            "strategy_metrics": val_result.strategy_metrics,
            "benchmark_metrics": val_result.benchmark_metrics,
        },
        "test_2025": {
            "strategy_metrics": test_result.strategy_metrics,
            "benchmark_metrics": test_result.benchmark_metrics,
        },
        "config": test_result.config,
    }
    with open(backtest_config.summary_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved backtest summary JSON to: %s", backtest_config.summary_metrics_path)

    logger.info("Backtesting pipeline completed successfully.")
    return summary_data


def main() -> None:
    try:
        run_backtest_pipeline()
    except Exception as exc:
        logger.exception("Backtest pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
