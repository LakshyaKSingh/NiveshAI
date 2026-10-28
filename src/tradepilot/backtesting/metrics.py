"""
Backtesting Performance Metrics Module
======================================
Computes financial portfolio performance and risk metrics (Sharpe, Drawdown,
CAGR, Win Rate, Turnover) and benchmark comparisons.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_backtest_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate comprehensive strategy performance and risk metrics.

    Parameters:
        equity_df: DataFrame containing daily marked-to-market equity.
        trades_df: DataFrame containing closed trades ledger.
        initial_capital: Initial capital allocated to portfolio.
        trading_days_per_year: Annualization factor (default 252).
        risk_free_rate: Annualized risk-free rate (default 0.0).

    Returns:
        Dict[str, Any]: Dictionary of calculated metrics.
    """
    if equity_df.empty or len(equity_df) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "turnover": 0.0,
            "total_costs": 0.0,
            "avg_exposure": 0.0,
            "time_in_market_pct": 0.0,
        }

    # Equity analytics
    final_equity = equity_df["total_equity"].iloc[-1]
    total_return = (final_equity / initial_capital) - 1.0

    num_days = len(equity_df)
    years = num_days / trading_days_per_year
    cagr = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) if years > 0 and final_equity > 0 else 0.0

    daily_returns = equity_df["daily_return"].iloc[1:]  # skip first day (0 return)
    daily_vol = daily_returns.std()
    ann_vol = daily_vol * np.sqrt(trading_days_per_year) if daily_vol > 0 else 0.0

    excess_daily_return = daily_returns.mean() - (risk_free_rate / trading_days_per_year)
    sharpe = (excess_daily_return / daily_vol * np.sqrt(trading_days_per_year)) if daily_vol > 0 else 0.0

    # Drawdown
    running_max = equity_df["total_equity"].cummax()
    drawdown = (equity_df["total_equity"] - running_max) / running_max
    max_drawdown = float(drawdown.min())  # negative float

    # Trade analytics
    total_trades = len(trades_df)
    if total_trades > 0:
        profitable_trades = trades_df[trades_df["net_pnl"] > 0]
        losing_trades = trades_df[trades_df["net_pnl"] < 0]

        win_rate = len(profitable_trades) / total_trades
        gross_profits = profitable_trades["net_pnl"].sum()
        gross_losses = abs(losing_trades["net_pnl"].sum())
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (np.inf if gross_profits > 0 else 0.0)

        avg_win_pct = profitable_trades["return_pct"].mean() if not profitable_trades.empty else 0.0
        avg_loss_pct = losing_trades["return_pct"].mean() if not losing_trades.empty else 0.0
        total_costs = float(trades_df["total_costs"].sum())

        total_traded_value = (trades_df["shares"] * trades_df["entry_price"]).sum() + (trades_df["shares"] * trades_df["exit_price"]).sum()
        avg_equity = equity_df["total_equity"].mean()
        turnover = (total_traded_value / (2.0 * avg_equity)) if avg_equity > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win_pct = 0.0
        avg_loss_pct = 0.0
        total_costs = 0.0
        turnover = 0.0

    avg_exposure = float(equity_df["gross_exposure"].mean())
    days_in_market = (equity_df["positions_count"] > 0).sum()
    time_in_market_pct = float(days_in_market / num_days)

    return {
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "annualized_return": float(cagr),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "total_trades": int(total_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 999.0,
        "avg_win_pct": float(avg_win_pct),
        "avg_loss_pct": float(avg_loss_pct),
        "turnover": float(turnover),
        "total_costs": float(total_costs),
        "avg_exposure": float(avg_exposure),
        "time_in_market_pct": float(time_in_market_pct),
    }


def calculate_benchmark_metrics(
    raw_ohlcv_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Compute performance metrics for an Equal-Weighted Buy-and-Hold benchmark
    over the identical universe and time period.

    Parameters:
        raw_ohlcv_df: Raw historical OHLCV data.
        start_date: Start date of evaluation.
        end_date: End date of evaluation.
        trading_days_per_year: Annualization constant (252).

    Returns:
        Dict[str, Any]: Benchmark performance dictionary.
    """
    df_period = raw_ohlcv_df[
        (raw_ohlcv_df["timestamp"] >= start_date) & (raw_ohlcv_df["timestamp"] <= end_date)
    ].copy()

    if df_period.empty:
        return {"total_return": 0.0, "annualized_return": 0.0, "annualized_volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    # Pivot to get daily closing prices per symbol
    pivot_close = df_period.pivot(index="timestamp", columns="symbol", values="close")
    pivot_close = pivot_close.ffill().dropna()

    if pivot_close.empty or len(pivot_close) < 2:
        return {"total_return": 0.0, "annualized_return": 0.0, "annualized_volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    # Equal-weighted daily returns of the universe
    daily_returns = pivot_close.pct_change().mean(axis=1).iloc[1:]
    cum_returns = (1.0 + daily_returns).cumprod()

    total_return = float(cum_returns.iloc[-1] - 1.0)
    num_days = len(daily_returns)
    years = num_days / trading_days_per_year
    cagr = float((cum_returns.iloc[-1]) ** (1.0 / years) - 1.0) if years > 0 and cum_returns.iloc[-1] > 0 else 0.0

    daily_vol = daily_returns.std()
    ann_vol = float(daily_vol * np.sqrt(trading_days_per_year)) if daily_vol > 0 else 0.0
    sharpe = float((daily_returns.mean() / daily_vol) * np.sqrt(trading_days_per_year)) if daily_vol > 0 else 0.0

    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = float(drawdown.min())

    return {
        "benchmark_name": "Equal_Weight_Buy_and_Hold",
        "total_return": total_return,
        "annualized_return": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
