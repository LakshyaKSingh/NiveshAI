"""
Backtesting module for event-driven historical simulation.
"""

from tradepilot.backtesting.engine import BacktestEngine, BacktestResult
from tradepilot.backtesting.execution import ExecutionHandler, Order, OrderSide, TradeFill
from tradepilot.backtesting.metrics import calculate_backtest_metrics, calculate_benchmark_metrics
from tradepilot.backtesting.portfolio import Portfolio, Position
from tradepilot.backtesting.signals import SignalGenerator

__all__ = [
    "SignalGenerator",
    "Order",
    "OrderSide",
    "TradeFill",
    "ExecutionHandler",
    "Position",
    "Portfolio",
    "calculate_backtest_metrics",
    "calculate_benchmark_metrics",
    "BacktestEngine",
    "BacktestResult",
]
