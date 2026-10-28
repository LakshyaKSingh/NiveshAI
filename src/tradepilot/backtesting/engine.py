"""
Backtesting Engine Module
=========================
Coordinates the chronological event-driven simulation:
Signal(t) -> Order Queue -> Execution(t+1 Open with slippage/fees) -> Mark-to-market(t+1 Close).
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from tradepilot.backtesting.execution import ExecutionHandler, Order, OrderSide
from tradepilot.backtesting.metrics import calculate_backtest_metrics, calculate_benchmark_metrics
from tradepilot.backtesting.portfolio import Portfolio
from tradepilot.config import DEFAULT_BACKTEST_CONFIG, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Contains all outputs from a backtest run."""
    strategy_metrics: Dict[str, Any]
    benchmark_metrics: Dict[str, Any]
    equity_curve: pd.DataFrame
    trade_ledger: pd.DataFrame
    config: Dict[str, Any]


class BacktestEngine:
    """
    Event-driven backtesting engine for historical simulation.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or DEFAULT_BACKTEST_CONFIG
        self.execution_handler = ExecutionHandler(
            slippage_bps=self.config.slippage_bps,
            commission_bps=self.config.commission_bps,
        )

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> BacktestResult:
        """
        Execute the event-driven backtest simulation.

        Parameters:
            ohlcv_df: Full OHLCV DataFrame (must include open, close, symbol, timestamp).
            signals_df: Generated signals DataFrame (timestamp, symbol, signal, prob_up, etc.).
            start_date: Optional simulation start filter.
            end_date: Optional simulation end filter.

        Returns:
            BacktestResult: Comprehensive backtest results, ledger, and metrics.
        """
        logger.info("Initializing backtest simulation...")

        # Filter price and signal data
        prices = ohlcv_df.copy()
        signals = signals_df.copy()

        if start_date is not None:
            prices = prices[prices["timestamp"] >= start_date]
            signals = signals[signals["timestamp"] >= start_date]
        if end_date is not None:
            prices = prices[prices["timestamp"] <= end_date]
            signals = signals[signals["timestamp"] <= end_date]

        # Extract unique dates in sorted chronological order
        unique_dates = sorted(prices["timestamp"].unique())
        if len(unique_dates) < 2:
            raise ValueError(f"Not enough trading sessions for backtest (found {len(unique_dates)}).")

        logger.info(
            "Backtest date range: %s to %s (%d trading sessions)",
            unique_dates[0].strftime("%Y-%m-%d"),
            unique_dates[-1].strftime("%Y-%m-%d"),
            len(unique_dates),
        )

        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions,
            allocation_per_position=self.config.allocation_per_position,
        )

        # Pre-index prices by (timestamp, symbol) for efficient lookups
        price_lookup = {}
        for _, row in prices.iterrows():
            price_lookup[(row["timestamp"], row["symbol"])] = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }

        # Index signals by (timestamp, symbol)
        signal_lookup = {}
        for _, row in signals.iterrows():
            signal_lookup[(row["timestamp"], row["symbol"])] = int(row["signal"])

        pending_orders: List[Order] = []

        # Chronological daily loop
        for t_idx, current_date in enumerate(unique_dates):
            current_close_prices: Dict[str, float] = {}

            # Gather today's available closing prices
            for (ts, sym), pdata in price_lookup.items():
                if ts == current_date:
                    current_close_prices[sym] = pdata["close"]

            # -------------------------------------------------------------
            # 1. Execute Pending Orders at today's Open price (generated at t-1)
            # -------------------------------------------------------------
            if pending_orders:
                # Process SELLs first to free up liquidity deterministically
                sell_orders = [o for o in pending_orders if o.side == OrderSide.SELL]
                buy_orders = sorted([o for o in pending_orders if o.side == OrderSide.BUY], key=lambda x: x.symbol)

                for order in sell_orders:
                    pdata = price_lookup.get((current_date, order.symbol))
                    if pdata:
                        fill = self.execution_handler.execute_order(
                            order=order,
                            open_price=pdata["open"],
                            execution_date=current_date,
                            available_cash=portfolio.cash,
                            max_position_value=0.0,
                        )
                        if fill:
                            portfolio.apply_fill(fill)

                for order in buy_orders:
                    if portfolio.can_open_position(order.symbol):
                        pdata = price_lookup.get((current_date, order.symbol))
                        if pdata:
                            max_alloc = portfolio.get_max_allocation_for_new_position()
                            fill = self.execution_handler.execute_order(
                                order=order,
                                open_price=pdata["open"],
                                execution_date=current_date,
                                available_cash=portfolio.cash,
                                max_position_value=max_alloc,
                            )
                            if fill:
                                portfolio.apply_fill(fill)

                pending_orders.clear()

            # -------------------------------------------------------------
            # 2. Mark portfolio to market at today's Close
            # -------------------------------------------------------------
            portfolio.mark_to_market(current_date, current_close_prices)

            # -------------------------------------------------------------
            # 3. Generate Exit and Entry Orders for execution tomorrow (t+1 Open)
            # -------------------------------------------------------------
            # Check holding period exit condition
            for sym, pos in list(portfolio.positions.items()):
                # Exit if held for specified holding_period_days
                if pos.bars_held >= self.config.holding_period_days:
                    pending_orders.append(
                        Order(symbol=sym, side=OrderSide.SELL, shares=pos.shares, signal_date=current_date)
                    )
                # Or exit if negative/exit signal received
                elif signal_lookup.get((current_date, sym), 0) == -1:
                    pending_orders.append(
                        Order(symbol=sym, side=OrderSide.SELL, shares=pos.shares, signal_date=current_date)
                    )

            # Check new buy signals for symbols not currently held or pending
            queued_sell_symbols = {o.symbol for o in pending_orders if o.side == OrderSide.SELL}
            for (ts, sym), sig in signal_lookup.items():
                if ts == current_date and sig == 1:
                    if sym not in portfolio.positions and sym not in queued_sell_symbols:
                        pending_orders.append(
                            Order(symbol=sym, side=OrderSide.BUY, signal_date=current_date)
                        )

        # Simulation complete: compile outputs
        equity_curve = portfolio.get_equity_curve_df()
        trade_ledger = portfolio.get_trade_ledger_df()

        strategy_metrics = calculate_backtest_metrics(
            equity_df=equity_curve,
            trades_df=trade_ledger,
            initial_capital=self.config.initial_capital,
        )

        benchmark_metrics = calculate_benchmark_metrics(
            raw_ohlcv_df=prices,
            start_date=unique_dates[0],
            end_date=unique_dates[-1],
        )

        config_dict = {
            "initial_capital": self.config.initial_capital,
            "max_positions": self.config.max_positions,
            "allocation_per_position": self.config.allocation_per_position,
            "entry_threshold": self.config.entry_threshold,
            "holding_period_days": self.config.holding_period_days,
            "slippage_bps": self.config.slippage_bps,
            "commission_bps": self.config.commission_bps,
            "allow_short": self.config.allow_short,
            "benchmark_name": self.config.benchmark_name,
        }

        return BacktestResult(
            strategy_metrics=strategy_metrics,
            benchmark_metrics=benchmark_metrics,
            equity_curve=equity_curve,
            trade_ledger=trade_ledger,
            config=config_dict,
        )
