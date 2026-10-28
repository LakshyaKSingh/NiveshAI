"""
Tests for Execution, Friction Accounting, and Portfolio Management
==================================================================
"""

import pandas as pd
import pytest

from tradepilot.backtesting.execution import ExecutionHandler, Order, OrderSide
from tradepilot.backtesting.portfolio import Portfolio


def test_execution_slippage_and_fees():
    # 10 bps slippage (0.0010) and 10 bps commission (0.0010)
    handler = ExecutionHandler(slippage_bps=10.0, commission_bps=10.0)

    # 1. Buy execution
    order_buy = Order(symbol="AAPL", side=OrderSide.BUY)
    fill_buy = handler.execute_order(
        order=order_buy,
        open_price=100.0,
        execution_date=pd.Timestamp("2024-01-02", tz="UTC"),
        available_cash=10_000.0,
        max_position_value=10_000.0,
    )

    assert fill_buy is not None
    # Buy price with adverse slippage: 100 * (1 + 0.001) = 100.10
    assert pytest.approx(fill_buy.fill_price) == 100.10
    assert fill_buy.shares == int(10_000 // (100.10 * 1.001))
    assert fill_buy.commission_cost > 0.0
    assert fill_buy.slippage_cost > 0.0

    # 2. Sell execution
    order_sell = Order(symbol="AAPL", side=OrderSide.SELL, shares=fill_buy.shares)
    fill_sell = handler.execute_order(
        order=order_sell,
        open_price=110.0,
        execution_date=pd.Timestamp("2024-01-08", tz="UTC"),
        available_cash=0.0,
        max_position_value=0.0,
    )

    assert fill_sell is not None
    # Sell price with adverse slippage: 110 * (1 - 0.001) = 109.89
    assert pytest.approx(fill_sell.fill_price) == 109.89


def test_portfolio_cash_accounting_and_round_trip():
    portfolio = Portfolio(initial_capital=10_000.0, max_positions=5, allocation_per_position=0.20)
    handler = ExecutionHandler(slippage_bps=0.0, commission_bps=0.0)

    # Buy 50 shares @ $100
    order_buy = Order(symbol="MSFT", side=OrderSide.BUY)
    fill_buy = handler.execute_order(
        order=order_buy,
        open_price=100.0,
        execution_date=pd.Timestamp("2024-01-02", tz="UTC"),
        available_cash=portfolio.cash,
        max_position_value=2_000.0,  # 20% limit
    )
    portfolio.apply_fill(fill_buy)

    assert fill_buy.shares == 20
    assert pytest.approx(portfolio.cash) == 8_000.0
    assert len(portfolio.positions) == 1

    # Mark to market @ $110
    portfolio.mark_to_market(
        current_date=pd.Timestamp("2024-01-02", tz="UTC"),
        close_prices={"MSFT": 110.0},
    )
    # Holdings value: 20 * 110 = 2200. Total equity = 8000 + 2200 = 10200
    assert pytest.approx(portfolio.total_equity) == 10_200.0

    # Sell 20 shares @ $110
    order_sell = Order(symbol="MSFT", side=OrderSide.SELL, shares=20)
    fill_sell = handler.execute_order(
        order=order_sell,
        open_price=110.0,
        execution_date=pd.Timestamp("2024-01-03", tz="UTC"),
        available_cash=portfolio.cash,
        max_position_value=0.0,
    )
    portfolio.apply_fill(fill_sell)

    assert len(portfolio.positions) == 0
    assert pytest.approx(portfolio.cash) == 10_200.0
    assert len(portfolio.trades) == 1
    assert pytest.approx(portfolio.trades[0].net_pnl) == 200.0


def test_portfolio_prevents_negative_cash():
    portfolio = Portfolio(initial_capital=500.0)
    handler = ExecutionHandler(slippage_bps=0.0, commission_bps=0.0)

    # Cannot buy a $1000 stock with $500 cash
    order = Order(symbol="Expensive", side=OrderSide.BUY)
    fill = handler.execute_order(
        order=order,
        open_price=1000.0,
        execution_date=pd.Timestamp("2024-01-02", tz="UTC"),
        available_cash=portfolio.cash,
        max_position_value=1000.0,
    )
    assert fill is None  # Zero shares affordable
