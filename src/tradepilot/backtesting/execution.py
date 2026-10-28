"""
Execution Module
================
Simulates realistic event-driven order fills with next-day open pricing,
adverse slippage, and transaction commissions.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional

import pandas as pd

from tradepilot.config import DEFAULT_BACKTEST_CONFIG

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    """Represents a trading order placed based on signal at date t to execute at date t+1."""
    symbol: str
    side: OrderSide
    target_value: Optional[float] = None
    shares: Optional[int] = None
    signal_date: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class TradeFill:
    """Represents an executed trade transaction."""
    symbol: str
    side: OrderSide
    shares: int
    fill_price: float
    base_price: float
    fill_date: pd.Timestamp
    slippage_cost: float
    commission_cost: float
    total_cost: float

    @property
    def gross_value(self) -> float:
        return self.shares * self.fill_price


class ExecutionHandler:
    """
    Executes orders at next-session Open prices with adverse slippage and fees.
    """

    def __init__(
        self,
        slippage_bps: float = DEFAULT_BACKTEST_CONFIG.slippage_bps,
        commission_bps: float = DEFAULT_BACKTEST_CONFIG.commission_bps,
    ):
        """
        Initialize the ExecutionHandler.

        Parameters:
            slippage_bps: Adverse slippage in basis points (1 bp = 0.0001 = 0.01%).
            commission_bps: Transaction fees in basis points.
        """
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps

    def execute_order(
        self,
        order: Order,
        open_price: float,
        execution_date: pd.Timestamp,
        available_cash: float,
        max_position_value: float,
    ) -> Optional[TradeFill]:
        """
        Execute an order given the next session's Open price and available capital.

        Parameters:
            order: The Order to fill.
            open_price: Next session's Open price.
            execution_date: The date on which execution occurs.
            available_cash: Current liquid cash in portfolio.
            max_position_value: Maximum capital allowed for this position.

        Returns:
            Optional[TradeFill]: Executed fill details, or None if order cannot be filled.
        """
        if open_price <= 0 or pd.isna(open_price):
            logger.warning("Rejecting order for %s: invalid open price %s", order.symbol, open_price)
            return None

        slippage_mult = self.slippage_bps / 10_000.0
        commission_mult = self.commission_bps / 10_000.0

        if order.side == OrderSide.BUY:
            # Adverse slippage increases buy fill price
            fill_price = open_price * (1.0 + slippage_mult)
            effective_cost_per_share = fill_price * (1.0 + commission_mult)

            # Determine shares to buy bounded by available cash and position limit
            budget = min(available_cash, max_position_value)
            shares = int(budget // effective_cost_per_share)

            if shares <= 0:
                logger.debug("Cannot fill BUY for %s: insufficient capital (budget=%.2f, cost=%.2f)", order.symbol, budget, effective_cost_per_share)
                return None

            gross_value = shares * fill_price
            slippage_cost = shares * abs(fill_price - open_price)
            commission_cost = gross_value * commission_mult
            total_cost = slippage_cost + commission_cost

            return TradeFill(
                symbol=order.symbol,
                side=OrderSide.BUY,
                shares=shares,
                fill_price=fill_price,
                base_price=open_price,
                fill_date=execution_date,
                slippage_cost=slippage_cost,
                commission_cost=commission_cost,
                total_cost=total_cost,
            )

        elif order.side == OrderSide.SELL:
            # Adverse slippage decreases sell fill price
            fill_price = open_price * (1.0 - slippage_mult)
            shares = order.shares or 0
            if shares <= 0:
                return None

            gross_value = shares * fill_price
            slippage_cost = shares * abs(fill_price - open_price)
            commission_cost = gross_value * commission_mult
            total_cost = slippage_cost + commission_cost

            return TradeFill(
                symbol=order.symbol,
                side=OrderSide.SELL,
                shares=shares,
                fill_price=fill_price,
                base_price=open_price,
                fill_date=execution_date,
                slippage_cost=slippage_cost,
                commission_cost=commission_cost,
                total_cost=total_cost,
            )

        return None
