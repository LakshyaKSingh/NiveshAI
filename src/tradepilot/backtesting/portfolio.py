"""
Portfolio Management Module
===========================
Maintains cash balances, open positions, daily marked-to-market equity curves,
and closed-trade records with strict risk limits.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional

import pandas as pd

from tradepilot.backtesting.execution import OrderSide, TradeFill
from tradepilot.config import DEFAULT_BACKTEST_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open active position."""
    symbol: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    current_price: float
    entry_costs: float
    bars_held: int = 0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.shares


@dataclass(frozen=True)
class CompletedTrade:
    """Record of a closed round-trip trade."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    gross_pnl: float
    net_pnl: float
    return_pct: float
    holding_days: int
    total_costs: float


class Portfolio:
    """
    Manages portfolio cash, position holdings, and equity tracking.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_BACKTEST_CONFIG.initial_capital,
        max_positions: int = DEFAULT_BACKTEST_CONFIG.max_positions,
        allocation_per_position: float = DEFAULT_BACKTEST_CONFIG.allocation_per_position,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.allocation_per_position = allocation_per_position

        self.positions: Dict[str, Position] = {}
        self.trades: List[CompletedTrade] = []
        self.equity_history: List[Dict[str, float]] = []

    @property
    def holdings_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.cash + self.holdings_value

    def get_max_allocation_for_new_position(self) -> float:
        """Calculate max capital allowed for an individual position based on current equity."""
        return self.total_equity * self.allocation_per_position

    def can_open_position(self, symbol: str) -> bool:
        """Check whether risk constraints permit opening a new position for symbol."""
        if symbol in self.positions:
            return False  # Do not pyramid existing positions
        if len(self.positions) >= self.max_positions:
            return False
        return self.cash > 100.0  # Require minimum working cash

    def apply_fill(self, fill: TradeFill) -> None:
        """Process an executed trade fill and update portfolio state."""
        if fill.side == OrderSide.BUY:
            total_cash_outflow = (fill.shares * fill.fill_price) + fill.commission_cost
            if total_cash_outflow > self.cash:
                raise ValueError(
                    f"Cash insolvency error: Attempting to spend {total_cash_outflow:.2f} with cash balance {self.cash:.2f}"
                )

            self.cash -= total_cash_outflow
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                shares=fill.shares,
                entry_date=fill.fill_date,
                entry_price=fill.fill_price,
                current_price=fill.fill_price,
                entry_costs=fill.total_cost,
                bars_held=0,
            )
            logger.debug("Opened position %s: %d shares @ %.2f", fill.symbol, fill.shares, fill.fill_price)

        elif fill.side == OrderSide.SELL:
            if fill.symbol not in self.positions:
                logger.warning("Attempted to sell unheld symbol: %s", fill.symbol)
                return

            pos = self.positions.pop(fill.symbol)
            cash_inflow = (fill.shares * fill.fill_price) - fill.commission_cost
            self.cash += cash_inflow

            gross_pnl = (fill.fill_price - pos.entry_price) * fill.shares
            total_costs = pos.entry_costs + fill.total_cost
            net_pnl = gross_pnl - total_costs
            cost_basis = pos.shares * pos.entry_price
            return_pct = net_pnl / cost_basis if cost_basis > 0 else 0.0

            trade = CompletedTrade(
                symbol=fill.symbol,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=fill.fill_date,
                exit_price=fill.fill_price,
                shares=fill.shares,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                return_pct=return_pct,
                holding_days=pos.bars_held,
                total_costs=total_costs,
            )
            self.trades.append(trade)
            logger.debug("Closed position %s: net PnL=%.2f (%.2f%%)", fill.symbol, net_pnl, return_pct * 100)

    def mark_to_market(self, current_date: pd.Timestamp, close_prices: Dict[str, float]) -> None:
        """
        Update positions with daily closing prices, increment holding days,
        and record daily portfolio valuation snapshot.
        """
        for sym, pos in list(self.positions.items()):
            if sym in close_prices and not pd.isna(close_prices[sym]) and close_prices[sym] > 0:
                pos.current_price = close_prices[sym]
            pos.bars_held += 1

        total_eq = self.total_equity
        gross_exposure = (self.holdings_value / total_eq) if total_eq > 0 else 0.0

        self.equity_history.append({
            "date": current_date,
            "cash": self.cash,
            "holdings_value": self.holdings_value,
            "total_equity": total_eq,
            "gross_exposure": gross_exposure,
            "positions_count": len(self.positions),
        })

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Return marked-to-market daily equity history as a DataFrame."""
        if not self.equity_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_history)
        df["daily_return"] = df["total_equity"].pct_change().fillna(0.0)
        df["cumulative_return"] = (df["total_equity"] / self.initial_capital) - 1.0
        return df

    def get_trade_ledger_df(self) -> pd.DataFrame:
        """Return completed round-trip trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=[
                "symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                "shares", "gross_pnl", "net_pnl", "return_pct", "holding_days", "total_costs"
            ])
        records = [
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "return_pct": t.return_pct,
                "holding_days": t.holding_days,
                "total_costs": t.total_costs,
            }
            for t in self.trades
        ]
        return pd.DataFrame(records)
