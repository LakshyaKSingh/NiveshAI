"""
Broker Interface Abstraction
============================
Defines the standard contract for brokerage adapters (Paper Alpaca and Mock).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import pandas as pd


@dataclass
class BrokerAccount:
    account_number: str
    status: str
    currency: str
    cash: float
    buying_power: float
    portfolio_value: float
    is_paper: bool
    trading_blocked: bool = False


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    avg_entry_price: float


@dataclass
class BrokerOrderResponse:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    status: str  # SUBMITTED, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED
    order_type: str = "market"
    time_in_force: str = "day"
    avg_fill_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class BrokerFill:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    timestamp: datetime


class BaseBrokerAdapter(ABC):
    """Abstract base class for all broker execution adapters."""

    @abstractmethod
    def is_paper(self) -> bool:
        """Return True if connected to paper trading environment."""
        pass

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        """Fetch current account balances, equity, and buying power."""
        pass

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Fetch all currently open positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Fetch open position for a specific symbol, or None if not held."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[BrokerOrderResponse]:
        """Fetch all pending/open orders."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[BrokerOrderResponse]:
        """Fetch order details by broker order ID or client order ID."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> BrokerOrderResponse:
        """Submit a new paper order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders and return count of cancelled orders."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get latest market price for a symbol."""
        pass

    @abstractmethod
    def get_latest_bars(self, symbol: str, limit: int = 40) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV bars for feature computation.
        Returns DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'].
        """
        pass
