"""
In-Memory Mock Broker Adapter
=============================
Provides a deterministic, high-fidelity paper trading simulation
for local development, automated testing, and offline execution.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import uuid
import pandas as pd
from tradepilot.execution.broker_base import (
    BaseBrokerAdapter,
    BrokerAccount,
    BrokerPosition,
    BrokerOrderResponse,
    BrokerFill,
)
from tradepilot.api.config import PROJECT_ROOT


class MockBrokerAdapter(BaseBrokerAdapter):
    """Simulates paper trading with portfolio state, fills, slippage, and fees."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        slippage_bps: float = 5.0,
        commission_bps: float = 5.0,
        historical_data_path: Optional[Path] = None,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.positions: Dict[str, Dict[str, float]] = {}  # symbol -> {shares, avg_price}
        self.orders: Dict[str, BrokerOrderResponse] = {}   # order_id -> BrokerOrderResponse
        self.fills: List[BrokerFill] = []
        self._raw_data_cache: Optional[pd.DataFrame] = None
        self.data_path = historical_data_path or (PROJECT_ROOT / "data" / "raw" / "stock_ohlcv_2021_2025.csv")

    def is_paper(self) -> bool:
        return True

    def _load_historical_data(self) -> pd.DataFrame:
        if self._raw_data_cache is None:
            if self.data_path.exists():
                df = pd.read_csv(self.data_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                self._raw_data_cache = df
            else:
                # Fallback synthetic frame for testing environments without raw data
                records = []
                now = datetime.now(timezone.utc)
                symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"]
                for sym in symbols:
                    for i in range(50):
                        records.append({
                            "symbol": sym,
                            "timestamp": now - pd.Timedelta(days=50 - i),
                            "open": 150.0 + i * 0.2,
                            "high": 152.0 + i * 0.2,
                            "low": 149.0 + i * 0.2,
                            "close": 151.0 + i * 0.2,
                            "volume": 1000000.0,
                        })
                self._raw_data_cache = pd.DataFrame(records)
        return self._raw_data_cache

    def get_latest_bars(self, symbol: str, limit: int = 40) -> pd.DataFrame:
        df = self._load_historical_data()
        sym_df = df[df["symbol"].str.upper() == symbol.upper()].sort_values("timestamp")
        if sym_df.empty:
            # Return synthetic DataFrame if symbol not found
            now = datetime.now(timezone.utc)
            rows = []
            for i in range(limit):
                rows.append({
                    "symbol": symbol.upper(),
                    "timestamp": now - pd.Timedelta(days=limit - i),
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 500000.0,
                })
            return pd.DataFrame(rows)
        return sym_df.tail(limit).copy().reset_index(drop=True)

    def get_latest_price(self, symbol: str) -> float:
        bars = self.get_latest_bars(symbol, limit=1)
        if not bars.empty:
            return float(bars.iloc[-1]["close"])
        return 150.0

    def get_account(self) -> BrokerAccount:
        portfolio_value = self.cash
        for sym, pos in self.positions.items():
            if pos["shares"] > 0:
                current_price = self.get_latest_price(sym)
                portfolio_value += pos["shares"] * current_price

        # Standard 2x buying power for marginable securities in paper trading
        buying_power = max(0.0, self.cash)

        return BrokerAccount(
            account_number="MOCK_PAPER_ACC_01",
            status="ACTIVE",
            currency="USD",
            cash=round(self.cash, 2),
            buying_power=round(buying_power, 2),
            portfolio_value=round(portfolio_value, 2),
            is_paper=True,
            trading_blocked=False,
        )

    def get_positions(self) -> List[BrokerPosition]:
        result = []
        for sym, pos in self.positions.items():
            shares = pos["shares"]
            if shares <= 0:
                continue
            current_price = self.get_latest_price(sym)
            market_value = shares * current_price
            cost_basis = shares * pos["avg_price"]
            unrealized_pl = market_value - cost_basis
            unrealized_plpc = unrealized_pl / cost_basis if cost_basis > 0 else 0.0

            result.append(
                BrokerPosition(
                    symbol=sym,
                    qty=shares,
                    market_value=round(market_value, 2),
                    cost_basis=round(cost_basis, 2),
                    unrealized_pl=round(unrealized_pl, 2),
                    unrealized_plpc=round(unrealized_plpc, 4),
                    current_price=round(current_price, 2),
                    avg_entry_price=round(pos["avg_price"], 2),
                )
            )
        return result

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        for pos in self.get_positions():
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None

    def get_open_orders(self) -> List[BrokerOrderResponse]:
        return [
            ord for ord in self.orders.values()
            if ord.status in ("SUBMITTED", "PARTIALLY_FILLED")
        ]

    def get_order(self, order_id: str) -> Optional[BrokerOrderResponse]:
        if order_id in self.orders:
            return self.orders[order_id]
        for ord in self.orders.values():
            if ord.client_order_id == order_id:
                return ord
        return None

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> BrokerOrderResponse:
        norm_side = side.lower()
        if norm_side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side}")
        if qty <= 0:
            raise ValueError(f"Quantity must be positive, got {qty}")

        order_id = f"mock-ord-{uuid.uuid4().hex[:12]}"
        cid = client_order_id or f"cid-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        base_price = self.get_latest_price(symbol)

        # Apply slippage: buy pays slightly higher, sell receives slightly lower
        slip_mult = 1.0 + (self.slippage_bps / 10000.0) if norm_side == "buy" else 1.0 - (self.slippage_bps / 10000.0)
        exec_price = round(base_price * slip_mult, 4)
        trade_val = qty * exec_price
        fee = round(trade_val * (self.commission_bps / 10000.0), 4)

        if norm_side == "buy":
            total_cost = trade_val + fee
            if total_cost > self.cash:
                # Reject if cash is insufficient
                order_resp = BrokerOrderResponse(
                    order_id=order_id,
                    client_order_id=cid,
                    symbol=symbol.upper(),
                    side=norm_side,
                    qty=qty,
                    filled_qty=0.0,
                    status="REJECTED",
                    order_type=order_type,
                    time_in_force=time_in_force,
                    avg_fill_price=None,
                    submitted_at=now,
                    updated_at=now,
                )
                self.orders[order_id] = order_resp
                return order_resp

            # Deduct cash & update position
            self.cash -= total_cost
            curr_pos = self.positions.get(symbol.upper(), {"shares": 0.0, "avg_price": 0.0})
            new_shares = curr_pos["shares"] + qty
            new_avg = ((curr_pos["shares"] * curr_pos["avg_price"]) + (qty * exec_price)) / new_shares
            self.positions[symbol.upper()] = {"shares": new_shares, "avg_price": new_avg}

        else:  # sell
            curr_pos = self.positions.get(symbol.upper(), {"shares": 0.0, "avg_price": 0.0})
            if curr_pos["shares"] < qty:
                # Reject if trying to short beyond held shares in simple paper mode
                order_resp = BrokerOrderResponse(
                    order_id=order_id,
                    client_order_id=cid,
                    symbol=symbol.upper(),
                    side=norm_side,
                    qty=qty,
                    filled_qty=0.0,
                    status="REJECTED",
                    order_type=order_type,
                    time_in_force=time_in_force,
                    avg_fill_price=None,
                    submitted_at=now,
                    updated_at=now,
                )
                self.orders[order_id] = order_resp
                return order_resp

            # Credit cash minus fees & update position
            net_proceeds = trade_val - fee
            self.cash += net_proceeds
            rem_shares = curr_pos["shares"] - qty
            if rem_shares <= 1e-6:
                self.positions.pop(symbol.upper(), None)
            else:
                self.positions[symbol.upper()]["shares"] = rem_shares

        # Record fill
        fill = BrokerFill(
            fill_id=f"fill-{uuid.uuid4().hex[:12]}",
            order_id=order_id,
            symbol=symbol.upper(),
            side=norm_side,
            qty=qty,
            price=exec_price,
            fee=fee,
            timestamp=now,
        )
        self.fills.append(fill)

        order_resp = BrokerOrderResponse(
            order_id=order_id,
            client_order_id=cid,
            symbol=symbol.upper(),
            side=norm_side,
            qty=qty,
            filled_qty=qty,
            status="FILLED",
            order_type=order_type,
            time_in_force=time_in_force,
            avg_fill_price=exec_price,
            submitted_at=now,
            updated_at=now,
        )
        self.orders[order_id] = order_resp
        return order_resp

    def cancel_order(self, order_id: str) -> bool:
        ord = self.orders.get(order_id)
        if ord and ord.status in ("SUBMITTED", "PARTIALLY_FILLED"):
            ord.status = "CANCELED"
            ord.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def cancel_all_orders(self) -> int:
        cancelled = 0
        for ord in self.orders.values():
            if ord.status in ("SUBMITTED", "PARTIALLY_FILLED"):
                ord.status = "CANCELED"
                ord.updated_at = datetime.now(timezone.utc)
                cancelled += 1
        return cancelled
