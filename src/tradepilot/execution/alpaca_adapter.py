"""
Alpaca Paper Trading Broker Adapter
===================================
Production-grade integration with Alpaca's paper-trading brokerage and data APIs.
Strictly enforces paper-trading endpoints and safeguards.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from tradepilot.execution.broker_base import (
    BaseBrokerAdapter,
    BrokerAccount,
    BrokerPosition,
    BrokerOrderResponse,
)
from tradepilot.api.config import settings


class AlpacaPaperBroker(BaseBrokerAdapter):
    """Integrates with Alpaca Paper API using official alpaca-py SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        key = api_key or settings.ALPACA_API_KEY
        secret = secret_key or settings.ALPACA_SECRET_KEY

        if not key or not secret:
            raise ValueError("Alpaca API credentials are missing.")

        # Safeguard: Refuse to initialize if not explicitly paper mode
        if not paper or not settings.ALPACA_PAPER or settings.TRADING_MODE != "PAPER":
            raise RuntimeError(
                "CRITICAL SAFEGUARD: TradePilot is strictly restricted to PAPER trading in this phase. "
                "Live trading is prohibited."
            )

        self.trading_client = TradingClient(
            api_key=key,
            secret_key=secret,
            paper=True,
            url_override=settings.ALPACA_BASE_URL,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=key,
            secret_key=secret,
            url_override=settings.ALPACA_DATA_URL,
        )

    def is_paper(self) -> bool:
        return True

    def get_account(self) -> BrokerAccount:
        acc = self.trading_client.get_account()
        return BrokerAccount(
            account_number=str(acc.account_number),
            status=str(acc.status),
            currency=str(acc.currency),
            cash=float(acc.cash),
            buying_power=float(acc.buying_power),
            portfolio_value=float(acc.portfolio_value),
            is_paper=True,
            trading_blocked=bool(acc.trading_blocked),
        )

    def get_positions(self) -> List[BrokerPosition]:
        positions = self.trading_client.get_all_positions()
        res = []
        for p in positions:
            res.append(
                BrokerPosition(
                    symbol=str(p.symbol),
                    qty=float(p.qty),
                    market_value=float(p.market_value),
                    cost_basis=float(p.cost_basis),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                    current_price=float(p.current_price),
                    avg_entry_price=float(p.avg_entry_price),
                )
            )
        return res

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        try:
            p = self.trading_client.get_open_position(symbol.upper())
            return BrokerPosition(
                symbol=str(p.symbol),
                qty=float(p.qty),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                current_price=float(p.current_price),
                avg_entry_price=float(p.avg_entry_price),
            )
        except Exception:
            return None

    def get_open_orders(self) -> List[BrokerOrderResponse]:
        orders = self.trading_client.get_orders()
        res = []
        for o in orders:
            res.append(
                BrokerOrderResponse(
                    order_id=str(o.id),
                    client_order_id=str(o.client_order_id),
                    symbol=str(o.symbol),
                    side=str(o.side).lower(),
                    qty=float(o.qty),
                    filled_qty=float(o.filled_qty or 0.0),
                    status=str(o.status).upper(),
                    order_type=str(o.order_type).lower(),
                    time_in_force=str(o.time_in_force).lower(),
                    avg_fill_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                    submitted_at=o.submitted_at,
                    updated_at=o.updated_at,
                )
            )
        return res

    def get_order(self, order_id: str) -> Optional[BrokerOrderResponse]:
        try:
            o = self.trading_client.get_order_by_id(order_id)
            return BrokerOrderResponse(
                order_id=str(o.id),
                client_order_id=str(o.client_order_id),
                symbol=str(o.symbol),
                side=str(o.side).lower(),
                qty=float(o.qty),
                filled_qty=float(o.filled_qty or 0.0),
                status=str(o.status).upper(),
                order_type=str(o.order_type).lower(),
                time_in_force=str(o.time_in_force).lower(),
                avg_fill_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                submitted_at=o.submitted_at,
                updated_at=o.updated_at,
            )
        except Exception:
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
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        req = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=order_side,
            time_in_force=tif,
            client_order_id=client_order_id,
        )

        o = self.trading_client.submit_order(order_data=req)
        return BrokerOrderResponse(
            order_id=str(o.id),
            client_order_id=str(o.client_order_id),
            symbol=str(o.symbol),
            side=str(o.side).lower(),
            qty=float(o.qty),
            filled_qty=float(o.filled_qty or 0.0),
            status=str(o.status).upper(),
            order_type=str(o.order_type).lower(),
            time_in_force=str(o.time_in_force).lower(),
            avg_fill_price=float(o.filled_avg_price) if o.filled_avg_price else None,
            submitted_at=o.submitted_at,
            updated_at=o.updated_at,
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def cancel_all_orders(self) -> int:
        res = self.trading_client.cancel_orders()
        return len(res)

    def get_latest_price(self, symbol: str) -> float:
        bars = self.get_latest_bars(symbol, limit=1)
        if not bars.empty:
            return float(bars.iloc[-1]["close"])
        raise ValueError(f"No price data available for {symbol}")

    def get_latest_bars(self, symbol: str, limit: int = 40) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed

        end = datetime.now(timezone.utc)

        start = end - timedelta(days=limit * 2 + 10)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[symbol.upper()],
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            barset = self.data_client.get_stock_bars(req)
            bars = barset[symbol.upper()]
            rows = []
            for b in bars:
                rows.append({
                    "symbol": symbol.upper(),
                    "timestamp": b.timestamp,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                })
            df = pd.DataFrame(rows)
            if not df.empty:
                return df.tail(limit).reset_index(drop=True)
        except Exception as e:
            # Fall back to local historical dataset if Alpaca market data feed is restricted
            from tradepilot.api.config import PROJECT_ROOT
            csv_path = PROJECT_ROOT / "data" / "raw" / "stock_ohlcv_2021_2025.csv"
            if csv_path.exists():
                raw_df = pd.read_csv(csv_path)
                sym_df = raw_df[raw_df["symbol"].str.upper() == symbol.upper()].sort_values("timestamp")
                if not sym_df.empty:
                    sym_df["timestamp"] = pd.to_datetime(sym_df["timestamp"])
                    return sym_df.tail(limit).copy().reset_index(drop=True)
            raise e

