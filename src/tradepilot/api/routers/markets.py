"""
Market Data Endpoints
=====================
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from tradepilot.api.schemas.market import (
    BarItem,
    HistoricalBarsResponse,
    LatestPriceItem,
    LatestPricesResponse,
    MarketProviderStatus,
)
from tradepilot.execution.mock_adapter import MockBrokerAdapter
from tradepilot.execution.alpaca_adapter import AlpacaPaperBroker
from tradepilot.api.config import settings

router = APIRouter(prefix="/api/v1/markets", tags=["Markets"])

# Default universe of symbols
APPROVED_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"]

# Instantiate cached adapter for market data queries
_broker = None


def get_market_broker():
    global _broker
    if _broker is None:
        if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY and settings.ALPACA_PAPER:
            try:
                _broker = AlpacaPaperBroker()
            except Exception:
                _broker = MockBrokerAdapter()
        else:
            _broker = MockBrokerAdapter()
    return _broker


@router.get("/symbols", response_model=List[str])
def get_supported_symbols():
    """List approved trading symbols in the TradePilot universe."""
    return APPROVED_SYMBOLS


@router.get("/bars", response_model=HistoricalBarsResponse)
def get_historical_bars(
    symbol: str = Query(..., description="Stock symbol (e.g. AAPL)"),
    limit: int = Query(default=40, ge=5, le=200, description="Number of daily bars"),
):
    """Fetch historical daily OHLCV bars for charts and feature verification."""
    broker = get_market_broker()
    sym = symbol.upper()
    try:
        df = broker.get_latest_bars(sym, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch bars for {sym}: {e}")

    bars = []
    for _, row in df.iterrows():
        bars.append(
            BarItem(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )

    return HistoricalBarsResponse(
        symbol=sym,
        timeframe="1D",
        count=len(bars),
        bars=bars,
    )


@router.get("/latest", response_model=LatestPricesResponse)
def get_latest_prices():
    """Get latest available closing/market prices for all universe symbols."""
    broker = get_market_broker()
    now = datetime.now(timezone.utc)
    prices = []

    for sym in APPROVED_SYMBOLS:
        try:
            df = broker.get_latest_bars(sym, limit=2)
            if not df.empty:
                last_bar = df.iloc[-1]
                prev_bar = df.iloc[-2] if len(df) > 1 else last_bar
                curr_p = float(last_bar["close"])
                prev_p = float(prev_bar["close"])
                chg_pct = round(((curr_p - prev_p) / prev_p) * 100.0, 2) if prev_p > 0 else 0.0

                prices.append(
                    LatestPriceItem(
                        symbol=sym,
                        price=curr_p,
                        change_pct=chg_pct,
                        volume=float(last_bar["volume"]),
                        timestamp=last_bar["timestamp"],
                    )
                )
        except Exception:
            continue

    return LatestPricesResponse(
        data_source="Alpaca Paper / Historical Engine",
        is_paper=True,
        timestamp=now,
        prices=prices,
    )


@router.get("/status", response_model=MarketProviderStatus)
def get_market_status():
    """Expose market data provider status, connectivity, and data staleness."""
    broker = get_market_broker()
    now = datetime.now(timezone.utc)
    staleness = 0.0

    try:
        aapl_bars = broker.get_latest_bars("AAPL", limit=1)
        if not aapl_bars.empty:
            last_ts = aapl_bars.iloc[-1]["timestamp"]
            if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            staleness = max(0.0, (now - last_ts).total_seconds())
    except Exception:
        staleness = 9999.0

    return MarketProviderStatus(
        provider="Alpaca Markets (Paper) / Local Engine",
        connected=True,
        paper_trading=True,
        staleness_seconds=round(staleness, 1),
        supported_symbols=APPROVED_SYMBOLS,
    )
