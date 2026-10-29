"""
Market Data Schemas
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class BarItem(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoricalBarsResponse(BaseModel):
    symbol: str
    timeframe: str
    count: int
    bars: List[BarItem]


class LatestPriceItem(BaseModel):
    symbol: str
    price: float
    change_pct: float
    volume: float
    timestamp: datetime


class LatestPricesResponse(BaseModel):
    data_source: str
    is_paper: bool
    timestamp: datetime
    prices: List[LatestPriceItem]


class MarketProviderStatus(BaseModel):
    provider: str
    connected: bool
    paper_trading: bool
    staleness_seconds: float
    supported_symbols: List[str]
