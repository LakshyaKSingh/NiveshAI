"""
Portfolio and Positions Schemas
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class PositionItem(BaseModel):
    symbol: str
    shares: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    exposure_pct: float
    updated_at: datetime


class PortfolioSummaryResponse(BaseModel):
    equity: float
    cash: float
    buying_power: float
    gross_exposure: float
    exposure_pct: float
    day_pnl: float
    day_pnl_pct: float
    total_pnl: float
    open_positions_count: int
    trading_mode: str
    last_updated: datetime


class EquityPoint(BaseModel):
    timestamp: datetime
    equity: float
    cash: float


class EquityHistoryResponse(BaseModel):
    points: List[EquityPoint]
