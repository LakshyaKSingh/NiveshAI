"""
Orders and Trades Schemas
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class OrderCreateRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    side: str = Field(..., pattern="^(buy|sell|BUY|SELL)$")
    qty: float = Field(..., gt=0)
    order_type: str = Field(default="market", pattern="^(market|limit)$")
    limit_price: Optional[float] = Field(default=None, gt=0)
    time_in_force: str = Field(default="day", pattern="^(day|gtc)$")


class OrderItem(BaseModel):
    id: int
    client_order_id: str
    broker_order_id: Optional[str] = None
    symbol: str
    side: str
    qty: float
    filled_qty: float
    status: str
    order_type: str
    time_in_force: str
    avg_fill_price: Optional[float] = None
    submitted_at: datetime
    updated_at: datetime


class OrdersListResponse(BaseModel):
    total: int
    orders: List[OrderItem]


class FillItem(BaseModel):
    id: int
    broker_order_id: str
    symbol: str
    side: str
    shares: float
    price: float
    fee: float
    timestamp: datetime


class FillsListResponse(BaseModel):
    total: int
    fills: List[FillItem]
