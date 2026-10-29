"""
Trading Agent Control Schemas
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class AgentStatusResponse(BaseModel):
    state: str  # STOPPED, RUNNING, PAUSED, ERROR
    mode: str   # Strictly PAPER
    last_heartbeat: Optional[datetime] = None
    last_cycle_at: Optional[datetime] = None
    error_message: Optional[str] = None
    is_paper_trading: bool = True
    broker_type: str = "mock"  # "alpaca_paper" or "mock"


class AgentCommandRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=256)


class AgentControlResponse(BaseModel):
    success: bool
    state: str
    message: str


class EmergencyStopResponse(BaseModel):
    success: bool
    state: str
    cancelled_orders_count: int
    message: str
