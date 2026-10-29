"""
Risk Management and Audit Schemas
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class RiskConfigResponse(BaseModel):
    max_position_size_pct: float
    max_concurrent_positions: int
    max_total_exposure_pct: float
    max_daily_loss_pct: float
    slippage_bps: float
    commission_bps: float
    entry_threshold: float
    max_consecutive_failures: int


class RiskStatusResponse(BaseModel):
    healthy: bool
    circuit_breaker_tripped: bool
    trip_reason: Optional[str] = None
    open_positions_count: int
    max_positions: int
    gross_exposure_pct: float
    max_exposure_pct: float
    current_daily_drawdown_pct: float
    max_daily_drawdown_pct: float
    last_evaluation: datetime


class RiskEventItem(BaseModel):
    id: int
    timestamp: datetime
    rule_name: str
    severity: str
    description: str
    blocked_order_id: Optional[str] = None


class RiskEventsResponse(BaseModel):
    total: int
    events: List[RiskEventItem]


class AuditLogItem(BaseModel):
    id: int
    timestamp: datetime
    user_id: Optional[int] = None
    username: Optional[str] = None
    action: str
    entity_type: str
    entity_id: Optional[str] = None
    details_json: Optional[str] = None
    ip_address: Optional[str] = None


class AuditLogsResponse(BaseModel):
    total: int
    logs: List[AuditLogItem]
