"""
Risk Management and Audit Trail Endpoints
=========================================
"""

from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tradepilot.api.schemas.risk import (
    RiskConfigResponse,
    RiskStatusResponse,
    RiskEventItem,
    RiskEventsResponse,
    AuditLogItem,
    AuditLogsResponse,
)
from tradepilot.api.dependencies import get_db, require_role
from tradepilot.db.models import RiskEventRecord, AuditLogRecord, User
from tradepilot.execution.agent import TradingAgent
from tradepilot.api.config import settings

router = APIRouter(prefix="/api/v1/risk", tags=["Risk & Activity"])

_agent_instance = None


def get_agent() -> TradingAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TradingAgent()
    return _agent_instance


@router.get("/config", response_model=RiskConfigResponse)
def get_risk_config():
    """Fetch active quantitative risk rules and thresholds."""
    agent = get_agent()
    r = agent.risk_engine
    return RiskConfigResponse(
        max_position_size_pct=r.max_position_size_pct,
        max_concurrent_positions=r.max_concurrent_positions,
        max_total_exposure_pct=r.max_total_exposure_pct,
        max_daily_loss_pct=r.max_daily_loss_pct,
        slippage_bps=settings.DEFAULT_SLIPPAGE_BPS,
        commission_bps=settings.DEFAULT_COMMISSION_BPS,
        entry_threshold=settings.ENTRY_THRESHOLD,
        max_consecutive_failures=r.max_consecutive_failures,
    )


@router.get("/status", response_model=RiskStatusResponse)
def get_risk_status():
    """Evaluate real-time risk utilization, circuit breakers, and exposure."""
    agent = get_agent()
    r = agent.risk_engine
    acc = agent.broker.get_account()
    positions = agent.broker.get_positions()

    gross_exposure = sum(p.market_value for p in positions)
    equity = max(1.0, acc.portfolio_value)
    exposure_pct = round((gross_exposure / equity) * 100.0, 2)

    total_unrealized = sum(p.unrealized_pl for p in positions)
    day_drawdown_pct = round(max(0.0, -total_unrealized / equity) * 100.0, 2)

    return RiskStatusResponse(
        healthy=not r.circuit_breaker_tripped,
        circuit_breaker_tripped=r.circuit_breaker_tripped,
        trip_reason=r.circuit_breaker_reason,
        open_positions_count=len(positions),
        max_positions=r.max_concurrent_positions,
        gross_exposure_pct=exposure_pct,
        max_exposure_pct=round(r.max_total_exposure_pct * 100.0, 2),
        current_daily_drawdown_pct=day_drawdown_pct,
        max_daily_drawdown_pct=round(r.max_daily_loss_pct * 100.0, 2),
        last_evaluation=r.last_evaluation_time,
    )


@router.get("/events", response_model=RiskEventsResponse)
def get_risk_events(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Retrieve history of risk limit violations and blocked order attempts."""
    records = (
        db.query(RiskEventRecord)
        .order_by(desc(RiskEventRecord.timestamp))
        .limit(limit)
        .all()
    )
    items = [
        RiskEventItem(
            id=r.id,
            timestamp=r.timestamp,
            rule_name=r.rule_name,
            severity=r.severity,
            description=r.description,
            blocked_order_id=r.blocked_order_id,
        )
        for r in records
    ]
    return RiskEventsResponse(total=len(items), events=items)


@router.get("/audit/logs", response_model=AuditLogsResponse)
def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """Retrieve append-only audit log entries for operational actions (Admin only)."""
    records = (
        db.query(AuditLogRecord)
        .order_by(desc(AuditLogRecord.timestamp))
        .limit(limit)
        .all()
    )
    items = [
        AuditLogItem(
            id=r.id,
            timestamp=r.timestamp,
            user_id=r.user_id,
            username=r.username,
            action=r.action,
            entity_type=r.entity_type,
            entity_id=r.entity_id,
            details_json=r.details_json,
            ip_address=r.ip_address,
        )
        for r in records
    ]
    return AuditLogsResponse(total=len(items), logs=items)
