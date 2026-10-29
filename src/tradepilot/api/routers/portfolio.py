"""
Portfolio and Positions Endpoints
=================================
"""

from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tradepilot.api.schemas.portfolio import (
    PositionItem,
    PortfolioSummaryResponse,
    EquityPoint,
    EquityHistoryResponse,
)
from tradepilot.api.dependencies import get_db
from tradepilot.db.models import PortfolioSnapshotRecord
from tradepilot.execution.agent import TradingAgent
from tradepilot.api.config import settings

router = APIRouter(prefix="/api/v1/portfolio", tags=["Portfolio"])

_agent_instance = None


def get_agent() -> TradingAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TradingAgent()
    return _agent_instance


@router.get("", response_model=PortfolioSummaryResponse)
def get_portfolio_summary(db: Session = Depends(get_db)):
    """Fetch current account equity, cash balance, buying power, and exposure."""
    agent = get_agent()
    acc = agent.broker.get_account()
    positions = agent.broker.get_positions()

    gross_exposure = sum(p.market_value for p in positions)
    equity = max(1.0, acc.portfolio_value)
    exposure_pct = round((gross_exposure / equity) * 100.0, 2)

    total_unrealized = sum(p.unrealized_pl for p in positions)
    day_pnl = total_unrealized
    day_pnl_pct = round((day_pnl / equity) * 100.0, 2)
    total_pnl = round(acc.portfolio_value - 100000.0, 2)

    return PortfolioSummaryResponse(
        equity=round(acc.portfolio_value, 2),
        cash=round(acc.cash, 2),
        buying_power=round(acc.buying_power, 2),
        gross_exposure=round(gross_exposure, 2),
        exposure_pct=exposure_pct,
        day_pnl=round(day_pnl, 2),
        day_pnl_pct=day_pnl_pct,
        total_pnl=total_pnl,
        open_positions_count=len(positions),
        trading_mode=settings.TRADING_MODE,
        last_updated=datetime.now(timezone.utc),
    )


@router.get("/positions", response_model=List[PositionItem])
def get_open_positions():
    """Retrieve all currently held open positions with unrealized PnL."""
    agent = get_agent()
    acc = agent.broker.get_account()
    positions = agent.broker.get_positions()
    now = datetime.now(timezone.utc)
    equity = max(1.0, acc.portfolio_value)

    res = []
    for p in positions:
        exp_pct = round((p.market_value / equity) * 100.0, 2)
        res.append(
            PositionItem(
                symbol=p.symbol,
                shares=p.qty,
                avg_entry_price=p.avg_entry_price,
                current_price=p.current_price,
                market_value=p.market_value,
                unrealized_pnl=p.unrealized_pl,
                unrealized_pnl_pct=round(p.unrealized_plpc * 100.0, 2),
                exposure_pct=exp_pct,
                updated_at=now,
            )
        )
    return res


@router.get("/equity", response_model=EquityHistoryResponse)
def get_equity_history(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Retrieve historical equity timeline points for portfolio performance charts."""
    snaps = (
        db.query(PortfolioSnapshotRecord)
        .order_by(PortfolioSnapshotRecord.timestamp.asc())
        .limit(limit)
        .all()
    )

    if not snaps:
        # Provide current baseline point if history is empty
        now = datetime.now(timezone.utc)
        return EquityHistoryResponse(
            points=[EquityPoint(timestamp=now, equity=100000.0, cash=100000.0)]
        )

    points = [
        EquityPoint(timestamp=s.timestamp, equity=s.equity, cash=s.cash)
        for s in snaps
    ]
    return EquityHistoryResponse(points=points)
