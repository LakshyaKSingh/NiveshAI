"""
Agent Lifecycle and Control Endpoints
====================================
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from tradepilot.api.schemas.agent import (
    AgentStatusResponse,
    AgentCommandRequest,
    AgentControlResponse,
    EmergencyStopResponse,
)
from tradepilot.api.dependencies import get_db, require_role
from tradepilot.db.models import User
from tradepilot.execution.agent import TradingAgent
from tradepilot.api.config import settings

router = APIRouter(prefix="/api/v1/agent", tags=["Agent Control"])

_agent_instance = None


def get_agent() -> TradingAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TradingAgent()
    return _agent_instance


@router.get("/status", response_model=AgentStatusResponse)
def get_agent_status(db: Session = Depends(get_db)):
    """Retrieve current agent state (STOPPED, RUNNING, PAUSED), mode, and heartbeat."""
    agent = get_agent()
    state = agent.get_state(db)
    broker_type = "alpaca_paper" if type(agent.broker).__name__ == "AlpacaPaperBroker" else "mock"

    return AgentStatusResponse(
        state=state.state,
        mode=state.mode,
        last_heartbeat=state.last_heartbeat,
        last_cycle_at=state.last_cycle_at,
        error_message=state.error_message,
        is_paper_trading=agent.broker.is_paper(),
        broker_type=broker_type,
    )


@router.post("/start", response_model=AgentControlResponse)
def start_trading_agent(
    req: AgentCommandRequest = None,
    current_user: User = Depends(require_role(["trader", "admin"])),
    db: Session = Depends(get_db),
):
    """Start automated trading strategy execution in PAPER trading mode."""
    agent = get_agent()
    try:
        state = agent.start(db, user_id=current_user.id)
        return AgentControlResponse(
            success=True,
            state=state.state,
            message=f"Agent started successfully in {state.mode} mode by {current_user.username}.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pause", response_model=AgentControlResponse)
def pause_trading_agent(
    req: AgentCommandRequest = None,
    current_user: User = Depends(require_role(["trader", "admin"])),
    db: Session = Depends(get_db),
):
    """Pause automated trading strategy (prevents new orders while maintaining positions)."""
    agent = get_agent()
    state = agent.pause(db, user_id=current_user.id)
    return AgentControlResponse(
        success=True,
        state=state.state,
        message=f"Agent paused by {current_user.username}.",
    )


@router.post("/stop", response_model=AgentControlResponse)
def stop_trading_agent(
    req: AgentCommandRequest = None,
    current_user: User = Depends(require_role(["trader", "admin"])),
    db: Session = Depends(get_db),
):
    """Stop automated strategy execution."""
    agent = get_agent()
    state = agent.stop(db, user_id=current_user.id)
    return AgentControlResponse(
        success=True,
        state=state.state,
        message=f"Agent stopped by {current_user.username}.",
    )


@router.post("/emergency-stop", response_model=EmergencyStopResponse)
def emergency_stop(
    req: AgentCommandRequest = None,
    current_user: User = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """
    CRITICAL: Global emergency kill-switch.
    Trips circuit breaker, blocks all subsequent orders, and cancels all pending broker orders.
    """
    agent = get_agent()
    cancelled = agent.emergency_stop(db, user_id=current_user.id)
    return EmergencyStopResponse(
        success=True,
        state="STOPPED",
        cancelled_orders_count=cancelled,
        message=f"CRITICAL: Emergency stop executed by {current_user.username}. Cancelled {cancelled} open orders.",
    )
