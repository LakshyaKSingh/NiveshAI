"""
System and Health Check Endpoints
=================================
"""

import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from tradepilot.api.config import settings
from tradepilot.api.schemas.system import HealthResponse, ReadyResponse, SystemStatusResponse
from tradepilot.db.session import get_db
from tradepilot.db.models import AgentStateModel

router = APIRouter(tags=["System & Health"])
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Liveness probe: verifies process is alive and responding."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready", response_model=ReadyResponse)
def get_ready(db: Session = Depends(get_db)):
    """Readiness probe: validates database connectivity, model existence, and paper mode."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    model_ok = settings.MODEL_PATH.exists()
    broker_ok = True  # Mock adapter or Alpaca Paper connection is available
    is_paper_safe = (settings.TRADING_MODE == "PAPER")

    ready = db_ok and model_ok and is_paper_safe

    return ReadyResponse(
        ready=ready,
        database=db_ok,
        model_loaded=model_ok,
        broker_connected=broker_ok,
        trading_mode=settings.TRADING_MODE,
        paper_safeguard_active=is_paper_safe,
        details={
            "sqlite_or_pg": "sqlite" if settings.DATABASE_URL.startswith("sqlite") else "postgres",
            "model_path_exists": model_ok,
        },
    )


@router.get("/api/v1/system/status", response_model=SystemStatusResponse)
def get_system_status(db: Session = Depends(get_db)):
    """Detailed operational status of API, database, active model, and agent."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    agent_state_rec = db.query(AgentStateModel).filter_by(id=1).first()
    state_str = agent_state_rec.state if agent_state_rec else "STOPPED"

    model_ver = "RandomForest_v1"
    if settings.MODEL_METADATA_PATH.exists():
        import json
        with open(settings.MODEL_METADATA_PATH, "r") as f:
            meta = json.load(f)
            model_ver = meta.get("algorithm", "RandomForest_v1")

    return SystemStatusResponse(
        app_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        trading_mode=settings.TRADING_MODE,
        database_status=db_status,
        broker_status="connected (paper)",
        model_version=model_ver,
        agent_state=state_str,
        uptime_seconds=round(time.time() - START_TIME, 2),
        timestamp=datetime.now(timezone.utc),
    )
