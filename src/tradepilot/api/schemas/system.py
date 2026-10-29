"""
System and Health Schemas
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: datetime


class ReadyResponse(BaseModel):
    ready: bool
    database: bool
    model_loaded: bool
    broker_connected: bool
    trading_mode: str
    paper_safeguard_active: bool
    details: Optional[Dict[str, Any]] = None


class SystemStatusResponse(BaseModel):
    app_name: str
    version: str
    environment: str
    trading_mode: str
    database_status: str
    broker_status: str
    model_version: str
    agent_state: str
    uptime_seconds: float
    timestamp: datetime
