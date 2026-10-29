"""
SQLAlchemy ORM Models for TradePilot
====================================
Defines schemas for Users, Agent State, Trading Cycles, Predictions,
Signals, Orders, Fills, Positions, Portfolio Snapshots, Risk Events, and Audit Logs.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Text,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column
from tradepilot.db.base import Base, TimestampMixin, utc_now


class User(Base, TimestampMixin):
    """Application user for authentication and role-based access control."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="viewer", nullable=False)  # admin, trader, viewer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AgentStateModel(Base):
    """Singleton tracking current trading agent lifecycle and heartbeat."""

    __tablename__ = "agent_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    state: Mapped[str] = mapped_column(String(32), default="STOPPED", nullable=False)  # STOPPED, RUNNING, PAUSED, ERROR
    mode: Mapped[str] = mapped_column(String(32), default="PAPER", nullable=False)      # Strictly PAPER
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_cycle_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class TradingCycleModel(Base):
    """Record of an automated or manual trading cycle execution."""

    __tablename__ = "trading_cycles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # UUID or timestamp-based ID
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING", nullable=False)  # RUNNING, COMPLETED, FAILED
    symbols_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    orders_generated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PredictionRecord(Base):
    """Model predictions generated per symbol during a cycle."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    prob_up: Mapped[float] = mapped_column(Float, nullable=False)
    prob_neutral: Mapped[float] = mapped_column(Float, nullable=False)
    prob_down: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(16), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)


class SignalRecord(Base):
    """Trading strategy signals evaluated from predictions."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    signal: Mapped[str] = mapped_column(String(16), nullable=False)  # BUY, EXIT, HOLD
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)


class OrderIntentRecord(Base):
    """Internal order intent created before broker submission and validated by risk."""

    __tablename__ = "order_intents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    cycle_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)  # BUY, SELL
    target_qty: Mapped[float] = mapped_column(Float, nullable=False)
    proposed_price: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)  # PENDING, APPROVED, REJECTED
    risk_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class BrokerOrderRecord(Base):
    """Actual order sent to paper brokerage and its lifecycle states."""

    __tablename__ = "broker_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(128), unique=True, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)  # buy, sell
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    filled_qty: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="SUBMITTED", nullable=False)  # SUBMITTED, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED
    order_type: Mapped[str] = mapped_column(String(32), default="market", nullable=False)
    time_in_force: Mapped[str] = mapped_column(String(16), default="day", nullable=False)
    avg_fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class OrderFillRecord(Base):
    """Individual trade execution fill reported by the broker."""

    __tablename__ = "order_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    broker_order_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class PositionSnapshotRecord(Base):
    """Periodic snapshot of open positions reconciled against broker state."""

    __tablename__ = "position_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    avg_entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    market_value: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False)


class PortfolioSnapshotRecord(Base):
    """Historical timeline of account equity, cash, and exposure."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    buying_power: Mapped[float] = mapped_column(Float, nullable=False)
    gross_exposure: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    exposure_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    day_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)


class RiskEventRecord(Base):
    """Log of risk limit violations, circuit-breaker trips, and blocked orders."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    rule_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), default="WARNING", nullable=False)  # INFO, WARNING, CRITICAL
    description: Mapped[str] = mapped_column(Text, nullable=False)
    blocked_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class AuditLogRecord(Base):
    """Append-only audit trail for administrative actions and state changes."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
