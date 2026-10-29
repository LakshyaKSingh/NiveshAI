"""
Orders and Trades Execution Endpoints
=====================================
"""

import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tradepilot.api.schemas.order import (
    OrderCreateRequest,
    OrderItem,
    OrdersListResponse,
    FillItem,
    FillsListResponse,
)
from tradepilot.api.dependencies import get_db, require_role
from tradepilot.db.models import BrokerOrderRecord, OrderFillRecord, OrderIntentRecord, RiskEventRecord, User
from tradepilot.execution.agent import TradingAgent

router = APIRouter(prefix="/api/v1", tags=["Orders & Trades"])

_agent_instance = None


def get_agent() -> TradingAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TradingAgent()
    return _agent_instance


@router.get("/orders", response_model=OrdersListResponse)
def get_orders(
    symbol: Optional[str] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List submitted paper orders with broker status transitions."""
    query = db.query(BrokerOrderRecord)
    if symbol:
        query = query.filter(BrokerOrderRecord.symbol == symbol.upper())
    if status_filter:
        query = query.filter(BrokerOrderRecord.status == status_filter.upper())

    records = query.order_by(desc(BrokerOrderRecord.submitted_at)).limit(limit).all()
    items = [
        OrderItem(
            id=r.id,
            client_order_id=r.client_order_id,
            broker_order_id=r.broker_order_id,
            symbol=r.symbol,
            side=r.side,
            qty=r.qty,
            filled_qty=r.filled_qty,
            status=r.status,
            order_type=r.order_type,
            time_in_force=r.time_in_force,
            avg_fill_price=r.avg_fill_price,
            submitted_at=r.submitted_at,
            updated_at=r.updated_at,
        )
        for r in records
    ]
    return OrdersListResponse(total=len(items), orders=items)


@router.get("/orders/{order_id}", response_model=OrderItem)
def get_order_by_id(
    order_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve detailed order record by client_order_id or broker_order_id."""
    record = (
        db.query(BrokerOrderRecord)
        .filter(
            (BrokerOrderRecord.client_order_id == order_id)
            | (BrokerOrderRecord.broker_order_id == order_id)
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found.")

    return OrderItem(
        id=record.id,
        client_order_id=record.client_order_id,
        broker_order_id=record.broker_order_id,
        symbol=record.symbol,
        side=record.side,
        qty=record.qty,
        filled_qty=record.filled_qty,
        status=record.status,
        order_type=record.order_type,
        time_in_force=record.time_in_force,
        avg_fill_price=record.avg_fill_price,
        submitted_at=record.submitted_at,
        updated_at=record.updated_at,
    )


@router.get("/trades", response_model=FillsListResponse)
def get_trades(
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List execution fills reported by the brokerage."""
    query = db.query(OrderFillRecord)
    if symbol:
        query = query.filter(OrderFillRecord.symbol == symbol.upper())
    records = query.order_by(desc(OrderFillRecord.timestamp)).limit(limit).all()

    items = [
        FillItem(
            id=r.id,
            broker_order_id=r.broker_order_id,
            symbol=r.symbol,
            side=r.side,
            shares=r.shares,
            price=r.price,
            fee=r.fee,
            timestamp=r.timestamp,
        )
        for r in records
    ]
    return FillsListResponse(total=len(items), fills=items)


@router.post("/orders", response_model=OrderItem, status_code=status.HTTP_201_CREATED)
def submit_manual_order(
    req: OrderCreateRequest,
    current_user: User = Depends(require_role(["trader", "admin"])),
    db: Session = Depends(get_db),
):
    """Submit a validated manual paper-trading order through Centralized Risk Engine."""
    agent = get_agent()
    sym = req.symbol.upper()
    norm_side = req.side.lower()
    now = datetime.now(timezone.utc)
    client_order_id = f"manual-{uuid.uuid4().hex[:12]}"

    acc = agent.broker.get_account()
    positions = agent.broker.get_positions()
    curr_price = agent.broker.get_latest_price(sym)

    # 1. Record intent
    intent = OrderIntentRecord(
        client_order_id=client_order_id,
        symbol=sym,
        side=norm_side,
        target_qty=req.qty,
        proposed_price=curr_price,
        status="PENDING",
        created_at=now,
    )
    db.add(intent)

    # 2. Centralized Risk Engine Evaluation
    risk_eval = agent.risk_engine.evaluate_order(
        symbol=sym,
        side=norm_side,
        qty=req.qty,
        price=curr_price,
        account=acc,
        current_positions=positions,
        client_order_id=client_order_id,
    )

    if not risk_eval.is_approved:
        intent.status = "REJECTED"
        intent.risk_notes = risk_eval.reason
        risk_event = RiskEventRecord(
            rule_name=risk_eval.rule_name or "MANUAL_ORDER_REJECTED",
            severity=risk_eval.severity,
            description=risk_eval.reason or "Blocked by risk engine.",
            blocked_order_id=client_order_id,
        )
        db.add(risk_event)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Order rejected by risk engine: {risk_eval.reason}",
        )

    # 3. Approved: Submit to paper broker
    intent.status = "APPROVED"
    broker_resp = agent.broker.submit_order(
        symbol=sym,
        qty=req.qty,
        side=norm_side,
        order_type=req.order_type,
        time_in_force=req.time_in_force,
        client_order_id=client_order_id,
    )

    b_order = BrokerOrderRecord(
        client_order_id=client_order_id,
        broker_order_id=broker_resp.order_id,
        symbol=sym,
        side=norm_side,
        qty=req.qty,
        filled_qty=broker_resp.filled_qty,
        status=broker_resp.status,
        order_type=broker_resp.order_type,
        time_in_force=broker_resp.time_in_force,
        avg_fill_price=broker_resp.avg_fill_price,
        submitted_at=now,
    )
    db.add(b_order)

    if broker_resp.status == "FILLED" and broker_resp.avg_fill_price:
        fill = OrderFillRecord(
            broker_order_id=broker_resp.order_id,
            symbol=sym,
            side=norm_side,
            shares=req.qty,
            price=broker_resp.avg_fill_price,
            fee=round(req.qty * broker_resp.avg_fill_price * 0.0005, 4),
            timestamp=now,
        )
        db.add(fill)

    db.commit()
    db.refresh(b_order)

    # Reconcile portfolio
    agent.reconcile_portfolio(db)

    return OrderItem(
        id=b_order.id,
        client_order_id=b_order.client_order_id,
        broker_order_id=b_order.broker_order_id,
        symbol=b_order.symbol,
        side=b_order.side,
        qty=b_order.qty,
        filled_qty=b_order.filled_qty,
        status=b_order.status,
        order_type=b_order.order_type,
        time_in_force=b_order.time_in_force,
        avg_fill_price=b_order.avg_fill_price,
        submitted_at=b_order.submitted_at,
        updated_at=b_order.updated_at,
    )
