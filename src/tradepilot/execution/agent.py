"""
Automated Trading Agent
=======================
Coordinates market data ingestion, ML feature engineering, model inference,
signal generation, risk engine evaluation, paper order execution, and reconciliation.
"""

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
import joblib
import pandas as pd
from sqlalchemy.orm import Session

from tradepilot.api.config import settings
from tradepilot.db.models import (
    AgentStateModel,
    TradingCycleModel,
    PredictionRecord,
    SignalRecord,
    OrderIntentRecord,
    BrokerOrderRecord,
    OrderFillRecord,
    PositionSnapshotRecord,
    PortfolioSnapshotRecord,
    RiskEventRecord,
    AuditLogRecord,
)
from tradepilot.execution.broker_base import BaseBrokerAdapter
from tradepilot.execution.mock_adapter import MockBrokerAdapter
from tradepilot.execution.alpaca_adapter import AlpacaPaperBroker
from tradepilot.execution.risk_engine import CentralizedRiskEngine
from tradepilot.features.build_features import compute_technical_features
from tradepilot.backtesting.signals import SignalGenerator

logger = logging.getLogger("tradepilot.agent")


class TradingAgent:
    """Manages the lifecycle and execution loop of the TradePilot trading strategy."""

    def __init__(
        self,
        broker_adapter: Optional[BaseBrokerAdapter] = None,
        risk_engine: Optional[CentralizedRiskEngine] = None,
        symbols: Optional[List[str]] = None,
    ):
        self.symbols = symbols or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"
        ]
        self.risk_engine = risk_engine or CentralizedRiskEngine()
        
        # Determine broker adapter: prefer AlpacaPaperBroker if credentials configured, otherwise fallback to MockBrokerAdapter
        if broker_adapter:
            self.broker = broker_adapter
        else:
            if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY and settings.ALPACA_PAPER:
                try:
                    self.broker = AlpacaPaperBroker()
                    logger.info("Initialized Alpaca Paper Trading Broker.")
                except Exception as exc:
                    logger.warning(f"Could not connect to Alpaca Paper ({exc}). Falling back to MockBrokerAdapter.")
                    self.broker = MockBrokerAdapter()
            else:
                self.broker = MockBrokerAdapter()
                logger.info("Initialized Mock Paper Trading Broker (deterministic simulation).")

        self.model = None
        self.model_metadata = {}
        self.signal_generator = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model artifact and verify metadata."""
        if settings.MODEL_PATH.exists():
            self.model = joblib.load(settings.MODEL_PATH)
            if settings.MODEL_METADATA_PATH.exists():
                with open(settings.MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                    self.model_metadata = json.load(f)
            
            from tradepilot.config import DEFAULT_DATA_CONFIG
            feature_names = self.model_metadata.get(
                "features",
                DEFAULT_DATA_CONFIG.feature_cols,
            )

            try:
                self.signal_generator = SignalGenerator(
                    pipeline=self.model,
                    feature_cols=feature_names,
                    entry_threshold=settings.ENTRY_THRESHOLD,
                )
                logger.info("Successfully loaded model artifact, metadata, and SignalGenerator.")
            except Exception as e:
                logger.error(f"Failed to initialize SignalGenerator: {e}")
        else:
            logger.warning(f"Model artifact not found at {settings.MODEL_PATH}.")


    def get_state(self, db: Session) -> AgentStateModel:
        """Retrieve current agent state record from database."""
        state = db.query(AgentStateModel).filter_by(id=1).first()
        if not state:
            state = AgentStateModel(id=1, state="STOPPED", mode="PAPER")
            db.add(state)
            db.commit()
            db.refresh(state)
        return state

    def start(self, db: Session, user_id: Optional[int] = None) -> AgentStateModel:
        """Transition agent to RUNNING state."""
        state = self.get_state(db)
        if state.mode != "PAPER":
            raise ValueError("Refusing to start: Trading mode must strictly be PAPER.")
        state.state = "RUNNING"
        state.error_message = None
        state.last_heartbeat = datetime.now(timezone.utc)
        self.risk_engine.reset_emergency_stop()

        audit = AuditLogRecord(
            user_id=user_id,
            action="AGENT_START",
            entity_type="AgentState",
            entity_id="1",
            details_json=json.dumps({"state": "RUNNING", "mode": state.mode}),
        )
        db.add(audit)
        db.commit()
        db.refresh(state)
        return state

    def pause(self, db: Session, user_id: Optional[int] = None) -> AgentStateModel:
        """Transition agent to PAUSED state (no new orders; monitors existing)."""
        state = self.get_state(db)
        state.state = "PAUSED"
        state.last_heartbeat = datetime.now(timezone.utc)

        audit = AuditLogRecord(
            user_id=user_id,
            action="AGENT_PAUSE",
            entity_type="AgentState",
            entity_id="1",
            details_json=json.dumps({"state": "PAUSED"}),
        )
        db.add(audit)
        db.commit()
        db.refresh(state)
        return state

    def stop(self, db: Session, user_id: Optional[int] = None) -> AgentStateModel:
        """Transition agent to STOPPED state."""
        state = self.get_state(db)
        state.state = "STOPPED"
        state.last_heartbeat = datetime.now(timezone.utc)

        audit = AuditLogRecord(
            user_id=user_id,
            action="AGENT_STOP",
            entity_type="AgentState",
            entity_id="1",
            details_json=json.dumps({"state": "STOPPED"}),
        )
        db.add(audit)
        db.commit()
        db.refresh(state)
        return state

    def emergency_stop(self, db: Session, user_id: Optional[int] = None) -> int:
        """Global kill switch: Trip circuit breaker, cancel all open orders, set state to STOPPED."""
        state = self.get_state(db)
        state.state = "STOPPED"
        state.error_message = "Emergency stop activated by operator."
        state.last_heartbeat = datetime.now(timezone.utc)

        self.risk_engine.trip_emergency_stop("Operator Emergency Stop")

        cancelled_count = self.broker.cancel_all_orders()

        risk_event = RiskEventRecord(
            rule_name="EMERGENCY_STOP",
            severity="CRITICAL",
            description=f"Global emergency stop activated. Cancelled {cancelled_count} open orders.",
        )
        audit = AuditLogRecord(
            user_id=user_id,
            action="EMERGENCY_STOP",
            entity_type="AgentState",
            entity_id="1",
            details_json=json.dumps({"cancelled_orders": cancelled_count}),
        )
        db.add(risk_event)
        db.add(audit)
        db.commit()
        return cancelled_count

    def reconcile_portfolio(self, db: Session) -> PortfolioSnapshotRecord:
        """Sync internal position & account records against the broker source of truth."""
        acc = self.broker.get_account()
        positions = self.broker.get_positions()
        now = datetime.now(timezone.utc)

        gross_exposure = sum(p.market_value for p in positions)
        equity = max(1.0, acc.portfolio_value)
        exposure_pct = gross_exposure / equity

        # Record position snapshots
        for p in positions:
            pos_snap = PositionSnapshotRecord(
                timestamp=now,
                symbol=p.symbol,
                shares=p.qty,
                avg_entry_price=p.avg_entry_price,
                current_price=p.current_price,
                market_value=p.market_value,
                unrealized_pnl=p.unrealized_pl,
                unrealized_pnl_pct=p.unrealized_plpc,
            )
            db.add(pos_snap)

        # Record portfolio snapshot
        port_snap = PortfolioSnapshotRecord(
            timestamp=now,
            equity=acc.portfolio_value,
            cash=acc.cash,
            buying_power=acc.buying_power,
            gross_exposure=gross_exposure,
            exposure_pct=exposure_pct,
            day_pnl=sum(p.unrealized_pl for p in positions),
            total_pnl=acc.portfolio_value - 100000.0,  # relative to baseline cash
        )
        db.add(port_snap)
        db.commit()
        db.refresh(port_snap)
        return port_snap

    def run_cycle(self, db: Session, target_symbols: Optional[List[str]] = None) -> TradingCycleModel:
        """
        Execute one full trading cycle:
        1. Safeguards check
        2. Market data fetch & feature computation
        3. Model inference & class probabilities
        4. Strategy signal generation
        5. Position sizing & risk engine validation
        6. Paper order submission & fill tracking
        7. Reconciliation & database persistence
        """
        cycle_id = f"cycle-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        cycle_record = TradingCycleModel(
            id=cycle_id,
            started_at=now,
            status="RUNNING",
            symbols_processed=0,
            orders_generated=0,
        )
        db.add(cycle_record)
        db.commit()

        # Update heartbeat
        state = self.get_state(db)
        state.last_heartbeat = now
        state.last_cycle_at = now
        db.commit()

        # Check agent state
        if state.state != "RUNNING":
            cycle_record.status = "COMPLETED"
            cycle_record.completed_at = datetime.now(timezone.utc)
            cycle_record.error_details = f"Skipped: Agent is in {state.state} state."
            db.commit()
            return cycle_record

        # Check trading mode
        if state.mode != "PAPER" or not self.broker.is_paper():
            cycle_record.status = "FAILED"
            cycle_record.completed_at = datetime.now(timezone.utc)
            cycle_record.error_details = "CRITICAL: Live trading mode rejected. Must be PAPER."
            state.state = "ERROR"
            state.error_message = cycle_record.error_details
            db.commit()
            return cycle_record

        symbols_to_run = target_symbols or self.symbols
        from tradepilot.config import DEFAULT_DATA_CONFIG
        feature_names = self.model_metadata.get(
            "features",
            DEFAULT_DATA_CONFIG.feature_cols,
        )



        acc = self.broker.get_account()
        current_positions = self.broker.get_positions()
        pos_map = {p.symbol.upper(): p for p in current_positions}
        orders_submitted = 0

        try:
            for symbol in symbols_to_run:
                sym = symbol.upper()
                try:
                    # 1. Fetch recent bars
                    bars_df = self.broker.get_latest_bars(sym, limit=45)
                    if bars_df.empty or len(bars_df) < 30:
                        logger.warning(f"Insufficient bars for {sym} ({len(bars_df)}). Skipping.")
                        continue

                    # 2. Compute features
                    feat_df = compute_technical_features(bars_df)
                    valid_feat = feat_df.dropna(subset=feature_names)
                    if valid_feat.empty:
                        logger.warning(f"No valid features for {sym} after dropna. Skipping.")
                        continue

                    latest_row = valid_feat.iloc[-1]
                    X_input = valid_feat[feature_names].iloc[[-1]]

                    # 3. Model Inference & Strategy Signal Generation via SignalGenerator
                    if self.signal_generator is not None:
                        sig_res = self.signal_generator.generate_signals(valid_feat.iloc[[-1]])
                        row_res = sig_res.iloc[0]
                        prob_up = float(row_res["prob_up"])
                        prob_neutral = float(row_res["prob_neutral"])
                        prob_down = float(row_res["prob_down"])
                        raw_sig = int(row_res["signal"])
                        sig_val = "BUY" if raw_sig == 1 else ("EXIT" if raw_sig == -1 else "HOLD")
                    else:
                        prob_down, prob_neutral, prob_up = 0.25, 0.40, 0.35
                        sig_val = "HOLD"

                    predicted_class = "UP" if prob_up > max(prob_down, prob_neutral) else (
                        "DOWN" if prob_down > prob_neutral else "NEUTRAL"
                    )

                    pred_rec = PredictionRecord(
                        cycle_id=cycle_id,
                        timestamp=now,
                        symbol=sym,
                        prob_up=prob_up,
                        prob_neutral=prob_neutral,
                        prob_down=prob_down,
                        predicted_class=predicted_class,
                        model_version=self.model_metadata.get("algorithm", "RandomForest_v1"),
                    )
                    db.add(pred_rec)

                    reasoning = f"prob_up={prob_up:.3f}, prob_down={prob_down:.3f}, threshold={settings.ENTRY_THRESHOLD:.2f}"
                    sig_rec = SignalRecord(
                        cycle_id=cycle_id,
                        timestamp=now,
                        symbol=sym,
                        signal=sig_val,
                        reasoning=reasoning,
                        threshold=settings.ENTRY_THRESHOLD,
                    )
                    db.add(sig_rec)


                    # 5. Position Sizing and Action
                    curr_price = float(latest_row["close"])
                    existing_pos = pos_map.get(sym)

                    order_side = None
                    target_qty = 0.0

                    if sig_val == "BUY" and (not existing_pos or existing_pos.qty <= 0):
                        # Max position size = 10% of portfolio equity
                        target_capital = min(acc.portfolio_value * settings.MAX_POSITION_SIZE_PCT, acc.buying_power)
                        shares_to_buy = math.floor(target_capital / curr_price)
                        if shares_to_buy > 0:
                            order_side = "buy"
                            target_qty = float(shares_to_buy)

                    elif sig_val == "EXIT" and existing_pos and existing_pos.qty > 0:
                        order_side = "sell"
                        target_qty = float(existing_pos.qty)

                    # 6. Risk Engine Gating
                    if order_side and target_qty > 0:
                        client_order_id = f"tp-{cycle_id}-{sym}-{order_side}"
                        intent = OrderIntentRecord(
                            client_order_id=client_order_id,
                            cycle_id=cycle_id,
                            symbol=sym,
                            side=order_side,
                            target_qty=target_qty,
                            proposed_price=curr_price,
                            status="PENDING",
                            created_at=now,
                        )
                        db.add(intent)

                        # Evaluate against risk engine
                        risk_eval = self.risk_engine.evaluate_order(
                            symbol=sym,
                            side=order_side,
                            qty=target_qty,
                            price=curr_price,
                            account=acc,
                            current_positions=current_positions,
                            client_order_id=client_order_id,
                        )

                        if risk_eval.is_approved:
                            intent.status = "APPROVED"
                            db.commit()

                            # 7. Submit Paper Order to Broker
                            broker_resp = self.broker.submit_order(
                                symbol=sym,
                                qty=target_qty,
                                side=order_side,
                                order_type="market",
                                time_in_force="day",
                                client_order_id=client_order_id,
                            )

                            b_order = BrokerOrderRecord(
                                client_order_id=client_order_id,
                                broker_order_id=broker_resp.order_id,
                                symbol=sym,
                                side=order_side,
                                qty=target_qty,
                                filled_qty=broker_resp.filled_qty,
                                status=broker_resp.status,
                                order_type=broker_resp.order_type,
                                time_in_force=broker_resp.time_in_force,
                                avg_fill_price=broker_resp.avg_fill_price,
                                submitted_at=now,
                            )
                            db.add(b_order)

                            # If filled immediately (as in mock adapter), record fill
                            if broker_resp.status == "FILLED" and broker_resp.avg_fill_price:
                                fill = OrderFillRecord(
                                    broker_order_id=broker_resp.order_id,
                                    symbol=sym,
                                    side=order_side,
                                    shares=target_qty,
                                    price=broker_resp.avg_fill_price,
                                    fee=round(target_qty * broker_resp.avg_fill_price * 0.0005, 4),
                                    timestamp=now,
                                )
                                db.add(fill)

                            orders_submitted += 1
                            logger.info(f"Order submitted: {order_side} {target_qty} {sym} -> {broker_resp.status}")

                        else:
                            intent.status = "REJECTED"
                            intent.risk_notes = risk_eval.reason
                            risk_event = RiskEventRecord(
                                rule_name=risk_eval.rule_name or "RISK_VIOLATION",
                                severity=risk_eval.severity,
                                description=risk_eval.reason or "Blocked by risk engine.",
                                blocked_order_id=client_order_id,
                            )
                            db.add(risk_event)
                            logger.warning(f"Order blocked by risk: {sym} {order_side} - {risk_eval.reason}")

                    cycle_record.symbols_processed += 1

                except Exception as sym_exc:
                    logger.error(f"Error processing symbol {sym}: {sym_exc}", exc_info=True)

            self.risk_engine.record_success()

        except Exception as cycle_exc:
            logger.error(f"Fatal error in trading cycle {cycle_id}: {cycle_exc}", exc_info=True)
            self.risk_engine.record_failure(str(cycle_exc))
            cycle_record.status = "FAILED"
            cycle_record.error_details = str(cycle_exc)

        # Reconcile portfolio after cycle
        self.reconcile_portfolio(db)

        cycle_record.status = "COMPLETED"
        cycle_record.completed_at = datetime.now(timezone.utc)
        cycle_record.orders_generated = orders_submitted
        db.commit()
        db.refresh(cycle_record)
        return cycle_record
