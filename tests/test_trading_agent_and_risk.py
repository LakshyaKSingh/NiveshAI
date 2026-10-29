"""
Trading Agent and Centralized Risk Engine Unit & Integration Tests
=================================================================
Validates risk limits, order deduplication, paper execution,
reconciliation, and emergency stop kill-switch behavior.
"""

import pytest
from tradepilot.execution.broker_base import BrokerAccount, BrokerPosition
from tradepilot.execution.mock_adapter import MockBrokerAdapter
from tradepilot.execution.risk_engine import CentralizedRiskEngine
from tradepilot.execution.agent import TradingAgent
from tradepilot.db.session import SessionLocal, init_db


@pytest.fixture
def mock_account():
    return BrokerAccount(
        account_number="TEST_ACC",
        status="ACTIVE",
        currency="USD",
        cash=100000.0,
        buying_power=100000.0,
        portfolio_value=100000.0,
        is_paper=True,
        trading_blocked=False,
    )


def test_risk_engine_approves_valid_order(mock_account):
    """Verify that a compliant order within all limits is approved."""
    engine = CentralizedRiskEngine()
    result = engine.evaluate_order(
        symbol="AAPL",
        side="buy",
        qty=10.0,
        price=150.0,
        account=mock_account,
        current_positions=[],
        client_order_id="test-order-1",
    )
    assert result.is_approved is True
    assert result.rule_name == "ALL_CHECKS_PASSED"


def test_risk_engine_enforces_max_position_size(mock_account):
    """Verify orders exceeding MAX_POSITION_SIZE_PCT (10% of equity) are rejected."""
    engine = CentralizedRiskEngine(max_position_size_pct=0.10)
    # Order value = 100 shares * $150 = $15,000, which is 15% of $100k equity (exceeds 10%)
    result = engine.evaluate_order(
        symbol="AAPL",
        side="buy",
        qty=100.0,
        price=150.0,
        account=mock_account,
        current_positions=[],
        client_order_id="test-order-2",
    )
    assert result.is_approved is False
    assert result.rule_name == "MAX_POSITION_SIZE"
    assert "exceeds max position size limit" in result.reason


def test_risk_engine_enforces_max_concurrent_positions(mock_account):
    """Verify that opening an 11th position when max is 10 is blocked."""
    engine = CentralizedRiskEngine(max_concurrent_positions=2)
    # Already holding 2 positions
    pos1 = BrokerPosition("AAPL", 10, 1500, 1500, 0, 0, 150, 150)
    pos2 = BrokerPosition("MSFT", 10, 2000, 2000, 0, 0, 200, 200)

    # Attempt to buy a 3rd distinct symbol
    result = engine.evaluate_order(
        symbol="GOOGL",
        side="buy",
        qty=5.0,
        price=100.0,
        account=mock_account,
        current_positions=[pos1, pos2],
        client_order_id="test-order-3",
    )
    assert result.is_approved is False
    assert result.rule_name == "MAX_CONCURRENT_POSITIONS"


def test_risk_engine_prevents_duplicate_orders(mock_account):
    """Verify that an order with a repeated client_order_id is blocked."""
    engine = CentralizedRiskEngine()
    cid = "duplicate-test-cid"
    r1 = engine.evaluate_order("AAPL", "buy", 5.0, 150.0, mock_account, [], client_order_id=cid)
    assert r1.is_approved is True

    # Same client_order_id submitted again
    r2 = engine.evaluate_order("AAPL", "buy", 5.0, 150.0, mock_account, [], client_order_id=cid)
    assert r2.is_approved is False
    assert r2.rule_name == "DUPLICATE_ORDER"


def test_risk_engine_emergency_stop_blocks_all_orders(mock_account):
    """Verify that tripping the emergency stop immediately blocks all orders."""
    engine = CentralizedRiskEngine()
    engine.trip_emergency_stop("Operator Drill")

    result = engine.evaluate_order("AAPL", "buy", 5.0, 150.0, mock_account, [], client_order_id="test-stop")
    assert result.is_approved is False
    assert result.rule_name == "EMERGENCY_STOP"
    assert result.severity == "CRITICAL"


def test_risk_engine_rejects_non_paper_mode(mock_account):
    """Verify that live accounts are strictly refused."""
    engine = CentralizedRiskEngine()
    mock_account.is_paper = False  # Simulate accidental live account

    result = engine.evaluate_order("AAPL", "buy", 5.0, 150.0, mock_account, [])
    assert result.is_approved is False
    assert result.rule_name == "PAPER_SAFEGUARD"


def test_mock_broker_order_execution_and_fees():
    """Verify MockBrokerAdapter executes orders with slippage, fees, and updates positions."""
    broker = MockBrokerAdapter(initial_cash=100000.0, slippage_bps=5.0, commission_bps=5.0)
    acc_initial = broker.get_account()
    assert acc_initial.cash == 100000.0

    # Submit buy order for 10 shares of AAPL
    order = broker.submit_order(symbol="AAPL", qty=10.0, side="buy")
    assert order.status == "FILLED"
    assert order.filled_qty == 10.0
    assert order.avg_fill_price > 0

    # Verify positions and cash
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10.0

    acc_after = broker.get_account()
    assert acc_after.cash < 100000.0
    assert len(broker.fills) == 1

    # Sell 5 shares
    sell_order = broker.submit_order(symbol="AAPL", qty=5.0, side="sell")
    assert sell_order.status == "FILLED"
    rem_pos = broker.get_position("AAPL")
    assert rem_pos.qty == 5.0


def test_trading_agent_lifecycle_and_reconciliation():
    """Test full TradingAgent workflow with DB persistence."""
    init_db()
    db = SessionLocal()
    agent = TradingAgent(broker_adapter=MockBrokerAdapter())

    # Test start
    state = agent.start(db)
    assert state.state == "RUNNING"
    assert state.mode == "PAPER"

    # Test run cycle on subset of symbols
    cycle = agent.run_cycle(db, target_symbols=["AAPL"])
    assert cycle.status == "COMPLETED"
    assert cycle.symbols_processed == 1

    # Test pause
    state = agent.pause(db)
    assert state.state == "PAUSED"

    # Test emergency stop
    cancelled = agent.emergency_stop(db)
    state = agent.get_state(db)
    assert state.state == "STOPPED"
    assert agent.risk_engine.circuit_breaker_tripped is True

    db.close()
