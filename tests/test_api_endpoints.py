"""
TradePilot API Integration Tests
================================
Validates all versioned REST API endpoints, authentication, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from tradepilot.api.app import app
from tradepilot.api.config import settings
from tradepilot.db.session import init_db

# Initialize database and tables
init_db()
client = TestClient(app)


def test_health_endpoint():
    """Verify liveness probe returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION
    assert "timestamp" in data


def test_ready_endpoint():
    """Verify readiness probe checks dependencies without exposing secrets."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["database"] is True
    assert data["model_loaded"] is True
    assert data["trading_mode"] == "PAPER"
    assert data["paper_safeguard_active"] is True


def test_system_status_endpoint():
    """Verify operational system status."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "TradePilot"
    assert data["trading_mode"] == "PAPER"
    assert data["database_status"] == "connected"
    assert data["uptime_seconds"] >= 0


def test_auth_login_and_me():
    """Test login with default admin credentials and verify /me profile endpoint."""
    login_resp = client.post(
        "/api/v1/auth/login-json",
        json={
            "username": settings.ADMIN_USERNAME,
            "password": settings.ADMIN_INITIAL_PASSWORD,
        },
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    assert token_data["role"] == "admin"

    token = token_data["access_token"]
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    user_info = me_resp.json()
    assert user_info["username"] == settings.ADMIN_USERNAME
    assert user_info["role"] == "admin"


def test_auth_login_invalid_credentials():
    """Test login failure with bad password."""
    login_resp = client.post(
        "/api/v1/auth/login-json",
        json={
            "username": settings.ADMIN_USERNAME,
            "password": "WrongPassword123!",
        },
    )
    assert login_resp.status_code == 401


def test_markets_symbols_and_bars():
    """Test retrieving approved symbols and daily OHLCV bars."""
    sym_resp = client.get("/api/v1/markets/symbols")
    assert sym_resp.status_code == 200
    symbols = sym_resp.json()
    assert "AAPL" in symbols
    assert "MSFT" in symbols

    bars_resp = client.get("/api/v1/markets/bars?symbol=AAPL&limit=10")
    assert bars_resp.status_code == 200
    bars_data = bars_resp.json()
    assert bars_data["symbol"] == "AAPL"
    assert len(bars_data["bars"]) > 0
    assert "close" in bars_data["bars"][0]


def test_markets_latest_and_status():
    """Test latest market prices and provider connectivity status."""
    latest_resp = client.get("/api/v1/markets/latest")
    assert latest_resp.status_code == 200
    prices = latest_resp.json()
    assert prices["is_paper"] is True
    assert len(prices["prices"]) > 0

    status_resp = client.get("/api/v1/markets/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["connected"] is True
    assert status_data["paper_trading"] is True


def test_active_model_and_predictions():
    """Test active model metadata inspection and predictions retrieval."""
    model_resp = client.get("/api/v1/models/active")
    assert model_resp.status_code == 200
    meta = model_resp.json()
    assert meta["num_features"] == 8
    assert "classes" in meta

    pred_resp = client.get("/api/v1/predictions/latest")
    assert pred_resp.status_code == 200
    assert "predictions" in pred_resp.json()


def test_portfolio_summary_and_positions():
    """Test portfolio overview, positions list, and equity history."""
    port_resp = client.get("/api/v1/portfolio")
    assert port_resp.status_code == 200
    summary = port_resp.json()
    assert summary["equity"] > 0
    assert summary["trading_mode"] == "PAPER"

    pos_resp = client.get("/api/v1/portfolio/positions")
    assert pos_resp.status_code == 200
    assert isinstance(pos_resp.json(), list)

    equity_resp = client.get("/api/v1/portfolio/equity")
    assert equity_resp.status_code == 200
    assert len(equity_resp.json()["points"]) > 0


def test_agent_status_and_lifecycle():
    """Test agent state querying and start/pause/stop transitions."""
    # Obtain auth token
    login_resp = client.post(
        "/api/v1/auth/login-json",
        json={
            "username": settings.ADMIN_USERNAME,
            "password": settings.ADMIN_INITIAL_PASSWORD,
        },
    )
    token = login_resp.json()["access_token"]
    auth_header = {"Authorization": f"Bearer {token}"}

    # Query status
    status_resp = client.get("/api/v1/agent/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["is_paper_trading"] is True

    # Pause agent
    pause_resp = client.post("/api/v1/agent/pause", headers=auth_header)
    assert pause_resp.status_code == 200
    assert pause_resp.json()["state"] == "PAUSED"

    # Stop agent
    stop_resp = client.post("/api/v1/agent/stop", headers=auth_header)
    assert stop_resp.status_code == 200
    assert stop_resp.json()["state"] == "STOPPED"


def test_risk_status_and_config():
    """Test risk engine status and configured limits."""
    cfg_resp = client.get("/api/v1/risk/config")
    assert cfg_resp.status_code == 200
    cfg = cfg_resp.json()
    assert cfg["max_position_size_pct"] == settings.MAX_POSITION_SIZE_PCT
    assert cfg["max_concurrent_positions"] == settings.MAX_CONCURRENT_POSITIONS

    status_resp = client.get("/api/v1/risk/status")
    assert status_resp.status_code == 200
    assert "healthy" in status_resp.json()
