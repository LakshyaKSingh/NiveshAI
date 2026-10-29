"""
API and Application Configuration
=================================
Manages environment variables, database URLs, JWT parameters,
broker credentials, and operational safeguards using Pydantic Settings.
"""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application & Environment
    PROJECT_NAME: str = "TradePilot"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = False

    # Security & Authentication
    SECRET_KEY: str = Field(
        default="tradepilot-insecure-dev-secret-key-change-in-production-min32chars",
        description="JWT signature secret key",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ADMIN_USERNAME: str = "admin"
    ADMIN_EMAIL: str = "admin@tradepilot.io"
    ADMIN_INITIAL_PASSWORD: str = "TradePilotAdmin2026!"


    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Database: Defaults to local SQLite, easily overridden with postgresql:// in AWS/Docker
    DATABASE_URL: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'tradepilot.db'}",
        description="SQLAlchemy database connection URI",
    )
    DB_ECHO: bool = False

    # Alpaca Brokerage (Strict Paper-Trading Verification)
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPACA_PAPER: bool = True
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    ALPACA_DATA_URL: str = "https://data.alpaca.markets"
    
    # Operational Mode Safeguard (Must be 'PAPER' in this phase)
    TRADING_MODE: str = "PAPER"  # Strictly enforces PAPER trading

    # Risk Management Defaults
    MAX_POSITION_SIZE_PCT: float = 0.10      # Max 10% equity in any single stock
    MAX_CONCURRENT_POSITIONS: int = 10       # Max 10 open positions
    MAX_TOTAL_EXPOSURE_PCT: float = 0.95      # Max 95% total gross capital exposure
    MAX_DAILY_LOSS_PCT: float = 0.05         # 5% max daily drawdown triggers circuit breaker
    DEFAULT_SLIPPAGE_BPS: float = 5.0        # 5 basis points adverse slippage
    DEFAULT_COMMISSION_BPS: float = 5.0      # 5 basis points commission
    ENTRY_THRESHOLD: float = 0.40            # P(UP) >= 0.40 to buy
    MAX_CONSECUTIVE_FAILURES: int = 3        # Circuit breaker trigger on repeated broker/data errors

    # Model Artifact Paths
    MODEL_PATH: Path = PROJECT_ROOT / "models" / "best_model.joblib"
    MODEL_METADATA_PATH: Path = PROJECT_ROOT / "models" / "model_metadata.json"


settings = Settings()
