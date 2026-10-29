"""
TradePilot FastAPI Application
==============================
Main entrypoint for TradePilot REST API, mounting all versioned routers,
CORS security policies, database initialization, and global error handlers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from tradepilot.api.config import settings
from tradepilot.db.session import init_db
from tradepilot.api.routers import (
    system,
    auth,
    markets,
    predictions,
    portfolio,
    orders,
    agent,
    risk,
)

logger = logging.getLogger("tradepilot.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle events: initialize DB and seed on startup."""
    logger.info("Initializing TradePilot Database and initial state...")
    init_db()
    logger.info(f"TradePilot API {settings.VERSION} online in {settings.TRADING_MODE} mode.")
    yield
    logger.info("TradePilot API shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-Structured REST API for TradePilot: Automated ML-Driven "
        "Paper Trading and Portfolio Management System."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Security Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(system.router)
app.include_router(auth.router)
app.include_router(markets.router)
app.include_router(predictions.router)
app.include_router(portfolio.router)
app.include_router(orders.router)
app.include_router(agent.router)
app.include_router(risk.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled server errors and format consistent JSON response."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please check system logs."},
    )
