"""
Database Session Management and Initialization
==============================================
Provides SQLAlchemy engine, session factory, FastAPI dependency,
and database initialization/seeding utilities.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from tradepilot.api.config import settings
from tradepilot.db.base import Base
import tradepilot.db.models  # ensure models are registered


# Configure engine with SQLite thread safety flags when using SQLite
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency providing a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(seed_admin: bool = True) -> None:
    """Create all tables and seed initial agent state and admin user if needed."""
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        # 1. Initialize AgentState singleton if missing
        agent_state = db.query(tradepilot.db.models.AgentStateModel).filter_by(id=1).first()
        if not agent_state:
            agent_state = tradepilot.db.models.AgentStateModel(
                id=1,
                state="STOPPED",
                mode="PAPER",
                error_message=None,
            )
            db.add(agent_state)

        # 2. Seed default admin user if missing
        if seed_admin:
            admin_user = (
                db.query(tradepilot.db.models.User)
                .filter_by(username=settings.ADMIN_USERNAME)
                .first()
            )
            if not admin_user:
                # Import password hasher lazily to avoid circular imports
                from tradepilot.api.security import get_password_hash

                admin_user = tradepilot.db.models.User(
                    username=settings.ADMIN_USERNAME,
                    email=settings.ADMIN_EMAIL,
                    hashed_password=get_password_hash(settings.ADMIN_INITIAL_PASSWORD),
                    role="admin",
                    is_active=True,
                )
                db.add(admin_user)

        db.commit()
