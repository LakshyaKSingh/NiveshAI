"""
Background Worker and Strategy Scheduler
=========================================
Runs scheduled trading cycles in an autonomous background process.
Ensures single-instance coordination and graceful shutdown.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from tradepilot.api.config import settings
from tradepilot.db.session import SessionLocal, init_db
from tradepilot.execution.agent import TradingAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tradepilot.worker")


class BackgroundWorker:
    """Autonomous trading worker with graceful lifecycle management."""

    def __init__(self, cycle_interval_seconds: int = 60):
        self.interval = cycle_interval_seconds
        self.is_running = False
        self.agent = TradingAgent()

    def handle_signal(self, signum, frame):
        logger.info(f"Received termination signal ({signum}). Initiating graceful worker shutdown...")
        self.is_running = False

    def run(self):
        logger.info("Initializing TradePilot Background Worker...")
        init_db()

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        self.is_running = True
        logger.info(f"Worker active. Cycle interval: {self.interval}s. Trading mode: {settings.TRADING_MODE}")

        while self.is_running:
            db = SessionLocal()
            try:
                state = self.agent.get_state(db)
                if state.state == "RUNNING":
                    logger.info("Agent is RUNNING. Executing scheduled trading cycle...")
                    cycle = self.agent.run_cycle(db)
                    logger.info(
                        f"Cycle {cycle.id} finished with status '{cycle.status}'. "
                        f"Symbols: {cycle.symbols_processed}, Orders: {cycle.orders_generated}"
                    )
                else:
                    # Update heartbeat even when idle or paused
                    state.last_heartbeat = datetime.now(timezone.utc)
                    db.commit()
                    logger.debug(f"Agent state is '{state.state}'. Standing by.")

            except Exception as e:
                logger.error(f"Worker encountered an unhandled cycle error: {e}", exc_info=True)
            finally:
                db.close()

            # Sleep in 1-second slices so shutdown signal is responsive
            for _ in range(self.interval):
                if not self.is_running:
                    break
                time.sleep(1)

        logger.info("Background Worker shutdown cleanly.")


if __name__ == "__main__":
    worker = BackgroundWorker()
    worker.run()
