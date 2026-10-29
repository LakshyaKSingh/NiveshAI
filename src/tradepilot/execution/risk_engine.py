"""
Centralized Risk Management Engine
==================================
Enforces quantitative trading limits, portfolio safeguards,
duplicate-order prevention, and emergency kill-switch circuit breakers.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Set
from tradepilot.api.config import settings
from tradepilot.execution.broker_base import BrokerAccount, BrokerPosition


@dataclass
class RiskValidationResult:
    """Result of risk evaluation on a proposed order intent."""
    is_approved: bool
    rule_name: Optional[str] = None
    reason: Optional[str] = None
    severity: str = "INFO"  # INFO, WARNING, CRITICAL


class CentralizedRiskEngine:
    """Evaluates orders and portfolio state against hard safety boundaries."""

    def __init__(
        self,
        max_position_size_pct: float = settings.MAX_POSITION_SIZE_PCT,
        max_concurrent_positions: int = settings.MAX_CONCURRENT_POSITIONS,
        max_total_exposure_pct: float = settings.MAX_TOTAL_EXPOSURE_PCT,
        max_daily_loss_pct: float = settings.MAX_DAILY_LOSS_PCT,
        max_consecutive_failures: int = settings.MAX_CONSECUTIVE_FAILURES,
        supported_symbols: Optional[List[str]] = None,
    ):
        self.max_position_size_pct = max_position_size_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_failures = max_consecutive_failures
        self.supported_symbols: Set[str] = set(
            s.upper() for s in (supported_symbols or [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"
            ])
        )

        # Operational state
        self.emergency_stop_active: bool = False
        self.circuit_breaker_tripped: bool = False
        self.circuit_breaker_reason: Optional[str] = None
        self.consecutive_failures: int = 0
        self.recent_client_order_ids: Set[str] = set()
        self.last_evaluation_time: datetime = datetime.now(timezone.utc)

    def trip_emergency_stop(self, reason: str = "Manual Emergency Stop Activated") -> None:
        """Manually trip global emergency stop."""
        self.emergency_stop_active = True
        self.circuit_breaker_tripped = True
        self.circuit_breaker_reason = reason

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop status after administrative review."""
        self.emergency_stop_active = False
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = None
        self.consecutive_failures = 0

    def record_failure(self, error_msg: str) -> bool:
        """Track failure count and trip circuit breaker if threshold reached."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = (
                f"Tripped: {self.consecutive_failures} consecutive failures. Last error: {error_msg}"
            )
            return True
        return False

    def record_success(self) -> None:
        """Reset failure counter on successful operation."""
        self.consecutive_failures = 0

    def evaluate_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        account: BrokerAccount,
        current_positions: List[BrokerPosition],
        client_order_id: Optional[str] = None,
        market_data_age_seconds: float = 0.0,
    ) -> RiskValidationResult:
        """Evaluate a candidate order against all risk rules."""
        self.last_evaluation_time = datetime.now(timezone.utc)
        sym = symbol.upper()
        norm_side = side.lower()

        # Rule 1: Emergency stop / Circuit breaker
        if self.emergency_stop_active or self.circuit_breaker_tripped:
            return RiskValidationResult(
                is_approved=False,
                rule_name="EMERGENCY_STOP",
                reason=f"Blocked: Circuit breaker or emergency stop is active ({self.circuit_breaker_reason})",
                severity="CRITICAL",
            )

        # Rule 2: Trading mode safeguard
        if not account.is_paper or settings.TRADING_MODE != "PAPER":
            return RiskValidationResult(
                is_approved=False,
                rule_name="PAPER_SAFEGUARD",
                reason="CRITICAL: Live trading mode detected or trading mode is not PAPER.",
                severity="CRITICAL",
            )

        # Rule 3: Account trading blocked
        if account.trading_blocked:
            return RiskValidationResult(
                is_approved=False,
                rule_name="ACCOUNT_BLOCKED",
                reason="Broker account status indicates trading is blocked.",
                severity="CRITICAL",
            )

        # Rule 4: Symbol validity
        if sym not in self.supported_symbols:
            return RiskValidationResult(
                is_approved=False,
                rule_name="UNSUPPORTED_SYMBOL",
                reason=f"Symbol {sym} is not in the approved trading universe.",
                severity="WARNING",
            )

        # Rule 5: Quantity validation
        if qty <= 0:
            return RiskValidationResult(
                is_approved=False,
                rule_name="INVALID_QUANTITY",
                reason=f"Order quantity must be positive, got {qty}",
                severity="WARNING",
            )

        # Rule 6: Duplicate order prevention
        if client_order_id:
            if client_order_id in self.recent_client_order_ids:
                return RiskValidationResult(
                    is_approved=False,
                    rule_name="DUPLICATE_ORDER",
                    reason=f"Order intent {client_order_id} was already submitted or processed.",
                    severity="WARNING",
                )

        # Rule 7: Data freshness
        if market_data_age_seconds > 86400 * 3:  # 3 days max staleness for daily bars over weekends
            return RiskValidationResult(
                is_approved=False,
                rule_name="STALE_MARKET_DATA",
                reason=f"Market data age ({market_data_age_seconds:.0f}s) exceeds freshness threshold.",
                severity="WARNING",
            )

        # Calculate current position map and portfolio metrics
        pos_map = {p.symbol.upper(): p for p in current_positions}
        existing_pos = pos_map.get(sym)
        portfolio_equity = max(1.0, account.portfolio_value)
        order_value = qty * price

        # Rule 8: Max position size check (for buys)
        if norm_side == "buy":
            existing_value = (existing_pos.qty * price) if existing_pos else 0.0
            total_after_order = existing_value + order_value
            max_allowed_val = portfolio_equity * self.max_position_size_pct
            if total_after_order > max_allowed_val * 1.02:  # allow 2% tolerance for price drift
                return RiskValidationResult(
                    is_approved=False,
                    rule_name="MAX_POSITION_SIZE",
                    reason=(
                        f"Order value ${total_after_order:.2f} exceeds max position size limit "
                        f"${max_allowed_val:.2f} ({self.max_position_size_pct * 100:.1f}% of equity)"
                    ),
                    severity="WARNING",
                )

        # Rule 9: Max concurrent positions check (for new buys)
        if norm_side == "buy" and (not existing_pos or existing_pos.qty <= 0):
            if len(pos_map) >= self.max_concurrent_positions:
                return RiskValidationResult(
                    is_approved=False,
                    rule_name="MAX_CONCURRENT_POSITIONS",
                    reason=(
                        f"Current open positions ({len(pos_map)}) reached maximum allowed "
                        f"({self.max_concurrent_positions})"
                    ),
                    severity="WARNING",
                )

        # Rule 10: Gross portfolio exposure check
        current_gross_exposure = sum(p.market_value for p in current_positions)
        new_gross_exposure = current_gross_exposure + (order_value if norm_side == "buy" else -order_value)
        max_allowed_exposure = portfolio_equity * self.max_total_exposure_pct
        if norm_side == "buy" and new_gross_exposure > max_allowed_exposure:
            return RiskValidationResult(
                is_approved=False,
                rule_name="MAX_GROSS_EXPOSURE",
                reason=(
                    f"New gross exposure ${new_gross_exposure:.2f} exceeds "
                    f"${max_allowed_exposure:.2f} ({self.max_total_exposure_pct * 100:.1f}% limit)"
                ),
                severity="WARNING",
            )

        # Rule 11: Available cash / buying power check
        if norm_side == "buy":
            if order_value > account.buying_power:
                return RiskValidationResult(
                    is_approved=False,
                    rule_name="INSUFFICIENT_BUYING_POWER",
                    reason=(
                        f"Required order capital ${order_value:.2f} exceeds available buying power "
                        f"${account.buying_power:.2f}"
                    ),
                    severity="WARNING",
                )

        # Rule 12: Sell quantity cannot exceed held shares
        if norm_side == "sell":
            held_shares = existing_pos.qty if existing_pos else 0.0
            if held_shares < qty:
                return RiskValidationResult(
                    is_approved=False,
                    rule_name="INSUFFICIENT_SHARES_FOR_SELL",
                    reason=f"Attempting to sell {qty} shares, but only {held_shares} held.",
                    severity="WARNING",
                )

        # Passed all rules
        if client_order_id:
            self.recent_client_order_ids.add(client_order_id)

        return RiskValidationResult(is_approved=True, rule_name="ALL_CHECKS_PASSED")
