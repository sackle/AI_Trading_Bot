from __future__ import annotations

import json
import logging
import resource
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone

import config

logger = logging.getLogger(__name__)

class Watchdog:
    """Autonomous operation watchdog for 24/7 unmonitored trading.

    Tracks API reliability, order fill rates, P&L trajectory, and system
    resources. Produces a health score (0-100) and triggers graduated
    responses when the score degrades:

      Score 70-100 → Normal operation
      Score 40-70  → Reduced exposure (halve position sizes, widen stops)
      Score 0-40   → Defensive mode (no new entries, tighten stops)
      Score 0      → Emergency (close all, alert, pause)
    """

    def __init__(self) -> None:
        self._api_errors: deque[datetime] = deque(maxlen=1000)
        self._order_results: deque[tuple[bool, datetime]] = deque(maxlen=1000)
        self._health_score: float = 100.0
        self._last_check: datetime = datetime.now(timezone.utc)
        self._alert_cooldown: datetime | None = None
        self._last_api_success: datetime | None = None
        
        # We also keep track of the exact last time an error occurred for the recovery logic
        self._last_error_time: datetime | None = None

        # --- P&L trajectory tracking ---
        self._equity_snapshots: deque[float] = deque(maxlen=60)
        self._peak_equity: float = 0.0
        self._pnl_drawdown_pct: float = 0.0

        # --- Memory tracking ---
        self._memory_limit_mb: int = getattr(config, "WATCHDOG_MEMORY_LIMIT_MB", 2048)

    def record_api_success(self) -> None:
        """Record a successful API call."""
        self._last_api_success = datetime.now(timezone.utc)
        self._update_score()

    def record_api_error(self, error: str) -> None:
        """Record an API error."""
        now = datetime.now(timezone.utc)
        self._api_errors.append(now)
        self._last_error_time = now
        logger.warning(f"API Error recorded: {error}")
        self._update_score()

    def record_order_result(self, success: bool) -> None:
        """Record order fill/fail."""
        now = datetime.now(timezone.utc)
        self._order_results.append((success, now))
        if not success:
            self._last_error_time = now
        self._update_score()

    def record_pnl_snapshot(self, equity: float) -> None:
        """Record current equity for rolling-window drawdown tracking."""
        if equity <= 0:
            return
        self._equity_snapshots.append(equity)
        # Peak is the max equity in the rolling window (≈ last 60 ticks).
        self._peak_equity = max(self._equity_snapshots)
        if self._peak_equity > 0:
            self._pnl_drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100.0
        else:
            self._pnl_drawdown_pct = 0.0

    def _get_memory_mb(self) -> float:
        """Current process RSS memory in megabytes."""
        try:
            # resource.getrusage returns ru_maxrss in bytes on Linux, KB on macOS
            usage = resource.getrusage(resource.RUSAGE_SELF)
            import sys
            if sys.platform == "darwin":
                return usage.ru_maxrss / (1024 * 1024)  # bytes → MB on macOS
            return usage.ru_maxrss / 1024  # KB → MB on Linux
        except Exception:
            return 0.0

    def _update_score(self) -> float:
        """Internal computation of the health score."""
        now = datetime.now(timezone.utc)
        
        # Recovery: +2 per minute of error-free operation
        minutes_since_check = (now - self._last_check).total_seconds() / 60.0
        self._last_check = now
        
        if self._last_error_time:
            minutes_since_error = (now - self._last_error_time).total_seconds() / 60.0
            if minutes_since_error > 0:
                # Add recovery only if we had an error in the past and some time passed
                # To avoid double counting, we just apply natural healing over time
                pass
        
        # Calculate stateless penalties based on 1-hour rolling window
        one_hour_ago = now - timedelta(hours=1)
        
        # Clean up old data
        while self._api_errors and self._api_errors[0] < one_hour_ago:
            self._api_errors.popleft()
            
        while self._order_results and self._order_results[0][1] < one_hour_ago:
            self._order_results.popleft()
            
        score = 100.0
        
        # Subtract 5 per API error in last hour (max -50)
        api_penalty = min(50.0, len(self._api_errors) * 5.0)
        score -= api_penalty
        
        # Subtract 10 per failed order in last hour (max -30)
        failed_orders = sum(1 for success, _ in self._order_results if not success)
        order_penalty = min(30.0, failed_orders * 10.0)
        score -= order_penalty
        
        # Subtract 20 if no successful API call in last 5 minutes
        if self._last_api_success is None or (now - self._last_api_success) > timedelta(minutes=5):
            score -= 20.0
            
        # Ensure score is within 0-100 bounds
        score = max(0.0, min(100.0, score))

        # --- P&L drawdown penalty ---
        if self._pnl_drawdown_pct > 5.0:
            score -= 25.0
        elif self._pnl_drawdown_pct > 3.0:
            score -= 10.0

        # --- Memory usage penalty ---
        mem_mb = self._get_memory_mb()
        if mem_mb > self._memory_limit_mb:
            score -= 10.0

        self._health_score = max(0.0, min(100.0, score))
        
        # If we have a recovery mechanism that strictly adds +2 per minute 
        # and overrides the penalty calculations, it can be applied to a stateful score.
        # But the prompt describes penalties based on rolling windows, which naturally
        # "heals" as errors fall out of the 1-hour window. To satisfy "+2 per minute", 
        # we can explicitly apply it as a bonus if there were errors recently but they stopped.
        if self._last_error_time:
            minutes_since_error = (now - self._last_error_time).total_seconds() / 60.0
            recovery_bonus = min(api_penalty + order_penalty, minutes_since_error * 2.0)
            self._health_score = min(100.0, self._health_score + recovery_bonus)
            
        return self._health_score

    @property
    def health_score(self) -> float:
        """Returns current health score."""
        return self._update_score()

    @property
    def health_status(self) -> str:
        """'normal', 'reduced', 'defensive', 'emergency'"""
        score = self.health_score
        if score > 70:
            return "normal"
        elif score > 40:
            return "reduced"
        elif score > 0:
            return "defensive"
        else:
            return "emergency"

    @property
    def should_reduce_exposure(self) -> bool:
        return self.health_status in ("reduced", "defensive", "emergency")

    @property
    def should_block_new_entries(self) -> bool:
        return self.health_status in ("defensive", "emergency")

    @property
    def sizing_multiplier(self) -> float:
        """1.0 for normal, 0.5 for reduced, 0.0 for defensive/emergency"""
        status = self.health_status
        if status == "normal":
            return 1.0
        elif status == "reduced":
            return 0.5
        else:
            return 0.0

    def check_and_alert(self) -> None:
        """If webhook URL configured, send POST with health status.
        
        Cooldown: 1 alert per 15 minutes max.
        """
        webhook_url = getattr(config, "WATCHDOG_ALERT_WEBHOOK_URL", "")
        if not webhook_url:
            return

        now = datetime.now(timezone.utc)
        if self._alert_cooldown and now < self._alert_cooldown:
            return

        status = self.health_status
        if status == "normal":
            # Optional: don't alert on normal, or do. We'll alert if called.
            pass
            
        payload = {
            "text": f"Watchdog Alert: Health status is {status} (Score: {self.health_score:.1f})"
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status in (200, 201, 204):
                    # Set cooldown for 15 minutes
                    self._alert_cooldown = now + timedelta(minutes=15)
                    logger.info(f"Alert sent successfully. Status: {status}")
                else:
                    logger.error(f"Failed to send alert. Status code: {response.status}")
        except Exception as e:
            logger.error(f"Error sending watchdog alert: {e}")

    def to_dict(self) -> dict:
        """Serializable snapshot for dashboard/state."""
        return {
            "health_score": round(self.health_score, 2),
            "health_status": self.health_status,
            "sizing_multiplier": self.sizing_multiplier,
            "should_block_new_entries": self.should_block_new_entries,
            "should_reduce_exposure": self.should_reduce_exposure,
            "api_errors_last_hour": len(self._api_errors),
            "failed_orders_last_hour": sum(1 for s, _ in self._order_results if not s),
            "pnl_drawdown_pct": round(self._pnl_drawdown_pct, 2),
            "peak_equity": round(self._peak_equity, 2),
            "memory_mb": round(self._get_memory_mb(), 1),
            "last_check_utc": self._last_check.isoformat() if self._last_check else None,
            "last_api_success_utc": self._last_api_success.isoformat() if self._last_api_success else None
        }
