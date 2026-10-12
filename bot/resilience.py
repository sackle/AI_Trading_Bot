"""
Network resilience primitives for the Alpaca broker layer.

Provides:

- ``retry_on_transient``  -- decorator that retries a callable when it raises
  a transient network error (connection reset, timeout, DNS failure, 503).
  Permanent errors (401, 403, 422) propagate immediately.

- ``CircuitBreaker``      -- tracks consecutive broker failures and transitions
  through CLOSED → OPEN → HALF_OPEN states so the bot can enter a degraded
  mode instead of hammering a broken endpoint.

- ``is_transient_error``  -- classifies an arbitrary exception as transient
  (worth retrying) vs permanent (should propagate).

All knobs are read from ``config`` at call time so they can be hot-changed
via .env without restarting the bot.
"""

from __future__ import annotations

import functools
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

import config

# ---------------------------------------------------------------------------
# Transient error classification
# ---------------------------------------------------------------------------

_TRANSIENT_SUBSTRINGS = (
    "connection reset",
    "read timed out",
    "timed out",
    "timeout",
    "503",
    "502",
    "429",
    "service unavailable",
    "bad gateway",
    "name resolution",
    "failed to resolve",
    "temporary failure in name resolution",
    "can't assign requested address",
    "connection refused",
    "connection aborted",
    "broken pipe",
    "eof occurred",
    "incomplete read",
    "too many open files",
)

_PERMANENT_SUBSTRINGS = (
    "401",
    "403",
    "422",
    "unauthorized",
    "forbidden",
    "unprocessable",
    "invalid",
    "not found",
    "already",
)


def is_transient_error(exc: BaseException) -> bool:
    """
    Returns True when *exc* looks like a transient network/infra problem
    that is worth retrying. Authentication and validation errors are never
    transient.
    """
    message = str(exc).lower()

    # Check permanent patterns first — they take priority.
    for pattern in _PERMANENT_SUBSTRINGS:
        if pattern in message:
            return False

    # OSError subclasses with known transient errnos.
    if isinstance(exc, OSError):
        return True

    for pattern in _TRANSIENT_SUBSTRINGS:
        if pattern in message:
            return True

    return False


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

T = TypeVar("T")


def retry_on_transient(
    func: Callable[..., T] | None = None,
    *,
    max_attempts: int | None = None,
    backoff_base: float | None = None,
    backoff_max: float = 30.0,
    label: str = "",
) -> Callable[..., T]:
    """
    Decorator that retries *func* up to *max_attempts* times when it raises a
    transient error.  Non-transient errors propagate immediately.

    Backoff between retries is exponential with jitter::

        delay = min(backoff_base * 2^attempt + jitter, backoff_max)

    When *max_attempts* or *backoff_base* are ``None`` the values are read
    from ``config.BROKER_READ_RETRY_ATTEMPTS`` and
    ``config.BROKER_READ_RETRY_BASE_DELAY_SECONDS`` at call time.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempts = max_attempts or config.BROKER_READ_RETRY_ATTEMPTS
            base_delay = backoff_base or config.BROKER_READ_RETRY_BASE_DELAY_SECONDS
            call_label = label or fn.__qualname__

            last_exc: BaseException | None = None

            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not is_transient_error(exc):
                        raise

                    if attempt < attempts - 1:
                        delay = min(
                            base_delay * (2 ** attempt) + random.uniform(0, 0.5),
                            backoff_max,
                        )
                        print(
                            f"[retry] {call_label} attempt {attempt + 1}/{attempts} "
                            f"failed ({type(exc).__name__}: {exc}); "
                            f"retrying in {delay:.1f}s…"
                        )
                        time.sleep(delay)

            # All attempts exhausted — re-raise the last exception.
            raise last_exc  # type: ignore[misc]

        return wrapper

    if func is not None:
        # Called as @retry_on_transient without parentheses.
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Tracks consecutive broker failures and transitions through three states:

    CLOSED    – normal operation.  Every success resets the failure counter.
    OPEN      – too many consecutive failures.  Calls are refused for a
                cooldown period so we don't hammer a broken endpoint.
    HALF_OPEN – cooldown elapsed; the next call is a probe.  If it succeeds
                we transition back to CLOSED; if it fails we go back to OPEN.

    Usage::

        cb = CircuitBreaker()
        if cb.allow_request():
            try:
                result = broker_call()
                cb.record_success()
            except Exception:
                cb.record_failure()
                raise
        else:
            # bot is in degraded mode
            ...
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        threshold: int | None = None,
        cooldown_seconds: float | None = None,
    ):
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._state: str = self.CLOSED
        self._consecutive_failures: int = 0
        self._last_failure_at: datetime | None = None
        self._opened_at: datetime | None = None

    # -- Config helpers (read at call time) ---------------------------------

    @property
    def threshold(self) -> int:
        return self._threshold or getattr(config, "CIRCUIT_BREAKER_THRESHOLD", 5)

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds or getattr(config, "CIRCUIT_BREAKER_COOLDOWN_SECONDS", 300.0)

    # -- Public API ---------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state, potentially transitioning OPEN → HALF_OPEN if cooldown elapsed."""
        if self._state == self.OPEN:
            elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
            if elapsed >= self.cooldown_seconds:
                self._state = self.HALF_OPEN
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def allow_request(self) -> bool:
        """Return True if a request should be attempted."""
        current = self.state
        if current == self.CLOSED:
            return True
        if current == self.HALF_OPEN:
            return True  # probe attempt
        return False  # OPEN — skip

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = self.CLOSED
        self._opened_at = None
        self._last_failure_at = None

    def record_failure(self) -> None:
        now = datetime.now(timezone.utc)
        self._consecutive_failures += 1
        self._last_failure_at = now

        if self._consecutive_failures >= self.threshold:
            self._state = self.OPEN
            self._opened_at = now

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "consecutive_failures": self._consecutive_failures,
            "threshold": self.threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "last_failure_at": self._last_failure_at.isoformat() if self._last_failure_at else None,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
        }
