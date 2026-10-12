"""
Earnings Calendar Filter — blocks new equity entries in the window
surrounding a company's earnings announcement.

Why this matters:
  Stocks routinely gap 5–25% on earnings day, instantly blowing through
  any technical stop-loss and causing outsized, unpredictable losses.
  This filter removes that tail risk entirely by refusing new entries
  in a configurable window around the event.

Data source: yfinance (free, no API key required).
  yf.Ticker(symbol).calendar  → next earnings date
  Updates are cached per symbol for 24 hours.

Default behaviour (all configurable in .env / config.py):
  EARNINGS_DAYS_BEFORE  = 2  → block 2 trading days BEFORE earnings
  EARNINGS_DAYS_AFTER   = 1  → block 1 trading day AFTER earnings
  EARNINGS_ENABLED      = true

Usage:
    cal = EarningsCalendar()
    result = cal.check("AAPL")
    if result.blocked:
        print(result.reason)   # "AAPL earnings in 1 day (2026-08-28)"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import config

logger = logging.getLogger("aegis.earnings")

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False
    logger.warning("yfinance not installed — earnings filter disabled")


@dataclass
class EarningsCheckResult:
    symbol: str
    blocked: bool
    reason: str
    earnings_date: Optional[date]
    days_away: Optional[int]


# Sentinel date meaning "no earnings found"
_NO_DATE = None

# In-memory cache: symbol → (earnings_date | None, fetched_at)
_cache: dict[str, tuple[Optional[date], datetime]] = {}
_CACHE_TTL = timedelta(hours=24)
_NO_EARNINGS_SYMBOLS = {"SPY", "QQQ", "GLD", "USO", "VNQ", "REET"}


def _supports_earnings_calendar(symbol: str) -> bool:
    """Avoid Yahoo fundamentals/calendar calls for indices, ETFs, and crypto."""
    if "/" in symbol or symbol in _NO_EARNINGS_SYMBOLS:
        return False
    return config.symbol_asset_class(symbol) not in {
        "equity_index",
        "commodity_etf",
        "crypto",
    }


def _fetch_next_earnings(symbol: str) -> Optional[date]:
    """
    Fetch the next earnings date for `symbol` via yfinance.
    Returns None if not found or yfinance is unavailable.
    """
    if not _YF_OK:
        return None
    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None:
            return None

        # yfinance returns a dict with "Earnings Date" key
        # Value can be a list of Timestamps or a single Timestamp
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date")
        else:
            # Some yf versions return a DataFrame
            try:
                raw = cal.loc["Earnings Date"]
            except (KeyError, TypeError):
                return None

        if raw is None:
            return None

        # Normalise to a list of dates
        if not isinstance(raw, (list, tuple)):
            raw = [raw]

        today = date.today()
        future_dates = []
        for item in raw:
            try:
                if hasattr(item, "date"):
                    d = item.date()
                else:
                    d = datetime.strptime(str(item)[:10], "%Y-%m-%d").date()
                if d >= today:
                    future_dates.append(d)
            except Exception:
                continue

        return min(future_dates) if future_dates else None

    except Exception as exc:
        logger.debug("yfinance earnings fetch failed for %s: %s", symbol, exc)
        return None


class EarningsCalendar:
    """
    Checks whether a symbol has earnings within the configured
    pre/post-earnings blackout window.
    """

    def __init__(self):
        self.enabled      = getattr(config, "EARNINGS_ENABLED", True)
        self.days_before  = int(getattr(config, "EARNINGS_DAYS_BEFORE", 2))
        self.days_after   = int(getattr(config, "EARNINGS_DAYS_AFTER", 1))

    def check(self, symbol: str) -> EarningsCheckResult:
        """
        Returns EarningsCheckResult.blocked=True if the symbol is
        inside the earnings blackout window.
        Always returns blocked=False for crypto, ETFs that don't report,
        or when the calendar is disabled.
        """
        if not self.enabled:
            return EarningsCheckResult(symbol, False, "", None, None)

        # Funds and indices never have corporate earnings announcements.
        if not _supports_earnings_calendar(symbol):
            return EarningsCheckResult(symbol, False, "", None, None)

        earnings_date = self._get_cached(symbol)

        if earnings_date is None:
            return EarningsCheckResult(symbol, False, "", None, None)

        today = date.today()
        days_away = (earnings_date - today).days

        # Block if inside the [-(days_after), +(days_before)] window
        if -self.days_after <= days_away <= self.days_before:
            if days_away == 0:
                reason = f"Earnings TODAY for {symbol} — entry blocked"
            elif days_away > 0:
                reason = f"{symbol} earnings in {days_away}d ({earnings_date}) — entry blocked"
            else:
                reason = f"{symbol} reported earnings {abs(days_away)}d ago ({earnings_date}) — cooling off"

            return EarningsCheckResult(
                symbol=symbol,
                blocked=True,
                reason=reason,
                earnings_date=earnings_date,
                days_away=days_away,
            )

        return EarningsCheckResult(
            symbol=symbol,
            blocked=False,
            reason="",
            earnings_date=earnings_date,
            days_away=days_away,
        )

    def next_earnings(self, symbol: str) -> Optional[date]:
        """Return the next earnings date for a symbol (cached)."""
        return self._get_cached(symbol)

    def prefetch(self, symbols: list[str]):
        """
        Warm up the cache for a list of symbols in the background.
        Call once on startup so first-tick checks are instant.
        """
        import threading
        eligible_symbols = [sym for sym in dict.fromkeys(symbols) if _supports_earnings_calendar(sym)]

        def _warm():
            for sym in eligible_symbols:
                try:
                    self._get_cached(sym)
                except Exception:
                    pass
        t = threading.Thread(target=_warm, daemon=True, name="earnings-prefetch")
        t.start()
        logger.info("Earnings calendar: prefetching %d symbols in background", len(eligible_symbols))

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_cached(self, symbol: str) -> Optional[date]:
        now = datetime.now(timezone.utc)
        if symbol in _cache:
            cached_date, fetched_at = _cache[symbol]
            if now - fetched_at < _CACHE_TTL:
                return cached_date

        earnings_date = _fetch_next_earnings(symbol)
        _cache[symbol] = (earnings_date, now)
        if earnings_date:
            logger.debug("Earnings cache: %s → %s", symbol, earnings_date)
        return earnings_date
