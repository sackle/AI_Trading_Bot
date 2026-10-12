"""
Runtime market filters that sit between a raw strategy signal and order
submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

import config
from bot.indicators import ema

_ET = ZoneInfo("America/New_York")


@dataclass
class FilterDecision:
    allowed: bool
    reasons: list[str]
    metrics: dict


def _is_equity(symbol: str) -> bool:
    return symbol in config.ALL_EQUITY_SYMBOLS


class MarketFilters:
    def evaluate(
        self,
        *,
        symbol: str,
        strategy_name: str,
        side: str,
        bars: pd.DataFrame,
        atr_value: float,
    ) -> FilterDecision:
        reasons: list[str] = []
        metrics: dict[str, float] = {}

        if bars.empty:
            return FilterDecision(False, ["No bars available for filter evaluation."], metrics)

        latest_bar_time = bars.index[-1]
        latest_price = float(bars["close"].iloc[-1])
        metrics["latest_price"] = latest_price

        if config.ENABLE_ATR_REGIME_FILTER and latest_price > 0:
            bounds = config.ATR_PCT_BOUNDS_BY_STRATEGY.get(strategy_name)
            if bounds:
                atr_pct = float(atr_value) / latest_price if atr_value else 0.0
                metrics["atr_pct"] = atr_pct
                low, high = bounds
                if atr_pct < low or atr_pct > high:
                    reasons.append(
                        f"ATR regime filter blocked entry ({atr_pct:.4f} outside {low:.4f}-{high:.4f})."
                    )

        if (
            config.ENABLE_TREND_ALIGNMENT_FILTER
            and config.TREND_ALIGNMENT_BY_STRATEGY.get(strategy_name, False)
            and len(bars) >= config.TREND_FILTER_EMA_PERIOD
        ):
            trend_ema = ema(bars["close"], config.TREND_FILTER_EMA_PERIOD).iloc[-1]
            if pd.notna(trend_ema):
                metrics["trend_ema"] = float(trend_ema)
                if side == "long" and latest_price < trend_ema:
                    reasons.append("Trend alignment filter blocked a long below the 200 EMA.")
                if side == "short" and latest_price > trend_ema:
                    reasons.append("Trend alignment filter blocked a short above the 200 EMA.")

        if config.ENABLE_EQUITY_TIME_FILTER and _is_equity(symbol):
            bar_et = latest_bar_time.tz_convert(_ET) if latest_bar_time.tzinfo else latest_bar_time.tz_localize("UTC").tz_convert(_ET)
            session_start = time(*config.REGULAR_EQUITY_SESSION_START_ET)
            session_end = time(*config.REGULAR_EQUITY_SESSION_END_ET)
            minutes_since_open = (bar_et.hour * 60 + bar_et.minute) - (session_start.hour * 60 + session_start.minute)
            minutes_until_close = (session_end.hour * 60 + session_end.minute) - (bar_et.hour * 60 + bar_et.minute)
            metrics["minutes_since_open"] = minutes_since_open
            metrics["minutes_until_close"] = minutes_until_close

            if 0 <= minutes_since_open < config.EQUITY_NO_NEW_ENTRIES_FIRST_MINUTES:
                reasons.append("Time-of-day filter blocked an entry near the opening session.")
            if 0 <= minutes_until_close < config.EQUITY_NO_NEW_ENTRIES_LAST_MINUTES:
                reasons.append("Time-of-day filter blocked an entry near the closing session.")

        return FilterDecision(allowed=not reasons, reasons=reasons, metrics=metrics)
