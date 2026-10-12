"""Shared utilities for distributed services."""
from __future__ import annotations

import config


def load_symbol_list(strategy_name: str) -> list[str]:
    """Return the list of symbols for a given strategy."""
    if strategy_name == "mean_reversion":
        return list(config.MEAN_REVERSION_SYMBOLS)
    elif strategy_name == "momentum_breakout":
        return list(config.MOMENTUM_SYMBOLS)
    elif strategy_name == "trend_following":
        return list(config.TREND_SYMBOLS)
    elif strategy_name == "red_green_breakout":
        # Red/Green breakout runs on ALL symbols
        return list(config.ALL_EQUITY_SYMBOLS) + list(config.ALL_CRYPTO_SYMBOLS)
    elif strategy_name == "all":
        return list(config.ALL_EQUITY_SYMBOLS) + list(config.ALL_CRYPTO_SYMBOLS)
    return []


def get_strategy_timeframe(strategy_name: str) -> tuple[int, str]:
    """Return (amount, unit) for a strategy's bar timeframe."""
    if strategy_name == "mean_reversion":
        return config.MEAN_REVERSION_TIMEFRAME_AMOUNT, config.MEAN_REVERSION_TIMEFRAME_UNIT
    elif strategy_name == "momentum_breakout":
        return config.MOMENTUM_TIMEFRAME_AMOUNT, config.MOMENTUM_TIMEFRAME_UNIT
    elif strategy_name == "trend_following":
        return config.TREND_TIMEFRAME_AMOUNT, config.TREND_TIMEFRAME_UNIT
    return 1, "Hour"
