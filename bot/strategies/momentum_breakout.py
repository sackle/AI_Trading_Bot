"""
Strategy 2: Momentum Breakout (BTC/USD), 1-hour candles.

20-period Donchian channel (high/low) with a volume-confirmation filter.
A close above the rolling 20-period high on >=1.5x average volume opens a
long. A close below the rolling 20-period low on the same volume
confirmation exits an existing long.

IMPORTANT: Alpaca does not support short-selling crypto (spot only), so
this strategy is long/flat only on BTC/USD -- the breakdown signal exits
an open long but never opens a fresh short. See README "Design notes".

Exit/risk management: an initial hard ATR stop plus a trailing stop of
`trail_atr_multiple` ATRs, both handled by risk_manager.py / portfolio.py.
"""

from typing import Optional, Tuple

import pandas as pd

from bot.indicators import donchian_channel


class MomentumBreakoutStrategy:
    name = "momentum_breakout"

    def __init__(self, symbol: str, lookback: int = 20, volume_multiplier: float = 1.5):
        self.symbol = symbol
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier
        self.min_bars_required = lookback + 2

    def _channel_and_volume(self, bars: pd.DataFrame) -> Tuple[float, float, float]:
        channel_high, channel_low = donchian_channel(bars, self.lookback)
        # Shift by 1 so "the 20-period high/low" excludes the bar being
        # tested against it -- no lookahead / self-inclusion.
        prior_high = channel_high.shift(1).iloc[-1]
        prior_low = channel_low.shift(1).iloc[-1]
        avg_volume = bars["volume"].rolling(self.lookback).mean().shift(1).iloc[-1]
        return prior_high, prior_low, avg_volume

    def evaluate_entry(self, bars: pd.DataFrame) -> Optional[str]:
        if len(bars) < self.min_bars_required:
            return None

        prior_high, _prior_low, avg_volume = self._channel_and_volume(bars)
        if pd.isna(prior_high) or pd.isna(avg_volume):
            return None

        current = bars.iloc[-1]
        volume_confirmed = avg_volume > 0 and current["volume"] >= self.volume_multiplier * avg_volume

        if current["close"] > prior_high and volume_confirmed:
            return "long"
        return None  # breakdown case is handled in evaluate_exit only -- no fresh shorts on crypto

    def evaluate_exit(self, bars: pd.DataFrame, side: str) -> bool:
        if side != "long" or len(bars) < self.min_bars_required:
            return False

        _prior_high, prior_low, avg_volume = self._channel_and_volume(bars)
        if pd.isna(prior_low) or pd.isna(avg_volume):
            return False

        current = bars.iloc[-1]
        volume_confirmed = avg_volume > 0 and current["volume"] >= self.volume_multiplier * avg_volume

        return bool(current["close"] < prior_low and volume_confirmed)
